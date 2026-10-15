"""Video ingestion: files (primary), USB cameras (``0``, ``1``...) and RTSP/HTTP URLs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

DEFAULT_FPS = 15.0
# Consecutive undecodable reads tolerated before a file is treated as ended
# (or a live source as disconnected). Isolated corrupt frames are skipped.
MAX_CONSECUTIVE_READ_FAILURES = 5


class SourceError(RuntimeError):
    """The source could not be opened or produced no decodable frames."""


class StreamLost(RuntimeError):
    """A live source stopped delivering frames and reconnection failed."""


@dataclass
class Frame:
    index: int
    t: float  # stream time in seconds
    image: np.ndarray


def redact(spec: str) -> str:
    """Remove credentials from a source URL before it is logged or returned."""
    if "://" not in spec:
        return spec
    try:
        parts = urlsplit(spec)
    except ValueError:
        return "<unparseable url>"
    if parts.username is None and parts.password is None:
        return spec
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))


def source_kind(spec: str) -> str:
    s = spec.strip()
    if s.isdigit():
        return "camera"
    if "://" in s:
        return "url"
    return "file"


def valid_frame(img: object) -> bool:
    return (
        isinstance(img, np.ndarray)
        and img.dtype == np.uint8
        and img.size > 0
        and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] in (1, 3, 4)))
    )


class VideoSource:
    # Clock for live-source timestamps; replaceable in tests.
    clock = staticmethod(time.monotonic)

    def __init__(self, spec: str, reconnect_attempts: int = 3, reconnect_delay_s: float = 1.0,
                 max_frames: int | None = None) -> None:
        self.spec = spec
        self.safe_spec = redact(spec)
        self.kind = source_kind(spec)
        s = spec.strip()
        self.target: str | int = int(s) if self.kind == "camera" else s
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self.max_frames = max_frames
        self.cap: cv2.VideoCapture | None = None
        self.fps = DEFAULT_FPS
        self.end_reason: str | None = None
        self.corrupt_frames = 0
        self.reconnects = 0
        self._first: np.ndarray | None = None
        self._first_index = 0

    @property
    def live(self) -> bool:
        return self.kind != "file"

    def _open_capture(self) -> cv2.VideoCapture:
        if self.kind == "file":
            path = Path(str(self.target))
            if not path.is_file():
                raise SourceError(f"video file not found: {self.safe_spec}")
            if path.stat().st_size == 0:
                raise SourceError(f"video file is empty: {self.safe_spec}")
        cap = cv2.VideoCapture(self.target)  # type: ignore[call-overload]
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"cannot open video source: {self.safe_spec}")
        return cap

    def _read(self) -> np.ndarray | None:
        assert self.cap is not None
        try:
            ok, img = self.cap.read()
        except cv2.error:
            return None
        return img if ok and valid_frame(img) else None

    def open(self) -> VideoSource:
        self.cap = self._open_capture()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and 1.0 <= fps <= 240.0 else DEFAULT_FPS
        img = None
        failures = 0
        while img is None and failures < MAX_CONSECUTIVE_READ_FAILURES:
            img = self._read()
            if img is None:
                failures += 1
        if img is None:
            self.close()
            raise SourceError(f"no decodable frames in source: {self.safe_spec}")
        self.corrupt_frames = failures
        self._first_index = failures
        self._first = img
        return self

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> VideoSource:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _reconnect(self) -> bool:
        for _ in range(self.reconnect_attempts):
            self.close()
            time.sleep(self.reconnect_delay_s)
            try:
                self.cap = self._open_capture()
                self.reconnects += 1
                return True
            except SourceError:
                continue
        return False

    def frames(self) -> Iterator[Frame]:
        if self.cap is None:
            raise SourceError("source not opened")
        start = self.clock()
        idx = 0 if self.live else self._first_index
        img: np.ndarray | None = self._first
        self._first = None
        while True:
            if img is None:
                failures = 0
                while img is None and failures < MAX_CONSECUTIVE_READ_FAILURES:
                    img = self._read()
                    if img is None:
                        failures += 1
                if img is None:
                    if not self.live:
                        self.end_reason = "end_of_file"
                        return
                    if self._reconnect():
                        continue
                    self.end_reason = "stream_lost"
                    raise StreamLost(f"live source stopped delivering frames: {self.safe_spec}")
                # Failures followed by a good frame were corrupt frames, not the end.
                self.corrupt_frames += failures
                idx += failures if not self.live else 0
            t = (self.clock() - start) if self.live else idx / self.fps
            yield Frame(idx, t, img)
            idx += 1
            img = None
            if self.max_frames is not None and idx >= self.max_frames:
                self.end_reason = "max_frames"
                return
