"""Reproducible degradation scenarios.

Deterministic transforms are applied to a source video (yours, or the bundled
*synthetic* scene, which is generated procedurally and contains no people or
real objects). Every output clip is listed in ``manifest.json`` with the exact
transform and parameters applied to each time segment.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .source import VideoSource

W, H, FPS = 640, 360, 15.0

Transform = Callable[[np.ndarray], np.ndarray]


def dark(img: np.ndarray, gain: float = 0.22) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=gain, beta=0)


def blur(img: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def occlude(img: np.ndarray, frac_w: float = 0.45, frac_h: float = 0.6, level: int = 70) -> np.ndarray:
    """Flat, soft-edged patch over the lower-left region (a smudge / sticker proxy)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.rectangle(mask, (0, int(h * (1 - frac_h))), (int(w * frac_w), h), 1.0, -1)
    mask = np.asarray(cv2.GaussianBlur(mask, (0, 0), 6), np.float32)[..., None]
    patch = np.full_like(img, level)
    return (img * (1 - mask) + patch * mask).astype(np.uint8)


def shift(img: np.ndarray, dx: float = 0.22, dy: float = 0.12, zoom: float = 1.3) -> np.ndarray:
    """Translate and zoom: a proxy for a knocked or re-pointed camera."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 4.0, zoom)
    m[0, 2] += dx * w
    m[1, 2] += dy * h
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


TRANSFORMS: dict[str, tuple[Transform | None, dict[str, Any]]] = {
    "healthy": (None, {}),
    "dark": (dark, {"gain": 0.22}),
    "blur": (blur, {"sigma": 6.0}),
    "occlusion": (occlude, {"frac_w": 0.45, "frac_h": 0.6, "level": 70}),
    "framing_shift": (shift, {"dx": 0.22, "dy": 0.12, "zoom": 1.3}),
}


# -- synthetic scene ------------------------------------------------------
def _background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H // 2):  # sky gradient
        img[y] = (200 - y // 3, 170 - y // 4, 120)
    for i in range(9):  # buildings with windows
        x0 = i * 72 + int(rng.integers(-10, 10))
        top = int(rng.integers(40, 140))
        col = tuple(int(c) for c in rng.integers(60, 140, 3))
        cv2.rectangle(img, (x0, top), (x0 + 60, H // 2 + 40), col, -1)
        for wy in range(top + 8, H // 2 + 30, 16):
            for wx in range(x0 + 6, x0 + 54, 12):
                lit = int(rng.integers(0, 2))
                cv2.rectangle(img, (wx, wy), (wx + 6, wy + 8), (60, 200, 230) if lit else (40, 40, 50), -1)
    ground = rng.integers(70, 120, (H - H // 2 - 40, W, 1)).astype(np.uint8).repeat(3, axis=2)
    img[H // 2 + 40:] = cv2.GaussianBlur(ground, (3, 3), 0)
    for x in range(0, W, 60):  # lane markings
        cv2.rectangle(img, (x, H - 70), (x + 30, H - 64), (230, 230, 230), -1)
    cv2.putText(img, "SENTINEL SYNTHETIC SCENE", (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    return img


def synthetic_frames(seconds: float, seed: int = 7, t_offset: float = 0.0) -> Iterator[np.ndarray]:
    bg = _background(seed)
    rng = np.random.default_rng(seed + 1)
    for i in range(int(seconds * FPS)):
        t = t_offset + i / FPS
        img = bg.copy()
        for k, (speed, y, color) in enumerate([(55, 250, (30, 30, 200)), (-40, 290, (200, 80, 30)), (70, 230, (40, 160, 40))]):
            x = int((t * speed + k * 211) % (W + 120)) - 60
            if speed < 0:
                x = W - x
            cv2.rectangle(img, (x, y), (x + 70, y + 30), color, -1)
            cv2.circle(img, (x + 15, y + 32), 7, (20, 20, 20), -1)
            cv2.circle(img, (x + 55, y + 32), 7, (20, 20, 20), -1)
        noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
        yield np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)  # type: ignore[attr-defined]
    if not w.isOpened():
        raise RuntimeError(f"cannot write video: {path}")
    return w


@dataclass
class Segment:
    scenario: str
    start_s: float
    end_s: float


DEMO_TIMELINE = [
    Segment("healthy", 0, 8), Segment("dark", 8, 16), Segment("healthy", 16, 24),
    Segment("blur", 24, 32), Segment("healthy", 32, 40), Segment("occlusion", 40, 48),
    Segment("healthy", 48, 56), Segment("framing_shift", 56, 64), Segment("healthy", 64, 74),
]


def _source_frames(source: str | None, seconds: float, t_offset: float) -> tuple[Iterator[np.ndarray], float]:
    if source is None:
        return synthetic_frames(seconds, t_offset=t_offset), FPS
    src = VideoSource(source).open()
    fps = src.fps
    skip = int(t_offset * fps)
    needed = int(seconds * fps)

    def gen() -> Iterator[np.ndarray]:
        try:
            for fr in src.frames():
                if fr.index < skip:
                    continue
                if fr.index >= skip + needed:
                    return
                yield fr.image
        finally:
            src.close()
    return gen(), fps


def build_demo(out_dir: str | Path, source: str | None = None, clip_seconds: float = 10.0) -> dict[str, Any]:
    """Write baseline + per-scenario clips + a combined timeline with recovery."""
    out = Path(out_dir)
    manifest: dict[str, Any] = {
        "source": source or "synthetic (procedurally generated, deterministic seed=7)",
        "synthetic": source is None,
        "note": ("Degradations are deterministic digital transforms, not physical camera faults. "
                 "Results on these clips do not validate detection of real lens dirt, darkness or camera movement."),
        "clips": [],
    }

    # Baseline clip: healthy footage from a different time window than the test clips.
    frames, fps = _source_frames(source, 20.0, t_offset=0.0)
    first = next(frames)
    size = (first.shape[1], first.shape[0])
    w = _writer(out / "baseline_healthy.mp4", fps, size)
    w.write(first)
    n = 1
    for img in frames:
        w.write(img)
        n += 1
    w.release()
    manifest["baseline"] = {"file": "baseline_healthy.mp4", "frames": n, "fps": fps, "scenario": "healthy"}

    for name, (fn, params) in TRANSFORMS.items():
        frames, fps = _source_frames(source, clip_seconds, t_offset=20.0)
        w = _writer(out / f"clip_{name}.mp4", fps, size)
        n = 0
        for img in frames:
            w.write(fn(img, **params) if fn else img)
            n += 1
        w.release()
        manifest["clips"].append({"file": f"clip_{name}.mp4", "frames": n, "fps": fps,
                                  "segments": [{"scenario": name, "transform": fn.__name__ if fn else None,
                                                "params": params, "start_s": 0.0, "end_s": n / fps}]})

    total = DEMO_TIMELINE[-1].end_s
    frames, fps = _source_frames(source, total, t_offset=20.0)
    w = _writer(out / "demo_timeline.mp4", fps, size)
    n = 0
    for img in frames:
        t = n / fps
        seg = next(s for s in DEMO_TIMELINE if s.start_s <= t < s.end_s)
        fn, params = TRANSFORMS[seg.scenario]
        w.write(fn(img, **params) if fn else img)
        n += 1
    w.release()
    manifest["clips"].append({
        "file": "demo_timeline.mp4", "frames": n, "fps": fps,
        "segments": [{"scenario": s.scenario, "transform": (TRANSFORMS[s.scenario][0].__name__  # type: ignore[union-attr]
                                                             if TRANSFORMS[s.scenario][0] else None),
                      "params": TRANSFORMS[s.scenario][1], "start_s": s.start_s, "end_s": s.end_s}
                     for s in DEMO_TIMELINE],
    })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
