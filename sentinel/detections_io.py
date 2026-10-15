"""Strict loader for Sentinel's internal detection-event format (JSONL).

Format: ``schema/detection_event.schema.json`` (version 1.0), one JSON object per
line, one line per frame. This is deliberately *Sentinel's* format, independent
of PipeGen: once a genuine PipeGen JSON metadata sample exists, a small
converter (``parse_pipegen_metadata`` in ``detector.py``) should produce these
records. Nothing here fabricates detections: frames without a record are
reported as "no data", never as "nothing detected".

Validation rules (enforced without needing the jsonschema package at runtime):

* every non-blank line is a JSON object with exactly the schema's keys;
* ``frame_index`` (non-negative int, not bool) and/or ``timestamp_s`` (finite, >= 0);
* ``confidence`` finite and within [0, 1]; ``bbox`` four finite numbers with w, h > 0;
* duplicate frames: identical repeats are dropped; conflicting repeats are an
  error in strict mode, and the first record wins in lenient mode;
* out-of-order records are accepted, sorted and counted;
* records for other streams are skipped and counted (multi-stream files);
* in strict mode any malformed record rejects the whole file, with line numbers.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .detector import Detection

DETECTION_SCHEMA_VERSION = "1.0"
_TOP_KEYS = {"schema_version", "stream_id", "source", "frame_index", "timestamp_s", "detections"}
_DET_KEYS = {"label", "confidence", "bbox", "track_id"}
MAX_REPORTED_ERRORS = 20


class DetectionEventError(ValueError):
    """The detections file is unreadable or (in strict mode) contains invalid records."""


@dataclass
class LoadReport:
    path: str
    stream_id: str
    records_read: int = 0
    records_used: int = 0
    malformed: int = 0
    other_stream: int = 0
    duplicates_identical: int = 0
    duplicates_conflicting: int = 0
    out_of_order: int = 0
    timestamp_mismatches: int = 0
    sources: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _num(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def validate_record(rec: object) -> list[str]:
    """Return a list of problems with one record (empty if valid)."""
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]
    errs: list[str] = []
    extra = set(rec) - _TOP_KEYS
    if extra:
        errs.append(f"unknown keys {sorted(extra)}")
    if rec.get("schema_version") != DETECTION_SCHEMA_VERSION:
        errs.append(f"schema_version must be {DETECTION_SCHEMA_VERSION!r}")
    for key in ("stream_id", "source"):
        if not isinstance(rec.get(key), str) or not rec[key].strip():
            errs.append(f"{key} must be a non-empty string")
    has_idx, has_ts = "frame_index" in rec, "timestamp_s" in rec
    if not has_idx and not has_ts:
        errs.append("frame_index or timestamp_s is required")
    if has_idx:
        fi = rec["frame_index"]
        if not isinstance(fi, int) or isinstance(fi, bool) or fi < 0:
            errs.append(f"frame_index must be a non-negative integer, got {fi!r}")
    if has_ts and (not _num(rec["timestamp_s"]) or rec["timestamp_s"] < 0):
        errs.append(f"timestamp_s must be a finite number >= 0, got {rec['timestamp_s']!r}")
    dets = rec.get("detections")
    if not isinstance(dets, list):
        errs.append("detections must be an array")
        return errs
    for i, d in enumerate(dets):
        where = f"detections[{i}]"
        if not isinstance(d, dict):
            errs.append(f"{where} is not an object")
            continue
        extra = set(d) - _DET_KEYS
        if extra:
            errs.append(f"{where}: unknown keys {sorted(extra)}")
        if not isinstance(d.get("label"), str) or not d["label"].strip():
            errs.append(f"{where}.label must be a non-empty string")
        c = d.get("confidence")
        if not _num(c) or not 0.0 <= c <= 1.0:  # type: ignore[operator]
            errs.append(f"{where}.confidence must be a number in [0, 1], got {c!r}")
        b = d.get("bbox")
        if not (isinstance(b, list) and len(b) == 4 and all(_num(x) for x in b)):
            errs.append(f"{where}.bbox must be [x, y, w, h] numbers")
        elif b[2] <= 0 or b[3] <= 0:
            errs.append(f"{where}.bbox width and height must be > 0")
        tid = d.get("track_id")
        if tid is not None and (not isinstance(tid, int) or isinstance(tid, bool)):
            errs.append(f"{where}.track_id must be an integer or null")
    return errs


def _to_detections(rec: dict[str, Any]) -> list[Detection]:
    return [
        Detection(d["label"], float(d["confidence"]),
                  (round(d["bbox"][0]), round(d["bbox"][1]), round(d["bbox"][2]), round(d["bbox"][3])))
        for d in rec["detections"]
    ]


class ExternalDetectionFeed:
    """Detections from another pipeline, associated with Sentinel's frames.

    Lookup order: exact ``frame_index``; otherwise the record whose
    ``timestamp_s`` is nearest to the frame's stream time, within
    ``time_tolerance_s``. No match returns ``None`` ("no data for this frame").
    """

    source = "external_pipeline"

    def __init__(self, by_frame: dict[int, list[Detection]], by_time: list[tuple[float, list[Detection]]],
                 name: str, time_tolerance_s: float) -> None:
        self.by_frame = by_frame
        self._times = [t for t, _ in sorted(by_time, key=lambda x: x[0])]
        self._dets = [d for _, d in sorted(by_time, key=lambda x: x[0])]
        self.name = name
        self.time_tolerance_s = time_tolerance_s

    def detect(self, frame: np.ndarray, index: int = 0, t: float | None = None) -> list[Detection] | None:
        if index in self.by_frame:
            return self.by_frame[index]
        if t is None or not self._times:
            return None
        i = bisect.bisect_left(self._times, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self._times) and abs(self._times[j] - t) <= self.time_tolerance_s and (
                best is None or abs(self._times[j] - t) < abs(self._times[best] - t)
            ):
                best = j
        return None if best is None else self._dets[best]


def load_detection_events(
    path: str | Path,
    stream_id: str,
    fps: float,
    strict: bool = True,
    time_tolerance_s: float | None = None,
) -> tuple[ExternalDetectionFeed, LoadReport]:
    p = Path(path)
    if not p.is_file():
        raise DetectionEventError(f"detections file not found: {p}")
    tol = time_tolerance_s if time_tolerance_s is not None else 0.5 / max(fps, 1e-6)
    report = LoadReport(path=str(p), stream_id=stream_id)
    keyed: dict[tuple[str, float], tuple[int, dict[str, Any]]] = {}
    last_key: float | None = None

    def problem(lineno: int, msg: str) -> None:
        report.malformed += 1
        if len(report.errors) < MAX_REPORTED_ERRORS:
            report.errors.append(f"line {lineno}: {msg}")

    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DetectionEventError(f"cannot read detections file {p}: {exc}") from exc

    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        report.records_read += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            problem(lineno, f"invalid JSON ({exc.msg})")
            continue
        errs = validate_record(rec)
        if errs:
            problem(lineno, "; ".join(errs))
            continue
        if rec["stream_id"] != stream_id:
            report.other_stream += 1
            continue
        if "frame_index" in rec:
            key: tuple[str, float] = ("frame", float(rec["frame_index"]))
            if "timestamp_s" in rec and abs(rec["timestamp_s"] - rec["frame_index"] / fps) > tol:
                report.timestamp_mismatches += 1
        else:
            key = ("time", float(rec["timestamp_s"]))
        order = key[1] / fps if key[0] == "frame" else key[1]  # compare in seconds, never index vs time
        if last_key is not None and order < last_key:
            report.out_of_order += 1
        last_key = order if last_key is None else max(last_key, order)
        if key in keyed:
            first_line, first = keyed[key]
            if first["detections"] == rec["detections"]:
                report.duplicates_identical += 1
            else:
                report.duplicates_conflicting += 1
                msg = f"conflicting duplicate of line {first_line} for {key[0]} {key[1]:g}"
                if strict:
                    problem(lineno, msg)
                elif len(report.errors) < MAX_REPORTED_ERRORS:
                    report.errors.append(f"line {lineno}: {msg} (kept line {first_line})")
            continue
        keyed[key] = (lineno, rec)
        if rec["source"] not in report.sources:
            report.sources.append(rec["source"])

    if strict and report.malformed:
        raise DetectionEventError(
            f"{report.malformed} invalid record(s) in {p} (strict mode):\n  " + "\n  ".join(report.errors)
        )
    if not keyed:
        raise DetectionEventError(
            f"no usable detection records for stream {stream_id!r} in {p} "
            f"({report.records_read} read, {report.other_stream} for other streams, {report.malformed} malformed)"
        )
    by_frame: dict[int, list[Detection]] = {}
    by_time: list[tuple[float, list[Detection]]] = []
    for (kind, k), (_, rec) in sorted(keyed.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if kind == "frame":
            by_frame[int(k)] = _to_detections(rec)
        else:
            by_time.append((k, _to_detections(rec)))
    report.records_used = len(keyed)
    name = "external:" + ",".join(report.sources)
    return ExternalDetectionFeed(by_frame, by_time, name, tol), report
