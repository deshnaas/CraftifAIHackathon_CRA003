"""Observe -> Measure -> Diagnose -> Respond -> Verify recovery."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .baseline import Baseline, build_baseline, obstruction_fraction
from .config import SentinelConfig
from .detector import Detection, Detector, NullDetector, detection_stats
from .health import HealthEngine, HealthState, resolve_thresholds
from .incidents import IncidentManager, JsonEventLog
from .metrics import FrameMetrics, compute_frame_metrics, thumb_similarity
from .source import SourceError, StreamLost, VideoSource, redact, source_kind

STATE_COLORS = {  # BGR
    HealthState.HEALTHY: (80, 180, 60),
    HealthState.DEGRADED: (0, 170, 255),
    HealthState.UNTRUSTED: (40, 40, 220),
    HealthState.RECOVERING: (220, 160, 40),
}


class EventSink(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...


def check_values(m: FrameMetrics, base: Baseline, cfg: SentinelConfig, latency_ms: float) -> dict[str, float | None]:
    return {
        "brightness": m.brightness / base.brightness if base.brightness > 0 else None,
        "sharpness": m.sharpness / base.sharpness if base.sharpness > 0 else None,
        "obstruction": obstruction_fraction(m.cell_std, base.cell_std, cfg),
        "framing": thumb_similarity(m.thumb, base.reference_thumb),
        "latency": latency_ms,
    }


def classify_input(spec: str) -> str:
    """Label the input honestly: synthetic demo footage vs real footage or cameras.

    ``synthetic_demo``  listed in a demo manifest generated from the synthetic scene
    ``derived_demo``    listed in a demo manifest derived from user-supplied footage
    ``video_file``      any other file (real footage, unverified)
    ``camera`` / ``network_stream``  live inputs
    """
    kind = source_kind(spec)
    if kind == "camera":
        return "camera"
    if kind == "url":
        return "network_stream"
    path = Path(spec)
    manifest = path.parent / "manifest.json"
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "video_file"
    listed = {c.get("file") for c in m.get("clips", [])} | {m.get("baseline", {}).get("file")}
    if path.name not in listed:
        return "video_file"
    return "synthetic_demo" if str(m.get("source", "")).startswith("synthetic") else "derived_demo"


@dataclass
class FrameResult:
    index: int
    t: float
    state: str
    checks: list[dict[str, Any]]
    metrics: dict[str, Any]
    latency_ms: float
    detection: dict[str, Any] | None
    suspected_categories: list[str]
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Monitor:
    def __init__(self, cfg: SentinelConfig, baseline: Baseline, out_dir: str | Path | None = None,
                 detector: Detector | None = None, sinks: list[EventSink] | None = None,
                 **incident_kwargs: Any) -> None:
        self.cfg = cfg
        self.baseline = baseline
        self.detector = detector or NullDetector()
        self.engine = HealthEngine(cfg, resolve_thresholds(cfg, baseline))
        incident_kwargs.setdefault("recurrence_window_s", cfg.recurrence_window_s)
        incident_kwargs.setdefault("detection_context_s", cfg.detection_context_s)
        self.incidents = IncidentManager(
            cfg.stream_id,
            {"brightness": baseline.brightness, "sharpness": baseline.sharpness},
            **incident_kwargs,
        )
        self.out_dir = Path(out_dir) if out_dir else None
        self.log = JsonEventLog(self.out_dir) if self.out_dir else None
        self.sinks = sinks or []
        self.sink_errors = 0
        self.events: list[dict[str, Any]] = []
        self.last_t = 0.0
        self.last_detections: list[Detection] | None = None
        self.resolution_mismatch: tuple[int, int] | None = None

    def process(self, index: int, t: float, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        m = compute_frame_metrics(frame)
        dets = self.detector.detect(frame, index, t)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_detections = dets
        stats = detection_stats(dets, self.detector)
        transitions = self.engine.update(t, check_values(m, self.baseline, self.cfg, latency_ms))
        summary = m.summary()
        events: list[dict[str, Any]] = []
        size = (m.width, m.height)
        if self.resolution_mismatch is None and size != (self.baseline.width, self.baseline.height):
            self.resolution_mismatch = size
            events.append(self.incidents.stream_warning(
                t, self.engine.state.value,
                f"frame size {m.width}x{m.height} differs from baseline {self.baseline.width}x{self.baseline.height}; "
                "framing and obstruction comparisons may be unreliable - re-baseline this camera",
            ))
        events += self.incidents.on_frame(t, self.engine, transitions, summary, stats)
        for ev in events:
            if ev["event_type"] == "incident_opened":
                self._snapshot(ev, frame)
        self._emit(events)
        self.last_t = t
        return FrameResult(index, round(t, 3), self.engine.state.value, self.engine.snapshot(), summary,
                           round(latency_ms, 2), stats, self.engine.suspected_categories(), events)

    def finish(self, reason: str) -> list[dict[str, Any]]:
        events = self.incidents.close_stream(self.last_t, reason)
        self._emit(events)
        return events

    def fail(self, message: str) -> list[dict[str, Any]]:
        events = [self.incidents.stream_error(self.last_t, self.engine.state.value, message)]
        events += self.incidents.close_stream(self.last_t, message)
        self._emit(events)
        return events

    def _emit(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        self.events.extend(events)
        if self.log:
            self.log.write(events, self.incidents.history)
        for sink in self.sinks:
            for ev in events:
                try:
                    sink.publish(ev)
                except Exception:  # an output sink must never stop monitoring
                    self.sink_errors += 1

    def _snapshot(self, ev: dict[str, Any], frame: np.ndarray) -> None:
        inc = self.incidents.current
        if not (self.out_dir and self.cfg.evidence_snapshots and inc):
            return
        rel = f"evidence/{inc.incident_id}.jpg"
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        inc.snapshot_path = rel
        ev["incident"]["snapshot_path"] = rel


def annotate(frame: np.ndarray, r: FrameResult, dets: list[Detection] | None) -> np.ndarray:
    if frame.ndim == 2:
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:
        img = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        img = np.ascontiguousarray(frame[:, :, :3])
        if img is frame or np.shares_memory(img, frame):
            img = img.copy()
    for d in dets or []:
        x, y, w, h = d.box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, f"{d.label} {d.confidence:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    color = STATE_COLORS[HealthState(r.state)]
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), color, -1)
    label = f"{r.state}  t={r.t:.1f}s"
    if r.suspected_categories:
        # While HEALTHY an active check is only being confirmed (dwell), not yet an incident.
        prefix = "confirming" if r.state == HealthState.HEALTHY.value else "suspected"
        label += f"  {prefix}: " + ", ".join(r.suspected_categories)
    cv2.putText(img, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def establish_baseline(spec: str, cfg: SentinelConfig, max_seconds: float | None = None) -> Baseline:
    metrics: list[FrameMetrics] = []
    with VideoSource(spec, cfg.reconnect_attempts, cfg.reconnect_delay_s) as src:
        for fr in src.frames():
            if max_seconds is not None and fr.t > max_seconds:
                break
            metrics.append(compute_frame_metrics(fr.image))
        fps = src.fps
    return build_baseline(metrics, cfg, redact(spec), fps)


@dataclass
class RunSummary:
    source: str
    input_kind: str
    frames: int
    end_reason: str
    final_state: str
    incidents: int
    mean_latency_ms: float | None
    max_latency_ms: float | None
    corrupt_frames: int
    reconnects: int
    out_dir: str | None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def write_run_manifest(out_dir: Path, spec: str, cfg: SentinelConfig, baseline: Baseline,
                       detector: Detector) -> dict[str, Any]:
    info = {
        "run_id": out_dir.name,
        "stream_id": cfg.stream_id,
        "source": redact(spec),
        "input_kind": classify_input(spec),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "detector": detector.name,
        "detector_source": getattr(detector, "source", "none"),
        "baseline_source": baseline.source,
        "baseline_warnings": baseline.warnings,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def run(
    spec: str,
    cfg: SentinelConfig,
    baseline: Baseline,
    out_dir: str | Path | None = None,
    detector: Detector | None = None,
    realtime: bool = False,
    on_frame: Callable[[FrameResult, np.ndarray, list[Detection] | None], None] | None = None,
    stop: threading.Event | None = None,
    pause: threading.Event | None = None,
    monitor: Monitor | None = None,
    extra_summary: dict[str, Any] | None = None,
    on_open: Callable[[VideoSource], None] | None = None,
) -> RunSummary:
    """Process a source to completion.

    Raises ``SourceError`` if the source cannot be opened (nothing is recorded).
    Once processing starts, a lost stream or an unexpected exception closes any
    open incident as unverified, emits a ``stream_error`` event and writes
    ``summary.json``; unexpected exceptions are then re-raised.
    """
    mon = monitor or Monitor(cfg, baseline, out_dir, detector)
    src = VideoSource(spec, cfg.reconnect_attempts, cfg.reconnect_delay_s).open()
    if on_open:
        on_open(src)
    out = Path(out_dir) if out_dir else None
    if out:
        write_run_manifest(out, spec, cfg, baseline, mon.detector)
    latencies: list[float] = []
    frames = 0
    error: str | None = None
    pending: BaseException | None = None
    wall0 = time.monotonic()
    try:
        for fr in src.frames():
            if stop is not None and stop.is_set():
                src.end_reason = "stopped"
                break
            if pause is not None and pause.is_set():
                paused_at = time.monotonic()
                while pause.is_set() and not (stop is not None and stop.is_set()):
                    time.sleep(0.05)
                wall0 += time.monotonic() - paused_at  # keep real-time pacing after resume
            res = mon.process(fr.index, fr.t, fr.image)
            latencies.append(res.latency_ms)
            frames += 1
            if on_frame:
                on_frame(res, fr.image, mon.last_detections)
            if realtime and not src.live:
                lag = fr.t - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
        mon.finish(src.end_reason or "end_of_stream")
    except StreamLost as exc:
        error = str(exc)
        mon.fail(error)
    except Exception as exc:
        src.end_reason = "internal_error"
        error = f"internal error: {type(exc).__name__}: {exc}"
        mon.fail(error)
        pending = exc
    finally:
        src.close()
    summary = RunSummary(
        source=redact(spec),
        input_kind=classify_input(spec),
        frames=frames,
        end_reason=src.end_reason or "unknown",
        final_state=mon.engine.state.value,
        incidents=len(mon.incidents.history),
        mean_latency_ms=round(float(np.mean(latencies)), 2) if latencies else None,
        max_latency_ms=round(float(np.max(latencies)), 2) if latencies else None,
        corrupt_frames=src.corrupt_frames,
        reconnects=src.reconnects,
        out_dir=str(out_dir) if out_dir else None,
        error=error,
        extra=dict(extra_summary or {}),
    )
    if mon.resolution_mismatch:
        summary.extra["resolution_mismatch"] = list(mon.resolution_mismatch)
    if mon.sink_errors:
        summary.extra["sink_errors"] = mon.sink_errors
    if out:
        (out / "summary.json").write_text(json.dumps(summary.to_dict(), indent=1), encoding="utf-8")
    if pending is not None:
        raise pending
    return summary


__all__ = ["Monitor", "RunSummary", "SourceError", "annotate", "classify_input", "establish_baseline", "run"]
