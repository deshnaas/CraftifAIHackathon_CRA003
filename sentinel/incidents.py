"""Incident lifecycle and JSON event emission.

An incident opens when the stream leaves HEALTHY, accumulates evidence while it
is DEGRADED / UNTRUSTED / RECOVERING, and is:

* ``resolved``   when the engine returns to HEALTHY after the recovery criteria;
* ``unverified`` when the stream ends or fails before recovery could be observed.

Findings are always labelled *suspected* input degradation: image metrics show
the input changed, not that model accuracy fell.
"""

from __future__ import annotations

import copy
import json
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .health import CATEGORY, HealthEngine, HealthState, Transition, held_for

SCHEMA_VERSION = "1.0"
_ONSET_LOOKBACK_S = 60.0
CLAIM = "suspected_input_degradation"
DISCLAIMER = (
    "Derived from image-quality measurements relative to a healthy baseline. "
    "Indicates the camera input changed measurably; it does not prove that "
    "detection accuracy fell and does not identify the physical cause."
)

ACTIONS = {
    "low_light": "Check scene lighting, IR illuminator and exposure settings; treat detections from this period as untrusted.",
    "blur": "Inspect the lens for dirt, condensation or focus drift; treat detections from this period as untrusted.",
    "obstruction": "Check for an object or debris blocking part of the field of view.",
    "framing_change": "Verify the camera mount and pointing. Re-baseline only after confirming the new view is intended.",
    "processing_latency": "Check host CPU/GPU load and pipeline back-pressure; frames may be dropped or stale.",
}
DEFAULT_ACTION = "Review the live view; keep downstream consumers in untrusted mode until recovery is verified."

SEVERITY = {
    HealthState.DEGRADED: "warning",
    HealthState.UNTRUSTED: "critical",
    HealthState.RECOVERING: "info",
    HealthState.HEALTHY: "info",
}
_SEV_RANK = {"info": 0, "warning": 1, "critical": 2}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass
class Incident:
    incident_id: str
    stream_id: str
    opened_at: str
    opened_stream_time_s: float
    status: str = "active"
    health_state: str = HealthState.DEGRADED.value
    severity: str = "warning"
    peak_severity: str = "warning"
    updated_at: str = ""
    resolved_at: str | None = None
    resolved_stream_time_s: float | None = None
    suspected_categories: list[str] = field(default_factory=list)
    categories_seen: list[str] = field(default_factory=list)
    measurements_at_open: list[dict[str, Any]] = field(default_factory=list)
    latest_measurements: list[dict[str, Any]] = field(default_factory=list)
    worst_values: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    detection: dict[str, Any] | None = None  # detection stats of the latest frame, if a detector exists
    detection_context: dict[str, Any] | None = None  # detection stats before and during the incident
    recurrence_of: str | None = None  # previous incident with overlapping category, if recent
    recurrence_count: int = 0
    snapshot_path: str | None = None
    claim: str = CLAIM
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        # Deep copy: emitted events must be snapshots, not views of a mutating incident.
        return copy.deepcopy(self.__dict__)


class _DetectionWindow:
    """Running detection statistics over a stream-time window. Frames without
    detection data are counted separately and never treated as zero detections."""

    def __init__(self) -> None:
        self.t0: float | None = None
        self.t1: float | None = None
        self.frames = 0
        self.with_data = 0
        self.total = 0

    def add(self, t: float, count: int | None) -> None:
        self.t0 = t if self.t0 is None else self.t0
        self.t1 = t
        self.frames += 1
        if count is not None:
            self.with_data += 1
            self.total += count

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_stream_time_s": None if self.t0 is None else round(self.t0, 3),
            "end_stream_time_s": None if self.t1 is None else round(self.t1, 3),
            "frames": self.frames,
            "frames_with_detection_data": self.with_data,
            "detections_total": self.total,
            "mean_detections_per_frame": round(self.total / self.with_data, 3) if self.with_data else None,
        }


class IncidentManager:
    def __init__(
        self,
        stream_id: str,
        baseline_summary: dict[str, float],
        id_factory: Callable[[str], str] = _new_id,
        clock: Callable[[], str] = _now,
        recurrence_window_s: float = 30.0,
        detection_context_s: float = 5.0,
    ) -> None:
        self.stream_id = stream_id
        self.baseline_summary = baseline_summary
        self.id_factory = id_factory
        self.clock = clock
        self.recurrence_window_s = recurrence_window_s
        self.detection_context_s = detection_context_s
        self.current: Incident | None = None
        self.history: list[Incident] = []
        self._recent: deque[tuple[float, int | None]] = deque()
        self._during: _DetectionWindow | None = None
        self._before: dict[str, Any] | None = None

    # -- events -----------------------------------------------------------
    def _event(self, kind: str, t: float, state: str, message: str, inc: Incident | None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.id_factory("EVT"),
            "event_type": kind,
            "emitted_at": self.clock(),
            "stream_id": self.stream_id,
            "stream_time_s": round(t, 3),
            "health_state": state,
            "message": message,
            "incident": None if inc is None else inc.to_dict(),
        }

    def on_frame(
        self,
        t: float,
        engine: HealthEngine,
        transitions: list[Transition],
        frame_summary: dict[str, Any],
        detection: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        count = None if detection is None else int(detection["count"])
        inc = self.current
        if inc is not None:
            self._accumulate(inc, engine, detection)
            if self._during is not None:
                self._during.add(t, count)
                inc.detection_context = {"before": self._before, "during": self._during.to_dict()}
        for tr in transitions:
            if tr.old is HealthState.HEALTHY and inc is None:
                inc = self._open(t, engine, frame_summary, detection)
                self._start_detection_context(inc, engine, t, count)
                self._record_transition(inc, tr)
                events.append(self._event("incident_opened", t, tr.new.value,
                                          f"{tr.old.value} -> {tr.new.value} ({tr.reason})", inc))
            elif inc is not None and tr.new is HealthState.HEALTHY:
                self._record_transition(inc, tr)
                inc.status = "resolved"
                inc.health_state = tr.new.value
                inc.severity = "info"
                inc.resolved_at = inc.updated_at = self.clock()
                inc.resolved_stream_time_s = round(t, 3)
                inc.evidence.append(f"t={t:.2f}s recovery verified: {tr.reason}")
                events.append(self._event("incident_resolved", t, tr.new.value, tr.reason, inc))
                self.current = inc = None
            elif inc is not None:
                self._record_transition(inc, tr)
                inc.health_state = tr.new.value
                inc.severity = SEVERITY[tr.new]
                if _SEV_RANK[inc.severity] > _SEV_RANK[inc.peak_severity]:
                    inc.peak_severity = inc.severity
                inc.updated_at = self.clock()
                inc.evidence.append(f"t={t:.2f}s {tr.old.value} -> {tr.new.value}: {tr.reason}")
                if tr.new is not HealthState.RECOVERING:
                    self._add_actions(inc, engine.suspected_categories())
                events.append(self._event("state_changed", t, tr.new.value,
                                          f"{tr.old.value} -> {tr.new.value} ({tr.reason})", inc))
        self._recent.append((t, count))
        # Keep enough history to reach back past the degradation onset (which precedes
        # the incident opening by up to the dwell/escalation times).
        while self._recent and t - self._recent[0][0] > self.detection_context_s + _ONSET_LOOKBACK_S:
            self._recent.popleft()
        return events

    def _start_detection_context(self, inc: Incident, engine: HealthEngine, t: float, count: int | None) -> None:
        """'before' covers detection_context_s ending at the degradation onset (earliest
        active check), so it reflects healthy input; 'during' runs from the onset on."""
        starts = [c.active_since for c in engine.checks.values() if c.active and c.active_since is not None]
        onset = min(starts) if starts else t
        before = _DetectionWindow()
        self._during = _DetectionWindow()
        for t_prev, c in self._recent:
            if onset - self.detection_context_s <= t_prev < onset:
                before.add(t_prev, c)
            elif t_prev >= onset:
                self._during.add(t_prev, c)
        self._during.add(t, count)
        self._before = before.to_dict()
        inc.detection_context = {"before": self._before, "during": self._during.to_dict()}

    def close_stream(self, t: float, reason: str) -> list[dict[str, Any]]:
        """Stream ended or failed: an open incident cannot be verified as recovered."""
        inc = self.current
        if inc is None:
            return []
        inc.status = "unverified"
        inc.updated_at = self.clock()
        inc.evidence.append(f"t={t:.2f}s stream closed before recovery was verified: {reason}")
        self.current = None
        return [self._event("incident_unverified", t, inc.health_state, reason, inc)]

    def stream_error(self, t: float, state: str, message: str) -> dict[str, Any]:
        return self._event("stream_error", t, state, message, self.current)

    def stream_warning(self, t: float, state: str, message: str) -> dict[str, Any]:
        return self._event("stream_warning", t, state, message, None)

    # -- helpers ----------------------------------------------------------
    def _open(self, t: float, engine: HealthEngine, frame_summary: dict[str, Any],
              detection: dict[str, Any] | None) -> Incident:
        now = self.clock()
        inc = Incident(
            incident_id=self.id_factory("INC"),
            stream_id=self.stream_id,
            opened_at=now,
            updated_at=now,
            opened_stream_time_s=round(t, 3),
            health_state=engine.state.value,
            severity=SEVERITY[engine.state],
            peak_severity=SEVERITY[engine.state],
            measurements_at_open=engine.snapshot(),
            detection=detection,
        )
        cats = engine.suspected_categories()
        inc.suspected_categories = list(cats)
        inc.categories_seen = list(cats)
        for c in engine.checks.values():
            if c.active and c.smoothed is not None:
                inc.worst_values[c.name] = round(c.smoothed, 4)
                th = c.thresholds
                held = held_for(c.active_since, t)
                inc.evidence.append(
                    f"t={t:.2f}s {c.name}={c.smoothed:.3f} is {th.direction} enter threshold "
                    f"{th.enter:.3f} (held {held:.1f}s; severe at {th.severe:.3f})"
                )
        inc.evidence.append(
            "frame: brightness {b} (baseline {bb}), sharpness {s} (baseline {bs}), {w}x{h}".format(
                b=frame_summary.get("brightness"), bb=round(self.baseline_summary["brightness"], 2),
                s=frame_summary.get("sharpness"), bs=round(self.baseline_summary["sharpness"], 5),
                w=frame_summary.get("width"), h=frame_summary.get("height"),
            )
        )
        self._add_actions(inc, cats)
        self._link_recurrence(inc, t)
        self.current = inc
        self.history.append(inc)
        return inc

    def _link_recurrence(self, inc: Incident, t: float) -> None:
        """Link (not merge) a new incident to a recently resolved one with an overlapping
        category, so a flapping camera shows up as one recurring problem."""
        for prev in reversed(self.history):
            if prev.resolved_stream_time_s is None:
                continue
            if t - prev.resolved_stream_time_s > self.recurrence_window_s:
                return
            if set(prev.categories_seen) & set(inc.categories_seen):
                inc.recurrence_of = prev.incident_id
                inc.recurrence_count = prev.recurrence_count + 1
                inc.evidence.append(
                    f"recurrence: {prev.incident_id} with an overlapping category resolved "
                    f"{t - prev.resolved_stream_time_s:.1f}s earlier (occurrence #{inc.recurrence_count + 1})"
                )
            return

    def _accumulate(self, inc: Incident, engine: HealthEngine, detection: dict[str, Any] | None) -> None:
        inc.latest_measurements = engine.snapshot()
        if detection is not None:
            inc.detection = detection
        for c in engine.checks.values():
            if not c.active or c.smoothed is None:
                continue
            prev = inc.worst_values.get(c.name)
            worse = prev is None or c.thresholds.beyond(c.smoothed, prev)
            if worse:
                inc.worst_values[c.name] = round(c.smoothed, 4)
        for cat in engine.suspected_categories():
            if cat not in inc.categories_seen:
                inc.categories_seen.append(cat)
                self._add_actions(inc, [cat])
        if engine.state is not HealthState.RECOVERING:
            inc.suspected_categories = engine.suspected_categories() or inc.suspected_categories

    @staticmethod
    def _add_actions(inc: Incident, cats: list[str]) -> None:
        for cat in cats or [""]:
            action = ACTIONS.get(cat, DEFAULT_ACTION)
            if action not in inc.recommended_actions:
                inc.recommended_actions.append(action)

    @staticmethod
    def _record_transition(inc: Incident, tr: Transition) -> None:
        inc.transitions.append(
            {"stream_time_s": round(tr.t, 3), "from": tr.old.value, "to": tr.new.value, "reason": tr.reason}
        )


class JsonEventLog:
    """Append-only JSON Lines event log plus a rewritten incidents snapshot."""

    def __init__(self, out_dir: str | Path) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.incidents_path = self.dir / "incidents.json"
        self.events_path.write_text("", encoding="utf-8")

    def write(self, events: list[dict[str, Any]], incidents: list[Incident]) -> None:
        if not events:
            return
        with self.events_path.open("a", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev) + "\n")
        self.write_snapshot(incidents)

    def write_snapshot(self, incidents: list[Incident]) -> None:
        tmp = self.incidents_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([i.to_dict() for i in incidents], indent=1), encoding="utf-8")
        tmp.replace(self.incidents_path)


__all__ = ["CATEGORY", "Incident", "IncidentManager", "JsonEventLog"]
