from __future__ import annotations

import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest

from sentinel import source as source_mod
from sentinel.config import SentinelConfig
from sentinel.incidents import ACTIONS
from sentinel.pipeline import Monitor, run
from sentinel.scenarios import blur, dark, occlude, shift
from sentinel.source import SourceError, StreamLost, VideoSource

SCHEMA = json.loads((Path(__file__).parent.parent / "schema" / "incident_event.schema.json").read_text())
FPS = 15.0


def play(mon: Monitor, frames, start_index: int = 0) -> list[dict]:
    events = []
    for i, f in enumerate(frames, start_index):
        events += mon.process(i, i / FPS, f).events
    return events


def validate(events: list[dict]) -> None:
    for ev in events:
        jsonschema.validate(ev, SCHEMA)
        json.loads(json.dumps(ev))  # fully JSON-serialisable


@pytest.mark.parametrize(
    ("transform", "category", "expected_state"),
    [(dark, "low_light", "UNTRUSTED"), (blur, "blur", "UNTRUSTED"),
     (occlude, "obstruction", "DEGRADED"), (shift, "framing_change", "DEGRADED")],
)
def test_each_scenario_opens_matching_incident(baseline, test_frames, ids, transform, category, expected_state):
    mon = Monitor(SentinelConfig(), baseline, id_factory=ids)
    events = play(mon, test_frames[:45])  # 3 s healthy
    assert events == [] and mon.engine.state.value == "HEALTHY"
    events = play(mon, [transform(f) for f in test_frames[45:105]], 45)  # 4 s degraded
    opened = [e for e in events if e["event_type"] == "incident_opened"]
    assert len(opened) == 1
    inc = opened[0]["incident"]
    assert inc["suspected_categories"] == [category]
    assert inc["status"] == "active"
    assert inc["claim"] == "suspected_input_degradation"
    assert inc["recommended_actions"] == [ACTIONS[category]]
    assert any("enter threshold" in ev for ev in inc["evidence"])
    assert mon.engine.state.value == expected_state
    validate(events)


def test_recovery_resolves_incident_only_after_recovery_window(baseline, test_frames, ids):
    cfg = SentinelConfig()
    mon = Monitor(cfg, baseline, id_factory=ids)
    play(mon, test_frames[:30])
    play(mon, [dark(f) for f in test_frames[30:90]], 30)
    inc = mon.incidents.current
    assert inc is not None
    events = play(mon, test_frames[90:210], 90)  # 8 s healthy again
    kinds = [e["event_type"] for e in events]
    assert kinds == ["state_changed", "incident_resolved"]
    recovering_t = events[0]["stream_time_s"]
    resolved_t = events[1]["stream_time_s"]
    assert resolved_t - recovering_t >= cfg.recover_after_s - 1e-6
    assert inc.status == "resolved" and mon.incidents.current is None
    assert [t["to"] for t in inc.transitions] == ["DEGRADED", "UNTRUSTED", "RECOVERING", "HEALTHY"]
    validate(events)


def test_end_of_stream_marks_active_incident_unverified(baseline, test_frames, ids):
    mon = Monitor(SentinelConfig(), baseline, id_factory=ids)
    play(mon, [blur(f) for f in test_frames[:45]])
    events = mon.finish("end_of_file")
    assert [e["event_type"] for e in events] == ["incident_unverified"]
    assert events[0]["incident"]["status"] == "unverified"
    assert mon.finish("again") == []
    validate(events)


def test_incident_ids_unique_and_history_persisted(baseline, test_frames, tmp_path):
    mon = Monitor(SentinelConfig(), baseline, out_dir=tmp_path)
    play(mon, [dark(f) for f in test_frames[:45]])
    play(mon, test_frames[45:120], 45)
    play(mon, [blur(f) for f in test_frames[120:165]], 120)
    mon.finish("end_of_file")
    history = json.loads((tmp_path / "incidents.json").read_text())
    assert len({i["incident_id"] for i in history}) == len(history) == 2
    assert [i["status"] for i in history] == ["resolved", "unverified"]
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    validate([json.loads(x) for x in lines])
    snap = history[0]["snapshot_path"]
    assert snap and (tmp_path / snap).is_file()


def test_no_detector_means_no_detection_stats(baseline, test_frames):
    mon = Monitor(SentinelConfig(), baseline)
    r = mon.process(0, 0.0, test_frames[0])
    assert r.detection is None


# -- sources --------------------------------------------------------------
def write_clip(path: Path, frames) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        w.write(f)
    w.release()
    return path


def test_missing_file_raises_source_error(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        VideoSource(str(tmp_path / "nope.mp4")).open()


def test_corrupt_file_raises_source_error(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    with pytest.raises(SourceError):
        VideoSource(str(bad)).open()


def test_run_with_invalid_source_raises(baseline, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 1000)
    with pytest.raises(SourceError):
        run(str(bad), SentinelConfig(), baseline, tmp_path / "out")


def test_file_end_of_stream(tmp_path, healthy_frames):
    clip = write_clip(tmp_path / "c.mp4", healthy_frames[:20])
    with VideoSource(str(clip)) as src:
        frames = list(src.frames())
    assert len(frames) == 20
    assert src.end_reason == "end_of_file"
    assert frames[-1].t == pytest.approx(19 / FPS)


def test_run_summary_on_file(tmp_path, healthy_frames, baseline):
    clip = write_clip(tmp_path / "c.mp4", healthy_frames[:30])
    s = run(str(clip), SentinelConfig(), baseline, tmp_path / "out")
    assert (s.frames, s.end_reason, s.final_state, s.incidents) == (30, "end_of_file", "HEALTHY", 0)
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["frames"] == 30


class FlakyCapture:
    """Fake live camera: delivers N frames then fails permanently."""

    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 0.0

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        pass


def test_live_stream_loss_reconnect_then_stream_lost(monkeypatch, test_frames):
    caps = [FlakyCapture(test_frames[:5]), FlakyCapture([], opened=False), FlakyCapture([], opened=False)]
    monkeypatch.setattr(source_mod.cv2, "VideoCapture", lambda target: caps.pop(0))
    monkeypatch.setattr(source_mod.time, "sleep", lambda s: None)
    src = VideoSource("0", reconnect_attempts=2, reconnect_delay_s=0).open()
    got = []
    with pytest.raises(StreamLost):
        for fr in src.frames():
            got.append(fr)
    assert len(got) == 5 and src.end_reason == "stream_lost"


def test_live_stream_reconnects_successfully(monkeypatch, test_frames):
    caps = [FlakyCapture(test_frames[:3]), FlakyCapture(test_frames[3:6])]
    monkeypatch.setattr(source_mod.cv2, "VideoCapture", lambda target: caps.pop(0) if caps else FlakyCapture([], False))
    monkeypatch.setattr(source_mod.time, "sleep", lambda s: None)
    src = VideoSource("rtsp://example.invalid/stream", reconnect_attempts=1, reconnect_delay_s=0).open()
    got = []
    with pytest.raises(StreamLost):
        for fr in src.frames():
            got.append(fr)
    # 3 frames from the first connection + 3 after reconnecting, then reconnection fails
    assert len(got) == 6 and src.end_reason == "stream_lost"


def fake_clock(step: float):
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += step
        return state["t"]
    return staticmethod(clock)


def test_run_records_stream_error_and_unverified_incident(monkeypatch, baseline, test_frames, tmp_path):
    frames = [dark(f) for f in test_frames[:45]]
    monkeypatch.setattr(source_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(VideoSource, "clock", fake_clock(1 / FPS))  # simulate 15 fps wall time
    cfg = SentinelConfig(reconnect_attempts=1, reconnect_delay_s=0)
    real = FlakyCapture(frames)
    calls = {"n": 0}

    def factory(target):
        calls["n"] += 1
        return real if calls["n"] == 1 else FlakyCapture([], False)

    monkeypatch.setattr(source_mod.cv2, "VideoCapture", factory)
    s = run("0", cfg, baseline, tmp_path)
    assert s.end_reason == "stream_lost"
    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    kinds = [e["event_type"] for e in events]
    assert "stream_error" in kinds and kinds[-1] == "incident_unverified"
    validate(events)


def test_live_source_timestamps_monotonic(monkeypatch, test_frames):
    monkeypatch.setattr(source_mod.cv2, "VideoCapture", lambda target: FlakyCapture(test_frames[:4]))
    src = VideoSource("1", reconnect_attempts=0).open()
    with pytest.raises(StreamLost):
        ts = []
        for fr in src.frames():
            ts.append(fr.t)
    assert ts == sorted(ts)
    assert np.all(np.diff(ts) >= 0)
