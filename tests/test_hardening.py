"""Hardening: ingestion failures, frame formats, baseline validation, config typing,
incident recurrence / snapshots, internal errors, resolution changes."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest

from sentinel import source as source_mod
from sentinel.baseline import Baseline, BaselineError, build_baseline
from sentinel.cli import main
from sentinel.config import SentinelConfig
from sentinel.metrics import DARK_PIXEL_LEVEL, compute_frame_metrics
from sentinel.pipeline import Monitor, classify_input, run
from sentinel.scenarios import blur, dark
from sentinel.source import SourceError, VideoSource, redact

SCHEMA = json.loads((Path(__file__).parent.parent / "schema" / "incident_event.schema.json").read_text())


class FakeCapture:
    """Scripted capture: each item is a frame, or None for a failed read."""

    def __init__(self, script, opened=True):
        self.script = list(script)
        self.opened = opened

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 15.0 if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        if not self.script:
            return False, None
        item = self.script.pop(0)
        return (item is not None), item

    def release(self):
        pass


def fake_file(monkeypatch, tmp_path, script):
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"x")  # non-empty so the existence/size checks pass
    monkeypatch.setattr(source_mod.cv2, "VideoCapture", lambda target: FakeCapture(script))
    return str(path)


# -- ingestion ------------------------------------------------------------
def test_isolated_corrupt_frames_are_skipped_not_treated_as_eof(monkeypatch, tmp_path, test_frames):
    good = test_frames[:6]
    script = [good[0], good[1], None, None, good[2], np.zeros((0, 0, 3), np.uint8), good[3], good[4]]
    with VideoSource(fake_file(monkeypatch, tmp_path, script)) as src:
        frames = list(src.frames())
    assert len(frames) == 5
    assert src.corrupt_frames == 3 and src.end_reason == "end_of_file"
    # timestamps keep counting through skipped frames
    assert [f.index for f in frames] == [0, 1, 4, 6, 7]


def test_long_run_of_failures_ends_file(monkeypatch, tmp_path, test_frames):
    script = [test_frames[0]] + [None] * 10 + [test_frames[1]]
    with VideoSource(fake_file(monkeypatch, tmp_path, script)) as src:
        frames = list(src.frames())
    assert len(frames) == 1 and src.end_reason == "end_of_file"


def test_invalid_frame_types_rejected(monkeypatch, tmp_path, test_frames):
    script = [test_frames[0].astype(np.float32), np.zeros((4, 4, 2), np.uint8), test_frames[1]]
    with VideoSource(fake_file(monkeypatch, tmp_path, script)) as src:
        frames = list(src.frames())
    assert len(frames) == 1 and src.corrupt_frames == 2
    assert frames[0].index == 2  # leading corrupt frames still advance the frame clock


def test_empty_file_and_zero_frame_video(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(SourceError, match="empty"):
        VideoSource(str(empty)).open()
    zero = tmp_path / "zero.mp4"
    w = cv2.VideoWriter(str(zero), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (64, 48))
    w.release()
    with pytest.raises(SourceError):
        VideoSource(str(zero)).open()


def test_credentials_redacted_everywhere(monkeypatch):
    url = "rtsp://admin:s3cret@10.0.0.5:554/stream1"
    assert redact(url) == "rtsp://***@10.0.0.5:554/stream1"
    assert redact("rtsp://10.0.0.5/s") == "rtsp://10.0.0.5/s"
    assert redact("C:/videos/a.mp4") == "C:/videos/a.mp4"
    monkeypatch.setattr(source_mod.cv2, "VideoCapture", lambda target: FakeCapture([], opened=False))
    with pytest.raises(SourceError) as ei:
        VideoSource(url).open()
    assert "s3cret" not in str(ei.value) and "admin" not in str(ei.value)


def test_classify_input_labels(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"source": "synthetic (x)", "clips": [{"file": "a.mp4"}]}))
    assert classify_input(str(tmp_path / "a.mp4")) == "synthetic_demo"
    assert classify_input(str(tmp_path / "other.mp4")) == "video_file"
    (tmp_path / "manifest.json").write_text(json.dumps({"source": "my.mp4", "clips": [{"file": "a.mp4"}]}))
    assert classify_input(str(tmp_path / "a.mp4")) == "derived_demo"
    assert classify_input("0") == "camera"
    assert classify_input("rtsp://h/s") == "network_stream"


# -- metrics ----------------------------------------------------------------
def test_dark_pixel_level_boundary():
    img = np.zeros((10, 10), np.uint8)
    img[:5] = DARK_PIXEL_LEVEL - 1
    img[5:] = DARK_PIXEL_LEVEL
    assert compute_frame_metrics(img).dark_fraction == pytest.approx(0.5)


def test_gray_single_channel_and_bgra_frames_supported(healthy_frames):
    bgr = healthy_frames[0]
    ref = compute_frame_metrics(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for img in (gray, gray[:, :, None], cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)):
        m = compute_frame_metrics(img)
        assert m.brightness == pytest.approx(ref.brightness, abs=1.0)
        assert m.sharpness == pytest.approx(ref.sharpness, rel=0.05)


def test_unsupported_frames_rejected():
    with pytest.raises(ValueError, match="dtype"):
        compute_frame_metrics(np.zeros((8, 8, 3), np.float32))
    with pytest.raises(ValueError, match="shape"):
        compute_frame_metrics(np.zeros((8, 8, 2), np.uint8))


# -- baseline -----------------------------------------------------------------
def metrics_of(frames):
    return [compute_frame_metrics(f) for f in frames]


def test_baseline_rejects_dark_footage(healthy_frames):
    with pytest.raises(BaselineError, match="too dark"):
        build_baseline(metrics_of([dark(f) for f in healthy_frames[:40]]), SentinelConfig(), "x", 15.0)


def test_baseline_rejects_mixed_resolution(healthy_frames):
    frames = healthy_frames[:35] + [cv2.resize(healthy_frames[0], (320, 180))]
    with pytest.raises(BaselineError, match="resolution"):
        build_baseline(metrics_of(frames), SentinelConfig(), "x", 15.0)


def test_baseline_warns_on_unstable_lighting(healthy_frames):
    frames = [f if i % 2 else cv2.convertScaleAbs(f, alpha=0.5) for i, f in enumerate(healthy_frames[:40])]
    base = build_baseline(metrics_of(frames), SentinelConfig(), "x", 15.0)
    assert any("lighting varied" in w for w in base.warnings)


def test_baseline_warns_on_dynamic_scene(healthy_frames):
    frames = [f if i % 2 else np.ascontiguousarray(f[::-1, ::-1]) for i, f in enumerate(healthy_frames[:40])]
    base = build_baseline(metrics_of(frames), SentinelConfig(), "x", 15.0)
    assert any("highly dynamic" in w for w in base.warnings)


def test_clean_baseline_has_no_warnings(baseline):
    assert baseline.warnings == []


def test_damaged_baseline_file_rejected(baseline, tmp_path):
    d = baseline.to_dict()
    d["cell_std"] = [[1.0, 2.0]]
    p = tmp_path / "b.json"
    p.write_text(json.dumps(d))
    with pytest.raises(BaselineError, match="cell_std"):
        Baseline.load(p)
    d = baseline.to_dict()
    d["brightness"] = 0
    p.write_text(json.dumps(d))
    with pytest.raises(BaselineError, match="brightness"):
        Baseline.load(p)
    d = baseline.to_dict()
    del d["warnings"]  # files written before warnings existed still load
    p.write_text(json.dumps(d))
    assert Baseline.load(p).warnings == []


# -- config -------------------------------------------------------------------
@pytest.mark.parametrize(("data", "msg"), [
    ({"smoothing_tau_s": "fast"}, "finite number"),
    ({"smoothing_tau_s": float("inf")}, "finite number"),
    ({"degrade_after_s": -1}, ">= 0"),
    ({"reconnect_attempts": 1.5}, "non-negative integer"),
    ({"reconnect_attempts": True}, "non-negative integer"),
    ({"baseline_min_frames": 1}, ">= 2"),
    ({"evidence_snapshots": "yes"}, "true or false"),
    ({"stream_id": ""}, "stream_id"),
    ({"checks": {"brightness": {"nope": 1}}}, "unknown keys"),
    ({"checks": {"brightness": {"enter": "x"}}}, "finite number"),
    ({"checks": {"framing": {"direction": "above"}}}, "framing"),
    ({"mqtt": {"port": 0}}, "port"),
    ({"mqtt": {"qos": 3}}, "qos"),
    ({"mqtt": {"topic": "a/#"}}, "wildcards"),
    ({"mqtt": {"password_env": "hunter2!"}}, "NAME of an environment variable"),
    ({"mqtt": {"password": "x"}}, "unknown keys"),
])
def test_config_rejects_bad_values(data, msg):
    with pytest.raises(ValueError, match=msg):
        SentinelConfig.from_dict(data)


def test_config_file_errors(tmp_path, capsys):
    bad = tmp_path / "c.json"
    bad.write_text("{nope")
    assert main(["--config", str(bad), "baseline", "x.mp4"]) == 2
    assert "not valid JSON" in capsys.readouterr().err


# -- incidents ----------------------------------------------------------------
def play(mon, frames, start=0):
    return [e for i, f in enumerate(frames, start) for e in mon.process(i, i / 15, f).events]


def test_emitted_events_are_immutable_snapshots(baseline, test_frames):
    mon = Monitor(SentinelConfig(), baseline)
    events = play(mon, [dark(f) for f in test_frames[:60]])
    opened = next(e for e in events if e["event_type"] == "incident_opened")
    n_evidence = len(opened["incident"]["evidence"])
    play(mon, test_frames[60:150], 60)  # recovery adds evidence and transitions to the live incident
    assert len(mon.incidents.history[0].evidence) > n_evidence
    assert len(opened["incident"]["evidence"]) == n_evidence  # the earlier event did not change


def test_relapse_stays_one_incident_and_recurrence_is_linked(baseline, test_frames):
    mon = Monitor(SentinelConfig(), baseline)
    events = play(mon, [dark(f) for f in test_frames[:45]])
    events += play(mon, test_frames[45:60], 45)                        # 1 s clear: RECOVERING
    events += play(mon, [dark(f) for f in test_frames[60:90]], 60)     # relapse -> same incident
    assert len(mon.incidents.history) == 1
    events += play(mon, test_frames[90:150], 90)                       # resolved
    events += play(mon, [dark(f) for f in test_frames[150:195]], 150)  # recurs within 30 s
    assert len(mon.incidents.history) == 2
    first, second = mon.incidents.history
    assert first.status == "resolved" and second.recurrence_of == first.incident_id
    assert second.recurrence_count == 1
    for e in events:
        jsonschema.validate(e, SCHEMA)


def test_unrelated_category_is_not_linked(baseline, test_frames):
    mon = Monitor(SentinelConfig(), baseline)
    play(mon, [dark(f) for f in test_frames[:45]])
    play(mon, test_frames[45:120], 45)
    play(mon, [blur(f) for f in test_frames[120:165]], 120)
    assert mon.incidents.history[1].recurrence_of is None


def test_escalation_tracked_in_peak_severity(baseline, test_frames):
    mon = Monitor(SentinelConfig(), baseline)
    play(mon, [dark(f) for f in test_frames[:60]])
    inc = mon.incidents.current
    assert inc.peak_severity == "critical"
    assert [t["to"] for t in inc.transitions] == ["DEGRADED", "UNTRUSTED"]


def test_resolution_change_emits_one_warning(baseline, test_frames):
    mon = Monitor(SentinelConfig(), baseline)
    small = [cv2.resize(f, (320, 180)) for f in test_frames[:20]]
    events = play(mon, small)
    warnings = [e for e in events if e["event_type"] == "stream_warning"]
    assert len(warnings) == 1 and "re-baseline" in warnings[0]["message"]
    jsonschema.validate(warnings[0], SCHEMA)


class ExplodingDetector:
    name = "exploding"
    source = "real_inference"

    def __init__(self, after):
        self.after = after

    def detect(self, frame, index=0, t=None):
        if index >= self.after:
            raise RuntimeError("model crashed")
        return []


def test_unexpected_exception_is_recorded_then_raised(tmp_path, baseline, test_frames):
    clip = tmp_path / "c.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
    for f in [dark(f) for f in test_frames[:45]]:
        w.write(f)
    w.release()
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="model crashed"):
        run(str(clip), SentinelConfig(), baseline, out, ExplodingDetector(after=40))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["end_reason"] == "internal_error" and "model crashed" in summary["error"]
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    assert [e["event_type"] for e in events][-2:] == ["stream_error", "incident_unverified"]


def test_cli_exit_code_3_on_internal_error(tmp_path, baseline, monkeypatch, capsys):
    import sentinel.cli as cli_mod

    def boom(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(cli_mod, "run", boom)
    baseline.save(tmp_path / "b.json")
    assert main(["run", "x.mp4", "--baseline", str(tmp_path / "b.json"), "--out", str(tmp_path / "o")]) == 3
    assert "internal error" in capsys.readouterr().err


def test_make_demo_is_deterministic_and_builds_baseline(tmp_path):
    from sentinel.scenarios import DEMO_TIMELINE, synthetic_frames
    a = list(synthetic_frames(1.0, t_offset=3.0))
    b = list(synthetic_frames(1.0, t_offset=3.0))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    scen = [s.scenario for s in DEMO_TIMELINE]
    for name in ("dark", "blur", "occlusion", "framing_shift"):
        i = scen.index(name)
        assert scen[i + 1] == "healthy"  # every degradation is followed by a recovery segment
    rc = main(["make-demo", "--out", str(tmp_path / "demo"), "--baseline-out", str(tmp_path / "b.json")])
    assert rc == 0
    manifest = json.loads((tmp_path / "demo" / "manifest.json").read_text())
    assert manifest["synthetic"] is True and "not physical camera faults" in manifest["note"]
    assert Baseline.load(tmp_path / "b.json").frames == 300
