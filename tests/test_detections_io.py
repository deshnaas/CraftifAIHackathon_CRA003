"""Strict loading and frame association of external detection events.

Uses tests/fixtures/detection_events.TEST_FIXTURE.jsonl: hand-written test data,
not output from PipeGen or any real detector.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from sentinel.cli import main
from sentinel.config import SentinelConfig
from sentinel.detections_io import DetectionEventError, load_detection_events, validate_record
from sentinel.pipeline import Monitor
from sentinel.scenarios import dark

ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "detection_events.TEST_FIXTURE.jsonl"
DET_SCHEMA = json.loads((ROOT / "schema" / "detection_event.schema.json").read_text())
INC_SCHEMA = json.loads((ROOT / "schema" / "incident_event.schema.json").read_text())
FRAME = np.zeros((4, 4, 3), np.uint8)


def rec(**over):
    base = {"schema_version": "1.0", "stream_id": "cam-01", "source": "unit_test", "frame_index": 0,
            "detections": [{"label": "car", "confidence": 0.5, "bbox": [1, 2, 3, 4]}]}
    base.update(over)
    return {k: v for k, v in base.items() if v is not ...}


def write(tmp_path, *records, raw_lines=()):
    p = tmp_path / "dets.jsonl"
    lines = [json.dumps(r) for r in records] + list(raw_lines)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_fixture_conforms_to_published_schema():
    for line in FIXTURE.read_text().splitlines():
        r = json.loads(line)
        jsonschema.validate(r, DET_SCHEMA)
        assert validate_record(r) == []


def test_fixture_loads_and_associates_by_frame_index_and_time():
    feed, report = load_detection_events(FIXTURE, "cam-01", fps=15.0)
    assert report.records_read == 5 and report.records_used == 4
    assert report.other_stream == 1 and report.out_of_order == 1 and report.malformed == 0
    assert feed.detect(FRAME, 0)[0].label == "car"
    assert feed.detect(FRAME, 1) == []  # detector ran, found nothing
    assert len(feed.detect(FRAME, 2)) == 2
    assert feed.detect(FRAME, 3, t=0.2) is None  # no record: unknown, not "zero detections"
    assert feed.detect(FRAME, 15, t=1.0 + 0.02)[0].confidence == 0.7  # nearest timestamp within tolerance
    assert feed.detect(FRAME, 16, t=1.0 + 0.1) is None  # outside tolerance (0.5 / fps)
    assert feed.name == "external:hand_written_test_fixture_not_pipegen"


@pytest.mark.parametrize(("bad", "msg"), [
    ({"frame_index": ...}, "frame_index or timestamp_s is required"),
    ({"frame_index": -1}, "non-negative integer"),
    ({"frame_index": True}, "non-negative integer"),
    ({"frame_index": 1.5}, "non-negative integer"),
    ({"timestamp_s": float("nan")}, "timestamp_s"),
    ({"schema_version": "2.0"}, "schema_version"),
    ({"stream_id": ""}, "stream_id"),
    ({"extra": 1}, "unknown keys"),
    ({"detections": {}}, "detections must be an array"),
    ({"detections": [{"label": "car", "confidence": 1.5, "bbox": [1, 2, 3, 4]}]}, "confidence"),
    ({"detections": [{"label": "car", "confidence": -0.1, "bbox": [1, 2, 3, 4]}]}, "confidence"),
    ({"detections": [{"label": "car", "confidence": "0.9", "bbox": [1, 2, 3, 4]}]}, "confidence"),
    ({"detections": [{"label": "car", "confidence": 0.9, "bbox": [1, 2, 0, 4]}]}, "width and height"),
    ({"detections": [{"label": "car", "confidence": 0.9, "bbox": [1, 2, 3]}]}, "bbox"),
    ({"detections": [{"label": "", "confidence": 0.9, "bbox": [1, 2, 3, 4]}]}, "label"),
    ({"detections": [{"label": "car", "confidence": 0.9, "bbox": [1, 2, 3, 4], "track_id": "a"}]}, "track_id"),
])
def test_invalid_records_rejected(bad, msg):
    errs = validate_record(rec(**bad))
    assert any(msg in e for e in errs), errs


def test_nan_confidence_rejected():
    assert validate_record(rec(detections=[{"label": "a", "confidence": float("nan"), "bbox": [0, 0, 1, 1]}]))


def test_strict_mode_rejects_file_with_line_numbers(tmp_path):
    p = write(tmp_path, rec(frame_index=0), raw_lines=["{not json", json.dumps(rec(frame_index=2, source=""))])
    with pytest.raises(DetectionEventError) as ei:
        load_detection_events(p, "cam-01", 15.0)
    assert "line 2: invalid JSON" in str(ei.value) and "line 3" in str(ei.value)


def test_lenient_mode_skips_bad_records(tmp_path):
    p = write(tmp_path, rec(frame_index=0), raw_lines=["{not json"])
    feed, report = load_detection_events(p, "cam-01", 15.0, strict=False)
    assert report.malformed == 1 and report.records_used == 1
    assert feed.detect(FRAME, 0) is not None


def test_identical_duplicates_dropped_conflicting_rejected(tmp_path):
    p = write(tmp_path, rec(frame_index=0), rec(frame_index=0))
    _, report = load_detection_events(p, "cam-01", 15.0)
    assert report.duplicates_identical == 1 and report.records_used == 1
    other = rec(frame_index=0, detections=[])
    p = write(tmp_path, rec(frame_index=0), other)
    with pytest.raises(DetectionEventError, match="conflicting duplicate"):
        load_detection_events(p, "cam-01", 15.0)
    feed, report = load_detection_events(p, "cam-01", 15.0, strict=False)
    assert report.duplicates_conflicting == 1
    assert len(feed.detect(FRAME, 0)) == 1  # first record wins in lenient mode


def test_out_of_order_records_sorted_and_counted(tmp_path):
    p = write(tmp_path, rec(frame_index=5), rec(frame_index=1), rec(frame_index=3))
    feed, report = load_detection_events(p, "cam-01", 15.0)
    assert report.out_of_order == 2
    assert all(feed.detect(FRAME, i) is not None for i in (1, 3, 5))


def test_timestamp_mismatch_reported(tmp_path):
    p = write(tmp_path, rec(frame_index=30, timestamp_s=0.0))
    _, report = load_detection_events(p, "cam-01", 15.0)
    assert report.timestamp_mismatches == 1


def test_no_records_for_stream_or_missing_file(tmp_path):
    with pytest.raises(DetectionEventError, match="no usable"):
        load_detection_events(write(tmp_path, rec(stream_id="cam-09")), "cam-01", 15.0)
    with pytest.raises(DetectionEventError, match="not found"):
        load_detection_events(tmp_path / "missing.jsonl", "cam-01", 15.0)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(DetectionEventError, match="no usable"):
        load_detection_events(empty, "cam-01", 15.0)


def test_external_detections_attach_to_incident_window(baseline, test_frames, tmp_path):
    # 2 s healthy with 2 detections per frame, then darkness with metadata only on even frames.
    recs = [rec(frame_index=i, detections=[{"label": "car", "confidence": 0.8, "bbox": [0, 0, 5, 5]}] * 2)
            for i in range(30)]
    recs += [rec(frame_index=i, detections=[]) for i in range(30, 90, 2)]
    feed, _ = load_detection_events(write(tmp_path, *recs), "cam-01", 15.0)
    # Smoothing off: the measured onset is then exact. With EMA smoothing the onset (and so
    # the end of the "before" window) lags the true change by roughly smoothing_tau_s.
    mon = Monitor(SentinelConfig(smoothing_tau_s=0.0), baseline, detector=feed)
    frames = test_frames[:30] + [dark(f) for f in test_frames[30:90]]
    events = [e for i, f in enumerate(frames) for e in mon.process(i, i / 15, f).events]
    opened = next(e for e in events if e["event_type"] == "incident_opened")["incident"]
    before = opened["detection_context"]["before"]
    assert before["frames_with_detection_data"] == before["frames"] > 0
    assert before["mean_detections_per_frame"] == 2.0
    during = mon.incidents.current.detection_context["during"]
    assert during["frames"] > during["frames_with_detection_data"] > 0  # gaps counted, not zero-filled
    assert during["detections_total"] == 0
    for e in events:
        jsonschema.validate(e, INC_SCHEMA)


def test_cli_run_with_detections_file(tmp_path, baseline, healthy_frames):
    import cv2
    clip = tmp_path / "c.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
    for f in healthy_frames[:20]:
        w.write(f)
    w.release()
    baseline.save(tmp_path / "b.json")
    rc = main(["run", str(clip), "--baseline", str(tmp_path / "b.json"), "--out", str(tmp_path / "out"),
               "--detections", str(FIXTURE)])
    assert rc == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["extra"]["detections_input"]["records_used"] == 4
    run_info = json.loads((tmp_path / "out" / "run.json").read_text())
    assert run_info["detector_source"] == "external_pipeline"


def test_cli_rejects_invalid_detections_file(tmp_path, baseline, capsys):
    baseline.save(tmp_path / "b.json")
    bad = write(tmp_path, rec(frame_index=0), raw_lines=["[]"])
    rc = main(["run", "x.mp4", "--baseline", str(tmp_path / "b.json"), "--detections", str(bad)])
    assert rc == 2 and "strict mode" in capsys.readouterr().err


def test_cli_rejects_conflicting_detector_options(tmp_path, baseline, capsys):
    baseline.save(tmp_path / "b.json")
    rc = main(["run", "x.mp4", "--baseline", str(tmp_path / "b.json"), "--detections", "a", "--model", "b"])
    assert rc == 2 and "only one of" in capsys.readouterr().err
