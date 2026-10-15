from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from sentinel.cli import main
from sentinel.config import SentinelConfig
from sentinel.detector import Detection, ExternalDetections, parse_pipegen_metadata
from sentinel.pipeline import Monitor
from sentinel.scenarios import dark

SCHEMA = json.loads((Path(__file__).parent.parent / "schema" / "incident_event.schema.json").read_text())


def test_external_detections_are_looked_up_by_frame_index(baseline, test_frames):
    # Test fixture only: hand-built detections, not a PipeGen output format.
    ext = ExternalDetections({0: [Detection("person", 0.9, (1, 2, 3, 4))], 1: []}, name="fixture")
    mon = Monitor(SentinelConfig(), baseline, detector=ext)
    r0 = mon.process(0, 0.0, test_frames[0])
    r1 = mon.process(1, 1 / 15, test_frames[1])
    r2 = mon.process(2, 2 / 15, test_frames[2])
    assert r0.detection == {"detector": "fixture", "source": "external_pipeline", "count": 1,
                            "mean_confidence": 0.9, "by_label": {"person": 1}}
    assert r1.detection is not None and r1.detection["count"] == 0
    assert r2.detection is None  # no metadata for this frame: unknown, not "nothing detected"


def test_external_detections_attached_to_incidents(baseline, test_frames):
    frames = [dark(f) for f in test_frames[:45]]
    ext = ExternalDetections({i: [Detection("car", 0.5, (0, 0, 10, 10))] for i in range(45)}, name="fixture")
    mon = Monitor(SentinelConfig(), baseline, detector=ext)
    events = [e for i, f in enumerate(frames) for e in mon.process(i, i / 15, f).events]
    opened = [e for e in events if e["event_type"] == "incident_opened"]
    assert opened and opened[0]["incident"]["detection"]["source"] == "external_pipeline"
    for e in events:
        jsonschema.validate(e, SCHEMA)


def test_pipegen_parser_refuses_until_format_verified(tmp_path):
    with pytest.raises(NotImplementedError, match="not verified"):
        parse_pipegen_metadata(tmp_path / "meta.json")


def test_cli_reports_unimplemented_pipegen_parser(tmp_path, baseline, capsys):
    baseline.save(tmp_path / "b.json")
    rc = main(["run", "x.mp4", "--baseline", str(tmp_path / "b.json"), "--pipegen-metadata", "meta.json"])
    assert rc == 2
    assert "not verified" in capsys.readouterr().err
