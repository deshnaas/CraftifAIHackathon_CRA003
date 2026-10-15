"""Incident lifecycle on the full demo timeline, and the progress view the dashboard shows.

The timeline is rebuilt in memory from the same deterministic synthetic scene and
transforms that ``make-demo`` encodes, so these tests do not depend on video codecs.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from sentinel.config import SentinelConfig
from sentinel.health import HealthEngine, HealthState, Thresholds
from sentinel.pipeline import Monitor
from sentinel.scenarios import DEMO_TIMELINE, FPS, TRANSFORMS, synthetic_frames

SCHEMA = json.loads((Path(__file__).parent.parent / "schema" / "incident_event.schema.json").read_text())
EXPECTED = {"dark": "low_light", "blur": "blur", "occlusion": "obstruction", "framing_shift": "framing_change"}


@pytest.fixture(scope="module")
def timeline_run(baseline):
    cfg = SentinelConfig()
    mon = Monitor(cfg, baseline)
    events = []
    progress_seen = []
    total = DEMO_TIMELINE[-1].end_s
    for i, img in enumerate(synthetic_frames(total, t_offset=20.0)):
        t = i / FPS
        seg = next(s for s in DEMO_TIMELINE if s.start_s <= t < s.end_s)
        fn, params = TRANSFORMS[seg.scenario]
        r = mon.process(i, t, fn(img, **params) if fn else img)
        events += r.events
        progress_seen.append(mon.engine.progress())
    mon.finish("end_of_file")
    return cfg, mon, events, progress_seen


def test_every_degradation_runs_the_full_lifecycle(timeline_run):
    cfg, mon, events, _ = timeline_run
    incidents = mon.incidents.history
    degraded_segments = [s for s in DEMO_TIMELINE if s.scenario != "healthy"]
    assert len(incidents) == len(degraded_segments) == 4
    for inc, seg in zip(incidents, degraded_segments, strict=True):
        assert seg.start_s <= inc.opened_stream_time_s < seg.end_s
        assert inc.categories_seen == [EXPECTED[seg.scenario]]
        assert inc.status == "resolved"
        states = [t["to"] for t in inc.transitions]
        assert states[0] == "DEGRADED" and states[-2:] == ["RECOVERING", "HEALTHY"]
        rec = next(t for t in reversed(inc.transitions) if t["to"] == "RECOVERING")
        assert rec["stream_time_s"] >= seg.end_s  # recovery starts only after clean footage resumes
        # resolved exactly when the configured clean-footage period has elapsed (within one frame)
        gap = inc.resolved_stream_time_s - rec["stream_time_s"]
        assert cfg.recover_after_s <= gap < cfg.recover_after_s + 1.5 / FPS


def test_no_incident_during_healthy_segments(timeline_run):
    _, mon, _, _ = timeline_run
    healthy = [s for s in DEMO_TIMELINE if s.scenario == "healthy"]
    for inc in mon.incidents.history:
        for s in healthy:
            # an incident may only *open* inside a degraded segment
            assert not (s.start_s <= inc.opened_stream_time_s < s.end_s)


def test_timeline_events_valid_and_ordered(timeline_run):
    _, _, events, _ = timeline_run
    kinds = [e["event_type"] for e in events]
    assert kinds.count("incident_opened") == kinds.count("incident_resolved") == 4
    assert "incident_unverified" not in kinds
    times = [e["stream_time_s"] for e in events]
    assert times == sorted(times)
    for e in events:
        jsonschema.validate(e, SCHEMA)


def test_progress_reports_confirmation_and_recovery_timers(timeline_run):
    cfg, _, _, progress = timeline_run
    pending = [p for p in progress if p["pending"]]
    assert pending and all(p["state"] == "HEALTHY" for p in pending)
    assert all(0 <= x["held_s"] < cfg.degrade_after_s + 1e-6 for p in pending for x in p["pending"])
    recovering = [p for p in progress if p["recovery"]]
    assert recovering and all(p["state"] == "RECOVERING" for p in recovering)
    assert max(p["recovery"]["clear_for_s"] for p in recovering) < cfg.recover_after_s
    assert all(p["recovery"]["needed_s"] == cfg.recover_after_s for p in recovering)
    assert any(p["escalation"] for p in progress if p["state"] == "DEGRADED")


def test_progress_before_any_frame_and_unit_timers():
    cfg = SentinelConfig(smoothing_tau_s=0.0, degrade_after_s=1.0, recover_after_s=2.0)
    e = HealthEngine(cfg, {"brightness": Thresholds("below", 0.5, 0.7, 0.3)})
    p = e.progress()
    assert p["state"] == "HEALTHY" and p["pending"] == [] and p["time_in_state_s"] == 0.0
    e.update(0.0, {"brightness": 0.4})
    e.update(0.5, {"brightness": 0.4})
    assert e.progress()["pending"][0]["held_s"] == pytest.approx(0.5)
    e.update(1.0, {"brightness": 0.4})
    assert e.state is HealthState.DEGRADED
    e.update(1.5, {"brightness": 0.9})
    e.update(2.5, {"brightness": 0.9})
    rec = e.progress()["recovery"]
    assert rec == {"clear_for_s": 1.0, "needed_s": 2.0}
