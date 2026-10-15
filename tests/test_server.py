from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import jsonschema
import pytest
from fastapi.testclient import TestClient

from sentinel.config import SentinelConfig
from sentinel.scenarios import dark
from sentinel.server import create_app

SCHEMA = json.loads((Path(__file__).parent.parent / "schema" / "incident_event.schema.json").read_text())


def make_env(tmp_path, baseline, frames):
    data = tmp_path / "data"
    data.mkdir()
    w = cv2.VideoWriter(str(data / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
    for f in frames:
        w.write(f)
    w.release()
    (data / "notes.txt").write_text("not a video")
    baseline.save(tmp_path / "baseline.json")
    app = create_app(data, tmp_path / "baseline.json", SentinelConfig(), runs_dir=tmp_path / "runs")
    return TestClient(app)


def wait_done(client, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get("/api/status").json()
        if s["run_status"] in ("finished", "error", "stopped"):
            return s
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_dashboard_end_to_end(tmp_path, baseline, test_frames):
    frames = test_frames[:30] + [dark(f) for f in test_frames[30:90]]
    with make_env(tmp_path, baseline, frames) as client:
        assert client.get("/").status_code == 200
        assert [v["name"] for v in client.get("/api/videos").json()] == ["clip.mp4"]
        assert client.get("/api/baseline").json()["frames"] == baseline.frames
        assert client.post("/api/run", json={"video": "clip.mp4", "realtime": False}).status_code == 200
        s = wait_done(client)
        assert s["run_status"] == "finished"
        assert s["frame"]["state"] == "UNTRUSTED"
        events = client.get("/api/events").json()
        assert [e["event_type"] for e in events][-1] == "incident_unverified"
        for e in events:
            jsonschema.validate(e, SCHEMA)
        incs = client.get("/api/incidents").json()
        assert len(incs) == 1 and incs[0]["categories_seen"] == ["low_light"]
        r = client.get("/api/incidents/download")
        assert "attachment" in r.headers["content-disposition"]
        assert client.get(f"/api/evidence/{incs[0]['incident_id']}.jpg").headers["content-type"] == "image/jpeg"
        assert client.get("/api/events?since=1").json() == events[1:]


def test_unknown_video_and_path_traversal_rejected(tmp_path, baseline, test_frames):
    with make_env(tmp_path, baseline, test_frames[:5]) as client:
        assert client.post("/api/run", json={"video": "../baseline.json"}).status_code == 404
        assert client.post("/api/run", json={"video": "notes.txt"}).status_code == 404
        assert client.get("/api/evidence/..%2F..%2Fx.jpg").status_code == 404


def test_missing_baseline_reported(tmp_path, baseline, test_frames):
    with make_env(tmp_path, baseline, test_frames[:5]) as client:
        (tmp_path / "baseline.json").unlink()
        r = client.post("/api/run", json={"video": "clip.mp4"})
        assert r.status_code == 409 and "baseline" in r.json()["detail"]


def new_client(tmp_path):
    app = create_app(tmp_path / "data", tmp_path / "baseline.json", SentinelConfig(), runs_dir=tmp_path / "runs")
    return TestClient(app)


def test_incidents_persist_across_server_restart(tmp_path, baseline, test_frames):
    frames = test_frames[:30] + [dark(f) for f in test_frames[30:90]]
    with make_env(tmp_path, baseline, frames) as client:
        client.post("/api/run", json={"video": "clip.mp4", "realtime": False})
        s = wait_done(client)
        run_id = s["run_id"]
        inc_id = client.get("/api/incidents").json()[0]["incident_id"]
    with new_client(tmp_path) as client:  # fresh process state, same runs directory
        assert client.get("/api/status").json()["run_status"] == "idle"
        runs = client.get("/api/runs").json()
        assert runs[0]["run_id"] == run_id and runs[0]["complete"] and runs[0]["incidents"] == 1
        assert runs[0]["input_kind"] == "video_file"
        incs = client.get("/api/incidents").json()  # defaults to the latest persisted run
        assert [i["incident_id"] for i in incs] == [inc_id]
        events = client.get(f"/api/events?run_id={run_id}").json()
        assert events[-1]["event_type"] == "incident_unverified"
        one = client.get(f"/api/incidents/{inc_id}").json()
        assert one["run_id"] == run_id and one["incident"]["stream_id"] == "cam-01"
        assert client.get(f"/api/evidence/{run_id}/{inc_id}.jpg").headers["content-type"] == "image/jpeg"


def test_incident_filters_and_correlation(tmp_path, baseline, test_frames):
    frames = test_frames[:30] + [dark(f) for f in test_frames[30:90]]
    with make_env(tmp_path, baseline, frames) as client:
        client.post("/api/run", json={"video": "clip.mp4", "realtime": False})
        wait_done(client)
        assert len(client.get("/api/incidents?status=unverified").json()) == 1
        assert client.get("/api/incidents?status=resolved").json() == []
        assert len(client.get("/api/incidents?stream_id=cam-01&category=low_light").json()) == 1
        assert client.get("/api/incidents?stream_id=cam-99").json() == []
        assert client.get("/api/incidents/INC-doesnotexist").status_code == 404


def test_api_errors(tmp_path, baseline, test_frames):
    with make_env(tmp_path, baseline, test_frames[:5]) as client:
        assert client.get("/api/incidents?run_id=../../etc").status_code == 404
        assert client.get("/api/incidents?run_id=nope").status_code == 404
        assert client.get("/api/events?since=-1").status_code == 422
        assert client.post("/api/run", json={}).status_code == 422
        assert client.post("/api/run", content=b"not json", headers={"Content-Type": "application/json"}).status_code == 422
        assert client.post("/api/pause").status_code == 409  # nothing running
        assert client.post("/api/resume").status_code == 409
        assert client.get("/api/evidence/run/bad%20id.jpg").status_code == 404
        assert client.get("/api/incidents").json() == []  # empty state, no runs yet
        assert client.get("/api/runs").json() == []
        assert client.get("/api/health").json() == {"ok": True}


def test_pause_and_resume(tmp_path, baseline, test_frames):
    with make_env(tmp_path, baseline, test_frames[:90]) as client:
        client.post("/api/run", json={"video": "clip.mp4", "realtime": True})
        for _ in range(100):
            if client.get("/api/status").json()["run_status"] == "running":
                break
            time.sleep(0.05)
        assert client.post("/api/pause").json()["status"] == "paused"
        t1 = client.get("/api/status").json()["frame"]
        time.sleep(0.6)
        t2 = client.get("/api/status").json()["frame"]
        assert t1 is None or t2["t"] == t1["t"] or t2["t"] - t1["t"] < 0.2  # at most the in-flight frame
        assert client.post("/api/resume").json()["status"] == "running"
        client.post("/api/stop")
        s = wait_done(client)
        assert s["run_status"] == "stopped"
        assert client.get("/api/runs").json()[0]["end_reason"] == "stopped"


def test_failed_source_is_recorded_in_history(tmp_path, baseline, test_frames):
    with make_env(tmp_path, baseline, test_frames[:5]) as client:
        (tmp_path / "data" / "broken.mp4").write_bytes(b"\x00" * 200)
        assert client.post("/api/run", json={"video": "broken.mp4", "realtime": False}).status_code == 200
        s = wait_done(client)
        assert s["run_status"] == "error" and "broken.mp4" in s["error"]
        events = client.get("/api/events").json()
        assert [e["event_type"] for e in events] == ["stream_error"]
        assert client.get("/api/runs").json()[0]["incidents"] == 0


def test_status_reports_baseline_and_input_labels(tmp_path, baseline, test_frames):
    with make_env(tmp_path, baseline, test_frames[:5]) as client:
        s = client.get("/api/status").json()
        assert s["baseline"]["ok"] and s["baseline"]["warnings"] == []
        assert s["mqtt"]["enabled"] is False
        assert client.get("/api/videos").json()[0]["input_kind"] == "video_file"
        (tmp_path / "baseline.json").write_text("{broken")
        s = client.get("/api/status").json()
        assert s["baseline"]["ok"] is False and "cannot load baseline" in s["baseline"]["error"]


def test_status_exposes_health_progress_and_timings(tmp_path, baseline, test_frames):
    frames = test_frames[:30] + [dark(f) for f in test_frames[30:90]]
    with make_env(tmp_path, baseline, frames) as client:
        s = client.get("/api/status").json()
        assert s["health"] is None  # no frames yet: nothing invented
        assert s["config"]["recover_after_s"] == SentinelConfig().recover_after_s
        client.post("/api/run", json={"video": "clip.mp4", "realtime": False})
        s = wait_done(client)
        h = s["health"]
        assert h["state"] == s["frame"]["state"] == "UNTRUSTED"
        assert h["pending"] == [] and h["recovery"] is None
        assert h["time_in_state_s"] >= 0


def test_status_exposes_whole_run_state_timeline(tmp_path, baseline, test_frames):
    frames = test_frames[:30] + [dark(f) for f in test_frames[30:90]] + test_frames[90:180]
    with make_env(tmp_path, baseline, frames) as client:
        s = client.get("/api/status").json()
        assert s["timeline"] == [] and s["duration_s"] is None
        client.post("/api/run", json={"video": "clip.mp4", "realtime": False})
        s = wait_done(client)
        tl = s["timeline"]
        assert [seg["state"] for seg in tl] == ["HEALTHY", "DEGRADED", "UNTRUSTED", "RECOVERING", "HEALTHY"]
        assert tl[0]["start_t"] == 0.0 and tl[-1]["end_t"] == s["frame"]["t"]
        assert all(a["end_t"] <= b["start_t"] for a, b in zip(tl, tl[1:], strict=False))
        assert s["duration_s"] == pytest.approx(len(frames) / 15.0, abs=0.2)
        # consistent with the incident record
        inc = client.get("/api/incidents").json()[0]
        rec = next(t for t in inc["transitions"] if t["to"] == "RECOVERING")
        assert next(seg for seg in tl if seg["state"] == "RECOVERING")["start_t"] == rec["stream_time_s"]
