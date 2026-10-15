"""Dashboard + JSON API (FastAPI).

GET  /api/health                     liveness
GET  /api/videos                     selectable clips (+ scenario manifest, input labels)
GET  /api/baseline                   current baseline summary and calibration warnings
POST /api/run                        {"video": name | camera index, "realtime": true}
POST /api/pause | /api/resume | /api/stop
GET  /api/status                     state, checks, latency, detection, recent series, run info
GET  /api/runs                       persisted runs (survive restarts)
GET  /api/events?run_id=&since=N     incident event feed (schema/incident_event.schema.json)
GET  /api/incidents?run_id=&stream_id=&status=&category=
GET  /api/incidents/download?run_id= same, as a downloadable file
GET  /api/incidents/{incident_id}    one incident (searches all runs when run_id is omitted)
GET  /api/evidence/{run_id}/{incident_id}.jpg   frame captured when the incident opened
GET  /api/stream.mjpg                annotated live preview
GET  /api/schema                     event JSON schema
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .baseline import Baseline, BaselineError
from .config import SentinelConfig
from .detector import Detection, build_detector
from .mqtt_sink import MqttSink
from .pipeline import FrameResult, Monitor, annotate, classify_input, run, write_run_manifest
from .source import SourceError, VideoSource

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
SCHEMA = ROOT / "schema" / "incident_event.schema.json"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
INCIDENT_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


class RunRequest(BaseModel):
    video: str
    realtime: bool = True


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class RunManager:
    def __init__(self, data_dir: Path, baseline_path: Path, cfg: SentinelConfig, model: str | None,
                 runs_dir: Path, sink: MqttSink | None = None) -> None:
        self.data_dir = data_dir
        self.baseline_path = baseline_path
        self.cfg = cfg
        self.model = model
        self.runs_dir = runs_dir
        self.sink = sink
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.monitor: Monitor | None = None
        self.source: VideoSource | None = None
        self.reset()

    def reset(self) -> None:
        self.status = "idle"
        self.error: str | None = None
        self.video: str | None = None
        self.run_id: str | None = None
        self.input_kind: str | None = None
        self.out_dir: Path | None = None
        self.latest: FrameResult | None = None
        self.jpeg: bytes | None = None
        self.series: deque[dict[str, Any]] = deque(maxlen=450)
        # Whole-run state history, run-length encoded (small: one entry per state change).
        self.timeline: list[dict[str, Any]] = []
        self.duration_s: float | None = None
        self.frame_seq = 0
        self.source = None

    # -- inputs -----------------------------------------------------------
    def baseline(self) -> Baseline:
        return Baseline.load(self.baseline_path)

    def manifest(self) -> dict[str, Any]:
        m = _read_json(self.data_dir / "manifest.json", {})
        return m if isinstance(m, dict) else {}

    def videos(self) -> list[dict[str, Any]]:
        clips = {c["file"]: c for c in self.manifest().get("clips", []) if isinstance(c, dict) and "file" in c}
        files = sorted(p for p in self.data_dir.glob("*") if p.suffix.lower() in VIDEO_EXT) if self.data_dir.is_dir() else []
        return [{"name": p.name, "segments": clips.get(p.name, {}).get("segments"),
                 "input_kind": classify_input(str(p))} for p in files]

    def resolve(self, name: str) -> str:
        if name.isdigit() and len(name) <= 2:
            return name  # USB camera index
        if name not in {v["name"] for v in self.videos()}:
            raise HTTPException(404, f"unknown video: {name}")
        return str(self.data_dir / name)

    # -- run control ------------------------------------------------------
    def _new_run_dir(self) -> Path:
        base = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = self.runs_dir / base
        n = 1
        while out.exists():
            out = self.runs_dir / f"{base}-{n}"
            n += 1
        return out

    def start(self, req: RunRequest) -> None:
        spec = self.resolve(req.video)
        try:
            base = self.baseline()
            detector = build_detector(self.model)
        except (BaselineError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc)) from exc
        self.stop()
        with self.lock:
            self.reset()
            self.video = req.video
            self.input_kind = classify_input(spec)
            self.out_dir = self._new_run_dir()
            self.run_id = self.out_dir.name
            self.monitor = Monitor(self.cfg, base, self.out_dir, detector, sinks=[self.sink] if self.sink else [])
            self.status = "starting"
            self.stop_event = threading.Event()
            self.pause_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, args=(spec, req.realtime, base, self.monitor), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def pause(self) -> None:
        if self.status != "running":
            raise HTTPException(409, f"cannot pause: run is {self.status}")
        self.pause_event.set()
        self.status = "paused"

    def resume(self) -> None:
        if self.status != "paused":
            raise HTTPException(409, f"cannot resume: run is {self.status}")
        self.pause_event.clear()
        self.status = "running"

    def _on_frame(self, r: FrameResult, img: np.ndarray, dets: list[Detection] | None) -> None:
        ok, buf = cv2.imencode(".jpg", annotate(img, r, dets), [cv2.IMWRITE_JPEG_QUALITY, 80])
        vals = {c["check"]: c["value"] for c in r.checks}
        with self.lock:
            self.latest = r
            if ok:
                self.jpeg = buf.tobytes()
                self.frame_seq += 1
            self.series.append({"t": r.t, "state": r.state, **vals})
            if self.timeline and self.timeline[-1]["state"] == r.state:
                self.timeline[-1]["end_t"] = r.t
            else:
                self.timeline.append({"state": r.state, "start_t": r.t, "end_t": r.t})

    def _on_open(self, src: VideoSource) -> None:
        self.source = src
        if not src.live and src.cap is not None:
            frames = src.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.duration_s = round(float(frames) / src.fps, 3) if frames and frames > 0 else None
        if self.status == "starting":
            self.status = "running"

    def _loop(self, spec: str, realtime: bool, base: Baseline, mon: Monitor) -> None:
        try:
            summary = run(spec, self.cfg, base, self.out_dir, realtime=realtime, on_frame=self._on_frame,
                          stop=self.stop_event, pause=self.pause_event, monitor=mon, on_open=self._on_open)
            with self.lock:
                if summary.error:
                    self.status, self.error = "error", summary.error
                else:
                    self.status = "stopped" if summary.end_reason == "stopped" else "finished"
        except SourceError as exc:
            # Nothing was processed; still persist the attempt so it shows in history.
            if self.out_dir:
                write_run_manifest(self.out_dir, spec, self.cfg, base, mon.detector)
            mon.fail(str(exc))
            with self.lock:
                self.status, self.error = "error", str(exc)
        except Exception as exc:  # run() already recorded the failure; surface it in the UI
            with self.lock:
                self.status, self.error = "error", f"internal error: {type(exc).__name__}: {exc}"

    # -- persisted history ------------------------------------------------
    def list_runs(self) -> list[dict[str, Any]]:
        if not self.runs_dir.is_dir():
            return []
        out = []
        for d in sorted((p for p in self.runs_dir.iterdir() if p.is_dir() and RUN_ID.match(p.name)), reverse=True):
            info = _read_json(d / "run.json", None)
            if not isinstance(info, dict):
                continue
            summary = _read_json(d / "summary.json", None)
            incidents = _read_json(d / "incidents.json", [])
            live = d.name == self.run_id and self.status in ("starting", "running", "paused")
            out.append({
                "run_id": d.name,
                "started_at": info.get("started_at"),
                "source": info.get("source"),
                "input_kind": info.get("input_kind"),
                "stream_id": info.get("stream_id"),
                "live": live,
                "complete": isinstance(summary, dict),
                "end_reason": summary.get("end_reason") if isinstance(summary, dict) else None,
                "incidents": len(incidents) if isinstance(incidents, list) else 0,
            })
        return out

    def _run_dir(self, run_id: str | None) -> Path | None:
        if run_id is None:
            if self.run_id:
                return self.runs_dir / self.run_id
            runs = self.list_runs()
            return self.runs_dir / runs[0]["run_id"] if runs else None
        if not RUN_ID.match(run_id) or not (self.runs_dir / run_id / "run.json").is_file():
            raise HTTPException(404, f"unknown run: {run_id}")
        return self.runs_dir / run_id

    def incidents(self, run_id: str | None) -> tuple[str | None, list[dict[str, Any]]]:
        d = self._run_dir(run_id)
        if d is None:
            return None, []
        if d.name == self.run_id and self.monitor is not None:
            return d.name, [i.to_dict() for i in self.monitor.incidents.history]
        data = _read_json(d / "incidents.json", [])
        return d.name, data if isinstance(data, list) else []

    def events(self, run_id: str | None, since: int) -> list[dict[str, Any]]:
        d = self._run_dir(run_id)
        if d is None:
            return []
        if d.name == self.run_id and self.monitor is not None:
            return self.monitor.events[since:]
        try:
            lines = (d / "events.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for ln in lines[since:]:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue  # a truncated last line from an interrupted run
        return out

    def snapshot(self) -> dict[str, Any]:
        try:
            b = self.baseline()
            baseline = {"ok": True, "source": b.source, "created_at": b.created_at, "frames": b.frames,
                        "resolution": [b.width, b.height], "warnings": b.warnings}
        except BaselineError as exc:
            baseline = {"ok": False, "error": str(exc)}
        with self.lock:
            mon = self.monitor
            inc = mon.incidents.current if mon else None
            src = self.source
            return {
                "run_status": self.status,
                "error": self.error,
                "video": self.video,
                "run_id": self.run_id,
                "input_kind": self.input_kind,
                "stream_id": self.cfg.stream_id,
                "detector": mon.detector.name if mon else ("none" if not self.model else f"onnx:{Path(self.model).name}"),
                "frame": None if self.latest is None else self.latest.to_dict() | {"events": None},
                "series": list(self.series),
                "timeline": [dict(seg) for seg in self.timeline],
                "duration_s": self.duration_s,
                "active_incident": inc.to_dict() if inc else None,
                "event_count": len(mon.events) if mon else 0,
                "corrupt_frames": src.corrupt_frames if src else 0,
                "resolution_mismatch": list(mon.resolution_mismatch) if mon and mon.resolution_mismatch else None,
                "health": mon.engine.progress() if mon and self.latest is not None else None,
                "config": {"degrade_after_s": self.cfg.degrade_after_s, "untrusted_after_s": self.cfg.untrusted_after_s,
                           "escalate_after_s": self.cfg.escalate_after_s, "recover_after_s": self.cfg.recover_after_s},
                "baseline": baseline,
                "mqtt": self.sink.status() if self.sink else {"enabled": False},
            }


def create_app(data_dir: Path, baseline_path: Path, cfg: SentinelConfig, model: str | None = None,
               runs_dir: Path | None = None, sink: MqttSink | None = None) -> FastAPI:
    sink = sink or MqttSink(cfg.mqtt, cfg.stream_id)
    mgr = RunManager(data_dir, baseline_path, cfg, model, runs_dir or ROOT / "runs" / "dashboard", sink)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        sink.start()
        yield
        mgr.stop()
        sink.close()

    app = FastAPI(title="Sentinel", version="0.2.0", lifespan=lifespan)
    app.state.manager = mgr

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/videos")
    def videos() -> list[dict[str, Any]]:
        return mgr.videos()

    @app.get("/api/baseline")
    def baseline() -> dict[str, Any]:
        try:
            d = mgr.baseline().to_dict()
        except BaselineError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {k: v for k, v in d.items() if k not in ("cell_std", "reference_thumb")}

    @app.post("/api/run")
    def start(req: RunRequest) -> dict[str, Any]:
        mgr.start(req)
        return {"status": "started", "video": req.video, "run_id": mgr.run_id}

    @app.post("/api/stop")
    def stop() -> dict[str, str]:
        mgr.stop()
        return {"status": mgr.status}

    @app.post("/api/pause")
    def pause() -> dict[str, str]:
        mgr.pause()
        return {"status": mgr.status}

    @app.post("/api/resume")
    def resume() -> dict[str, str]:
        mgr.resume()
        return {"status": mgr.status}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return mgr.snapshot()

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return mgr.list_runs()

    @app.get("/api/events")
    def events(since: int = 0, run_id: str | None = None) -> list[dict[str, Any]]:
        if since < 0:
            raise HTTPException(422, "since must be >= 0")
        return mgr.events(run_id, since)

    def _filtered(run_id: str | None, stream_id: str | None, status: str | None,
                  category: str | None) -> tuple[str | None, list[dict[str, Any]]]:
        rid, incs = mgr.incidents(run_id)
        if stream_id:
            incs = [i for i in incs if i.get("stream_id") == stream_id]
        if status:
            incs = [i for i in incs if i.get("status") == status]
        if category:
            incs = [i for i in incs if category in i.get("categories_seen", [])]
        return rid, incs

    @app.get("/api/incidents")
    def incidents(run_id: str | None = None, stream_id: str | None = None, status: str | None = None,
                  category: str | None = None) -> list[dict[str, Any]]:
        return _filtered(run_id, stream_id, status, category)[1]

    @app.get("/api/incidents/download")
    def download(run_id: str | None = None, stream_id: str | None = None) -> JSONResponse:
        rid, incs = _filtered(run_id, stream_id, None, None)
        name = f"sentinel_incidents_{rid or 'none'}.json"
        return JSONResponse(incs, headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/incidents/{incident_id}")
    def incident(incident_id: str, run_id: str | None = None) -> dict[str, Any]:
        if not INCIDENT_ID.match(incident_id):
            raise HTTPException(404, "unknown incident")
        run_ids = [run_id] if run_id else ([mgr.run_id] if mgr.run_id else []) + [r["run_id"] for r in mgr.list_runs()]
        for rid in dict.fromkeys(run_ids):
            _, incs = mgr.incidents(rid)
            for i in incs:
                if i.get("incident_id") == incident_id:
                    return {"run_id": rid, "incident": i}
        raise HTTPException(404, "unknown incident")

    @app.get("/api/evidence/{run_id}/{incident_id}.jpg")
    def evidence(run_id: str, incident_id: str) -> FileResponse:
        d = mgr._run_dir(run_id)
        if d is None or not INCIDENT_ID.match(incident_id):
            raise HTTPException(404)
        path = d / "evidence" / f"{incident_id}.jpg"
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/evidence/{incident_id}.jpg")
    def evidence_current(incident_id: str) -> FileResponse:
        """Backward-compatible route: evidence from the current (or latest) run."""
        d = mgr._run_dir(None)
        if d is None:
            raise HTTPException(404)
        return evidence(d.name, incident_id)

    @app.get("/api/schema")
    def schema() -> JSONResponse:
        return JSONResponse(json.loads(SCHEMA.read_text(encoding="utf-8")))

    @app.get("/api/stream.mjpg")
    def stream() -> StreamingResponse:
        def gen() -> Iterator[bytes]:
            last = -1
            idle = 0
            while idle < 600:  # close after ~60 s without new frames
                with mgr.lock:
                    seq, jpg = mgr.frame_seq, mgr.jpeg
                if jpg is not None and seq != last:
                    last, idle = seq, 0
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                else:
                    idle += 1
                    time.sleep(0.1)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
