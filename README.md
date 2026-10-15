# Sentinel

**Reliability monitoring for edge computer-vision pipelines.** A camera can stay online while its input becomes less trustworthy, because of darkness, blur, obstruction or a changed view. Sentinel measures this against a healthy baseline and raises evidence-backed incidents as JSON. It tracks recovery explicitly and resolves an incident only once recovery has been observed.

> **Status: local software MVP (CPU).** It is **not** a validated CraftifAI Track A submission.
> PipeGen, TensorRT and the live NVIDIA run have not been done (no NVIDIA hardware was available).
> See [REQUIREMENTS_STATUS.md](REQUIREMENTS_STATUS.md) and [GPU_VALIDATION.md](GPU_VALIDATION.md).
>
> **Demo evidence (synthetic footage):** [evidence/DEMO_REPORT.md](evidence/DEMO_REPORT.md): screenshots of every health state, the raw incident JSON from a real dashboard run, measured detection and recovery times, and a reproduction script (`scripts/make_evidence.ps1`).

Workflow: **Observe → Measure → Diagnose → Respond → Verify recovery**.

## Quick start (Windows)

Run these from the project folder, in PowerShell or Git Bash. On Linux or macOS, use `.venv/bin/python`.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

**Generate the demo data** (synthetic clips, manifest and baseline) with one command:

```bash
.venv/Scripts/python -m sentinel make-demo
```

**Launch the application** (dashboard and API at http://127.0.0.1:8000):

```bash
.venv/Scripts/python -m sentinel serve
```

In the dashboard, pick `demo_timeline.mp4` and press **Run / replay**.

### Other commands

| Command | What it does |
|---|---|
| `python -m sentinel baseline VIDEO [--seconds N] [--out data/baseline.json]` | Calibrate on footage you declare healthy. Rejects dark, blank or mixed-resolution footage, and warns about unstable lighting or highly dynamic scenes. |
| `python -m sentinel run SOURCE [--out DIR] [--realtime]` | Monitor a file, a camera index (`0`) or an `rtsp://` URL. Writes `run.json`, `events.jsonl`, `incidents.json`, `summary.json` and `evidence/*.jpg`. |
| `... run SOURCE --detections FILE.jsonl [--detections-lenient]` | Attach detections produced elsewhere, in Sentinel's [detection-event schema](schema/detection_event.schema.json). Validation is strict by default. |
| `... run SOURCE --model yolov8n.onnx --labels coco.txt` | Optional CPU detector through OpenCV DNN. No weights are bundled and this path is **untested**. |
| `... run SOURCE --pipegen-metadata FILE` | Reserved. It refuses to run until a converter is written against a genuine PipeGen output sample. |
| `python -m sentinel make-demo --source my_healthy.mp4` | Derive the same scenarios from your own footage. The clips are labelled `derived_demo`. |
| `python -m sentinel --config cfg.json …` | Override thresholds and timings, and enable MQTT. Example: `{"checks": {"brightness": {"enter": 0.5}}, "recover_after_s": 5}`. |

Exit codes: `0` ok, `2` invalid input (missing or corrupt source, bad config, baseline or detections file), `3` unexpected internal error. Events and the summary are still written in that case.

Checks: `.venv/Scripts/python -m pytest -q` · `.venv/Scripts/ruff check sentinel tests` · `.venv/Scripts/mypy sentinel`

## Demo script (about 3 minutes)

1. Start `serve`, select `demo_timeline.mp4` and press **Run / replay**. The preview shows a **SYNTHETIC DEMO FOOTAGE** badge.
2. At 0–8 s the stream is HEALTHY.
3. At 8 s the footage is darkened. The state goes to DEGRADED, then UNTRUSTED ("severity: critical"), with *suspected low_light* and the recommended action.
4. At 16 s normal footage resumes. The state shows RECOVERING for 3 s, then HEALTHY, and the incident becomes *resolved*.
5. The same pattern repeats for blur (24 s), occlusion (40 s) and camera shift (56 s). The panel under the video shows the ground-truth transform from the manifest.
6. Click **Pause** and then **Resume** (stream time freezes and continues). Expand an incident to see its values, thresholds, evidence and snapshot. Click **View JSON**, **Download JSON** or **Event feed**.
7. Play `clip_dark.mp4`. It ends while still dark, so the incident is marked **unverified**, not resolved.
8. Restart `serve`. Past runs and their incidents are still listed in the run selector.

## Outputs

- **Incident events:** [`schema/incident_event.schema.json`](schema/incident_event.schema.json) (v1.0). Events are `incident_opened`, `state_changed`, `incident_resolved`, `incident_unverified`, `stream_error` and `stream_warning`. Every incident carries `claim: "suspected_input_degradation"`.
- **Persistence:** each run has its own folder under `runs/` (the dashboard uses `runs/dashboard/<run_id>/`). It holds `events.jsonl` (append-only) and `incidents.json` (rewritten atomically). Runs survive restarts.
- **API:**
  - `/api/runs`
  - `/api/incidents?run_id=&stream_id=&status=&category=`
  - `/api/incidents/{incident_id}`
  - `/api/incidents/download`
  - `/api/events?run_id=&since=N`
  - `/api/evidence/{run_id}/{incident_id}.jpg`
  - `/api/status`, `/api/pause`, `/api/resume`, `/api/stop`, `/api/stream.mjpg`, `/api/schema`, `/api/health`
- **MQTT (optional, off by default):** set `"mqtt": {"enabled": true, "host": "…", "topic": "sentinel/{stream_id}/events"}` in the config.
  - Credentials come only from the environment variables named in `username_env` and `password_env`.
  - If the broker is down, monitoring continues and failures are counted.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). In short:
1. source
2. per-frame metrics
3. baseline-relative check values
4. EMA smoothing, hysteresis and dwell-timer state machine
5. incident manager
6. JSONL/JSON persistence, then the API, dashboard and optional MQTT

## Limitations

- **No accuracy claims.** Incidents mean the input changed measurably compared with the baseline. They do not show that detection accuracy dropped. No ground-truth evaluation has been done.
- **Synthetic validation only.** The scenarios are digital transforms of a procedurally generated scene. Real lens dirt, real camera movement and real night conditions have not been tested. Expect to tune thresholds per camera.
- **Proxies.**
  - Obstruction catches only occluders flatter than the scene.
  - Framing change can also fire on large scene-content changes.
  - Global brightness can miss local darkness.
  - Sensor noise can mask blur.
- **Detector.** No real detector ran. The ONNX backend is untested. External detections were tested only with a hand-written fixture.
- **Live sources.** Camera and RTSP handling (reconnect, corrupt frames) was tested only with simulated captures. MQTT was tested with a fake client and against an unreachable port, not against a real broker.
- **Baseline is static.** Re-baseline deliberately after an intended view change.
- **No GPU path.** Sentinel does not use PipeGen, TensorRT or DeepStream yet. See GPU_VALIDATION.md.

## Documents

[evidence/DEMO_REPORT.md](evidence/DEMO_REPORT.md) · [evidence/ARCHITECTURE.md](evidence/ARCHITECTURE.md) · [REQUIREMENTS_STATUS.md](REQUIREMENTS_STATUS.md) · [TEST_RESULTS.md](TEST_RESULTS.md) · [GPU_VALIDATION.md](GPU_VALIDATION.md) · [ARCHITECTURE.md](ARCHITECTURE.md)
