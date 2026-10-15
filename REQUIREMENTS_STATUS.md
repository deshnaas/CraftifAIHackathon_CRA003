# Requirements status

> **Current result: local prototype. It is NOT a validated Track A submission.**
> None of the four mandatory Track A requirements has been met, because they all require
> PipeGen v0.5.1 running on an NVIDIA GPU or a Jetson, and neither was available.

Last updated: 2026-10-15.

## Sources consulted

| Document | Access | Notes |
|---|---|---|
| *Embedded SF: CraftifAI Physical AI Hackathon — Official Problem Statement* (Google Doc in the user's Drive) | ✅ Read | Defines Tracks A and B, mandatory requirements, judging and the setup checklist. |
| PipeGen v0.5.1 Lab (`docs.google.com/document/d/15HpN0QE…`) | ❌ **Not accessible** | Not shared with the signed-in Google account: Drive returns "not found", and the web link redirects to sign-in. The lab's real syntax, IR format, commands and outputs are therefore **unverified**. Ask the organisers to share it. |
| craftifai.com homepage and `/products/pipegen` | ✅ Read | Supported hardware (NVIDIA Turing or newer, x86_64 or Jetson), the five inputs and outputs, build-time-only usage, and the benchmark report contents. See GPU_VALIDATION.md §1. |
| `developer.craftifai.com/docs/pipegen` | ❌ 404 | Linked from the product page, but returned "Page not found" on 2026-10-15. |
| `github.com/craftifai/PipeGen_Benchmark_Repositories` | ✅ Listed | Source pipelines for PipeGen to ingest, including ONNX-YOLOv8. These are **not** generated outputs, so they don't show the JSON metadata format. |
| `ESP32_BLE_FirmGen_Simple_PRD.pdf` | ✅ Read | ESP32 firmware PRD: BLE advertising as `FirmGen_XX`, UART logs at 115200 baud. |

## PRD / track mismatch: resolved

The ESP32 BLE PRD is a **FirmGen (Track B)** sample. It specifies firmware for an ESP32 board (BLE advertising plus UART logs) and nothing about vision.
The problem statement lists it on the Track A line next to the PipeGen lab, but its content belongs to Track B.
**It does not apply to Sentinel's Track A submission**, and it has not been merged into these requirements.
It would become relevant only for the optional *Closed Loop* bonus, which needs FirmGen v0.4.1, ESP-IDF v5.5 and a physical ESP32. That bonus is out of scope here.

## Track A mandatory requirements

| # | Requirement (official wording, summarised) | Status | Evidence | Blocker |
|---|---|---|---|---|
| A1 | Full PipeGen flow: Requirement → Pipeline IR → Model → Compiler → Run | ❌ Not started | none | PipeGen v0.5.1 needs x86-64 Linux with an NVIDIA GPU (or a Jetson on JetPack 6), an X11 desktop and a CraftifAI account. The lab document could not be read. |
| A2 | At least one model compiled to a TensorRT engine, with PipeGen's speed report | ❌ Not started | none | Needs an NVIDIA GPU and TensorRT. This laptop has Intel Arc graphics. |
| A3 | Live run on a video file, USB, CSI or RTSP source on an NVIDIA GPU or Jetson | ❌ Not met | Sentinel runs live on video files on **CPU** only. This does not satisfy A3. | Needs an NVIDIA machine. |
| A4 | Actionable output (JSON/MQTT/RTSP/alerts/analytics) | 🟡 Partial: local only | Sentinel writes schema-validated JSON incident events (`events.jsonl`, `incidents.json`) and serves `/api/events` and `/api/incidents`. See TEST_RESULTS.md. | These outputs come from Sentinel's CPU pipeline, not from a PipeGen-generated pipeline. |

### Judging risk: "deep integration"

The problem statement says the chosen tool must be **load-bearing**, and that bolting it on is grounds for disqualification.
Sentinel currently runs **without** PipeGen.
To qualify, the PipeGen-generated DeepStream pipeline has to be the perception core: it would produce the detections and JSON/MQTT metadata, and Sentinel would add reliability monitoring on the same stream.
GPU_VALIDATION.md describes this plan. Whether PipeGen can host Sentinel's image-quality stage as a pipeline node is **unknown** until the lab is read.

### Submission evidence still required (from the problem statement)

- [ ] Screenshot of the Pipeline IR node graph
- [ ] PipeGen TensorRT speed report
- [ ] Repository including the PipeGen export folder
- [ ] Project description: inputs, tasks, models, precision, outputs
- [ ] Live demo on hardware at the venue

> The problem statement gives a **4:00 pm** submission deadline. Please check how much time you have left before you plan GPU work.

## Sentinel MVP features (local)

| Feature | Status | Evidence |
|---|---|---|
| Video file ingestion; missing, empty, corrupt or zero-frame files; corrupt frames skipped; end of file | ✅ | `test_hardening.py`, `test_incidents_pipeline.py`; CLI exit code 2 |
| USB camera / RTSP ingestion with reconnect; credentials redacted | 🟡 Implemented; tested only with simulated captures | no real camera or RTSP server tested |
| Brightness, blur, obstruction, framing-change and latency metrics | ✅ with documented limits | tests; ARCHITECTURE.md |
| Baseline calibration with rejection and warnings for unsuitable footage | ✅ | `test_hardening.py` |
| Configurable, type-checked thresholds; smoothing; hysteresis | ✅ | `test_health.py`, `test_hardening.py` |
| HEALTHY / DEGRADED / UNTRUSTED / RECOVERING with recovery verification | ✅ | `test_health.py`, scenario runs |
| Incidents: one per episode, escalation, resolution, unverified, recurrence linking | ✅ | `test_hardening.py`, `test_incidents_pipeline.py` |
| JSON event schema, persistence across restarts, lookup by incident or stream, download | ✅ | `test_server.py` |
| Internal detection-event schema with a strict loader and frame association | ✅ tested with a hand-written fixture only | `test_detections_io.py` |
| PipeGen detections into Sentinel incidents | ❌ Converter not written (needs a genuine PipeGen output sample) | `parse_pipegen_metadata()` refuses by design |
| Optional ONNX CPU detector | 🟡 Untested (no weights) | — |
| MQTT output (optional, off by default) | 🟡 Tested with a fake client and a real paho client against an unreachable port; **not against a real broker** | `test_mqtt.py` |
| Dashboard: preview, pause/resume/stop, health and severity, metrics, detection panel, incidents, JSON view and download, loading/empty/error/disconnected states, synthetic-vs-real label, run history | ✅ | Manual browser check; see TEST_RESULTS.md |
| One-command demo generation and one-command launch | ✅ | `make-demo` then `serve`; clean start in TEST_RESULTS.md |
