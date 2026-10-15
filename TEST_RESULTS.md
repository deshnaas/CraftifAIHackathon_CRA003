# Test results

Latest results: 2026-10-15, from a **clean start**. That means a fresh copy of `sentinel/`, `tests/`, `schema/`, `pyproject.toml` and `requirements.txt`, a new virtualenv from `requirements.txt`, then the documented commands only.

## Environment

- ASUS Vivobook S 14 (S5406SA), Intel Arc graphics, **no NVIDIA GPU** (`nvidia-smi` not present)
- Windows 11 Home 10.0.26200, Python 3.14.3
- opencv-python-headless 5.0.0 (CPU, bundled FFmpeg), numpy 2.5.4, fastapi 0.143.2, paho-mqtt 2.1.0
- No CUDA, TensorRT, DeepStream or PipeGen installed

## Automated checks

| Check | Command | Result |
|---|---|---|
| Tests | `.venv/Scripts/python -m pytest -q` | **141 passed** (73 s) after the dashboard-quality pass. Before it: 135 passed (84 s on the clean start). |
| Lint | `.venv/Scripts/ruff check sentinel tests` | All checks passed |
| Types | `.venv/Scripts/mypy sentinel` | No issues found in 15 source files |
| Frontend | `node --check` on the dashboard's inline script | Syntax OK. There is no frontend lint, type or build tooling: the dashboard is plain HTML/JS with no build step. |

The baseline before this round was **51 passed**, with ruff and mypy clean (13 files). Every earlier test still passes unchanged. One server test caught a regression I introduced (a removed evidence route), and I restored the route rather than editing the test.

| Test file | Tests | Covers |
|---|---|---|
| `test_metrics_baseline.py` | 12 | metric values, blur, gain invariance, downscaling, baseline round trip, obstruction proxy |
| `test_health.py` | 15 | threshold boundaries, hysteresis, no flicker, dwell, escalation, recovery window, relapse, t = 0 regression, config |
| `test_incidents_pipeline.py` | 17 | one incident per scenario, resolution, unverified at EOF, persistence, missing or corrupt files, live reconnect and loss |
| `test_external_detections.py` | 4 | frame-indexed external detections, `--pipegen-metadata` refusal |
| `test_hardening.py` | 40 | corrupt frames (leading, mid-stream, invalid dtype or shape), empty and zero-frame videos, URL credential redaction, input labels, dark-pixel boundary, gray/BGRA frames, baseline rejection and warnings, damaged baseline files, 16 config type/range cases, immutable event snapshots, relapse kept as one incident, recurrence linking, escalation, resolution-change warning, internal error recorded then raised, CLI exit code 3, deterministic demo plus one-command baseline |
| `test_detections_io.py` | 29 | fixture conforms to the schema, association by index and timestamp, missing frame returns "no data", 17 malformed-record cases (including NaN, out-of-range, string and bool values), strict vs lenient mode, identical and conflicting duplicates, out-of-order records, timestamp mismatch, empty or missing files, detection context before and during an incident, CLI `--detections` |
| `test_mqtt.py` | 9 | disabled by default, topic formatting, credentials from environment only and never in status, missing env var, publish exceptions and non-zero return codes counted, paho missing, isolated sink failure, **real paho client** against an unreachable port (non-blocking) |
| `test_lifecycle.py` | 5 | full demo timeline in memory: each degradation opens one incident in its own segment, runs DEGRADED → … → RECOVERING → HEALTHY, and resolves exactly `recover_after_s` after recovery starts (within one frame); no incidents in healthy segments; events valid and ordered; the progress view reports confirmation, escalation and recovery timers |
| `test_server.py` | 10 | end-to-end run, path traversal, missing baseline, **persistence across a server restart**, filters and lookup by ID, API errors (404/409/422), pause/resume/stop, failed source recorded in history, baseline and input labels in status |

## Scenario results (synthetic footage, CPU, no detector)

Demo timeline (`demo_timeline.mp4`, 74 s, 1110 frames, 640×360 @ 15 fps, `input_kind: synthetic_demo`). These results are identical to the previous round:

| Applied transform (ground truth) | Incident opened (s) | Diagnosed category | Peak state | Resolved (s) |
|---|---|---|---|---|
| dark, 8–16 s | 9.40 | low_light | UNTRUSTED | 19.47 |
| blur, 24–32 s | 25.47 | blur | UNTRUSTED | 35.40 |
| occlusion, 40–48 s | 41.67 | obstruction | DEGRADED | 51.40 |
| framing_shift, 56–64 s | 57.73 | framing_change | DEGRADED | 67.33 |

There were no incidents during the healthy segments and no corrupt frames. Mean latency was 5.46 ms/frame (max 22.55 ms).

Single-scenario clips (clean start):

| Clip | Final state | Incidents | Mean / max latency (ms) |
|---|---|---|---|
| healthy | HEALTHY | 0 | 3.93 / 14.89 |
| dark | UNTRUSTED | 1 (unverified at EOF) | 3.91 / 9.99 |
| blur | UNTRUSTED | 1 (unverified at EOF) | 3.97 / 10.82 |
| occlusion | DEGRADED | 1 (unverified at EOF) | 4.11 / 10.15 |
| framing_shift | DEGRADED | 1 (unverified at EOF) | 5.08 / 14.66 |

CLI checks:
- `missing.mp4` → "video file not found", exit 2
- corrupt file → "cannot open video source", exit 2
- `--detections` with the test fixture → 4 frame records used, 1 out of order, exit 0
- `--pipegen-metadata` → refuses with "format not verified", exit 2

## Dashboard-quality pass (2026-10-15)

**Lifecycle on the encoded `demo_timeline.mp4`** (CLI run). Each incident resolved exactly 3.000 s (`recover_after_s`) after recovery began:

| Category | Transitions | Recovering → resolved (s) |
|---|---|---|
| low_light | DEGRADED → UNTRUSTED → RECOVERING → HEALTHY | 16.467 → 19.467 |
| blur | DEGRADED → UNTRUSTED → RECOVERING → HEALTHY | 32.4 → 35.4 |
| obstruction | DEGRADED → RECOVERING → HEALTHY | 48.4 → 51.4 |
| framing_change | DEGRADED → RECOVERING → HEALTHY | 64.333 → 67.333 |

**Checked in the browser:**
- **Live run.** The UI showed each stage: OK, then "confirming 0.6 / 1.0 s", Warning (DEGRADED), Critical (UNTRUSTED), then "Verifying 0.5 / 3.0 s", with the incident labelled *recovering*, then *resolved*.
- **Layout.** No horizontal page overflow at 375 px, 498 px (pane), 768 px or 1366 px. The checks table fits at every width (two secondary threshold columns are hidden below 640 px; their values are in each check's help text).
- **Accessibility.** Every control has an accessible name, and `:focus-visible` outlines show on selects, buttons, inputs and the help toggles. The JSON dialog moves focus to Close and returns it afterwards.
- **Errors.**
  - An unknown incident ID gives "No incident with ID … in any run."
  - A rejected pause gives a banner: "Could not pause: cannot pause: run is finished".
  - A failed video load gives "Video preview could not be loaded…".
  - Disconnected and reconnected states are shown with banners.
- **Not checked.** Real keyboard Tab navigation: synthetic key presses did not reach the embedded browser pane, so focus styling was verified programmatically instead.

**Bugs fixed in this pass:**
- During the 1 s confirmation window the big state said HEALTHY while "suspected: low_light" was shown. It now says "confirming", in both the dashboard and the video overlay.
- Recovering incidents were shown as plain "active".
- The chart's axis labels were drawn in a translucent band colour.
- The chart's last time label was clipped.
- The real-time checkbox's accessible name was "on".

## Manual dashboard check (browser, earlier round)

I checked these against a server started from this folder:
- idle and empty states
- live run with the SYNTHETIC DEMO FOOTAGE badge and the ground-truth segment
- UNTRUSTED / critical severity with the recommended action
- pause (stream time frozen at 18.267 s) and resume
- run history and incidents restored after a server restart
- evidence snapshots served for a persisted run
- incident JSON dialog
- disconnected banner when the server was stopped

The check found and fixed two dashboard bugs:
- a render error (writing to an element that had just been replaced) was shown as "disconnected"
- snapshots were missing for the latest persisted run after a restart

## Bugs found and fixed this round

- `Incident.to_dict()` made a shallow copy, so in-memory events served by `/api/events` changed after they were emitted. It is now a deep copy (`test_emitted_events_are_immutable_snapshots`).
- One corrupt frame in a file ended the run as if it were end of file. Leading corrupt frames made `open()` fail. Both are now skipped and counted.
- RTSP URLs with credentials appeared in errors and summaries. They are now redacted everywhere.
- An unexpected exception during processing left the open incident unclosed and wrote no summary.
- Unknown keys inside a check config raised `TypeError` (a crash) instead of a validation error. Wrong value types weren't caught at all.
- The out-of-order check in the detection loader compared frame indices with seconds.
- "Before" detection context ended at incident open, so it included already-degraded frames. It now ends at the degradation onset.

## What these results do NOT show

- **Real footage or physical faults.** Everything uses a procedurally generated scene with digital transforms.
- **Model accuracy.** No ground truth was used, no detector ran, and the external detections are a hand-written fixture.
- **Live hardware.** USB, RTSP and reconnect were tested only with simulated captures. MQTT was not tested against a real broker.
- **ONNX detector backend.** Never executed.
- **GPU, TensorRT or PipeGen.** Not available. See GPU_VALIDATION.md.

## Evidence pack verification (2026-10-15)

The demo evidence pack is in [`evidence/`](evidence/DEMO_REPORT.md). These checks were run after it was written:

| Check | Command | Result |
|---|---|---|
| Tests | `.venv/Scripts/python -m pytest -q -p no:cacheprovider` | **141 passed** (49 s) |
| Lint | `.venv/Scripts/ruff check sentinel tests scripts` | All checks passed |
| Types | `.venv/Scripts/mypy sentinel scripts/summarize_run.py` | No issues found in 16 source files |
| Dashboard JS | `node --check` on the inline script | OK |
| Raw evidence integrity | `sha256sum -c SHA256SUMS.txt` in `evidence/dashboard_run_20261015-215543/` | 11/11 OK |
| Reproduction script | `powershell -ExecutionPolicy Bypass -File scripts/make_evidence.ps1 -SkipInstall` | All steps exit 0 (log in `evidence/repro_20261015-220222/commands.log`) |
| Secret / personal-data scan | `grep` of `evidence/` and `scripts/` for credentials, user name, email, absolute paths; EXIF check of screenshots | No matches; no EXIF |
| Mermaid diagram | parsed and rendered with mermaid 10.9.1 in the browser | OK |
