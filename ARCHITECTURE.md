# Architecture

```
            Observe              Measure                 Diagnose                 Respond / Verify
┌──────────┐   frames   ┌──────────────────┐ values ┌───────────────────┐ transitions ┌──────────────────┐
│ source.py├───────────▶│ metrics.py       ├───────▶│ health.py         ├────────────▶│ incidents.py     │
│ file/USB/│            │ + baseline.py    │        │ EMA + hysteresis  │             │ lifecycle, JSON  │
│ RTSP     │            │ (relative ratios)│        │ + dwell timers    │             │ events, evidence │
└──────────┘            └──────────────────┘        └───────────────────┘             └────────┬─────────┘
      │       ┌──────────────────────────────────┐                                             │
      └──────▶│ detector.py / detections_io.py   │── detection stats (real or external only) ──┤
              └──────────────────────────────────┘                                             ▼
                          pipeline.py: Monitor + run() ──▶ run.json, events.jsonl, incidents.json, summary.json,
                                                           evidence/*.jpg ──▶ server.py API / dashboard, mqtt_sink.py
```

## Components

| Module | Responsibility |
|---|---|
| `source.py` | Opens files, camera indices and URLs. **Removes credentials from URLs** before they appear in any message or output. Rejects missing, empty and undecodable sources. Skips isolated corrupt or invalid frames (up to 5 in a row, counted as `corrupt_frames`), including at the start of a file. Treats a longer failure run as end of file, or as a disconnect on live sources. Reconnects live sources, then raises `StreamLost`. File timestamps come from frame index ÷ fps (deterministic, and skipped frames still advance time). Live timestamps come from a monotonic clock. |
| `metrics.py` | Per-frame measurements on a ≤640 px grayscale copy. Accepts BGR, BGRA, grayscale and single-channel `uint8` frames, and rejects other types. |
| `baseline.py` | Builds the baseline from healthy footage and validates it (see below). Checks the integrity of stored baseline files. |
| `config.py` | Every threshold and timing value, with type checks. Thresholds must be ordered for hysteresis. Includes the optional MQTT section, which stores environment-variable *names* only. |
| `health.py` | Smoothed check values, active/severe flags, and HEALTHY / DEGRADED / UNTRUSTED / RECOVERING. Chooses the suspected category. |
| `incidents.py` | Incident lifecycle, recurrence linking, detection context, and versioned JSON events (deep-copied snapshots). |
| `detector.py` | Detector interface: `detect(frame, index, t)`. `NullDetector` is the default and reports nothing. `OnnxYoloDetector` is untested. `parse_pipegen_metadata` deliberately refuses (see the boundary section). |
| `detections_io.py` | Strict loader for Sentinel's internal detection-event JSONL. Associates records with frames by index or by timestamp. |
| `pipeline.py` | `Monitor` handles each frame. `run()` is the **single** run loop used by both the CLI and the server: stop and pause, real-time pacing, run manifest, summary, error handling. |
| `mqtt_sink.py` | Optional non-blocking MQTT publisher. |
| `scenarios.py` | Deterministic synthetic scene, degradation transforms, demo timeline and manifest (labelled `synthetic: true`). |
| `server.py` + `static/index.html` | FastAPI API, persisted run history, MJPEG preview and the dashboard (plain HTML/JS with no build step). |

## Measurements and checks

| Check | Value fed to the engine | Default thresholds (enter / exit / severe) | Limitations |
|---|---|---|---|
| brightness | mean gray ÷ baseline median | < 0.55 / ≥ 0.70 / < 0.30 | Global only. A bright spotlight in a dark scene can mask local darkness. |
| sharpness | var(Laplacian) ÷ var(image), ÷ baseline | < 0.40 / ≥ 0.55 / < 0.20 | Insensitive to gain. Strong noise *raises* it and can mask blur. Changes in scene content move it. |
| obstruction | fraction of baseline-textured 8×6 cells whose std falls below 0.2× baseline, normalised by the median cell ratio | > max(0.15, p95+0.05) / < 0.08 / > 0.40 | Only catches occluders *flatter* than the scene. |
| framing | NCC of a smoothed 64×36 thumbnail against the baseline reference | < p5−0.25 / ≥ p5−0.15 / < p5−0.50 | Large scene changes also lower it. A proxy, not proof that the camera moved. |
| latency | processing time per frame in ms (metrics + detector lookup) | > 250 / < 150 / > 1000 | Sentinel's own processing, not end-to-end camera latency. |

Diagnosis rules: low light suppresses sharpness, obstruction and framing; framing change suppresses sharpness and obstruction; blur suppresses obstruction. Suppressed checks still count towards the health state.

## Baseline calibration and validation

| Condition | Result |
|---|---|
| Fewer than `baseline_min_frames` frames, no texture, or resolution changes mid-clip | **Rejected** (`BaselineError`) |
| Median brightness below `baseline_min_brightness` (25) | **Rejected**: too dark to be a healthy reference |
| Brightness coefficient of variation above 0.25 | Warning: lighting unstable during calibration |
| View-similarity p5 below 0.6 | Warning: scene too dynamic for the framing proxy |
| Obstruction p95 above 0.1 | Warning: the threshold is raised automatically |
| Stored file with wrong array shapes, non-positive values or non-finite numbers | **Rejected** on load |

Warnings are saved in `baseline.json`, printed by the CLI and shown on the dashboard. At runtime, a frame size different from the baseline's emits one `stream_warning` event.

## State machine

```
HEALTHY ──(any check active ≥ degrade_after_s = 1 s)──▶ DEGRADED ──(severe ≥ 2 s, or degraded ≥ 15 s)──▶ UNTRUSTED
   ▲                                                       │  ▲                                              │
   │                                         all clear     ▼  │ relapse                          all clear     │
   └────────(all clear ≥ recover_after_s = 3 s)──────── RECOVERING ◀───────────────────────────────────────────┘
```

Three mechanisms prevent flicker: EMA smoothing (τ = 0.5 s), separate enter and exit thresholds, and dwell timers. Time differences use explicit `None` checks, so a condition starting at stream time 0.0 is handled correctly.

## Incident lifecycle

- **Open** when the state leaves HEALTHY. A frame snapshot is saved to `evidence/<id>.jpg`.
- **One incident per episode.** A relapse during RECOVERING continues the same incident, so it is never duplicated.
- **Escalation** DEGRADED → UNTRUSTED is recorded in `transitions` and `peak_severity`.
- **Resolved** only on RECOVERING → HEALTHY, after the full recovery window.
- **Unverified** when the stream ends or fails first. A lost stream or an internal error also emits `stream_error`.
- **Recurrence.** A new incident within `recurrence_window_s` (30 s) of a resolved one with an overlapping category is *linked* (`recurrence_of`, `recurrence_count`), not merged.
- **Detection context.** When a detector or an external feed supplies data, `detection_context.before` covers `detection_context_s` (5 s) ending at the degradation onset (the earliest active check), and `during` runs from the onset onwards.
  - Frames without detection data are counted separately and never treated as zero detections.
  - With smoothing on, the measured onset lags the true change by roughly `smoothing_tau_s`.
- **Snapshots.** Events are deep copies, so an emitted event never changes afterwards.
- **No automatic actions.** Sentinel never retrains, recompiles or reconfigures anything. It notifies, sets an explicit UNTRUSTED state and recommends operator actions.

## Persistence and correlation

Each run folder contains:
- `run.json`: run ID, stream ID, redacted source, `input_kind`, detector, baseline warnings
- `events.jsonl`
- `incidents.json`
- `summary.json`: written even after errors
- `evidence/`

The server lists these folders after a restart. Incidents can be correlated by `incident_id` (global lookup across runs) and filtered by `stream_id`, `status`, `category` and `run_id`.

`input_kind` keeps demo footage distinct from real input:
- `synthetic_demo`: listed in a synthetic demo manifest
- `derived_demo`
- `video_file`
- `camera`
- `network_stream`

## Boundary with a future PipeGen-generated DeepStream app

```
PipeGen DeepStream app (NVIDIA)                      Sentinel (any machine with Python)
  run.sh ─▶ TensorRT detector ─▶ JSON metadata ──▶ [converter: NOT WRITTEN] ──▶ detection_event JSONL (schema v1.0)
     ▲                                                                                  │ --detections
     └──────────── same raw video file / camera ──────────▶ Sentinel quality checks ◀──┘
```

- Sentinel reads the **raw** source, never PipeGen's annotated output.
- The only missing piece is the converter, from PipeGen's real JSON metadata to [`schema/detection_event.schema.json`](schema/detection_event.schema.json). Write it as `parse_pipegen_metadata` or as a standalone script, and only against a genuine output sample.
- Frame association uses `frame_index`, with `timestamp_s` as a fallback. The frame-numbering convention must be checked on real data.
