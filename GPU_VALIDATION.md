# PipeGen / NVIDIA feasibility and validation checklist

Nothing in the validation checklist has been done. Tick an item only when the evidence named next to it exists in `evidence/`.

Last updated: 2026-10-15.

## 1. What is verified, and where it comes from

| Fact | Source |
|---|---|
| Install PipeGen **v0.5.1**; `pipegen --version` must print `0.5.1` | Hackathon problem statement, setup checklist |
| Needs an **x86-64 Linux PC with an NVIDIA GPU, or a Jetson on JetPack 6**, in an **X11 desktop session**; keep videos under `/home` | Hackathon problem statement, setup checklist |
| Sign in with a CraftifAI account; downloads at `developer.craftifai.com/dashboard?tab=downloads` | Problem statement. The page itself requires sign-in. |
| Five stages: Requirement, Pipeline IR, Model, Compiler, Run. ONNX is compiled to TensorRT in FP32/FP16/INT8. A DeepStream app with a `run.sh` is generated. Outputs: display, saved video, RTSP, JSON metadata, MQTT | Problem statement, technology primer |
| GPU support: "x86_64 systems with an NVIDIA GPU from Turing onward, including Windows through WSL2"; also Jetson AGX Orin and Orin Nano | craftifai.com homepage FAQ |
| Runs on your own machine and is not a hosted service. It is used at build time. Output is a self-contained directory (`run.sh`, config, compiled libraries) that runs as a standard GStreamer/DeepStream app with no runtime dependency on CraftifAI | craftifai.com/products/pipegen FAQ |
| Inputs: USB/CSI camera, video file, RTSP, Intel RealSense. Outputs: on-screen display, recorded video, RTSP publish, **MQTT metadata, JSON metadata**. It does not generate other protocols | craftifai.com/products/pipegen FAQ |
| Every pipeline ships with a benchmark report: latency, throughput, per-layer timing, engine size and precision, power and temperature where available, and accuracy against the source model | craftifai.com/products/pipegen |
| A public catalog of source pipelines exists at `github.com/craftifai/PipeGen_Benchmark_Repositories`. It includes `ONNX-YOLOv8-Object-Detection`, whose weights are fetched by `models/download.sh` and whose README says they carry the Ultralytics GPL licence | GitHub API listing, read on 2026-10-15 |

### Not verified (the blocking unknowns)

- **The PipeGen v0.5.1 Lab** (`docs.google.com/document/d/15HpN0QE…`) is not shared with the signed-in Google account: Drive returns "not found" and the web link redirects to sign-in. Ask the organisers to share it.
- **`developer.craftifai.com/docs/pipegen`**, linked from the product page, returned **404** on 2026-10-15.
- Unknown without the lab:
  - exact install commands
  - the Pipeline IR format
  - how each stage is driven (GUI or CLI)
  - the **JSON metadata schema**
  - where outputs are written
  - how `run.sh` is invoked
  - driver, CUDA and DeepStream versions
  - minimum disk space
- **Conflict to resolve:** the homepage allows Windows through WSL2, but the hackathon checklist requires Linux with an X11 desktop. **Follow the hackathon checklist.**

## 2. Feasibility on this laptop

**Not feasible.** The ASUS Vivobook S 14 (S5406SA) has Intel Arc graphics and no NVIDIA GPU (`nvidia-smi` is absent). Both official routes (native Linux and WSL2) need an NVIDIA GPU from Turing onward. No installation attempt is worthwhile here.

## 3. Remote NVIDIA environments

| Option | Usable? | Why |
|---|---|---|
| **Venue hardware desk or CraftifAI mentor machine** | ✅ Most realistic | The problem statement expects a live demo on hardware at the venue and points teams to mentors at the hardware desk. Ask for an x86 NVIDIA PC or a Jetson with PipeGen 0.5.1 already set up. |
| Teammate's laptop or PC with an NVIDIA RTX/GTX 16-series or newer, running native Ubuntu with an X11 session | ✅ | Matches the official checklist exactly. A USB camera from the kit can be plugged in directly. |
| Cloud GPU VM (Turing or newer, e.g. a T4- or A10-class instance) with Ubuntu desktop and an **X11** remote desktop | ⚠️ Possible but risky | It meets the stated hardware rule. But GPU quota approval can take hours, setup time is significant, PipeGen sign-in and install there are unverified, and GPU-accelerated display over remote desktop can fail. A USB camera cannot be attached, so use a video file under `/home` or an RTSP stream (both are valid live-run inputs). |
| Notebook-only GPU environments (Colab, Kaggle) | ❌ | No X11 desktop session, no persistent system install, no DeepStream app run. These do not meet the official setup checklist. |
| WSL2 on this laptop | ❌ | It still needs an NVIDIA GPU, and the hackathon checklist asks for Linux with X11. |

## 4. Smallest authentic integration (the Sentinel / PipeGen boundary)

```
video file / USB cam ──▶ PipeGen-generated DeepStream app (TensorRT YOLOv8) ──▶ JSON metadata (+ MQTT)
        │                                                                              │
        │                       converter (NOT WRITTEN: needs a genuine PipeGen sample)│
        │                                                                              ▼
        │                                     detection-event JSONL (schema/detection_event.schema.json v1.0)
        ▼                                                                              │ --detections
Sentinel quality checks on the SAME raw source ──▶ health state + incidents (detection_context from TensorRT)
```

- PipeGen is the perception core. In the submitted demo, Sentinel never runs its own detector.
- Sentinel reads the **raw** source, not PipeGen's annotated RTSP output, because drawn boxes would distort its image metrics.
- **Implemented and tested locally:**
  - **Detection-event schema and strict loader** (`sentinel/detections_io.py`, 29 tests). It validates every record, rejects malformed or conflicting duplicates (or skips them with `--detections-lenient`), sorts out-of-order records, and filters by `stream_id`.
  - **Frame association** by `frame_index`, falling back to the nearest `timestamp_s` within half a frame. A missing frame means "no data", never zero detections.
  - **Incident detection context:** detection statistics before the degradation onset and during the incident.
  - **`--pipegen-metadata`** stays reserved, and `parse_pipegen_metadata()` raises until the converter exists.
- **The only remaining software piece** is the converter from PipeGen's JSON metadata to detection-event records. Write it against a real sample and validate each output record with `detections_io.validate_record`. Add a test that uses the real sample (kept in `evidence/`, not in `tests/fixtures/`).
- **Unknowns to check on real data:**
  - whether PipeGen's frame numbering starts at 0 and counts every decoded frame
  - how it encodes boxes (absolute pixels vs normalised)
  - whether it writes a record for frames with no detections; if not, the converter must emit `"detections": []` only where the detector demonstrably ran

## 5. Local work vs NVIDIA-only work

| Done and tested on this laptop | Needs NVIDIA hardware plus PipeGen |
|---|---|
| ✅ Sentinel monitor, dashboard, persistence, API (135 tests passing) | Install PipeGen 0.5.1 and confirm `pipegen --version` |
| ✅ Detection-event schema, loader, association, CLI `--detections` | Requirement → IR → Model → Compiler → Run |
| ✅ Optional MQTT publisher (not tested against a real broker) | TensorRT engine and PipeGen benchmark report |
| ✅ Demo scenarios, incident JSON schema | Live run on video/USB/RTSP |
| Re-baseline on real footage, once any is available | Capture real JSON metadata, write the converter and its test |
| Point Sentinel's MQTT at PipeGen's broker (optional) | Joint run: PipeGen and Sentinel on the same source |

## 6. Validation checklist (on the NVIDIA machine)

The environment checks are standard Linux and NVIDIA tools, not PipeGen commands. The PipeGen steps must follow the lab. Save outputs to `evidence/`.

**Environment** (`evidence/env.txt`)
- [ ] `nvidia-smi`: GPU model (Turing or newer) and driver version
- [ ] `uname -m` prints `x86_64`, and `cat /etc/os-release` shows a Linux distribution (or record the JetPack 6 version on a Jetson)
- [ ] `echo $XDG_SESSION_TYPE` prints `x11`
- [ ] `df -h /home`: record free space (the required amount is in the lab)
- [ ] Install PipeGen v0.5.1 following the lab, then `pipegen --version` prints `0.5.1`. Sign in with the CraftifAI account.

**PipeGen flow** (record each action in `evidence/pipegen_log.md`)
- [ ] **Requirement.** Draft, not validated: *"Detect people and vehicles in a video file (and a USB camera), and publish per-frame detections as JSON metadata and MQTT."*
- [ ] **Pipeline IR.** Screenshot the node graph (required for submission).
- [ ] **Model.** Record the model PipeGen selected (e.g. the catalog's ONNX-YOLOv8), its source and licence.
- [ ] **Compiler.** Build the TensorRT engine (FP16; INT8 earns the bonus). Save the benchmark report exactly as produced (required for submission).
- [ ] **Run.** Run the generated `run.sh` on a video under `/home`, then on the USB camera. Save the console log and a screen recording.
- [ ] Copy a sample of the JSON metadata (and MQTT messages, if enabled) to `evidence/pipegen_output/`.

**Sentinel integration** (Sentinel itself runs on Linux too: `python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`)
- [ ] Write the converter against the saved sample and add its test. Confirm that frame numbering matches Sentinel's frame index on the same file.
- [ ] Build a baseline from healthy footage of the real camera: `.venv/bin/python -m sentinel baseline /home/<user>/healthy.mp4` (resolve any baseline warnings).
- [ ] Convert, then run Sentinel on the same file: `.venv/bin/python -m sentinel run /home/<user>/test.mp4 --detections <converted.jsonl> --out runs/pipegen`
- [ ] Check that incidents contain `detection.source = "external_pipeline"` and a plausible `detection_context`.
- [ ] On the real camera, cover the lens, dim the lights and knock the mount. Record which incidents fired and which did not. Do not claim a capability the recording does not show.

**Submission**
- [ ] Commit the IR screenshot, benchmark report and PipeGen export folder to the repo.
- [ ] Update REQUIREMENTS_STATUS.md with links to the evidence.
- [ ] Submit through the PipeGen form in the problem statement.
