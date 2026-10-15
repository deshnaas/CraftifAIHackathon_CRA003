"""Command-line entry point: ``python -m sentinel <command>``.

Exit codes: 0 ok, 2 bad input (missing/corrupt source, invalid config,
baseline or detections file), 3 unexpected internal error (events and
summary are still written for the run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .baseline import Baseline, BaselineError
from .config import SentinelConfig
from .detections_io import DetectionEventError, load_detection_events
from .detector import Detector, build_detector, parse_pipegen_metadata
from .mqtt_sink import MqttSink
from .pipeline import Monitor, establish_baseline, run
from .scenarios import build_demo
from .source import SourceError


def _print_baseline(base: Baseline) -> None:
    print(json.dumps({k: v for k, v in base.to_dict().items() if k not in ("cell_std", "reference_thumb")}, indent=1))
    for w in base.warnings:
        print(f"warning: {w}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sentinel", description="Edge vision pipeline reliability monitor")
    p.add_argument("--config", help="JSON config file (defaults are used when omitted)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("make-demo", help="generate synthetic demo clips AND the baseline (one command)")
    d.add_argument("--out", default="data/demo")
    d.add_argument("--source", help="your own healthy video (default: synthetic scene)")
    d.add_argument("--baseline-out", default="data/baseline.json")
    d.add_argument("--no-baseline", action="store_true", help="only write the clips")

    b = sub.add_parser("baseline", help="establish a baseline from healthy footage")
    b.add_argument("source")
    b.add_argument("--out", default="data/baseline.json")
    b.add_argument("--seconds", type=float, help="use only the first N seconds")

    r = sub.add_parser("run", help="monitor a video file, camera index or RTSP URL")
    r.add_argument("source")
    r.add_argument("--baseline", default="data/baseline.json")
    r.add_argument("--out", default="runs/latest")
    r.add_argument("--model", help="optional YOLOv8-style ONNX model for CPU detection (untested)")
    r.add_argument("--labels", help="optional labels file, one per line")
    r.add_argument("--detections", help="detections JSONL in Sentinel's detection-event schema "
                   "(schema/detection_event.schema.json), e.g. converted from a PipeGen run on the same source")
    r.add_argument("--detections-lenient", action="store_true",
                   help="skip invalid records instead of rejecting the whole detections file")
    r.add_argument("--pipegen-metadata", help="raw PipeGen JSON metadata (converter not implemented until a "
                   "real PipeGen output file has been inspected)")
    r.add_argument("--realtime", action="store_true", help="pace file playback at its native fps")

    s = sub.add_parser("serve", help="start the dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data", default="data/demo")
    s.add_argument("--baseline", default="data/baseline.json")
    s.add_argument("--runs", default="runs/dashboard", help="where dashboard runs are persisted")
    s.add_argument("--model")

    a = p.parse_args(argv)
    try:
        cfg = SentinelConfig.load(a.config)
        if a.cmd == "make-demo":
            m = build_demo(a.out, a.source)
            print(f"wrote {len(m['clips'])} clips + baseline footage to {a.out} (manifest.json); "
                  f"source: {m['source']}")
            if not a.no_baseline:
                base = establish_baseline(str(Path(a.out) / m["baseline"]["file"]), cfg)
                base.save(a.baseline_out)
                print(f"baseline written to {a.baseline_out} ({base.frames} frames)")
                for w in base.warnings:
                    print(f"warning: {w}", file=sys.stderr)
        elif a.cmd == "baseline":
            base = establish_baseline(a.source, cfg, a.seconds)
            base.save(a.out)
            _print_baseline(base)
        elif a.cmd == "run":
            if sum(bool(x) for x in (a.model, a.detections, a.pipegen_metadata)) > 1:
                raise ValueError("use only one of --model, --detections, --pipegen-metadata")
            base = Baseline.load(a.baseline)
            extra: dict[str, object] = {}
            detector: Detector
            if a.pipegen_metadata:
                detector = parse_pipegen_metadata(a.pipegen_metadata)
            elif a.detections:
                detector, report = load_detection_events(a.detections, cfg.stream_id, base.fps,
                                                         strict=not a.detections_lenient)
                extra["detections_input"] = report.to_dict()
                print(f"detections: {report.records_used} frame records used, {report.malformed} malformed, "
                      f"{report.duplicates_conflicting} conflicting duplicates, {report.out_of_order} out of order",
                      file=sys.stderr)
            else:
                detector = build_detector(a.model, a.labels)
            sink = MqttSink(cfg.mqtt, cfg.stream_id)
            sink.start()
            try:
                mon = Monitor(cfg, base, a.out, detector, sinks=[sink])
                summary = run(a.source, cfg, base, a.out, realtime=a.realtime, monitor=mon, extra_summary=extra)
            finally:
                if cfg.mqtt.enabled:
                    extra_status = sink.status()
                    print(f"mqtt: {json.dumps(extra_status)}", file=sys.stderr)
                sink.close()
            print(json.dumps(summary.to_dict(), indent=1))
        elif a.cmd == "serve":
            import uvicorn

            from .server import create_app
            app = create_app(Path(a.data), Path(a.baseline), cfg, a.model, runs_dir=Path(a.runs))
            uvicorn.run(app, host=a.host, port=a.port)
    except (SourceError, BaselineError, DetectionEventError, FileNotFoundError, ValueError,
            NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
