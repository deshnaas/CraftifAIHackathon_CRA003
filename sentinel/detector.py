"""Optional object-detector integration.

Detection is optional: the health monitor works with ``NullDetector``, which
reports *no* detection statistics (never fake ones).

``OnnxYoloDetector`` runs a user-supplied YOLOv8-style ONNX export through
OpenCV DNN on CPU. No weights ship with this repo, and this backend has not
been validated here (see TEST_RESULTS.md).

External detections (on the NVIDIA target: the PipeGen-generated DeepStream
app) enter through Sentinel's own versioned detection-event format, loaded and
validated by ``detections_io.load_detection_events`` (``--detections``).
``ExternalDetections`` is the minimal in-memory form keyed by frame index.
The only missing piece is a converter from PipeGen's JSON metadata to that
format, which must be written against a real output file:
``parse_pipegen_metadata`` deliberately refuses until then (GPU_VALIDATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h in source pixels


class Detector(Protocol):
    name: str
    source: str  # "real_inference" (ran here) or "external_pipeline" (ran elsewhere)

    def detect(self, frame: np.ndarray, index: int = 0, t: float | None = None) -> list[Detection] | None: ...


class NullDetector:
    name = "none"
    source = "none"

    def detect(self, frame: np.ndarray, index: int = 0, t: float | None = None) -> list[Detection] | None:
        return None


class ExternalDetections:
    """Detections produced by another pipeline, looked up by frame index.

    A frame with no entry returns ``None`` (unknown), not an empty list, so a
    gap in the external metadata is never reported as "nothing detected".
    """

    source = "external_pipeline"

    def __init__(self, by_frame: dict[int, list[Detection]], name: str) -> None:
        self.by_frame = by_frame
        self.name = name

    def detect(self, frame: np.ndarray, index: int = 0, t: float | None = None) -> list[Detection] | None:
        return self.by_frame.get(index)


def parse_pipegen_metadata(path: str | Path) -> ExternalDetections:
    """Parse the JSON metadata written by a PipeGen-generated pipeline.

    Not implemented on purpose: PipeGen's metadata format has not been seen yet,
    and guessing it would produce detections that look real but are not.
    Implement this against a real output file (GPU_VALIDATION.md, step 3).
    """
    raise NotImplementedError(
        "PipeGen JSON metadata format not verified yet; implement parse_pipegen_metadata() "
        f"against a real PipeGen output file before using {path}"
    )


class OnnxYoloDetector:
    def __init__(self, model_path: str, labels: list[str] | None = None, input_size: int = 640,
                 conf_threshold: float = 0.4, nms_threshold: float = 0.5) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"detector model not found: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.name = f"onnx:{Path(model_path).name}"
        self.source = "real_inference"
        self.labels = labels or []
        self.size = input_size
        self.conf = conf_threshold
        self.nms = nms_threshold

    def detect(self, frame: np.ndarray, index: int = 0, t: float | None = None) -> list[Detection] | None:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = np.squeeze(self.net.forward())  # (4 + classes, N) for YOLOv8 exports
        if out.ndim != 2:
            return []
        if out.shape[0] < out.shape[1]:
            out = out.T
        scores = out[:, 4:]
        cls = scores.argmax(axis=1)
        conf = scores.max(axis=1)
        keep = conf >= self.conf
        boxes, confs, ids = [], [], []
        sx, sy = w / self.size, h / self.size
        for (cx, cy, bw, bh), c, k in zip(out[keep, :4], conf[keep], cls[keep], strict=True):
            boxes.append([int((cx - bw / 2) * sx), int((cy - bh / 2) * sy), int(bw * sx), int(bh * sy)])
            confs.append(float(c))
            ids.append(int(k))
        idx = cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.nms) if boxes else []
        result = []
        for i in np.array(idx).flatten():
            label = self.labels[ids[i]] if ids[i] < len(self.labels) else str(ids[i])
            x, y, bw, bh = boxes[i]
            result.append(Detection(label, confs[i], (x, y, bw, bh)))
        return result


def detection_stats(dets: list[Detection] | None, detector: Detector) -> dict[str, Any] | None:
    if dets is None:
        return None
    by_label: dict[str, int] = {}
    for d in dets:
        by_label[d.label] = by_label.get(d.label, 0) + 1
    return {
        "detector": detector.name,
        "source": detector.source,
        "count": len(dets),
        "mean_confidence": round(float(np.mean([d.confidence for d in dets])), 3) if dets else None,
        "by_label": by_label,
    }


def build_detector(model_path: str | None, labels_path: str | None = None) -> Detector:
    if not model_path:
        return NullDetector()
    labels = None
    if labels_path:
        labels = [ln.strip() for ln in Path(labels_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return OnnxYoloDetector(model_path, labels)
