"""Per-frame image-quality measurements.

All metrics are computed on a grayscale copy downscaled to at most
``ANALYSIS_WIDTH`` pixels wide so cost is independent of input resolution.

Metrics (see ARCHITECTURE.md for rationale and limitations):

* ``brightness``      mean gray level, 0-255.
* ``dark_fraction``   fraction of pixels below ``DARK_PIXEL_LEVEL``.
* ``sharpness``       variance of the Laplacian divided by the variance of the
                      image (gain-normalised Laplacian variance). Plain
                      Laplacian variance scales with the square of contrast, so
                      a darkened frame would also look "blurred"; dividing by
                      image variance removes that gain dependence.
* ``cell_std``        per-cell standard deviation on a coarse grid, used by the
                      obstruction proxy (compared against the baseline later).
* ``thumb``           heavily smoothed, zero-mean / unit-variance thumbnail used
                      by the framing-change proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ANALYSIS_WIDTH = 640
DARK_PIXEL_LEVEL = 40
GRID_COLS = 8
GRID_ROWS = 6
THUMB_SIZE = (64, 36)  # (w, h)


@dataclass
class FrameMetrics:
    width: int
    height: int
    brightness: float
    dark_fraction: float
    sharpness: float
    cell_std: np.ndarray  # shape (GRID_ROWS, GRID_COLS)
    thumb: np.ndarray  # shape (THUMB_SIZE[1], THUMB_SIZE[0]), float32, normalised

    def summary(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "brightness": round(self.brightness, 2),
            "dark_fraction": round(self.dark_fraction, 4),
            "sharpness": round(self.sharpness, 5),
        }


def to_analysis_gray(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    if frame.dtype != np.uint8:
        raise ValueError(f"unsupported frame dtype {frame.dtype}; expected uint8")
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"unsupported frame shape {frame.shape}")
    h, w = gray.shape[:2]
    if w > ANALYSIS_WIDTH:
        scale = ANALYSIS_WIDTH / w
        gray = cv2.resize(gray, (ANALYSIS_WIDTH, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return gray


def brightness_stats(gray: np.ndarray) -> tuple[float, float]:
    return float(gray.mean()), float((gray < DARK_PIXEL_LEVEL).mean())


def normalized_laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    var = float(g.var())
    if var < 1e-6:
        return 0.0
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(lap.var()) / var


def cell_std(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    out = np.zeros((GRID_ROWS, GRID_COLS), np.float32)
    g = gray.astype(np.float32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = g[r * h // GRID_ROWS:(r + 1) * h // GRID_ROWS, c * w // GRID_COLS:(c + 1) * w // GRID_COLS]
            out[r, c] = float(cell.std()) if cell.size else 0.0
    return out


def normalized_thumb(gray: np.ndarray) -> np.ndarray:
    # Strong blur first so the framing proxy is insensitive to focus blur and noise.
    small = cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    small = np.asarray(cv2.GaussianBlur(small, (5, 5), 1.5), np.float32)
    std = float(small.std())
    return (small - small.mean()) / std if std > 1e-6 else np.zeros_like(small)


def thumb_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two normalised thumbnails, in [-1, 1]."""
    if not a.any() or not b.any():
        return 0.0
    return float((a * b).mean())


def compute_frame_metrics(frame: np.ndarray) -> FrameMetrics:
    h, w = frame.shape[:2]
    gray = to_analysis_gray(frame)
    brightness, dark = brightness_stats(gray)
    return FrameMetrics(
        width=w,
        height=h,
        brightness=brightness,
        dark_fraction=dark,
        sharpness=normalized_laplacian_variance(gray),
        cell_std=cell_std(gray),
        thumb=normalized_thumb(gray),
    )
