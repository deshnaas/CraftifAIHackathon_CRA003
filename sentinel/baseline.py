"""Baseline established from footage the operator declares healthy.

Thresholds are expressed relative to this baseline so that one config can be
reused across cameras with different scenes, exposure and optics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import SentinelConfig
from .metrics import GRID_COLS, GRID_ROWS, THUMB_SIZE, FrameMetrics, thumb_similarity

BASELINE_VERSION = 1


class BaselineError(ValueError):
    pass


@dataclass
class Baseline:
    source: str
    frames: int
    width: int
    height: int
    fps: float
    brightness: float  # median
    dark_fraction: float  # median
    sharpness: float  # median
    cell_std: np.ndarray  # median per cell
    reference_thumb: np.ndarray  # mean normalised thumbnail
    similarity_p5: float  # 5th percentile similarity of healthy frames to the reference
    obstruction_p95: float  # 95th percentile obstruction fraction on healthy frames
    created_at: str
    warnings: list[str] = field(default_factory=list)  # calibration caveats an operator should see

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": BASELINE_VERSION,
            "source": self.source,
            "frames": self.frames,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "brightness": self.brightness,
            "dark_fraction": self.dark_fraction,
            "sharpness": self.sharpness,
            "cell_std": self.cell_std.round(3).tolist(),
            "reference_thumb": self.reference_thumb.round(4).tolist(),
            "similarity_p5": self.similarity_p5,
            "obstruction_p95": self.obstruction_p95,
            "created_at": self.created_at,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Baseline:
        if not isinstance(d, dict) or d.get("version") != BASELINE_VERSION:
            raise BaselineError(f"unsupported baseline version: {d.get('version') if isinstance(d, dict) else d!r}")
        base = cls(
            source=d["source"],
            frames=d["frames"],
            width=d["width"],
            height=d["height"],
            fps=d["fps"],
            brightness=d["brightness"],
            dark_fraction=d["dark_fraction"],
            sharpness=d["sharpness"],
            cell_std=np.asarray(d["cell_std"], np.float32),
            reference_thumb=np.asarray(d["reference_thumb"], np.float32),
            similarity_p5=d["similarity_p5"],
            obstruction_p95=d["obstruction_p95"],
            created_at=d["created_at"],
            warnings=list(d.get("warnings", [])),
        )
        base.check_integrity()
        return base

    def check_integrity(self) -> None:
        """Reject damaged or hand-edited baseline files before they produce nonsense ratios."""
        if self.cell_std.shape != (GRID_ROWS, GRID_COLS):
            raise BaselineError(f"baseline cell_std has shape {self.cell_std.shape}, expected {(GRID_ROWS, GRID_COLS)}")
        if self.reference_thumb.shape != (THUMB_SIZE[1], THUMB_SIZE[0]):
            raise BaselineError(f"baseline reference_thumb has shape {self.reference_thumb.shape}")
        for name in ("brightness", "sharpness", "fps"):
            v = getattr(self, name)
            if not isinstance(v, int | float) or not math.isfinite(v) or v <= 0:
                raise BaselineError(f"baseline {name} must be a positive number, got {v!r}")
        if not (np.isfinite(self.cell_std).all() and np.isfinite(self.reference_thumb).all()):
            raise BaselineError("baseline contains non-finite values")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, BaselineError):
                raise
            raise BaselineError(f"cannot load baseline {path}: {exc}") from exc


def obstruction_fraction(cell: np.ndarray, base_cell: np.ndarray, cfg: SentinelConfig) -> float:
    """Fraction of baseline-textured cells whose texture has collapsed.

    Each cell's std is divided by its baseline std, then by the median of those
    ratios, so a uniform gain change (darkening) does not register as obstruction.
    Only detects occluders that are *less textured* than the scene they cover.
    """
    eligible = base_cell >= cfg.obstruction_min_cell_std
    if not eligible.any():
        return 0.0
    ratios = cell[eligible] / base_cell[eligible]
    med = float(np.median(ratios))
    if med < 1e-6:
        return 1.0
    return float((ratios / med < cfg.obstruction_cell_ratio).mean())


def build_baseline(
    metrics: list[FrameMetrics], cfg: SentinelConfig, source: str, fps: float
) -> Baseline:
    if len(metrics) < cfg.baseline_min_frames:
        raise BaselineError(
            f"baseline needs at least {cfg.baseline_min_frames} frames, got {len(metrics)}"
        )
    sizes = {(m.width, m.height) for m in metrics}
    if len(sizes) > 1:
        raise BaselineError(f"baseline footage changes resolution mid-clip: {sorted(sizes)}")
    sharp = np.array([m.sharpness for m in metrics])
    if np.median(sharp) <= 0:
        raise BaselineError("baseline footage has no measurable texture (blank frames?)")
    bright = np.array([m.brightness for m in metrics])
    if float(np.median(bright)) < cfg.baseline_min_brightness:
        raise BaselineError(
            f"baseline footage is too dark to be a healthy reference (median brightness "
            f"{np.median(bright):.1f} < {cfg.baseline_min_brightness:g}); record healthy footage in normal light"
        )
    ref = np.mean([m.thumb for m in metrics], axis=0).astype(np.float32)
    ref = (ref - ref.mean()) / max(float(ref.std()), 1e-6)
    cells = np.median([m.cell_std for m in metrics], axis=0).astype(np.float32)
    sims = [thumb_similarity(m.thumb, ref) for m in metrics]
    obst = [obstruction_fraction(m.cell_std, cells, cfg) for m in metrics]
    warnings: list[str] = []
    cv = float(bright.std() / max(bright.mean(), 1e-6))
    if cv > cfg.baseline_max_brightness_cv:
        warnings.append(f"lighting varied during calibration (brightness CV {cv:.2f}); "
                        "brightness thresholds may fire on normal variation")
    sim_p5 = float(np.percentile(sims, 5))
    if sim_p5 < cfg.baseline_min_similarity:
        warnings.append(f"scene is highly dynamic (view similarity p5 {sim_p5:.2f}); "
                        "framing-change check is unreliable for this camera")
    obst_p95 = float(np.percentile(obst, 95))
    if obst_p95 > 0.1:
        warnings.append(f"healthy footage already shows texture loss in {obst_p95:.0%} of cells; "
                        "obstruction threshold raised accordingly")
    return Baseline(
        source=source,
        frames=len(metrics),
        width=metrics[0].width,
        height=metrics[0].height,
        fps=fps,
        brightness=float(np.median([m.brightness for m in metrics])),
        dark_fraction=float(np.median([m.dark_fraction for m in metrics])),
        sharpness=float(np.median(sharp)),
        cell_std=cells,
        reference_thumb=ref,
        similarity_p5=sim_p5,
        obstruction_p95=obst_p95,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        warnings=warnings,
    )
