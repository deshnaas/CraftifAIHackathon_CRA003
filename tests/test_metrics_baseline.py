from __future__ import annotations

import cv2
import numpy as np
import pytest

from sentinel.baseline import Baseline, BaselineError, build_baseline, obstruction_fraction
from sentinel.config import SentinelConfig
from sentinel.metrics import ANALYSIS_WIDTH, compute_frame_metrics, to_analysis_gray
from sentinel.scenarios import blur, dark, occlude


def test_brightness_and_dark_fraction_of_uniform_frames():
    m = compute_frame_metrics(np.full((120, 160, 3), 200, np.uint8))
    assert m.brightness == pytest.approx(200.0)
    assert m.dark_fraction == 0.0
    m = compute_frame_metrics(np.full((120, 160, 3), 10, np.uint8))
    assert m.dark_fraction == 1.0


def test_dark_fraction_counts_pixels_below_level():
    img = np.full((100, 100, 3), 200, np.uint8)
    img[:25] = 5  # 25% of rows dark
    assert compute_frame_metrics(img).dark_fraction == pytest.approx(0.25)


def test_uniform_frame_has_zero_sharpness():
    assert compute_frame_metrics(np.full((50, 50, 3), 128, np.uint8)).sharpness == 0.0


def test_blur_lowers_sharpness(healthy_frames):
    f = healthy_frames[0]
    assert compute_frame_metrics(blur(f)).sharpness < 0.2 * compute_frame_metrics(f).sharpness


def test_sharpness_is_roughly_gain_invariant(healthy_frames):
    f = healthy_frames[0]
    s, s_dark = compute_frame_metrics(f).sharpness, compute_frame_metrics(dark(f)).sharpness
    assert s_dark == pytest.approx(s, rel=0.3)  # plain Laplacian variance would drop ~20x


def test_large_frames_are_downscaled_for_analysis():
    gray = to_analysis_gray(np.zeros((1080, 1920, 3), np.uint8))
    assert gray.shape[1] == ANALYSIS_WIDTH
    m = compute_frame_metrics(np.zeros((1080, 1920, 3), np.uint8))
    assert (m.width, m.height) == (1920, 1080)


def test_empty_frame_rejected():
    with pytest.raises(ValueError):
        compute_frame_metrics(np.zeros((0, 0, 3), np.uint8))


def test_baseline_requires_min_frames(healthy_frames):
    cfg = SentinelConfig(baseline_min_frames=30)
    ms = [compute_frame_metrics(f) for f in healthy_frames[:10]]
    with pytest.raises(BaselineError, match="at least 30"):
        build_baseline(ms, cfg, "x", 15.0)


def test_baseline_rejects_blank_footage():
    ms = [compute_frame_metrics(np.full((60, 80, 3), 90, np.uint8)) for _ in range(40)]
    with pytest.raises(BaselineError, match="texture"):
        build_baseline(ms, SentinelConfig(), "x", 15.0)


def test_baseline_values_and_roundtrip(baseline, tmp_path):
    assert 80 < baseline.brightness < 160
    assert baseline.similarity_p5 > 0.9
    assert baseline.obstruction_p95 == pytest.approx(0.0)
    path = tmp_path / "b.json"
    baseline.save(path)
    loaded = Baseline.load(path)
    assert loaded.brightness == pytest.approx(baseline.brightness)
    np.testing.assert_allclose(loaded.reference_thumb, baseline.reference_thumb, atol=1e-3)


def test_baseline_load_errors(tmp_path):
    with pytest.raises(BaselineError):
        Baseline.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"version": 99}')
    with pytest.raises(BaselineError):
        Baseline.load(bad)


def test_obstruction_proxy(baseline, test_frames):
    cfg = SentinelConfig()
    f = test_frames[0]

    def frac(img):
        return obstruction_fraction(compute_frame_metrics(img).cell_std, baseline.cell_std, cfg)

    assert frac(f) == 0.0
    assert frac(dark(f)) == 0.0  # uniform gain change is not obstruction
    assert frac(occlude(f)) >= 0.15
    covered = f.copy()
    cv2.rectangle(covered, (0, 0), (f.shape[1], f.shape[0]), (0, 0, 0), -1)
    assert frac(covered) == 1.0
