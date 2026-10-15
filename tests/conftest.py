from __future__ import annotations

import itertools

import numpy as np
import pytest

from sentinel.baseline import Baseline, build_baseline
from sentinel.config import SentinelConfig
from sentinel.metrics import compute_frame_metrics
from sentinel.scenarios import synthetic_frames


@pytest.fixture(scope="session")
def healthy_frames() -> list[np.ndarray]:
    return list(synthetic_frames(4.0))


@pytest.fixture(scope="session")
def test_frames() -> list[np.ndarray]:
    # a different time window than the baseline footage
    return list(synthetic_frames(14.0, t_offset=30.0))


@pytest.fixture(scope="session")
def baseline(healthy_frames: list[np.ndarray]) -> Baseline:
    cfg = SentinelConfig()
    return build_baseline([compute_frame_metrics(f) for f in healthy_frames], cfg, "synthetic", 15.0)


@pytest.fixture
def ids():
    counter = itertools.count(1)
    return lambda prefix: f"{prefix}-{next(counter):04d}"
