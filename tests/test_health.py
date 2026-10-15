from __future__ import annotations

import pytest

from sentinel.config import SentinelConfig
from sentinel.health import HealthEngine, HealthState, Thresholds, resolve_thresholds

DT = 0.1


def engine(**overrides) -> HealthEngine:
    cfg = SentinelConfig(smoothing_tau_s=0.0, degrade_after_s=1.0, untrusted_after_s=2.0,
                         escalate_after_s=100.0, recover_after_s=3.0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    th = {"brightness": Thresholds("below", enter=0.5, exit=0.7, severe=0.3)}
    return HealthEngine(cfg, th)


def feed(e: HealthEngine, value: float, seconds: float, t0: float) -> tuple[float, list]:
    transitions = []
    t = t0
    for _ in range(round(seconds / DT)):
        transitions += e.update(round(t, 6), {"brightness": value})
        t += DT
    return t, transitions


def test_value_exactly_at_enter_threshold_is_not_active():
    e = engine()
    e.update(0.0, {"brightness": 0.5})
    assert not e.checks["brightness"].active


def test_value_just_past_enter_threshold_is_active():
    e = engine()
    e.update(0.0, {"brightness": 0.4999})
    assert e.checks["brightness"].active


def test_hysteresis_keeps_check_active_between_enter_and_exit():
    e = engine()
    e.update(0.0, {"brightness": 0.4})
    e.update(0.1, {"brightness": 0.6})  # above enter, below exit
    assert e.checks["brightness"].active
    e.update(0.2, {"brightness": 0.71})
    assert not e.checks["brightness"].active


def test_no_flicker_when_value_oscillates_around_enter_threshold():
    e = engine()
    t, trs = feed(e, 1.0, 1.0, 0.0)
    for i in range(60):
        trs += e.update(t, {"brightness": 0.49 if i % 2 else 0.51})
        t += DT
    # active once, never clears (0.51 < exit 0.7), so exactly one HEALTHY->DEGRADED transition
    assert [(x.old, x.new) for x in trs] == [(HealthState.HEALTHY, HealthState.DEGRADED)]


def test_brief_dip_shorter_than_dwell_does_not_degrade():
    e = engine()
    t, trs = feed(e, 1.0, 1.0, 0.0)
    t, trs2 = feed(e, 0.4, 0.5, t)
    t, trs3 = feed(e, 1.0, 1.0, t)
    assert trs + trs2 + trs3 == []
    assert e.state is HealthState.HEALTHY


def test_degraded_after_dwell_and_untrusted_after_severe_dwell():
    e = engine()
    t, _ = feed(e, 1.0, 1.0, 0.0)
    t, trs = feed(e, 0.2, 3.5, t)
    assert [x.new for x in trs] == [HealthState.DEGRADED, HealthState.UNTRUSTED]
    assert trs[0].t - 1.0 == pytest.approx(1.0, abs=DT + 1e-6)
    assert trs[1].t - 1.0 == pytest.approx(2.0, abs=DT + 1e-6)


def test_condition_present_from_first_frame_still_degrades():
    # regression: an active_since of 0.0 must not be treated as "unset"
    e = engine()
    _, trs = feed(e, 0.4, 1.5, 0.0)
    assert [x.new for x in trs] == [HealthState.DEGRADED]


def test_escalates_to_untrusted_when_degraded_too_long():
    e = engine(escalate_after_s=5.0)
    _, trs = feed(e, 0.45, 8.0, 0.0)
    assert [x.new for x in trs] == [HealthState.DEGRADED, HealthState.UNTRUSTED]


def test_recovery_requires_full_recovery_window():
    e = engine()
    t, _ = feed(e, 0.4, 2.0, 0.0)
    assert e.state is HealthState.DEGRADED
    t, trs = feed(e, 1.0, 2.9, t)
    assert e.state is HealthState.RECOVERING
    t, trs2 = feed(e, 1.0, 0.5, t)
    assert e.state is HealthState.HEALTHY
    assert [x.new for x in trs + trs2] == [HealthState.RECOVERING, HealthState.HEALTHY]


def test_relapse_during_recovery_returns_to_degraded():
    e = engine()
    t, _ = feed(e, 0.4, 2.0, 0.0)
    t, _ = feed(e, 1.0, 1.0, t)
    assert e.state is HealthState.RECOVERING
    _, trs = feed(e, 0.4, 0.2, t)
    assert e.state is HealthState.DEGRADED
    assert trs[0].reason.startswith("relapse")


def test_smoothing_delays_reaction_to_single_outlier():
    e = engine(smoothing_tau_s=0.5)
    t, _ = feed(e, 1.0, 1.0, 0.0)
    e.update(t, {"brightness": 0.0})  # one-frame glitch
    assert not e.checks["brightness"].active


def test_missing_values_are_ignored():
    e = engine()
    e.update(0.0, {"brightness": None})
    e.update(0.1, {"brightness": float("nan")})
    assert e.checks["brightness"].smoothed is None


def test_resolve_thresholds_is_baseline_relative(baseline):
    th = resolve_thresholds(SentinelConfig(), baseline)
    assert th["framing"].enter == pytest.approx(baseline.similarity_p5 - 0.25)
    assert th["framing"].severe < th["framing"].enter < th["framing"].exit
    assert th["obstruction"].enter >= baseline.obstruction_p95 + 0.05


def test_disabled_check_is_not_evaluated(baseline):
    cfg = SentinelConfig.from_dict({"checks": {"latency": {"enabled": False}}})
    assert "latency" not in resolve_thresholds(cfg, baseline)


def test_config_validation():
    with pytest.raises(ValueError, match="unknown config"):
        SentinelConfig.from_dict({"nope": 1})
    with pytest.raises(ValueError, match="ordered"):
        SentinelConfig.from_dict({"checks": {"brightness": {"enter": 0.9, "exit": 0.5}}})
    with pytest.raises(ValueError, match="unknown check"):
        SentinelConfig.from_dict({"checks": {"bogus": {}}})
