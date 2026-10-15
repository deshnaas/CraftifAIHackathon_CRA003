"""Deterministic health state machine.

HEALTHY --(a check active >= degrade_after_s)--> DEGRADED
DEGRADED --(severe >= untrusted_after_s, or degraded >= escalate_after_s)--> UNTRUSTED
DEGRADED / UNTRUSTED --(no check active)--> RECOVERING
RECOVERING --(any check active)--> DEGRADED / UNTRUSTED
RECOVERING --(no check active for recover_after_s)--> HEALTHY

Each check value is EMA-smoothed, and a check uses separate enter / exit
thresholds (hysteresis), so a value hovering at one threshold cannot flicker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .baseline import Baseline
from .config import CheckSpec, SentinelConfig


class HealthState(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNTRUSTED = "UNTRUSTED"
    RECOVERING = "RECOVERING"


CATEGORY = {
    "brightness": "low_light",
    "sharpness": "blur",
    "obstruction": "obstruction",
    "framing": "framing_change",
    "latency": "processing_latency",
}

# When the primary condition is active, these checks are expected side effects
# and are not reported as separate suspected causes (they still count for state).
# The framing proxy works on heavily smoothed thumbnails, so blur does not move
# it; a re-pointed / zoomed view however changes texture statistics.
SUPPRESSES = {
    "brightness": {"sharpness", "obstruction", "framing"},
    "framing": {"sharpness", "obstruction"},
    "sharpness": {"obstruction"},
}


@dataclass(frozen=True)
class Thresholds:
    direction: str
    enter: float
    exit: float
    severe: float

    def beyond(self, value: float, level: float) -> bool:
        return value < level if self.direction == "below" else value > level


def resolve_thresholds(cfg: SentinelConfig, base: Baseline) -> dict[str, Thresholds]:
    out: dict[str, Thresholds] = {}
    for name, spec in cfg.checks.items():
        if not spec.enabled:
            continue
        out[name] = _resolve(name, spec, base)
    return out


def _resolve(name: str, spec: CheckSpec, base: Baseline) -> Thresholds:
    if name == "framing":
        p5 = base.similarity_p5
        return Thresholds("below", p5 - spec.enter, p5 - spec.exit, p5 - spec.severe)
    if name == "obstruction":
        # never alarm on obstruction levels already seen in healthy footage
        floor = base.obstruction_p95
        enter = max(spec.enter, floor + 0.05)
        exit_ = max(spec.exit, floor + 0.02)
        return Thresholds("above", enter, min(exit_, enter), max(spec.severe, enter))
    return Thresholds(spec.direction, spec.enter, spec.exit, spec.severe)


@dataclass
class CheckState:
    name: str
    thresholds: Thresholds
    raw: float | None = None
    smoothed: float | None = None
    active: bool = False
    severe: bool = False
    active_since: float | None = None
    severe_since: float | None = None

    def as_dict(self) -> dict[str, object]:
        t = self.thresholds
        return {
            "check": self.name,
            "category": CATEGORY.get(self.name, self.name),
            "value": None if self.smoothed is None else round(self.smoothed, 4),
            "raw_value": None if self.raw is None else round(self.raw, 4),
            "direction": t.direction,
            "enter_threshold": round(t.enter, 4),
            "exit_threshold": round(t.exit, 4),
            "severe_threshold": round(t.severe, 4),
            "active": self.active,
            "severe": self.severe,
        }


@dataclass
class Transition:
    t: float
    old: HealthState
    new: HealthState
    reason: str


@dataclass
class HealthEngine:
    cfg: SentinelConfig
    thresholds: dict[str, Thresholds]
    state: HealthState = HealthState.HEALTHY
    state_since: float = 0.0
    last_t: float | None = None
    checks: dict[str, CheckState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.checks = {n: CheckState(n, th) for n, th in self.thresholds.items()}

    # -- per-check update -------------------------------------------------
    def _update_check(self, c: CheckState, value: float | None, t: float, dt: float) -> None:
        if value is None or math.isnan(value):
            return
        c.raw = value
        if c.smoothed is None or self.cfg.smoothing_tau_s <= 0:
            c.smoothed = value
        else:
            alpha = 1.0 - math.exp(-dt / self.cfg.smoothing_tau_s) if dt > 0 else 1.0
            c.smoothed += alpha * (value - c.smoothed)
        th = c.thresholds
        if c.active:
            # clear only once the value is back past the exit threshold
            if not th.beyond(c.smoothed, th.exit):
                c.active, c.active_since = False, None
        elif th.beyond(c.smoothed, th.enter):
            c.active, c.active_since = True, t
        now_severe = c.active and th.beyond(c.smoothed, th.severe)
        if now_severe and not c.severe:
            c.severe_since = t
        elif not now_severe:
            c.severe_since = None
        c.severe = now_severe

    # -- state machine ----------------------------------------------------
    def update(self, t: float, values: dict[str, float | None]) -> list[Transition]:
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        if self.last_t is None:
            self.state_since = t
        self.last_t = t
        for name, c in self.checks.items():
            self._update_check(c, values.get(name), t, dt)

        cfg = self.cfg
        active = [c for c in self.checks.values() if c.active]
        qualified = [c for c in active if held_for(c.active_since, t) >= cfg.degrade_after_s]
        sustained_severe = [
            c for c in active if c.severe and held_for(c.severe_since, t) >= cfg.untrusted_after_s
        ]
        transitions: list[Transition] = []

        def go(new: HealthState, reason: str) -> None:
            transitions.append(Transition(t, self.state, new, reason))
            self.state, self.state_since = new, t

        s = self.state
        if s is HealthState.HEALTHY:
            if sustained_severe:
                go(HealthState.UNTRUSTED, _names("severe", sustained_severe))
            elif qualified:
                go(HealthState.DEGRADED, _names("active", qualified))
        elif s is HealthState.DEGRADED:
            if sustained_severe:
                go(HealthState.UNTRUSTED, _names("severe", sustained_severe))
            elif not active:
                go(HealthState.RECOVERING, "all checks back within exit thresholds")
            elif t - self.state_since >= cfg.escalate_after_s:
                go(HealthState.UNTRUSTED, f"degraded for {cfg.escalate_after_s:g}s without recovery")
        elif s is HealthState.UNTRUSTED:
            if not active:
                go(HealthState.RECOVERING, "all checks back within exit thresholds")
        elif s is HealthState.RECOVERING:
            if sustained_severe:
                go(HealthState.UNTRUSTED, "relapse: " + _names("severe", sustained_severe))
            elif active:
                go(HealthState.DEGRADED, "relapse: " + _names("active", active))
            elif t - self.state_since >= cfg.recover_after_s:
                go(HealthState.HEALTHY, f"all checks clear for {cfg.recover_after_s:g}s")
        return transitions

    def suspected_categories(self) -> list[str]:
        active = {c.name for c in self.checks.values() if c.active}
        suppressed: set[str] = set()
        for name in active:
            suppressed |= SUPPRESSES.get(name, set())
        return [CATEGORY[n] for n in self.checks if n in active and n not in suppressed]

    def snapshot(self) -> list[dict[str, object]]:
        return [c.as_dict() for c in self.checks.values()]

    def progress(self) -> dict[str, Any]:
        """Read-only view of the timers driving the next transition, for operators.

        ``pending``: checks active while still HEALTHY (confirming before DEGRADED).
        ``recovery``: while RECOVERING, how long all checks have been clear vs required.
        ``escalation``: while DEGRADED, time towards automatic escalation to UNTRUSTED.
        """
        t = self.last_t if self.last_t is not None else 0.0
        cfg = self.cfg
        out: dict[str, Any] = {
            "state": self.state.value,
            "state_since_s": round(self.state_since, 3),
            "time_in_state_s": round(t - self.state_since, 3) if self.last_t is not None else 0.0,
            "pending": [],
            "recovery": None,
            "escalation": None,
        }
        if self.state is HealthState.HEALTHY:
            out["pending"] = [
                {"check": c.name, "category": CATEGORY.get(c.name, c.name),
                 "held_s": round(held_for(c.active_since, t), 3), "needed_s": cfg.degrade_after_s}
                for c in self.checks.values() if c.active
            ]
        elif self.state is HealthState.RECOVERING:
            out["recovery"] = {"clear_for_s": round(t - self.state_since, 3), "needed_s": cfg.recover_after_s}
        elif self.state is HealthState.DEGRADED:
            out["escalation"] = {"degraded_for_s": round(t - self.state_since, 3), "escalate_after_s": cfg.escalate_after_s}
        return out


def held_for(since: float | None, t: float) -> float:
    # Explicit None check: a condition that starts at stream time 0.0 is valid.
    return 0.0 if since is None else t - since


def _names(kind: str, checks: list[CheckState]) -> str:
    return f"{kind}: " + ", ".join(c.name for c in checks)
