"""Configuration. Every threshold is configurable; defaults are starting points
tuned on the bundled synthetic demo, not universal truths for every camera."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

Direction = Literal["below", "above"]


@dataclass
class CheckSpec:
    """A check fires (``enter``) when the smoothed value crosses ``enter`` in
    the bad direction and clears only once it returns past ``exit`` (hysteresis).
    Crossing ``severe`` contributes to the UNTRUSTED state."""

    enabled: bool
    direction: Direction
    enter: float
    exit: float
    severe: float


def _default_checks() -> dict[str, CheckSpec]:
    return {
        # value = brightness / baseline brightness
        "brightness": CheckSpec(True, "below", enter=0.55, exit=0.70, severe=0.30),
        # value = sharpness / baseline sharpness (gain-normalised Laplacian variance)
        "sharpness": CheckSpec(True, "below", enter=0.40, exit=0.55, severe=0.20),
        # value = fraction of baseline-textured grid cells that have lost their texture
        "obstruction": CheckSpec(True, "above", enter=0.15, exit=0.08, severe=0.40),
        # value = normalised cross-correlation of thumbnail with baseline reference
        # (thresholds here are offsets below the baseline 5th percentile; resolved at runtime)
        "framing": CheckSpec(True, "below", enter=0.25, exit=0.15, severe=0.50),
        # value = per-frame processing latency in ms
        "latency": CheckSpec(True, "above", enter=250.0, exit=150.0, severe=1000.0),
    }


@dataclass
class MqttConfig:
    """Optional MQTT publication of incident events. Credentials are never stored
    in config: only the *names* of environment variables that hold them."""

    enabled: bool = False
    host: str = "localhost"
    port: int = 1883
    topic: str = "sentinel/{stream_id}/events"
    client_id: str = "sentinel-{stream_id}"
    qos: int = 1
    tls: bool = False
    username_env: str | None = None
    password_env: str | None = None
    keepalive_s: int = 30
    max_queued: int = 1000


_NUMERIC = {
    "smoothing_tau_s", "degrade_after_s", "untrusted_after_s", "escalate_after_s", "recover_after_s",
    "obstruction_cell_ratio", "obstruction_min_cell_std", "reconnect_delay_s",
    "baseline_min_brightness", "baseline_max_brightness_cv", "baseline_min_similarity",
    "recurrence_window_s", "detection_context_s",
}
_INTEGER = {"baseline_min_frames", "reconnect_attempts"}
_BOOL = {"evidence_snapshots"}


def _is_number(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


@dataclass
class SentinelConfig:
    stream_id: str = "cam-01"
    checks: dict[str, CheckSpec] = field(default_factory=_default_checks)
    smoothing_tau_s: float = 0.5  # EMA time constant applied to every check value
    degrade_after_s: float = 1.0  # a check must stay active this long before DEGRADED
    untrusted_after_s: float = 2.0  # a severe condition this long -> UNTRUSTED
    escalate_after_s: float = 15.0  # DEGRADED this long without recovery -> UNTRUSTED
    recover_after_s: float = 3.0  # all checks clear this long in RECOVERING -> HEALTHY
    baseline_min_frames: int = 30
    baseline_min_brightness: float = 25.0  # darker "healthy" footage is rejected as a reference
    baseline_max_brightness_cv: float = 0.25  # above this, warn: lighting unstable during calibration
    baseline_min_similarity: float = 0.6  # below this, warn: scene too dynamic for the framing proxy
    obstruction_cell_ratio: float = 0.2  # cell texture below this fraction of baseline = "lost"
    obstruction_min_cell_std: float = 4.0  # ignore cells that were flat in the baseline
    reconnect_attempts: int = 3
    reconnect_delay_s: float = 1.0
    evidence_snapshots: bool = True
    recurrence_window_s: float = 30.0  # a new incident this soon after a resolved one is linked to it
    detection_context_s: float = 5.0  # detection stats window kept from before an incident opens
    mqtt: MqttConfig = field(default_factory=MqttConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SentinelConfig:
        if not isinstance(data, dict):
            raise ValueError("config must be a JSON object")
        cfg = cls()
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for key, value in data.items():
            if key == "checks":
                if not isinstance(value, dict):
                    raise ValueError("checks must be an object")
                for name, spec in value.items():
                    if name not in cfg.checks:
                        raise ValueError(f"unknown check: {name}")
                    if not isinstance(spec, dict):
                        raise ValueError(f"checks.{name} must be an object")
                    bad = set(spec) - {f.name for f in fields(CheckSpec)}
                    if bad:
                        raise ValueError(f"checks.{name}: unknown keys {sorted(bad)}")
                    cfg.checks[name] = CheckSpec(**{**asdict(cfg.checks[name]), **spec})
            elif key == "mqtt":
                if not isinstance(value, dict):
                    raise ValueError("mqtt must be an object")
                bad = set(value) - {f.name for f in fields(MqttConfig)}
                if bad:
                    raise ValueError(f"mqtt: unknown keys {sorted(bad)}")
                cfg.mqtt = MqttConfig(**{**asdict(cfg.mqtt), **value})
            else:
                setattr(cfg, key, value)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> SentinelConfig:
        if path is None:
            return cls()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"config {path} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def validate(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        for key in _NUMERIC:
            v = getattr(self, key)
            if not _is_number(v):
                raise ValueError(f"{key} must be a finite number, got {v!r}")
            if v < 0:
                raise ValueError(f"{key} must be >= 0")
        for key in _INTEGER:
            v = getattr(self, key)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{key} must be a non-negative integer, got {v!r}")
        if self.baseline_min_frames < 2:
            raise ValueError("baseline_min_frames must be >= 2")
        for key in _BOOL:
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be true or false")
        for name, c in self.checks.items():
            if not isinstance(c.enabled, bool):
                raise ValueError(f"{name}: enabled must be true or false")
            for attr in ("enter", "exit", "severe"):
                if not _is_number(getattr(c, attr)):
                    raise ValueError(f"{name}.{attr} must be a finite number")
            if name == "framing":
                # offsets below the baseline similarity: larger = worse, always "below"
                if c.direction != "below":
                    raise ValueError("framing: direction must be 'below'")
                ok = c.exit <= c.enter <= c.severe
            elif c.direction == "below":
                ok = c.severe <= c.enter <= c.exit
            elif c.direction == "above":
                ok = c.exit <= c.enter <= c.severe
            else:
                raise ValueError(f"{name}: direction must be 'below' or 'above'")
            if not ok:
                raise ValueError(f"{name}: thresholds must be ordered (exit/enter/severe) for hysteresis")
        m = self.mqtt
        if not isinstance(m.enabled, bool) or not isinstance(m.tls, bool):
            raise ValueError("mqtt.enabled and mqtt.tls must be true or false")
        if not isinstance(m.port, int) or not 0 < m.port < 65536:
            raise ValueError("mqtt.port must be 1-65535")
        if m.qos not in (0, 1, 2):
            raise ValueError("mqtt.qos must be 0, 1 or 2")
        if not isinstance(m.topic, str) or not m.topic or any(ch in m.topic for ch in "#+"):
            raise ValueError("mqtt.topic must be a non-empty topic without wildcards")
        for attr in ("username_env", "password_env"):
            v = getattr(m, attr)
            if v is not None and (not isinstance(v, str) or not v.isidentifier()):
                raise ValueError(f"mqtt.{attr} must be the NAME of an environment variable, not a credential")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
