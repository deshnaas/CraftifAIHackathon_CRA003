"""Optional MQTT publication of incident events.

Disabled by default. When enabled it never blocks or crashes monitoring:
connection happens in the background (``connect_async`` + ``loop_start``),
publish failures are counted, and if ``paho-mqtt`` is not installed the sink
reports itself unavailable. Credentials come only from environment variables
named in the config and never appear in status, logs or payloads.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from typing import Any

from .config import MqttConfig


class MqttSink:
    def __init__(self, cfg: MqttConfig, stream_id: str,
                 client_factory: Callable[[str], Any] | None = None) -> None:
        self.cfg = cfg
        self.stream_id = stream_id
        self.topic = cfg.topic.format(stream_id=stream_id)
        self.client_id = cfg.client_id.format(stream_id=stream_id)
        self._factory = client_factory
        self._client: Any = None
        self._lock = threading.Lock()
        self.state = "disabled" if not cfg.enabled else "not_started"
        self.connected = False
        self.published = 0
        self.failed = 0
        self.last_error: str | None = None

    def _default_factory(self, client_id: str) -> Any:
        import paho.mqtt.client as mqtt  # optional dependency

        return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)

    def start(self) -> None:
        if not self.cfg.enabled:
            return
        try:
            client = (self._factory or self._default_factory)(self.client_id)
        except ImportError:
            self.state = "unavailable"
            self.last_error = "paho-mqtt is not installed (pip install paho-mqtt)"
            return
        user = os.environ.get(self.cfg.username_env) if self.cfg.username_env else None
        password = os.environ.get(self.cfg.password_env) if self.cfg.password_env else None
        if self.cfg.username_env and user is None:
            self.last_error = f"environment variable {self.cfg.username_env} is not set"
        if user is not None:
            client.username_pw_set(user, password)
        if self.cfg.tls:
            client.tls_set()
        client.max_queued_messages_set(self.cfg.max_queued)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        self._client = client
        try:
            client.connect_async(self.cfg.host, self.cfg.port, self.cfg.keepalive_s)
            client.loop_start()
            self.state = "connecting"
        except Exception as exc:  # network/config errors must not stop monitoring
            self.state = "error"
            self.last_error = f"connect failed: {type(exc).__name__}"

    @staticmethod
    def _ok(reason_code: Any) -> bool:
        if hasattr(reason_code, "is_failure"):
            return not reason_code.is_failure
        return int(reason_code) == 0

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        with self._lock:
            self.connected = self._ok(reason_code)
            self.state = "connected" if self.connected else "error"
            if not self.connected:
                self.last_error = f"broker refused connection: {reason_code}"

    def _on_disconnect(self, client: Any, userdata: Any, flags: Any = None, reason_code: Any = None,
                       properties: Any = None) -> None:
        with self._lock:
            self.connected = False
            if self.state != "closed":
                self.state = "disconnected"

    def publish(self, event: dict[str, Any]) -> None:
        if self._client is None:
            return
        try:
            info = self._client.publish(self.topic, json.dumps(event), qos=self.cfg.qos)
            rc = getattr(info, "rc", 0)
            if rc == 0:
                self.published += 1
            else:
                self.failed += 1
                self.last_error = f"publish returned rc={rc} (not connected or queue full)"
        except Exception as exc:
            self.failed += 1
            self.last_error = f"publish failed: {type(exc).__name__}"

    def close(self) -> None:
        if self._client is None:
            return
        self.state = "closed"
        try:
            self._client.disconnect()
            self._client.loop_stop()
        except Exception:
            pass
        self._client = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.cfg.enabled,
            "state": self.state,
            "connected": self.connected,
            "broker": f"{self.cfg.host}:{self.cfg.port}" if self.cfg.enabled else None,
            "topic": self.topic if self.cfg.enabled else None,
            "published": self.published,
            "failed": self.failed,
            "last_error": self.last_error,
        }
