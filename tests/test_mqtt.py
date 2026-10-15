"""Optional MQTT sink: disabled by default, failure-tolerant, no credentials leaked."""

from __future__ import annotations

import json
import socket
import time

import pytest

from sentinel.config import MqttConfig, SentinelConfig
from sentinel.mqtt_sink import MqttSink
from sentinel.pipeline import Monitor
from sentinel.scenarios import dark


class FakeInfo:
    def __init__(self, rc):
        self.rc = rc


class FakeClient:
    def __init__(self, client_id, fail_publish=False, rc=0):
        self.client_id = client_id
        self.published = []
        self.fail_publish = fail_publish
        self.rc = rc
        self.auth = None
        self.on_connect = self.on_disconnect = None

    def username_pw_set(self, u, p):
        self.auth = (u, p)

    def tls_set(self):
        pass

    def max_queued_messages_set(self, n):
        pass

    def connect_async(self, host, port, keepalive):
        self.target = (host, port)

    def loop_start(self):
        self.on_connect(self, None, {}, 0, None)

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def publish(self, topic, payload, qos=0):
        if self.fail_publish:
            raise OSError("broken pipe")
        self.published.append((topic, json.loads(payload), qos))
        return FakeInfo(self.rc)


def test_disabled_by_default_does_nothing():
    sink = MqttSink(SentinelConfig().mqtt, "cam-01")
    sink.start()
    sink.publish({"x": 1})
    assert sink.status()["state"] == "disabled" and sink.published == 0


def test_publishes_incident_events_with_formatted_topic(baseline, test_frames):
    holder = {}

    def factory(cid):
        holder["c"] = FakeClient(cid)
        return holder["c"]
    sink = MqttSink(MqttConfig(enabled=True, qos=1), "cam-07", client_factory=factory)
    sink.start()
    assert sink.status()["connected"] is True
    cfg = SentinelConfig(stream_id="cam-07")
    mon = Monitor(cfg, baseline, sinks=[sink])
    for i, f in enumerate([dark(f) for f in test_frames[:45]]):
        mon.process(i, i / 15, f)
    msgs = holder["c"].published
    assert msgs and all(t == "sentinel/cam-07/events" and q == 1 for t, _, q in msgs)
    assert msgs[0][1]["event_type"] == "incident_opened"
    assert holder["c"].client_id == "sentinel-cam-07"


def test_credentials_only_from_env_and_never_in_status(monkeypatch):
    monkeypatch.setenv("SENT_USER", "operator")
    monkeypatch.setenv("SENT_PASS", "correct-horse-battery")
    holder = {}

    def factory(cid):
        holder["c"] = FakeClient(cid)
        return holder["c"]
    sink = MqttSink(MqttConfig(enabled=True, username_env="SENT_USER", password_env="SENT_PASS"), "cam-01", factory)
    sink.start()
    assert holder["c"].auth == ("operator", "correct-horse-battery")
    assert "correct-horse-battery" not in json.dumps(sink.status())
    assert "password" not in json.dumps(SentinelConfig().to_dict()["mqtt"]).replace("password_env", "")


def test_missing_env_credential_reported_without_value(monkeypatch):
    monkeypatch.delenv("NOPE_USER", raising=False)
    sink = MqttSink(MqttConfig(enabled=True, username_env="NOPE_USER"), "cam-01", FakeClient)
    sink.start()
    assert "NOPE_USER is not set" in sink.status()["last_error"]


def test_publish_failures_counted_and_never_stop_monitoring(baseline, test_frames):
    sink = MqttSink(MqttConfig(enabled=True), "cam-01", lambda cid: FakeClient(cid, fail_publish=True))
    sink.start()
    mon = Monitor(SentinelConfig(), baseline, sinks=[sink])
    for i, f in enumerate([dark(f) for f in test_frames[:45]]):
        mon.process(i, i / 15, f)
    assert mon.incidents.current is not None
    assert sink.failed > 0 and "OSError" in sink.last_error


def test_nonzero_publish_rc_counted():
    sink = MqttSink(MqttConfig(enabled=True), "cam-01", lambda cid: FakeClient(cid, rc=4))
    sink.start()
    sink.publish({"a": 1})
    assert sink.failed == 1 and sink.published == 0


def test_paho_missing_reports_unavailable():
    def factory(cid):
        raise ImportError("no paho")
    sink = MqttSink(MqttConfig(enabled=True), "cam-01", factory)
    sink.start()
    sink.publish({"a": 1})
    assert sink.status()["state"] == "unavailable"


def test_sink_exception_in_monitor_is_isolated(baseline, test_frames):
    class Broken:
        def publish(self, event):
            raise RuntimeError("sink bug")
    mon = Monitor(SentinelConfig(), baseline, sinks=[Broken()])
    for i, f in enumerate([dark(f) for f in test_frames[:45]]):
        mon.process(i, i / 15, f)
    assert mon.sink_errors > 0 and mon.incidents.current is not None


def test_real_paho_client_with_unreachable_broker_does_not_block():
    pytest.importorskip("paho.mqtt.client")
    with socket.socket() as s:  # find a closed local port
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    sink = MqttSink(MqttConfig(enabled=True, host="127.0.0.1", port=port), "cam-01")
    t0 = time.perf_counter()
    sink.start()
    for _ in range(3):
        sink.publish({"event_type": "test"})
    elapsed = time.perf_counter() - t0
    time.sleep(0.3)
    status = sink.status()
    sink.close()
    assert elapsed < 1.0
    assert status["connected"] is False
