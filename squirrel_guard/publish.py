"""Publish events to the recorder host over MQTT. The camstack bridge there stores them and sends notifications."""
from __future__ import annotations

import base64
import json
import logging
import os

import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)


class Publisher:
    def __init__(self, camera: str):
        self.camera = camera
        self.topic = f"squirrelguard/{camera}/event"
        host = os.environ.get("SG_MQTT_HOST")
        self.client = None
        if not host:
            log.info("SG_MQTT_HOST not set: events stay local")
            return
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"squirrel-guard-{camera}")
        c.username_pw_set(os.environ.get("SG_MQTT_USER"), os.environ.get("SG_MQTT_PASS"))
        c.reconnect_delay_set(1, 60)
        c.on_connect = lambda cl, ud, fl, rc, pr: log.info("MQTT connected to %s (%s)", host, rc)
        c.on_disconnect = lambda cl, ud, fl, rc, pr: log.warning("MQTT disconnected (%s)", rc)
        c.connect_async(host, int(os.environ.get("SG_MQTT_PORT", "1883")), keepalive=60)
        c.loop_start()  # background thread; QoS 1 messages queue while disconnected
        self.client = c

    def event(self, record: dict, jpeg: bytes | None) -> None:
        if not self.client:
            return
        payload = dict(record, camera=self.camera)
        if jpeg:
            payload["snapshot_jpeg_b64"] = base64.b64encode(jpeg).decode()
        info = self.client.publish(self.topic, json.dumps(payload), qos=1)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.warning("MQTT publish queued/failed rc=%s", info.rc)

    def clip(self, meta: dict, mp4: bytes) -> None:
        """Rendered event clip (boxes burned in) for the recorder host bridge to post as a second notification."""
        if not self.client:
            return
        payload = dict(meta, camera=self.camera, clip_mp4_b64=base64.b64encode(mp4).decode())
        # QoS 0: the recorder host's inline Suricata cannot parse >1 MB MQTT PUBLISHes, then drops the PUBACK as
        # "PUBLISH not seen before PUBACK" (2229001), which stalled the session into a redelivery loop.
        self.client.publish(f"squirrelguard/{self.camera}/clip", json.dumps(payload), qos=0)

    def deter(self, req: dict) -> None:
        """Ask the recorder host bridge to play a deterrent sound on the camera (gpuserver cannot reach the camera's network)."""
        if self.client:
            self.client.publish(f"squirrelguard/{self.camera}/deter", json.dumps(req), qos=1)

    def close(self) -> None:
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
