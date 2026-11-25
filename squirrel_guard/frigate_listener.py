"""Frigate mode: Frigate detects, squirrel-guard confirms and acts.

frigate/events (MQTT) -> new squirrel event -> Frigate snapshot (crop) -> local VLM verifier -> deterrent
-> publish to squirrelguard/<camera>/event (recorder host bridge: Postgres, ntfy, email).

Run:  python -m squirrel_guard.frigate_listener -c config.toml
Env:  SG_MQTT_HOST/PORT/USER/PASS (broker shared with Frigate), SG_FRIGATE_API (default http://frigate:5000)
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import logging
import os
import queue
import threading
import time
import tomllib
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import requests

from .actuator import Actuator
from .annotate import annotate
from .publish import Publisher
from .verifier import Verifier

log = logging.getLogger("squirrel_guard.frigate")


def fetch_jpeg(api: str, event_id: str, crop: bool) -> bytes | None:
    url = f"{api}/api/events/{event_id}/snapshot.jpg"
    params = {"crop": 1, "bbox": 0, "quality": 90} if crop else {"bbox": 0, "quality": 95}
    for attempt in range(5):  # the snapshot can lag the event message by a moment
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200 and r.content:
                return r.content
        except requests.RequestException as e:
            log.debug("snapshot fetch failed: %s", e)
        time.sleep(0.5 * (attempt + 1))
    return None


def event_from_api(api: str, event_id: str) -> dict | None:
    """GET /api/events/<id> and map it to the MQTT `after` shape the worker expects."""
    try:
        r = requests.get(f"{api}/api/events/{event_id}", timeout=10)
        if r.status_code != 200:
            return None
        e = r.json()
    except (requests.RequestException, ValueError):
        return None
    data = e.get("data") or {}
    return {"id": e.get("id"), "camera": e.get("camera"), "label": e.get("label"),
            "start_time": e.get("start_time"), "top_score": data.get("top_score") or data.get("score") or e.get("top_score"),
            "box": None,
            "rel_box": data.get("box"),  # relative [x, y, w, h]
            "false_positive": e.get("false_positive")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = tomllib.loads(Path(args.config).read_text())
    fcfg, ver_cfg, act_cfg = cfg.get("frigate", {}), cfg["verifier"], cfg["actuator"]
    api = os.environ.get("SG_FRIGATE_API", fcfg.get("api", "http://frigate:5000")).rstrip("/")
    labels = {l.lower() for l in fcfg.get("labels", ["squirrel"])}
    cameras = set(fcfg.get("cameras", []))  # empty = all
    min_score = float(fcfg.get("min_score", 0.0))
    verify_labels = {l.lower() for l in fcfg.get("verify_labels", ["squirrel"])}
    ncfg = cfg.get("notify", {})
    spacing = {k.lower(): float(v) for k, v in ncfg.get("min_interval_s", {"squirrel": 120, "person": 300}).items()}
    review_url = ncfg.get("review_url", "").rstrip("/")
    last_notify: dict[str, float] = {}

    ann_cfg = cfg.get("announce", {})
    announce_labels = {l.lower() for l in ann_cfg.get("labels", [])} if ann_cfg.get("enabled") else set()
    last_announce: dict[str, float] = {}

    def announce_due(label: str) -> bool:
        now = time.monotonic()
        if now - last_announce.get(label, -1e9) < float(ann_cfg.get("min_interval_s", 300)):
            return False
        last_announce[label] = now
        return True

    def in_quiet_hours() -> bool:
        q = act_cfg.get("quiet_hours")  # ["21:30", "07:00"]
        if not q:
            return False
        now = dt.datetime.now().strftime("%H:%M")
        start, end = q
        return (start <= now or now < end) if start > end else (start <= now < end)
    ccfg = cfg.get("clips", {})
    clip_q: queue.Queue = queue.Queue(maxsize=20)

    verifier = Verifier(ver_cfg["url"], model=ver_cfg.get("model", ""), timeout=ver_cfg.get("timeout", 20)) if ver_cfg.get("enabled", True) else None
    actuator = Actuator(act_cfg.get("command", ""), act_cfg.get("cooldown_s", 60), act_cfg.get("dry_run", True))
    # connect publishers up front so the first event is not queued behind the MQTT handshake
    publishers: dict[str, Publisher] = {cam: Publisher(cam) for cam in cameras}
    pub_lock = threading.Lock()

    def get_publisher(cam: str) -> Publisher:
        # NOT dict.setdefault(cam, Publisher(cam)): that builds a new client every call, and two clients with the
        # same MQTT client id make the broker drop each other in a reconnect loop.
        with pub_lock:
            if cam not in publishers:
                publishers[cam] = Publisher(cam)
            return publishers[cam]

    seen: collections.OrderedDict[str, float] = collections.OrderedDict()  # event ids already handled
    work: queue.Queue = queue.Queue(maxsize=50)

    def enqueue(after: dict) -> None:
        eid = after["id"]
        seen[eid] = time.time()
        while len(seen) > 500:
            seen.popitem(last=False)
        try:
            work.put_nowait(after)
        except queue.Full:
            log.warning("work queue full, dropping event %s", eid)

    def on_review(rv: dict) -> None:
        # frigate/reviews: covers manual/API-created events too, which never appear on frigate/events (0.18)
        after = rv.get("after") or {}
        data = after.get("data") or {}
        if (cameras and after.get("camera") not in cameras) or not labels & {o.split(":", 1)[0].strip().lower() for o in data.get("objects") or []}:  # objects look like "squirrel: <sub_label>"
            return
        for eid in data.get("detections") or []:
            if eid in seen:
                continue
            seen[eid] = time.time()  # claim before the API call so concurrent review updates don't double-process
            ev = event_from_api(api, eid)
            if ev and (ev.get("label") or "").lower() in labels and not ev.get("false_positive"):
                enqueue(ev)

    def on_message(client, userdata, msg):
        try:
            ev = json.loads(msg.payload)
        except ValueError:
            return
        if msg.topic == "frigate/reviews":
            if ev.get("type") in ("new", "update"):
                threading.Thread(target=on_review, args=(ev,), daemon=True).start()  # API call off the MQTT thread
            return
        after = ev.get("after") or {}
        eid, label, cam = after.get("id"), (after.get("label") or "").lower(), after.get("camera")
        if not eid or label not in labels or (cameras and cam not in cameras):
            return
        if ev.get("type") == "end" or after.get("false_positive") or eid in seen:
            return
        score = float(after.get("top_score") or after.get("score") or 0)
        if score < min_score:
            return
        enqueue(after)

    def worker():
        while True:
            after = work.get()
            eid, cam = after["id"], after["camera"]
            t0 = time.perf_counter()
            crop = fetch_jpeg(api, eid, crop=True)
            clean = fetch_jpeg(api, eid, crop=False)
            sc = float(after.get("top_score") or after.get("score") or 0)
            snap_time = (after.get("snapshot") or {}).get("frame_time") or after.get("start_time") or time.time()
            full = annotate(clean, after.get("label"), sc, snap_time,
                            box_abs=(after.get("snapshot") or {}).get("box") or after.get("box"),
                            box_rel=after.get("rel_box")) if clean else None
            label = (after.get("label") or "").lower()
            verdict = None
            if label in verify_labels:
                if verifier and crop:
                    img = cv2.imdecode(np.frombuffer(crop, np.uint8), cv2.IMREAD_COLOR)
                    verdict = verifier(img) if img is not None else None
                confirmed = verdict == label if verifier else True
                if verifier and verdict is None:
                    confirmed = ver_cfg.get("trust_detector_if_down", False)
                if not confirmed:
                    action = "rejected"
                elif in_quiet_hours():
                    action = "quiet"  # confirmed, but no sound overnight
                else:
                    action = actuator.fire()
                    if action == "fired" and act_cfg.get("mode") == "mqtt":
                        get_publisher(cam).deter({"camera": cam, "sound": act_cfg.get("sound", "random"),
                                                  "volume": int(act_cfg.get("volume", 80)), "frigate_event": eid})
            else:  # e.g. person: trusted as detected, notify (+ spoken announcement, below)
                confirmed, action = True, "notify"
                if label in announce_labels and announce_due(label):
                    get_publisher(cam).deter({"camera": cam, "sound": ann_cfg.get("sound", "person_announce"),
                                              "volume": int(ann_cfg.get("volume", 90)), "frigate_event": eid})
                    action = "notify+announce"
            now = time.monotonic()
            notify = confirmed and now - last_notify.get(label, -1e9) >= spacing.get(label, 120)
            if notify:
                last_notify[label] = now
            started = after.get("start_time") or time.time()
            record = {
                "ts": dt.datetime.fromtimestamp(started).strftime("%Y%m%d-%H%M%S-%f")[:-3],
                "time": dt.datetime.fromtimestamp(started).astimezone().isoformat(timespec="milliseconds"),
                "label": after.get("label"), "score": round(float(after.get("top_score") or after.get("score") or 0), 3),
                "box": [int(v) for v in (after.get("box") or [])][:4] or None,
                "verifier": verdict, "confirmed": confirmed, "action": action,
                "detect_ms": round((time.perf_counter() - t0) * 1000),  # here: snapshot + verify latency
                "snapshot": f"{eid}.jpg", "frigate_event": eid, "notify": notify,
                "click": f"{review_url}/explore?event_id={eid}" if review_url else None,
            }
            pub = get_publisher(cam)
            pub.event(record, full or crop)
            log.info("EVENT %s", record)
            if notify and ccfg.get("enabled", False):
                try:
                    clip_q.put_nowait((eid, cam, label, record["click"]))
                except queue.Full:
                    log.warning("clip queue full, skipping clip for %s", eid)

    def clip_worker():
        from .clips import render
        from .detector import Detector
        det = Detector(ccfg["model"], conf=float(ccfg.get("conf", 0.3)), threads=int(ccfg.get("threads", 6)))
        keep = {l.lower() for l in ccfg.get("labels", ["squirrel", "person"])}
        wait_s = float(ccfg.get("wait_for_end_s", 180))
        settle = float(ccfg.get("settle_s", 15))

        def make_clip(eid: str, cam: str, label: str, click: str | None) -> None:
            t_end = time.time() + wait_s
            while True:  # the clip exists only after the event ends and its segments reach storage
                if time.time() > t_end:
                    log.info("no clip for %s (event did not end with a recording within %.0fs)", eid, wait_s)
                    return
                try:
                    e = requests.get(f"{api}/api/events/{eid}", timeout=10).json()
                except (requests.RequestException, ValueError):
                    e = None
                if e is not None and (not isinstance(e, dict) or "id" not in e):
                    log.info("event %s no longer exists in Frigate (dropped as a fragment); no clip", eid)
                    return
                # end_time can be set ahead of time (manual events): wait until it is really past + settle
                if e and e.get("end_time") and e.get("has_clip") and time.time() >= e["end_time"] + settle:
                    break
                time.sleep(5)
            r = None
            for _ in range(6):  # 400 = recordings for the window not in storage yet
                try:
                    r = requests.get(f"{api}/api/events/{eid}/clip.mp4", timeout=60)
                    if r.status_code == 200 and r.content:
                        break
                except requests.RequestException:
                    pass
                r = None
                time.sleep(10)
            if r is None:
                log.warning("clip download failed for %s after retries", eid)
                return
            t0 = time.perf_counter()
            mp4 = render(r.content, det, keep, out_width=int(ccfg.get("width", 960)), out_fps=float(ccfg.get("fps", 12)),
                         detect_every=int(ccfg.get("detect_every", 2)), max_seconds=float(ccfg.get("max_seconds", 30)))
            if not mp4:
                return
            get_publisher(cam).clip({"camera": cam, "label": label, "frigate_event": eid, "click": click,
                                     "filename": f"{eid}.mp4"}, mp4)
            log.info("CLIP %s %s %.1f MB rendered in %.1fs", cam, eid, len(mp4) / 1e6, time.perf_counter() - t0)

        while True:
            item = clip_q.get()
            try:
                make_clip(*item)
            except Exception:  # never let one bad event kill the worker thread
                log.exception("clip failed for %s", item[0])

    threading.Thread(target=worker, daemon=True).start()
    if ccfg.get("enabled", False):
        threading.Thread(target=clip_worker, daemon=True).start()

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="squirrel-guard-frigate")
    c.username_pw_set(os.environ.get("SG_MQTT_USER"), os.environ.get("SG_MQTT_PASS"))
    c.reconnect_delay_set(1, 60)

    def on_connect(cl, ud, fl, rc, pr):
        log.info("MQTT connected (%s); listening on frigate/events + frigate/reviews for %s", rc, sorted(labels))
        cl.subscribe([("frigate/events", 0), ("frigate/reviews", 0)])

    c.on_connect = on_connect
    c.on_message = on_message
    log.info("Frigate API %s; verifier %s (for %s); actuator dry_run=%s; clips %s", api,
             ver_cfg["url"] if verifier else "off", sorted(verify_labels), actuator.dry_run,
             "on" if ccfg.get("enabled") else "off")
    c.connect(os.environ["SG_MQTT_HOST"], int(os.environ.get("SG_MQTT_PORT", "1883")), keepalive=60)
    c.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
