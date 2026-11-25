#!/usr/bin/env python3
"""frigate-stall-guard - restart Frigate when a camera's capture is hung; alert when a camera is offline.

Why: on 2026-11-10 at 20:14, Frigate 0.18.2's watchdog for v4 hung inside stop_ffmpeg(). It logged
"Waiting for ffmpeg to exit gracefully..." and that call never returned. This came after a burst of
"exceeded fps limit" restarts, caused by the camera's Wi-Fi stalling and then bursting.

For the next 15 h, v4 served one frozen, corrupted frame at a reported 100 fps. There were no detections,
alerts or review items, and Frigate reported no error. Only a container restart cleared it.

Run every minute by a systemd timer. For each camera in GUARD_CAMERAS (space separated):
  hung     The latest frame is byte-identical for FROZEN_RUNS runs while Frigate still counts frames,
           or camera_fps > 2 x detect fps + 5 (an impossible rate) for FROZEN_RUNS runs.
           Action: docker restart frigate (at most once per RESTART_GAP, at most MAX_RESTARTS a day),
           plus a Telegram notice.
  offline  camera_fps == 0 for OFFLINE_RUNS runs.
           Action: one alert, and one more when the camera is back. No restart: Frigate retries the
           stream by itself, and a restart cannot bring a camera back.
Notices go to the homelab notify relay (app-host Caddy -> Apprise -> Telegram). The bearer token comes from the
systemd credential "notify". State lives in $STATE_DIRECTORY/state.json. Standard library only.
"""
import hashlib
import json
import os
import subprocess
import time
import urllib.request

CAMERAS = os.environ.get("GUARD_CAMERAS", "v4 frontyard").split()
FROZEN_RUNS = int(os.environ.get("FROZEN_RUNS", "4"))
OFFLINE_RUNS = int(os.environ.get("OFFLINE_RUNS", "10"))
RESTART_GAP = int(os.environ.get("RESTART_GAP", "1800"))
MAX_RESTARTS = int(os.environ.get("MAX_RESTARTS", "4"))
CONTAINER = os.environ.get("FRIGATE_CONTAINER", "frigate")
NOTIFY_URL = os.environ.get("NOTIFY_URL", "https://notify.home.example.org/notify/homelab")
STATE = os.path.join(os.environ.get("STATE_DIRECTORY", "/var/lib/frigate-stall-guard"), "state.json")


def api(path, timeout=15):
    """Frigate's internal API (port 5000 is not published), fetched from inside the container."""
    r = subprocess.run(["docker", "exec", CONTAINER, "curl", "-sf", "-m", "10", "http://127.0.0.1:5000" + path],
                       capture_output=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{path}: curl exit {r.returncode}")
    return r.stdout


def notify(title, body):
    print(f"notice: {title}: {body}", flush=True)
    try:
        with open(os.path.join(os.environ["CREDENTIALS_DIRECTORY"], "notify")) as f:
            token = f.read().strip()
        req = urllib.request.Request(NOTIFY_URL, method="POST",
                                     data=json.dumps({"title": title, "body": body, "tag": "critical"}).encode(),
                                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:      # an Apprise 424 (one target failed) still reached Telegram
        print(f"notify failed: {e!r}", flush=True)


def load():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"cams": {}, "restarts": []}


def save(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, STATE)


def main():
    st = load()
    now = time.time()
    st["restarts"] = [t for t in st.get("restarts", []) if now - t < 86400]
    try:
        stats = json.loads(api("/api/stats"))["cameras"]
        cfg = json.loads(api("/api/config"))["cameras"]
    except Exception as e:
        print(f"frigate API unavailable: {e!r}", flush=True)     # container down/starting: not our job
        return
    hung = []
    for cam in CAMERAS:
        if cam not in stats or cam not in cfg:
            continue
        c = st["cams"].setdefault(cam, {"hash": "", "same": 0, "fast": 0, "zero": 0, "offline": False})
        fps = float(stats[cam].get("camera_fps") or 0)
        limit = 2 * float(cfg[cam]["detect"]["fps"]) + 5
        try:
            h = hashlib.sha256(api(f"/api/{cam}/latest.jpg?h=180")).hexdigest()
        except Exception:
            h = ""
        c["same"] = c["same"] + 1 if h and h == c["hash"] else 0
        c["hash"] = h
        c["fast"] = c["fast"] + 1 if fps > limit else 0
        c["zero"] = c["zero"] + 1 if fps == 0 else 0
        if fps > 0 and (c["same"] >= FROZEN_RUNS or c["fast"] >= FROZEN_RUNS):
            why = (f"frame frozen for {c['same']} min" if c["same"] >= FROZEN_RUNS
                   else f"impossible {fps:.0f} fps for {c['fast']} min")
            hung.append(f"{cam} ({why})")
        if c["zero"] >= OFFLINE_RUNS and not c["offline"]:
            c["offline"] = True
            notify(f"📷 Camera {cam} offline", f"Frigate has had no frames from {cam} for {c['zero']} min: no "
                   "detections or alerts from it until it is back. Check its power / Wi-Fi (go2rtc on the recorder host).")
        elif fps > 0 and c["offline"] and c["same"] == 0:
            c["offline"] = False
            notify(f"📷 Camera {cam} back", f"{cam} is delivering frames again ({fps:.0f} fps).")
    if hung:
        recent = [t for t in st["restarts"] if now - t < RESTART_GAP]
        if not recent and len(st["restarts"]) < MAX_RESTARTS:
            r = subprocess.run(["docker", "restart", CONTAINER], capture_output=True, text=True, timeout=180)
            st["restarts"].append(now)
            for c in st["cams"].values():
                c.update(same=0, fast=0, hash="")
            notify("📷 Frigate restarted: camera capture hung",
                   f"{', '.join(hung)}. No detections or alerts were possible from it. "
                   f"docker restart {CONTAINER}: {'ok' if r.returncode == 0 else r.stderr.strip()[:200]}.")
        elif not st.get("gave_up_notified") and len(st["restarts"]) >= MAX_RESTARTS:
            st["gave_up_notified"] = now
            notify("📷 Frigate still hung after restarts",
                   f"{', '.join(hung)}. Restarted {len(st['restarts'])}x in 24 h, now leaving it alone: "
                   "check the camera itself (frozen encoder?) and Frigate's log.")
    if st.get("gave_up_notified") and now - st["gave_up_notified"] > 86400:
        st.pop("gave_up_notified")
    save(st)


if __name__ == "__main__":
    main()
