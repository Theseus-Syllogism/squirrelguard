"""Configure Uptime Kuma on the recorder host: admin account, ntfy alert channel, camstack monitors. Re-runnable.

Credentials are read from the environment (piped from /srv/camstack/.env, never printed):
  KUMA_USER KUMA_PASS NTFY_ALERT_USER NTFY_ALERT_PASS   (push token is written to fd 3)
Usage: ssh db-host 'grep -E "^(KUMA_|NTFY_ALERT_)" /srv/camstack/.env' | .venv/bin/python scripts/kuma_setup.py
"""
import os
import sys

from uptime_kuma_api import MonitorType, NotificationType, UptimeKumaApi

env = dict(l.rstrip("\n").split("=", 1) for l in sys.stdin if "=" in l and not l.startswith("#"))
env.update({k: v for k, v in os.environ.items() if k.startswith(("KUMA_", "NTFY_ALERT_"))})
URL = env.get("KUMA_URL", "http://192.0.2.20:3002")

api = UptimeKumaApi(URL, timeout=30)
if api.need_setup():
    api.setup(env["KUMA_USER"], env["KUMA_PASS"])
    print("admin account created")
api.login(env["KUMA_USER"], env["KUMA_PASS"])

# Alerts go to a separate ntfy topic; ntfy is reached over the compose network (no host hop, no Suricata).
notes = {n["name"]: n for n in api.get_notifications()}
if "ntfy camstack-alerts" not in notes:
    r = api.add_notification(
        name="ntfy camstack-alerts", type=NotificationType.NTFY, isDefault=True, applyExisting=True,
        ntfyserverurl="http://ntfy:80", ntfytopic="camstack-alerts", ntfyPriority=4,
        ntfyusername=env["NTFY_ALERT_USER"], ntfypassword=env["NTFY_ALERT_PASS"])
    note_id = r["id"]
    print("ntfy notification added")
else:
    note_id = notes["ntfy camstack-alerts"]["id"]

M = MonitorType
monitors = [
    dict(type=M.PORT, name="Camera RTSP (198.51.100.103:554)", hostname="198.51.100.103", port=554),
    dict(type=M.PORT, name="Camera SSH (198.51.100.103:22)", hostname="198.51.100.103", port=22, interval=300),
    dict(type=M.PORT, name="go2rtc RTSP", hostname="go2rtc", port=8554),
    dict(type=M.PORT, name="MQTT broker", hostname="192.0.2.20", port=1883),  # mosquitto runs on the host network
    dict(type=M.HTTP, name="ntfy", url="http://ntfy:80/v1/health"),
    dict(type=M.HTTP, name="Grafana", url="http://192.0.2.20:3001/api/health"),
    dict(type=M.HTTP, name="File Browser", url="http://filebrowser:80/health"),
    dict(type=M.PUSH, name="Recorder writing segments", interval=600),  # library generates the token
]
existing = {m["name"] for m in api.get_monitors()}
for m in monitors:
    if m["name"] in existing:
        continue
    m.setdefault("interval", 60)
    m.setdefault("retryInterval", 60)
    m.setdefault("maxretries", 2)
    api.add_monitor(notificationIDList=[note_id], **m)
    print("added monitor:", m["name"])
# Print the push monitor's token as KEY=VALUE on fd 3 if open (caller stores it; never shown on stdout)
push = next(m for m in api.get_monitors() if m["name"] == "Recorder writing segments")
try:
    with os.fdopen(3, "w") as fd3:
        fd3.write(f"KUMA_PUSH_TOKEN={push['pushToken']}\n")
except OSError:
    pass
api.disconnect()
print("done")
