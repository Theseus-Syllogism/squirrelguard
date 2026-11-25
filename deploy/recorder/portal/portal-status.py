#!/usr/bin/env python3
"""Write status.json for the camstack portal: service health, camera, recording freshness, disk, IDS mirror.
Runs every minute from portal-status.timer (root: reads the recordings dir and interface counters)."""
import json, os, socket, ssl, subprocess, time, urllib.error, urllib.request
from pathlib import Path

OUT = Path(os.environ.get("PORTAL_DIR", "/srv/camstack/portal/www")) / "status.json"
REC = Path("/mnt/cloning/camstack-video")
STATE = Path("/run/portal-status.mirror")
HTTP = {
    "frigate": "https://192.0.2.30:8971/api/version",   # gpuserver (the recorder host pass-through is gone; SSO via nginx)
    "persons": "http://127.0.0.1:8092/",
    "files": "http://127.0.0.1:8091/health",
    "go2rtc": "http://127.0.0.1:1984/",
    "grafana": "http://127.0.0.1:3001/api/health",
    "ntfy": "http://127.0.0.1:8090/v1/health",
    "kuma": "http://127.0.0.1:3002/",
}
TCP = {"mqtt": ("127.0.0.1", 1883)}
NOVERIFY = ssl.create_default_context(); NOVERIFY.check_hostname = False; NOVERIFY.verify_mode = ssl.CERT_NONE


def http(url):
    t = time.monotonic()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=5, context=NOVERIFY)
        code = r.status
    except urllib.error.HTTPError as e:   # 401/403 = service answering behind its login
        code = e.code
    except Exception as e:
        return {"up": False, "error": type(e).__name__}
    return {"up": code < 500, "code": code, "ms": round((time.monotonic() - t) * 1000)}


def tcp(host, port):
    t = time.monotonic()
    try:
        socket.create_connection((host, port), timeout=3).close()
        return {"up": True, "ms": round((time.monotonic() - t) * 1000)}
    except OSError as e:
        return {"up": False, "error": type(e).__name__}


def newest_segment():
    best = 0.0
    for p in REC.glob("*/*.mkv") if REC.is_dir() else []:
        best = max(best, p.stat().st_mtime)
    for p in REC.glob("*.mkv") if REC.is_dir() else []:
        best = max(best, p.stat().st_mtime)
    return round(time.time() - best) if best else None


def mirror():
    p = Path("/sys/class/net/mirror0/statistics/tx_bytes")
    if not p.exists():
        return {"up": False}
    now, tx = time.time(), int(p.read_text())
    mbps = None
    try:
        t0, tx0 = (float(v) for v in STATE.read_text().split())
        if now > t0 and tx >= tx0:
            mbps = round((tx - tx0) * 8 / (now - t0) / 1e6, 1)
    except (OSError, ValueError):
        pass
    STATE.write_text(f"{now} {tx}")
    filt = subprocess.run(["tc", "filter", "show", "dev", "wlx00c0cab8fc2a", "ingress"], capture_output=True, text=True).stdout
    return {"up": "mirred" in filt, "mbps": mbps}


st = {"time": time.time(), "services": {k: http(u) for k, u in HTTP.items()} | {k: tcp(*a) for k, a in TCP.items()}}
CAMERAS = {"frontyard": ("198.51.100.103", 554), "v4": ("198.51.100.104", 554)}
st["cameras"] = {}
for name, addr in CAMERAS.items():
    seg = [p.stat().st_mtime for p in (REC / name).glob("*.mkv")] if (REC / name).is_dir() else []
    st["cameras"][name] = tcp(*addr) | {"newest_age_s": round(time.time() - max(seg)) if seg else None}
st["camera"] = st["cameras"]["frontyard"]                     # kept for older portal pages
st["recording"] = {"newest_age_s": newest_segment()}
d = os.statvfs("/mnt/cloning") if os.path.ismount("/mnt/cloning") else None
st["disk"] = {"free_gb": round(d.f_bavail * d.f_frsize / 1e9), "used_pct": round(100 * (1 - d.f_bavail / d.f_blocks))} if d else {}
st["mirror"] = mirror()
tmp = OUT.with_suffix(".tmp")
tmp.write_text(json.dumps(st))
tmp.chmod(0o644)
tmp.replace(OUT)
