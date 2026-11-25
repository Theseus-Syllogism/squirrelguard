#!/usr/bin/env python3
"""Push high-severity alerts from the recorder host IDS (gpuserver suricata-recorder) to ntfy topic camstack-alerts.

Follows /var/log/suricata/recorder/eve.json (survives logrotate). Severity 1 only, minus known-benign noise;
one push per (signature, source) per 6 h, at most 10 pushes per hour.
Env (/etc/suricata/recorder-notify.env): NTFY_URL, NTFY_TOKEN.
"""
import json, os, time, urllib.request

EVE = "/var/log/suricata/recorder/eve.json"
URL = os.environ.get("NTFY_URL", "http://192.0.2.20:8090/camstack-alerts")
TOKEN = os.environ["NTFY_TOKEN"]
# benign on this network (reviewed 2026-11-05): banner-based regreSSHion (OpenSSH 9.6p1-3ubuntu13.20 is patched),
# SSH/HTTP/TLS on non-standard ports (2200, 8554, 8971 by design), LAN multicast discovery, TLS1.0 probe,
# recorder host's DNS via lan-gw, apt mirror responses without Content-Type
IGNORE_SIDS = {3321289, 3300305, 3300303, 3300298, 3300145, 3300153, 3300154, 3300246, 9000110, 9000045}
IGNORE_PREFIX = ("SURICATA ", "TGI HUNT")
DEDUPE_S, MAX_PER_H = 6 * 3600, 10
seen, sent, skipped = {}, [], 0


def push(e):
    a = e["alert"]
    body = (f"{a['signature']}\n{e.get('src_ip')}:{e.get('src_port', '')} -> {e.get('dest_ip')}:{e.get('dest_port', '')} "
            f"{e.get('proto', '')}\nsid {a['signature_id']}, {a.get('category', '')}, action {a.get('action')}")
    req = urllib.request.Request(URL, data=body.encode(), method="POST", headers={
        "Authorization": f"Bearer {TOKEN}", "Title": "recorder host network IDS alert", "Priority": "high", "Tags": "rotating_light"})
    urllib.request.urlopen(req, timeout=10).read()


def follow(path):
    f, ino = None, None
    while True:
        if f is None:
            try:
                f = open(path); ino = os.fstat(f.fileno()).st_ino; f.seek(0, 2)
            except OSError:
                time.sleep(5); continue
        line = f.readline()
        if line:
            yield line
            continue
        time.sleep(1)
        try:
            if os.stat(path).st_ino != ino:   # rotated: finish the old file, then reopen
                rest = f.read()
                for l in rest.splitlines(True):
                    yield l
                f.close(); f = None; ino = None
                f = open(path); ino = os.fstat(f.fileno()).st_ino
        except OSError:
            pass


for line in follow(EVE):
    if '"event_type":"alert"' not in line:
        continue
    try:
        e = json.loads(line)
    except ValueError:
        continue
    a = e["alert"]
    if a.get("severity") != 1 or a["signature_id"] in IGNORE_SIDS or a["signature"].startswith(IGNORE_PREFIX):
        continue
    now = time.time()
    key = (a["signature_id"], e.get("src_ip"))
    if now - seen.get(key, 0) < DEDUPE_S:
        continue
    sent[:] = [t for t in sent if now - t < 3600]
    if len(sent) >= MAX_PER_H:
        skipped += 1
        continue
    seen[key] = now
    try:
        push(e)
        sent.append(now)
        print(f"pushed sid {a['signature_id']} from {e.get('src_ip')} ({skipped} rate-limited so far)", flush=True)
    except Exception as ex:
        print(f"ntfy push failed: {ex}", flush=True)
