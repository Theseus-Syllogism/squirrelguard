"""easelscan: photograph the easel pad, straighten it, read it with the local vision LLM, archive it.

  python -m easelscan scan [--force] [--no-read]     one scan (reads only when the writing changed)
  python -m easelscan watch                          scan every EASEL_INTERVAL_S seconds

Environment:
  EASEL_CAPTURE_CMD  command printing concatenated MJPEG frames on stdout
                     (default: ssh to the Pi; on the Pi the key is locked to the capture command)
  EASEL_CALIB        calib.json (corners of the sheet in the upright frame)
  EASEL_OUT          archive directory
  EASEL_LLM_URL      OpenAI-compatible vision LLM (gpuserver swap gateway, /q35 = Qwen 35B with vision: http://192.0.2.30:5990/q35)
  EASEL_VOCAB        comma-separated words the writer often uses (improves reading)
  NTFY_URL, NTFY_PUB_TOKEN   push a summary when a changed page has been read (optional)
Privacy: only the straightened page crop is stored; the camera frame (room, people) is discarded in memory.
A person in front of the page (occlusion) skips the scan.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2

from . import outputs, page, reader

E = os.environ
DEFAULT_CAPTURE = ("ssh -o BatchMode=yes -o ConnectTimeout=10 pi@198.51.100.1 "
                   "'ffmpeg -hide_banner -loglevel error -f v4l2 -input_format mjpeg -video_size 1920x1080 "
                   "-framerate 5 -i /dev/video0 -frames:v 30 -c:v copy -f mjpeg -'")


def capture(n_keep: int = 8) -> list:
    blob = subprocess.run(E.get("EASEL_CAPTURE_CMD", DEFAULT_CAPTURE), shell=True, capture_output=True,
                          timeout=90).stdout
    frames = page.split_jpegs(blob)
    if len(frames) < 3:
        raise RuntimeError(f"capture returned {len(frames)} frames")
    return frames[-n_keep:]                      # the first frames are exposure/focus settling


def scan_once(force: bool = False, do_read: bool = True) -> dict:
    out = Path(E.get("EASEL_OUT", "data/easel"))
    out.mkdir(parents=True, exist_ok=True)
    calib = page.load_calib(E.get("EASEL_CALIB", Path.home() / ".config/easelscan/calib.json"))
    now = dt.datetime.now().astimezone()
    frames = capture()
    pages = [page.straighten(f, calib) for f in frames]
    del frames                                   # full camera frames never leave memory
    pg = page.average(pages)
    occ = page.occluded(pg, pages)
    if occ > 0.03 and not force:
        return {"status": "skipped", "reason": f"page occluded ({occ:.0%})"}
    cl = page.clean(pg)
    state = out / "last_clean.png"
    prev = cv2.imread(str(state), cv2.IMREAD_GRAYSCALE) if state.exists() else None
    ch, share = page.changed(prev, cl)
    if not ch and not force:
        return {"status": "unchanged", "ink_change": round(share, 4)}
    cv2.imwrite(str(state), cl)
    if not do_read:
        return {"status": "captured", "ink_change": round(share, 4)}
    vocab = [w.strip() for w in E.get("EASEL_VOCAB", "").split(",") if w.strip()]
    d, meta = reader.read_page(cl, E.get("EASEL_LLM_URL", "http://192.0.2.30:5990/q35"), vocab, now.date())
    sd = outputs.write_scan(out, now, cl, d, meta)
    rem = d.get("reminders", [])
    body = (f"{len(d.get('sections', []))} sections, {len(rem)} reminders, {len(d.get('tasks', []))} tasks"
            + ("".join(f"\n⏰ {r['text']}" + (f" ({r['due']})" if r.get('due') else "") for r in rem[:8])))
    try:
        outputs.ntfy(E.get("NTFY_URL", ""), E.get("NTFY_PUB_TOKEN"), "Easel notes scanned", body)
    except Exception as e:  # noqa: BLE001 - a push failure must not lose the scan
        print("ntfy failed:", e, file=sys.stderr)
    return {"status": "read", "dir": str(sd), "ink_change": round(share, 4), "seconds": meta["seconds"],
            "reminders": len(rem), "sections": len(d.get("sections", []))}


def main():
    ap = argparse.ArgumentParser(prog="easelscan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "watch"])
    ap.add_argument("--force", action="store_true", help="read even if unchanged/occluded")
    ap.add_argument("--no-read", action="store_true", help="capture + straighten only (no LLM)")
    a = ap.parse_args()
    if a.cmd == "scan":
        print(json.dumps(scan_once(a.force, not a.no_read)))
        return
    every = float(E.get("EASEL_INTERVAL_S", "600"))
    while True:
        try:
            print(dt.datetime.now().isoformat(timespec="seconds"), json.dumps(scan_once()), flush=True)
        except Exception as e:  # noqa: BLE001 - keep watching
            print(dt.datetime.now().isoformat(timespec="seconds"), "scan failed:", e, flush=True)
        time.sleep(every)


if __name__ == "__main__":
    main()
