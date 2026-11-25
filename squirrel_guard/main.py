"""squirrel-guard: RTSP -> motion gate -> YOLO detector -> local VLM verifier -> deterrent, with an event log.

Run:  .venv/bin/python -m squirrel_guard.main -c config.toml
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import logging
import os
import time
import tomllib
from pathlib import Path

import cv2

from .actuator import Actuator
from .detector import Detector
from .motion import MotionGate
from .publish import Publisher
from .source import frames
from .verifier import Verifier

log = logging.getLogger("squirrel_guard")


def load_env(path: Path) -> None:
    """Minimal KEY=VALUE loader for the untracked .env (keeps the RTSP password out of config and git)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def annotate(frame, dets):
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(out, f"{d.label} {d.score:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg_path = Path(args.config)
    cfg = tomllib.loads(cfg_path.read_text())
    load_env(cfg_path.parent / ".env")
    url = os.environ.get("SG_RTSP_URL") or cfg["camera"].get("url")
    if not url:
        raise SystemExit("set SG_RTSP_URL in .env (rtsp://user:pass@host:554/ch1)")

    cam, det_cfg, ver_cfg, act_cfg, trig = cfg["camera"], cfg["detector"], cfg["verifier"], cfg["actuator"], cfg["trigger"]
    out_dir = Path(cfg.get("output", {}).get("dir", "data/events"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_all = cfg.get("output", {}).get("save_all_detections", True)

    gate = MotionGate(**cfg.get("motion", {}))
    detector = Detector(det_cfg["model"], conf=det_cfg.get("conf", 0.25), threads=det_cfg.get("threads", 4))
    targets = {t.lower() for t in det_cfg.get("targets", ["Squirrel"])}
    verifier = Verifier(ver_cfg["url"], model=ver_cfg.get("model", ""), timeout=ver_cfg.get("timeout", 60)) if ver_cfg.get("enabled", True) else None
    actuator = Actuator(act_cfg.get("command", ""), act_cfg.get("cooldown_s", 60), act_cfg.get("dry_run", True))
    publisher = Publisher(cam.get("name", "frontyard"))

    # require hits in `need` of the last `window` processed frames before spending a verifier call
    window = collections.deque(maxlen=trig.get("window", 6))
    need = trig.get("need", 3)
    log.info("model classes: %d; targets: %s; verifier: %s; actuator dry_run=%s",
             len(detector.names), sorted(targets), "on" if verifier else "off", actuator.dry_run)

    stats = collections.Counter()
    t_stats = time.monotonic()
    for frame in frames(url, cam.get("width", 640), cam.get("height", 360), cam.get("fps", 3)):
        stats["frames"] += 1
        if time.monotonic() - t_stats > 300:
            log.info("last 5 min: %s", dict(stats))
            stats.clear()
            t_stats = time.monotonic()

        if not gate(frame):
            window.append(False)
            continue
        stats["motion"] += 1
        t0 = time.perf_counter()
        dets = detector(frame)
        ms = (time.perf_counter() - t0) * 1000
        hits = [d for d in dets if d.label.lower() in targets]
        window.append(bool(hits))
        if dets:
            log.debug("%.0f ms: %s", ms, ", ".join(f"{d.label}:{d.score:.2f}" for d in dets[:5]))

        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        if dets and save_all:
            # training data for later fine-tuning: every frame where the detector saw anything
            cv2.imwrite(str(out_dir / f"{ts}_raw.jpg"), frame)
        if not hits or sum(window) < need:
            continue

        stats["candidates"] += 1
        best = hits[0]
        verdict = None
        if verifier:
            t1 = time.perf_counter()
            verdict = verifier(verifier.crop(frame, best.box))
            log.info("verifier said %r in %.1fs", verdict, time.perf_counter() - t1)
        confirmed = verdict == "squirrel" if verifier else True
        if verifier and verdict is None:
            confirmed = ver_cfg.get("trust_detector_if_down", False)

        action = actuator.fire() if confirmed else "rejected"
        stats[action] += 1
        window.clear()
        snap_name = f"{ts}_event.jpg"
        annotated = annotate(frame, dets)
        cv2.imwrite(str(out_dir / snap_name), annotated)
        record = {"ts": ts, "time": dt.datetime.now().astimezone().isoformat(timespec="milliseconds"),
                  "label": best.label, "score": round(best.score, 3), "box": list(best.box),
                  "verifier": verdict, "confirmed": confirmed, "action": action, "detect_ms": round(ms),
                  "snapshot": snap_name}
        with open(out_dir / "events.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        publisher.event(record, jpg.tobytes() if ok else None)
        log.info("EVENT %s", record)


if __name__ == "__main__":
    main()
