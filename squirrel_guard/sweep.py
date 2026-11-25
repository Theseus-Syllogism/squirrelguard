"""Small-person sweeper: finds distant walkers that Frigate's detector misses, and makes them Frigate events.

Why: Frigate runs YOLOv8s-OIV7 (kept for its squirrel class). On a ~65 px walker 40 m away on v4 it scores person
0.02-0.08 (needs 0.35), and still only 0.1-0.24 when zoomed. The COCO-trained YOLOv8s-pose that the profiler already
uses scores the same person 0.53-0.78 once the region is zoomed 2x (2026-11-10 15:38 miss, docs/sweep.md).

How: once a second per camera, take Frigate's latest frame (detect resolution), find small moving blobs by frame
differencing, run the COCO person model on a zoomed tile (default 320 px -> 640 model input) around each, and keep
a short track. A person seen in >= 2 sweeps with no Frigate person event in progress becomes a Frigate event
(POST /api/events/<camera>/person/create, with recording), which is then queued for the normal profiler analysis.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests

from squirrel_guard.pose import PoseEstimator

log = logging.getLogger("sweep")


@dataclass
class Track:
    box: tuple                                  # x1, y1, x2, y2 (detect-frame px)
    first: float
    last: float
    hits: int = 1
    best: float = 0.0
    emitted: bool = False
    boxes: list = field(default_factory=list)


class Sweeper(threading.Thread):
    def __init__(self, api: str, cameras: list[str], pose_model: str, enqueue, cfg: dict):
        super().__init__(daemon=True, name="sweep")
        self.api, self.cameras, self.enqueue = api.rstrip("/"), cameras, enqueue
        self.pose = PoseEstimator(pose_model, conf=float(cfg.get("sweep_min_score", 0.5)),
                                  threads=int(cfg.get("sweep_threads", 2)))
        self.interval = 1.0 / float(cfg.get("sweep_fps", 1.0))
        self.tile = int(cfg.get("sweep_tile", 320))
        self.max_tiles = int(cfg.get("sweep_max_tiles", 6))
        self.max_h_frac = float(cfg.get("sweep_max_height_frac", 0.25))   # bigger people: Frigate finds them
        self.min_h = int(cfg.get("sweep_min_height", 28))
        self.hits_needed = int(cfg.get("sweep_hits", 2))
        self.duration = int(cfg.get("sweep_event_s", 20))
        self.settle = float(cfg.get("settle_s", 15))
        self.prev: dict[str, np.ndarray] = {}
        self.tracks: dict[str, list[Track]] = {c: [] for c in cameras}

    # -- frame + motion ---------------------------------------------------------------------------------------
    def frame(self, cam: str):
        try:
            r = requests.get(f"{self.api}/api/{cam}/latest.jpg", params={"quality": 92}, timeout=10)
            if r.status_code == 200 and r.content:
                return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        except requests.RequestException:
            pass
        return None

    def motion_boxes(self, cam: str, img: np.ndarray) -> list[tuple]:
        H, W = img.shape[:2]
        s = 640 / W
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img, (640, int(H * s))), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        prev, self.prev[cam] = self.prev.get(cam), g
        if prev is None or prev.shape != g.shape:
            return []
        m = cv2.dilate((cv2.absdiff(g, prev) > 18).astype(np.uint8), np.ones((5, 5), np.uint8))
        m[: int(0.035 * m.shape[0])] = 0                              # on-screen clock / title row
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        out = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < 6 or h / s > self.max_h_frac * H:
                continue
            out.append((x / s, y / s, (x + w) / s, (y + h) / s, area))
        out.sort(key=lambda b: -b[4])
        return out[: self.max_tiles * 3]

    # -- person check on zoomed tiles -------------------------------------------------------------------------
    def people(self, img: np.ndarray, blobs: list[tuple]) -> list[tuple]:
        H, W = img.shape[:2]
        found, tiles = [], []
        for x1, y1, x2, y2, _ in blobs:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if any(abs(cx - tx) < self.tile * 0.35 and abs(cy - ty) < self.tile * 0.35 for tx, ty in tiles):
                continue                                              # already covered by a tile
            tiles.append((cx, cy))
            if len(tiles) > self.max_tiles:
                break
            X1 = int(np.clip(cx - self.tile / 2, 0, W - self.tile))
            Y1 = int(np.clip(cy - self.tile / 2 - self.tile * 0.1, 0, H - self.tile))
            for p in self.pose(img[Y1:Y1 + self.tile, X1:X1 + self.tile]):
                b = (p.box[0] + X1, p.box[1] + Y1, p.box[2] + X1, p.box[3] + Y1)
                h = b[3] - b[1]
                if self.min_h <= h <= self.max_h_frac * H:
                    found.append((b, p.score))
        return found

    # -- tracking + Frigate events ----------------------------------------------------------------------------
    def frigate_busy(self, cam: str) -> bool:
        try:
            r = requests.get(f"{self.api}/api/events", params={"camera": cam, "label": "person", "in_progress": 1,
                                                                "limit": 5}, timeout=10)
            return r.status_code == 200 and len(r.json()) > 0
        except (requests.RequestException, ValueError):
            return True                                               # unsure: do not duplicate

    def emit(self, cam: str, t: Track, W: int, H: int):
        x1, y1, x2, y2 = t.box
        rel = [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]
        body = {"score": round(t.best, 3), "duration": self.duration, "include_recording": True,
                "pre_capture": max(3, int(time.time() - t.first) + 2),
                "draw": {"boxes": [{"box": rel, "color": [255, 160, 0], "score": int(t.best * 100)}]}}
        try:
            r = requests.post(f"{self.api}/api/events/{cam}/person/create", json=body, timeout=15)
            eid = r.json().get("event_id") if r.status_code == 200 else None
        except (requests.RequestException, ValueError):
            eid = None
        if not eid:
            log.warning("could not create a Frigate event for a swept person on %s (%s)", cam,
                        getattr(locals().get("r"), "status_code", "?"))
            return
        log.info("swept person on %s: score %.2f, %d px tall -> Frigate event %s", cam, t.best, int(y2 - y1), eid)
        boxes = [[b[0] / W, b[1] / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H] for b in t.boxes[-8:]]

        def later():
            try:
                a = requests.get(f"{self.api}/api/events/{eid}", timeout=15).json()
            except (requests.RequestException, ValueError):
                return
            a.setdefault("data", {})
            a["data"].setdefault("box", rel)
            a["_sweep"] = {"boxes": boxes, "score": t.best}
            self.enqueue(a)
        threading.Timer(self.duration + self.settle, later).start()

    def step(self, cam: str):
        img = self.frame(cam)
        if img is None:
            return
        H, W = img.shape[:2]
        now = time.time()
        found = self.people(img, self.motion_boxes(cam, img))
        tracks = [t for t in self.tracks[cam] if now - t.last < 6]
        for b, sc in found:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            best = min(tracks, key=lambda t: np.hypot((t.box[0] + t.box[2]) / 2 - cx, (t.box[1] + t.box[3]) / 2 - cy),
                       default=None)
            if best is not None and np.hypot((best.box[0] + best.box[2]) / 2 - cx,
                                             (best.box[1] + best.box[3]) / 2 - cy) < 3 * (b[3] - b[1]) + 60:
                best.box, best.last, best.hits, best.best = b, now, best.hits + 1, max(best.best, sc)
                best.boxes.append(b)
            else:
                tracks.append(Track(b, now, now, 1, sc, boxes=[b]))
        for t in tracks:
            if not t.emitted and t.hits >= self.hits_needed:
                t.emitted = True
                if self.frigate_busy(cam):
                    log.debug("swept person on %s, but Frigate already has a person event", cam)
                else:
                    self.emit(cam, t, W, H)
        self.tracks[cam] = tracks

    def run(self):
        log.info("small-person sweeper on %s: %.1f fps, %d px tiles, COCO person model", self.cameras,
                 1 / self.interval, self.tile)
        while True:
            t0 = time.time()
            for cam in self.cameras:
                try:
                    self.step(cam)
                except Exception:  # noqa: BLE001 - keep sweeping
                    log.exception("sweep step failed on %s", cam)
            time.sleep(max(0.05, self.interval - (time.time() - t0)))
