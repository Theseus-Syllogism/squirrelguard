"""Vehicle occupant faces: find faces INSIDE a vehicle box and build per-occupant face templates for one visit.

OPT-IN ([profiler] vehicle_occupant_faces = true, see docs/faceid-v2.md "Vehicle occupants"). The templates only
leave the profiler to be compared, on the recorder host, with ENROLLED household/known templates (facematch.decide_any, the
same gates and thresholds as people). The bridge stores the count and the decision; it never stores the vector or a
crop of a face that did not match, never creates an identity from an occupant and never clusters unknown drivers.

PRIVACY RULES are those of faceid.py: names only for enrolled consenting people, no race/ethnicity output (InspireFace
attributes stay off here), no outside lookups, no super-resolution (crops are upscaled bilinearly so the detector sees
small faces at a usable size: interpolation adds no identity detail).

PIPELINE (per sampled full-resolution frame with the visit's vehicle detected)
  vehicle box -> cabin region (upper part of the box: windshield + side windows, a small margin)
  -> overlapping square tiles of `tile_px` frame pixels, each upscaled to the engine's detector size
  -> InspireFace detect (all faces, not just the largest) -> dedupe faces found twice in overlapping tiles
  -> group the faces of the visit into occupants (embedding cosine to each group's running mean, plus the face's
     horizontal position in the vehicle box as a tie-break) -> one FaceAggregator per occupant (quality-weighted,
     outlier rejection: the person path's aggregator) -> InspireFace template + AdaFace template (faceid_v2) on the
     same pooled faces.
Within-visit grouping only: groups never outlive the visit and are never compared with each other across visits.

CPU: at most `max_cpu_s` of detector time per visit (tiles are cut only where a vehicle was detected).
"""
from __future__ import annotations

import dataclasses
import logging
import math
import time

import cv2
import numpy as np

from .faceid import AggParams, FaceAggregator, FaceObs, l2n

log = logging.getLogger("squirrel_guard.occupants")
PERSON_LABELS = {"person", "man", "woman", "boy", "girl"}   # detector labels of people (profiler.PEOPLE)


@dataclasses.dataclass
class OccParams:
    region_top: float = -0.05      # cabin region, as fractions of the vehicle box height (negative: above the box;
    region_bottom: float = 0.70    # a low camera clips roofs) ...
    region_margin: float = 0.04    # ... and a horizontal margin as a fraction of the box width
    tile_px: int = 320             # tile side in frame pixels (detector sees it at det_px: 2x for 320 -> 640)
    det_px: int = 640              # upscale target = the engine's detect_pixel_level
    max_upscale: float = 4.0       # small cabins: never more than this (interpolation only)
    overlap: float = 0.25
    max_tiles: int = 12            # per frame (largest vehicles get coarser tiles instead)
    min_vehicle_px: int = 96       # vehicle box side below this: occupants cannot be resolved, skip the frame
    group_sim: float = 0.30        # InspireFace cosine to an occupant group's mean to join it
    max_groups: int = 4
    max_cpu_s: float = 12.0
    keep_crops: int = 1            # evidence crop per occupant (the bridge keeps it ONLY for an enrolled match)


def cabin_region(box, frame_shape, p: OccParams) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if min(w, h) < p.min_vehicle_px:
        return None
    H, W = frame_shape[:2]
    X1, X2 = max(0.0, x1 - p.region_margin * w), min(float(W), x2 + p.region_margin * w)
    Y1, Y2 = max(0.0, y1 + p.region_top * h), min(float(H), y1 + p.region_bottom * h)
    if X2 - X1 < 16 or Y2 - Y1 < 16:
        return None
    return int(X1), int(Y1), int(math.ceil(X2)), int(math.ceil(Y2))


def tiles(region, p: OccParams) -> list[tuple[int, int, int, int]]:
    """Overlapping square tiles covering the region (one tile when it is small). The tile side grows when more than
    max_tiles would be needed, so the cost per frame stays bounded."""
    X1, Y1, X2, Y2 = region
    w, h = X2 - X1, Y2 - Y1
    if max(w, h) <= p.tile_px * 1.25:          # small cabin: the whole region is one tile
        return [(X1, Y1, X2, Y2)]
    side = p.tile_px
    while True:
        step = max(1, int(side * (1 - p.overlap)))
        nx = 1 if w <= side else 1 + math.ceil((w - side) / step)
        ny = 1 if h <= side else 1 + math.ceil((h - side) / step)
        if nx * ny <= p.max_tiles:
            break
        side = int(side * 1.25)
    xs = [X1] if w <= side else [X1 + min(i * step, w - side) for i in range(nx)]
    ys = [Y1] if h <= side else [Y1 + min(j * step, h - side) for j in range(ny)]
    return [(int(x), int(y), int(x + min(side, w)), int(y + min(side, h))) for y in ys for x in xs]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def dedupe(obs: list[FaceObs], thr: float = 0.3) -> list[FaceObs]:
    """One face per physical face: overlapping tiles find the same face twice; keep the larger/more confident."""
    out: list[FaceObs] = []
    for o in sorted(obs, key=lambda o: -(o.size * o.det)):
        if o.box is None or all(_iou(o.box, k.box) < thr for k in out if k.box is not None):
            out.append(o)
    return out


def _inside(face_box, person_box, frac: float = 0.6) -> bool:
    """Is at least `frac` of the face box inside the person box?"""
    ix = max(0.0, min(face_box[2], person_box[2]) - max(face_box[0], person_box[0]))
    iy = max(0.0, min(face_box[3], person_box[3]) - max(face_box[1], person_box[1]))
    a = max(1e-6, (face_box[2] - face_box[0]) * (face_box[3] - face_box[1]))
    return ix * iy / a >= frac


@dataclasses.dataclass
class _Group:
    agg: FaceAggregator
    mean: np.ndarray
    n: int = 0
    rel_x: float = 0.5             # mean horizontal position in the vehicle box (0 = left edge)


class OccupantSampler:
    """Per-visit occupant face collector. on_frame(frame, vehicle_box, t) per sampled frame; result() at the end.
    `engine`: faceid.FaceEngine / faceid_v2.FaceEngineV2 (all faces per tile); `ada`: faceid_v2.AdaFace or None."""

    def __init__(self, engine, params: AggParams | None = None, occ: OccParams | None = None, ada=None):
        self.engine, self.ada = engine, ada
        self.p = dataclasses.replace(params or AggParams(), keep_crops=(occ or OccParams()).keep_crops)
        self.o = occ or OccParams()
        self.groups: list[_Group] = []
        self.frames = self.frames_used = self.tiles_run = self.faces_seen = self.dropped_groups = 0
        self.pedestrians = 0                  # faces dropped as belonging to a person standing outside
        self.max_size = 0.0
        self.cpu_s = 0.0

    def done(self) -> bool:
        return self.cpu_s >= self.o.max_cpu_s

    def on_frame(self, fr: np.ndarray, vbox, t: float = 0.0, people=()) -> None:
        """people: person boxes on this frame. A face inside a STANDING person's box (one that reaches below the
        cabin, i.e. someone on the ground next to or in front of the vehicle) is a pedestrian, not an occupant."""
        self.frames += 1
        if vbox is None or self.done():
            return
        reg = cabin_region(vbox, fr.shape, self.o)
        if reg is None:
            return
        self.frames_used += 1
        found: list[FaceObs] = []
        t0 = time.process_time()
        for X1, Y1, X2, Y2 in tiles(reg, self.o):
            img = fr[Y1:Y2, X1:X2]
            if img.size == 0:
                continue
            s = min(self.o.max_upscale, self.o.det_px / max(img.shape[:2]))
            if s > 1.0:                       # the detector's input size; bilinear, adds no detail
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            else:
                s = 1.0
            self.tiles_run += 1
            try:
                found += self.engine.faces(np.ascontiguousarray(img), offset=(X1, Y1), scale=s, t=t, frame=fr,
                                           largest_only=False, want_crop=lambda o: True)
            except Exception:
                log.exception("occupant face analysis failed")
            if time.process_time() - t0 + self.cpu_s >= self.o.max_cpu_s:
                break
        self.cpu_s += time.process_time() - t0
        vx1, vx2 = float(vbox[0]), float(vbox[2])
        cabin_bottom = float(vbox[1]) + self.o.region_bottom * (float(vbox[3]) - float(vbox[1]))
        standing = [b for b in people or () if float(b[3]) > cabin_bottom + 0.1 * (float(vbox[3]) - float(vbox[1]))]
        for f in dedupe(found):
            if f.box and any(_inside(f.box, b) for b in standing):
                self.pedestrians += 1
                continue
            self.faces_seen += 1
            self.max_size = max(self.max_size, f.size)
            rx = ((f.box[0] + f.box[2]) / 2 - vx1) / max(1.0, vx2 - vx1) if f.box else 0.5
            self._assign(f, rx)

    def _assign(self, f: FaceObs, rx: float) -> None:
        v = l2n(f.vec)
        best, bs = None, -1.0
        for g in self.groups:
            s = float(v @ g.mean) - 0.05 * abs(rx - g.rel_x)     # position: a small tie-break only
            if s > bs:
                best, bs = g, s
        if best is None or bs < self.o.group_sim:
            if len(self.groups) >= self.o.max_groups:
                self.dropped_groups += 1
                return
            best = _Group(FaceAggregator(self.p), v.copy(), 0, rx)
            self.groups.append(best)
        best.n += 1
        best.mean = l2n(best.mean * (best.n - 1) + v)
        best.rel_x += (rx - best.rel_x) / best.n
        best.agg.add(f)

    def _template(self, g: _Group) -> dict:
        r = g.agg.result()
        r["model"] = getattr(self.engine, "pack", None)
        r["rel_x"] = round(g.rel_x, 2)
        r.pop("faces", None)                  # per-face table: not needed downstream
        r.pop("crops_b64", None)              # one crop at most (best_crop_b64), dropped by the bridge unless matched
        r["apparent"] = {}                    # occupants: no apparent expression/age/gender at all
        if self.ada is None or r.get("vec") is None:
            return r
        from . import faceid_v2
        faces = [f for _, f in g.agg.pool if getattr(f, "aligned", None) is not None]
        if faces:
            try:
                V, n = self.ada.embed([f.aligned for f in faces])
                t = faceid_v2.aggregate(faces, V, g.agg.p)
                if t is not None:
                    r["v2"] = {"version": faceid_v2.V2_VERSION, "model": self.ada.name,
                               "vec": [round(float(x), 5) for x in t], "n_faces": len(faces),
                               "ada_norm_med": round(float(np.median(n)), 2)}
            except Exception:
                log.exception("occupant AdaFace template failed")
        for _, f in g.agg.pool:
            f.aligned = None
        return r

    def result(self, max_templates: int = 3) -> dict:
        """{"occupants_detected": n, "occupant_faces": [faceid-like template dicts, best first], stats...}.
        occupants_detected counts occupant groups with at least one face of >= min_px (a usable face)."""
        groups = sorted(self.groups, key=lambda g: -sum(w for w, _ in g.agg.pool))
        usable = [g for g in groups if g.agg.pool]
        out = {"occupants_detected": len(usable), "faces_seen": self.faces_seen, "frames": self.frames,
               "frames_with_vehicle": self.frames_used, "tiles": self.tiles_run, "cpu_s": round(self.cpu_s, 2),
               "max_size_px": round(self.max_size, 1), "groups_dropped": self.dropped_groups,
               "pedestrian_faces": self.pedestrians,
               "occupant_faces": [self._template(g) for g in usable[:max_templates]]}
        if not usable:
            out["reason"] = "no_face" if self.faces_seen == 0 else "too_small"
        return out



def sample_clip(clip: bytes, det_boxes, sampler: OccupantSampler, fps: float = 2.0, max_s: float = 20.0,
                max_frames: int = 8, t0: float = 0.0) -> int:
    """Two passes over one recording clip, so the CPU budget covers the whole visit instead of its first seconds:
    1) decode at `fps`, det_boxes(frame) -> vehicle box or None, or (vehicle box, person boxes); 2) decode again and
    run the face tiles on up to
    `max_frames` frames with a vehicle, evenly spread (largest boxes first among ties). Frames are never buffered.
    Returns the number of frames given to the sampler."""
    from .kinematics import decode
    gen, _ = decode(clip, fps, max_s)
    boxes = []
    for i, fr in enumerate(gen):
        b = det_boxes(fr)
        b, ppl = b if isinstance(b, tuple) and len(b) == 2 and (b[0] is None or len(b[0]) == 4) else (b, ())
        if b is not None:
            boxes.append((i, (b, list(ppl))))
    if not boxes:
        return 0
    k = min(max_frames, len(boxes))
    pick = {boxes[int(round(j * (len(boxes) - 1) / max(1, k - 1)))][0]: None for j in range(k)}
    want = {i: b for i, b in boxes if i in pick}
    gen, rfps = decode(clip, fps, max_s)
    n = 0
    for i, fr in enumerate(gen):
        if i in want:
            sampler.on_frame(fr, want[i][0], t=t0 + i / rfps, people=want[i][1])
            n += 1
            if sampler.done():
                break
    return n
