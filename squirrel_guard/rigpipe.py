"""Glue between clips, whole-body keypoints and the IK rig (squirrel_guard.rig), shared by the demo builders
(scripts/replay_demo.py --rig, scripts/multicam_demo.py --rig, scripts/rigfit.py) and the live profiler.

  pick + keypoints:  kp = extract(frames, pose, wholebody)            -> per-camera dict (kp, box, size, ir)
  fit + attach:      attach_rig(tr, {"v4": (kp, cam, shift)}, fps)     -> tr["rig"], tr["skeleton"] from the rig

With a rig the replay's PATH should come from it: res, views = fit_for_replay(...); re-run trajectory.estimate with
ground_override=rig_ground(res, ...); then attach_rig(tr, ..., res=res, views=views).

A replay ("tr", trajectory.estimate output) and the keypoints must share a time base: frame index i of the primary
camera = replay frame (i - tr["frame0"]). Secondary cameras carry a shift (their frame i + shift = primary frame i).
"""
from __future__ import annotations

import logging
import time

import numpy as np

from . import rig

log = logging.getLogger(__name__)
TORSO = [5, 6, 11, 12]          # COCO shoulders + hips
VERSION = "rigpipe/1"
# Head pose + jaw ("face") and wrist rotation + finger curls ("hands") on top of the body (rig.fit parts=...).
# Set by the profiler from [profiler] rig_parts; () = body only, exactly as before. Used by fit_views here and in fusion.
PARTS: tuple = ()


def single_person(frames, pose, on_frame=None, min_torso=0.3):
    """Per frame the most plausible detection of the ONE person in the scene (score x torso confidence, preferring
    continuity with the previous pick), or None. Unlike kinematics.track this never suspends: partial views behind
    furniture still count (each joint is gated by its own confidence later)."""
    out, size, prev = [], None, None
    for fr in frames:
        size = (fr.shape[1], fr.shape[0])
        best, bs = None, 0.0
        for d in pose(fr):
            torso = float(np.mean(d.kp[TORSO, 2]))
            if torso < min_torso:
                continue
            sc = d.score * torso
            if prev is not None:
                cx, px = (d.box[0] + d.box[2]) / 2, (prev.box[0] + prev.box[2]) / 2
                sc *= 1.0 / (1.0 + abs(cx - px) / (0.25 * size[0]))
            if sc > bs:
                best, bs = d, sc
        out.append(best)
        prev = best or prev
        if on_frame:
            on_frame(fr, best)
    return out, size


def extract(frames, pose, wb, picks: list | None = None, every: int = 1) -> dict:
    """Frames -> whole-body keypoints of the one person. picks: per-frame Pose|None already chosen (e.g. the replay's
    own track) to reuse instead of picking again; every: run RTMW on every n-th frame only (others NaN)."""
    from .wholebody import is_grayscale
    kps, boxes, irs = [], [], []
    t0 = time.time()

    def on_frame(fr, p):
        i = len(kps)
        irs.append(is_grayscale(fr))
        if p is None or i % every:
            kps.append(np.full((133, 3), np.nan, np.float32))
            boxes.append([np.nan] * 4)
        else:
            kps.append(wb(fr, p.box))
            boxes.append(list(p.box))

    if picks is None:
        _, size = single_person(frames, pose, on_frame=on_frame)
    else:
        size = None
        for fr, p in zip(frames, picks):
            size = (fr.shape[1], fr.shape[0])
            on_frame(fr, p)
    return {"kp": np.stack(kps) if kps else np.zeros((0, 133, 3), np.float32),
            "box": np.array(boxes, np.float32).reshape(-1, 4), "size": size, "ir": np.array(irs, bool),
            "calibrated": not getattr(wb, "raw", True),
            "seconds": round(time.time() - t0, 1)}


def on_timebase(kpd: dict, n: int, shift: int = 0):
    """Keypoints of one camera on the primary's time base: (n, 133, 3) with 0..1 confidence, box heights (n,)."""
    kp, box = kpd["kp"].astype(float), kpd["box"].astype(float)
    out = np.full((n, 133, 3), np.nan)
    bh = np.full(n, np.nan)
    for i in range(n):
        j = i + shift
        if 0 <= j < len(kp):
            out[i], bh[i] = kp[j], box[j, 3] - box[j, 1]
    if not kpd.get("calibrated", False):         # keypoints saved before WholeBody calibrated its scores (raw x-l)
        out[..., 2] = rig.conf_of(out[..., 2])
    return out, bh


def fit_views(cams: dict, fps: float, n: int, init_xy=None, stature0: float = 1.70, iters=(300, 900),
              verbose: bool = False) -> tuple[dict, list]:
    """cams: {name: (kpd from extract, Camera, shift)} (first = primary). -> (rig.fit result, views)."""
    views = []
    for name, (kpd, cam, shift) in cams.items():
        kp, bh = on_timebase(kpd, n, shift)
        views.append({"name": name, "cam": cam, "size": tuple(int(v) for v in kpd["size"]), "kp": kp, "box_h": bh})
    res = rig.fit(views, n, 1.0 / fps, init_xy=init_xy, stature0=stature0, iters=iters, verbose=verbose,
                  parts=PARTS or None)
    return res, views


def _ratio(tr: dict, fps: float) -> float:
    """rig frames per replay frame (the rig may run at a higher rate than the replay; both start at the clip start)."""
    rf = float(tr.get("fps") or fps)
    return fps / rf if rf > 0 else 1.0


def init_from_replay(tr: dict, n: int, fps: float | None = None) -> np.ndarray:
    xy = np.full((n, 2), np.nan)
    f0 = int(tr.get("frame0", 0))
    r = _ratio(tr, fps) if fps else 1.0
    for i, m in enumerate(tr.get("mean") or []):
        j = int(round((f0 + i) * r))
        if 0 <= j < n:
            xy[j] = m
    return xy


def fit_for_replay(tr: dict, cams: dict, fps: float, n: int | None = None, stature0: float | None = None,
                   iters=(300, 900)):
    """Fit the rig for a replay (initialised from its path) -> (rig.fit result, views) or (None, None)."""
    first = next(iter(cams.values()))[0]
    n = n or len(first["kp"])
    if n < 3:
        return None, None
    s0 = stature0 or (tr.get("multicam") or {}).get("stature_m") or (tr.get("metrics") or {}).get("head_height_assumed_m") \
        or 1.70
    t0 = time.time()
    res, views = fit_views(cams, fps, n, init_xy=init_from_replay(tr, n, fps), stature0=float(s0), iters=iters)
    res["fit_s"] = round(time.time() - t0, 1)
    if not res["seen"].any():
        return None, None
    return res, views


def rig_ground(res: dict, n_track: int, track_fps: float, rig_fps: float, sigma: float = 0.05) -> list:
    """Per frame of the replay's track (n_track frames at track_fps): (pelvis ground xy, 2x2 cov) from the rig where it
    is fitted, else None. For trajectory.estimate(ground_override=...): the rig then drives the path, so the path,
    stick figure and body mesh agree (the rig uses every camera, constant bone lengths and joint limits)."""
    out = [None] * n_track
    r = rig_fps / track_fps if track_fps > 0 else 1.0
    M = res["markers"]
    for j in range(n_track):
        k = int(round(j * r))
        if 0 <= k < len(M) and res["seen"][k]:
            xy = (M[k, rig.L_HIP, :2] + M[k, rig.R_HIP, :2]) / 2
            if np.isfinite(xy).all():
                out[j] = (xy.astype(float), np.eye(2) * sigma * sigma)
    return out


def merge_ground(primary: list, fallback: list | None) -> list:
    """Element-wise: primary where not None, else fallback."""
    if fallback is None:
        return primary
    return [p if p is not None else (fallback[i] if i < len(fallback) else None) for i, p in enumerate(primary)]


def attach_rig(tr: dict, cams: dict, fps: float, n: int | None = None, stature0: float | None = None,
               iters=(300, 900), replace_skeleton: bool = True, res: dict | None = None,
               views: list | None = None) -> dict | None:
    """Fit the rig to the cameras' keypoints and store it in the replay: tr["rig"] (segments, stature, per-frame
    markers 0-22 + joint angles on the replay's t, gait, per-camera error) and, by default, the replay's stick figure
    (tr["skeleton"], COCO-17) from the rig. Returns the rig block, or None when no frame has a usable view."""
    first = next(iter(cams.values()))[0]
    n = n or len(first["kp"])
    t0 = time.time()
    if res is None:
        res, views = fit_for_replay(tr, cams, fps, n=n, stature0=stature0, iters=iters)
    if res is None:
        return None
    gait = rig.gait_metrics(res, 1.0 / fps)
    f0, m = int(tr.get("frame0", 0)), len(tr.get("t") or [])
    r = _ratio(tr, fps)
    idx = [int(round(k * r)) for k in range(f0, f0 + m)]   # rig frame shown at each replay frame
    ok = lambda i: 0 <= i < n and bool(res["seen"][i])  # noqa: E731
    A = res["angles_deg"]
    block = {"version": VERSION,
             "method": "squirrel_guard.rig: procedural skeleton, constant segment lengths, soft joint limits, IK fit "
                       f"to RTMW whole-body keypoints from {', '.join(cams)}",
             "cameras": list(cams), "fps": round(float(fps), 3), "stature_m": round(res["stature_m"], 3),
             "segments_m": {k: round(v, 3) for k, v in res["segments_m"].items()},
             "markers_format": "COCO-WholeBody 0-22 (body 17 + feet 6), metres, world frame",
             "markers": [[[round(float(v), 3) for v in p] for p in res["markers"][i]] if ok(i) else None for i in idx],
             "angles_deg": {f"{nm}{'_' + s if s else ''}": [round(float(A[i, k]), 1) if ok(i) else None for i in idx]
                            for k, (nm, s) in enumerate(rig.ANGLES)},
             "gait": gait,
             "error": {v["name"]: pv for v, pv in zip(views, res["per_view"])},
             "frames_fitted": int(sum(ok(i) for i in idx)),
             "seconds": round(res.get("fit_s", 0) + time.time() - t0, 1)}
    if res.get("parts"):
        block["parts"] = parts_block(res["parts"], idx, ok)
    tr["rig"] = block
    if replace_skeleton:
        tr["skeleton"] = [[[round(float(v), 3) for v in p] for p in res["markers"][i, :17]] if ok(i) else None
                          for i in idx]
    return block


def parts_block(P: dict, idx: list, ok) -> dict:
    """rig.fit(..., parts=...)["parts"] on the replay's frames: head yaw/pitch/roll + jaw where the FACE was usable,
    hand frames where that HAND was usable (None elsewhere: unusable parts are prior-only, never shown as measured).
    Orientation only; the face points never identify anyone."""
    r3 = lambda a: [round(float(x), 3) for x in np.asarray(a).ravel()]  # noqa: E731
    fu, hu = P["face_usable"], P["hand_usable"]
    use = lambda m, i: ok(i) and i < len(m) and bool(m[i])            # noqa: E731
    return {"parts": P.get("parts"), "usable_rate": P.get("usable_rate"),
            "head_ypr_deg": [[round(float(v), 1) for v in P["head_ypr_deg"][i]] if use(fu, i) else None for i in idx],
            "jaw_deg": [round(float(P["jaw_deg"][i]), 1) if use(fu, i) else None for i in idx],
            "head_R": [r3(P["head_R"][i]) if use(fu, i) else None for i in idx],
            "hand_R": {s: [r3(P["hand_R"][s][i]) if use(hu[s], i) else None for i in idx] for s in ("L", "R")},
            "note": "head from the 68 face points + body; hands from 21 points each; None = not measured that frame"}
