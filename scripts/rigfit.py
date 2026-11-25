#!/usr/bin/env python
"""Fit the procedural IK rig (squirrel_guard.rig) to a multi-camera visit and write it into its replay.

  .venv/bin/python scripts/rigfit.py KP.npz REPLAY.json --calib v4=v4.json --calib frontyard=frontyard.json
      [--use v4,frontyard] [--heldout frontyard] [--write] [--report OUT.json]

KP.npz from scripts/rigfit_keypoints.py; REPLAY.json from scripts/multicam_demo.py (primary camera = its
"camera"; secondary time offsets = multicam.offsets; the smoothed path initialises the pelvis). --use picks the
cameras the fit may see; --heldout projects the fitted rig into a camera it did NOT see and reports the error
there (an honest accuracy check), next to the same check for the replay's existing skeleton. --write stores
"rig" (segment lengths, stature, angles, gait, per-view error) in REPLAY.json and replaces its stick figure
("skeleton", COCO-17) with the rig's markers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import rig, rigpipe  # noqa: E402,F401
from squirrel_guard.geometry import Camera  # noqa: E402


def heldout_error(markers, cam, size, kp, box_h, conf_min=0.5):
    """Median reprojection error (fraction of box height, and px) of 3D markers (n, k, 3, NaN allowed) vs a view's
    keypoints (n, >=k, 3 with 0..1 confidence)."""
    import torch
    k = markers.shape[1]
    X = torch.tensor(np.nan_to_num(markers), dtype=torch.float64)
    uv, depth = rig.project(X, cam, size)
    uv = uv.numpy()
    ok = np.isfinite(markers).all(-1) & np.isfinite(kp[:, :k, 0]) & (np.nan_to_num(kp[:, :k, 2]) >= conf_min) \
        & (depth.numpy() > 0)
    if not ok.any():
        return None
    e = np.linalg.norm(uv - kp[:, :k, :2], axis=-1)
    bh = np.nan_to_num(box_h, nan=1.0)[:, None].repeat(k, 1)
    return {"median_err_box": round(float(np.median((e / bh)[ok])), 4), "median_err_px": round(float(np.median(e[ok])), 1),
            "p90_err_box": round(float(np.percentile((e / bh)[ok], 90)), 4), "points": int(ok.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kp"); ap.add_argument("replay")
    ap.add_argument("--calib", action="append", required=True, help="NAME=CALIB.json")
    ap.add_argument("--use", help="comma-separated cameras the fit sees (default: all)")
    ap.add_argument("--heldout", help="camera to evaluate on without fitting to it")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--iters", default="300,900")
    ap.add_argument("--offset", action="append", default=[], help="NAME=FRAMES time shift of a secondary camera "
                    "(default: the replay's multicam.offsets)")
    a = ap.parse_args()
    z = np.load(a.kp)
    tr = json.loads(Path(a.replay).read_text())
    cams = {k: Camera.load(v) for k, v in (s.split("=", 1) for s in a.calib)}
    primary = (tr.get("multicam", {}).get("primary") or tr["camera"].get("name")
               or (a.use.split(",")[0] if a.use else None))     # single-camera replays name no camera
    offs = {k: v["shift_frames"] for k, v in tr.get("multicam", {}).get("offsets", {}).items()}
    offs.update({k: int(v) for k, v in (o.split("=", 1) for o in a.offset)})
    offs[primary] = 0
    n = len(z[f"{primary}_kp"])
    fps = float(z[f"{primary}_fps"])
    dt = 1.0 / fps
    use = a.use.split(",") if a.use else [c for c in cams if f"{c}_kp" in z.files]

    def on_primary_timebase(name):
        kp, box = z[f"{name}_kp"].astype(float), z[f"{name}_box"].astype(float)
        k = offs[name]
        out_kp = np.full((n, 133, 3), np.nan)
        out_bh = np.full(n, np.nan)
        for i in range(n):
            j = i + k
            if 0 <= j < len(kp):
                out_kp[i], out_bh[i] = kp[j], box[j, 3] - box[j, 1]
        if not (f"{name}_calibrated" in z.files and bool(z[f"{name}_calibrated"])):
            out_kp[..., 2] = rig.conf_of(out_kp[..., 2])          # legacy files: raw x-l peaks
        return out_kp, out_bh, tuple(int(v) for v in z[f"{name}_size"])

    views = []
    for name in use:
        kp, bh, size = on_primary_timebase(name)
        views.append({"name": name, "cam": cams[name], "size": size, "kp": kp, "box_h": bh})
    init = np.full((n, 2), np.nan)
    f0 = int(tr["frame0"])
    for i, m in enumerate(tr["mean"]):
        if 0 <= f0 + i < n:
            init[f0 + i] = m
    stature0 = (tr.get("multicam") or {}).get("stature_m") or 1.70
    res = rig.fit(views, n, dt, init_xy=init, stature0=float(stature0),
                  iters=tuple(int(v) for v in a.iters.split(",")))
    gait = rig.gait_metrics(res, dt)
    report = {"use": use, "stature_m": round(res["stature_m"], 3),
              "segments_m": {k: round(v, 3) for k, v in res["segments_m"].items()},
              "train_error": {v["name"]: pv for v, pv in zip(views, res["per_view"])}, "gait": gait,
              "frames_seen": int(res["seen"].sum()), "frames": n}
    # bone-length consistency of plain per-frame triangulation (what the rig replaces), when 2+ views are used
    if a.heldout:
        kp, bh, size = on_primary_timebase(a.heldout)
        M = res["markers"].copy()
        M[~res["seen"]] = np.nan
        report["heldout"] = {"camera": a.heldout, "rig": heldout_error(M, cams[a.heldout], size, kp, bh)}
        # the replay's existing stick figure (COCO-17 per replay frame) on the same frames
        S = np.full((n, 17, 3), np.nan)
        for i, s in enumerate(tr.get("skeleton") or []):
            if s and 0 <= f0 + i < n:
                S[f0 + i] = [[np.nan] * 3 if j is None else j for j in s]
        S[~res["seen"]] = np.nan
        report["heldout"]["replay_skeleton"] = heldout_error(S, cams[a.heldout], size, kp, bh)
        R17 = M[:, :17]
        both = np.isfinite(S).all(-1) & np.isfinite(R17).all(-1)
        report["heldout"]["rig_same_joints_as_replay"] = heldout_error(np.where(both[..., None], R17, np.nan),
                                                                       cams[a.heldout], size, kp, bh)
    print(json.dumps(report, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))
    if a.write:
        from squirrel_guard import rigpipe
        cams_kp = {}
        for name in use:
            cams_kp[name] = ({"kp": z[f"{name}_kp"], "box": z[f"{name}_box"], "size": tuple(z[f"{name}_size"]),
                              "calibrated": bool(z[f"{name}_calibrated"]) if f"{name}_calibrated" in z.files else False},
                             cams[name], offs[name])
        blk = rigpipe.attach_rig(tr, cams_kp, fps, n=n, stature0=float(stature0),
                                 iters=tuple(int(v) for v in a.iters.split(",")))
        Path(a.replay).write_text(json.dumps(tr, separators=(",", ":")))
        print(f"wrote rig into {a.replay}: stature {blk['stature_m']} m, {blk['frames_fitted']} frames")

if __name__ == "__main__":
    main()
