#!/usr/bin/env python
"""End-to-end 3D-replay demo from a static-camera clip of ONE walking person of known height.

  .venv/bin/python scripts/replay_demo.py CLIP.mp4 [--out deploy/recorder/replay/demo] [--name demo] [--height 1.75]
        [--fps 8] [--max-s 40] [--hfov 55] [--tilt 25] [--calib camera.json] [--frames DIR]

Pipeline: kinematics.decode + track (replay_assets.Collector as on_frame) -> geometry.fit_walk on
kinematics.stance_points (known walker height) -> kinematics.gait -> trajectory.estimate + skeleton3d ->
replay_assets.build -> replay_assets.attach -> OUT/NAME.json + NAME_replay_{atlas.webp,bg.jpg,ground.json}.
Open the viewer with ?data=demo/demo.json.

Calibration notes. fit_walk only searches downward pitch (0..89 deg). Stock clips are often shot from eye level
or below, looking slightly UP, so the observations are first re-projected into a virtual camera rotated --tilt deg
further down (exact for a pinhole: pixel -> ray -> rotate -> pixel), fit_walk runs unchanged on those, and the
rotation is composed back. Heights do not depend on yaw, so the result is exact up to a world yaw. Pass --tilt 0
for a camera that already looks down (a real Wyze mount). A walker moving along the optical axis cannot separate
focal length from distance (every ground distance scales with f), so the focal length is fixed by --hfov (deg);
the metric scale of the demo then follows from that assumption.
--frames DIR also writes the analysed frames listed in --dump-t (seconds) as PNGs, for alignment checks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import gaze, kinematics, replay_assets, rigpipe, trajectory  # noqa: E402
from squirrel_guard.geometry import Camera, fit_walk, nominal_K, rot_from_angles  # noqa: E402
from squirrel_guard.pose import PoseEstimator  # noqa: E402


def _rx(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)  # same convention as geometry.rot_from_angles: positive = forward tips down
    return np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])


def fit_walk_tilted(obs, size, height_m: float, hfov: float, tilt: float) -> tuple[Camera, dict]:
    """geometry.fit_walk for cameras that may look UP: fit in a virtual camera tilted `tilt` deg further down."""
    W, H = size
    K = nominal_K(W, H, hfov)
    M = _rx(tilt)                                       # real_from_virtual (camera frames)
    Ki = np.linalg.inv(K)

    def to_virtual(p):
        r = M.T @ (Ki @ np.array([p[0], p[1], 1.0]))
        q = K @ (r / r[2])
        return (float(q[0]), float(q[1]))

    vobs = [(to_virtual(f), to_virtual(h)) for f, h in obs]
    camv, stats = fit_walk(vobs, size, height_m, K=K, D=np.zeros(5), fit_focal=False)
    R = camv.R @ M.T                                    # world_from_real = world_from_virtual @ virtual_from_real
    cam = Camera(K, np.zeros(5), R, camv.C, size)
    fwd = R[:, 2]
    stats.update({"pitch_deg": round(math.degrees(math.asin(-fwd[2])), 2), "pitch_note": "positive = looking down",
                  "virtual_tilt_deg": tilt, "hfov_assumed_deg": hfov, "intrinsics": "nominal (assumed hfov)"})
    # same residuals in the real camera (sanity check: must equal the virtual fit's)
    z = cam.heights([f for f, _ in obs], [h for _, h in obs], size)
    z = z[np.isfinite(z) & (z > 0)]
    stats["residual_cm_median_abs_check"] = round(float(np.median(np.abs(z - height_m))) * 100, 1)
    return cam, stats


def pitch_from_standing(trk, size, height_m: float, cam_h: float, hfov: float, k1: float) -> tuple[Camera, dict]:
    """Camera at a KNOWN height (e.g. on a desk) whose feet-visible frames are few: try pitches from 60 deg up to
    60 deg down and keep the one that makes the visible person height_m tall (median over frames that show both
    feet and head). Yaw is free, roll is assumed level."""
    K = nominal_K(*size, hfov)
    D = np.array([k1, 0.1 * abs(k1), 0, 0, 0])
    pairs = [(fp[0], ((p.box[0] + p.box[2]) / 2, p.box[1])) for p in trk if p is not None
             and (fp := trajectory.foot_point(p, size)) and trajectory.head_point(p, size)]
    if len(pairs) < 3:
        sys.exit(f"pitch: only {len(pairs)} frames show feet and head; pass --pitch")
    best = None
    for pitch in np.arange(-60.0, 60.01, 0.25):
        cam = Camera(K, D, rot_from_angles(float(pitch), 0.0), [0.0, 0.0, cam_h], size)
        z = cam.heights([a for a, _ in pairs], [b for _, b in pairs], size)
        z = z[np.isfinite(z) & (z > 0)]
        if len(z) >= 3:
            err = abs(float(np.median(z)) - height_m)
            if best is None or err < best[0]:
                best = (err, float(pitch), cam)
    if best is None:
        sys.exit("pitch: no pitch gives finite heights")
    return best[2], {"method": "pitch_from_standing", "frames": len(pairs), "pitch_deg": best[1],
                     "pitch_note": "positive = looking down", "camera_height_m_assumed": cam_h,
                     "hfov_assumed_deg": hfov, "k1_assumed": k1, "height_residual_cm": round(best[0] * 100, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", default=str(ROOT / "deploy/recorder/replay/demo"))
    ap.add_argument("--name", default="demo")
    ap.add_argument("--height", type=float, default=1.75)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--max-s", type=float, default=40.0)
    ap.add_argument("--hfov", type=float, default=55.0)
    ap.add_argument("--tilt", type=float, default=25.0)
    ap.add_argument("--pose-model", default=str(ROOT / "models/yolov8s-pose.onnx"))
    ap.add_argument("--seg-model", default=str(ROOT / "models/yolov8s-seg-320.onnx"))
    ap.add_argument("--calib", help="use a saved calibration JSON instead of fitting")
    ap.add_argument("--cam-height", type=float, help="camera height (m) is KNOWN: solve only the pitch from frames "
                    "that show feet and head (desk-mounted camera looking up); implies --head-plane")
    ap.add_argument("--pitch", type=float, help="with --cam-height: use this pitch (deg, negative = looking up) "
                    "instead of solving it")
    ap.add_argument("--k1", type=float, default=0.0, help="radial distortion for --cam-height (Wyze v3 ~ -0.25)")
    ap.add_argument("--head-plane", action="store_true", help="when the feet are out of frame, place the person by "
                    "the top of the head on the plane Z = --height")
    ap.add_argument("--rig", action="store_true", help="also fit the IK rig (RTMW whole-body keypoints, "
                    "squirrel_guard.rig) to this camera and use it for the stick figure; adds \"rig\"")
    ap.add_argument("--rig-model", default=str(ROOT / "models/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122/end2end.onnx"))
    ap.add_argument("--frames", help="directory for the --dump-t frames (PNG, full resolution, as analysed)")
    ap.add_argument("--dump-t", default="", help="comma-separated times (s) of analysed frames to dump")
    a = ap.parse_args()

    pose = PoseEstimator(a.pose_model)
    frames, fps = kinematics.decode(Path(a.clip).read_bytes(), a.fps, a.max_s)
    coll = replay_assets.Collector()
    dump_at = {int(round(float(s) * fps)) for s in a.dump_t.split(",") if s.strip()}
    dumped = {}

    def on_frame(fr, p):
        k = coll.n
        coll.on_frame(fr, p)
        if a.frames and k in dump_at:
            dumped[k] = fr.copy()

    t0 = time.perf_counter()
    trk, size = kinematics.track(frames, pose, on_frame=on_frame)
    t_track = time.perf_counter() - t0
    print(f"track: {len(trk)} frames @ {fps:.3f} fps, {sum(p is not None for p in trk)} with the person, "
          f"{t_track:.1f} s (collector {coll.ms:.0f} ms of it: {len(coll.crops)} crops, {len(coll.bg)} bg frames)")

    if a.cam_height and a.pitch is not None:
        K, D = nominal_K(*size, a.hfov), np.array([a.k1, 0.1 * abs(a.k1), 0, 0, 0])
        cam = Camera(K, D, rot_from_angles(a.pitch, 0.0), [0.0, 0.0, a.cam_height], size)
        stats = {"method": "given", "pitch_deg": a.pitch, "pitch_note": "positive = looking down",
                 "camera_height_m_assumed": a.cam_height, "hfov_assumed_deg": a.hfov, "k1_assumed": a.k1}
        a.head_plane = True
    elif a.cam_height:
        cam, stats = pitch_from_standing(trk, size, a.height, a.cam_height, a.hfov, a.k1)
        a.head_plane = True
    elif a.calib:
        cam, stats = Camera.load(a.calib), {"method": "file", "file": a.calib}
    else:
        obs = [sp for p in trk if p is not None and (sp := kinematics.stance_points(p, size))]
        print(f"calibration: {len(obs)} stance observations")
        cam, stats = fit_walk_tilted(obs, size, a.height, a.hfov, a.tilt) if a.tilt else \
            fit_walk(obs, size, a.height, K=nominal_K(*size, a.hfov), D=np.zeros(5), fit_focal=False)
    print("camera:", json.dumps(stats))

    gait = kinematics.gait(trk, fps, cam, size)
    print("gait:", json.dumps(gait))
    t1 = time.perf_counter()
    hh = a.height if a.head_plane else None
    tr = trajectory.estimate(trk, fps, size, cam, gait=gait, heights_px=True, head_height=hh)
    if tr is None:
        sys.exit("no trajectory (too few ground points)")
    sk = trajectory.skeleton3d(trk, cam, size, head_height=hh)
    tr["skeleton"] = sk[tr["frame0"]:tr["frame0"] + len(tr["t"])]
    gz = gaze.estimate(trk, tr, cam, size, a.height)
    if gz:
        tr["gaze"] = gz
        print("gaze:", json.dumps(gz["summary"]))
    t_traj = time.perf_counter() - t1
    tr["source"] = (f"real camera clip {Path(a.clip).name}: camera {a.cam_height} m high (assumed), pitch solved so the "
                    f"walker is {a.height} m; head-plane positions where the feet are out of frame; hfov {a.hfov} deg assumed"
                    if a.cam_height else
                    f"demo: stock clip {Path(a.clip).name}, camera calibrated from the walker ({a.height} m, hfov {a.hfov} deg assumed)")
    tr["calibration"] = stats
    print("metrics:", json.dumps(tr["metrics"]))

    if a.rig:
        from squirrel_guard.wholebody import WholeBody
        fr2, _ = kinematics.decode(Path(a.clip).read_bytes(), a.fps, a.max_s)
        kpd = rigpipe.extract(fr2, pose, WholeBody(a.rig_model))
        rc = {"camera": (kpd, cam, 0)}
        res, views = rigpipe.fit_for_replay(tr, rc, fps, n=len(trk),
                                            stature0=a.height if a.head_plane or a.cam_height else None)
        blk = None
        if res is not None:                    # the rig drives the path: re-estimate it from the rig's pelvis
            keep = {k: tr[k] for k in ("source", "calibration") if k in tr}
            tr2 = trajectory.estimate(trk, fps, size, cam, gait=gait, heights_px=True, head_height=hh,
                                      ground_override=rigpipe.rig_ground(res, len(trk), fps, fps))
            if tr2:
                tr = tr2
                tr.update(keep)
                gz = gaze.estimate(trk, tr, cam, size, a.height)
                if gz:
                    tr["gaze"] = gz
            blk = rigpipe.attach_rig(tr, rc, fps, n=len(trk), res=res, views=views)
        if blk:
            print(f"rig: stature {blk['stature_m']} m, {blk['frames_fitted']} frames, error "
                  f"{blk['error']['camera']['median_err_px']} px, keypoints {kpd['seconds']} s + fit {blk['seconds']} s")
            print("rig gait:", json.dumps(blk["gait"]))
    t2 = time.perf_counter()
    masker = replay_assets.PersonMasker(a.seg_model, threads=4)   # like the profiler's GaitEmbedder
    t_load = time.perf_counter() - t2
    replay, atlas, bg = replay_assets.build(coll, cam, tr, masker, fps)
    out = Path(a.out)
    replay_assets.attach(tr, replay, atlas, bg, out, a.name)
    (out / f"{a.name}.json").write_text(json.dumps(tr, separators=(",", ":")))
    st = tr["replay"]["stats"]
    files = {f: (out / f).stat().st_size for f in [f"{a.name}.json", *tr["assets"].values()]}
    print(f"assets: {st['cutouts']} cut-outs ({st['mask_missed']} without a mask), "
          f"ground {replay['ground']['cells']} cells + backdrop {replay['ground']['backdrop_cells']}; "
          f"files " + ", ".join(f"{k} {v / 1024:.0f} KB" for k, v in files.items()) + f"; total {sum(files.values()) / 1024:.0f} KB")
    print(f"time: track+pose {t_track:.1f} s (collector {coll.ms:.0f} ms), trajectory+skeleton {t_traj:.2f} s, "
          f"seg model load {t_load:.2f} s, build {st['build_s']:.2f} s (segmentation {st['segmentation_ms']:.0f} ms, "
          f"background {st['background_ms']:.0f} ms)")
    if a.frames:
        d = Path(a.frames)
        d.mkdir(parents=True, exist_ok=True)
        cam.save(d / f"{a.name}_camera.json", **stats)
        for k, fr in sorted(dumped.items()):
            cv2.imwrite(str(d / f"{a.name}_frame_{k:04d}.png"), fr)
        print(f"dumped {len(dumped)} frames + camera to {d}")


if __name__ == "__main__":
    main()
