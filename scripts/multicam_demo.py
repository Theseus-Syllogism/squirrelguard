#!/usr/bin/env python
"""3D-replay demo from 2+ jointly calibrated cameras watching the same walk (scripts/multicam_rig.py calibrations).

  .venv/bin/python scripts/multicam_demo.py --cam v4=V4.mp4,calib/v4.json --cam frontyard=FY.mp4,calib/frontyard.json
      [--out deploy/recorder/replay/demo] [--name multicam] [--fps 8] [--max-s 32] [--max-offset 3] [--splat S.splat]

The first --cam is the PRIMARY: its frames drive the replay's cut-outs, projected background and timeline. Each
clip is tracked on its own (kinematics.track). Then:
  * time offset: clips cut from different recorders start a little apart. For every candidate shift of each
    secondary clip (in frames, up to --max-offset s) the 17 body joints seen confidently by both cameras are cast
    as rays; the shift with the smallest median ray-to-ray gap wins (rays of the same joint at the same instant meet).
  * triangulation: at every primary frame with a matching secondary pose, each joint is the midpoint of the
    closest approach of its rays (pairs with a gap over --max-gap m are dropped). This needs no floor contact, so it
    works where furniture hides the feet, and the ground position (mean of the hip and shoulder joints) and
    stature are MEASURED, not assumed.
  * single-camera frames: the person is placed by the top of the head on the plane Z = measured stature
    (trajectory head plane), from the primary camera.
  * the trajectory then runs through the usual Kalman/RTS + GP pipeline (trajectory.estimate with ground_override);
    the skeleton is the triangulated one where available and the primary camera's flat-card one elsewhere.
The output is the normal replay JSON plus "cameras_extra" (every secondary camera, drawn as a frustum in the viewer)
and "multicam" (offsets, triangulated frame count, ray gaps, stature).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import fusion, gaze, kinematics, replay_assets, rigpipe, trajectory  # noqa: E402
from squirrel_guard.fusion import closest  # noqa: E402  (moved to squirrel_guard.fusion)
from squirrel_guard.rigpipe import single_person  # noqa: E402,F401  (moved; kept importable from here)
from squirrel_guard.geometry import Camera  # noqa: E402
from squirrel_guard.pose import PoseEstimator  # noqa: E402

CONF = 0.4
TORSO = [5, 6, 11, 12]          # shoulders + hips (COCO)
HEAD = [0, 1, 2, 3, 4]          # nose, eyes, ears


def joint_rays(cam, p, size):
    ok = p.kp[:, 2] >= CONF
    return ok, cam.rays(p.kp[:, :2], size)


def card_skeleton(cam, p, size, xy):
    """trajectory.skeleton3d's flat cut-out, but standing at a GIVEN ground point (the fused position) instead of
    this camera's own foot pixel, which furniture often hides."""
    G = np.array([xy[0], xy[1], 0.0])
    nrm = np.array([cam.C[0] - xy[0], cam.C[1] - xy[1], 0.0])
    if np.linalg.norm(nrm) < 1e-6:
        return None
    nrm /= np.linalg.norm(nrm)
    d = cam.rays(p.kp[:, :2], size)
    out = []
    for j in range(17):
        den = float(nrm @ d[j])
        s = float(nrm @ (G - cam.C)) / den if abs(den) > 1e-6 else -1
        if p.kp[j, 2] < 0.3 or s <= 0:
            out.append(None)
            continue
        X = cam.C + s * d[j]
        out.append([round(float(X[0]), 3), round(float(X[1]), 3), round(max(0.0, float(X[2])), 3)])
    return out


def _kp17(track) -> np.ndarray:
    return np.stack([p.kp if p is not None else np.full((17, 3), np.nan) for p in track]).astype(float)


def best_offset(cA, tA, sA, cB, tB, sB, max_shift):
    """Shift k (frames) such that tB[i + k] matches tA[i]; minimises the median joint ray gap (fusion.gap_curve)."""
    rA = fusion.view_rays(cA, _kp17(tA), sA, CONF, joints=list(range(17)))
    rB = fusion.view_rays(cB, _kp17(tB), sB, CONF, joints=list(range(17)))
    bs = fusion.best_shift(fusion.gap_curve(cA, rA, cB, rB, -max_shift, max_shift))
    return None if bs is None else (int(round(bs[0])), bs[1], bs[2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", action="append", required=True, help="NAME=CLIP,CALIB.json (first = primary)")
    ap.add_argument("--out", default=str(ROOT / "deploy/recorder/replay/demo"))
    ap.add_argument("--name", default="multicam")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--max-s", type=float, default=32.0)
    ap.add_argument("--max-offset", type=float, default=3.0, help="largest clip start difference searched (s)")
    ap.add_argument("--max-gap", type=float, default=0.25, help="drop joint ray pairs further apart than this (m)")
    ap.add_argument("--event", action="append", default=[], help="NAME=FRIGATE_EVENT_ID of the stored visit per "
                    "camera; the primary's becomes \"event\" so the replay pane shows its related-sighting analysis")
    ap.add_argument("--splat", help="a .splat in the same world frame (multicam_rig --splat-work + ply_to_splat)")
    ap.add_argument("--splat-note", default="")
    ap.add_argument("--rig", action="store_true", help="also fit the IK rig (RTMW whole-body keypoints of every camera, "
                    "squirrel_guard.rig) and use it for the stick figure; adds \"rig\" (angles, segments, gait)")
    ap.add_argument("--rig-model", default=str(ROOT / "models/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122/end2end.onnx"))
    ap.add_argument("--pose-model", default=str(ROOT / "models/yolov8s-pose.onnx"))
    ap.add_argument("--seg-model", default=str(ROOT / "models/yolov8s-seg-320.onnx"))
    a = ap.parse_args()

    pose = PoseEstimator(a.pose_model)
    cams = []
    t0 = time.perf_counter()
    for i, spec in enumerate(a.cam):
        name, rest = spec.split("=", 1)
        clip, calib = rest.split(",")
        cam = Camera.load(calib)
        if cam is None:
            sys.exit(f"bad calibration {calib}")
        # one time grid for every camera (kinematics.decode would give 25/3 = 8.33 vs 15/2 = 7.5 fps)
        frames, fps = fusion.grid_frames(Path(clip).read_bytes(), a.fps, a.max_s)
        coll = replay_assets.Collector() if i == 0 else None
        trk, size = single_person(frames, pose, on_frame=coll.on_frame if coll else None)
        print(f"{name}: {len(trk)} frames @ {fps:.2f} fps, {sum(p is not None for p in trk)} with the person")
        cams.append({"name": name, "clip": clip, "cam": cam, "trk": trk, "size": size, "fps": fps, "coll": coll})
    t_track = time.perf_counter() - t0
    P = cams[0]
    fps, n = P["fps"], len(P["trk"])

    # time offsets + triangulation (primary with each secondary; joints averaged over secondaries)
    joints = np.full((n, 17, 3), np.nan)
    jw = np.zeros((n, 17))
    gaps_all, offsets = [], {}
    for S in cams[1:]:
        bo = best_offset(P["cam"], P["trk"], P["size"], S["cam"], S["trk"], S["size"], int(round(a.max_offset * fps)))
        if bo is None:
            print(f"{S['name']}: too few common frames to synchronise; skipped")
            continue
        k, med, cnt = bo
        offsets[S["name"]] = {"shift_frames": k, "shift_s": round(k / fps, 3), "median_ray_gap_m": round(med, 3),
                              "frames_compared": cnt}
        print(f"{S['name']}: offset {k:+d} frames ({k / fps:+.2f} s), median ray gap {med * 100:.1f} cm over {cnt} frames")
        for i, pa in enumerate(P["trk"]):
            j = i + k
            if pa is None or not (0 <= j < len(S["trk"])) or S["trk"][j] is None:
                continue
            oa, ra = joint_rays(P["cam"], pa, P["size"])
            ob, rb = joint_rays(S["cam"], S["trk"][j], S["size"])
            m = np.nonzero(oa & ob)[0]
            if len(m) < 3:
                continue
            X, g = closest(P["cam"].C, ra[m], S["cam"].C, rb[m])
            for jj, x, gg in zip(m, X, g):
                if np.isfinite(gg) and gg <= a.max_gap and -0.2 < x[2] < 2.6:
                    w = 1.0 / (0.02 + gg)
                    prev = np.nan_to_num(joints[i, jj]) * jw[i, jj]
                    jw[i, jj] += w
                    joints[i, jj] = (prev + w * x) / jw[i, jj]
                    gaps_all.append(gg)

    names = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
             "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
    jz = {nm: round(float(np.nanmedian(joints[:, k, 2])), 3) for k, nm in enumerate(names)
          if np.isfinite(joints[:, k, 2]).sum() >= 3}
    sw = np.linalg.norm(joints[:, 5] - joints[:, 6], axis=1)
    rng_m = np.linalg.norm(np.nanmean(joints[:, TORSO], 1) - P["cam"].C, axis=1)
    print("triangulated joint heights (median, m):", jz)
    print(f"shoulder width {np.nanmedian(sw):.2f} m (adult ~0.36-0.42); range from primary {np.nanmedian(rng_m):.2f} m")
    # measured stature: top of head ~ 0.11 m above the eye/ear level
    head_z = [np.nanmean(joints[i, HEAD, 2]) for i in range(n) if np.isfinite(joints[i, HEAD, 2]).sum() >= 2]
    stature = float(np.median(head_z) + 0.11) if len(head_z) >= 3 else 1.75
    stature_measured = len(head_z) >= 3

    override, n_tri = [None] * n, 0
    for i in range(n):
        tor = joints[i, TORSO]
        ok = np.isfinite(tor[:, 0])
        if ok.sum() >= 2:
            xy = tor[ok, :2].mean(0)
            sig = 0.08 + 0.5 * float(np.median(np.array(gaps_all))) if gaps_all else 0.12
            override[i] = (xy, np.eye(2) * sig * sig)
            n_tri += 1
    # frames seen by one camera only: head top on the plane Z = stature (feet are often hidden indoors), from the
    # primary camera, else from a secondary at its synchronised frame
    n_head = 0
    for i in range(n):
        if override[i] is not None:
            continue
        cands = [(P, i)] + [(S, i + offsets[S["name"]]["shift_frames"]) for S in cams[1:] if S["name"] in offsets]
        for C, j in cands:
            p = C["trk"][j] if 0 <= j < len(C["trk"]) else None
            hp = trajectory.head_point(p, C["size"]) if p is not None else None
            m = trajectory.measure(C["cam"], hp[0], C["size"], 0.7 * hp[1], z=stature) if hp else None
            if m is not None and np.isfinite(m[0]).all():
                override[i] = m
                n_head += 1
                break
    print(f"head-plane placed {n_head} more frames")
    print(f"triangulated {n_tri}/{n} primary frames; stature {'measured' if stature_measured else 'ASSUMED'} "
          f"{stature:.2f} m; joint ray gap median {np.median(gaps_all) * 100 if gaps_all else float('nan'):.1f} cm")

    cam, size, trk = P["cam"], P["size"], P["trk"]
    gait = kinematics.gait(trk, fps, cam, size)
    tr = trajectory.estimate(trk, fps, size, cam, gait=gait, heights_px=True, head_height=stature,
                             ground_override=override)
    if tr is None:
        sys.exit("no trajectory")
    sk = [None] * n
    for i in range(n):
        if np.isfinite(joints[i, TORSO, 0]).sum() >= 2:
            sk[i] = [[round(float(v), 3) for v in x] if np.isfinite(x[0]) else None for x in joints[i]]
        elif override[i] is not None and trk[i] is not None:
            sk[i] = card_skeleton(cam, trk[i], size, override[i][0])
    tr["skeleton"] = sk[tr["frame0"]:tr["frame0"] + len(tr["t"])]
    gz = gaze.estimate(trk, tr, cam, size, stature)
    if gz:
        tr["gaze"] = gz
    tr["metrics"]["frames_triangulated"] = n_tri
    tr["metrics"]["frames_head_plane_multicam"] = n_head
    tr["metrics"]["head_height_assumed_m"] = None if stature_measured else stature
    tr["metrics"]["stature_measured_m"] = round(stature, 3) if stature_measured else None
    tr["multicam"] = {"primary": P["name"], "offsets": offsets, "frames_triangulated": n_tri,
                      "joint_ray_gap_median_m": round(float(np.median(gaps_all)), 3) if gaps_all else None,
                      "stature_m": round(stature, 3), "stature_measured": stature_measured,
                      "joint_height_median_m": jz, "shoulder_width_median_m": round(float(np.nanmedian(sw)), 3)}
    tr["cameras_extra"] = [{"name": S["name"], "C": S["cam"].C.tolist(), "R": S["cam"].R.tolist(),
                            "K": S["cam"].K.tolist(), "D": S["cam"].D.tolist(), "size": list(S["cam"].size)}
                           for S in cams[1:]]
    tr["camera"]["name"] = P["name"]
    ev = dict(e.split("=", 1) for e in a.event)
    if P["name"] in ev:
        tr["event"] = ev[P["name"]]
    tr["events_by_camera"] = ev
    tr["source"] = (f"real clips from {len(cams)} cameras ({', '.join(c['name'] for c in cams)}), calibrated together "
                    f"by scripts/multicam_rig.py; positions triangulated where two cameras see the person, else head "
                    f"plane at the measured stature")
    tr["calibration"] = {"method": "multicam-vggt", **{k: v for k, v in cam.meta.items()
                                                       if k in ("date", "note", "pitch_deg", "camera_height_m", "hfov_deg")}}
    print("metrics:", json.dumps(tr["metrics"]))

    if a.rig:
        from squirrel_guard.wholebody import WholeBody
        wb = WholeBody(a.rig_model)
        rc = {}
        for C in cams:
            if C is not P and C["name"] not in offsets:
                continue
            fr2, _ = fusion.grid_frames(Path(C["clip"]).read_bytes(), a.fps, a.max_s)
            kpd = rigpipe.extract(fr2, pose, wb)
            rc[C["name"]] = (kpd, C["cam"], 0 if C is P else offsets[C["name"]]["shift_frames"])
            print(f"rig keypoints {C['name']}: {len(kpd['kp'])} frames, {kpd['seconds']} s")
        res, views = rigpipe.fit_for_replay(tr, rc, fps, n=n)
        blk = None
        if res is not None:                    # the rig drives the path where fitted, triangulation elsewhere
            keep = {k: tr[k] for k in tr if k not in ("t", "mean", "cov", "velocity", "observed", "speed", "speed_std",
                                                      "footprints", "prediction", "metrics", "head_z", "frame0", "fps",
                                                      "version", "camera", "skeleton", "gaze")}
            tr2 = trajectory.estimate(trk, fps, size, cam, gait=gait, heights_px=True, head_height=stature,
                                      ground_override=rigpipe.merge_ground(rigpipe.rig_ground(res, n, fps, fps), override))
            if tr2:
                tr2["camera"]["name"] = tr["camera"].get("name")
                tr2["metrics"].update({k: v for k, v in tr["metrics"].items() if k not in tr2["metrics"]})
                tr = tr2
                tr.update(keep)
                gz = gaze.estimate(trk, tr, cam, size, stature)
                if gz:
                    tr["gaze"] = gz
            blk = rigpipe.attach_rig(tr, rc, fps, n=n, res=res, views=views)
        if blk:
            print(f"rig: stature {blk['stature_m']} m, {blk['frames_fitted']} frames, error "
                  + ", ".join(f"{k} {v['median_err_px']:.0f}px" for k, v in blk["error"].items() if v["median_err_px"])
                  + f"; gait {json.dumps(blk['gait'])}")
    masker = replay_assets.PersonMasker(a.seg_model, threads=4)
    replay, atlas, bg = replay_assets.build(P["coll"], cam, tr, masker, fps)
    out = Path(a.out)
    replay_assets.attach(tr, replay, atlas, bg, out, a.name)
    if a.splat:
        dst = out / f"{a.name}_splat.splat"
        shutil.copy(a.splat, dst)
        tr["assets"]["splat"] = dst.name
        tr["splat_note"] = a.splat_note
    (out / f"{a.name}.json").write_text(json.dumps(tr, separators=(",", ":")))
    print(f"wrote {out / (a.name + '.json')} (track {t_track:.0f} s)")


if __name__ == "__main__":
    main()
