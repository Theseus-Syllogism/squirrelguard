#!/usr/bin/env python
"""Hands + face in the rig on a real multi-camera visit: usability rates and head-pose agreement with MediaPipe.

  .venv/bin/python scripts/rig_handface_eval.py KP.npz REPLAY.json [--calib NAME=CALIB.json ...] [--use a,b]
      [--mp NAME=MP.npz ...] [--stature0 1.70] [--report OUT.json] [--save FIT.npz]

KP.npz from scripts/rigfit_keypoints.py (133 points); REPLAY.json gives the primary camera, the secondary time
offsets, the path that initialises the pelvis and, when --calib is not given, the calibrations (camera /
cameras_extra). MP.npz from scripts/mp_headpose.py (same decode, so frame j of camera NAME).

The rig is fitted twice on the same data: body only (head orientation from nose/eyes/ears, as before) and with
parts=("face", "hands"). For each camera with a MediaPipe file, both rigs' head frames are expressed in that camera
and compared with MediaPipe's: a constant rotation (template definitions differ: MediaPipe's "up" is nasion->chin)
is fitted on the compared frames and reported as "bias"; "residual" is what remains frame by frame (geodesic
degrees) plus per-axis camera-relative yaw/pitch/roll differences. Hand usability counts frames where the part
passed rig.PART_GATE (confidence, size in px, wrist consistency, residual check).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import rig, rigpipe  # noqa: E402
from squirrel_guard.geometry import Camera  # noqa: E402


def cams_from_replay(tr: dict) -> dict:
    out = {}
    for c in [tr["camera"]] + list(tr.get("cameras_extra") or []):
        if c.get("name"):
            out[c["name"]] = Camera(c["K"], c.get("D"), c["R"], c["C"], c["size"])
    return out


def head_R_body_only(res: dict) -> np.ndarray:
    """World head frames of a body-only fit (rebuilt from its angles)."""
    import torch
    T = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    H = T(res["stature_m"])
    seg = {k: T(v) for k, v in res["segments_m"].items()}
    _, fr = rig._chain(T(res["root"]), T(np.radians(res["yaw_deg"])), T(np.radians(res["angles_deg"])), seg, H)
    return fr["head"][0].numpy()


def polar(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ S @ Vt


def geo(A, B):
    c = (np.einsum("...ij,...ij->...", A, B) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def cam_ypr(Rc: np.ndarray) -> np.ndarray:
    """Head frames in OpenCV camera coordinates -> camera-relative yaw (+ = face turned to image right), pitch (+ = up),
    roll (+ = clockwise in the image), degrees."""
    fwd, up = Rc[..., :, 1], Rc[..., :, 2]
    yaw = np.degrees(np.arctan2(fwd[..., 0], -fwd[..., 2]))
    pitch = np.degrees(np.arcsin(np.clip(-fwd[..., 1], -1, 1)))
    roll = np.degrees(np.arctan2(up[..., 0], -up[..., 1]))
    return np.stack([yaw, pitch, roll], -1)


def compare(Rrig_c: np.ndarray, Rmp: np.ndarray, m: np.ndarray) -> dict | None:
    if m.sum() < 3:
        return {"frames": int(m.sum())}
    A, B = Rrig_c[m], Rmp[m]
    Q = polar(np.einsum("nji,njk->ik", B, A))          # Rrig ~ Rmp @ Q
    Bq = B @ Q
    res = geo(A, Bq)
    ya, yb = cam_ypr(A), cam_ypr(Bq)
    d = (ya - yb + 180) % 360 - 180
    out = {"frames": int(m.sum()), "bias_deg": round(float(geo(Q, np.eye(3))), 1),
           "residual_deg_median": round(float(np.median(res)), 1), "residual_deg_p90": round(float(np.percentile(res, 90)), 1),
           "no_alignment_deg_median": round(float(np.median(geo(A, B))), 1)}
    for k, nm in enumerate(("yaw", "pitch", "roll")):
        out[f"{nm}_absdiff_median"] = round(float(np.median(np.abs(d[:, k]))), 1)
        out[f"{nm}_mp_range_p5_p95"] = [round(float(v), 1) for v in np.percentile(yb[:, k], [5, 95])]
        if np.std(yb[:, k]) > 1.0 and np.std(ya[:, k]) > 1e-6:
            out[f"{nm}_corr"] = round(float(np.corrcoef(ya[:, k], yb[:, k])[0, 1]), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kp")
    ap.add_argument("replay")
    ap.add_argument("--calib", action="append", default=[])
    ap.add_argument("--use")
    ap.add_argument("--mp", action="append", default=[])
    ap.add_argument("--stature0", type=float)
    ap.add_argument("--iters", default="300,900")
    ap.add_argument("--report")
    ap.add_argument("--save")
    a = ap.parse_args()
    z = np.load(a.kp)
    tr = json.loads(Path(a.replay).read_text())
    cams = cams_from_replay(tr)
    cams.update({k: Camera.load(v) for k, v in (s.split("=", 1) for s in a.calib)})
    primary = (tr.get("multicam") or {}).get("primary") or tr["camera"].get("name") or "v4"
    offs = {k: v["shift_frames"] for k, v in (tr.get("multicam") or {}).get("offsets", {}).items()}
    offs[primary] = 0
    use = a.use.split(",") if a.use else [c for c in cams if f"{c}_kp" in z.files]
    use = [primary] + [c for c in use if c != primary]
    n = len(z[f"{primary}_kp"])
    fps = float(z[f"{primary}_fps"])
    kpds = {}
    for c in use:
        kpds[c] = ({"kp": z[f"{c}_kp"], "box": z[f"{c}_box"], "size": tuple(int(v) for v in z[f"{c}_size"]),
                    "calibrated": bool(z[f"{c}_calibrated"]) if f"{c}_calibrated" in z.files else False},
                   cams[c], offs.get(c, 0))
    s0 = a.stature0 or (tr.get("multicam") or {}).get("stature_m") or 1.70
    init = rigpipe.init_from_replay(tr, n, fps)
    iters = tuple(int(v) for v in a.iters.split(","))
    t0 = time.time()
    res0, views = rigpipe.fit_views(kpds, fps, n, init_xy=init, stature0=float(s0), iters=iters)
    t_body = time.time() - t0
    t0 = time.time()
    res1 = rig.fit(views, n, 1.0 / fps, init_xy=init, stature0=float(s0), iters=iters, verbose=False,
                   parts=("face", "hands"))
    t_parts = time.time() - t0
    P = res1["parts"]
    seen = res1["seen"]
    rep = {"primary": primary, "use": use, "frames": n, "fps": round(fps, 2), "frames_body_seen": int(seen.sum()),
           "fit_s": {"body_only": round(t_body, 1), "with_parts": round(t_parts, 1)},
           "stature_m": {"body_only": round(res0["stature_m"], 3), "with_parts": round(res1["stature_m"], 3)},
           "body_markers_moved_cm_median": round(float(np.nanmedian(np.linalg.norm(
               res1["markers"][seen][:, 5:] - res0["markers"][seen][:, 5:], axis=-1))) * 100, 2),
           "usable_rate_of_seen_frames": P["usable_rate"], "face_scale": round(P["face_scale"], 3),
           "hand_scale": round(P["hand_scale"], 3), "per_camera": {}}
    for v, pv, pv0 in zip(views, P["per_view"], res1["per_view"]):
        kp = v["kp"]
        present = np.isfinite(kp[:, 0, 0])
        d = {"frames_with_person": int(present.sum()), "body_err_box": pv0["median_err_box"]}
        for name, g in pv.items():
            d[name] = {**g, "usable_rate_of_person_frames": round(g["usable"] / max(1, int(present.sum())), 3)}
        # how big are the hands where RTMW is confident about them (px, the extent of confident points)
        for name, idx in (("L", rig.LHAND_IDX), ("R", rig.RHAND_IDX), ("face", rig.FACE_IDX)):
            c = np.nan_to_num(kp[:, idx, 2])
            ext = [float(np.ptp(kp[i, idx][c[i] >= 0.5, :2], 0).max()) for i in range(n) if (c[i] >= 0.5).sum() >= 5]
            d.setdefault(name, {})["confident_extent_px_median"] = round(float(np.median(ext)), 1) if ext else None
            d[name]["frames_conf_pass"] = int(((c >= 0.5).sum(1) >= (34 if name == "face" else 14)).sum())
        rep["per_camera"][v["name"]] = d
    # head pose vs MediaPipe
    Rw1 = P["head_R"]
    Rw0 = head_R_body_only(res0)
    rep["head_vs_mediapipe"] = {}
    for spec in a.mp:
        name, path = spec.split("=", 1)
        mp = np.load(path)
        cam = cams[name]
        sh = offs.get(name, 0)
        Rmp = np.full((n, 3, 3), np.nan)
        okm = np.zeros(n, bool)
        for i in range(n):
            j = i + sh
            if 0 <= j < len(mp["ok"]) and mp["ok"][j]:
                Rmp[i], okm[i] = mp["R"][j], True
        R0c = np.einsum("ji,njk->nik", cam.R, Rw0)                 # camera = R^T world
        R1c = np.einsum("ji,njk->nik", cam.R, Rw1)
        base = okm & seen
        fu = P["face_usable"]
        rep["head_vs_mediapipe"][name] = {
            "mediapipe_faces": int(okm.sum()),
            "mediapipe_ipd_px_median": round(float(np.nanmedian(mp["ipd_px"])), 1) if okm.any() else None,
            "body_only_all": compare(R0c, Rmp, base),
            "with_face_all": compare(R1c, Rmp, base),
            "body_only_on_face_usable": compare(R0c, Rmp, base & fu),
            "with_face_on_face_usable": compare(R1c, Rmp, base & fu),
            "with_face_on_face_not_usable": compare(R1c, Rmp, base & ~fu)}
    print(json.dumps(rep, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))
    if a.save:
        np.savez_compressed(a.save, markers=res1["markers"], seen=seen, yaw_deg=res1["yaw_deg"], root=res1["root"],
                            markers133=P["markers133"], head_R=P["head_R"], head_R_body_only=Rw0,
                            face_usable=P["face_usable"], hand_usable_L=P["hand_usable"]["L"],
                            hand_usable_R=P["hand_usable"]["R"], hand_R_L=P["hand_R"]["L"], hand_R_R=P["hand_R"]["R"],
                            angles_x_deg=P["angles_x_deg"], angles_deg=res1["angles_deg"], fps=fps)


if __name__ == "__main__":
    main()
