#!/usr/bin/env python
"""Independent head pose from MediaPipe face mesh, per frame, in a calibrated camera's frame (for checking the rig).

  ~/.cache/squirrel-guard-test/facefit-venv/bin/python scripts/mp_headpose.py OUT.npz --clip CLIP.mp4 \
      --kp KP.npz --name testcam --calib CALIB.json [--fps 8] [--max-s 32]
  (--replay REPLAY.json --name NAME reads the calibration from the replay's camera / cameras_extra instead.)

Frames are decoded exactly as scripts/rigfit_keypoints.py does (kinematics.decode, same fps), so frame i here is
frame i of NAME_kp in KP.npz. The face window comes from that frame's confident RTMW face points. MediaPipe's legacy
face mesh (478 points, Apache-2.0, bundled with mediapipe <= 0.10.21: no download) runs on the crop
(facefit.landmarks_in_window); landmarks are undistorted into a virtual pinhole view centred on the face
(facefit.undistort_landmarks), a rigid reference shape (the visit's own mean face, frontalised at the nasion: x = eye
axis, y = nasion->chin, z = depth) is fitted per frame (Umeyama), and the rotation is turned back into the real
camera frame. Output per frame: R (3, 3) head frame in OpenCV camera coordinates with the rig's axes as columns
(subject's right, forward, up), ok flag, face size (IPD px). MediaPipe's own "up" is the nasion->chin line, so the
rig and this estimate differ by a constant rotation (template definitions): compare after fitting that constant.
Face geometry here is used for head orientation only: nothing is stored about the face itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import facefit, kinematics  # noqa: E402
from squirrel_guard import facegeom as fg  # noqa: E402
from squirrel_guard.geometry import Camera  # noqa: E402

# rig head axes (subject's right, forward, up) in facegeom's frontal basis (eye axis R->L, nasion->chin, depth away)
M_RIG = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])


def cam_from_replay(path: str, name: str) -> Camera:
    tr = json.loads(Path(path).read_text())
    for c in [tr["camera"]] + list(tr.get("cameras_extra") or []):
        if c.get("name") == name:
            return Camera(c["K"], c.get("D"), c["R"], c["C"], c["size"])
    raise SystemExit(f"camera {name} not in {path}")


def view_rotation(P: np.ndarray, cam: Camera, frame_size) -> np.ndarray:
    """The face-centred virtual view's axes in the real camera frame (same construction as undistort_landmarks)."""
    import cv2
    sx, sy = cam.size[0] / frame_size[0], cam.size[1] / frame_size[1]
    n = cv2.undistortPoints((P[:, :2] * (sx, sy)).reshape(-1, 1, 2), cam.K, cam.D).reshape(-1, 2)
    v = np.c_[n, np.ones(len(n))]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    c = v[:facefit.N_SURF].mean(0)
    c /= np.linalg.norm(c)
    x = np.cross([0.0, 1.0, 0.0], c)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(c, x), c], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--kp", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--calib")
    ap.add_argument("--replay")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--max-s", type=float, default=32.0)
    ap.add_argument("--min-conf", type=float, default=0.3)
    a = ap.parse_args()
    cam = Camera.load(a.calib) if a.calib else cam_from_replay(a.replay, a.name)
    z = np.load(a.kp)
    kp = z[f"{a.name}_kp"].astype(float)
    lm = facefit.SolutionLandmarker(0.3)
    frames, _ = kinematics.decode(Path(a.clip).read_bytes(), a.fps, a.max_s)
    meshes, Rv, ipd = {}, {}, {}
    for i, fr in enumerate(frames):
        if i >= len(kp):
            break
        f = kp[i, 23:91]
        good = np.isfinite(f[:, 0]) & (np.nan_to_num(f[:, 2]) >= a.min_conf)
        if good.sum() < 20:
            continue
        lo, hi = f[good, :2].min(0), f[good, :2].max(0)
        c, side = (lo + hi) / 2, 1.5 * float(max(hi - lo))
        if side < 16:
            continue
        o = facefit.landmarks_in_window(fr, (c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2), lm,
                                        cam=a.name, idx=i)
        if o is None:
            continue
        size = (fr.shape[1], fr.shape[0])
        meshes[i] = facefit.undistort_landmarks(o.P, cam, size)[:facefit.N_SURF]
        Rv[i] = view_rotation(o.P, cam, size)
        ipd[i] = o.ipd_px
    n = len(kp)
    R = np.full((n, 3, 3), np.nan)
    ok = np.zeros(n, bool)
    ipd_a = np.full(n, np.nan)
    if meshes:
        # reference = the visit's mean face in the frontal basis (2 rounds of align + average)
        ref = None
        for _ in range(3):
            Qs = []
            for i, P in meshes.items():
                Q, _ = fg.frontalise(P)
                Q = Q[:facefit.N_SURF] * (1, 1, -1)                 # back to depth-away (camera-like) handedness
                if ref is not None:
                    s, Ra, t = facefit.umeyama(Q, ref)
                    Q = s * Q @ Ra.T + t
                Qs.append(Q)
            ref = np.median(np.stack(Qs), 0)
            ref -= ref.mean(0)
        for i, P in meshes.items():
            s, Ra, t = facefit.umeyama(ref, P - P.mean(0))
            R[i] = Rv[i] @ Ra @ M_RIG
            ok[i] = True
            ipd_a[i] = ipd[i]
    np.savez_compressed(a.out, R=R, ok=ok, ipd_px=ipd_a, name=a.name)
    print(json.dumps({"camera": a.name, "frames": n, "faces": int(ok.sum()),
                      "ipd_px_median": round(float(np.nanmedian(ipd_a)), 1) if ok.any() else None}))


if __name__ == "__main__":
    main()
