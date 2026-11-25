"""Where a person was facing, frame by frame, from the five COCO head keypoints (nose, eyes, ears) and the camera.

Analysis by synthesis: a template head (nose 8 cm in front of the ear line, eyes 3 cm either side of it, ears 7 cm
either side of the head centre; the same layout tests/synth_walk.py uses) is placed at the person's 3D head
position (from the trajectory: ground point + eye height), turned to every candidate yaw in 5-degree steps and
projected through the real camera (lens distortion included). The yaw whose projection best matches the observed
head keypoints wins, after a similarity fit (scale + shift) so the head's apparent size does not matter.
Visibility is evidence too: a keypoint the template says faces away from the camera but the detector saw with high
confidence (or the reverse) is penalised, which is what separates "facing the camera" from "facing away".

This is HEAD direction, not eye gaze: at 1080p from a wide-angle lens a face is a few dozen pixels, so eyes cannot
be read. Expect roughly +-20-30 degrees; the per-frame confidence comes from how sharply the fit prefers one yaw.
Output angles are world yaw in degrees (0 = world +X, 90 = world +Y, counter-clockwise from above, as the
trajectory's headings).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import Camera

# template head in the person's frame: (forward, left, up) metres relative to the head centre (ear midpoint)
TEMPLATE = np.array([[0.08, 0.00, 0.00],     # nose
                     [0.08, 0.03, 0.03],     # left eye
                     [0.08, -0.03, 0.03],    # right eye
                     [0.00, 0.07, 0.00],     # left ear
                     [0.00, -0.07, 0.00]])   # right ear
# outward directions of those points (what makes them visible): face points look forward, ears sideways
NORMALS = np.array([[1, 0, 0], [1, 0.3, 0], [1, -0.3, 0], [0, 1, 0], [0, -1, 0]], float)
YAWS = np.arange(-180, 180, 5.0)
KP_MIN = 0.3
EYE_DROP_M = 0.12                 # eyes/ears sit ~12 cm below the top of the head
HUMAN_FOV_DEG = 120               # binocular visual field (horizontal), drawn faint
FOCUS_DEG = 60                    # central field where attention usually is, drawn stronger


def _project(cam: Camera, P: np.ndarray, frame_size) -> np.ndarray:
    Rcw = np.asarray(cam.R, float).T
    rvec, _ = cv2.Rodrigues(Rcw)
    tvec = (-Rcw @ np.asarray(cam.C, float)).reshape(3, 1)
    px, _ = cv2.projectPoints(P.reshape(-1, 1, 3), rvec, tvec, cam.K, cam.D)
    px = px.reshape(-1, 2)
    px[:, 0] *= frame_size[0] / cam.size[0]
    px[:, 1] *= frame_size[1] / cam.size[1]
    return px


def head_yaw(kp: np.ndarray, head_xyz, cam: Camera, frame_size) -> tuple[float, float, float] | None:
    """(world yaw deg, confidence 0-1, uncertainty sd deg) for one frame, or None when the head is not readable.
    Confidence 1 = sd 15 deg (the floor: template and keypoint noise), 0 = sd 75 deg or more."""
    vis = np.clip(kp[:5, 2], 0, 1)
    # Facing away hides nose and eyes: that is still evidence (visibility term below), just vaguer. Only give up
    # when the whole head region is unconfident AND no ear is seen, i.e. the head is probably not in view.
    if (vis >= KP_MIN).sum() == 0 and vis.max() < 0.02:
        return None
    H = np.asarray(head_xyz, float)
    to_cam = np.asarray(cam.C, float) - H
    to_cam /= np.linalg.norm(to_cam) + 1e-9
    obs = kp[:5, :2]
    seen = vis >= KP_MIN
    costs = []
    for yd in YAWS:
        a = math.radians(yd)
        f = np.array([math.cos(a), math.sin(a), 0.0])
        l = np.array([-math.sin(a), math.cos(a), 0.0])
        u = np.array([0.0, 0.0, 1.0])
        M = np.c_[f, l, u]
        P = H + TEMPLATE @ M.T
        facing = (NORMALS @ M.T) @ to_cam                       # > 0: the point faces the camera
        px = _project(cam, P, frame_size)
        cost = 0.0
        if seen.sum() >= 2:                                      # similarity fit of the seen points, then residual
            A, B = px[seen], obs[seen]
            ca, cb = A.mean(0), B.mean(0)
            sa = np.linalg.norm(A - ca) + 1e-6
            sb = np.linalg.norm(B - cb) + 1e-6
            r = (A - ca) / sa - (B - cb) / sb
            cost += float((r ** 2).sum() * (vis[seen] / vis[seen].sum()).sum())
        # visibility agreement: a confidently seen point should face the camera; a facing point should be seen
        cost += 0.6 * float(np.sum(np.clip(-facing, 0, 1) * vis))
        cost += 0.3 * float(np.sum(np.clip(facing, 0, 1) * (1 - vis) * (facing > 0.5)))
        costs.append(cost)
    costs = np.asarray(costs)
    i = int(np.argmin(costs))
    ang = np.radians(YAWS)

    def circ(temp):
        w = np.exp(-(costs - costs[i]) / temp)
        w /= w.sum()
        cx, cy = (w * np.cos(ang)).sum(), (w * np.sin(ang)).sum()
        return math.degrees(math.atan2(cy, cx)), float(np.hypot(cx, cy))
    yaw, _ = circ(0.05)                 # direction: sharp posterior (best fit, lightly averaged)
    _, R = circ(0.25)                   # uncertainty: broad posterior (how many yaws fit almost as well)
    sd = max(15.0, math.degrees(math.sqrt(max(0.0, -2 * math.log(max(R, 1e-6))))))   # circular sd, floored at 15 deg
    return yaw, float(np.clip(1.0 - (sd - 15.0) / 60.0, 0.05, 1.0)), sd


def estimate(track: list, tr: dict, cam: Camera, size, height_m: float = 1.75) -> dict | None:
    """Per trajectory frame: smoothed head yaw (deg or None), confidence, and eye height; plus a summary
    (share of observed time facing the camera, longest look away from the walking direction)."""
    f0, n = int(tr["frame0"]), len(tr["t"])
    mean = np.asarray(tr["mean"], float)
    eye_z = height_m - EYE_DROP_M
    raw: list = [None] * n
    for k in range(n):
        i = f0 + k
        p = track[i] if i < len(track) else None
        if p is None or not tr["observed"][k]:
            continue
        r = head_yaw(np.asarray(p.kp, float), (mean[k, 0], mean[k, 1], eye_z), cam, size)
        if r is not None:
            raw[k] = r
    if sum(r is not None for r in raw) < 3:
        return None
    # circular smoothing over +-2 frames, confidence-weighted; frames without a reading stay None
    yaw: list = [None] * n
    conf: list = [None] * n
    sd: list = [None] * n
    for k in range(n):
        if raw[k] is None:
            continue
        vx = vy = ws = 0.0
        for j in range(max(0, k - 2), min(n, k + 3)):
            if raw[j] is None:
                continue
            wj = raw[j][1] * (1.0 if j == k else 0.5)
            vx += wj * math.cos(math.radians(raw[j][0]))
            vy += wj * math.sin(math.radians(raw[j][0]))
            ws += wj
        yaw[k] = round(math.degrees(math.atan2(vy, vx)), 1)
        conf[k] = round(raw[k][1], 2)
        sd[k] = round(raw[k][2], 0)
    # summary relative to the camera and to the walking direction
    vel = np.asarray(tr.get("velocity") or np.zeros((n, 2)), float)
    toward_cam = away_walk = 0
    worst = (0.0, None)
    good = [k for k in range(n) if yaw[k] is not None and (conf[k] or 0) >= 0.4]
    for k in good:
        dx, dy = cam.C[0] - mean[k, 0], cam.C[1] - mean[k, 1]
        d_cam = (yaw[k] - math.degrees(math.atan2(dy, dx)) + 180) % 360 - 180
        if abs(d_cam) <= 30:
            toward_cam += 1
        sp = float(np.hypot(*vel[k]))
        if sp > 0.4:
            d_walk = (yaw[k] - math.degrees(math.atan2(vel[k, 1], vel[k, 0])) + 180) % 360 - 180
            if abs(d_walk) > 60:
                away_walk += 1
                if abs(d_walk) > abs(worst[0]):
                    worst = (d_walk, tr["t"][k])
    return {"yaw": yaw, "conf": conf, "sd_deg": sd, "eye_z": round(eye_z, 2), "fov_deg": HUMAN_FOV_DEG, "focus_deg": FOCUS_DEG,
            "summary": {"frames_read": len(good),
                        "facing_camera_share": round(toward_cam / len(good), 2) if good else None,
                        "looking_off_path_share": round(away_walk / len(good), 2) if good else None,
                        "largest_off_path_deg": round(worst[0], 0) if worst[1] is not None else None,
                        "largest_off_path_t": worst[1]},
            "note": "head direction from nose/eyes/ears (not eye gaze); sd_deg per frame, never below 15 deg"}
