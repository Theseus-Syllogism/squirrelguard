"""Procedural skeleton rig fitted by inverse kinematics to 2D whole-body keypoints from one or more calibrated cameras.

Why: per-frame keypoints (and per-frame triangulation) give limbs that change length from frame to frame, knees
that bend backwards, and nothing at all where one camera loses a joint. A rig fixes what is fixed about a person
(bone lengths, joint ranges) and only lets the joint angles move, so every camera's evidence over the whole visit
constrains one consistent body.

Model (world frame of geometry.Camera: metres, Z up)
  - Kinematic tree: pelvis root (position + yaw/pitch/roll) -> spine (flex/lateral/twist) -> neck -> head (3 dof);
    neck -> shoulders -> upper arm (3 dof) -> elbow (flexion only, 0..150 deg) -> wrist;
    pelvis -> hips -> thigh (flex/abduction/twist) -> knee (flexion only, 0..150 deg) -> ankle (pitch/roll) ->
    heel, big toe, small toe.
  - Segment lengths: anthropometric ratios of stature (Drillis & Contini 1966, the standard biomechanics table),
    one free stature and one free log-multiplier per segment (left = right), all CONSTANT over the visit, with a
    weak prior that keeps them near human proportions. Head marker layout (eyes, ears, nose) is a fixed adult
    template scaled with stature.
  - Joint limits: soft quadratic penalties outside anatomical ranges (LIMITS, degrees).
Fit (torch, CPU is enough: ~30 parameters per frame)
  - Data term: for every camera and frame, the rig's marker points are projected through the calibrated camera
    (K, D incl. radial distortion, R, C) and compared with the keypoints; residuals are divided by the person's box
    height in that view (near and far cameras weigh alike), Huber-robust, weighted by keypoint confidence.
  - Temporal smoothness on joint positions (second differences) and angles; floor: heels/toes not below Z = 0, and
    ground contact (the lowest foot point ~ on the floor in observed frames: walking/standing, not jumping or lying),
    which with a calibrated camera height removes the single-view size/distance ambiguity.
  - Stage 1 fits root pose + stature on torso/head markers with the limbs at rest; stage 2 fits everything.
Outputs per frame: rig marker points (COCO-WholeBody body + feet), joint angles (deg), per-view reprojection error;
per visit: segment lengths. gait_metrics() turns the angle curves + heel heights into gait parameters.
"""
from __future__ import annotations

import math

import numpy as np

# COCO-WholeBody indices used as rig markers
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
L_BIG, L_SMALL, L_HEEL, R_BIG, R_SMALL, R_HEEL = 17, 18, 19, 20, 21, 22
MARKERS = list(range(23))                     # body 17 + feet 6
TORSO_HEAD = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_HIP, R_HIP]

# segment length ratios of stature (Drillis & Contini), marker-to-marker (COCO markers sit near the joint centres)
RATIO = {"hip_half": 0.055, "spine": 0.29, "sh_half": 0.11, "neck": 0.095, "upper_arm": 0.186, "forearm": 0.146,
         "thigh": 0.245, "shank": 0.246, "ankle_h": 0.039, "heel": 0.04, "toe": 0.12}
SEGS = list(RATIO)
# head template (fraction of stature) in the head frame (x = subject's right, y = forward, z = up), origin = head centre
# Right-handed body frame: x = the subject's RIGHT, y = forward, z = up (so the subject's left is -x). An earlier
# version put the left at +x, a mirror image: the fit then matched left/right markers by turning the figure around
# (nose pointing away from the camera, knees bending forward) while positions still matched.
HEAD_T = {NOSE: (0.0, 0.062, -0.005), L_EYE: (-0.019, 0.05, 0.012), R_EYE: (0.019, 0.05, 0.012),
          L_EAR: (-0.043, 0.0, 0.0), R_EAR: (0.043, 0.0, 0.0)}
# soft joint limits, degrees: (low, high)
LIMITS = {"spine_flex": (-30, 60), "spine_lat": (-30, 30), "spine_twist": (-45, 45),
          "head_flex": (-50, 60), "head_lat": (-40, 40), "head_twist": (-80, 80),
          "sh_flex": (-60, 180), "sh_abd": (-30, 180), "sh_twist": (-90, 90), "elbow": (0, 150),
          "hip_flex": (-30, 120), "hip_abd": (-20, 45), "hip_twist": (-45, 45), "knee": (0, 150),
          "ankle_pitch": (-50, 30), "ankle_roll": (-25, 25)}
# per-frame angle parameters (name, side or None)
ANGLES = [("root_pitch", None), ("root_roll", None), ("spine_flex", None), ("spine_lat", None), ("spine_twist", None),
          ("head_flex", None), ("head_lat", None), ("head_twist", None)] + \
         [(n, s) for s in ("L", "R") for n in ("sh_flex", "sh_abd", "sh_twist", "elbow", "hip_flex", "hip_abd",
                                               "hip_twist", "knee", "ankle_pitch", "ankle_roll")]
ANG_IDX = {(n, s): i for i, (n, s) in enumerate(ANGLES)}


def conf_of(raw):
    """RTMW raw SimCC peak -> 0..1 weight (measured: ~0.5-1 = unsure/occluded, 3-6 = confident)."""
    return np.clip((np.asarray(raw, float) - 1.0) / 2.5, 0.0, 1.0)


def _rx(a):
    import torch
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([torch.stack([o, z, z], -1), torch.stack([z, c, -s], -1), torch.stack([z, s, c], -1)], -2)


def _ry(a):
    import torch
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)


def _rz(a):
    import torch
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def _mv(R, v):
    """Batched R (n,3,3) @ v (3,) or (n,3)."""
    import torch
    v = v if v.dim() == 2 else v.expand(R.shape[0], 3)
    return torch.einsum("nij,nj->ni", R, v)


def _chain(root, yaw, ang, seg, H):
    """Body kinematic chain -> (marker points (n, 23, 3), frames {"head": (Rh, head_c), "L"/"R": (Re, wrist)})."""
    import torch
    n = root.shape[0]
    A = lambda name, side=None: ang[:, ANG_IDX[(name, side)]]  # noqa: E731
    e = lambda x, y, z: torch.stack([torch.as_tensor(x, dtype=root.dtype).expand(n) if not torch.is_tensor(x) else x.expand(n),  # noqa: E731
                                     torch.as_tensor(y, dtype=root.dtype).expand(n) if not torch.is_tensor(y) else y.expand(n),
                                     torch.as_tensor(z, dtype=root.dtype).expand(n) if not torch.is_tensor(z) else z.expand(n)], -1)
    P = [None] * 23
    fr = {}
    Rr = _rz(yaw) @ _rx(A("root_pitch")) @ _ry(A("root_roll"))
    Rs = Rr @ _rx(A("spine_flex")) @ _ry(-A("spine_lat")) @ _rz(A("spine_twist"))
    neck = root + _mv(Rs, e(0.0, 0.0, seg["spine"]))
    Rh = Rs @ _rx(-A("head_flex")) @ _ry(-A("head_lat")) @ _rz(A("head_twist"))
    head_c = neck + _mv(Rh, e(0.0, 0.0, seg["neck"]))
    fr["head"] = (Rh, head_c)
    for k, (x, y, z) in HEAD_T.items():
        P[k] = head_c + _mv(Rh, e(x * H, y * H, z * H))
    for side, sg, SH, EL, WR, HIP, KN, AN, BIG, SMALL, HEEL in (
            ("L", -1.0, L_SH, L_EL, L_WR, L_HIP, L_KN, L_AN, L_BIG, L_SMALL, L_HEEL),
            ("R", 1.0, R_SH, R_EL, R_WR, R_HIP, R_KN, R_AN, R_BIG, R_SMALL, R_HEEL)):
        sh = neck + _mv(Rs, e(sg * seg["sh_half"], 0.0, 0.0))
        Ra = Rs @ _rx(A("sh_flex", side)) @ _ry(-sg * A("sh_abd", side)) @ _rz(sg * A("sh_twist", side))
        el = sh + _mv(Ra, e(0.0, 0.0, -seg["upper_arm"]))
        Re = Ra @ _rx(A("elbow", side))
        wr = el + _mv(Re, e(0.0, 0.0, -seg["forearm"]))
        fr[side] = (Re, wr)
        hip = root + _mv(Rr, e(sg * seg["hip_half"], 0.0, 0.0))
        Rt = Rr @ _rx(A("hip_flex", side)) @ _ry(-sg * A("hip_abd", side)) @ _rz(sg * A("hip_twist", side))
        kn = hip + _mv(Rt, e(0.0, 0.0, -seg["thigh"]))
        Rk = Rt @ _rx(-A("knee", side))
        an = kn + _mv(Rk, e(0.0, 0.0, -seg["shank"]))
        Rf = Rk @ _rx(A("ankle_pitch", side)) @ _ry(-sg * A("ankle_roll", side))
        heel = an + _mv(Rf, e(0.0, -seg["heel"], -seg["ankle_h"]))
        big = an + _mv(Rf, e(-sg * 0.01 * H, seg["toe"], -seg["ankle_h"]))
        small = an + _mv(Rf, e(sg * 0.025 * H, seg["toe"] * 0.8, -seg["ankle_h"]))
        P[SH], P[EL], P[WR], P[HIP], P[KN], P[AN] = sh, el, wr, hip, kn, an
        P[BIG], P[SMALL], P[HEEL] = big, small, heel
    return torch.stack(P, 1), fr


def forward(root, yaw, ang, seg, H):
    """Rig -> marker points (n, 23, 3). root (n,3) pelvis centre; yaw (n,); ang (n, len(ANGLES)) radians;
    seg dict of segment lengths (tensors, metres); H stature (tensor)."""
    return _chain(root, yaw, ang, seg, H)[0]


# ------------------------------------------------------------------------------------------- hands and face (optional)
# COCO-WholeBody 133: FACE 23-90 (iBUG-68 order: 0-16 jaw from the subject's RIGHT to left, 17-26 brows, 27-35 nose,
# 36-47 eyes (36-41 = subject's right eye), 48-67 mouth), LEFT_HAND 91-111, RIGHT_HAND 112-132 (21 points: 0 wrist,
# 1-4 thumb CMC..tip, 5-8 index MCP..tip, 9-12 middle, 13-16 ring, 17-20 little finger).
FACE0, LHAND0, RHAND0 = 23, 91, 112
FACE_IDX, LHAND_IDX, RHAND_IDX = list(range(23, 91)), list(range(91, 112)), list(range(112, 133))
# Mean adult face, 68 points, centimetres for a 1.75 m adult, (x = subject's right, z = up, y = forward) with the
# origin at the pupil midpoint. Hand-built from standard craniofacial anthropometry (biocular width 9 cm,
# intercanthal 3.2 cm, mouth 5 cm, nasion-menton 12.4 cm, bizygomatic ~13.5 cm, nose tip ~3 cm proud of the eyes);
# stored (x, z, y) as written, reordered below. A generic template: head angles carry a constant bias of a few degrees
# for any particular face (measured against MediaPipe in docs/rigfit.md), the frame-to-frame changes do not.
_FACE_XZY_CM = [
    (7.0, -1.5, -5.5), (6.9, -3.8, -5.0), (6.6, -6.0, -4.3), (6.1, -8.1, -3.4), (5.3, -9.9, -2.3), (4.2, -11.4, -1.0),
    (2.9, -12.5, 0.2), (1.5, -13.2, 1.0), (0.0, -13.4, 1.3), (-1.5, -13.2, 1.0), (-2.9, -12.5, 0.2), (-4.2, -11.4, -1.0),
    (-5.3, -9.9, -2.3), (-6.1, -8.1, -3.4), (-6.6, -6.0, -4.3), (-6.9, -3.8, -5.0), (-7.0, -1.5, -5.5),
    (5.0, 2.0, -1.2), (4.1, 2.8, -0.3), (3.0, 3.1, 0.3), (1.9, 3.0, 0.7), (0.9, 2.6, 0.9),
    (-0.9, 2.6, 0.9), (-1.9, 3.0, 0.7), (-3.0, 3.1, 0.3), (-4.1, 2.8, -0.3), (-5.0, 2.0, -1.2),
    (0.0, 1.0, 0.8), (0.0, -0.5, 1.7), (0.0, -1.8, 2.5), (0.0, -3.0, 3.2),
    (1.4, -3.6, 1.6), (0.75, -3.85, 2.0), (0.0, -4.0, 2.2), (-0.75, -3.85, 2.0), (-1.4, -3.6, 1.6),
    (4.5, 0.1, -0.9), (3.7, 0.55, -0.1), (2.6, 0.55, 0.0), (1.6, 0.0, -0.3), (2.6, -0.45, -0.1), (3.7, -0.45, -0.2),
    (-1.6, 0.0, -0.3), (-2.6, 0.55, 0.0), (-3.7, 0.55, -0.1), (-4.5, 0.1, -0.9), (-3.7, -0.45, -0.2), (-2.6, -0.45, -0.1),
    (2.5, -7.3, 0.6), (1.6, -6.6, 1.6), (0.6, -6.3, 2.1), (0.0, -6.4, 2.2), (-0.6, -6.3, 2.1), (-1.6, -6.6, 1.6),
    (-2.5, -7.3, 0.6), (-1.6, -8.0, 1.5), (-0.7, -8.3, 1.9), (0.0, -8.35, 2.0), (0.7, -8.3, 1.9), (1.6, -8.0, 1.5),
    (2.2, -7.3, 0.8), (0.7, -7.05, 1.8), (0.0, -7.05, 1.9), (-0.7, -7.05, 1.8), (-2.2, -7.3, 0.8), (-0.7, -7.4, 1.8),
    (0.0, -7.4, 1.9), (0.7, -7.4, 1.8)]
FACE_T = np.array([(x, y, z) for x, z, y in _FACE_XZY_CM]) / 100.0 / 1.75     # fraction of stature, head-face frame
FACE_ORIGIN = np.array([0.0, 0.05, 0.012])                                   # pupil midpoint in HEAD_T's frame
# jaw opening: rotation about the temporomandibular axis (x) through JAW_PIVOT; per-point weight (lower lip/chin 1)
JAW_PIVOT = np.array([0.0, -7.0, -2.5]) / 100.0 / 1.75
JAW_W = np.zeros(68)
JAW_W[2:15] = [0.2, 0.5, 0.8, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.2]
JAW_W[[48, 54, 60, 64]] = 0.5
JAW_W[[55, 56, 57, 58, 59, 65, 66, 67]] = 1.0
# Hand, RIGHT hand in its own frame (origin = wrist, fingers along -z, thumb toward +y, palm facing -x = medially with
# the arm hanging: the relaxed "palms in" pose), centimetres for a 1.75 m adult (hand length 0.108 stature, Drillis &
# Contini; palm ~0.52 of it). The left hand is the mirror image (x -> -x).
HAND_MCP_CM = {5: (0.0, 2.3, -9.0), 9: (0.0, 0.2, -9.5), 13: (0.0, -1.8, -8.9), 17: (0.0, -3.6, -8.0)}
HAND_PHAL_CM = {5: (4.0, 2.4, 2.0), 9: (4.4, 2.8, 2.2), 13: (4.0, 2.6, 2.1), 17: (3.2, 1.9, 1.8)}
HAND_FAN_DEG = {5: 6.0, 9: 0.0, 13: -5.0, 17: -10.0}                 # finger spread about x (toward +y = thumb side)
THUMB_CMC_CM = (-0.8, 2.4, -2.5)
THUMB_PHAL_CM = (4.5, 3.2, 2.8)
THUMB_DIR = np.array([-0.35, 0.55, -0.75]) / np.linalg.norm([-0.35, 0.55, -0.75])
FINGERS = ("thumb", "index", "middle", "ring", "little")
LIMITS_X = {"jaw": (-3, 35), "wr_flex": (-70, 80), "wr_dev": (-35, 25), "wr_pron": (-85, 85), "curl": (-10, 100)}
# extra per-frame angles (fitted only with fit(..., parts=...)): jaw, then per hand wrist (3) + finger curls (5)
ANGLES_X = [("jaw", None)] + [(n, s) for s in ("L", "R") for n in ("wr_flex", "wr_dev", "wr_pron")
                              + tuple(f"curl_{f}" for f in FINGERS)]
ANGX_IDX = {(n, s): i for i, (n, s) in enumerate(ANGLES_X)}


def _rot_axis(axis, a):
    """Rotation about a fixed unit axis (3,) by angles a (n,) -> (n, 3, 3) (Rodrigues)."""
    import torch
    k = torch.as_tensor(axis, dtype=a.dtype)
    K = torch.zeros((3, 3), dtype=a.dtype)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    s, c = torch.sin(a)[:, None, None], torch.cos(a)[:, None, None]
    return torch.eye(3, dtype=a.dtype) + s * K + (1 - c) * (K @ K)


def _hand_local(angx, side: str, scale):
    """Finger curls -> 21 hand points in the hand frame of `side` (n, 21, 3), metres (scale = metres per template cm)."""
    import torch
    n = angx.shape[0]
    dt = angx.dtype
    C = lambda f: angx[:, ANGX_IDX[(f"curl_{f}", side)]]  # noqa: E731
    pts = [torch.zeros((n, 3), dtype=dt)] * 21
    flex_axis = (0.0, 1.0, 0.0)                         # right hand: rotation about +y curls the fingers toward -x
    for f, base in zip(FINGERS[1:], (5, 9, 13, 17)):
        c = C(f)
        fan = math.radians(HAND_FAN_DEG[base])
        d0 = torch.tensor([0.0, math.sin(fan), -math.cos(fan)], dtype=dt)
        p = torch.tensor(HAND_MCP_CM[base], dtype=dt).expand(n, 3) * scale
        pts[base] = p
        tot = torch.zeros_like(c)
        for j, (ln, share) in enumerate(zip(HAND_PHAL_CM[base], (1.0, 1.0, 0.7))):
            tot = tot + share * c
            p = p + _mv(_rot_axis(flex_axis, tot), d0) * (ln * scale)
            pts[base + 1 + j] = p
    c = C("thumb")
    ax = np.cross(THUMB_DIR, [-1.0, 0.0, 0.0])
    ax /= np.linalg.norm(ax)
    p = torch.tensor(THUMB_CMC_CM, dtype=dt).expand(n, 3) * scale
    pts[1] = p
    d0 = torch.tensor(THUMB_DIR, dtype=dt)
    tot = torch.zeros_like(c)
    for j, (ln, share) in enumerate(zip(THUMB_PHAL_CM, (0.5, 1.0, 0.8))):
        tot = tot + share * c
        p = p + _mv(_rot_axis(ax, tot), d0) * (ln * scale)
        pts[2 + j] = p
    P = torch.stack(pts, 1)
    if side == "L":                                      # mirror image of the right-hand template
        P = P * torch.tensor([-1.0, 1.0, 1.0], dtype=dt)
    return P


def wrist_rot(angx, side: str):
    """Hand frame relative to the forearm frame (n, 3, 3): pronation about the forearm axis, then flexion (toward the
    palm), then radial deviation (toward the thumb)."""
    sg = -1.0 if side == "L" else 1.0
    a = lambda nm: angx[:, ANGX_IDX[(nm, side)]]  # noqa: E731
    return _rz(sg * a("wr_pron")) @ _ry(sg * a("wr_flex")) @ _rx(a("wr_dev"))


def forward_parts(root, yaw, ang, angx, seg, H, face_scale=None, hand_scale=None):
    """Rig with face and hands -> (points (n, 133, 3) in COCO-WholeBody order, frames dict: "head" (Rh, head_c),
    "L"/"R" hand frames (n, 3, 3) world (columns: subject's right, thumb/forward, up for the hanging hand), "jaw")."""
    import torch
    X23, fr = _chain(root, yaw, ang, seg, H)
    n = root.shape[0]
    dt = root.dtype
    fs = H * (face_scale if face_scale is not None else 1.0)
    Rh, head_c = fr["head"]
    F = torch.tensor(FACE_T, dtype=dt)
    piv = torch.tensor(JAW_PIVOT, dtype=dt)
    jw = torch.tensor(JAW_W, dtype=dt)
    jaw = angx[:, ANGX_IDX[("jaw", None)]]
    a = -jaw[:, None] * jw[None]                                  # (n, 68): opening = rotation about -x
    rel = F[None] - piv
    y = torch.cos(a) * rel[..., 1] - torch.sin(a) * rel[..., 2]
    z = torch.sin(a) * rel[..., 1] + torch.cos(a) * rel[..., 2]
    Fl = torch.stack([rel[..., 0].expand(n, 68), y, z], -1) + piv
    Fh = (Fl + torch.tensor(FACE_ORIGIN, dtype=dt)) * fs              # head frame, metres
    face = head_c[:, None] + torch.einsum("nab,nkb->nka", Rh, Fh)
    hands, hfr = {}, {}
    hsc = H / 1.75 / 100.0 * (hand_scale if hand_scale is not None else 1.0)
    for side in ("L", "R"):
        Re, wr = fr[side]
        Rw = Re @ wrist_rot(angx, side)
        loc = _hand_local(angx, side, hsc)
        hands[side] = wr[:, None] + torch.einsum("nab,nkb->nka", Rw, loc)
        hfr[side] = Rw
    X = torch.cat([X23, face, hands["L"], hands["R"]], 1)
    return X, {"head": fr["head"], "L": hfr["L"], "R": hfr["R"], "jaw": jaw}


def head_ypr(Rh) -> np.ndarray:
    """World head rotation(s) (..., 3, 3) (columns: subject's right, forward, up) -> (..., 3) degrees: yaw (heading of
    the face, same convention as the rig yaw: 0 = facing +y, counter-clockwise from above), pitch (+ = looking up),
    roll (+ = head tilted toward the subject's right shoulder)."""
    R = np.asarray(Rh, float)
    fwd, up, right = R[..., :, 1], R[..., :, 2], R[..., :, 0]
    yaw = np.degrees(np.arctan2(-fwd[..., 0], fwd[..., 1]))
    pitch = np.degrees(np.arcsin(np.clip(fwd[..., 2], -1, 1)))
    roll = np.degrees(np.arctan2(-right[..., 2], up[..., 2]))
    return np.stack([yaw, pitch, roll], -1)


def project(X, cam, size):
    """World points (..., 3) -> pixels (..., 2) in a frame of `size`, OpenCV model incl. k1, k2, p1, p2, k3."""
    import torch
    R = torch.as_tensor(cam.R, dtype=X.dtype)
    C = torch.as_tensor(cam.C, dtype=X.dtype)
    Xc = (X - C) @ R                                     # R is world_from_camera: camera = R^T (X - C)
    z = Xc[..., 2].clamp_min(1e-3)
    x, y = Xc[..., 0] / z, Xc[..., 1] / z
    k1, k2, p1, p2, k3 = (float(v) for v in (list(cam.D) + [0] * 5)[:5])
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    sx, sy = size[0] / cam.size[0], size[1] / cam.size[1]
    u = (cam.K[0, 0] * xd + cam.K[0, 2]) * sx
    v = (cam.K[1, 1] * yd + cam.K[1, 2]) * sy
    return torch.stack([u, v], -1), Xc[..., 2]


def _huber(e, d):
    import torch
    a = e.abs()
    return torch.where(a < d, 0.5 * a * a / d, a - 0.5 * d)


def fit(views: list[dict], n: int, dt: float, init_xy: np.ndarray | None = None, stature0: float = 1.70,
        iters: tuple[int, int] = (300, 900), w_smooth: float = 2.0, w_contact: float = 1.0,
        stature_sd: float | None = None, verbose: bool = True, parts=None, iters_parts: int = 400) -> dict:
    """views: [{"cam": Camera, "size": (W, H), "kp": (n, 23+, 3) pixels + 0..1 confidence (NaN = none),
    "box_h": (n,) person box height px}] already on ONE time base of n frames (shift secondaries before calling).
    init_xy: (n, 2) rough pelvis ground positions (e.g. the replay trajectory); NaN rows are interpolated.
    parts: optional subset of ("face", "hands"): with 133-point keypoints, a third stage fits head orientation + jaw
    to the 68 face points and wrist rotation + finger curls to the hand points (confidence- and size-gated, see
    PART_GATE) and adds res["parts"] (fit_parts docstring). Without parts the result is exactly as before."""
    import torch
    torch.manual_seed(0)
    dtp = torch.float64
    obs = []
    for v in views:
        kp = np.asarray(v["kp"], float)[:, :23]
        w = np.nan_to_num(kp[..., 2]) * np.isfinite(kp[..., 0])
        bh = np.nan_to_num(np.asarray(v["box_h"], float), nan=1.0).clip(1.0)
        obs.append({"cam": v["cam"], "size": v["size"], "uv": torch.tensor(np.nan_to_num(kp[..., :2]), dtype=dtp),
                    "w": torch.tensor(w, dtype=dtp), "bh": torch.tensor(bh, dtype=dtp)[:, None]})
    seen = np.zeros(n, bool)
    feet_seen = np.zeros(n, bool)
    legs_seen = {"L": np.zeros(n, bool), "R": np.zeros(n, bool)}
    for o in obs:
        W = o["w"].numpy()
        seen |= (W[:, TORSO_HEAD] > 0.3).sum(1) >= 3
        feet_seen |= (W[:, [L_AN, R_AN, L_HEEL, R_HEEL, L_BIG, R_BIG]] > 0.5).sum(1) >= 2
        legs_seen["L"] |= (W[:, [L_HIP, L_KN, L_AN]] > 0.5).all(1)
        legs_seen["R"] |= (W[:, [R_HIP, R_KN, R_AN]] > 0.5).all(1)
    if stature_sd is None:                      # one camera cannot separate size from distance well: hold the prior
        stature_sd = 0.15 if len(obs) > 1 else 0.06
    seen_t = torch.tensor(seen & feet_seen)       # contact only where the feet are actually observed
    # initial root: given xy (interpolated over gaps), pelvis height ~0.53 stature; yaw: facing the first camera
    if init_xy is None:
        init_xy = np.zeros((n, 2))
    xy = np.array(init_xy, float)
    bad = ~np.isfinite(xy).all(1)
    if bad.all():
        xy[:] = 0.0
    elif bad.any():
        idx = np.arange(n)
        for d in range(2):
            xy[bad, d] = np.interp(idx[bad], idx[~bad], xy[~bad, d])
    c0 = np.asarray(views[0]["cam"].C, float)
    to_cam = c0[:2] - xy
    yaw0 = np.arctan2(-to_cam[:, 0], to_cam[:, 1])            # body +y (forward) towards the camera
    # front/back from the primary view: subject's left shoulder on the image RIGHT = facing the camera
    kp0 = np.asarray(views[0]["kp"], float)
    away = np.isfinite(kp0[:, L_SH, 0]) & np.isfinite(kp0[:, R_SH, 0]) & (kp0[:, L_SH, 0] < kp0[:, R_SH, 0])
    yaw0[away] += math.pi
    root = torch.tensor(np.c_[xy, np.full(n, 0.53 * stature0)], dtype=dtp, requires_grad=True)
    yaw = torch.tensor(yaw0, dtype=dtp, requires_grad=True)
    ang = torch.zeros((n, len(ANGLES)), dtype=dtp)
    with torch.no_grad():                                     # slightly bent elbows/knees: away from the limit kink
        for s in ("L", "R"):
            ang[:, ANG_IDX[("elbow", s)]] = math.radians(10)
            ang[:, ANG_IDX[("knee", s)]] = math.radians(5)
    ang.requires_grad_(True)
    logH = torch.tensor(math.log(stature0), dtype=dtp, requires_grad=True)
    logm = torch.zeros(len(SEGS), dtype=dtp, requires_grad=True)
    lo = torch.tensor([math.radians(LIMITS[a][0]) if a in LIMITS else -math.pi for a, _ in ANGLES], dtype=dtp)
    hi = torch.tensor([math.radians(LIMITS[a][1]) if a in LIMITS else math.pi for a, _ in ANGLES], dtype=dtp)
    lo[ANG_IDX[("root_pitch", None)]], hi[ANG_IDX[("root_pitch", None)]] = math.radians(-20), math.radians(30)
    lo[ANG_IDX[("root_roll", None)]], hi[ANG_IDX[("root_roll", None)]] = math.radians(-20), math.radians(20)

    def segs():
        H = torch.exp(logH)
        return {s: RATIO[s] * H * torch.exp(logm[i]) for i, s in enumerate(SEGS)}, H

    def loss_fn(markers, full: bool):
        seg, H = segs()
        X = forward(root, yaw, ang, seg, H)
        L = 0.0
        for o in obs:
            uv, depth = project(X, o["cam"], o["size"])
            w = o["w"] if full else o["w"] * torch.tensor([1.0 if k in TORSO_HEAD else 0.0 for k in range(23)],
                                                          dtype=dtp)
            err = (uv - o["uv"]).norm(dim=-1) / o["bh"]
            L = L + (w * _huber(err, 0.02)).sum() / max(1.0, float(o["w"].sum())) * 100.0
            L = L + 10.0 * torch.relu(0.2 - depth).sum()       # stay in front of the cameras
        acc = (X[2:] - 2 * X[1:-1] + X[:-2]) / (dt * dt)
        L = L + w_smooth * 1e-3 * (acc ** 2).sum(-1).mean()
        aacc = (ang[2:] - 2 * ang[1:-1] + ang[:-2]) / (dt * dt)
        L = L + w_smooth * 1e-4 * (aacc ** 2).mean()
        yacc = (yaw[2:] - 2 * yaw[1:-1] + yaw[:-2]) / (dt * dt)
        L = L + w_smooth * 1e-4 * (yacc ** 2).mean()
        L = L + 50.0 * ((torch.relu(lo - ang) ** 2) + (torch.relu(ang - hi) ** 2)).sum(-1).mean()
        L = L + 20.0 * (logm ** 2).sum() / (0.08 ** 2) * 1e-3 + ((torch.exp(logH) - stature0) / stature_sd) ** 2 * 1e-2
        feet = X[:, [L_HEEL, R_HEEL, L_BIG, R_BIG, L_SMALL, R_SMALL], 2]
        L = L + 100.0 * (torch.relu(-feet) ** 2).sum(-1).mean()
        if w_contact > 0:
            # ground contact: walking/standing = at least one foot ON the floor (soft minimum of the foot heights ~ 0).
            # With a calibrated camera height this fixes the size/distance ambiguity of a single view.
            low = -0.02 * torch.logsumexp(-feet / 0.02, dim=1)
            if bool(seen_t.any()):
                L = L + w_contact * ((low[seen_t] / 0.03) ** 2).mean()
        return L, X

    stages = [(iters[0], [root, yaw, logH], False, 0.03), (iters[1], [root, yaw, ang, logH, logm], True, 0.01)]
    for it, ps, full, lr in stages:
        opt = torch.optim.Adam(ps, lr=lr)
        for k in range(it):
            opt.zero_grad()
            L, _ = loss_fn(None, full)
            L.backward()
            opt.step()
            if verbose and (k % 300 == 0 or k == it - 1):
                print(f"  rig fit {'full' if full else 'root'} step {k:4d} loss {L.item():.4f}")
    parts_res = None
    if parts:
        parts_res = _fit_parts(tuple(parts), views, obs, root, yaw, ang, segs, lambda: loss_fn(None, True)[0], seen,
                               dt, w_smooth, iters_parts, verbose)
    with torch.no_grad():
        seg, H = segs()
        X = forward(root, yaw, ang, seg, H)
        per_view = []
        for o in obs:
            uv, _ = project(X, o["cam"], o["size"])
            err = ((uv - o["uv"]).norm(dim=-1) / o["bh"]).numpy()
            w = o["w"].numpy()
            m = w > 0.5
            per_view.append({"median_err_box": float(np.median(err[m])) if m.any() else None,
                             "median_err_px": float(np.median((err * o["bh"].numpy())[m])) if m.any() else None,
                             "points": int(m.sum())})
    return {"markers": X.numpy(), "angles_deg": np.degrees(ang.detach().numpy()), "yaw_deg": np.degrees(yaw.detach().numpy()),
            "root": root.detach().numpy(), "segments_m": {s: float(v) for s, v in seg.items()}, "stature_m": float(H),
            "seen": seen, "feet_seen": feet_seen, "legs_seen": legs_seen, "per_view": per_view,
            **({"parts": parts_res} if parts_res is not None else {})}


# Hard gates for the face and hand terms, per view and frame (these cameras see hands at 20-80 px):
#   min_pts points with confidence >= conf; the part's expected size in that view (focal x metric size / depth, from
#   the stage-2 body) >= min_px; hands: the hand's wrist point within wrist_tol hand lengths of the body wrist (catches
#   swapped or stray hands); after the fit, a view-frame whose median residual exceeds max_res part sizes is dropped
#   and the stage refitted (the RTMW points disagree with any rigid face / plausible hand).
PART_GATE = {"face": {"min_pts": 34, "conf": 0.5, "min_px": 24.0, "max_res": 0.12, "size_m": 0.10},
             "hand": {"min_pts": 14, "conf": 0.5, "min_px": 24.0, "wrist_tol": 0.35, "max_res": 0.20, "size_m": 0.108}}
HEAD_COLS = [ANG_IDX[("head_flex", None)], ANG_IDX[("head_lat", None)], ANG_IDX[("head_twist", None)]]


def _fit_parts(parts, views, obs, root, yaw, ang, segs, body_loss, seen, dt, w_smooth, iters, verbose):
    """Stage 3 of fit(): head orientation (head_flex/lat/twist, now also from the 68 face points), jaw, wrists and
    finger curls; the rest of the body stays as stage 2 left it (the body data term still holds the head markers).
    Returns res["parts"]:
      markers133 (n,133,3) world; angles_x_deg (n, len(ANGLES_X)) (ANGX_IDX); head_R (n,3,3) world head frame
      (columns: subject's right, forward, up); head_ypr_deg (n,3) world yaw/pitch/roll (head_ypr); jaw_deg (n,);
      hand_R {"L","R"} (n,3,3) world hand frames; palm_normal / finger_dir {"L","R"} (n,3) unit vectors;
      face_usable, hand_usable {"L","R"} (n,) bool (some view passed the gate, after the residual check);
      usable_rate {"face","L","R"}: fraction of frames with the body seen in which the part was usable;
      per_view [{"face"/"L"/"R": {"gated_in", "usable", "median_err"}}] (err in part sizes), face_scale, hand_scale.
    Where a part is not usable, its angles come from priors + smoothness only (neutral, interpolated): use the masks."""
    import torch
    dtp = root.dtype
    n = root.shape[0]
    want_face, want_hands = "face" in parts, "hands" in parts
    groups = ([("face", FACE_IDX)] if want_face else []) + ([("L", LHAND_IDX), ("R", RHAND_IDX)] if want_hands else [])
    angx = torch.zeros((n, len(ANGLES_X)), dtype=dtp)
    with torch.no_grad():
        for s_ in ("L", "R"):
            for f in FINGERS:
                angx[:, ANGX_IDX[(f"curl_{f}", s_)]] = math.radians(20)
    log_fs = torch.zeros((), dtype=dtp, requires_grad=True)
    log_hs = torch.zeros((), dtype=dtp, requires_grad=True)
    lo = torch.tensor([math.radians(LIMITS_X["curl" if a.startswith("curl") else a][0]) for a, _ in ANGLES_X], dtype=dtp)
    hi = torch.tensor([math.radians(LIMITS_X["curl" if a.startswith("curl") else a][1]) for a, _ in ANGLES_X], dtype=dtp)
    hmask = torch.zeros(len(ANGLES), dtype=dtp)
    hmask[HEAD_COLS] = 1.0

    def fwd():
        seg, H = segs()
        return forward_parts(root, yaw, ang, angx, seg, H, torch.exp(log_fs), torch.exp(log_hs))

    # observations and gates (expected part size in px from the stage-2 body)
    with torch.no_grad():
        X0, _ = fwd()
        seg0, H0 = segs()
    pobs = []
    for v, o in zip(views, obs):
        kp = np.asarray(v["kp"], float)
        if kp.shape[1] < 133:
            pobs.append(None)
            continue
        cam, size = v["cam"], v["size"]
        fpx = float(cam.K[0, 0]) * size[0] / cam.size[0]
        _, depth = project(X0, cam, size)
        depth = depth.numpy()
        conf = np.nan_to_num(kp[..., 2]) * np.isfinite(kp[..., 0])
        rec = {"cam": cam, "size": size, "uv": torch.tensor(np.nan_to_num(kp[:, :, :2]), dtype=dtp), "g": {}}
        for name, idx in groups:
            gk = "face" if name == "face" else "hand"
            G = PART_GATE[gk]
            ref = NOSE if name == "face" else (L_WR if name == "L" else R_WR)
            spx = fpx * G["size_m"] * float(H0) / 1.75 / np.clip(depth[:, ref], 0.2, None)
            c = conf[:, idx]
            ok = ((c >= G["conf"]).sum(1) >= G["min_pts"]) & (spx >= G["min_px"]) & seen & (depth[:, ref] > 0.2)
            if name != "face":
                bw = L_WR if name == "L" else R_WR
                d = np.linalg.norm(kp[:, idx[0], :2] - kp[:, bw, :2], axis=1)
                ok &= np.nan_to_num(d / spx, nan=9.0) <= G["wrist_tol"]
            rec["g"][name] = {"idx": idx, "spx": torch.tensor(spx, dtype=dtp)[:, None], "ok0": ok.copy(), "ok": ok,
                              "c": torch.tensor(c * (c >= G["conf"]), dtype=dtp)}
        pobs.append(rec)

    def data_term(X, per_frame=False):
        L = 0.0
        pf = {}
        for rec in pobs:
            if rec is None:
                continue
            uv, _ = project(X, rec["cam"], rec["size"])
            for name, g in rec["g"].items():
                w = g["c"] * torch.tensor(g["ok"], dtype=dtp)[:, None]
                err = (uv[:, g["idx"]] - rec["uv"][:, g["idx"]]).norm(dim=-1) / g["spx"]
                hub = _huber(err, 0.05)
                if per_frame:
                    pf[(id(rec), name)] = (w * hub).sum(1)
                else:
                    L = L + (w * hub).sum() / max(1.0, float(w.sum())) * 100.0
        return pf if per_frame else L

    # coarse per-frame start for the wrists (palm vs back of the hand is a local-minimum trap)
    if want_hands:
        with torch.no_grad():
            grid = [(p, f_) for p in (-70, -35, 0, 35, 70) for f_ in (-40, 0, 40)]
            for side in ("L", "R"):
                best, best_e = None, None
                for p, f_ in grid:
                    angx[:, ANGX_IDX[("wr_pron", side)]] = math.radians(p)
                    angx[:, ANGX_IDX[("wr_flex", side)]] = math.radians(f_)
                    X, _ = fwd()
                    pf = data_term(X, per_frame=True)
                    e = sum((v for (k, nm), v in pf.items() if nm == side), torch.zeros(n, dtype=dtp))
                    if best is None:
                        best, best_e = torch.tensor([[p, f_]] * n, dtype=dtp), e
                    else:
                        m = e < best_e
                        best[m] = torch.tensor([p, f_], dtype=dtp)
                        best_e = torch.where(m, e, best_e)
                pr = best[:, 0].numpy()
                fl = best[:, 1].numpy()
                angx[:, ANGX_IDX[("wr_pron", side)]] = torch.tensor(np.radians(pr), dtype=dtp)
                angx[:, ANGX_IDX[("wr_flex", side)]] = torch.tensor(np.radians(fl), dtype=dtp)
    angx.requires_grad_(True)
    curl_cols = {s_: [ANGX_IDX[(f"curl_{f}", s_)] for f in FINGERS[1:]] for s_ in ("L", "R")}

    def run(k_iters):
        h = ang.register_hook(lambda g: g * hmask)
        opt = torch.optim.Adam([{"params": [angx], "lr": 0.02}, {"params": [ang], "lr": 0.005},
                                {"params": [log_fs, log_hs], "lr": 0.005}])
        for k in range(k_iters):
            opt.zero_grad()
            X, _ = fwd()
            L = body_loss() + data_term(X)
            L = L + 50.0 * ((torch.relu(lo - angx) ** 2) + (torch.relu(angx - hi) ** 2)).sum(-1).mean()
            aacc = (angx[2:] - 2 * angx[1:-1] + angx[:-2]) / (dt * dt)
            L = L + w_smooth * 1e-4 * (aacc ** 2).mean()
            L = L + 1e-2 * (angx ** 2).sum(-1).mean()
            for s_ in ("L", "R"):
                cc = angx[:, curl_cols[s_]]
                L = L + 0.5 * ((cc - cc.mean(1, keepdim=True)) ** 2).sum(-1).mean()
            L = L + ((log_fs / 0.10) ** 2 + (log_hs / 0.10) ** 2) * 1e-2
            L.backward()
            opt.step()
            if verbose and (k % 200 == 0 or k == k_iters - 1):
                print(f"  rig fit parts step {k:4d} loss {L.item():.4f}")
        h.remove()

    run(iters)
    # residual check: drop view-frames the rigid face / plausible hand cannot explain, refit once if any
    changed = False
    with torch.no_grad():
        X, _ = fwd()
        for rec in pobs:
            if rec is None:
                continue
            uv, _ = project(X, rec["cam"], rec["size"])
            for name, g in rec["g"].items():
                G = PART_GATE["face" if name == "face" else "hand"]
                err = ((uv[:, g["idx"]] - rec["uv"][:, g["idx"]]).norm(dim=-1) / g["spx"]).numpy()
                c = g["c"].numpy() > 0
                med = np.array([np.median(err[i][c[i]]) if c[i].any() else np.inf for i in range(n)])
                g["err"] = med
                bad = g["ok"] & (med > G["max_res"])
                if bad.any():
                    g["ok"] = g["ok"] & ~bad
                    changed = True
    if changed:
        run(max(100, iters // 2))
    with torch.no_grad():
        X, fr = fwd()
        out_pv = []
        for rec in pobs:
            pv = {}
            if rec is not None:
                uv, _ = project(X, rec["cam"], rec["size"])
                for name, g in rec["g"].items():
                    err = ((uv[:, g["idx"]] - rec["uv"][:, g["idx"]]).norm(dim=-1) / g["spx"]).numpy()
                    c = g["c"].numpy() > 0
                    m = g["ok"][:, None] & c
                    pv[name] = {"gated_in": int(g["ok0"].sum()), "usable": int(g["ok"].sum()),
                                "median_err": round(float(np.median(err[m])), 4) if m.any() else None}
            out_pv.append(pv)
        usable = {}
        for name, _ in groups:
            u = np.zeros(n, bool)
            for rec in pobs:
                if rec is not None and name in rec["g"]:
                    u |= rec["g"][name]["ok"]
            usable[name] = u
        Rh = fr["head"][0].numpy()
        hand_R = {s_: fr[s_].numpy() for s_ in ("L", "R")}
        res = {"markers133": X.numpy(), "angles_x_deg": np.degrees(angx.detach().numpy()),
               "head_R": Rh, "head_ypr_deg": head_ypr(Rh), "jaw_deg": np.degrees(fr["jaw"].detach().numpy()),
               "hand_R": hand_R,
               "palm_normal": {s_: hand_R[s_][:, :, 0] * (1.0 if s_ == "L" else -1.0) for s_ in ("L", "R")},
               "finger_dir": {s_: -hand_R[s_][:, :, 2] for s_ in ("L", "R")},
               "face_usable": usable.get("face", np.zeros(n, bool)),
               "hand_usable": {s_: usable.get(s_, np.zeros(n, bool)) for s_ in ("L", "R")},
               "usable_rate": {k: round(float(v[seen].mean()), 3) if seen.any() else 0.0 for k, v in usable.items()},
               "per_view": out_pv, "face_scale": float(torch.exp(log_fs)), "hand_scale": float(torch.exp(log_hs)),
               "parts": list(parts), "gate": PART_GATE}
    return res


def gait_metrics(res: dict, dt: float, min_speed: float = 0.35) -> dict:
    """Gait parameters from a fitted rig: walking frames (pelvis speed >= min_speed), heel strikes (heel height local
    minima while that foot moves forward), cadence, step/stride length, knee/hip flexion range per side, symmetry."""
    X, A = res["markers"], res["angles_deg"]
    seen = res["seen"]
    pel = res["root"][:, :2]
    v = np.r_[0.0, np.linalg.norm(np.diff(pel, axis=0), axis=1) / dt]
    v = np.convolve(v, np.ones(3) / 3, mode="same")
    walk = (v >= min_speed) & seen
    out = {"walking_frames": int(walk.sum()), "frames": int(len(v)),
           "speed_mps_walking_median": round(float(np.median(v[walk])), 2) if walk.any() else None}
    if walk.sum() < 6:
        out["note"] = "too little walking for gait parameters (needs >= 6 walking frames)"
        return out
    strikes = {}
    for side, heel in (("L", L_HEEL), ("R", R_HEEL)):
        z = X[:, heel, 2]
        s = [i for i in range(1, len(z) - 1) if walk[i] and z[i] <= z[i - 1] and z[i] < z[i + 1] and z[i] < 0.06]
        keep = []
        for i in s:                                   # one strike per foot per >= 0.6 s (stride >= 2 x 0.3 s step)
            if not keep or (i - keep[-1]) * dt >= 0.6:
                keep.append(i)
        strikes[side] = keep
    allst = sorted((i, s) for s in strikes for i in strikes[s])
    ok = lambda a, b: a[1] != b[1] and 0.3 <= (b[0] - a[0]) * dt <= 1.2   # noqa: E731  alternate feet, human step time
    steps = [(b[0] - a[0]) * dt for a, b in zip(allst, allst[1:]) if ok(a, b)]
    out["steps_used"] = len(steps)
    if len(steps) >= 3:
        out["cadence_spm"] = round(60.0 / float(np.median(steps)), 1)
        out["step_time_s_cv"] = round(float(np.std(steps) / np.mean(steps)), 3)
    lens = []
    for a, b in zip(allst, allst[1:]):
        if ok(a, b):
            ha = X[a[0], L_HEEL if a[1] == "L" else R_HEEL, :2]
            hb = X[b[0], L_HEEL if b[1] == "L" else R_HEEL, :2]
            lens.append(float(np.linalg.norm(hb - ha)))
    if len(lens) >= 3:
        out["step_length_m_median"] = round(float(np.median(lens)), 2)
    legs = res.get("legs_seen") or {"L": np.ones(len(v), bool), "R": np.ones(len(v), bool)}
    for name in ("knee", "hip_flex"):
        rom = {}
        for side in ("L", "R"):
            m = walk & legs[side]                     # only frames where that leg is actually observed
            if m.sum() < 6:
                continue
            a = A[m, ANG_IDX[(name, side)]]
            rom[side] = round(float(np.percentile(a, 95) - np.percentile(a, 5)), 1)
        if len(rom) < 2:
            out[f"{name}_rom_deg"] = rom or None
            continue
        out[f"{name}_rom_deg"] = rom
        if max(rom.values()) > 0:
            out[f"{name}_symmetry"] = round(min(rom.values()) / max(rom.values()), 2)
    out["heel_strikes"] = {s: len(v) for s, v in strikes.items()}
    out["leg_frames_observed"] = {s: int((walk & legs[s]).sum()) for s in ("L", "R")}
    if len(steps) < 3:
        out["note"] = ("fewer than 3 alternating steps: cadence/step length not reported (standing, shuffling or turning; "
                       f"at {1 / dt:.0f} fps a step is only ~{max(1, round(0.5 / dt))} frames)")
    return out
