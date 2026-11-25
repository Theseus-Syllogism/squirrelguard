"""World-grounded SMPL body driven by the IK rig (squirrel_guard.rig): a 3D body mesh for every replay with a rig.

  body = SMPLModel("models/smpl/SMPL_NEUTRAL.npz")
  res = fit_to_rig(body, markers, seen)            # markers (n, 23, 3) world metres (rig.markers), seen (n,) bool
  blk = body3d_block(body, res, ...)               # replay "body3d" block + mesh bytes (viewer.js format)

Why: WHAM (scripts/body3d_wham.py) places its body from ONE camera and, indoors with hidden feet, landed metres off;
the rig already holds the person's metric 3D joints (fixed bone lengths, joint limits, every calibrated camera), so
the body only has to be dressed onto it. SMPL gives the surface (6890 vertices) the rig lacks.

SMPL (Loper et al. 2015) forward pass in plain torch from the official .npz (v_template, shapedirs, posedirs,
J_regressor, weights, kintree_table, f): shape blend -> rest joints -> per-joint rotations (axis-angle) -> forward
kinematics -> linear blend skinning (+ pose correctives for the final mesh). No smplx/chumpy dependency.
LICENCE: the SMPL model and everything derived from it (meshes, body.bin files) are non-commercial: they stay on the
user's own machines (git-ignored, never published); the model file is mounted at run time, not in any image.

Frames: SMPL's canonical frame is x = subject's left, y = up, z = forward; the rig's body frame (and the world) is
right-handed with x = subject's right, y = forward, z = up: BODY_FROM_SMPL maps one to the other (a rotation).
Fit (per visit, CPU): shared shape (10 betas), per frame heading (yaw about world z, initialised from the rig),
root rotation, 23 joint rotations and translation; loss = Huber distance of SMPL's markers (joints for shoulders..
ankles, vertices for nose/eyes/ears/heels/toes) to the rig's markers + pose/shape priors + temporal smoothness.
"""
from __future__ import annotations

import math
import time

import numpy as np

# rig marker (COCO-WholeBody 0-22) -> SMPL joint index ("j", k) or vertex index ("v", k)
MARKER_SRC = {0: ("v", 332), 1: ("v", 2800), 2: ("v", 6260), 3: ("v", 583), 4: ("v", 4071),
              5: ("j", 16), 6: ("j", 17), 7: ("j", 18), 8: ("j", 19), 9: ("j", 20), 10: ("j", 21),
              11: ("j", 1), 12: ("j", 2), 13: ("j", 4), 14: ("j", 5), 15: ("j", 7), 16: ("j", 8),
              17: ("v", 3216), 18: ("v", 3226), 19: ("v", 3387), 20: ("v", 6617), 21: ("v", 6624), 22: ("v", 6787)}
# COCO hip/shoulder keypoints sit outside/above SMPL's joint centres: they pull less
MARKER_W = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 1, 1, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 0.8, 0.5, 0.8, 0.8, 0.5, 0.8])
BODY_FROM_SMPL = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
# SMPL has no finger or jaw joints (SMPL-X / SMPL+H would: their model files need registration at MPI and are not on
# this machine). The rig's hands and face (rig.fit(..., parts=...)) drive SMPL's head (joint 15) and wrists (20/21)
# instead: fingertip vertices (SMPL-H topology = SMPL's 6890 vertices) give the hand frame, compared with the rig's.
FINGERTIPS = {"L": (2746, 2319, 2445, 2556, 2673), "R": (6191, 5782, 5905, 6016, 6133)}   # thumb, index .. little
SMPL_WRIST = {"L": 20, "R": 21}
SMPL_HEAD = 15
VERSION = "smplfit/1"


def _rodrigues(aa):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    import torch
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    eye = torch.eye(3, dtype=aa.dtype).expand(K.shape)
    return eye + s * K + (1 - c) * (K @ K)


class SMPLModel:
    def __init__(self, path: str):
        import torch
        z = np.load(path)
        T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
        self.v_template = T(z["v_template"])
        self.shapedirs = T(z["shapedirs"][:, :, :10])
        self.posedirs = T(z["posedirs"]).reshape(-1, 207)                # (6890*3, 207)
        self.J_regressor = T(z["J_regressor"])
        self.weights = T(z["weights"])
        self.parents = [int(p) if p < 1e6 else -1 for p in z["kintree_table"][0]]
        self.faces = np.asarray(z["f"], np.int64)
        self.mv = [MARKER_SRC[i][1] for i in range(23) if MARKER_SRC[i][0] == "v"]
        self.mv_rows = [i for i in range(23) if MARKER_SRC[i][0] == "v"]
        self.mj = [MARKER_SRC[i][1] for i in range(23) if MARKER_SRC[i][0] == "j"]
        self.mj_rows = [i for i in range(23) if MARKER_SRC[i][0] == "j"]

    def rest(self, betas):
        """betas (10,) -> shaped rest vertices (6890, 3), rest joints (24, 3)."""
        import torch
        v = self.v_template + torch.einsum("vck,k->vc", self.shapedirs, betas)
        return v, self.J_regressor @ v

    def pose(self, betas, rots, vert_ids=None, correctives: bool = False):
        """rots (n, 24, 3, 3) local joint rotations (0 = root) in SMPL's frame -> (posed vertices (n, V, 3) for
        vert_ids (all if None), posed joints (n, 24, 3)), SMPL frame, root joint at its rest position."""
        import torch
        v0, J = self.rest(betas)
        n = rots.shape[0]
        G = [None] * 24
        for k in range(24):
            p = self.parents[k]
            t = J[k] - (J[p] if p >= 0 else 0)
            A = torch.zeros((n, 4, 4), dtype=rots.dtype)
            A[:, :3, :3], A[:, :3, 3], A[:, 3, 3] = rots[:, k], t, 1.0
            G[k] = A if p < 0 else G[p] @ A
        G = torch.stack(G, 1)                                             # (n, 24, 4, 4)
        Jp = G[:, :, :3, 3]
        # remove the rest joint location: skinning transforms
        Gs = G.clone()
        Gs[:, :, :3, 3] = Jp - torch.einsum("njab,jb->nja", G[:, :, :3, :3], J)
        ids = slice(None) if vert_ids is None else vert_ids
        vr = v0[ids]
        if correctives:
            eye = torch.eye(3, dtype=rots.dtype)
            pf = (rots[:, 1:] - eye).reshape(n, -1)                        # (n, 207)
            off = (pf @ self.posedirs.T).reshape(n, -1, 3)
            vr = vr[None] + (off if vert_ids is None else off[:, vert_ids])
        else:
            vr = vr[None].expand(n, -1, 3)
        W = self.weights[ids]                                              # (V, 24)
        T = torch.einsum("vj,njab->nvab", W, Gs)
        V = torch.einsum("nvab,nvb->nva", T[..., :3, :3], vr) + T[..., :3, 3]
        return V, Jp


def _rz(a):
    import torch
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def smpl_hand_frame(V_tips, wrist):
    """Fingertip vertices (..., 5, 3) (thumb, index .. little) and wrist joint (..., 3) -> (finger direction,
    across = little -> index), unit vectors, same definitions as rig_hand_frame()."""
    import torch
    d = torch.nn.functional.normalize(V_tips[..., 1:, :].mean(-2) - wrist, dim=-1)
    a = V_tips[..., 1, :] - V_tips[..., 4, :]
    a = torch.nn.functional.normalize(a - (a * d).sum(-1, keepdim=True) * d, dim=-1)
    return d, a


def rig_hand_frame(R):
    """rig hand frames (..., 3, 3) (rig.forward_parts) -> (finger direction, across little -> index) (curl-free)."""
    return -R[..., :, 2], R[..., :, 1]


def fit_to_rig(body: SMPLModel, markers, seen, yaw0=None, dt: float = 1 / 8, iters: int = 400,
               verbose: bool = False, head_R=None, head_ok=None, hand_R=None, hand_ok=None,
               w_orient: float = 2.0) -> dict:
    """markers (n, 23, 3) world metres (NaN rows allowed), seen (n,) bool -> per-frame SMPL world transform and
    pose (fitted on the seen frames; others are skipped). yaw0 (n,) rig heading in radians (body +y = forward).
    Optional, from rig.fit(..., parts=...)["parts"]: head_R (n, 3, 3) world head frames + head_ok (n,) bool
    (face_usable), hand_R {"L"/"R": (n, 3, 3)} + hand_ok {"L"/"R": (n,) bool}: SMPL's head joint and hands are pulled
    toward those orientations on the usable frames (w_orient). Without them the fit is exactly as before."""
    import torch
    torch.manual_seed(0)
    t0 = time.time()
    M = np.asarray(markers, float)
    idx = np.nonzero(np.asarray(seen, bool) & np.isfinite(M).all((1, 2)))[0]
    if len(idx) == 0:
        return {"frames": [], "ms": 0.0}
    tgt = torch.tensor(M[idx], dtype=torch.float64)
    n = len(idx)
    BS = torch.tensor(BODY_FROM_SMPL, dtype=torch.float64)
    pelvis = (tgt[:, 11] + tgt[:, 12]) / 2
    if yaw0 is None:                                     # heading from the hips (right-handed: forward = up x (R - L))
        lr = (tgt[:, 12] - tgt[:, 11]).numpy()
        fwd = np.cross([0, 0, 1.0], lr)
        yaw0 = np.arctan2(-fwd[:, 0], fwd[:, 1])
    else:
        yaw0 = np.asarray(yaw0, float)[idx]
    yaw = torch.tensor(yaw0, dtype=torch.float64, requires_grad=True)
    root_aa = torch.zeros((n, 3), dtype=torch.float64, requires_grad=True)
    body_aa = torch.zeros((n, 23, 3), dtype=torch.float64)
    with torch.no_grad():                                # relaxed arms (A-pose -> down): shoulders ~70 deg
        body_aa[:, 15, 2], body_aa[:, 16, 2] = -1.2, 1.2  # joints 16/17 = L/R shoulder (index - 1 in body_aa)
    body_aa.requires_grad_(True)
    betas = torch.zeros(10, dtype=torch.float64, requires_grad=True)
    trans = pelvis.clone().requires_grad_(True)
    w = torch.tensor(MARKER_W, dtype=torch.float64)
    valid = torch.isfinite(tgt).all(-1)
    tgt0 = torch.nan_to_num(tgt)
    orient = head_R is not None or hand_R is not None
    tips = [i for s_ in ("L", "R") for i in FINGERTIPS[s_]]
    vid = torch.tensor(body.mv + (tips if orient else []))
    nmv = len(body.mv)
    T_ = lambda a: torch.tensor(np.asarray(a, float)[idx], dtype=torch.float64)  # noqa: E731
    tgt_head = T_(head_R) if head_R is not None else None
    ok_head = torch.tensor(np.asarray(head_ok, bool)[idx]) if head_R is not None else None
    tgt_hand = {s_: T_(hand_R[s_]) for s_ in ("L", "R")} if hand_R is not None else None
    ok_hand = {s_: torch.tensor(np.asarray(hand_ok[s_], bool)[idx]) for s_ in ("L", "R")} if hand_R is not None else None
    aux = {}

    def markers_of():
        rots = _rodrigues(torch.cat([root_aa[:, None], body_aa], 1))
        V, Jp = body.pose(betas, rots, vert_ids=vid)
        P = torch.zeros((n, 23, 3), dtype=torch.float64)
        P[:, body.mv_rows] = V[:, :nmv]
        P[:, body.mj_rows] = Jp[:, body.mj]
        Rw = _rz(yaw) @ BS                                 # world_from_smpl per frame
        pel = (Jp[:, 1] + Jp[:, 2]) / 2                    # SMPL hip-joint midpoint = the translation anchor
        if orient:
            Vt = torch.einsum("nab,nkb->nka", Rw, V[:, nmv:] - pel[:, None])
            Jw = torch.einsum("nab,nkb->nka", Rw, Jp - pel[:, None])
            aux["tips"] = {"L": Vt[:, :5], "R": Vt[:, 5:]}
            aux["wrist"] = {s_: Jw[:, SMPL_WRIST[s_]] for s_ in ("L", "R")}
            Rg = [None] * 24                               # global joint rotations (SMPL frame)
            for k in range(24):
                p_ = body.parents[k]
                Rg[k] = rots[:, k] if p_ < 0 else Rg[p_] @ rots[:, k]
            aux["head"] = Rw @ Rg[SMPL_HEAD] @ BS.T        # head frame in the rig's axes (rest pose = body frame)
        return torch.einsum("nab,nkb->nka", Rw, P - pel[:, None]) + trans[:, None]

    def orient_loss():
        L = 0.0
        if tgt_head is not None and bool(ok_head.any()):
            L = L + ((aux["head"] - tgt_head) ** 2).sum((-1, -2))[ok_head].mean()
        if tgt_hand is not None:
            for s_ in ("L", "R"):
                if bool(ok_hand[s_].any()):
                    d, a = smpl_hand_frame(aux["tips"][s_], aux["wrist"][s_])
                    dr, ar = rig_hand_frame(tgt_hand[s_])
                    L = L + (((d - dr) ** 2).sum(-1) + ((a - ar) ** 2).sum(-1))[ok_hand[s_]].mean()
        return L

    opt = torch.optim.Adam([yaw, root_aa, body_aa, betas, trans], lr=0.02)
    for k in range(iters):
        opt.zero_grad()
        X = markers_of()
        d = (X - tgt0).norm(dim=-1)
        hub = torch.where(d < 0.03, 0.5 * d * d / 0.03, d - 0.015)
        L = (w * hub * valid).sum() / valid.sum().clamp_min(1) * 50.0
        L = L + 1e-3 * (body_aa ** 2).sum(-1).mean() * 23 + 1e-2 * (betas ** 2).sum()
        L = L + 2e-2 * (root_aa ** 2).sum(-1).mean()       # yaw carries the heading; root only tilts
        if orient:
            L = L + w_orient * orient_loss()
        if n > 2:
            acc = (body_aa[2:] - 2 * body_aa[1:-1] + body_aa[:-2])
            L = L + 2e-2 * (acc ** 2).sum(-1).mean()
        L.backward()
        opt.step()
        if verbose and (k % 100 == 0 or k == iters - 1):
            print(f"  smpl fit step {k:4d} loss {L.item():.4f} marker err median {d[valid].median().item() * 100:.1f} cm")
    with torch.no_grad():
        X = markers_of()
        err = (X - tgt0).norm(dim=-1)[valid].numpy()
        ori = {}
        if orient:
            ori = {"orient": orientation_errors(aux, tgt_head, ok_head, tgt_hand, ok_hand)}
    return {**ori, "idx": idx, "yaw": yaw.detach(), "root_aa": root_aa.detach(), "body_aa": body_aa.detach(),
            "betas": betas.detach(), "trans": trans.detach(), "markers": X.numpy(),
            "err_cm_median": round(float(np.median(err)) * 100, 1), "err_cm_p90": round(float(np.percentile(err, 90)) * 100, 1),
            "ms": round((time.time() - t0) * 1000)}


def orientation_errors(aux, tgt_head, ok_head, tgt_hand, ok_hand) -> dict:
    """Median angle (deg) between SMPL's head frame / hand directions and the rig's, on the usable frames."""
    import torch
    out = {}
    if tgt_head is not None and bool(ok_head.any()):
        c = ((torch.einsum("nij,nij->n", aux["head"], tgt_head) - 1) / 2).clamp(-1, 1)
        out["head_deg_median"] = round(float(torch.rad2deg(torch.acos(c))[ok_head].median()), 1)
    if tgt_hand is not None:
        for s_ in ("L", "R"):
            if bool(ok_hand[s_].any()):
                d, a = smpl_hand_frame(aux["tips"][s_], aux["wrist"][s_])
                dr, ar = rig_hand_frame(tgt_hand[s_])
                e = torch.rad2deg(torch.acos((d * dr).sum(-1).clamp(-1, 1)))[ok_hand[s_]]
                ea = torch.rad2deg(torch.acos((a * ar).sum(-1).clamp(-1, 1)))[ok_hand[s_]]
                out[f"hand_{s_}_finger_dir_deg_median"] = round(float(e.median()), 1)
                out[f"hand_{s_}_across_deg_median"] = round(float(ea.median()), 1)
    return out


def world_vertices(body: SMPLModel, res: dict, which=None):
    """Posed world vertices (with pose correctives) for fitted frames res["idx"][which] -> (m, 6890, 3), pelvis (m, 3)."""
    import torch
    sel = np.arange(len(res["idx"])) if which is None else np.asarray(which)
    with torch.no_grad():
        rots = _rodrigues(torch.cat([res["root_aa"][sel][:, None], res["body_aa"][sel]], 1))
        V, Jp = body.pose(res["betas"], rots, correctives=True)
        Rw = _rz(res["yaw"][sel]) @ torch.tensor(BODY_FROM_SMPL, dtype=torch.float64)
        pel = (Jp[:, 1] + Jp[:, 2]) / 2
        Vw = torch.einsum("nab,nvb->nva", Rw, V - pel[:, None]) + res["trans"][sel][:, None]
        root = torch.einsum("nab,nb->na", Rw, Jp[:, 0] - pel) + res["trans"][sel]
    return Vw.numpy(), root.numpy()


def body3d_block(body: SMPLModel, res: dict, n_replay: int, frame_of_replay, name: str, mesh_step: int = 1) -> tuple[dict, bytes, bytes]:
    """Fit result -> (replay "body3d" block, vertices .bin bytes, faces .bin bytes) in the viewer's format.
    frame_of_replay(i) -> the fit's frame index (into res["idx"] space: the rig frame) shown at replay frame i."""
    pos = {int(f): k for k, f in enumerate(res["idx"])}
    slots = []
    for i in range(n_replay):
        k = pos.get(int(frame_of_replay(i)))
        slots.append(k)
    mesh_rows = [i for i, k in enumerate(slots) if k is not None and i % max(1, mesh_step) == 0]
    ks = [slots[i] for i in mesh_rows]
    V, root = world_vertices(body, res, ks) if ks else (np.zeros((0, 6890, 3)), np.zeros((0, 3)))
    roots = [None] * n_replay
    for r, (i, k) in enumerate(zip(mesh_rows, ks)):
        roots[i] = [round(float(x), 4) for x in root[r]]
    joints = []
    for i, k in enumerate(slots):
        if k is None:
            joints.append(None)
        else:
            P = res["markers"][k][:17]
            joints.append([[round(float(x), 3) for x in p] for p in P])
    rel = (V - root[:, None]).astype("<f2") if len(ks) else np.zeros((0, 6890, 3), "<f2")
    blk = {"version": VERSION,
           "method": "SMPL dressed onto the IK rig (squirrel_guard.smplfit): shared shape, per-frame pose fitted to "
                     f"the rig's 3D markers; marker error {res['err_cm_median']} cm median, {res['err_cm_p90']} cm p90",
           "licence": "SMPL-derived (non-commercial): local use only, never published",
           "model": "SMPL neutral", "frame": "replay world, metres, Z up", "root_mode": "rig",
           "joints_format": "COCO-17 (same order as skeleton)", "joints": joints, "root": roots,
           "betas": [round(float(b), 4) for b in res["betas"]],
           "stats": {"fit_ms": res["ms"], "marker_err_cm_median": res["err_cm_median"],
                     "marker_err_cm_p90": res["err_cm_p90"], "replay_frames_with_body": len(mesh_rows),
                     "replay_frames": n_replay},
           "mesh": {"vertices": f"{name}_body.bin", "faces": f"{name}_body_faces.bin", "vertex_count": 6890,
                    "face_count": int(len(body.faces)), "index": mesh_rows,
                    "encoding": "float16 little-endian, (len(index), vertex_count, 3), relative to root[index[i]]"}}
    return blk, rel.tobytes(), body.faces.astype("<u2").tobytes()


def body_for_replay(body: SMPLModel, tr: dict, name: str, mesh_step: int = 1, iters: int = 400,
                    verbose: bool = False) -> tuple[dict, bytes, bytes] | None:
    """A replay with "rig" -> (body3d block, vertices bytes, faces bytes) or None (no rig / too few markers)."""
    rg = tr.get("rig") or {}
    if not rg.get("markers"):
        return None
    n = len(tr["t"])
    M = np.full((n, 23, 3), np.nan)
    for i, fr in enumerate(rg["markers"][:n]):
        if fr:
            M[i] = [[np.nan] * 3 if p is None else p for p in fr]
    seen = np.isfinite(M).all(-1).sum(1) >= 12
    if not seen.any():
        return None
    kw = {}
    pb = rg.get("parts") or {}
    if pb.get("head_R"):                       # rig head/hands (rigpipe.parts_block) drive SMPL's head and wrists
        def mats(lst):
            R = np.tile(np.eye(3), (n, 1, 1))
            okm = np.zeros(n, bool)
            for i, v in enumerate(lst[:n]):
                if v:
                    R[i], okm[i] = np.asarray(v, float).reshape(3, 3), True
            return R, okm
        kw["head_R"], kw["head_ok"] = mats(pb["head_R"])
        hr = {s_: mats((pb.get("hand_R") or {}).get(s_) or []) for s_ in ("L", "R")}
        kw["hand_R"], kw["hand_ok"] = {s_: v[0] for s_, v in hr.items()}, {s_: v[1] for s_, v in hr.items()}
    res = fit_to_rig(body, M, seen, dt=1.0 / float(tr.get("fps") or 8), iters=iters, verbose=verbose, **kw)
    if not len(res.get("idx", [])):
        return None
    return body3d_block(body, res, n, lambda i: i, name, mesh_step=mesh_step)
