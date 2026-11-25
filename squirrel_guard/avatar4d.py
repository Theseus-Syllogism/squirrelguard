"""4D dynamic Gaussian-splat avatar of one person from a calibrated multi-camera clip, driven by the SMPL fit.

  prep   (project venv, CPU):  fit SMPL to the rig markers (smplfit.fit_to_rig), decode + undistort both clips,
                               crop each person view to a fixed 384x512 window, YOLO person masks -> prep.npz
  masks  (a venv with transformers SAM, optional): replace the masks with SAM ViT-B box-prompted masks
  train  (splat venv, CUDA + gsplat): optimise the avatar, evaluate held-out frames / camera -> model.pt, eval.json
  export: compact binary for the browser (deploy/recorder/replay/avatar4d.html) -> multicam_avatar.bin/.json

METHOD (own implementation; ideas from the published avatar-splatting literature, no code taken from any repo)
  * Canonical space = SMPL rest pose (zero pose, fitted betas, SMPL frame). N Gaussians are initialised on surface
    samples of the rest mesh (area-weighted), each carrying a fixed 24-joint skinning-weight vector interpolated
    barycentrically from the three vertices of its triangle.
  * Per Gaussian: canonical position, log-scale (3), rotation quaternion, opacity, colour as SH degree 1.
  * Per frame f: world bone transforms A_f (24, 4x4) = world_from_smpl(yaw, pelvis, trans) @ SMPL skinning transform
    (from the rig-driven SMPL fit). A Gaussian moves by its blended transform T = sum_j w_j A_fj (linear blend
    skinning), its covariance by M Sigma M^T (M = T's 3x3). No pose correctives (they are absorbed by the offsets).
  * Non-rigid clothing: a small MLP(posenc(x_canonical), z_f) -> canonical offset (bounded, 0.12 m * tanh), where
    z_f is a 16-d per-frame latent. A held-out frame gets z interpolated from its neighbours (it was never seen).
    A per-frame translation correction (few cm) absorbs SMPL fit drift against the images.
  * Per camera: a colour gain/bias (camera 0 fixed); a held-out camera keeps the identity (no test-time fitting).
  * Loss: masked L1 + 0.2 (1 - SSIM) on the person (ground truth composited on black), alpha vs mask L1,
    offset magnitude, temporal smoothness of offsets (consecutive frames, sampled Gaussians), latent/translation
    priors, anisotropy (longest / middle axis above 4 is penalised: needles were a problem in the static splats)
    and a hard cap on scales.
  * Baseline ("smplcolor"): the same renderer with Gaussians frozen on the SMPL surface (no offsets, fixed small
    isotropic scale, opaque) and only colour trained: a per-sample coloured SMPL mesh.

LIMITS
  * Only frames with rig markers (and a SMPL fit) are used; the fit's error (a few cm) shows as blur/doubling.
  * The coat is modelled as offsets from the body surface: hem/flaps that swing far from SMPL are approximate;
    the offset MLP needs training views at similar latent, so held-out frames look smoother than training frames.
  * SH is evaluated in world frame (not rotated with the body): degree 1 only, small effect.
  * Masks (YOLO 320 or SAM) include whatever the segmenter calls "person"; occluders cut holes the alpha loss sees.

LICENCE: SMPL (Loper et al. 2015) is non-commercial. Everything this module writes (prep.npz, model.pt, the
multicam_avatar.* export, renders) is SMPL-derived: local use only, never committed, published or deployed.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

CROP_W, CROP_H = 384, 512
C0 = 0.28209479177387814


# ------------------------------------------------------------------------------------------------------------- prep
def _markers(tr: dict) -> np.ndarray:
    rg = tr["rig"]
    n = len(tr["t"])
    M = np.full((n, 23, 3), np.nan)
    for i, fr in enumerate(rg["markers"][:n]):
        if fr:
            M[i] = [[np.nan] * 3 if p is None else p for p in fr]
    return M


def bone_transforms(body, res) -> tuple[np.ndarray, np.ndarray]:
    """SMPL fit -> (world bone transforms A (n, 24, 4, 4), shaped rest vertices v0 (6890, 3)) such that
    world vertex (no correctives) = sum_j w_j A_j [v0; 1]."""
    import torch
    from .smplfit import BODY_FROM_SMPL, _rodrigues, _rz
    with torch.no_grad():
        rots = _rodrigues(torch.cat([res["root_aa"][:, None], res["body_aa"]], 1))
        v0, J = body.rest(res["betas"])
        n = rots.shape[0]
        G = [None] * 24
        for k in range(24):
            p = body.parents[k]
            t = J[k] - (J[p] if p >= 0 else 0)
            A = torch.zeros((n, 4, 4), dtype=rots.dtype)
            A[:, :3, :3], A[:, :3, 3], A[:, 3, 3] = rots[:, k], t, 1.0
            G[k] = A if p < 0 else G[p] @ A
        G = torch.stack(G, 1)
        Jp = G[:, :, :3, 3]
        Gs = G.clone()
        Gs[:, :, :3, 3] = Jp - torch.einsum("njab,jb->nja", G[:, :, :3, :3], J)
        Rw = _rz(res["yaw"]) @ torch.tensor(BODY_FROM_SMPL, dtype=torch.float64)
        pel = (Jp[:, 1] + Jp[:, 2]) / 2
        W = torch.zeros((n, 4, 4), dtype=torch.float64)
        W[:, :3, :3] = Rw
        W[:, :3, 3] = res["trans"] - torch.einsum("nab,nb->na", Rw, pel)
        W[:, 3, 3] = 1
        A = torch.einsum("nab,njbc->njac", W, Gs)
    return A.numpy(), v0.numpy()


def _crop_window(box, img_w, img_h, margin=0.12):
    """Box (x0, y0, x1, y1) -> crop (x0, y0, w, h) with CROP_W:CROP_H aspect around it (may leave the image)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max(y1 - y0, (x1 - x0) * CROP_H / CROP_W) * (1 + 2 * margin)
    w = h * CROP_W / CROP_H
    return cx - w / 2, cy - h / 2, w, h


def prepare(clips: dict, calibs: dict, replay: str, out: str, smpl: str, shifts: dict | None = None,
            fps: float = 8.0, iters: int = 400, seg_model: str | None = None, verbose: bool = True) -> dict:
    """clips/calibs: {camera name: path} (first = primary, the replay's frame base). shifts: {name: frame shift}
    (secondary frame = primary frame + shift; default from replay["multicam"]["offsets"])."""
    import cv2
    import torch
    from .geometry import Camera
    from .kinematics import decode
    from .smplfit import SMPLModel, fit_to_rig
    t0 = time.time()
    tr = json.loads(Path(replay).read_text())
    names = list(clips)
    if shifts is None:
        offs = (tr.get("multicam") or {}).get("offsets") or {}
        shifts = {k: int(v["shift_frames"]) for k, v in offs.items()}
    shifts = {n: int(shifts.get(n, 0)) if k else 0 for k, n in enumerate(names)}
    M = _markers(tr)
    seen = np.isfinite(M).all(-1).sum(1) >= 12
    body = SMPLModel(smpl)
    torch.set_num_threads(4)
    res = fit_to_rig(body, M, seen, dt=1.0 / float(tr.get("fps") or 8), iters=iters, verbose=verbose)
    idx = np.asarray(res["idx"])
    A, v0 = bone_transforms(body, res)
    if verbose:
        print(f"SMPL fit: {len(idx)} frames, marker err {res['err_cm_median']} cm median / {res['err_cm_p90']} p90")
    # posed world vertices (with correctives) only for crop boxes
    import squirrel_guard.smplfit as sf
    Vw, _ = sf.world_vertices(body, res)
    masker = None
    if seg_model:
        from .replay_assets import PersonMasker
        masker = PersonMasker(seg_model, threads=4)
    views = {"img": [], "mask": [], "valid": [], "K": [], "viewmat": [], "frame": [], "cam": [], "box": []}
    cams_meta = []
    for ci, name in enumerate(names):
        cam = Camera.load(calibs[name])
        W, H = cam.size
        need = {int(i) + shifts[name]: k for k, i in enumerate(idx)}
        frames, eff = decode(Path(clips[name]).read_bytes(), fps, 60.0)
        map1, map2 = cv2.initUndistortRectifyMap(cam.K, cam.D, None, cam.K, (W, H), cv2.CV_32FC1)
        vm = np.eye(4)
        vm[:3, :3] = cam.R.T
        vm[:3, 3] = -cam.R.T @ cam.C
        cams_meta.append({"name": name, "size": [W, H], "K": cam.K.tolist(), "shift_frames": shifts[name],
                          "decode_fps": eff})
        got = 0
        for j, fr in enumerate(frames):
            k = need.get(j)
            if k is None:
                continue
            if fr.shape[1] != W:
                fr = cv2.resize(fr, (W, H))
            # project the posed body (pinhole, the undistorted image's model)
            Pc = (Vw[k] - cam.C) @ cam.R
            if (Pc[:, 2] < 0.2).any():
                continue
            uv = Pc[:, :2] / Pc[:, 2:] @ cam.K[:2, :2].T + cam.K[:2, 2]
            box = [*uv.min(0), *uv.max(0)]
            if box[2] < 0 or box[0] > W or box[3] < 0 or box[1] > H:
                continue
            cx, cy, cw, ch = _crop_window(box, W, H)
            s = CROP_H / ch
            # affine crop of the undistorted frame: crop pixel = s * (undistorted pixel - (cx, cy))
            Aff = np.array([[s, 0, -s * cx], [0, s, -s * cy]])
            mx = cv2.warpAffine(map1, Aff, (CROP_W, CROP_H), flags=cv2.INTER_LINEAR, borderValue=-1)
            my = cv2.warpAffine(map2, Aff, (CROP_W, CROP_H), flags=cv2.INTER_LINEAR, borderValue=-1)
            img = cv2.remap(fr, mx, my, cv2.INTER_LINEAR, borderValue=0)
            valid = ((mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)).astype(np.uint8)
            Kc = np.array([[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]]) @ cam.K
            bc = [(box[0] - cx) * s, (box[1] - cy) * s, (box[2] - cx) * s, (box[3] - cy) * s]
            m = None
            if masker is not None:
                pad = 0.06 * (bc[3] - bc[1])
                m = masker._mask(img, [bc[0] - pad, bc[1] - pad, bc[2] + pad, bc[3] + pad])
            if m is None:
                m = np.zeros((CROP_H, CROP_W), np.uint8)
            views["img"].append(img)
            views["mask"].append((m > 127).astype(np.uint8))
            views["valid"].append(valid)
            views["K"].append(Kc)
            views["viewmat"].append(vm)
            views["frame"].append(k)
            views["cam"].append(ci)
            views["box"].append(bc)
            got += 1
        if verbose:
            print(f"{name}: {got} views (shift {shifts[name]:+d} frames, decode {eff:.2f} fps)")
    np.savez_compressed(
        out, A=A.astype(np.float32), v0=v0.astype(np.float32), faces=body.faces.astype(np.int32),
        weights=body.weights.numpy().astype(np.float32), rig_frame=idx.astype(np.int32),
        img=np.stack(views["img"]), mask=np.stack(views["mask"]), valid=np.stack(views["valid"]),
        K=np.stack(views["K"]).astype(np.float32), viewmat=np.stack(views["viewmat"]).astype(np.float32),
        frame=np.array(views["frame"], np.int32), cam=np.array(views["cam"], np.int32),
        box=np.array(views["box"], np.float32),
        meta=json.dumps({"cameras": cams_meta, "replay": str(replay), "fps": tr.get("fps"),
                         "smpl_err_cm_median": res["err_cm_median"], "smpl_err_cm_p90": res["err_cm_p90"],
                         "betas": res["betas"].tolist(), "seconds": round(time.time() - t0, 1)}))
    return {"frames": len(idx), "views": len(views["img"]), "seconds": round(time.time() - t0, 1)}


def sam_masks(prep: str, dev: str = "cuda") -> dict:
    """Replace prep.npz masks by SAM ViT-B masks prompted with the projected SMPL box (+ the YOLO mask's box
    union when it is larger: the coat reaches past the body)."""
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor
    z = dict(np.load(prep, allow_pickle=False))
    sp = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dt = torch.float16 if dev == "cuda" else torch.float32
    sm = SamModel.from_pretrained("facebook/sam-vit-base", torch_dtype=dt).to(dev).eval()
    out, ious = [], []
    for img, bc, ym in zip(z["img"], z["box"], z["mask"]):
        b = np.array(bc, float)
        pad = 0.04 * (b[3] - b[1])
        b = b + [-pad, -pad, pad, pad]
        if ym.any():
            ys, xs = np.nonzero(ym)
            b = np.array([min(b[0], xs.min()), min(b[1], ys.min()), max(b[2], xs.max()), max(b[3], ys.max())])
        b = np.clip(b, 0, [CROP_W - 1, CROP_H - 1, CROP_W - 1, CROP_H - 1])
        pil = Image.fromarray(img[:, :, ::-1])
        with torch.no_grad():
            inp = sp(pil, input_boxes=[[b.tolist()]], return_tensors="pt").to(dev)
            o = sm(pixel_values=inp["pixel_values"].to(dt), input_boxes=inp["input_boxes"].to(dt),
                   multimask_output=True)
            ms = sp.image_processor.post_process_masks(o.pred_masks.float().cpu(), inp["original_sizes"].cpu(),
                                                       inp["reshaped_input_sizes"].cpu())[0][0].numpy()
            sc = o.iou_scores.float().cpu().numpy().reshape(-1)
        k = int(np.argmax(sc))
        m = ms[k].astype(np.uint8)
        out.append(m)
        if ym.any():
            ious.append(float((m & ym).sum() / max(1, (m | ym).sum())))
    z["mask_yolo"] = z["mask"]
    z["mask"] = np.stack(out)
    np.savez_compressed(prep, **z)
    return {"views": len(out), "iou_vs_yolo_median": round(float(np.median(ious)), 3) if ious else None}


# ------------------------------------------------------------------------------------------------------------ model
def _surface_samples(v0, faces, weights, n, rng):
    a, b, c = v0[faces[:, 0]], v0[faces[:, 1]], v0[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    fi = rng.choice(len(faces), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    bw = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    f = faces[fi]
    x = (bw[:, :, None] * v0[f]).sum(1)
    w = (bw[:, :, None] * weights[f]).sum(1)
    nrm = np.cross(b[fi] - a[fi], c[fi] - a[fi])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return x, w, nrm


def _posenc(x, nf=4):
    import torch
    out = [x]
    for k in range(nf):
        out += [torch.sin((2.0 ** k) * math.pi * x), torch.cos((2.0 ** k) * math.pi * x)]
    return torch.cat(out, -1)


def _quat_to_rot(q):
    import torch
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def _ssim_map(x, y):
    import torch
    import torch.nn.functional as F
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = ((g[:, None] * g[None, :]) / g.sum() ** 2).expand(3, 1, 11, 11)
    mu_x, mu_y = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


class Avatar:
    """Canonical Gaussians + skinning + non-rigid offsets. All tensors on `dev`."""

    def __init__(self, v0, faces, weights, n_frames, n_cams, n=40000, mode="full", dev="cuda", seed=0, zdim=16):
        import torch
        self.dev, self.mode, self.n_frames, self.zdim = dev, mode, n_frames, zdim
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        x, w, nrm = _surface_samples(v0, faces, weights, n, rng)
        T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
        self.weights = T(w)                                           # (N, 24) fixed
        s0 = math.log(0.012 if mode == "full" else 0.010)
        P = torch.nn.Parameter
        self.p = torch.nn.ParameterDict({
            "xyz": P(T(x)),
            "scales": P(torch.full((n, 3), s0, device=dev)),
            "quats": P(torch.cat([torch.ones(n, 1, device=dev), torch.zeros(n, 3, device=dev)], 1)),
            "opac": P(torch.full((n,), 2.0 if mode == "full" else 6.0, device=dev)),
            "sh0": P(torch.zeros(n, 1, 3, device=dev)),
            "shN": P(torch.zeros(n, 3, 3, device=dev)),
            "z": P(torch.zeros(n_frames, zdim, device=dev)),
            "dt": P(torch.zeros(n_frames, 3, device=dev)),
            "gain": P(torch.zeros(n_cams, 3, device=dev)),
            "bias": P(torch.zeros(n_cams, 3, device=dev)),
        })
        self.mlp = torch.nn.Sequential(torch.nn.Linear(27 + zdim, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128),
                                       torch.nn.SiLU(), torch.nn.Linear(128, 3)).to(dev)
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()
        self.trainable = ["sh0"] if mode == "smplcolor" else list(self.p.keys())
        self.held_frames: set[int] = set()

    def latent(self, f: int, key: str = "z"):
        """Per-frame latent (or translation fix, key="dt"); a held-out frame's value is interpolated from the
        nearest trained frames."""
        if f not in self.held_frames:
            return self.p[key][f]
        lo = next((g for g in range(f - 1, -1, -1) if g not in self.held_frames), None)
        hi = next((g for g in range(f + 1, self.n_frames) if g not in self.held_frames), None)
        if lo is None:
            return self.p[key][hi]
        if hi is None:
            return self.p[key][lo]
        a = (f - lo) / (hi - lo)
        return (1 - a) * self.p[key][lo] + a * self.p[key][hi]

    def offsets(self, f: int, sel=None):
        import torch
        if self.mode == "smplcolor":
            return None
        x = self.p["xyz"] if sel is None else self.p["xyz"][sel]
        z = self.latent(f)[None].expand(len(x), -1)
        return 0.12 * torch.tanh(self.mlp(torch.cat([_posenc(x), z], -1)))

    def world(self, A_f, f: int):
        """A_f (24, 4, 4) -> world means (N, 3), covariances (N, 3, 3), offsets (N, 3) or None."""
        import torch
        T = torch.einsum("nj,jab->nab", self.weights, A_f)
        off = self.offsets(f)
        xc = self.p["xyz"] + (off if off is not None else 0)
        M = T[:, :3, :3]
        mean = torch.einsum("nab,nb->na", M, xc) + T[:, :3, 3]
        if self.mode != "smplcolor":
            mean = mean + self.latent(f, "dt")
        R = _quat_to_rot(self.p["quats"])
        S = torch.exp(self.p["scales"])
        L = M @ R * S[:, None, :]
        return mean, L @ L.transpose(1, 2), off

    def colors(self):
        import torch
        return torch.cat([self.p["sh0"], self.p["shN"]], 1)

    def opacity(self):
        import torch
        return torch.sigmoid(self.p["opac"])

    def render(self, A_f, f, viewmat, K, W=CROP_W, H=CROP_H, cam: int | None = None, sh_degree=1):
        import torch
        from gsplat import rasterization
        mean, cov, off = self.world(A_f, f)
        out, alpha, info = rasterization(mean, None, None, self.opacity(), self.colors(), viewmat[None], K[None],
                                         W, H, sh_degree=sh_degree if self.mode == "full" else 0, packed=True,
                                         near_plane=0.05, covars=cov)
        img = out[0, ..., :3]
        if cam is not None and self.mode == "full":
            img = img * torch.exp(self.p["gain"][cam]) + self.p["bias"][cam] * alpha[0]
        return img.clamp(0, 1), alpha[0, ..., 0], off


# ------------------------------------------------------------------------------------------------------------ train
def _metrics(img, gt, mask_gt, alpha):
    import torch
    mse = ((img - gt) ** 2).mean()
    psnr = float(-10 * torch.log10(mse.clamp_min(1e-10)))
    ss = float(_ssim_map(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean())
    m = mask_gt > 0.5
    # PSNR over the person's bounding region (the dilated union of GT mask and render alpha)
    reg = (m | (alpha > 0.5)).float()
    import torch.nn.functional as F
    reg = F.max_pool2d(reg[None, None], 9, 1, 4)[0, 0] > 0
    mse_r = ((img - gt) ** 2)[reg].mean() if reg.any() else mse
    pr = float(-10 * torch.log10(mse_r.clamp_min(1e-10)))
    inter = ((alpha > 0.5) & m).sum().float()
    uni = ((alpha > 0.5) | m).sum().float().clamp_min(1)
    return {"psnr": psnr, "psnr_person": pr, "ssim": ss, "mask_iou": float(inter / uni)}


def train(prep: str, out_dir: str, iters: int = 8000, n_gauss: int = 40000, holdout_every: int = 8,
          holdout_cam: int | None = None, mode: str = "full", seed: int = 0, log=print, fill_reg: float = 0.0,
          tag: str | None = None) -> dict:
    import torch
    dev = "cuda"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(prep, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    A = torch.tensor(z["A"], device=dev)
    nF = len(A)
    imgs = torch.tensor(z["img"][..., ::-1].copy(), device=dev)                  # uint8 RGB (V, H, W, 3)
    masks = torch.tensor(z["mask"], device=dev).float()
    valid = torch.tensor(z["valid"], device=dev).float()
    Ks = torch.tensor(z["K"], device=dev)
    VM = torch.tensor(z["viewmat"], device=dev)
    frame, cam = z["frame"], z["cam"]
    n_cams = len(meta["cameras"])
    held_frames = set(range(0, nF, holdout_every)) if holdout_every else set()
    if holdout_cam is not None:
        held_frames = set()
    test = [v for v in range(len(frame)) if frame[v] in held_frames or (holdout_cam is not None and cam[v] == holdout_cam)]
    train_v = [v for v in range(len(frame)) if v not in set(test)]
    av = Avatar(z["v0"], z["faces"], z["weights"], nF, n_cams, n=n_gauss, mode=mode, seed=seed)
    av.held_frames = held_frames
    # initial colour: mean colour inside the masks (grey-ish start), so early gradients are about geometry
    with torch.no_grad():
        mc = (imgs[train_v].float() / 255 * masks[train_v, ..., None]).sum((0, 1, 2)) / masks[train_v].sum().clamp_min(1)
        av.p["sh0"][:] = ((mc - 0.5) / C0)[None, None]
    lr = {"xyz": 4e-4, "scales": 5e-3, "quats": 1e-3, "opac": 5e-2, "sh0": 5e-3, "shN": 5e-3 / 20, "z": 1e-3,
          "dt": 5e-4, "gain": 1e-3, "bias": 1e-3}
    groups = [{"params": [av.p[k]], "lr": lr[k], "name": k} for k in av.trainable]
    if mode == "full":
        groups.append({"params": list(av.mlp.parameters()), "lr": 5e-4, "name": "mlp"})
    opt = torch.optim.Adam(groups, eps=1e-15)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 ** (s / iters))
    rng = np.random.default_rng(seed)
    if fill_reg > 0:
        # visibility of the initial (surface) Gaussians in the training views; recomputed once mid-training
        vis_w = (visibility(av, A, Ks, VM, masks, valid, frame, train_v) / 4.0).clamp(max=1.0)
        mj, mdist = mirror_partner(av.p["xyz"].detach())
        nb = _knn(av.p["xyz"].detach(), 8)
        log(f"fill_reg {fill_reg}: {int((vis_w < 0.5).sum())} of {n_gauss} Gaussians seen in < 2 training views")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    log(f"train {mode}: {len(train_v)} train views, {len(test)} test views, {nF} frames, {n_gauss} Gaussians")
    for step in range(iters):
        v = int(rng.choice(train_v))
        f, c = int(frame[v]), int(cam[v])
        gt = imgs[v].float() / 255
        m, val = masks[v], valid[v]
        img, alpha, off = av.render(A[f], f, VM[v], Ks[v], cam=c)
        gtm = gt * m[..., None]
        # person-weighted: the dilated mask counts 4x more than the (black) background, which fills most of the crop
        pw = 1 + 4 * (torch.nn.functional.max_pool2d(m[None, None], 15, 1, 7)[0, 0] > 0).float()
        w = (val * pw)[..., None]
        l1 = ((img - gtm).abs() * w).sum() / (w.sum() * 3).clamp_min(1)
        ss = (_ssim_map(img.permute(2, 0, 1)[None], gtm.permute(2, 0, 1)[None]) * val).sum() / (val.sum() * 3).clamp_min(1)
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        if mode == "full":
            loss = loss + 0.3 * ((alpha - m).abs() * val).sum() / val.sum().clamp_min(1)
            ls = av.p["scales"].sort(dim=1).values
            loss = loss + 0.01 * torch.relu(torch.exp(ls[:, 2] - ls[:, 1]) - 4.0).mean()
            loss = loss + 0.3 * (off ** 2).sum(-1).mean()
            loss = loss + 2e-3 * (av.p["z"] ** 2).sum(-1).mean() + 5.0 * (av.p["dt"] ** 2).sum(-1).mean()
            loss = loss + 0.05 * (av.p["gain"] ** 2).sum() + 0.05 * (av.p["bias"] ** 2).sum()
            # temporal smoothness of the offsets: neighbouring frame, random subset of Gaussians
            g = f + (1 if f + 1 < nF else -1)
            if g not in held_frames:
                sel = torch.randint(0, n_gauss, (4096,), device=dev)
                loss = loss + 20.0 * ((av.offsets(f, sel) - av.offsets(g, sel)) ** 2).sum(-1).mean()
        if fill_reg > 0:
            if step == iters // 2:
                with torch.no_grad():
                    vis_w = (visibility(av, A, Ks, VM, masks, valid, frame, train_v) / 4.0).clamp(max=1.0)
            sel = torch.randint(0, n_gauss, (8192,), device=dev)
            u = 1.0 - vis_w[sel]
            c = av.p["sh0"][:, 0]
            cm = c[mj[sel]].detach()
            wm = u * vis_w[mj[sel]] * (mdist[sel] < 0.03).float()       # only toward a mirror partner that was seen
            ln = ((c[sel] - c[nb[sel]].mean(1).detach()) ** 2).sum(-1)
            loss = loss + fill_reg * ((wm * ((c[sel] - cm) ** 2).sum(-1)).mean() + (u * ln).mean())
            if off is not None:
                loss = loss + fill_reg * 10.0 * (u[:, None] * off[sel] ** 2).sum(-1).mean()
        loss.backward()
        with torch.no_grad():
            if av.p["gain"].grad is not None:
                av.p["gain"].grad[0] = 0
                av.p["bias"].grad[0] = 0
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        with torch.no_grad():
            av.p["scales"].clamp_(max=math.log(0.04), min=math.log(5e-4))
        if step % 1000 == 0 or step == iters - 1:
            log(f"  step {step:5d} loss {loss.item():.4f} l1 {l1.item():.4f} "
                f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB {time.time() - t0:.0f}s")
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    tag = tag or mode
    res = evaluate(av, A, imgs, masks, Ks, VM, frame, cam, test, out, tag=tag)
    res.update({"mode": mode, "iters": iters, "gaussians": n_gauss, "train_views": len(train_v),
                "test_views": len(test), "holdout_every": holdout_every, "holdout_cam": holdout_cam,
                "train_minutes": round(secs / 60, 2), "peak_vram_gib": round(peak, 2),
                "smpl_err_cm_median": meta.get("smpl_err_cm_median"), "fill_reg": fill_reg})
    torch.save({"p": {k: v.detach().cpu() for k, v in av.p.items()}, "mlp": av.mlp.state_dict(),
                "weights": av.weights.cpu(), "mode": mode, "held_frames": sorted(held_frames),
                "n_frames": nF, "n_cams": n_cams, "holdout_cam": holdout_cam, "holdout_every": holdout_every},
               out / f"model_{tag}.pt")
    (out / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    log(json.dumps({k: v for k, v in res.items() if k != "per_view"}))
    return res


def evaluate(av, A, imgs, masks, Ks, VM, frame, cam, test, out: Path, tag: str) -> dict:
    import cv2
    import torch
    rows, per = [], []
    with torch.no_grad():
        for v in test:
            f, c = int(frame[v]), int(cam[v])
            img, alpha, _ = av.render(A[f], f, VM[v], Ks[v], cam=c)
            gt = imgs[v].float() / 255 * masks[v][..., None]
            mt = _metrics(img, gt, masks[v], alpha)
            mt.update({"view": int(v), "frame": f, "cam": c})
            per.append(mt)
            np.save(out / f"render_{tag}_{v:04d}.npy", (img.cpu().numpy() * 255).astype(np.uint8))
    summ = {}
    for c in sorted(set(p["cam"] for p in per)):
        ps = [p for p in per if p["cam"] == c]
        summ[f"cam{c}"] = {k: round(float(np.mean([p[k] for p in ps])), 3) for k in ("psnr", "psnr_person", "ssim", "mask_iou")}
        summ[f"cam{c}"]["n"] = len(ps)
    allm = {k: round(float(np.mean([p[k] for p in per])), 3) for k in ("psnr", "psnr_person", "ssim", "mask_iou")} if per else {}
    return {"heldout": allm, "by_camera": summ, "per_view": per}


# ------------------------------------------------------------------------------------------- unseen-side colour fill
# A body region no training camera saw has no photometric gradient: its Gaussians keep the initial mean colour (and
# whatever offsets the MLP extrapolates). Three licence-free fills, measured on the held-out-camera split in
# docs/avatar4d.md:
#   visibility(): per Gaussian, the number of TRAINING views in which it is the front surface (projected inside the
#     view, inside the person mask, within `tol` of the rendered depth there). Held-out views are never used.
#   fill_colours(mode): post-hoc, on a trained model. "mirror": an unseen Gaussian takes the colour of its left/right
#     mirror partner in the canonical SMPL rest pose when that one was seen; "diffuse": harmonic interpolation of the
#     colour over a k-nearest-neighbour graph of the canonical positions, seen Gaussians held fixed (the smoothest
#     colour field consistent with what was seen); "mirror+diffuse": mirror first, then diffuse the rest.
#   train(fill_reg=w): the same priors as losses during training, weighted by (1 - visibility): unseen colours pulled
#     toward the mirror partner and the neighbour mean, unseen offsets toward zero (the body surface).
def _knn(x, k: int, chunk: int = 4096):
    """x (N, 3) torch -> indices (N, k) of the k nearest other points."""
    import torch
    out = []
    for i in range(0, len(x), chunk):
        d = torch.cdist(x[i:i + chunk], x)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        out.append(d.topk(k, largest=False).indices)
    return torch.cat(out)


def mirror_partner(x, chunk: int = 4096):
    """Canonical SMPL positions (x = subject's left) -> index of the nearest point to each point's mirror image."""
    import torch
    m = x * torch.tensor([-1.0, 1.0, 1.0], device=x.device)
    out, dist = [], []
    for i in range(0, len(x), chunk):
        d = torch.cdist(m[i:i + chunk], x)
        v, j = d.min(1)
        out.append(j)
        dist.append(v)
    return torch.cat(out), torch.cat(dist)


def visibility(av, A, Ks, VM, masks, valid, frame, views, tol: float = 0.04):
    """Count, per Gaussian, the given views in which it is the visible front surface of the person."""
    import torch
    from gsplat import rasterization
    cnt = torch.zeros(len(av.p["xyz"]), device=av.dev)
    with torch.no_grad():
        for v in views:
            f = int(frame[v])
            mean, cov, _ = av.world(A[f], f)
            out, alpha, _ = rasterization(mean, None, None, av.opacity(), av.colors(), VM[v][None], Ks[v][None],
                                          CROP_W, CROP_H, sh_degree=0 if av.mode != "full" else 1, packed=True,
                                          near_plane=0.05, covars=cov, render_mode="RGB+ED")
            D = out[0, ..., 3]
            Xc = mean @ VM[v][:3, :3].T + VM[v][:3, 3]
            z = Xc[:, 2]
            uv = Xc[:, :2] / z.clamp_min(1e-3)[:, None] @ Ks[v][:2, :2].T + Ks[v][:2, 2]
            u, w = uv[:, 0].round().long(), uv[:, 1].round().long()
            inb = (z > 0.05) & (u >= 0) & (u < CROP_W) & (w >= 0) & (w < CROP_H)
            uc, wc = u.clamp(0, CROP_W - 1), w.clamp(0, CROP_H - 1)
            ok = inb & (masks[v][wc, uc] > 0.5) & (valid[v][wc, uc] > 0.5) & (alpha[0, wc, uc, 0] > 0.5)
            ok &= (z - D[wc, uc]).abs() < tol
            cnt += ok.float()
    return cnt


def fill_colours(av, vis, mode: str = "mirror+diffuse", seen_min: float = 2.0, k: int = 8, iters: int = 400) -> dict:
    """Post-hoc fill of the dc colour of Gaussians seen in fewer than seen_min training views (see above).
    Unseen Gaussians also lose their view-dependent SH (shN = 0). Returns counts."""
    import torch
    with torch.no_grad():
        x = av.p["xyz"]
        seen = vis >= seen_min
        c = av.p["sh0"][:, 0].clone()
        n_unseen = int((~seen).sum())
        n_mirror = 0
        filled = seen.clone()
        if "mirror" in mode:
            j, dj = mirror_partner(x)
            use = (~seen) & seen[j] & (dj < 0.03)
            c[use] = c[j[use]]
            filled |= use
            n_mirror = int(use.sum())
        if "diffuse" in mode:
            nb = _knn(x, k)
            free = ~filled
            for _ in range(iters):                        # Jacobi iterations of the graph Laplace equation
                c[free] = c[nb[free]].mean(1)
        av.p["sh0"][:, 0] = c
        av.p["shN"][~seen] = 0
    return {"gaussians": len(x), "unseen": n_unseen, "mirror_filled": n_mirror, "mode": mode, "seen_min": seen_min}


def fill_eval(model: str, prep: str, out_dir: str, modes=("mirror", "diffuse", "mirror+diffuse"),
              holdout_cam: int | None = None, seen_min: float = 2.0, log=print) -> dict:
    """Load a trained model, compute visibility in its TRAINING views, apply each post-hoc fill and evaluate on the
    held-out views (same split as train()). Writes eval_<model>_fill_<mode>.json; returns {mode: heldout metrics}."""
    import torch
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(model, map_location="cpu")
    hc = ck.get("holdout_cam", holdout_cam) if holdout_cam is None else holdout_cam
    av0, z = load_model(model, prep)
    dev = av0.dev
    A = torch.tensor(z["A"], device=dev)
    imgs = torch.tensor(z["img"][..., ::-1].copy(), device=dev)
    masks = torch.tensor(z["mask"], device=dev).float()
    valid = torch.tensor(z["valid"], device=dev).float()
    Ks, VM = torch.tensor(z["K"], device=dev), torch.tensor(z["viewmat"], device=dev)
    frame, cam = z["frame"], z["cam"]
    held = set(ck["held_frames"])
    test = [v for v in range(len(frame)) if frame[v] in held or (hc is not None and cam[v] == hc)]
    train_v = [v for v in range(len(frame)) if v not in set(test)]
    vis = visibility(av0, A, Ks, VM, masks, valid, frame, train_v)
    stem = Path(model).stem.replace("model_", "")
    res = {"visibility": {"gaussians": len(vis), "seen_lt_1": int((vis < 1).sum()), "seen_lt_2": int((vis < 2).sum()),
                          "median_views": float(vis.median())}}
    base = evaluate(av0, A, imgs, masks, Ks, VM, frame, cam, test, out, tag=f"{stem}_nofill")
    res["none"] = base["heldout"]
    for m in modes:
        av, _ = load_model(model, prep)
        info = fill_colours(av, vis, m, seen_min=seen_min)
        r = evaluate(av, A, imgs, masks, Ks, VM, frame, cam, test, out, tag=f"{stem}_fill_{m.replace('+', '_')}")
        r["fill"] = info
        (out / f"eval_{stem}_fill_{m.replace('+', '_')}.json").write_text(json.dumps(r, indent=1))
        res[m] = r["heldout"]
        log(f"{stem} fill {m}: {json.dumps(r['heldout'])} ({info})")
    return res


# ----------------------------------------------------------------------------------------------------------- export
def load_model(path: str, prep: str, dev="cuda"):
    import torch
    ck = torch.load(path, map_location=dev)
    z = np.load(prep, allow_pickle=False)
    av = Avatar(z["v0"], z["faces"], z["weights"], ck["n_frames"], ck["n_cams"], n=len(ck["p"]["xyz"]),
                mode=ck["mode"], dev=dev)
    with torch.no_grad():
        for k, v in ck["p"].items():
            av.p[k].data = v.to(dev)
        av.weights = ck["weights"].to(dev)
    av.mlp.load_state_dict(ck["mlp"])
    av.held_frames = set(ck["held_frames"])
    return av, z


def export(model: str, prep: str, out_base: str, min_opacity: float = 0.05) -> dict:
    """-> out_base.bin + out_base.json. Layout (little endian), all offsets in the JSON:
      canonical: xyz f32 (N,3) | cov6 f32 (N,6) canonical covariance (xx xy xz yy yz zz) | rgba u8 (N,4) (SH dc
      colour, opacity) | joints u8 (N,4) | jweights u8 (N,4) (top-4 skinning, sum 255)
      frames: bones f32 (F,24,3,4) world from canonical (row-major 3x4), incl. the per-frame translation fix
      offsets: i8 (F,N,3) canonical non-rigid offsets * offset_scale."""
    import torch
    av, z = load_model(model, prep)
    with torch.no_grad():
        op = av.opacity()
        keep = (op > min_opacity).nonzero()[:, 0]
        n = len(keep)
        xyz = av.p["xyz"][keep].cpu().numpy().astype("<f4")
        R = _quat_to_rot(av.p["quats"][keep])
        S = torch.exp(av.p["scales"][keep])
        L = R * S[:, None, :]
        cov = (L @ L.transpose(1, 2)).cpu().numpy()
        cov6 = np.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1).astype("<f4")
        gain = torch.exp(av.p["gain"][0])
        rgb = ((av.p["sh0"][keep, 0] * C0 + 0.5) * gain).clamp(0, 1).cpu().numpy()
        rgba = np.concatenate([rgb, op[keep, None].cpu().numpy()], 1)
        rgba = np.round(rgba * 255).astype(np.uint8)
        w = av.weights[keep]
        tw, tj = w.topk(4, dim=1)
        tw = tw / tw.sum(1, keepdim=True)
        jw = np.round(tw.cpu().numpy() * 255).astype(np.int32)
        jw[:, 0] += 255 - jw.sum(1)
        A = torch.tensor(z["A"], device=av.dev).clone()
        nF = len(A)
        if av.mode == "full":
            A[:, :, :3, 3] += torch.stack([av.latent(f, "dt") for f in range(len(A))])[:, None, :]
        bones = A[:, :, :3, :].cpu().numpy().astype("<f4")
        offs = []
        for f in range(nF):
            o = av.offsets(f)
            offs.append(o[keep].cpu().numpy() if o is not None else np.zeros((n, 3)))
        offs = np.stack(offs)
    oscale = float(np.abs(offs).max()) / 127 if np.abs(offs).max() > 0 else 1.0
    offq = np.clip(np.round(offs / oscale), -127, 127).astype(np.int8)
    parts = [("xyz", xyz), ("cov6", cov6), ("rgba", rgba), ("joints", tj.cpu().numpy().astype(np.uint8)),
             ("jweights", jw.astype(np.uint8)), ("bones", bones), ("offsets", offq)]
    meta_in = json.loads(str(z["meta"]))
    hdr = {"version": "avatar4d/1", "count": n, "frames": nF, "rig_frame": z["rig_frame"].tolist(),
           "replay_fps": meta_in.get("fps"), "offset_scale": oscale, "frame": "replay world, metres, Z up",
           "licence": "SMPL-derived (non-commercial): local use only, never published",
           "layout": {}, "note": "world = sum_k w_k * bones[f][j_k] @ [xyz + offsets[f] * offset_scale; 1]"}
    buf, pos = [], 0
    for name, arr in parts:
        b = np.ascontiguousarray(arr).tobytes()
        pad = (-pos) % 4
        if pad:
            buf.append(b"\0" * pad)
            pos += pad
        hdr["layout"][name] = {"offset": pos, "bytes": len(b), "dtype": str(arr.dtype), "shape": list(arr.shape)}
        buf.append(b)
        pos += len(b)
    Path(out_base + ".bin").write_bytes(b"".join(buf))
    Path(out_base + ".json").write_text(json.dumps(hdr, indent=1))
    return {"gaussians": n, "frames": nF, "bytes": pos}
