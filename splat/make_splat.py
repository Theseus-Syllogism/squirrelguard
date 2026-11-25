#!/usr/bin/env python3
"""Phone walk-around video of the yard -> camera poses (COLMAP via pycolmap) -> Gaussian splat (gsplat) -> .ply

Runs on the laptop's RTX 4070 (8 GB) in low-memory mode: packed rasterization, MCMC densification with a hard
cap on the number of Gaussians, downscaled training images, SH degree 2. Close games/LLMs that hold VRAM first.

  ~/src/splat/.venv/bin/python make_splat.py frames VIDEO WORK [--fps 2] [--max-frames 300] [--long-side 1600]
  ~/src/splat/.venv/bin/python make_splat.py sfm WORK                  # SIFT + sequential matching + mapping
  ~/src/splat/.venv/bin/python make_splat.py sfm WORK --method vggt [--refine]   # VGGT poses + depth (GPU)
  ~/src/splat/.venv/bin/python make_splat.py train WORK [--iters 30000] [--cap 1000000] [--factor 1]
  ~/src/splat/.venv/bin/python make_splat.py all VIDEO WORK ...        # all three
  ~/src/splat/.venv/bin/python make_splat.py compare WORK_A WORK_B --out DIR   # held-out views side by side

Output: WORK/splat.ply (standard 3DGS layout; the replay viewer loads it with ?splat=). COLMAP/VGGT output has no
real-world scale or orientation: align it to the camera calibration / ground plane afterwards.
Environment: ~/src/splat/.venv = Python 3.10, torch 2.4.1+cu124, gsplat 1.5.3+pt24cu124, pycolmap 4.2.1, einops
(VGGT runs from the source tree ~/src/vggt via vggt_infer.py in a subprocess; weights facebook/VGGT-1B, CC BY-NC).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------------------------------------ 1. frames
def cmd_frames(a) -> None:
    work = Path(a.work)
    raw, out = work / "raw", work / "images"
    for d in (raw, out):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", a.video, "-vf",
                    f"fps={a.fps},scale='if(gt(iw,ih),{a.long_side},-2)':'if(gt(iw,ih),-2,{a.long_side})'",
                    "-q:v", "2", str(raw / "%05d.jpg")], check=True)
    frames = sorted(raw.glob("*.jpg"))
    sharp = []
    for f in frames:
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
    sharp = np.array(sharp)
    # drop the blurriest frame in every window of 3 (motion blur ruins SfM and splats), then cap the count
    keep = [i for i in range(len(frames)) if sharp[i] >= np.min(sharp[max(0, i - 1):i + 2]) or len(frames) < 30]
    keep = [i for i in keep if sharp[i] >= np.percentile(sharp, 15)]
    if len(keep) > a.max_frames:
        keep = [keep[int(k)] for k in np.linspace(0, len(keep) - 1, a.max_frames)]
    for n, i in enumerate(keep):
        shutil.copy(frames[i], out / f"{n:05d}.jpg")
    shutil.rmtree(raw)
    print(f"frames: {len(frames)} extracted at {a.fps} fps, kept {len(keep)} sharp ones in {out}")


# --------------------------------------------------------------------------------------------- 2. structure
def cmd_sfm(a) -> None:
    if getattr(a, "method", "colmap") == "vggt":
        return cmd_sfm_vggt(a)
    import pycolmap
    work = Path(a.work)
    db, imgs, sparse = work / "database.db", work / "images", work / "sparse"
    db.unlink(missing_ok=True)
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir(parents=True)
    t = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"                       # phone lens: fx fy cx cy k1 k2 p1 p2
    pycolmap.extract_features(db, imgs, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    n = len(list(imgs.glob("*.jpg")))
    if n <= 120:
        pycolmap.match_exhaustive(db)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 15
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pairing)
    recs = pycolmap.incremental_mapping(db, imgs, sparse)
    if not recs:
        sys.exit("SfM failed: no reconstruction (too little overlap/texture, or too much blur)")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"sfm: {best.num_reg_images()}/{n} images registered, {best.num_points3D()} points, "
          f"mean reprojection {best.compute_mean_reprojection_error():.2f} px ({time.time() - t:.0f}s)")
    best.write(sparse)
    und = work / "undistorted"
    shutil.rmtree(und, ignore_errors=True)
    pycolmap.undistort_images(und, sparse, imgs)           # -> PINHOLE images + model for training
    print(f"undistorted into {und}")


def _vggt_points(ext, Kn, depth, conf, rgb, keep_pct: float, max_points: int):
    """Dense coloured points from VGGT depth maps: keep the top keep_pct % most confident pixels, sample max_points.
    ext (S,3,4) cam_from_world and Kn (S,3,3) at the network resolution of depth (S,h,w)."""
    thr = np.percentile(conf, 100 - keep_pct)
    f, y, x = np.nonzero(conf >= thr)
    if len(f) > max_points:
        pick = np.sort(np.random.default_rng(0).choice(len(f), max_points, replace=False))
        f, y, x = f[pick], y[pick], x[pick]
    d = depth[f, y, x]
    cam = np.stack([(x - Kn[f, 0, 2]) / Kn[f, 0, 0] * d, (y - Kn[f, 1, 2]) / Kn[f, 1, 1] * d, d], 1)
    R, t = ext[f, :, :3], ext[f, :, 3]
    xyz = np.einsum("nji,nj->ni", R, cam - t)                      # R^T (cam - t)
    return xyz, rgb[f, y, x]


def _write_text_model(path: Path, cams: dict, ext, ids: dict, names: list, cam_of: dict) -> None:
    """COLMAP text model (SIMPLE_PINHOLE cameras, poses, no points) that pycolmap.Reconstruction() can read.
    cams: camera id -> (W, H, f, cx, cy); cam_of: image name -> camera id (one shared camera, or one per image)."""
    import pycolmap
    path.mkdir(parents=True, exist_ok=True)
    (path / "cameras.txt").write_text("".join(f"{c} SIMPLE_PINHOLE {W} {H} {f!r} {cx!r} {cy!r}\n"
                                              for c, (W, H, f, cx, cy) in cams.items()))
    lines = []
    for n, e in zip(names, ext):
        qx, qy, qz, qw = pycolmap.Rotation3d(e[:, :3]).quat
        lines.append(f"{ids[n]} {qw!r} {qx!r} {qy!r} {qz!r} {e[0, 3]!r} {e[1, 3]!r} {e[2, 3]!r} {cam_of[n]} {n}\n\n")
    (path / "images.txt").write_text("".join(lines))
    (path / "points3D.txt").write_text("")


def _umeyama(src, dst):
    """s, R, t with dst ~ s R src + t."""
    ms, md = src.mean(0), dst.mean(0)
    U, S, Vt = np.linalg.svd((dst - md).T @ (src - ms) / len(src))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((src - ms) ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def cmd_sfm_vggt(a) -> None:
    """VGGT (feed-forward, seconds on the GPU) instead of incremental SfM; optional quick pycolmap BA (--refine).
    Writes the same sparse/ + undistorted/{sparse,images} layout as the COLMAP path, so `train` is unchanged."""
    import pycolmap
    work = Path(a.work)
    imgs, sparse, und = work / "images", work / "sparse", work / "undistorted"
    names = sorted(p.name for p in imgs.glob("*.jpg"))
    npz = work / "vggt.npz"
    t0 = time.time()
    subprocess.run([a.vggt_python, str(Path(__file__).resolve().parent / "vggt_infer.py"), str(imgs), str(npz),
                    "--repo", a.vggt_repo, "--weights", a.vggt_weights, "--chunk", str(a.vggt_chunk),
                    "--device", a.vggt_device], check=True)
    z = np.load(npz)
    meta = json.loads(str(z["meta"]))
    assert list(z["names"]) == names, "vggt.npz does not match WORK/images"
    ext, Kn, rgb = z["extrinsic"], z["intrinsic"], z["rgb"]
    depth, conf = z["depth"].astype(np.float32), z["conf"]
    h, w = depth.shape[1:]
    H0, W0 = cv2.imread(str(imgs / names[0])).shape[:2]
    sx, sy = W0 / w, H0 / h
    # one shared pinhole camera (phone video: fixed lens); VGGT puts the principal point at the centre
    fx = fy = float(np.mean([np.median(Kn[:, 0, 0]) * sx, np.median(Kn[:, 1, 1]) * sy]))  # square pixels
    # --camera-per-image: frames from different cameras (fixed bench/security cameras), each keeps VGGT's own focal
    f_each = [float(np.mean([Kn[i, 0, 0] * sx, Kn[i, 1, 1] * sy])) for i in range(len(names))]
    t_net = time.time() - t0
    print(f"vggt: {len(names)} frames at {w}x{h} in {t_net:.0f}s (model load {meta['load_s']}s, inference "
          f"{meta['inference_s']}s, {meta['chunks']} chunk(s), peak VRAM {meta['peak_vram_gib']} GiB); "
          f"focal {fx:.0f} px at {W0}x{H0}")

    db = work / "database.db"
    db.unlink(missing_ok=True)
    stats = {}
    if a.refine:
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_PINHOLE"
        reader.camera_params = f"{fx},{W0 / 2},{H0 / 2}"
        mode = pycolmap.CameraMode.PER_IMAGE if a.camera_per_image else pycolmap.CameraMode.SINGLE
        pycolmap.extract_features(db, imgs, camera_mode=mode, reader_options=reader)
        if len(names) <= 120:
            pycolmap.match_exhaustive(db)
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap, pairing.quadratic_overlap = 15, True
            pycolmap.match_sequential(db, pairing_options=pairing)
        d = pycolmap.Database.open(db)
        dbimgs = {i.name: i for i in d.read_all_images()}
        d.close()
        ids, cam_of = {n: dbimgs[n].image_id for n in names}, {n: dbimgs[n].camera_id for n in names}
    else:
        ids = {n: i + 1 for i, n in enumerate(names)}
        cam_of = {n: (i + 1 if a.camera_per_image else 1) for i, n in enumerate(names)}
    if not a.camera_per_image:
        cam_of = {n: cam_of[names[0]] for n in names}
    cams = {cam_of[n]: (W0, H0, f_each[i] if a.camera_per_image else fx, W0 / 2, H0 / 2) for i, n in enumerate(names)}
    model = work / "vggt_model"
    shutil.rmtree(model, ignore_errors=True)
    _write_text_model(model, cams, ext, ids, names, cam_of)
    rec = pycolmap.Reconstruction(model)

    if a.refine:
        tri = work / "vggt_tri"
        shutil.rmtree(tri, ignore_errors=True)
        tri.mkdir()
        rec = pycolmap.triangulate_points(rec, db, imgs, tri, clear_points=True)
        stats["tri_points"], stats["tri_reproj_px"] = rec.num_points3D(), rec.compute_mean_reprojection_error()
        c_before = np.array([rec.images[ids[n]].projection_center() for n in names])
        opts = pycolmap.BundleAdjustmentOptions()
        opts.refine_focal_length, opts.refine_principal_point, opts.refine_extra_params = True, False, False
        pycolmap.bundle_adjustment(rec, opts)
        # BA moves the gauge: bring the result back into VGGT's frame and scale so the depth maps still fit
        c_after = np.array([rec.images[ids[n]].projection_center() for n in names])
        s, R, t = _umeyama(c_after, c_before)
        rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
        c_after = np.array([rec.images[ids[n]].projection_center() for n in names])
        stats["ba_reproj_px"] = rec.compute_mean_reprojection_error()
        stats["ba_center_shift_rel"] = float(np.median(np.linalg.norm(c_after - c_before, axis=1)) /
                                             np.median(np.linalg.norm(c_before - c_before.mean(0), axis=1)))
        f_each = [float(rec.cameras[cam_of[n]].params[0]) for n in names]
        fx = fy = float(np.median(f_each))
        ext = np.array([rec.images[ids[n]].cam_from_world().matrix() for n in names])
        Kn = np.array([[[f / sx, 0, w / 2], [0, f / sy, h / 2], [0, 0, 1]] for f in f_each])
        print(f"refine: {stats['tri_points']} SIFT points triangulated on VGGT poses at "
              f"{stats['tri_reproj_px']:.2f} px -> bundle adjusted {stats['ba_reproj_px']:.2f} px, focal "
              f"{fx:.0f} px, camera centres moved {100 * stats['ba_center_shift_rel']:.1f} % of the scene radius")
    xyz, col = _vggt_points(ext, Kn, depth, conf, rgb, a.vggt_conf, a.vggt_points)
    for p, c in zip(xyz, col):
        rec.add_point3D(p, pycolmap.Track(), c)

    for d in (sparse, und):
        shutil.rmtree(d, ignore_errors=True)
    sparse.mkdir(parents=True)
    rec.write(sparse)
    (und / "sparse").mkdir(parents=True)
    rec.write(und / "sparse")
    (und / "images").mkdir()
    for n in names:                                      # pinhole already: "undistorted" = the frames themselves
        try:
            (und / "images" / n).hardlink_to(imgs / n)
        except OSError:
            shutil.copy(imgs / n, und / "images" / n)
    shutil.rmtree(model)
    secs = time.time() - t0
    summary = {"method": "vggt", "refine": a.refine, "frames": len(names), "points": rec.num_points3D(),
               "seconds": round(secs, 1), "vggt": meta, "focal_px": round(fx, 1), **stats}
    if a.camera_per_image:
        summary["focal_px_each"] = {n: round(f, 1) for n, f in zip(names, f_each)}
    (work / "sfm_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"sfm: {len(names)}/{len(names)} images posed by VGGT{' + BA' if a.refine else ''}, "
          f"{rec.num_points3D()} points ({secs:.0f}s)")


# ------------------------------------------------------------------------------------------------- 3. train
def _load(work: Path, factor: int):
    import pycolmap
    import torch
    rec = pycolmap.Reconstruction(work / "undistorted" / "sparse")
    imgs, Ks, viewmats = [], [], []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = cam.params[:4] if cam.model.name != "SIMPLE_PINHOLE" else (*cam.params[:1], *cam.params)
        im = cv2.imread(str(work / "undistorted" / "images" / img.name))[:, :, ::-1]
        if factor > 1:
            im = cv2.resize(im, (im.shape[1] // factor, im.shape[0] // factor), interpolation=cv2.INTER_AREA)
        s = im.shape[1] / cam.width
        K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]], np.float32)
        M = np.eye(4, dtype=np.float32)
        M[:3, :4] = img.cam_from_world().matrix()
        imgs.append(torch.from_numpy(np.ascontiguousarray(im)).float() / 255.0)
        Ks.append(torch.from_numpy(K))
        viewmats.append(torch.from_numpy(M))
    pts = np.array([p.xyz for p in rec.points3D.values()], np.float32)
    rgb = np.array([p.color for p in rec.points3D.values()], np.float32) / 255.0
    return imgs, Ks, viewmats, pts, rgb


def _ssim(x, y):
    """SSIM on (1, 3, H, W) images with an 11x11 Gaussian window (plain torch; no compiled extension needed)."""
    import torch
    import torch.nn.functional as F
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = (g[:, None] * g[None, :]) / g.sum() ** 2
    w = w.expand(3, 1, 11, 11)
    mu_x, mu_y = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()


def cmd_train(a) -> None:
    import torch
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy
    work, dev = Path(a.work), "cuda"
    torch.manual_seed(a.seed)                  # initial quaternions and MCMC noise (gsplat itself is not bit-exact)
    imgs, Ks, viewmats, pts, rgb = _load(work, a.factor)
    n_img = len(imgs)
    test_ids = set(range(0, n_img, 8)) if n_img >= 16 else set()
    if a.holdout:                              # name the held-out views (leave-one-camera-out tests with few views)
        names = sorted(p.name for p in (work / "undistorted" / "images").iterdir())
        test_ids = {i for i, n in enumerate(names) if any(h in n for h in a.holdout.split(","))}
    train_ids = [i for i in range(n_img) if i not in test_ids]
    H, W = imgs[0].shape[:2]
    print(f"train: {n_img} images at {W}x{H} (factor {a.factor}), {len(pts)} SfM points, cap {a.cap} Gaussians")

    # initial Gaussians from the SfM points: scale = mean distance to 3 nearest neighbours
    P = torch.from_numpy(pts).to(dev)
    nn = []
    for i in range(0, len(P), 512):            # blocks of 512 rows keep cdist small on an 8 GB card
        dd = torch.cdist(P[i:i + 512], P)
        nn.append(dd.topk(4, largest=False).values[:, 1:].mean(1))
    dist = torch.cat(nn).clamp_min(1e-4)
    sh_deg = a.sh_degree
    C0 = 0.28209479177387814
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(P.clone()),
        "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.rand(len(P), 4, device=dev)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((len(P),), 0.5, device=dev))),
        "sh0": torch.nn.Parameter(((torch.from_numpy(rgb).to(dev) - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(len(P), (sh_deg + 1) ** 2 - 1, 3, device=dev)),
    })
    extent = float((P - P.mean(0)).norm(dim=1).median()) * 1.1
    lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    strategy = MCMCStrategy(cap_max=a.cap, refine_start_iter=500, refine_stop_iter=int(a.iters * 0.8),
                            refine_every=100)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state()
    mean_sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / a.iters))
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(a.iters):
        i = int(rng.choice(train_ids))
        gt = imgs[i].to(dev)[None]
        sh_now = min(sh_deg, step // 1000)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        out, alphas, info = rasterization(
            params["means"], torch.nn.functional.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors, viewmats[i].to(dev)[None], Ks[i].to(dev)[None], W, H,
            sh_degree=sh_now, packed=True, near_plane=0.01, far_plane=1e10)
        img = out[..., :3].clamp(0, 1)
        l1 = (img - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - _ssim(img.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        if a.aniso_weight > 0:
            # anti-spike: penalise Gaussians whose axis ratio exceeds --max-ratio (after PhysGaussian's anisotropy
            # regulariser), so thin needles are replaced by several rounder splats
            # longest / MIDDLE axis: a needle (spike) has one long axis; flat discs on walls stay allowed
            ls = params["scales"].sort(dim=1).values
            ratio = torch.exp(ls[:, 2] - ls[:, 1])
            loss = loss + a.aniso_weight * torch.relu(ratio - a.max_ratio).mean()
        strategy.step_pre_backward(params, opts, state, step, info)
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        mean_sched.step()
        strategy.step_post_backward(params, opts, state, step, info, lr=mean_sched.get_last_lr()[0])
        if a.max_scale > 0:
            with torch.no_grad():                # hard cap on any axis (scene units), also after densification
                params["scales"].clamp_(max=math.log(a.max_scale))
        if step % 1000 == 0 or step == a.iters - 1:
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            print(f"  step {step:6d}  loss {loss.item():.4f}  gaussians {len(params['means']):,}  "
                  f"peak VRAM {mem:.2f} GiB  {time.time() - t0:.0f}s", flush=True)

    # evaluation on held-out views
    psnrs = []
    with torch.no_grad():
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        for i in sorted(test_ids):
            out, _, _ = rasterization(params["means"], torch.nn.functional.normalize(params["quats"], dim=-1),
                                      torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), colors,
                                      viewmats[i].to(dev)[None], Ks[i].to(dev)[None], W, H, sh_degree=sh_deg,
                                      packed=True)
            mse = ((out[0, ..., :3].clamp(0, 1) - imgs[i].to(dev)) ** 2).mean()
            psnrs.append(float(-10 * torch.log10(mse)))
            if i == min(test_ids):
                cv2.imwrite(str(work / "heldout_render.jpg"), (out[0, ..., :3].clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8))
                cv2.imwrite(str(work / "heldout_truth.jpg"), (imgs[i].numpy()[:, :, ::-1] * 255).astype(np.uint8))
    write_ply(work / "splat.ply", params, sh_deg)
    summary = {"images": n_img, "train": len(train_ids), "test": len(test_ids), "gaussians": len(params["means"]),
               "iters": a.iters, "factor": a.factor, "size": [W, H], "minutes": round((time.time() - t0) / 60, 1),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               "heldout_psnr": round(float(np.mean(psnrs)), 2) if psnrs else None}
    (work / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print("done:", json.dumps(summary))


def write_ply(path: Path, p, sh_deg: int) -> None:
    """Standard 3DGS .ply (x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*), as the INRIA code writes it."""
    from plyfile import PlyData, PlyElement
    m = p["means"].detach().cpu().numpy()
    dc = p["sh0"].detach().transpose(1, 2).reshape(len(m), -1).cpu().numpy()
    rest = p["shN"].detach().transpose(1, 2).reshape(len(m), -1).cpu().numpy()
    op = p["opacities"].detach().cpu().numpy()[:, None]
    sc = p["scales"].detach().cpu().numpy()
    q = torch_normalize(p["quats"].detach().cpu().numpy())
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(dc.shape[1])] +
             [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"] + [f"scale_{i}" for i in range(3)] +
             [f"rot_{i}" for i in range(4)])
    data = np.concatenate([m, np.zeros_like(m), dc, rest, op, sc, q], 1).astype(np.float32)
    el = np.empty(len(m), dtype=[(n, "f4") for n in names])
    el[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(el, "vertex")]).write(str(path))
    print(f"wrote {path} ({path.stat().st_size / 2 ** 20:.0f} MB)")


def read_ply(path: Path, dev: str = "cuda"):
    """Inverse of write_ply -> (params dict of tensors, sh degree)."""
    import torch
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    col = lambda pre: np.stack([v[n] for n in sorted((n for n in names if n.startswith(pre)),  # noqa: E731
                                                     key=lambda s: int(s.rsplit("_", 1)[1]))], 1)
    n_rest = sum(n.startswith("f_rest_") for n in names) // 3
    sh_deg = int(round(math.sqrt(n_rest + 1))) - 1
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)  # noqa: E731
    p = {"means": T(np.stack([v["x"], v["y"], v["z"]], 1)), "opacities": T(v["opacity"]),
         "scales": T(col("scale_")), "quats": T(col("rot_")),
         "sh0": T(col("f_dc_")).reshape(-1, 3, 1).transpose(1, 2),
         "shN": T(col("f_rest_")).reshape(-1, 3, n_rest).transpose(1, 2)}
    return p, sh_deg


def cmd_compare(a) -> None:
    """Held-out views (the ones `train` never saw) rendered from each WORK's own splat and its own camera for that
    frame. One PNG per view, one row per WORK: its training image of that frame | its render, PSNR in the label.
    The truth differs per WORK when the pose source differs: COLMAP's undistorted frames are cropped/rescaled."""
    import torch
    from gsplat import rasterization
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for w in a.works:
        work = Path(w)
        imgs, Ks, viewmats, _, _ = _load(work, a.factor)
        n = len(imgs)
        test = sorted(set(range(0, n, 8))) if n >= 16 else list(range(n))
        pick = [test[int(k)] for k in np.linspace(0, len(test) - 1, min(a.views, len(test)))]
        p, sh = read_ply(work / "splat.ply")
        H, W = imgs[0].shape[:2]
        with torch.no_grad():
            for i in pick:
                r, _, _ = rasterization(p["means"], torch.nn.functional.normalize(p["quats"], dim=-1),
                                        torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                        torch.cat([p["sh0"], p["shN"]], 1), viewmats[i].cuda()[None],
                                        Ks[i].cuda()[None], W, H, sh_degree=sh, packed=True)
                r = r[0, ..., :3].clamp(0, 1).cpu().numpy()
                psnr = -10 * math.log10(float(((r - imgs[i].numpy()) ** 2).mean()))
                rows.setdefault(i, []).append([(imgs[i].numpy(), f"{work.name}: real frame {i} (held out)"),
                                               (r, f"{work.name}: held-out render, PSNR {psnr:.2f}")])
        del p
        torch.cuda.empty_cache()
    tw = 640
    for i, grid in rows.items():
        lines = []
        for row in grid:
            tiles = []
            for im, label in row:
                t = (im[:, :, ::-1] * 255).astype(np.uint8)
                t = np.ascontiguousarray(cv2.resize(t, (tw, round(t.shape[0] * tw / t.shape[1])),
                                                    interpolation=cv2.INTER_AREA))
                cv2.rectangle(t, (0, 0), (t.shape[1], 24), (0, 0, 0), -1)
                cv2.putText(t, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(t)
            lines.append(np.concatenate(tiles, 1))
        hmax = max(x.shape[0] for x in lines)
        lines = [np.pad(x, ((0, hmax - x.shape[0]), (0, 0), (0, 0))) for x in lines]
        path = out / f"compare_view{i:03d}.png"
        cv2.imwrite(str(path), np.concatenate(lines, 0))
        print(f"wrote {path}")


def torch_normalize(q):
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("frames", "all"):
        s = sub.add_parser(name)
        s.add_argument("video"); s.add_argument("work")
        s.add_argument("--fps", type=float, default=2.0); s.add_argument("--max-frames", type=int, default=300)
        s.add_argument("--long-side", type=int, default=1600)
    sub.add_parser("sfm").add_argument("work")
    for name in ("sfm", "all"):
        s = sub.choices[name]
        s.add_argument("--method", choices=["colmap", "vggt"], default="colmap",
                       help="colmap: SIFT + incremental SfM (CPU, minutes); vggt: feed-forward VGGT (GPU, seconds)")
        s.add_argument("--refine", action="store_true",
                       help="vggt: SIFT matching + triangulation on VGGT's poses + one bundle adjustment")
        s.add_argument("--vggt-python", default=sys.executable, help="interpreter that runs vggt_infer.py")
        s.add_argument("--vggt-repo", default=str(Path.home() / "src/vggt"))
        s.add_argument("--vggt-weights", default="facebook/VGGT-1B",
                       help="HF repo id or local model.safetensors (VGGT-1B is CC BY-NC 4.0: non-commercial)")
        s.add_argument("--vggt-chunk", type=int, default=80, help="max frames per VGGT pass (anchored chunks above)")
        s.add_argument("--vggt-device", default="cuda", choices=["cuda", "cpu"])
        s.add_argument("--camera-per-image", action="store_true",
                       help="one camera per image (frames from different fixed cameras) instead of one shared lens")
        s.add_argument("--vggt-conf", type=float, default=50.0, help="keep this %% of most confident depth pixels")
        s.add_argument("--vggt-points", type=int, default=150_000, help="initial points sampled from the depth maps")
    c = sub.add_parser("compare", help="render the same held-out views from several trained WORK dirs into a PNG")
    c.add_argument("works", nargs="+"); c.add_argument("--out", required=True)
    c.add_argument("--views", type=int, default=3); c.add_argument("--factor", type=int, default=2)
    for name in ("train", "all"):
        s = sub.choices[name] if name in sub.choices else sub.add_parser(name)
        if name == "train":
            s.add_argument("work")
        s.add_argument("--iters", type=int, default=30000); s.add_argument("--cap", type=int, default=1_000_000)
        s.add_argument("--factor", type=int, default=1); s.add_argument("--sh-degree", type=int, default=2)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--holdout", default="", help="comma-separated parts of image names to keep out of training")
        s.add_argument("--aniso-weight", type=float, default=0.0,
                       help="anti-spike loss weight on axis ratios above --max-ratio (0 = off; try 0.1)")
        s.add_argument("--max-ratio", type=float, default=10.0, help="allowed longest/middle axis ratio (needles above it are penalised)")
        s.add_argument("--max-scale", type=float, default=0.0, help="clamp every axis to this size (0 = off)")
    a = ap.parse_args()
    if a.cmd in ("frames", "all"):
        cmd_frames(a)
    if a.cmd in ("sfm", "all"):
        cmd_sfm(a)
    if a.cmd in ("train", "all"):
        cmd_train(a)
    if a.cmd == "compare":
        cmd_compare(a)


if __name__ == "__main__":
    main()
