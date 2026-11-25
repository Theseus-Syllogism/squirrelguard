#!/usr/bin/env python3
"""VGGT worker for make_splat.py `sfm --method vggt`: images -> per-frame camera, depth map and depth confidence.

Runs as a subprocess so the 1B-parameter model's memory is returned before gsplat starts, and so it can use a
different interpreter (--vggt-python in make_splat.py) if VGGT ever needs its own venv. Needs torch, torchvision,
Pillow, safetensors, huggingface_hub, einops and the VGGT source tree (https://github.com/facebookresearch/vggt,
unpacked at ~/src/vggt; nothing is installed from it, it goes on sys.path).

  python vggt_infer.py IMAGES_DIR OUT.npz [--repo ~/src/vggt] [--weights facebook/VGGT-1B] [--chunk 80]

VRAM: the laptop's 8 GB card usually has only ~3 GB free (the user's llama-server holds ~4.8 GB), and plain VGGT
(1.26B parameters with the point head, ~2.8 GB of weights even in bf16, plus activations) does not fit next to it.
So: the point-map and tracking heads are not loaded (depth + camera unprojection is the more accurate point source
per the paper); the aggregator (0.9B, the alternating frame/global-attention transformer) runs in bf16 with its
weights in pinned CPU memory, each block copied in just before it runs; its cached layer outputs go to CPU memory;
the camera head (0.2B) and depth head (DPT, 33M) then run in fp32 as upstream does, the depth head 2 frames at a
time. Peak VRAM is then activations, which grow with the frame count (global attention sees all frames at once).
Inputs are resized to 518 px on the long side (multiples of 14, VGGT's training resolution).

More frames than --chunk: the sequence is processed in chunks that all contain the same evenly spaced anchor frames
(anchor 0 first, so every chunk has the same world frame up to scale); each chunk is aligned to the first by a
similarity transform fitted on the anchors' unprojected depth points (same pixels in every chunk).

Output .npz: names, extrinsic (S,3,4 cam_from_world, OpenCV), intrinsic (S,3,3 at the network resolution),
depth (S,h,w), conf (S,h,w), rgb (S,h,w,3 uint8), meta (JSON: size, chunks, timings, peak VRAM).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_images(paths, long_side=518):
    import torch
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    w0, h0 = ims[0].size
    s = long_side / max(w0, h0)
    w = max(14, round(w0 * s / 14) * 14)
    h = max(14, round(h0 * s / 14) * 14)
    arr = np.stack([np.asarray(im.resize((w, h), Image.Resampling.BICUBIC)) for im in ims])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0, arr, (h, w)


def build_model(repo: Path, weights: str, dev: str):
    import torch
    sys.path.insert(0, str(repo))
    from vggt.models.vggt import VGGT
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    model = VGGT(enable_point=False, enable_track=False)
    wp = Path(weights) if Path(weights).exists() else Path(hf_hub_download(weights, "model.safetensors"))
    sd = load_file(str(wp))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    assert not missing, f"missing weights: {missing[:5]}"
    assert all(k.startswith(("point_head.", "track_head.")) for k in unexpected), unexpected[:5]
    model.eval()
    if dev == "cuda":
        model.aggregator.to(torch.bfloat16)
        for t in list(model.parameters()) + list(model.buffers()):   # stays on the CPU, streamed in by _OnDevice
            t.data = t.data.pin_memory()
    return model


class _OnDevice:
    """Temporarily point a module's parameters/buffers at device copies; the pinned CPU tensors stay where they
    are, so going back is free. No-op on CPU."""

    def __init__(self, mod, dev):
        self.items = list(mod.parameters()) + list(mod.buffers()) if dev == "cuda" else []
        self.dev = dev

    def __enter__(self):
        self.saved = [t.data for t in self.items]
        for t in self.items:
            t.data = t.data.to(self.dev, non_blocking=True)

    def __exit__(self, *exc):
        for t, d in zip(self.items, self.saved):
            t.data = d


def aggregate(agg, images, dev: str, fchunk: int = 8):
    """Same computation as vggt.models.aggregator.Aggregator.forward (eval mode, aa_block_size 1), streamed: the
    patch embedding (DINOv2 ViT-L) and each of the 24 frame + 24 global attention blocks (~38 MB each in bf16) are
    copied to the GPU just before they run; patch embedding and frame blocks run fchunk frames at a time; the cached
    layer outputs the heads need (S x P x 2048 each) are moved to CPU memory as they are produced."""
    import torch
    from vggt.models.aggregator import slice_expand_and_flatten
    B, S, _, H, W = images.shape
    d = images.device
    images = ((images - agg._resnet_mean.to(d)) / agg._resnet_std.to(d)).view(B * S, 3, H, W)
    patch = []
    with _OnDevice(agg.patch_embed, dev):
        for s in range(0, B * S, fchunk):
            pt = agg.patch_embed(images[s:s + fchunk])
            patch.append(pt["x_norm_patchtokens"] if isinstance(pt, dict) else pt)
    patch = torch.cat(patch)
    tokens = torch.cat([slice_expand_and_flatten(agg.camera_token.to(d), B, S),
                        slice_expand_and_flatten(agg.register_token.to(d), B, S), patch], dim=1)
    del patch
    pos = agg.position_getter(B * S, H // agg.patch_size, W // agg.patch_size, device=d)
    if agg.patch_start_idx > 0:
        pos = torch.cat([torch.zeros(B * S, agg.patch_start_idx, 2, device=d, dtype=pos.dtype), pos + 1], dim=1)
    _, P, C = tokens.shape
    out = []
    assert agg.aa_block_size == 1 and list(agg.aa_order) == ["frame", "global"]
    for i in range(agg.aa_block_num):
        tokens = tokens.view(B * S, P, C)
        with _OnDevice(agg.frame_blocks[i], dev):
            tokens = torch.cat([agg.frame_blocks[i](tokens[s:s + fchunk], pos=pos[s:s + fchunk])
                                for s in range(0, B * S, fchunk)])
        fr = tokens.view(B, S, P, C)
        with _OnDevice(agg.global_blocks[i], dev):
            tokens = agg.global_blocks[i](tokens.view(B, S * P, C), pos=pos.view(B, S * P, 2))
        if i in agg.cached_layer_indices:
            cat = torch.cat([fr, tokens.view(B, S, P, C)], dim=-1)
            out.append(cat.to("cpu") if dev == "cuda" else cat)
        else:
            out.append(None)
        del fr
    return out, agg.patch_start_idx


def run_pass(model, images, dev):
    """images (S,3,h,w) in [0,1] -> extrinsic (S,3,4), intrinsic (S,3,3), depth (S,h,w), conf (S,h,w) as numpy."""
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    x = images.to(dev)[None]
    S = x.shape[1]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            toks, psi = aggregate(model.aggregator, x, dev)
        with _OnDevice(model.camera_head, dev):
            pose = model.camera_head([None] * (len(toks) - 1) + [toks[-1][:, :, :1].to(dev).float()])[-1]
        depth, conf = [], []
        with _OnDevice(model.depth_head, dev):
            for s in range(0, S, 2):                # DPT works frame by frame: fp32 copies of 2 frames at a time
                sub = [t[:, s:s + 2].to(dev).float() if t is not None else None for t in toks]
                dd, cc = model.depth_head(sub, images=x[:, s:s + 2], patch_start_idx=psi, frames_chunk_size=None)
                depth.append(dd[0, ..., 0].cpu()); conf.append(cc[0].cpu())
                del sub
        del toks
        ext, intr = pose_encoding_to_extri_intri(pose, x.shape[-2:])
    return (ext[0].cpu().numpy().astype(np.float64), intr[0].cpu().numpy().astype(np.float64),
            torch.cat(depth).float().numpy(), torch.cat(conf).float().numpy())


def unproject(ext, K, depth, ys, xs):
    """World points of pixels (ys, xs) of one frame."""
    d = depth[ys, xs]
    cam = np.stack([(xs - K[0, 2]) / K[0, 0] * d, (ys - K[1, 2]) / K[1, 1] * d, d], 1)
    R, t = ext[:, :3], ext[:, 3]
    return (cam - t) @ R           # R^T (x - t)


def umeyama(src, dst):
    """s, R, t with dst ~ s R src + t (least squares similarity)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images"); ap.add_argument("out")
    ap.add_argument("--repo", default=str(Path.home() / "src/vggt"))
    ap.add_argument("--weights", default="facebook/VGGT-1B", help="HF repo id or a local model.safetensors")
    ap.add_argument("--chunk", type=int, default=80, help="max frames per VGGT pass")
    ap.add_argument("--anchors", type=int, default=0, help="shared anchor frames per chunk (default chunk/4)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu: fp32, slow, for testing only")
    a = ap.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation
    import torch
    dev = a.device
    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        sys.exit(f"no images in {a.images}")
    t0 = time.time()
    images, rgb, (h, w) = load_images(paths)
    model = build_model(Path(a.repo).expanduser(), a.weights, dev)
    t_load = time.time() - t0
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    n = len(paths)
    t1 = time.time()
    if n <= a.chunk:
        ext, intr, depth, conf = run_pass(model, images, dev)
        chunks = 1
    else:
        na = a.anchors or max(2, a.chunk // 4)
        anchors = [int(round(i)) for i in np.linspace(0, n - 1, na)]
        rest = [i for i in range(n) if i not in anchors]
        per = a.chunk - na
        groups = [rest[i:i + per] for i in range(0, len(rest), per)]
        ext = np.zeros((n, 3, 4)); intr = np.zeros((n, 3, 3))
        depth = np.zeros((n, h, w), np.float32); conf = np.zeros((n, h, w), np.float32)
        ref_pts = None
        ys, xs = np.mgrid[0:h:4, 0:w:4]
        ys, xs = ys.ravel(), xs.ravel()
        for gi, g in enumerate(groups):
            idx = anchors + g
            e, k, d, c = run_pass(model, images[idx], dev)
            # anchor points of this chunk (same pixels in every chunk) -> similarity to chunk 0's frame and scale
            pts = np.concatenate([unproject(e[j], k[j], d[j], ys, xs) for j in range(na)])
            cw = np.concatenate([c[j][ys, xs] for j in range(na)])
            if ref_pts is None:
                ref_pts, ref_c = pts, cw
                s, R, t = 1.0, np.eye(3), np.zeros(3)
            else:
                good = (cw > np.percentile(cw, 50)) & (ref_c > np.percentile(ref_c, 50))
                s, R, t = umeyama(pts[good], ref_pts[good])
                res = np.linalg.norm(s * pts[good] @ R.T + t - ref_pts[good], axis=1)
                print(f"  chunk {gi}: aligned on {good.sum()} anchor points, scale {s:.3f}, "
                      f"median residual {np.median(res):.4f} (scene median depth {np.median(d):.3f})", flush=True)
            for j, fi in enumerate(idx):
                if gi > 0 and fi in anchors:
                    continue                     # anchors keep chunk 0's prediction
                Rc, tc = e[j][:, :3], e[j][:, 3]
                # x_new = s R x + t  =>  s * cam = Rc R^T (x_new - t) + s tc ; depth scales by s
                Rn = Rc @ R.T
                ext[fi] = np.concatenate([Rn, (s * tc - Rn @ t)[:, None]], 1)
                intr[fi], depth[fi], conf[fi] = k[j], d[j] * s, c[j]
        chunks = len(groups)
    if dev == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0.0
    meta = {"frames": n, "size_hw": [h, w], "chunks": chunks, "load_s": round(t_load, 1),
            "inference_s": round(t_inf, 1), "peak_vram_gib": round(peak, 2)}
    np.savez(a.out, names=np.array([p.name for p in paths]), extrinsic=ext, intrinsic=intr,
             depth=depth.astype(np.float32), conf=conf.astype(np.float32), rgb=rgb, meta=json.dumps(meta))
    print("vggt:", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
