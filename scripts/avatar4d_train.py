#!/usr/bin/env python3
"""4D Gaussian-splat avatar from the two-camera demo clip (squirrel_guard.avatar4d; method, limits, licence there).

  # 1. prep (project venv: CPU torch, cv2, onnxruntime)
  .venv/bin/python scripts/avatar4d_train.py prep --clip v4=V4.mp4 --clip frontyard=FY.mp4 \
      --calib v4=calib/v4.json --calib frontyard=calib/frontyard.json \
      --replay deploy/recorder/replay/demo/multicam.json [--shift frontyard=-15] --out AV/prep.npz
  # 2. optional SAM masks (~/src/pixarmesh/.venv: transformers SAM ViT-B)
  ~/src/pixarmesh/.venv/bin/python scripts/avatar4d_train.py masks --prep AV/prep.npz
  # 3. train + evaluate (~/src/splat/.venv: CUDA torch + gsplat); --holdout-cam 1 trains on v4 only
  ~/src/splat/.venv/bin/python scripts/avatar4d_train.py train --prep AV/prep.npz --out AV/run_frames --iters 8000
  ~/src/splat/.venv/bin/python scripts/avatar4d_train.py train ... --mode smplcolor        # baseline
  # 3b. unseen-side fill: post-hoc on a trained model, or as a training prior
  ~/src/splat/.venv/bin/python scripts/avatar4d_train.py fill --prep AV/prep.npz --model AV/run_cam/model_full.pt --out AV/fill --holdout-cam 1
  ~/src/splat/.venv/bin/python scripts/avatar4d_train.py train ... --holdout-cam 1 --fill-reg 1.0 --tag full_fillreg
  # 4. browser export (multicam_avatar.bin/.json), contact sheet, turntable
  ~/src/splat/.venv/bin/python scripts/avatar4d_train.py export --prep AV/prep.npz --model AV/run_frames/model_full.pt --out AV/multicam_avatar
  ~/src/splat/.venv/bin/python scripts/avatar4d_train.py renders --prep AV/prep.npz --run AV/run_frames --out AV/renders

Outputs are SMPL-derived (non-commercial): keep them under ~/.cache/squirrel-guard-test/avatar/, never in git.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from squirrel_guard import avatar4d  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _kv(items):
    return dict(x.split("=", 1) for x in items or [])


def cmd_prep(a):
    shifts = {k: int(v) for k, v in _kv(a.shift).items()} or None
    r = avatar4d.prepare(_kv(a.clip), _kv(a.calib), a.replay, a.out, a.smpl, shifts=shifts, fps=a.fps,
                         iters=a.fit_iters, seg_model=a.seg_model)
    print(json.dumps(r))


def cmd_masks(a):
    print(json.dumps(avatar4d.sam_masks(a.prep)))


def cmd_train(a):
    avatar4d.train(a.prep, a.out, iters=a.iters, n_gauss=a.gaussians, holdout_every=a.holdout_every,
                   holdout_cam=a.holdout_cam, mode=a.mode, seed=a.seed, fill_reg=a.fill_reg, tag=a.tag)


def cmd_fill(a):
    print(json.dumps(avatar4d.fill_eval(a.model, a.prep, a.out, modes=a.modes.split(","), holdout_cam=a.holdout_cam,
                                        seen_min=a.seen_min), indent=1))


def cmd_export(a):
    print(json.dumps(avatar4d.export(a.model, a.prep, a.out)))


def cmd_renders(a):
    """Contact sheet (GT | avatar | baseline) of held-out views, and a turntable + walk novel-view sequence."""
    import cv2
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Path(a.run)
    ev = json.loads((run / "eval_full.json").read_text())
    z = np.load(a.prep, allow_pickle=False)
    per = ev["per_view"]
    pick = per[:: max(1, len(per) // a.sheet_rows)][: a.sheet_rows]
    rows = []
    for p in pick:
        v = p["view"]
        gt = (z["img"][v] * z["mask"][v][..., None]).astype(np.uint8)
        ren = np.load(run / f"render_full_{v:04d}.npy")[:, :, ::-1]
        bp = run / f"render_smplcolor_{v:04d}.npy"
        base = np.load(bp)[:, :, ::-1] if bp.exists() else np.zeros_like(ren)
        row = np.hstack([gt, ren, base])
        cv2.putText(row, f"view {v} cam{p['cam']} f{p['frame']} PSNR {p['psnr_person']:.1f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(cv2.resize(row, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cols = 2
    while len(rows) % cols:
        rows.append(np.zeros_like(rows[0]))
    sheet = np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)])
    cv2.imwrite(str(out / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # turntable at the middle frame + orbiting camera following the walk
    av, _ = avatar4d.load_model(str(run / "model_full.pt"), a.prep)
    A = torch.tensor(z["A"], device="cuda")
    nF = len(A)
    W, H = 512, 640
    K = torch.tensor([[700.0, 0, W / 2], [0, 700.0, H / 2], [0, 0, 1]], device="cuda")
    tdir = out / "turntable"
    tdir.mkdir(exist_ok=True)
    n = a.turntable_frames
    with torch.no_grad():
        for i in range(n):
            f = (nF // 2) if i < n // 2 else int((i - n // 2) * (nF - 1) / max(1, n - n // 2 - 1))
            mean, _, _ = av.world(A[f], f)
            c = mean.mean(0)
            ang = 2 * math.pi * i / (n // 2)
            eye = c + torch.tensor([3.2 * math.cos(ang), 3.2 * math.sin(ang), 0.3], device="cuda")
            fwd = torch.nn.functional.normalize(c - eye, dim=0)
            right = torch.nn.functional.normalize(torch.linalg.cross(fwd, torch.tensor([0, 0, 1.0], device="cuda")), dim=0)
            down = torch.linalg.cross(fwd, right)
            Rcw = torch.stack([right, down, fwd])                       # rows: camera axes in world (x right, y down, z fwd)
            vm = torch.eye(4, device="cuda")
            vm[:3, :3] = Rcw
            vm[:3, 3] = -Rcw @ eye
            img, _, _ = av.render(A[f], f, vm, K, W=W, H=H, cam=0)
            im = (img.cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite(str(tdir / f"{i:04d}.png"), im)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", str(tdir / "%04d.png"),
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", str(out / "turntable.mp4")], check=False)
    print(json.dumps({"sheet": str(out / "contact_sheet.jpg"), "turntable": str(out / "turntable.mp4")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--clip", action="append", required=True, help="name=path (first = primary)")
    p.add_argument("--calib", action="append", required=True, help="name=geometry.Camera json")
    p.add_argument("--replay", required=True)
    p.add_argument("--shift", action="append", help="name=frames (secondary = primary + shift); default from replay")
    p.add_argument("--smpl", default=str(ROOT / "models/smpl/SMPL_NEUTRAL.npz"))
    p.add_argument("--seg-model", default=str(ROOT / "models/yolov8s-seg-320.onnx"))
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--fit-iters", type=int, default=400)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("masks")
    p.add_argument("--prep", required=True)
    p.set_defaults(fn=cmd_masks)
    p = sub.add_parser("train")
    p.add_argument("--prep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--gaussians", type=int, default=40000)
    p.add_argument("--holdout-every", type=int, default=8)
    p.add_argument("--holdout-cam", type=int, default=None, help="camera index held out entirely (frames all trained)")
    p.add_argument("--mode", choices=["full", "smplcolor"], default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fill-reg", type=float, default=0.0, help="visibility-weighted mirror/neighbour colour prior")
    p.add_argument("--tag", help="name for model_<tag>.pt / eval_<tag>.json (default: the mode)")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("fill", help="post-hoc unseen-side colour fill of a trained model, evaluated on held-out views")
    p.add_argument("--prep", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--modes", default="mirror,diffuse,mirror+diffuse")
    p.add_argument("--holdout-cam", type=int, default=None, help="for checkpoints saved before holdout_cam was stored")
    p.add_argument("--seen-min", type=float, default=2.0)
    p.set_defaults(fn=cmd_fill)
    p = sub.add_parser("export")
    p.add_argument("--prep", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="output base path (writes .bin and .json)")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("renders")
    p.add_argument("--prep", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sheet-rows", type=int, default=8)
    p.add_argument("--turntable-frames", type=int, default=96)
    p.set_defaults(fn=cmd_renders)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
