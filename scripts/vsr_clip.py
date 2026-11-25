#!/usr/bin/env python
"""Enhance the detection region of a Frigate alert/detection clip with multi-frame super-resolution (BasicVSR++ x4).

  ~/src/splat/.venv/bin/python scripts/vsr_clip.py EVENT_ID --out data/vsr          # fetches clip + track from gpuserver
  ~/src/splat/.venv/bin/python scripts/vsr_clip.py --clip c.mp4 --box x0,y0,x1,y1 --out data/vsr

The crop follows the object's track (Frigate path_data, interpolated; fixed box when there is none), cut from the
native-resolution recording. Outputs in <out>/<event>/:
  compare.mp4   left: the camera crop resized (bicubic, what the camera recorded)  right: BasicVSR++ output
                (default --prescale 0.5: antialiased x0.5 then x4 = x2 over camera pixels),
                stamped "ENHANCED (multi-frame SR) - not original pixels"
  best.png      the frame with the largest box, bicubic | enhanced
  selftest.json real-detail check on this clip's content: the model-input crop shrunk x4 (antialiased), restored by
                bicubic vs BasicVSR++, PSNR/SSIM against it (gain > 0 dB = detail genuinely recovered)
Run on the laptop GPU (RTX 4070); gpuserver's GPUs are full with the LLM. Outputs are local (data/ is git-ignored).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squirrel_guard import vsr  # noqa: E402

LABEL = "ENHANCED (multi-frame SR) - not original pixels"


def aa_resize(img: np.ndarray, scale: float) -> np.ndarray:
    """Antialiased bicubic resize (MATLAB-imresize-like): the degradation BasicVSR++ REDS4 was trained on."""
    t = torch.from_numpy(img).permute(2, 0, 1).float()[None]
    t = F.interpolate(t, scale_factor=scale, mode="bicubic", antialias=True, align_corners=False)
    return t[0].clamp(0, 255).round().byte().permute(1, 2, 0).numpy()


def frigate(path: str) -> bytes:
    return subprocess.run(["ssh", "gpuserver", f"docker exec frigate curl -s -m 60 http://127.0.0.1:5000{path}"],
                          capture_output=True, check=True).stdout


def read_frames(clip: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames, fps


def crop_track(frames, fps, ev, box_px, side_cap):
    """Per-frame crop windows (x0, y0) of a fixed size following the object."""
    H, W = frames[0].shape[:2]
    x0, y0, x1, y1 = box_px
    bw, bh = x1 - x0, y1 - y0
    side = int(min(side_cap, max(96, 1.35 * max(bw, bh))))
    side -= side % 4
    cx = np.full(len(frames), (x0 + x1) / 2.0)
    cy = np.full(len(frames), (y0 + y1) / 2.0)
    keep = np.ones(len(frames), bool)
    path = (ev or {}).get("data", {}).get("path_data") or []
    if len(path) >= 2 and ev:
        t0 = ev["start_time"]                                  # Frigate event start (incl. pre-capture) = clip start
        ts = np.array([p[1] for p in path]) - t0
        px = np.array([p[0][0] for p in path]) * W
        py = np.array([p[0][1] for p in path]) * H
        t = np.arange(len(frames)) / fps
        keep = (t >= ts[0] - 0.5) & (t <= ts[-1] + 0.5)
        cx, cy = np.interp(t, ts, px), np.interp(t, ts, py)
        k = max(1, int(fps // 4))                              # smooth the path a little
        cx = np.convolve(np.pad(cx, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
        cy = np.convolve(np.pad(cy, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    xs = np.clip((cx - side / 2).astype(int), 0, W - side)
    ys = np.clip((cy - side / 2).astype(int), 0, H - side)
    idx = np.nonzero(keep)[0]
    return [(int(i), int(xs[i]), int(ys[i])) for i in idx], side


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255 ** 2 / m)


def ssim(a, b):
    a, b = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float64) for x in (a, b))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    sa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    return float(np.mean(((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))))


def stamp(img):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, LABEL, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event", nargs="?")
    ap.add_argument("--clip")
    ap.add_argument("--box", help="x0,y0,x1,y1 in clip pixels (with --clip)")
    ap.add_argument("--path-json", help='with --clip: {"box": [x,y,w,h] relative, "path_data": [[[cx,cy],t_s],...]} '
                                        "(relative centres, seconds from clip start): the crop follows it")
    ap.add_argument("--out", default="data/vsr")
    ap.add_argument("--side-cap", type=int, default=320, help="max crop side (input px); output is 4x")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--prescale", type=float, default=0.5,
                    help="antialiased shrink before x4 SR (0.5 -> net x2 over camera pixels; 1.0 -> raw x4). The "
                         "shrink removes H.264 blocking/noise the model would otherwise sharpen into fake texture")
    a = ap.parse_args()

    ev = None
    if a.event:
        name = a.event
        ev = json.loads(frigate(f"/api/events/{a.event}"))
        od = Path(a.out) / name
        od.mkdir(parents=True, exist_ok=True)
        clip = od / "clip.mp4"
        if not clip.exists():
            clip.write_bytes(frigate(f"/api/events/{a.event}/clip.mp4"))
    else:
        clip = Path(a.clip)
        name = clip.stem
        if a.path_json:
            pj = json.loads(Path(a.path_json).read_text())
            ev = {"start_time": 0.0, "data": {"box": pj["box"], "path_data": pj["path_data"]}}
        od = Path(a.out) / name
        od.mkdir(parents=True, exist_ok=True)
    frames, fps = read_frames(clip)
    H, W = frames[0].shape[:2]
    if ev:
        bx, by, bw, bh = ev["data"]["box"]                     # relative x, y, w, h
        box_px = (bx * W, by * H, (bx + bw) * W, (by + bh) * H)
    else:
        box_px = tuple(float(v) for v in a.box.split(","))
    track, side = crop_track(frames, fps, ev, box_px, a.side_cap)
    min_side = int(np.ceil(64 / a.prescale / 4) * 4)        # BasicVSR++ needs >= 64 px input after the prescale
    if side < min_side:
        side = min_side
        track = [(i, int(np.clip(x - (min_side - side) // 2, 0, W - min_side)),
                  int(np.clip(y - (min_side - side) // 2, 0, H - min_side))) for i, x, y in track]
    if len(track) > a.max_frames:                              # keep the middle of the track
        s = (len(track) - a.max_frames) // 2
        track = track[s:s + a.max_frames]
    crops = [frames[i][y:y + side, x:x + side] for i, x, y in track]
    print(f"{clip.name}: {W}x{H} @ {fps:.1f} fps, {len(crops)} frames, crop {side}px -> {4 * side}px", flush=True)

    net = vsr.load()
    inp = crops if a.prescale == 1.0 else [aa_resize(c, a.prescale) for c in crops]
    sr = vsr.upscale(net, inp)
    o = sr[0].shape[0]
    bic = [cv2.resize(c, (o, o), interpolation=cv2.INTER_CUBIC) for c in crops]   # camera pixels, same output size

    vw = cv2.VideoWriter(str(od / "compare_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * o, o))
    for b, s in zip(bic, sr):
        vw.write(cv2.cvtColor(np.hstack([b, stamp(s)]), cv2.COLOR_RGB2BGR))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(od / "compare_raw.mp4"), "-c:v", "libx264",
                    "-crf", "16", "-pix_fmt", "yuv420p", str(od / "compare.mp4")], check=True)
    (od / "compare_raw.mp4").unlink()
    mid = len(sr) // 2
    cv2.imwrite(str(od / "best.png"), cv2.cvtColor(np.hstack([bic[mid], stamp(sr[mid])]), cv2.COLOR_RGB2BGR))

    # self-test (same degradation as the real run): truth = the model-input crop; shrink it x4 (antialiased),
    # restore with bicubic vs BasicVSR++, compare with the truth
    src = inp if inp[0].shape[0] >= 256 else crops          # x4 shrink must stay >= 64 px (model minimum)
    gt = [c[: c.shape[0] - c.shape[0] % 16, : c.shape[1] - c.shape[1] % 16] for c in src]
    gs = gt[0].shape[0]
    if gs < 256:                                            # x4 shrink would fall below the model's 64 px minimum
        res = {"frames": len(gt), "crop_px": gs, "selftest": "skipped: crop smaller than 256 px"}
        (od / "selftest.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    lo = [aa_resize(g, 0.25) for g in gt]
    rb = [cv2.resize(l, (gs, gs), interpolation=cv2.INTER_CUBIC) for l in lo]
    rs = vsr.upscale(net, lo)
    res = {
        "frames": len(gt), "crop_px": gs, "prescale": a.prescale, "output_px": o,
        "psnr_bicubic": round(float(np.mean([psnr(x, g) for x, g in zip(rb, gt)])), 2),
        "psnr_basicvsrpp": round(float(np.mean([psnr(x, g) for x, g in zip(rs, gt)])), 2),
        "ssim_bicubic": round(float(np.mean([ssim(x, g) for x, g in zip(rb, gt)])), 4),
        "ssim_basicvsrpp": round(float(np.mean([ssim(x, g) for x, g in zip(rs, gt)])), 4),
    }
    res["psnr_gain_db"] = round(res["psnr_basicvsrpp"] - res["psnr_bicubic"], 2)
    k = len(gt) // 2
    cv2.imwrite(str(od / "selftest.png"), cv2.cvtColor(np.hstack([rb[k], rs[k], gt[k]]), cv2.COLOR_RGB2BGR))
    (od / "selftest.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
