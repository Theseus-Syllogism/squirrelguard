#!/usr/bin/env python
"""One person-free frame per camera from a visit's recordings, for a per-clip room scene (scripts/scene_worker.py).

  ~/src/pixarmesh/.venv/bin/python scripts/scene_frames.py OUT --clip v4=V4.mp4 --clip frontyard=FY.mp4 \
      [--samples 12] [--size v4=2560x1440 ...] [--report OUT/frames.json]

For each camera, --samples frames spread over its clip are searched for people (Grounding DINO tiny, prompt
"person."; boxes grown by 15 %). If some frame has no person, the one nearest the clip's middle is used as is.
Otherwise the least covered frame is used and its person boxes are filled with the per-pixel median of the other
frames that show the same view (mean grey difference outside people < --max-diff; a camera nudged during the visit
is not blended in) where they are not inside a person box; pixels left uncovered are reported as person_px (the
splat/objects see part of a person there). first_last_diff (grey levels) hints that the view changed in the clip. Writes
OUT/<cam>.jpg at the calibration's native size (--size; the clip is resized if it differs) and a JSON report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def sample(path: str, n: int):
    if path.endswith(".jpg"):                     # a single still (camera without a recording of the visit)
        f = cv2.imread(path)
        return [f] if f is not None else []
    cap = cv2.VideoCapture(path)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out = []
    if tot <= 0:                                  # unknown length: read all, keep n evenly
        allf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            allf.append(f)
        idx = np.linspace(0, len(allf) - 1, min(n, len(allf))).round().astype(int) if allf else []
        return [allf[i] for i in idx]
    for i in np.linspace(tot * 0.02, tot * 0.98 - 1, n).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--clip", action="append", required=True, help="CAM=clip.mp4")
    ap.add_argument("--size", action="append", default=[], help="CAM=WxH native calibration size")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--box-thr", type=float, default=0.3)
    ap.add_argument("--max-diff", type=float, default=10.0,
                    help="mean grey-level difference outside people for a frame to count as the same view")
    ap.add_argument("--report")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = {k: tuple(int(v) for v in s.lower().split("x")) for k, s in (t.split("=") for t in a.size)}
    gid = "IDEA-Research/grounding-dino-tiny"
    gp = AutoProcessor.from_pretrained(gid)
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(gid).to(a.device).eval()
    report = {}
    t0 = time.time()
    for spec in a.clip:
        cam, path = spec.split("=", 1)
        frames = sample(path, a.samples)
        if not frames:
            raise SystemExit(f"{cam}: no frames decoded from {path}")
        frames = [f for f in frames if float(f.std()) >= 8]     # black/blank frames (lens covered, stream down)
        if not frames:
            report[cam] = {"frames": 0, "method": "blank: no usable frame", "person_px": None, "first_last_diff": None}
            print(cam, "blank: skipped")
            continue
        if cam in sizes and frames[0].shape[1::-1] != sizes[cam]:
            frames = [cv2.resize(f, sizes[cam], interpolation=cv2.INTER_AREA) for f in frames]
        h, w = frames[0].shape[:2]
        masks = []
        for f in frames:
            pil = Image.fromarray(f[:, :, ::-1])
            with torch.no_grad():
                inp = gp(images=pil, text="person.", return_tensors="pt").to(a.device)
                res = gp.post_process_grounded_object_detection(gm(**inp), inp.input_ids, threshold=a.box_thr,
                                                                text_threshold=0.25, target_sizes=[(h, w)])[0]
            m = np.zeros((h, w), bool)
            for b in res["boxes"].cpu().numpy():
                bw, bh = b[2] - b[0], b[3] - b[1]
                x0, y0 = int(max(0, b[0] - 0.15 * bw)), int(max(0, b[1] - 0.15 * bh))
                x1, y1 = int(min(w, b[2] + 0.15 * bw)), int(min(h, b[3] + 0.15 * bh))
                if bw * bh < 0.9 * w * h:              # a box over the whole frame is a false "person"
                    m[y0:y1, x0:x1] = True
            masks.append(m)
        cover = [float(m.mean()) for m in masks]
        free = [i for i, c in enumerate(cover) if c == 0]
        if free:
            mid = len(frames) / 2
            k = min(free, key=lambda i: abs(i - mid))
            img, how, left = frames[k], f"person-free frame {k + 1}/{len(frames)}", 0.0
        else:
            # base = the least covered frame; its person boxes are filled with the per-pixel median of the OTHER
            # frames that show the same view there (a camera nudged during the visit must not ghost the room)
            k = int(np.argmin(cover))
            img, base_m = frames[k].copy(), masks[k]
            g = lambda f: cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32)
            gm_ = lambda m: cv2.resize(m.astype(np.uint8), (320, 180), interpolation=cv2.INTER_NEAREST) > 0
            gb, mb = g(frames[k]), gm_(base_m)
            same = []
            for j, f in enumerate(frames):
                if j == k:
                    continue
                both = ~mb & ~gm_(masks[j])
                if both.mean() > 0.2 and float(np.abs(g(f) - gb)[both].mean()) < a.max_diff:
                    same.append(j)
            ys, xs = np.nonzero(base_m)
            left = float(base_m.mean())
            if same and len(ys):
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                blk = np.stack([frames[j][y0:y1, x0:x1] for j in same]).astype(np.float32)
                mm = np.stack([masks[j][y0:y1, x0:x1] for j in same])
                blk[np.broadcast_to(mm[..., None], blk.shape)] = np.nan
                med = np.nanmedian(blk, axis=0)
                fill = base_m[y0:y1, x0:x1] & np.isfinite(med).all(-1)
                img[y0:y1, x0:x1][fill] = med[fill].clip(0, 255).astype(np.uint8)
                left = float((base_m[y0:y1, x0:x1] & ~fill).sum() / (h * w))
            how = f"frame {k + 1}/{len(frames)}, person boxes filled from {len(same)} matching frames"
        stable = None
        if len(frames) > 1:                            # did the view itself change during the clip? (camera moved)
            g0 = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32)
            g1 = cv2.resize(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32)
            free = ~(cv2.resize((masks[0] | masks[-1]).astype(np.uint8), (320, 180),
                                interpolation=cv2.INTER_NEAREST) > 0)
            stable = round(float(np.abs(g0 - g1)[free].mean()), 1) if free.mean() > 0.2 else None
        cv2.imwrite(str(out / f"{cam}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        report[cam] = {"frames": len(frames), "person_cover": [round(c, 3) for c in cover], "method": how,
                       "person_px": round(left, 4), "size": [w, h],
                       "first_last_diff": stable}
        print(cam, json.dumps(report[cam]))
    report["_seconds"] = round(time.time() - t0, 1)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
