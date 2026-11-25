#!/usr/bin/env python
"""Enhance one Frigate event for HUMAN VIEWING on the laptop GPU (RTX 4070): a lucky-imaging face still and a
BasicVSR++ close-up clip, both stamped ENHANCED, next to the camera's own pixels. Never used for recognition.

  ~/src/splat/.venv/bin/python scripts/enhance_event.py EVENT_ID [--out data/enhance]

Outputs in <out>/<event>/:
  face_still.png      camera frame (bicubic) | lucky-imaging composite | uncertainty map   (faces >= 12 px; when the
                      stack is `suspect`, only the camera frame is shown, with the reason)
  closeup.mp4         person crop: camera (bicubic) | BasicVSR++ (scripts/vsr_clip.py)
  report.json         face sizes, frames used, consistency/uncertainty, timings
Pose + AdaFace run as TensorRT fp16 engines (squirrel_guard/trt.py); falls back to onnxruntime CPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import lucky  # noqa: E402
from squirrel_guard.faceid_v2 import AdaFace  # noqa: E402
from squirrel_guard.faceseq import FaceSeq  # noqa: E402
from squirrel_guard.faceseq_agg import MIN_FACE_PX  # noqa: E402

M = Path.home() / ".cache/squirrel-guard/models"
STACK_MAX_PX = 40      # self-tests: stacking gains +1..+3.4 dB at 17-34 px; at 60-90 px it softens a moving face


def frigate(path: str) -> bytes:
    return subprocess.run(["ssh", "gpuserver", f"docker exec frigate curl -s -m 120 http://127.0.0.1:5000{path}"],
                          capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event", nargs="?", help="Frigate event id (or use --clip for a recording without an event)")
    ap.add_argument("--clip", help="a clip file instead of an event (people are found with zoomed tiles)")
    ap.add_argument("--name", help="output folder name with --clip")
    ap.add_argument("--out", default=str(ROOT / "data/enhance"))
    a = ap.parse_args()
    name = a.event or a.name or Path(a.clip).stem
    od = Path(a.out) / name
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.event:
        ev = json.loads(frigate(f"/api/events/{a.event}"))
        clip = od / "clip.mp4"
        if not clip.exists():
            clip.write_bytes(frigate(f"/api/events/{a.event}/clip.mp4"))
    else:
        ev, clip = {"label": "person", "camera": None, "data": {}}, Path(a.clip)
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    try:
        backend = "trt"
        fs = FaceSeq(str(M / "yolov8s-pose.onnx"), backend=backend)
        ada = AdaFace(M / "adaface_ir101_webface12m.onnx", backend=backend)
    except Exception as e:  # noqa: BLE001
        print("TensorRT unavailable, using CPU:", e)
        backend = "ort"
        fs = FaceSeq(str(M / "yolov8s-pose.onnx"))
        ada = AdaFace(M / "adaface_ir101_webface12m.onnx", threads=4)
    rep = {"event": a.event or str(clip), "label": ev.get("label"), "camera": ev.get("camera"), "frames": len(frames),
           "backend": backend}
    # start from the person nearest Frigate's box, in the first frame that has people
    H, W = frames[0].shape[:2]
    box_rel = (ev.get("data") or {}).get("box")
    start = None
    if box_rel:
        bx, by, bw, bh = box_rel
        cx, cy = (bx + bw / 2) * W, (by + bh / 2) * H
        for i, f in enumerate(frames[::2]):
            ps = fs.pose(f)
            if ps:
                p = min(ps, key=lambda q: abs((q.box[0] + q.box[2]) / 2 - cx) + abs((q.box[1] + q.box[3]) / 2 - cy))
                start = (2 * i, p.box)
                break
    else:   # no event box: zoomed 320 px tiles (like the sweeper) until someone is found
        for i in range(0, len(frames), max(1, int(fps // 2))):
            f, found = frames[i], []
            for ty in range(0, H - 319, 240):
                for tx in range(0, W - 319, 240):
                    for p in fs.pose(f[ty:ty + 320, tx:tx + 320]):
                        if p.score >= 0.5:
                            found.append((p.box[0] + tx, p.box[1] + ty, p.box[2] + tx, p.box[3] + ty))
            if found:
                b = max(found, key=lambda b: b[3] - b[1])
                start = (i, b)
                bx, by, bw, bh = b[0] / W, b[1] / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H
                break
        if start is None:
            bx, by, bw, bh = 0.4, 0.3, 0.2, 0.4
    obs = []
    if start:
        obs = fs.track(frames[start[0]:], fps, start[1], step=1)
        for o in obs:
            o.frame += start[0]
    usable = [o for o in obs if o.face_px >= MIN_FACE_PX]
    rep.update(faces_found=len(obs), faces_usable=len(usable),
               face_px_max=round(max((o.face_px for o in obs), default=0), 1))
    small = [o for o in usable if o.face_px <= STACK_MAX_PX]
    if len(usable) >= 4 and len(small) < 4:
        best = max(usable, key=lambda o: o.sharpness * min(1.0, 1 - abs(o.yaw_proxy)))
        X0, Y0 = (best.pts5.mean(0) - 0.85 * best.face_px).astype(int).clip(0)
        s = int(1.7 * best.face_px)
        cv2.imwrite(str(od / "face_still.png"), cv2.resize(frames[best.frame][Y0:Y0 + s, X0:X0 + s], (384, 384),
                                                          interpolation=cv2.INTER_CUBIC))
        rep["face_still"] = (f"face already {best.face_px:.0f} px: best single camera frame {best.frame} shown "
                             f"(stacking softens faces above {STACK_MAX_PX} px)")
    elif len(usable) >= 4:
        usable = small
        _, N = ada.embed([o.crop112 for o in usable])
        res = lucky.composite(frames, usable, N, scale=2)
        big = lambda x: cv2.resize(x, (384, 384), interpolation=cv2.INTER_NEAREST)
        um = cv2.applyColorMap(np.clip(res["uncertainty"] * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        if res["suspect"]:
            ref = big(res["reference_bicubic"])
            cv2.putText(ref, "stack unreliable: camera frame only", (6, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
            sheet = ref
        else:
            sheet = np.hstack([big(res["reference_bicubic"]), big(lucky.stamp(res["composite"])), big(um)])
        cv2.imwrite(str(od / "face_still.png"), sheet)
        rep["face_still"] = {k: res[k] for k in ("frames_used", "reference_frame", "consistency_db",
                                                  "face_uncertainty", "suspect")}
    else:
        rep["face_still"] = f"skipped: {len(usable)} faces >= {MIN_FACE_PX:.0f} px (need 4)"
    # close-up clip of the person with BasicVSR++
    cmd = [sys.executable, str(ROOT / "scripts/vsr_clip.py"), "--clip", str(clip), "--box",
           ",".join(str(int(v)) for v in ((bx * W, by * H, (bx + bw) * W, (by + bh) * H))), "--out", str(od / "vsr")]
    if not a.event and obs:   # follow the person (face-sequence track) instead of a fixed crop
        pb = [(o.frame / fps, o.person_box) for o in obs]
        bw = max(b[2] - b[0] for _, b in pb) / W           # crop sized for the person at their largest
        bh = max(b[3] - b[1] for _, b in pb) / H
        (od / "path.json").write_text(json.dumps({"box": [bx, by, bw, bh], "path_data": [
            [[(b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H], t] for t, b in pb]}))
        cmd += ["--path-json", str(od / "path.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    vdir = od / "vsr" / Path(clip).stem
    if (vdir / "compare.mp4").exists():
        (vdir / "compare.mp4").replace(od / "closeup.mp4")
        try:
            rep["closeup_selftest"] = json.loads((vdir / "selftest.json").read_text())
        except (OSError, ValueError):
            pass
    else:
        rep["closeup"] = "failed: " + r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "failed"
    rep["seconds"] = round(time.time() - t0, 1)
    (od / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
