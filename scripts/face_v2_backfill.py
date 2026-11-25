#!/usr/bin/env python
"""Re-embed ENROLLED face templates with AdaFace (docs/faceid-v2.md): run the production face sampler (InspireFace
detection + 5-point alignment + the same aggregator) on each enrolment's source clip and print the v2 template as JSON
{event: v2 block}. The bridge/persons UI then stores them as face_templates rows with model = the AdaFace model name.

  .venv/bin/python scripts/face_v2_backfill.py --model models/adaface_ir101_webface12m.onnx CLIP.mp4 [...] > v2.json
Clip names must be sg_<frigate_event>.mp4 (or <frigate_event>.mp4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import faceid, faceid_v2  # noqa: E402
from squirrel_guard.kinematics import decode, track  # noqa: E402
from squirrel_guard.pose import PoseEstimator  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--model", default=str(ROOT / "models/adaface_ir101_webface12m.onnx"))
    ap.add_argument("--pack", default="Megatron")
    a = ap.parse_args()
    pose = PoseEstimator(str(ROOT / "models/yolov8s-pose.onnx"), threads=4)
    eng = faceid_v2.FaceEngineV2(a.pack)
    ada = faceid_v2.AdaFace(a.model, threads=4)
    out = {}
    for c in a.clips:
        ev = Path(c).stem.removeprefix("sg_")
        frames, _ = decode(Path(c).read_bytes(), 8.0, 1e9)
        s = faceid_v2.FaceSamplerV2(eng, faceid.AggParams(), ada=ada)

        def cap(gen):
            for fr in gen:
                if fr.shape[1] > 2560:
                    fr = cv2.resize(fr, (2560, round(fr.shape[0] * 2560 / fr.shape[1])), interpolation=cv2.INTER_AREA)
                yield fr
        track(cap(frames), pose, on_frame=s.on_frame)
        r = s.result()
        out[ev] = r.get("v2")
        print(ev, "v2" if r.get("v2") else "no v2", (r.get("v2") or {}).get("n_faces"), file=sys.stderr, flush=True)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
