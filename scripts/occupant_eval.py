"""Vehicle occupant faces on real vehicle clips: how often a face is found inside the vehicle box, its size, and the
bridge decision against the enrolled templates (the production occupant path: squirrel_guard.occupants on
FaceEngineV2 + AdaFace, decided by deploy/recorder/bridge/facematch.decide_any with its default thresholds).

Usage (clips and templates stay local, never in git):
  .venv/bin/python scripts/occupant_eval.py --clips DIR --enrolled enrolled.json [--fps 4] [--max-s 30] [--crops OUT]
  DIR: <frigate event id>[_a|_b].mp4 recording clips (Frigate /api/events/<id>/clip.mp4, or /api/<cam>/start/../end/..
       windows for long events); enrolled.json: {"templates": [{template_id, identity_id, label, kind, model, vec}]}
       (the recorder host face_templates rows).  --crops writes the best face crop per occupant group for a visual check
       (delete them afterwards).
Prints one line per clip and a summary; --json writes the per-clip numbers.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy/recorder/bridge"))

from squirrel_guard import faceid, faceid_v2, occupants  # noqa: E402
from squirrel_guard.detector import Detector  # noqa: E402
from squirrel_guard.vehicle import vehicle_label  # noqa: E402

import facematch  # noqa: E402

PACK = str(Path.home() / ".inspireface/ms/models/tunmxy--InspireFace/snapshots/master/Megatron")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--enrolled", required=True)
    ap.add_argument("--pack", default=PACK)
    ap.add_argument("--adaface", default=str(ROOT / "models/adaface_ir101_webface12m.onnx"))
    ap.add_argument("--detector", default=str(ROOT / "models/yolov8s-oiv7.onnx"))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--whole-frame", action="store_true", help="positive control on person clips: no vehicle box")
    ap.add_argument("--detect-level", type=int, default=640)
    ap.add_argument("--max-cpu-s", type=float, default=12.0)
    ap.add_argument("--crops")
    ap.add_argument("--json")
    a = ap.parse_args()
    enrolled = json.loads(Path(a.enrolled).read_text())["templates"]
    det = Detector(a.detector, conf=0.3, threads=4)
    eng = faceid_v2.FaceEngineV2(a.pack, detect_level=a.detect_level, expression=False, attributes=False)
    ada = faceid_v2.AdaFace(a.adaface, threads=4)
    occ = occupants.OccParams(det_px=a.detect_level, max_cpu_s=a.max_cpu_s)
    rows = []
    sizes = []
    for clip in sorted(Path(a.clips).glob(a.glob)):
        data = clip.read_bytes()
        if len(data) < 1000:
            continue
        t0 = time.time()
        smp = occupants.OccupantSampler(eng, faceid.AggParams(min_px=24, max_faces=96), occ, ada=ada)
        vh = []
        all_faces = []
        orig = eng.faces

        def spy(*args, **kw):
            out = orig(*args, **kw)
            all_faces.extend(out)
            return out
        eng.faces = spy

        def boxes(fr):
            if a.whole_frame:                    # positive control: the whole frame is the "cabin"
                return (0, 0, fr.shape[1], int(fr.shape[0] / occ.region_bottom))
            dd = det(fr)
            ds = [d for d in dd if vehicle_label(d.label)]
            if not ds:
                return None
            b = max(ds, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])).box
            vh.append(min(b[2] - b[0], b[3] - b[1]))
            return b, [d.box for d in dd if d.label.lower() in occupants.PERSON_LABELS]
        occupants.sample_clip(data, boxes, smp, fps=a.fps, max_s=a.max_s, max_frames=a.max_frames)
        eng.faces = orig
        r = smp.result()
        decs = []
        for k, o in enumerate(r["occupant_faces"]):
            d = facematch.decide_any(o, enrolled)
            decs.append({"status": d["status"], "sim": d.get("sim"), "engine": d.get("engine"), "reasons": d["reasons"],
                         "identity": d.get("identity"), "n_faces": o.get("n_faces"), "size_px": o.get("size_px"),
                         "quality": o.get("quality"), "consistency": o.get("consistency"),
                         "liveness": (o.get("liveness") or {}).get("score"),
                         "isf_sim": (d.get("shadow") or {}).get("sim")})
            if a.crops and o.get("best_crop_b64"):
                Path(a.crops).mkdir(parents=True, exist_ok=True)
                (Path(a.crops) / f"{clip.stem}_occ{k}.jpg").write_bytes(base64.b64decode(o["best_crop_b64"]))
        fs = [f.size for f in occupants.dedupe(all_faces)]
        sizes += [f.size for f in all_faces]
        row = {"clip": clip.stem, "frames": r["frames"], "frames_vehicle": r["frames_with_vehicle"],
               "vehicle_px_med": float(np.median(vh)) if vh else None, "tiles": r["tiles"], "faces_seen": r["faces_seen"],
               "face_px": [round(x, 1) for x in sorted(fs)][-8:], "groups": r["occupants_detected"],
               "cpu_s": r["cpu_s"], "wall_s": round(time.time() - t0, 1), "decisions": decs}
        rows.append(row)
        print(json.dumps(row), flush=True)
    n = len(rows)
    with_face = sum(1 for r in rows if r["faces_seen"])
    usable = sum(1 for r in rows if r["groups"])
    st = {}
    for r in rows:
        for d in r["decisions"]:
            st[d["status"]] = st.get(d["status"], 0) + 1
    s = np.asarray(sizes) if sizes else np.zeros(0)
    summ = {"clips": n, "clips_with_any_face": with_face, "clips_with_usable_face_(>=24px)": usable,
            "faces_detected": int(len(s)),
            "face_px_percentiles_10_50_90": [round(float(np.percentile(s, q)), 1) for q in (10, 50, 90)] if len(s) else None,
            "faces_>=32px": int((s >= 32).sum()), "decisions": st}
    print("SUMMARY", json.dumps(summ))
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
