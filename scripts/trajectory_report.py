#!/usr/bin/env python
"""Numbers for the trajectory model on synthetic walks (several noise seeds) and the viewer's sample.json.

  .venv/bin/python scripts/trajectory_report.py [--seeds 5] [--sample deploy/recorder/replay/sample.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
import synth_walk as sw  # noqa: E402
from test_trajectory import FPS, GAIT, OCCLUSION, evaluate  # noqa: E402
from squirrel_guard import trajectory as tj  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sample", default=str(ROOT / "deploy/recorder/replay/sample.json"))
    a = ap.parse_args()
    keys = ["rmse_m", "rmse_occlusion_m", "coverage95", "coverage95_occlusion", "speed_mae_mps", "footprints",
            "true_steps", "footprint_spacing_m", "footprint_spacing_expected_m", "pred_1s_err_m", "pred_1s_sd_m",
            "pred_1s_inside95", "pred_2s_err_m", "pred_2s_sd_m", "pred_2s_inside95"]
    for name, mk, occ in [("straight", sw.straight, None), ("curve+stop", sw.curve_with_stop, OCCLUSION),
                          ("leaving", sw.leaving, OCCLUSION)]:
        rows = [evaluate(mk(), seed=s, occlusion=occ) for s in range(a.seeds)]
        print(f"\n{name}  ({a.seeds} seeds, mean)")
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if not vals:
                continue
            print(f"  {k:30s} {np.mean([float(v) for v in vals]):.3f}")
        print("  metrics[0]:", json.dumps(rows[0]["metrics"]))

    # sample for the viewer: curve with a stop, 20 % dropped frames and a 1.5 s occlusion while moving
    cam = sw.wyze_camera()
    path = sw.curve_with_stop()
    trk, truth = sw.synth_track(path, cam, fps=FPS, occlusion=(9.0, 1.5), seed=3)
    res = tj.estimate(trk, FPS, sw.SIZE, cam, gait=GAIT, heights_px=True,
                      zones=[{"name": "porch", "polygon": [[-2, 3], [2, 3], [2, 6], [-2, 6]]}])
    res["skeleton"] = tj.skeleton3d(trk, cam, sw.SIZE)[res["frame0"]:res["frame0"] + len(res["t"])]
    res["truth"] = {"t": [round(float(v), 3) for v in truth["t"]], "xy": np.round(truth["xy"], 3).tolist(),
                    "note": "synthetic ground truth; absent for real recordings"}
    res["zones"] = [{"name": "porch", "polygon": [[-2, 3], [2, 3], [2, 6], [-2, 6]]}]
    res["source"] = "synthetic: tests/synth_walk.curve_with_stop, 1.75 m person, 112 steps/min, 3 px noise"
    out = Path(a.sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, separators=(",", ":")))
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB, {len(res['t'])} frames, "
          f"{sum(res['observed'])} observed, {len(res['footprints'])} footprints)")


if __name__ == "__main__":
    main()
