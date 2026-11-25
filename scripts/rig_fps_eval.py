#!/usr/bin/env python
"""Rig frame rate vs gait quality on a synthetic walker with KNOWN cadence (tests/test_fusion._walker: 1.1 s stride =
109.1 steps/min, 0.9 m/s), seen by 1 or 2 calibrated cameras with keypoint noise as a fraction of the box height.
For each rate: rig fit time, cadence error, steps found, step-time CV, stature.

  .venv/bin/python scripts/rig_fps_eval.py [--fps 8 12 16 20 25] [--seconds 8] [--noise 0.01 0.02] [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import rig  # noqa: E402
from tests.test_fusion import CAMS, _walker  # noqa: E402
from tests.test_rig import _cam  # noqa: E402

TRUE_CADENCE = 60.0 / 0.55


def views_for(X, cams, rng, noise_frac, drop=0.1):
    import torch
    out = []
    for cam in cams:
        uv, _ = rig.project(torch.tensor(X), cam, cam.size)
        uv = uv.numpy()
        bh = uv[..., 1].max(1) - uv[..., 1].min(1)
        uv = uv + rng.normal(0, 1, uv.shape) * (noise_frac * bh)[:, None, None]
        conf = np.where(rng.random(uv.shape[:2]) < drop, 0.0, 0.9)
        kp = np.concatenate([uv, conf[..., None]], -1)
        out.append({"cam": cam, "size": cam.size, "kp": kp, "box_h": bh})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, nargs="+", default=[8, 12, 16, 20, 25])
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.01, 0.02])
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    allc = [_cam(C, [0.0, 4.0, 0.9]) for C in CAMS.values()]
    rows = []
    for ncam in a.cams:
        for noise in a.noise:
            for fps in a.fps:
                res_s = []
                for seed in range(a.seeds):
                    rng = np.random.default_rng(seed)
                    n = int(round(a.seconds * fps))
                    X = _walker(n, 1 / fps, t0=0.37 * seed)
                    vs = views_for(X, allc[:ncam], rng, noise)
                    init = X[:, [rig.L_HIP, rig.R_HIP], :2].mean(1) + rng.normal(0, 0.3, (n, 2))
                    t0 = time.process_time()
                    w0 = time.time()
                    r = rig.fit(vs, n, 1 / fps, init_xy=init, stature0=1.70, verbose=False)
                    cpu, wall = time.process_time() - t0, time.time() - w0
                    g = rig.gait_metrics(r, 1 / fps)
                    res_s.append({"cpu_s": cpu, "wall_s": wall, "steps": g.get("steps_used", 0),
                                  "cadence": g.get("cadence_spm"), "cv": g.get("step_time_s_cv"),
                                  "step_len": g.get("step_length_m_median"), "stature": r["stature_m"]})
                cad = [x["cadence"] for x in res_s if x["cadence"]]
                row = {"cams": ncam, "noise_box": noise, "fps": fps, "frames": int(round(a.seconds * fps)),
                       "fit_cpu_s": round(float(np.mean([x["cpu_s"] for x in res_s])), 1),
                       "fit_wall_s": round(float(np.mean([x["wall_s"] for x in res_s])), 1),
                       "cadence_reported": f"{len(cad)}/{len(res_s)}",
                       "cadence_err_spm": round(float(np.mean([abs(c - TRUE_CADENCE) for c in cad])), 1) if cad else None,
                       "steps_used": [x["steps"] for x in res_s],
                       "step_time_cv": [x["cv"] for x in res_s],
                       "step_length_m": [x["step_len"] for x in res_s],
                       "stature_m": round(float(np.mean([x["stature"] for x in res_s])), 3)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({"true_cadence_spm": round(TRUE_CADENCE, 1), "true_step_length_m":
                                            round(0.9 * 0.55, 3), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
