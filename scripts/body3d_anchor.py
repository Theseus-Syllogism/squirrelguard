#!/usr/bin/env python
"""Re-anchor a WHAM body3d block (scripts/body3d_wham.py) onto a replay's own trajectory and merge it in.

  python scripts/body3d_anchor.py BODY3D.json REPLAY.json [--out OUT.json] [--foot-z 0.08]

For when WHAM's single-camera placement is unreliable (furniture hiding the feet, short baseline) but the replay
has a better position, e.g. multi-camera triangulation (scripts/multicam_demo.py). Per frame the whole body (joints
and mesh root) is translated so the hip midpoint sits over the replay's smoothed position and the lower ankle is at
--foot-z above the floor; WHAM's pose, shape and orientation are kept. Frames are matched by index (body3d_wham
samples the replay's own t). Writes REPLAY.json + the anchored "body3d" block to OUT (default <replay>_body3d.json:
SMPL-derived, git-ignored, never published); mesh files are referenced by name, so the .bin files must sit next
to it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("body3d"); ap.add_argument("replay")
    ap.add_argument("--foot-z", type=float, default=0.08)
    ap.add_argument("--out")
    a = ap.parse_args()
    b = json.loads(Path(a.body3d).read_text())["body3d"]
    rp = Path(a.replay)
    tr = json.loads(rp.read_text())
    mean = tr["mean"]
    if len(b["joints"]) != len(mean):
        raise SystemExit(f"frame counts differ: body3d {len(b['joints'])} vs replay {len(mean)}")
    shifts = []
    for i, J in enumerate(b["joints"]):
        if not J:
            continue
        P = np.array([[np.nan] * 3 if j is None else j for j in J], float)
        hip = np.nanmean(P[[11, 12]], 0)
        ank = np.nanmin(P[[15, 16], 2])
        if not (np.isfinite(hip).all() and np.isfinite(ank)):
            continue
        dlt = np.array([mean[i][0] - hip[0], mean[i][1] - hip[1], a.foot_z - ank])
        shifts.append(np.linalg.norm(dlt))
        b["joints"][i] = [None if j is None else [round(float(v), 3) for v in np.array(j) + dlt] for j in J]
        if b.get("root") and b["root"][i] is not None:
            b["root"][i] = [round(float(v), 3) for v in np.array(b["root"][i]) + dlt]
    b["anchor"] = {"method": "scripts/body3d_anchor.py: translated onto the replay trajectory, lower ankle at "
                             f"{a.foot_z} m", "frames": len(shifts),
                   "shift_m_median": round(float(np.median(shifts)), 2) if shifts else None}
    tr["body3d"] = b
    out = Path(a.out) if a.out else rp.with_name(rp.stem + "_body3d.json")
    out.write_text(json.dumps(tr, separators=(",", ":")))
    print(f"anchored {len(shifts)} frames (median shift {b['anchor']['shift_m_median']} m) -> {out}")


if __name__ == "__main__":
    main()
