#!/usr/bin/env python
"""Rig-driven 3D body for a replay: SMPL dressed onto the replay's IK rig (squirrel_guard.smplfit).

  .venv/bin/python scripts/body3d_rig.py REPLAY.json [--out OUT.json] [--smpl models/smpl/SMPL_NEUTRAL.npz]
      [--mesh-step 1] [--iters 400]

REPLAY.json must carry "rig" (scripts/rigfit.py --write, replay_demo.py/multicam_demo.py --rig, or the live
profiler). Writes OUT.json = the replay + "body3d" (default: <replay stem>_body3d.json next to it; SMPL-derived:
git-ignored) and <name>_body.bin + <name>_body_faces.bin next to OUT (local only, never published).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import smplfit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay")
    ap.add_argument("--out")
    ap.add_argument("--smpl", default=str(ROOT / "models/smpl/SMPL_NEUTRAL.npz"))
    ap.add_argument("--mesh-step", type=int, default=1)
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()
    rp = Path(a.replay)
    tr = json.loads(rp.read_text())
    rg = tr.get("rig") or {}
    if not rg.get("markers"):
        sys.exit("replay has no rig (fit one first: scripts/rigfit.py --write or --rig)")
    out = Path(a.out) if a.out else rp.with_name(rp.stem + "_body3d.json")
    name = out.stem[:-len("_body3d")] if out.stem.endswith("_body3d") else out.stem
    n = len(tr["t"])
    body = smplfit.SMPLModel(a.smpl)
    r = smplfit.body_for_replay(body, tr, name, mesh_step=a.mesh_step, iters=a.iters, verbose=True)
    if r is None:
        sys.exit("no frame with enough rig markers")
    blk, vb, fb = r
    (out.parent / blk["mesh"]["vertices"]).write_bytes(vb)
    (out.parent / blk["mesh"]["faces"]).write_bytes(fb)
    tr["body3d"] = blk
    out.write_text(json.dumps(tr, separators=(",", ":")))
    st = blk["stats"]
    print(f"body: {st['replay_frames_with_body']}/{n} frames, marker error {st['marker_err_cm_median']} cm median "
          f"({st['marker_err_cm_p90']} p90), fit {st['fit_ms'] / 1000:.1f} s, betas {blk['betas'][:3]}...; "
          f"{len(vb) / 2 ** 20:.1f} MB vertices -> {out}")


if __name__ == "__main__":
    main()
