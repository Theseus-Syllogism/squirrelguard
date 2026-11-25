#!/usr/bin/env python
"""Package a calibration frame's room scene (splat + 3D objects) for the replay viewer, so the persons UI attaches it
to EVERY replay made with that calibration (fused and single-camera; persons-ui attach_scene).

  python3 scripts/publish_scene.py --frame "multicam-vggt:2026-11-08T05:20:v4+frontyard+testcam" \
      --splat demo/fused0516_splat.splat --objects demo/fused0516_objects.json --out STAGING
  then copy STAGING/* to the recorder host's /srv/camstack/replay/scenes/

Writes <slug>.splat, <slug>_objects.json (+ <slug>_objects/*.glb, file paths made absolute: /replay/scenes/...) and
the manifest <slug>.json. The splat and objects must be in that calibration's world frame (scripts/multicam_rig.py
--splat-work, splat/make_splat.py, scripts/rig_objects.py on the same frames).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path


def slug(frame: str) -> str:          # same rule as persons-ui scene_slug
    return re.sub(r"[^A-Za-z0-9]+", "-", str(frame)).strip("-")[:120]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--splat")
    ap.add_argument("--objects")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--name", help="file name instead of the frame's slug (per-clip builds: scene_worker.py)")
    ap.add_argument("--url-base", default="/replay/scenes/", help="URL of the directory the files are copied to")
    ap.add_argument("--extra", help="JSON object merged into the manifest")
    a = ap.parse_args()
    s, out = a.name or slug(a.frame), Path(a.out)
    base = a.url_base.rstrip("/") + "/"
    out.mkdir(parents=True, exist_ok=True)
    man = {"frame": a.frame, "built": time.strftime("%Y-%m-%d %H:%M"), "note": a.note}
    if a.splat:
        shutil.copyfile(a.splat, out / f"{s}.splat")
        man["splat"] = f"{s}.splat"
    if a.objects:
        src = Path(a.objects)
        o = json.loads(src.read_text())
        od = out / f"{s}_objects"
        od.mkdir(exist_ok=True)
        for ob in o.get("objects", []):
            f = src.parent / ob["file"]
            shutil.copyfile(f, od / f.name)
            ob["file"] = f"{base}{s}_objects/{f.name}"
        (out / f"{s}_objects.json").write_text(json.dumps(o, indent=1))
        man["objects"] = f"{s}_objects.json"
    if a.extra:
        man.update(json.loads(a.extra))
    (out / f"{s}.json").write_text(json.dumps(man, indent=1))
    print(s, man)


if __name__ == "__main__":
    main()
