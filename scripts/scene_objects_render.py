#!/usr/bin/env python3
"""Turntable contact sheets for scene_objects_*.py output (offscreen pyrender, EGL).

  PYOPENGL_PLATFORM=egl ~/src/pixarmesh/.venv/bin/python scripts/scene_objects_render.py OUT/objects.json [OUT/triposr.json ...]

Per JSON: one PNG per object (8 views at 45 deg steps, 15 deg elevation, a 1 m reference cube to the right for
scale) at <json stem>_<i>.png, plus <json stem>_scene.png (all objects placed in the world, with a 2.5 m ceiling
plane outline and the camera) next to the JSON.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

GLTF_TO_WORLD = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)   # rotate +90 deg about X


def load(path):
    m = trimesh.load(path, force="mesh")
    m.apply_transform(GLTF_TO_WORLD)
    return m


def look(eye, target):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, u, -f, eye
    return M


def render(meshes, eye, target, size=(400, 400), yfov=0.8):
    sc = pyrender.Scene(ambient_light=[0.35] * 3, bg_color=[255, 255, 255, 255])
    for m, col in meshes:
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=col, roughnessFactor=0.8, doubleSided=True)
        sc.add(pyrender.Mesh.from_trimesh(m, material=None if col is None else mat, smooth=False))
    cp = look(np.asarray(eye, float), np.asarray(target, float))
    sc.add(pyrender.PerspectiveCamera(yfov=yfov), pose=cp)
    sc.add(pyrender.DirectionalLight(intensity=3.0), pose=cp)
    r = pyrender.OffscreenRenderer(*size)
    img = r.render(sc)[0]; r.delete()
    return img


def main() -> None:
    for jp in map(Path, sys.argv[1:]):
        d = json.loads(jp.read_text()); base = jp.parent; allm = []
        for k, o in enumerate(d["objects"]):
            m = load(base / o["file"]); col = (0.55, 0.65, 0.8, 1.0)
            if getattr(m.visual, "kind", None) == "vertex": col = None
            ext = float(np.ptp(m.vertices, 0).max()); cube = trimesh.creation.box([1, 1, 1])
            cube.apply_translation([ext * 0.5 + 0.7, 0, 0.5 + m.vertices[:, 2].min()])
            views = []
            for az in range(0, 360, 45):
                a = np.radians(az); dist = max(ext, 1.0) * 2.6
                eye = [dist * np.cos(a), dist * np.sin(a), dist * 0.27]
                views.append(render([(m, col), (cube, (0.9, 0.5, 0.4, 0.35))], eye, [0.35, 0, 0]))
            sheet = np.concatenate([np.concatenate(views[:4], 1), np.concatenate(views[4:], 1)], 0)
            Image.fromarray(sheet).save(base / f"{jp.stem}_{k:02d}.png")
            mw = m.copy(); mw.apply_translation(o["position"]); allm.append((mw, col))
        floor = trimesh.creation.box([6, 6, 0.01]); floor.apply_translation([0, 2.5, 0])
        ceil = trimesh.creation.box([6, 6, 0.01]); ceil.apply_translation([0, 2.5, 2.5])
        cam = trimesh.creation.box([0.12, 0.08, 0.08]); cam.apply_translation([0, 0, 0.869])
        geo = allm + [(floor, (0.8, 0.8, 0.8, 1)), (ceil, (0.9, 0.9, 0.6, 0.25)), (cam, (1, 0.2, 0.2, 1))]
        views = [render(geo, e, [0, 2.5, 1.0], size=(640, 480), yfov=1.0) for e in ([-5, -3, 3.5], [5, -3, 3.5], [0, 1.0, 9])]
        Image.fromarray(np.concatenate(views, 1)).save(base / f"{jp.stem}_scene.png")
        print("rendered", jp)


if __name__ == "__main__":
    main()
