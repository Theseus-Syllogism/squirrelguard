"""Blender 5.3 (native Gaussian splats): render the site scene from one or more cameras.

  ~/.local/opt/blender-5.3.0-linux-x64/blender -b --factory-startup --python scripts/blender/site_render.py -- JOB.json

JOB.json:
  {"splats": ["data/site/site_splat.ply", ...],          # 3DGS PLYs in site-enu (x east, y north, z up, metres)
   "meshes": [{"file": "x.glb", "location": [x,y,z], "rotation_deg": [rx,ry,rz], "scale": 1}],   # optional
   "cameras": [{"name": "v4", "position": [x,y,z], "yaw_deg": 270, "pitch_deg": 0, "roll_deg": 0,
                "hfov_deg": 72, "width": 1280, "height": 720, "out": "out/v4.png"}],
   "engine": "BLENDER_EEVEE", "samples": 16, "world_rgb": [0.6, 0.7, 0.85]}

yaw: compass heading the camera looks toward (0 north, 90 east, 180 south, 270 west); pitch + = up.
"""
import json
import math
import sys

import bpy
from mathutils import Euler

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
for f in job.get("splats", []):
    bpy.ops.wm.ply_import(filepath=f)
for m in job.get("meshes", []):
    bpy.ops.import_scene.gltf(filepath=m["file"])
    for o in bpy.context.selected_objects:
        if o.parent is None:
            o.location = m.get("location", (0, 0, 0))
            o.rotation_euler = [math.radians(v) for v in m.get("rotation_deg", (0, 0, 0))]
            o.scale = [m.get("scale", 1)] * 3

world = bpy.data.worlds.new("w")
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs[0].default_value = (*job.get("world_rgb", [0.6, 0.7, 0.85]), 1)
sc.world = world
try:
    sc.render.engine = job.get("engine", "BLENDER_EEVEE")
except TypeError as e:
    print("engine fallback:", e)
if hasattr(sc, "eevee"):
    sc.eevee.taa_render_samples = job.get("samples", 16)
sc.view_settings.view_transform = "Standard"      # splat colours are display-referred (sRGB)

for c in job["cameras"]:
    cam = bpy.data.cameras.new(c["name"])
    cam.sensor_fit = "HORIZONTAL"
    cam.angle = math.radians(c["hfov_deg"])
    cam.clip_start, cam.clip_end = 0.05, 2000
    o = bpy.data.objects.new(c["name"], cam)
    sc.collection.objects.link(o)
    o.location = c["position"]
    # Blender camera looks down -Z with +Y up. Start looking north (rx=90), then yaw clockwise from north.
    o.rotation_euler = Euler((math.radians(90 + c.get("pitch_deg", 0)), math.radians(c.get("roll_deg", 0)),
                              math.radians(-c.get("yaw_deg", 0))), "XYZ")
    sc.camera = o
    sc.render.resolution_x, sc.render.resolution_y = c.get("width", 1280), c.get("height", 720)
    sc.render.resolution_percentage = 100
    sc.render.filepath = c["out"]
    bpy.ops.render.render(write_still=True)
    print("RENDERED", c["out"])
