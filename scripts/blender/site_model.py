"""Blender 5.3: object/model render of the site from GIS data (companion of site_render.py, which renders the splat).

  blender -b --factory-startup --gpu-backend opengl --python scripts/blender/site_model.py -- JOB.json

JOB.json: {"dsm": "data/site-naip/dsm.npy" (surface heights, row 0 = north, cell_m each), "cell_m": 0.5,
           "half_m": 110, "ortho": "data/site-naip/ortho.png", "crop_m": 70 (optional half-width kept),
           "splats": [...] (optional, drawn instead of the surface when "surface": false),
           "vehicles": [plan.json vehicles], "frustums": [plan.json cameras, drawn as view wedges],
           "cameras": [as site_render.py]}
The surface is the lidar height grid as one mesh, textured top-down with the orthophoto (walls take roof colour).
Vehicles are plain boxes (they are not in the lidar); a frustum is the camera's horizontal field of view to range_m.
"""
import json, math, sys
import bpy, numpy as np
from mathutils import Euler

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def mat(name, rgba, emit=0.0, image=None):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = rgba; b.inputs["Roughness"].default_value = 0.85
    if "Alpha" in b.inputs: b.inputs["Alpha"].default_value = rgba[3]
    if emit:
        b.inputs["Emission Color"].default_value = rgba; b.inputs["Emission Strength"].default_value = emit
    if image:
        t = m.node_tree.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(image)
        m.node_tree.links.new(t.outputs["Color"], b.inputs["Base Color"])
    if rgba[3] < 1 and hasattr(m, "surface_render_method"): m.surface_render_method = "BLENDED"
    return m


def add_mesh(name, verts, faces, material, uvs=None, smooth=False):
    me = bpy.data.meshes.new(name); me.from_pydata(verts, [], faces); me.update()
    if uvs is not None:
        uv = me.uv_layers.new(name="uv"); li = np.zeros(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", li)
        uv.data.foreach_set("uv", np.asarray(uvs, np.float32)[li].ravel())
    if smooth: me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.materials.append(material); o = bpy.data.objects.new(name, me); sc.collection.objects.link(o); return o


half = job.get("half_m", 110.0); cell = job.get("cell_m", 0.5)
H = np.load(job["dsm"]); crop = job.get("crop_m", half)
n = H.shape[0]; xs = -half + (np.arange(n) + 0.5) * cell; ys = half - (np.arange(n) + 0.5) * cell
keep_x = np.abs(xs - job.get("center", [0, 0])[0]) <= crop; keep_y = np.abs(ys - job.get("center", [0, 0])[1]) <= crop
H = H[np.ix_(keep_y, keep_x)]; xs = xs[keep_x]; ys = ys[keep_y]
if job.get("surface", True) and not job.get("lod2"):
    X, Y = np.meshgrid(xs, ys); V = np.stack([X, Y, H], -1).reshape(-1, 3); r, c = H.shape
    i = (np.arange(r - 1)[:, None] * c + np.arange(c - 1)[None]).ravel()
    F = np.stack([i, i + c, i + c + 1, i + 1], 1)
    UV = np.stack([(X + half) / (2 * half), (Y + half) / (2 * half)], -1).reshape(-1, 2)
    add_mesh("site_surface", V.tolist(), F.tolist(), mat("ortho", (1, 1, 1, 1), image=job["ortho"]), UV)
for f in job.get("splats", []):
    bpy.ops.wm.ply_import(filepath=f)




# ---------------------------------------------------------------------------------------------------------------
# LOD2 street model (scripts/site_lod2.py -> lod2.json): procedural buildings, roads, trees, vehicles, sun.
def node(m, kind, **kw):
    n = m.node_tree.nodes.new(kind)
    for k, v in kw.items():
        if k in n.inputs:
            n.inputs[k].default_value = v
        else:
            setattr(n, k, v)
    return n


def link(m, a, b):
    m.node_tree.links.new(a, b)


def uv_xy(m):
    uv = node(m, "ShaderNodeUVMap"); sep = node(m, "ShaderNodeSeparateXYZ"); link(m, uv.outputs[0], sep.inputs[0]); return sep


def math_(m, op, a, b=None):
    n = node(m, "ShaderNodeMath", operation=op)
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            link(m, x, n.inputs[i])
    return n.outputs[0]


def wall_material(name, rgb, seed):
    """Siding bands + a regular window grid in wall UV metres (u along the wall, v height above ground)."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); b.inputs["Roughness"].default_value = 0.8
    xy = uv_xy(m); u, v = xy.outputs[0], xy.outputs[1]
    fx = math_(m, "FRACT", math_(m, "DIVIDE", math_(m, "ADD", u, seed % 7 * 0.37), 3.1))
    fy = math_(m, "FRACT", math_(m, "DIVIDE", math_(m, "SUBTRACT", v, 0.95), 2.8))
    win = math_(m, "MULTIPLY", math_(m, "MULTIPLY", math_(m, "GREATER_THAN", fx, 0.30), math_(m, "LESS_THAN", fx, 0.70)),
                math_(m, "MULTIPLY", math_(m, "GREATER_THAN", fy, 0.02), math_(m, "LESS_THAN", fy, 0.47)))
    win = math_(m, "MULTIPLY", win, math_(m, "GREATER_THAN", v, 0.9))
    band = math_(m, "FRACT", math_(m, "MULTIPLY", v, 5.0))
    shade = math_(m, "ADD", 0.86, math_(m, "MULTIPLY", math_(m, "GREATER_THAN", band, 0.82), -0.12))
    mix = node(m, "ShaderNodeMix", data_type="RGBA"); mix.inputs[6].default_value = (*[c for c in rgb], 1)
    mix.inputs[7].default_value = (0.05, 0.07, 0.09, 1); link(m, win, mix.inputs[0])
    col = node(m, "ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY"); col.inputs[0].default_value = 1
    link(m, mix.outputs[2], col.inputs[6]); cc = node(m, "ShaderNodeCombineColor"); [link(m, shade, cc.inputs[i]) for i in range(3)]
    link(m, cc.outputs[0], col.inputs[7]); link(m, col.outputs[2], b.inputs["Base Color"])
    rough = math_(m, "SUBTRACT", 0.85, math_(m, "MULTIPLY", win, 0.75)); link(m, rough, b.inputs["Roughness"])
    bump = node(m, "ShaderNodeBump", Strength=0.25); link(m, band, bump.inputs["Height"]); link(m, bump.outputs[0], b.inputs["Normal"])
    return m


def shingle_material(name, rgb):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); b.inputs["Roughness"].default_value = 0.92
    uv = node(m, "ShaderNodeUVMap")
    br = node(m, "ShaderNodeTexBrick", offset=0.5, squash=1.0)
    br.inputs["Scale"].default_value = 1.0; br.inputs["Brick Width"].default_value = 0.34; br.inputs["Row Height"].default_value = 0.14
    br.inputs["Mortar Size"].default_value = 0.012; br.inputs["Color1"].default_value = (*rgb, 1)
    br.inputs["Color2"].default_value = (*[c * 0.78 for c in rgb], 1); br.inputs["Mortar"].default_value = (*[c * 0.45 for c in rgb], 1)
    link(m, uv.outputs[0], br.inputs["Vector"]); link(m, br.outputs["Color"], b.inputs["Base Color"])
    bump = node(m, "ShaderNodeBump", Strength=0.35); link(m, br.outputs["Fac"], bump.inputs["Height"]); link(m, bump.outputs[0], b.inputs["Normal"])
    return m


def noisy_material(name, rgb, scale=6.0, var=0.25, rough=0.9, bump=0.15, metal=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    nz = node(m, "ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = scale; nz.inputs["Detail"].default_value = 8
    ramp = node(m, "ShaderNodeValToRGB"); ramp.color_ramp.elements[0].color = (*[c * (1 - var) for c in rgb], 1)
    ramp.color_ramp.elements[1].color = (*[min(c * (1 + var), 1) for c in rgb], 1)
    link(m, nz.outputs["Fac"], ramp.inputs["Fac"]); link(m, ramp.outputs["Color"], b.inputs["Base Color"])
    if bump:
        bp = node(m, "ShaderNodeBump", Strength=bump); link(m, nz.outputs["Fac"], bp.inputs["Height"]); link(m, bp.outputs[0], b.inputs["Normal"])
    return m


def build_lod2(L, gz):
    import random
    wall_cols = [(0.78, 0.72, 0.62), (0.66, 0.66, 0.63), (0.74, 0.69, 0.58), (0.60, 0.64, 0.66), (0.80, 0.77, 0.70), (0.57, 0.52, 0.45)]
    roof_cols = [(0.20, 0.20, 0.21), (0.26, 0.22, 0.19), (0.30, 0.29, 0.28), (0.17, 0.18, 0.20)]
    walls = [wall_material(f"wall{i}", c, i) for i, c in enumerate(wall_cols)]
    roofs = [shingle_material(f"roof{i}", c) for i, c in enumerate(roof_cols)]
    trim = noisy_material("trim", (0.85, 0.85, 0.82), rough=0.6, bump=0)
    skip = set(job.get("exclude_buildings", []))     # e.g. the house a camera is mounted in (walls block its view)
    for bd in L["buildings"]:
        if bd["id"] in skip:
            continue
        rnd = random.Random(bd["id"]); fp = np.array(bd["footprint"]); g0 = bd["ground"] - 0.15; eave = bd["eave"]
        if np.cross(fp[1] - fp[0], fp[2] - fp[1]) < 0 and sum((fp[(i + 1) % len(fp)][0] - fp[i][0]) * (fp[(i + 1) % len(fp)][1] + fp[i][1]) for i in range(len(fp))) > 0:
            fp = fp[::-1]
        V, F, UV = [], [], []
        run = 0.0
        for i in range(len(fp)):
            a, b_ = fp[i], fp[(i + 1) % len(fp)]; d = float(np.linalg.norm(b_ - a)); k = len(V)
            V += [(*a, g0), (*b_, g0), (*b_, eave), (*a, eave)]; F.append((k, k + 1, k + 2, k + 3))
            UV += [(run, 0), (run + d, 0), (run + d, eave - g0), (run, eave - g0)]; run += d
        add_mesh(f"b{bd['id']}_walls", V, F, walls[rnd.randrange(len(walls))], UV)
        r = bd["rect"]; c = np.array(r["c"]); u = np.array(r["u"]); v = np.array([-u[1], u[0]])
        hl, hw, ov = r["hl"] + 0.45, r["hw"] + 0.45, 0.45
        ridge = bd["ridge"] if bd["roof"] != "flat" else eave + 0.15; rise = ridge - eave
        P = lambda s, t, z: (*(c + u * s + v * t), z)
        if bd["roof"] == "flat":
            rv = [P(-hl, -hw, ridge), P(hl, -hw, ridge), P(hl, hw, ridge), P(-hl, hw, ridge)]; rf = [(0, 1, 2, 3)]
            ruv = [(0, 0), (2 * hl, 0), (2 * hl, 2 * hw), (0, 2 * hw)]
        else:
            ins = hl if bd["roof"] == "gable" else max(hl - hw, 0.3)
            e0 = eave - ov * rise / max(hw, 0.5)                       # eave line lowered by the overhang slope
            rv = [P(-hl, -hw, e0), P(hl, -hw, e0), P(hl, hw, e0), P(-hl, hw, e0), P(-ins, 0, ridge), P(ins, 0, ridge)]
            rf = [(0, 1, 5, 4), (2, 3, 4, 5), (3, 0, 4), (1, 2, 5)]
            sl = math.hypot(hw, rise)
            ruv = [(0, 0), (2 * hl, 0), (0, 0), (2 * hl, 0), (hl - ins, sl), (hl + ins, sl)]
        # split UVs per face (shared verts would smear): duplicate vertices per face
        V2, F2, U2 = [], [], []
        for f in rf:
            k = len(V2); V2 += [rv[i] for i in f]; U2 += [ruv[i] for i in f]; F2.append(tuple(range(k, k + len(f))))
        add_mesh(f"b{bd['id']}_roof", V2, F2, roofs[rnd.randrange(len(roofs))], U2)
        if bd["roof"] == "gable":                                      # gable-end triangles above the eave
            for sgn in (-1, 1):
                s0 = sgn * r["hl"]; add_mesh(f"b{bd['id']}_gable{sgn}", [P(s0, -r["hw"], eave), P(s0, r["hw"], eave), P(s0, 0, ridge - 0.05)], [(0, 1, 2)], trim)
    asphalt = noisy_material("asphalt", (0.16, 0.16, 0.17), scale=40, var=0.18, rough=0.95, bump=0.2)
    concrete = noisy_material("concrete", (0.62, 0.61, 0.58), scale=25, var=0.10, rough=0.9, bump=0.1)
    for rd in L["roads"]:
        pl = np.array(rd["line"]); w = rd["width"] / 2; V, F = [], []
        for i, p in enumerate(pl):
            d = pl[min(i + 1, len(pl) - 1)][:2] - pl[max(i - 1, 0)][:2]; d = d / (np.linalg.norm(d) + 1e-9); nrm = np.array([-d[1], d[0]])
            z = gz(*p[:2]) + 0.06
            V += [(*(p[:2] + nrm * w), z), (*(p[:2] - nrm * w), z)]
            if i:
                k = 2 * i; F.append((k - 2, k - 1, k + 1, k))
        add_mesh("road_" + rd["kind"], V, F, concrete if rd["kind"] == "footway" else asphalt)
    leaf = noisy_material("leaves", (0.20, 0.30, 0.12), scale=3, var=0.35, rough=0.85, bump=0.6)
    if job.get("season") == "winter":                 # bare deciduous crowns: faint brown-grey twig volume
        lb = next(n for n in leaf.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); lb.inputs["Alpha"].default_value = 0.22
        ramp = next(n for n in leaf.node_tree.nodes if n.type == "VALTORGB")
        ramp.color_ramp.elements[0].color = (0.16, 0.13, 0.11, 1); ramp.color_ramp.elements[1].color = (0.36, 0.31, 0.27, 1)
        if hasattr(leaf, "surface_render_method"): leaf.surface_render_method = "BLENDED"
    bark = noisy_material("bark", (0.22, 0.17, 0.12), scale=20, var=0.2, rough=0.95, bump=0.3)
    tex = bpy.data.textures.new("canopy", "CLOUDS"); tex.noise_scale = 0.9
    for t in L["trees"]:
        x, y, h, r = t["x"], t["y"], t["h"], t["r"]; z = gz(x, y)
        bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.12 + 0.02 * h, depth=h * 0.55, location=(x, y, z + h * 0.27))
        bpy.context.object.data.materials.append(bark)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1, location=(x, y, z + h * 0.62))
        o = bpy.context.object; o.scale = (r, r, max(h * 0.42, r * 0.8)); o.data.materials.append(leaf)
        d = o.modifiers.new("d", "DISPLACE"); d.texture = tex; d.strength = 0.35; d.mid_level = 0.5
    for i, vh in enumerate(L.get("vehicles", [])):
        build_vehicle(vh, gz, i)


def build_vehicle(v, gz, i):
    """Low-poly procedural vehicle: body, cabin (pickup: front 45 %, SUV: front 80 %), four wheels."""
    L, W = v["length"], v["width"]; a = math.radians(90 - v["heading_deg"]); cx, cy = v["center"]; z0 = gz(cx, cy)
    paint = noisy_material(f"paint{i}", tuple(v.get("rgb", (0.6, 0.6, 0.62))), scale=2, var=0.04, rough=0.25, bump=0, metal=0.6)
    glass = noisy_material(f"glass{i}", (0.05, 0.06, 0.07), scale=2, var=0.02, rough=0.08, bump=0)
    tyre = noisy_material(f"tyre{i}", (0.04, 0.04, 0.04), scale=10, var=0.1, rough=0.8, bump=0)
    def box(name, s0, s1, w, z_lo, z_hi, mat, bev=0.12):
        bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.object; o.name = name
        o.scale = ((s1 - s0), w, (z_hi - z_lo)); sm = (s0 + s1) / 2
        o.location = (cx + math.cos(a) * sm, cy + math.sin(a) * sm, z0 + (z_lo + z_hi) / 2); o.rotation_euler = (0, 0, a)
        o.data.materials.append(mat); bm = o.modifiers.new("bev", "BEVEL"); bm.width = bev; bm.segments = 3
    pickup = v.get("type", "suv") == "pickup"
    box(v["name"] + "_body", -L / 2, L / 2, W, 0.45, 1.15, paint)
    c0, c1 = (0.05 * L, 0.45 * L - 0.2) if pickup else (-0.42 * L, 0.18 * L)
    if pickup:
        c0, c1 = -0.05 * L, 0.18 * L
    box(v["name"] + "_cabin", c0, c1, W * 0.92, 1.1, 1.85 if pickup else 1.9, glass, 0.15)
    for s in (-L * 0.32, L * 0.32):
        for t in (-W / 2 + 0.1, W / 2 - 0.1):
            bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.4, depth=0.3)
            o = bpy.context.object; o.rotation_euler = (math.pi / 2, 0, a)
            o.location = (cx + math.cos(a) * s - math.sin(a) * t, cy + math.sin(a) * s + math.cos(a) * t, z0 + 0.4); o.data.materials.append(tyre)

def ground_at(x, y):
    return float(H[np.abs(ys - y).argmin(), np.abs(xs - x).argmin()])


veh = mat("vehicle", (0.85, 0.2, 0.15, 1))
for v in job.get("vehicles", []):
    L, W = v["length"] / 2, v["width"] / 2; a = math.radians(90 - v["heading_deg"]); ca, sa = math.cos(a), math.sin(a)
    cx, cy = v["center"]; z0 = min(ground_at(cx + dx, cy + dy) for dx in (-3, 0, 3) for dy in (-3, 0, 3)) + 0.05
    P = [(cx + ca * x - sa * y, cy + sa * x + ca * y, z0 + z) for z in (v["z0"] - 0.3, v["z1"]) for x, y in ((-L, -W), (L, -W), (L, W), (-L, W))]
    add_mesh(v["name"], P, [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)], veh)
fov = mat("fov", (0.15, 0.55, 1.0, 0.33), emit=0.6); mark = mat("cam", (0.05, 0.3, 0.9, 1), emit=1.5)
for c in job.get("frustums", []):
    x, y, z = c["position"]; z0 = ground_at(x, y); R = c.get("range_m", 30); hf = math.radians(c["hfov_deg"]) / 2
    yaw = math.radians(90 - c["yaw_deg"]); st = 24; P = [(x, y, z0 + z)]
    for k in range(st + 1):
        a = yaw - hf + 2 * hf * k / st; P.append((x + R * math.cos(a), y + R * math.sin(a), z0 + z + R * math.tan(math.radians(c.get("pitch_deg", 0))) + 0.6))
    add_mesh(c["name"] + "_fov", P, [(0, k, k + 1) for k in range(1, st + 1)], fov)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.9, location=(x, y, z0 + z + 0.9)); bpy.context.object.data.materials.append(mark)

if job.get("lod2"):
    Lj = json.load(open(job["lod2"])); Lj["vehicles"] = job.get("lod2_vehicles", [])
    Gg = np.load(job["ground_grid"]); gh = job.get("half_m", 110.0)
    def gz(x, y):
        i = min(max(int(x + gh), 0), Gg.shape[1] - 1); j = min(max(int(y + gh), 0), Gg.shape[0] - 1); return float(Gg[j, i])
    build_lod2(Lj, gz)
    cr = job.get("crop_m", 90); cx0, cy0 = job.get("center", [0, 0]); st = 1.0
    xs_ = np.arange(cx0 - cr, cx0 + cr + st, st); ys_ = np.arange(cy0 - cr, cy0 + cr + st, st)
    X_, Y_ = np.meshgrid(xs_, ys_); Z_ = np.vectorize(gz)(X_, Y_); r_, c_ = Z_.shape
    i_ = (np.arange(r_ - 1)[:, None] * c_ + np.arange(c_ - 1)[None]).ravel()
    gm = mat("ground_photo", (1, 1, 1, 1), image=job["ortho"])
    gb = next(n for n in gm.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); gb.inputs["Roughness"].default_value = 0.95
    nz = gm.node_tree.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 60; bp = gm.node_tree.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.2; gm.node_tree.links.new(nz.outputs["Fac"], bp.inputs["Height"]); gm.node_tree.links.new(bp.outputs[0], gb.inputs["Normal"])
    add_mesh("ground", np.stack([X_, Y_, Z_], -1).reshape(-1, 3).tolist(), np.stack([i_, i_ + 1, i_ + c_ + 1, i_ + c_], 1).tolist(), gm,
             np.stack([(X_ + gh) / (2 * gh), (Y_ + gh) / (2 * gh)], -1).reshape(-1, 2))

# textured meshes already in site coordinates (e.g. a fitted image-to-3D vehicle from scripts/vehicle_fit.py)
for ob in job.get("objs", []):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=ob["file"], forward_axis="Y", up_axis="Z")
    tm = bpy.data.materials.new("objtex"); tm.use_nodes = True
    tb = next(n for n in tm.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); tb.inputs["Roughness"].default_value = ob.get("roughness", 0.45)
    ti = tm.node_tree.nodes.new("ShaderNodeTexImage"); ti.image = bpy.data.images.load(ob["texture"])
    tm.node_tree.links.new(ti.outputs[0], tb.inputs["Base Color"])
    for o in set(bpy.data.objects) - before:
        if o.type == "MESH":
            o.data.materials.clear(); o.data.materials.append(tm)

# facade projection: paint a real (undistorted, masked RGBA) camera frame onto the building walls that camera sees
fp = job.get("facade_projection")
if fp:
    c = fp["camera"]; pcam = bpy.data.cameras.new("proj"); pcam.sensor_fit = "HORIZONTAL"; pcam.angle = math.radians(c["hfov_deg"])
    po = bpy.data.objects.new("proj_cam", pcam); sc.collection.objects.link(po); po.location = c["position"]
    po.rotation_euler = Euler((math.radians(90 + c.get("pitch_deg", 0)), 0, math.radians(-c.get("yaw_deg", 0))), "XYZ")
    img = bpy.data.images.load(fp["image"]); iw, ih = img.size
    sc.render.resolution_x, sc.render.resolution_y = iw, ih          # UV Project uses the scene aspect for camera projectors
    fwd = po.rotation_euler.to_matrix() @ __import__("mathutils").Vector((0, 0, -1))   # matrix_world is stale until a depsgraph update
    for o in list(sc.objects):
        if o.type == "MESH" and o.name.endswith("_walls"):
            o.data.uv_layers.new(name="proj")
            md = o.modifiers.new("proj", "UV_PROJECT"); md.uv_layer = "proj"; md.projector_count = 1
            md.projectors[0].object = po; md.aspect_x, md.aspect_y = iw, ih
    for m in [m for m in bpy.data.materials if m.name.startswith("wall")]:
        nt = m.node_tree; b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        base = b.inputs["Base Color"].links[0].from_socket
        uvn = nt.nodes.new("ShaderNodeUVMap"); uvn.uv_map = "proj"
        tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img; tex.extension = "CLIP"; nt.links.new(uvn.outputs[0], tex.inputs[0])
        geo = nt.nodes.new("ShaderNodeNewGeometry"); dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = (-fwd.x, -fwd.y, -fwd.z); nt.links.new(geo.outputs["Normal"], dot.inputs[0])
        face = nt.nodes.new("ShaderNodeMath"); face.operation = "GREATER_THAN"; face.inputs[1].default_value = 0.15
        nt.links.new(dot.outputs["Value"], face.inputs[0])
        fac = nt.nodes.new("ShaderNodeMath"); fac.operation = "MULTIPLY"; nt.links.new(tex.outputs["Alpha"], fac.inputs[0]); nt.links.new(face.outputs[0], fac.inputs[1])
        mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; nt.links.new(fac.outputs[0], mix.inputs[0])
        nt.links.new(base, mix.inputs[6]); nt.links.new(tex.outputs["Color"], mix.inputs[7]); nt.links.new(mix.outputs[2], b.inputs["Base Color"])

sun = bpy.data.lights.new("sun", "SUN"); sun.energy = job.get("sun", 2.2); so = bpy.data.objects.new("sun", sun); sc.collection.objects.link(so)
el, az = job.get("sun_el", 40), job.get("sun_az", 220)       # degrees; azimuth from north, clockwise
so.rotation_euler = Euler((math.radians(90 - el), 0, math.radians(180 - az)), "XYZ"); sun.angle = math.radians(0.6)
world = bpy.data.worlds.new("w"); world.use_nodes = True
bg = next(nd for nd in world.node_tree.nodes if nd.type == "BACKGROUND"); bg.inputs[0].default_value = (*job.get("world_rgb", [0.78, 0.84, 0.92]), 1)
bg.inputs[1].default_value = 0.55; sc.world = world
try:
    sc.render.engine = job.get("engine", "BLENDER_EEVEE")
except TypeError as e:
    print("engine fallback:", e)
if hasattr(sc, "eevee"): sc.eevee.taa_render_samples = job.get("samples", 32)
sc.view_settings.view_transform = "Standard"
for c in job["cameras"]:
    cam = bpy.data.cameras.new(c["name"]); cam.sensor_fit = "HORIZONTAL"; cam.angle = math.radians(c["hfov_deg"]); cam.clip_start, cam.clip_end = 0.05, 2000
    o = bpy.data.objects.new(c["name"], cam); sc.collection.objects.link(o); o.location = c["position"]
    o.rotation_euler = Euler((math.radians(90 + c.get("pitch_deg", 0)), 0, math.radians(-c.get("yaw_deg", 0))), "XYZ")
    sc.camera = o; sc.render.resolution_x, sc.render.resolution_y = c.get("width", 1600), c.get("height", 900)
    sc.render.resolution_percentage = 100; sc.render.filepath = c["out"]; bpy.ops.render.render(write_still=True); print("RENDERED", c["out"])
