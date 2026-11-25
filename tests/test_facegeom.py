"""Face geometry (squirrel_guard/facegeom.py) on a synthetic mesh; MediaPipe and InspireFace are mocked.

Run:  .venv/bin/python tests/test_facegeom.py      (or pytest). No network, no GPU, no model files, no mediapipe.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy/recorder/bridge"))
from squirrel_guard import facegeom as fg  # noqa: E402
from squirrel_guard import faceid as fi  # noqa: E402

import facegeomcmp  # noqa: E402

RNG = np.random.default_rng(3)


def synth_face(nose_w=0.62, mouth_w=0.78, nose_tip=0.65, asym=0.0, seed=3) -> np.ndarray:
    """(478,3) mesh in FRONTAL IPD units (x subject right->left, y down, z toward the camera), mirror symmetric
    except for `asym` added to one eye corner. Iris centres at x = -/+0.5 (IPD = 1), nasion at the origin."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((478, 3))
    xs = rng.uniform(-1.1, 1.1, 478)
    ys = rng.uniform(-0.6, 2.0, 478)
    Q[:, 0], Q[:, 1] = xs, ys
    Q[:, 2] = -0.35 * xs ** 2 - 0.1 * (ys - 0.6) ** 2
    pts = {fg.IRIS_R: (-0.5, 0.0, 0.0), fg.IRIS_L: (0.5, 0.0, 0.0),
           fg.EYE_R_OUT: (-0.72, 0.0, -0.12), fg.EYE_R_IN: (-0.27, 0.0, -0.05),
           fg.EYE_L_IN: (0.27, 0.0, -0.05), fg.EYE_L_OUT: (0.72, 0.0, -0.12),
           fg.EYE_R_TOP: (-0.5, -0.08, 0.0), fg.EYE_L_TOP: (0.5, -0.08, 0.0),
           fg.BROW_R: (-0.5, -0.35, 0.05), fg.BROW_L: (0.5, -0.35, 0.05),
           fg.NASION: (0.0, 0.0, 0.0), fg.NOSE_TIP: (0.0, 0.55, -0.05 + nose_tip), fg.SUBNASALE: (0.0, 0.7, 0.35),
           fg.ALA_R: (-nose_w / 2, 0.62, 0.15), fg.ALA_L: (nose_w / 2, 0.62, 0.15),
           fg.MOUTH_R: (-mouth_w / 2, 1.05, 0.1), fg.MOUTH_L: (mouth_w / 2, 1.05, 0.1),
           fg.LIP_TOP: (0.0, 0.98, 0.25), fg.LIP_BOTTOM: (0.0, 1.2, 0.2),
           fg.CHEEK_R: (-1.1, 0.3, -0.9), fg.CHEEK_L: (1.1, 0.3, -0.9),
           fg.CHEEKS_R: (-0.6, 0.45, -0.1), fg.CHEEKS_L: (0.6, 0.45, -0.1),
           fg.JAW_R: (-0.9, 1.3, -0.7), fg.JAW_L: (0.9, 1.3, -0.7),
           fg.CHIN: (0.0, 1.75, 0.0), fg.GLABELLA: (0.0, -0.25, 0.05)}
    for i, p in pts.items():
        Q[i] = p
    for i in fg.MIDLINE:
        if i not in pts:
            Q[i, 0] = 0.0
    Q[fg.EYE_L_OUT, 1] += asym
    return Q


def to_image(Q, ipd_px=60.0, yaw_deg=0.0, centre=(128, 128)):
    """Frontal mesh -> image-space MediaPipe-like landmarks (x right, y down, z smaller = nearer), yawed."""
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    P = Q * (1, 1, -1) @ R.T * ipd_px
    P[:, 0] += centre[0]
    P[:, 1] += centre[1]
    return P.astype(np.float32)


def test_features_known_values_and_invariance():
    Q = synth_face()
    f0 = fg.features_of(to_image(Q))
    assert abs(f0["nose_width"] - 0.62) < 1e-3 and abs(f0["mouth_width"] - 0.78) < 1e-3, f0
    assert abs(f0["eye_spacing"] - 0.54) < 1e-3 and abs(f0["eye_width_r"] - 0.45) < 1e-3
    assert abs(f0["nose_protrusion"] - (0.60 + 0.05)) < 1e-3, f0          # tip z 0.60 vs inner corners -0.05
    assert f0["asymmetry"] < 1e-3
    assert f0["cheek_depth"] > 0.5 and f0["jaw_depth"] > 0.5
    # scale and yaw (the mesh is frontalised in 3D) do not change the numbers
    f1 = fg.features_of(to_image(Q, ipd_px=23.0, yaw_deg=20.0, centre=(40, 77)))
    for k in fg.FEATURES:
        assert abs(f0[k] - f1[k]) < 2e-3, (k, f0[k], f1[k])
    yaw, _ = fg.pose_of(to_image(Q, yaw_deg=20.0))
    assert abs(abs(yaw) - 20.0) < 1.0, yaw
    assert fg.features_of(to_image(synth_face(asym=0.1)))["asymmetry"] > 0.005


def test_robust_aggregate_ignores_outliers():
    vals = [0.6 + 0.005 * RNG.normal() for _ in range(9)] + [1.4]
    r = fg.robust(vals)
    assert abs(r["value"] - 0.6) < 0.01 and r["spread"] < 0.02 and r["n"] == 10
    assert fg.robust([])["value"] is None and fg.robust([0.5])["spread"] == 0.0


def _visit(nose_w, n=8, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    per, meta = [], []
    for _ in range(n):
        Q = synth_face(nose_w=nose_w + noise * rng.normal())
        per.append(fg.features_of(to_image(Q)))
        meta.append({"size": 90.0, "yaw": 5.0, "pitch": 2.0})
    return fg.aggregate(per, meta)


def test_compare_flags_a_different_shape_and_bridge_copy_agrees():
    enr = fg.merge([_visit(0.62, seed=1), _visit(0.63, seed=2)])
    same = fg.compare(_visit(0.62, seed=3), enr)
    other = fg.compare(_visit(0.90, seed=4), enr)
    assert same["explanatory"] and "embedding" in same["note"]
    assert same["features"]["nose_width"]["verdict"] == "consistent"
    assert other["features"]["nose_width"]["verdict"] == "differs" and other["features"]["nose_width"]["z"] > 3
    # the bridge's numpy-free copy gives identical results
    v = _visit(0.7, seed=5)
    assert facegeomcmp.merge([v, enr]) == fg.merge([v, enr])
    assert facegeomcmp.compare(v, enr) == fg.compare(v, enr)
    assert facegeomcmp.FEATURES == fg.FEATURES and facegeomcmp.WITHIN_SD == fg.WITHIN_SD
    assert facegeomcmp.STABLE_FEATURES == fg.STABLE_FEATURES
    assert fg.compare(None, enr) is None and fg.merge([]) is None


class MockMesher:
    """Stands in for FaceMesher: returns the synthetic mesh placed in the crop (yawed like the face)."""

    def __init__(self, fail_every=0):
        self.calls, self.fail_every = 0, fail_every

    def __call__(self, crop):
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            return None
        h = crop.shape[0]
        return to_image(synth_face(seed=self.calls), ipd_px=h / 4.5, centre=(h / 2, h / 2.6))


def _obs(size=80, yaw=0.0, pitch=0.0, t=0):
    o = fi.FaceObs(vec=fi.l2n(RNG.normal(size=fi.EMB_DIM)), quality=0.5, size=size, yaw=yaw, pitch=pitch, t=t,
                   box=(500.0, 300.0, 500.0 + size, 300.0 + size * 1.2))
    o.lmk106 = np.c_[np.linspace(500, 500 + size, 106), np.linspace(300, 300 + size, 106)].astype(np.float32)
    return o


def test_collector_pose_filter_bounds_and_payload():
    frame = np.broadcast_to(np.linspace(40, 220, 2560, dtype=np.uint8)[None, :, None], (1440, 2560, 3)).copy()
    m = MockMesher(fail_every=5)
    c = fg.GeomCollector(m, fg.GeomParams(keep=4))
    for i in range(12):
        c.offer(frame, _obs(size=60 + i, yaw=5.0, t=i), w=0.1 * i)
    c.offer(frame, _obs(size=200, yaw=50.0), w=99)          # turned away: never kept
    c.offer(frame, _obs(size=20), w=99)                     # too small: ignored
    assert len(c.cands) == 4 and c.off_pose == 1 and c.offered == 13
    assert [round(x.w, 1) for x in c.cands] == [1.1, 1.0, 0.9, 0.8]
    r = c.result()
    assert r["n_faces"] >= 3 and r["explanatory"] and m.calls == 4
    assert abs(r["features"]["nose_width"]["value"] - 0.62) < 0.01
    assert r["size_px"] > 60 and r["yaw_abs_med"] == 5.0
    assert all("f" in f for f in r["faces"])
    for k in ("overlay_jpg_b64", "depth_jpg_b64"):
        assert r[k] and len(r[k]) < 40000, (k, len(r[k] or ""))
    gl = json.dumps(r["mesh"])
    assert len(gl) <= 20000, len(gl)
    V, F = fg.gltf_arrays(r["mesh"])
    assert V.shape == (468, 3) and len(F) > 500 and F.max() < 468
    assert np.all(np.array(r["mesh"]["accessors"][0]["max"]) <= 32767)
    stripped = fg.strip_images(r)
    assert "mesh" not in stripped and "overlay_jpg_b64" not in stripped and stripped["features"]
    assert len(json.dumps(stripped)) < 8000


def test_collector_empty_and_mesh_failures():
    assert fg.GeomCollector(MockMesher()).result()["reason"] == "no_frontal_face"
    c = fg.GeomCollector(lambda crop: None)
    c.offer(np.zeros((1000, 1000, 3), np.uint8), _obs(size=60), 1.0)
    assert c.result()["reason"] == "no_mesh"


def test_render_html_self_contained():
    c = fg.GeomCollector(MockMesher())
    frame = RNG.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    for i in range(5):
        c.offer(frame, _obs(size=90, t=i), 1.0 + i)
    r = c.result()
    enr = fg.merge([_visit(0.62, seed=1)])
    h = fg.render_html(r, fg.compare(r, enr))
    assert "<canvas" in h and "Nose width" in h and "consistent" in h and "embedding" in h
    assert "http://" not in h and "https://" not in h          # no outside resources
    assert "Jaw width" not in h and "Jaw width" in fg.render_html(r, show_all=True)   # noisy hidden by default
    assert "No near-frontal face" in fg.render_html({"n_faces": 0})
    # no race/ethnicity anywhere in the payload or the page
    blob = (json.dumps(fg.strip_images(r)) + h).lower()
    assert "race" not in blob.replace("trace", "") and "ethnic" not in blob


class _Engine:
    """FaceEngine stand-in for FaceSampler: one face per call, growing."""
    pack = "mock"

    def faces(self, img, offset=(0, 0), scale=1.0, t=0.0, want_crop=None, frame=None, largest_only=True):
        return [_obs(size=70 + t, yaw=3.0, t=t)]


def test_sampler_hook_and_profiler_attach():
    from squirrel_guard import profiler
    eng = _Engine()
    face = (eng, {"facegeom_faces": 3})
    fs = fi.FaceSampler(eng, fi.AggParams(), backoff=False)
    profiler.facegeom_attach(fs, face)                       # no mesher on the engine: no geometry
    assert fs.geom is None
    eng.geom_mesher = MockMesher()
    profiler.facegeom_attach(fs, face)
    assert fs.geom is not None and fs.geom.p.keep == 3
    fr = RNG.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    for i in range(6):
        fs.on_box(fr, [400, 200, 700, 1000], None, t=i)
    r = fs.result()
    assert r["facegeom"]["n_faces"] == 3 and r["vec"] is not None
    assert profiler.facegeom_setup(eng, {}) is None and eng.geom_mesher is None   # flag off -> no mesher


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
