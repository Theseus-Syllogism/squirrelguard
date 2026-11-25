"""faceid_v2 helpers without any model files (AdaFace and InspireFace are faked). Run: .venv/bin/python tests/test_faceid_v2.py"""
import numpy as np

from squirrel_guard import faceid_v2 as v2
from squirrel_guard.faceid import AggParams, FaceObs, l2n


def test_align112_maps_reference_points_onto_themselves():
    img = np.random.default_rng(0).integers(0, 255, (112, 112, 3), np.uint8)
    out = v2.align112(img, v2.ARCFACE_REF)
    assert out.shape == (112, 112, 3)
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 1.0


def test_ir_score_grey_vs_colour():
    grey = np.full((20, 20, 3), 90, np.uint8)
    col = grey.copy()
    col[..., 2] = 200
    assert v2.is_ir(grey) and not v2.is_ir(col)


def test_calibration_threshold_and_fusion():
    imp = np.linspace(-0.2, 0.3, 1001)
    c = v2.Calibration.from_impostors(imp, fars=(0.01,))
    assert 0.29 < c.thr_by_far[0.01] <= 0.3
    cals = {"a": c, "b": c}
    assert v2.fuse({"a": 0.5, "b": None}, cals) == float(c.z(0.5))


def test_gallery_max_over_templates():
    g = v2.Gallery()
    e = np.eye(512, dtype=np.float32)
    g.add(1, e[0])
    g.add(1, e[1], ("side", "ir"))
    assert g.scores(e[1])[1] > 0.99


class _Eng:
    pack = "fake"

    def faces(self, img, offset=(0, 0), scale=1.0, t=0.0, frame=None, want_crop=None):
        o = FaceObs(vec=l2n(np.ones(512, np.float32)), quality=0.5, size=64, t=t)
        o.aligned = np.zeros((112, 112, 3), np.uint8)
        o.ir = 10.0
        return [o]


class _Ada:
    name = "fake_ada"

    def embed(self, crops):
        V = np.zeros((len(crops), 512), np.float32)
        V[:, 3] = 1
        return V, np.full(len(crops), 20.0, np.float32)


def test_sampler_v2_adds_v2_block_and_keeps_vec():
    s = v2.FaceSamplerV2(_Eng(), AggParams(), ada=_Ada(), backoff=False)
    fr = np.zeros((480, 640, 3), np.uint8)
    for i in range(5):
        s.on_box(fr, (200, 50, 300, 400), None, t=i)
    r = s.result()
    assert r["vec"] is not None and r["v2"]["model"] == "fake_ada"
    assert abs(r["v2"]["vec"][3] - 1.0) < 1e-3 and r["v2"]["ir_frac"] == 0.0
    assert all(getattr(f, "aligned", None) is None for _, f in s.agg.pool)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
