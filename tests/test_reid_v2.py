"""reid_v2: metadata-driven preprocessing, L2 output, visit averaging, preset fallback, and the profiler factory
keeping the old OSNet embedder by default.

No network and no real model: tiny ONNX graphs (mean colour per channel -> fixed linear map) stand in for the
re-ID networks, so the expected vectors can be computed by hand.
Run:  .venv/bin/python -m pytest tests/test_reid_v2.py      (pytest optional)
      .venv/bin/python tests/test_reid_v2.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import reid, reid_v2  # noqa: E402

PROJ = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], np.float32)   # 3 -> 4


def tiny_model(path: Path, h: int, w: int, meta: dict | None) -> None:
    """images (N, 3, h, w) -> GlobalAveragePool -> (N, 3) -> @ PROJ -> features (N, 4)."""
    nodes = [helper.make_node("GlobalAveragePool", ["images"], ["gap"]),
             helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
             helper.make_node("MatMul", ["flat", "proj"], ["features"])]
    g = helper.make_graph(nodes, "tiny",
                          [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["n", 3, h, w])],
                          [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", 4])],
                          [numpy_helper.from_array(PROJ, "proj")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    for k, v in (meta or {}).items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, str(path))


def expected(bgr: tuple, mean, std) -> np.ndarray:
    rgb = np.array(bgr[::-1], np.float32) / 255.0
    v = ((rgb - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)) @ PROJ
    return v / np.linalg.norm(v)


def crop(bgr: tuple, h: int = 200, w: int = 90) -> np.ndarray:
    return np.full((h, w, 3), bgr, np.uint8)


def test_metadata_preprocessing_and_l2():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "anything.onnx"
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        tiny_model(p, 384, 128, {"sg_input_h": "384", "sg_input_w": "128", "sg_mean": json.dumps(mean),
                                 "sg_std": json.dumps(std), "sg_model": "tiny"})
        e = reid_v2.Embedder(str(p), threads=1)
        assert e.hw == (384, 128) and e.name == "tiny" and e.dim == 4
        assert e.preprocess(crop((10, 20, 30))).shape == (3, 384, 128)
        out = e([crop((10, 20, 200)), crop((240, 30, 5))])
        assert out.shape == (2, 4) and np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)
        assert np.allclose(out[0], expected((10, 20, 200), mean, std), atol=1e-4)   # BGR -> RGB, (x - mean) / std
        assert np.allclose(out[1], expected((240, 30, 5), mean, std), atol=1e-4)


def test_one_averages_then_renormalises():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "clipreid_vitb16_msmt17.int8.onnx"          # no metadata: preset by file name
        tiny_model(p, 256, 128, None)
        e = reid_v2.Embedder(str(p), threads=1)
        assert e.hw == (256, 128) and e.name == "clipreid_vitb16_msmt17"
        cs = [crop((10, 20, 200)), crop((240, 30, 5)), crop((90, 90, 90))]
        v = e.one(cs)
        ref = e(cs).mean(0)
        assert np.allclose(v, ref / np.linalg.norm(ref), atol=1e-6) and abs(np.linalg.norm(v) - 1) < 1e-5


def test_unknown_file_without_metadata_is_refused():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mystery.onnx"
        tiny_model(p, 256, 128, None)
        try:
            reid_v2.Embedder(str(p), threads=1)
        except ValueError as ex:
            assert "no sg_* metadata" in str(ex)
        else:
            raise AssertionError("expected ValueError")


def test_factory_default_is_osnet_and_v2_is_opt_in():
    with tempfile.TemporaryDirectory() as d:
        old = Path(d) / "osnet_x0_25_msmt17.onnx"
        new = Path(d) / "solider_swin_t_msmt17.onnx"
        tiny_model(old, 256, 128, None)
        tiny_model(new, 384, 128, None)
        assert isinstance(reid_v2.load_embedder({"reid_model": str(old)}, threads=1), reid.Embedder)
        assert isinstance(reid_v2.load_embedder({"reid_model": str(old), "reid_backend": "osnet",
                                                 "reid_v2_model": str(new)}, threads=1), reid.Embedder)
        v2 = reid_v2.load_embedder({"reid_model": str(old), "reid_backend": "v2", "reid_v2_model": str(new)}, threads=1)
        assert isinstance(v2, reid_v2.Embedder) and v2.hw == (384, 128)
        assert reid_v2.load_shadow({"reid_model": str(old)}) is None
        assert isinstance(reid_v2.load_shadow({"reid_shadow_model": str(new)}, threads=1), reid_v2.Embedder)
        for bad in ({"reid_model": str(old), "reid_backend": "v2"}, {"reid_model": str(old), "reid_backend": "x"}):
            try:
                reid_v2.load_embedder(bad, threads=1)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for {bad}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
