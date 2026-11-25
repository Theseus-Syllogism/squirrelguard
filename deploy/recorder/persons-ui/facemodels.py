"""Prior 3D face model index for the persons UI (numpy only): read the fitted faces stored by the bridge
(snapshots/evidence/<event>_facefit.glb, written by squirrel_guard/facefit.py), align them, build an identity's mean
face, and measure how each visit's face deviates from it or from another visit of the SAME identity.

POLICY: explanatory evidence only. Geometry is compared only between models already attributed to one identity (by the
face embedding / the identity decisions); it is never used to search, rank or name people, and never mapped to any
group. Depth is the landmark model's relative estimate unless two calibrated cameras measured the scale.

GLB layout (facefit.gltf_lods): one scene per LOD; LOD1 = the native MediaPipe topology, 478 points in a fixed order
(468 surface + 10 iris), int16 positions (stride 8) scaled by the node scale, uint16/uint32 indices. Fixed vertex order
means two visits' faces correspond point by point: alignment is a similarity Procrustes fit, no matching needed.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

IRIS_R, IRIS_L = 468, 473
EYE_CORNERS = (33, 133, 362, 263)
_CACHE: dict = {}          # path -> (mtime, V, T)


def read_lod(path: str | Path, name: str = "LOD1"):
    """GLB -> (vertices (n, 3) float64 in the file's units, triangles (m, 3) int) of the named LOD scene."""
    p = Path(path)
    mt = p.stat().st_mtime
    hit = _CACHE.get((str(p), name))
    if hit and hit[0] == mt:
        return hit[1], hit[2]
    b = p.read_bytes()
    magic, _ver, _len = struct.unpack_from("<III", b, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB")
    jl, jt = struct.unpack_from("<II", b, 12)
    g = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl, _bt = struct.unpack_from("<II", b, off)
    binc = b[off + 8: off + 8 + bl]
    scenes = g["scenes"]
    si = next((i for i, s in enumerate(scenes) if s.get("name") == name), g.get("scene", 0))
    node = g["nodes"][scenes[si]["nodes"][0]]
    mesh = g["meshes"][node["mesh"]]
    prim = mesh["primitives"][0]
    scale = np.asarray(node.get("scale", [1, 1, 1]), float)

    def acc(i):
        a = g["accessors"][i]
        v = g["bufferViews"][a["bufferView"]]
        ct = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        it = np.dtype(ct).itemsize
        stride = v.get("byteStride") or nc * it
        start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        raw = np.frombuffer(binc, dtype=np.uint8, count=stride * (a["count"] - 1) + nc * it, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(a["count"], nc * it), strides=(stride, 1))
        return np.ascontiguousarray(rows).view(ct).reshape(a["count"], nc)

    V = acc(prim["attributes"]["POSITION"]).astype(np.float64) * scale
    T = acc(prim["indices"]).reshape(-1, 3).astype(np.int64)
    _CACHE[(str(p), name)] = (mt, V, T)
    return V, T


def ipd(V: np.ndarray) -> float:
    if len(V) > IRIS_L:
        return float(np.linalg.norm(V[IRIS_R] - V[IRIS_L]))
    a, b, c, d = EYE_CORNERS
    return float(np.linalg.norm((V[a] + V[b]) / 2 - (V[c] + V[d]) / 2))


def normalise(V: np.ndarray) -> np.ndarray:
    """Centre and scale to inter-pupil distance = 1 (models in metres and in IPD units become comparable)."""
    W = V - V.mean(0)
    return W / max(ipd(W), 1e-9)


def align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Similarity (rotation, scale, translation) of A onto B, least squares (Umeyama)."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A0 ** 2).sum(1).mean()
    return s * A0 @ R.T + mb


def mean_shape(shapes: list[np.ndarray], weights: list[float] | None = None, iters: int = 6):
    """Generalised Procrustes: -> (mean (n, 3) in IPD units, aligned shapes)."""
    w = np.asarray(weights if weights else [1.0] * len(shapes), float)
    w = w / w.sum()
    S = [normalise(x) for x in shapes]
    m = S[int(np.argmax(w))].copy()
    for _ in range(iters):
        al = [align(x, m) for x in S]
        m = normalise(np.einsum("k,kvc->vc", w, np.stack(al)))
    return m, [align(x, m) for x in S]


def deviation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-point distance (IPD units) of shape A from shape B after aligning A onto B (both normalised)."""
    a, b = normalise(A), normalise(B)
    return np.linalg.norm(align(a, b) - b, axis=1)


def summary(dev: np.ndarray) -> dict:
    return {"rms_pct": round(100 * float(np.sqrt((dev ** 2).mean())), 2),
            "p90_pct": round(100 * float(np.percentile(dev, 90)), 2), "max_pct": round(100 * float(dev.max()), 2)}


def write_glb(V: np.ndarray, T: np.ndarray, name: str = "mean") -> bytes:
    """Minimal GLB (float32 positions, uint16/32 indices, one scene) for a mean face."""
    V = np.asarray(V, "<f4")
    big = len(V) > 65535
    I = np.asarray(T, "<u4" if big else "<u2").ravel()
    pb, ib = V.tobytes(), I.tobytes()
    pad = lambda x: x + b"\0" * ((-len(x)) % 4)  # noqa: E731
    buf = pad(pb) + pad(ib)
    g = {"asset": {"version": "2.0", "generator": "persons-ui facemodels"},
         "scene": 0, "scenes": [{"name": name, "nodes": [0]}], "nodes": [{"name": name, "mesh": 0}],
         "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
         "buffers": [{"byteLength": len(buf)}],
         "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pb), "target": 34962},
                         {"buffer": 0, "byteOffset": len(pad(pb)), "byteLength": len(ib), "target": 34963}],
         "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(V), "type": "VEC3",
                        "min": V.min(0).tolist(), "max": V.max(0).tolist()},
                       {"bufferView": 1, "componentType": 5125 if big else 5123, "count": len(I), "type": "SCALAR"}],
         "extras": {"units": "IPD (inter-pupil distance = 1)",
                    "note": "mean of an identity's fitted faces; explanatory evidence only, never an identity decision"}}
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(buf), 0x004E4942) + buf
    return out
