"""facefit (squirrel_guard/facefit.py) on SYNTHETIC faces only: no mediapipe, no model files, no network, no images.

Run:  .venv/bin/python tests/test_facefit.py      (or pytest). Needs numpy + cv2 (Camera rays / projection).
The synthetic face is the MediaPipe topology laid out by a harmonic (Tutte) embedding with an ellipsoid depth and a
nose bump: a smooth, plausible surface with the right landmark order, not anyone's face.
"""
from __future__ import annotations

import io
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squirrel_guard import facefit as ff  # noqa: E402
from squirrel_guard import facegeom as fg  # noqa: E402
from squirrel_guard.geometry import Camera  # noqa: E402


# -- fixtures -----------------------------------------------------------------------------------------------------
def synth_shape(nose=0.55, width=1.0, seed=0) -> np.ndarray:
    """(478, 3) synthetic face in facegeom's frontal frame (IPD = 1, nasion origin, z toward the camera)."""
    T = ff.topology(True)
    loop = max(ff.boundary_loops(T), key=len)
    k = loop.index(10)                                   # forehead top first
    loop = loop[k:] + loop[:k]
    n = len(loop)
    ang = np.pi / 2 + 2 * np.pi * np.arange(n) / n
    P = np.zeros((ff.N_SURF, 2))
    P[loop] = np.c_[1.6 * width * np.cos(ang), -2.0 * np.sin(ang)]
    if P[234, 0] > P[454, 0]:                            # subject's right cheek on the image left
        P[loop, 0] *= -1
    E = ff.unique_edges(T)
    A = np.zeros((ff.N_SURF, ff.N_SURF))
    A[E[:, 0], E[:, 1]] = A[E[:, 1], E[:, 0]] = 1
    L = np.diag(A.sum(1)) - A
    inner = np.setdiff1d(np.arange(ff.N_SURF), loop)
    P[inner] = np.linalg.solve(L[np.ix_(inner, inner)], -L[np.ix_(inner, loop)] @ P[loop])
    x, y = P[:, 0], P[:, 1]
    z = 0.9 * np.sqrt(np.clip(1 - (x / (1.7 * width)) ** 2 - (y / 2.1) ** 2, 0, 1))
    nt = P[fg.NOSE_TIP]
    z += nose * np.exp(-((x - nt[0]) ** 2 + (y - nt[1]) ** 2) / 0.08)
    V = np.c_[x, y, z]
    rng = np.random.default_rng(seed)
    V += rng.normal(0, 0.004, V.shape)                    # a little individual detail
    out = np.zeros((ff.N_ALL, 3))
    out[:ff.N_SURF] = V
    for c, ring, (a, b) in ((468, (469, 470, 471, 472), (fg.EYE_R_OUT, fg.EYE_R_IN)),
                            (473, (474, 475, 476, 477), (fg.EYE_L_IN, fg.EYE_L_OUT))):
        ctr = V[[a, b]].mean(0) + (0, 0, 0.02)
        out[c] = ctr
        for j, r in enumerate(ring):
            out[r] = ctr + 0.06 * np.array([math.cos(j * math.pi / 2), math.sin(j * math.pi / 2), 0])
    Q, _ = fg.frontalise(out * (1, 1, -1))               # into the gauge (IPD = 1 between iris centres)
    return Q


def rot(yaw, pitch, roll) -> np.ndarray:
    a, b, c = (math.radians(v) for v in (yaw, pitch, roll))
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def frames_of(S, F=30, seed=1, sigma_xy=0.4, sigma_z=2.0, depth_scale=1.0):
    """Landmarks in image px (MediaPipe convention: z in px, smaller = nearer) of S seen in F poses."""
    rng = np.random.default_rng(seed)
    U = S * (1, 1, -1)
    U = U * (1, 1, depth_scale)
    X, meta = [], []
    for f in range(F):
        R = rot(rng.uniform(-30, 30), rng.uniform(-15, 15), rng.uniform(-10, 10))
        s = rng.uniform(20, 60)
        t = np.array([rng.uniform(100, 500), rng.uniform(100, 300), 0.0])
        P = s * U @ R.T + t
        P[:, :2] += rng.normal(0, sigma_xy, (len(P), 2))
        P[:, 2] += rng.normal(0, sigma_z, len(P))
        X.append(P)
        meta.append((s, R, t))
    return np.stack(X), meta


def obs_of(X, cam="c", sharp=100.0):
    out = []
    for i, P in enumerate(X):
        ipd = float(np.linalg.norm(P[fg.IRIS_R, :2] - P[fg.IRIS_L, :2]))
        y, p = fg.pose_of(P)
        out.append(ff.FrameObs(cam, i, i / 8.0, P.astype(np.float32), ipd, 4 * ipd, sharp, False, y, p))
    return out


def rms(A, B):
    return float(np.sqrt(((A[:ff.N_SURF] - B[:ff.N_SURF]) ** 2).sum(1).mean()))


# -- topology ---------------------------------------------------------------------------------------------------
def test_topology_manifold_and_oriented():
    for closed, nf, nloops in ((True, 898, 1), (False, 852, 4)):
        T = ff.topology(closed)
        assert T.shape == (nf, 3) and T.min() == 0 and T.max() == ff.N_SURF - 1
        de = [(a, b) for t in T for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))]
        assert len(de) == len(set(de)), "consistently oriented, manifold"
        E = ff.unique_edges(T)
        assert len(ff.boundary_loops(T)) == nloops
        assert ff.N_SURF - len(E) + len(T) == 2 - nloops       # Euler characteristic of a sphere minus holes
    assert set(ff.FEATURE_VERTS) <= set(range(ff.N_SURF))
    S = synth_shape()
    n = ff.vertex_normals(S[:ff.N_SURF], ff.topology(True))
    assert np.mean(n[:, 2] > 0) > 0.9, "normals face the camera in the frontal frame"


# -- fit ----------------------------------------------------------------------------------------------------------
def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 3))
    R = rot(20, -10, 5)
    B = 3.5 * A @ R.T + (1, 2, 3)
    s, R2, t = ff.umeyama(A, B)
    assert abs(s - 3.5) < 1e-9 and np.allclose(R2, R) and np.allclose(t, (1, 2, 3))


def test_fit_recovers_shape_and_beats_per_frame_median():
    S = synth_shape()
    X, meta = frames_of(S, F=30, sigma_xy=0.6, sigma_z=3.0)
    wf = ff.frame_base_weights(obs_of(X), ff.FitParams())
    r = ff.fit_identity(X, wf, params=ff.FitParams(boot=8))
    e_fit = rms(r.S, S)
    med = np.median(np.stack([fg.frontalise(x)[0] for x in X]), 0)
    e_med = rms(med, S)
    assert e_fit < 0.03, e_fit
    assert e_fit < e_med, (e_fit, e_med)
    # poses: recovered yaw within a few degrees
    yaw_true = np.array([ff.angles_of(R)[0] for _, R, _ in meta])
    assert np.median(np.abs(r.angles[:, 0] - yaw_true)) < 2.0
    # reprojection ~ the 2D noise
    assert np.median(r.resid_px) < 1.5
    assert r.vertex_sd is not None and np.all(np.isfinite(r.vertex_sd)) and np.median(r.vertex_sd) < 0.05
    assert set(r.boot_features) == set(fg.FEATURES)
    f_true, f_fit = ff.features(S), ff.features(r.S)
    for k in ("eye_spacing", "nose_width", "face_height"):
        assert abs(f_fit[k] - f_true[k]) < 0.03, (k, f_fit[k], f_true[k])


def test_features_match_facegeom_convention():
    S = synth_shape()
    P = 37.0 * (S * (1, 1, -1)) @ rot(10, 5, -3).T + (200, 150, 0)
    a, b = fg.features_of(P), ff.features(S)
    for k in fg.FEATURES:
        assert abs(a[k] - b[k]) < 2e-3, (k, a[k], b[k])


def test_outlier_frame_and_expression_are_downweighted():
    S = synth_shape()
    X, _ = frames_of(S, F=24, seed=4, sigma_xy=0.2, sigma_z=0.5)
    other = synth_shape(nose=0.9, width=1.25, seed=9)            # a different face in two frames
    X2, _ = frames_of(other, F=2, seed=5)
    X[3], X[11] = X2[0], X2[1]
    # 6 frames with an open mouth: the lower lip and the chin drop by 0.2 IPD (jaw opening), the upper face stays
    mouth_y = (S[ff.LIP_IN_TOP, 1] + S[ff.LIP_IN_BOTTOM, 1]) / 2
    drop = np.where(S[:, 1] > mouth_y, 0.2, 0.0)
    for f in (5, 6, 7, 15, 16, 17):
        s = np.linalg.norm(X[f, fg.IRIS_R, :2] - X[f, fg.IRIS_L, :2])
        down = X[f, fg.CHIN, :2] - X[f, fg.NASION, :2]
        down /= np.linalg.norm(down)
        X[f, :, :2] += (s * drop)[:, None] * down[None, :]
    wf = ff.frame_base_weights(obs_of(X), ff.FitParams())
    r = ff.fit_identity(X, wf, params=ff.FitParams(boot=0))
    assert r.expr[[5, 6, 7], 0].min() > 0.3 and np.median(r.expr[:, 0]) < 0.2
    assert r.w_frame[[3, 11]].max() < 0.5 * np.median(r.w_frame), r.w_frame[[3, 11]]
    f_true, f_fit = ff.features(S), ff.features(r.S)
    assert abs(f_fit["face_height"] - f_true["face_height"]) < 0.04, (f_fit["face_height"], f_true["face_height"])
    assert abs(f_fit["nose_width"] - f_true["nose_width"]) < 0.03


def test_small_faces_regularised():
    S = synth_shape()
    X, _ = frames_of(S, F=4, seed=6, sigma_xy=2.5, sigma_z=6.0)
    wf = np.ones(4)
    p0 = ff.FitParams(boot=0, lam=0.0)
    p1 = ff.FitParams(boot=0, lam=30.0)
    r0, r1 = ff.fit_identity(X, wf, params=p0), ff.fit_identity(X, wf, params=p1)
    T = ff.topology(True)
    def rough(V):
        E = ff.unique_edges(T)
        return float(np.mean(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))
    # with a prior (the visit median), the Laplacian term smooths: no rougher than without it
    assert rough(r1.S[:ff.N_SURF]) <= rough(r0.S[:ff.N_SURF]) * 1.01


# -- two views ----------------------------------------------------------------------------------------------------
def _cam(C, yaw_deg, size=(1920, 1080), f=1200.0):
    a = math.radians(yaw_deg)
    # camera looking along world +Y rotated about Z; OpenCV axes: x right, y down, z forward
    R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    K = [[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]]
    return Camera(K, [0, 0, 0, 0, 0], Rz @ R0, C, size)


def test_triangulate_and_multiview_scale_depth():
    S = synth_shape()
    U = S * (1, 1, -1)
    ca, cb = _cam([0, 0, 1.0], 0.0), _cam([-0.25, -0.1, 1.0], 8.0)
    size = (1920, 1080)
    ipd_m, k_true = 0.063, 1.4
    rng = np.random.default_rng(2)
    pairs = []
    for i in range(6):
        # face 1.3 m in front of camera a, facing it (face z axis = -world Y), slight head turns
        Rf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)       # face x -> world X, y(down) -> -Z, z(in) -> +Y
        Rf = Rf @ rot(rng.uniform(-15, 15), rng.uniform(-8, 8), 0)
        t = np.array([rng.uniform(-0.1, 0.1), 1.3 + rng.uniform(-0.1, 0.1), 1.55])
        Xw = ipd_m * (U * (1, 1, k_true)) @ Rf.T + t
        Xt, gap = ff.triangulate(ca, ff.project(ca, Xw, size), size, cb, ff.project(cb, Xw, size), size)
        assert np.nanmax(gap) < 1e-6 and np.allclose(Xt, Xw, atol=1e-6)
        obs = []
        for cam in (ca, cb):
            px = ff.project(cam, Xw, size) + rng.normal(0, 0.3, (ff.N_ALL, 2))
            P = np.c_[px, np.zeros(ff.N_ALL)]
            obs.append(ff.FrameObs("a" if cam is ca else "b", i, i / 8, P.astype(np.float32),
                                   float(np.linalg.norm(px[fg.IRIS_R] - px[fg.IRIS_L])), 100, 100, False))
        pairs.append(tuple(obs))
    mv = ff.multiview(U, pairs, ca, size, cb, size)
    assert mv["scale_used"] and mv["depth_used"], mv
    assert abs(mv["ipd_mm"] - 63.0) < 2.5, mv["ipd_mm"]          # lateral scale: ~2 % perspective bias at 1.3 m
    assert abs(mv["depth_scale"] - k_true) < 0.15, mv["depth_scale"]
    S2 = ff.apply_depth_scale(S, mv["depth_scale"])
    assert abs(ff.features(S2)["nose_protrusion"] / ff.features(S)["nose_protrusion"] - mv["depth_scale"]) < 1e-2
    m = ff.metric_features(ff.features(S2), None, mv["ipd_mm"], mv["ipd_mm_sd"])
    assert abs(m["eye_spacing"]["mm"] - ff.features(S2)["eye_spacing"] * mv["ipd_mm"]) < 0.1
    few = ff.multiview(U, pairs[:2], ca, size, cb, size)
    assert not few["used"] and "pairs" in few["why"]


def test_undistort_identity_without_distortion():
    cam = _cam([0, 0, 1], 0)
    S = synth_shape()
    P = 40 * (S * (1, 1, -1)) + (900, 500, 0)
    assert np.allclose(ff.undistort_landmarks(P, cam, (1920, 1080), recentre=False), P, atol=1e-3)
    assert np.allclose(ff.undistort_landmarks(P / 2, cam, (960, 540), recentre=False), P / 2, atol=1e-3)


def test_recentred_view_removes_off_axis_stretch():
    """A face far off-axis in a rectilinear image looks radially stretched; the face-centred view undoes it."""
    cam = _cam([0, 0, 1.0], 0.0, f=900.0)
    size = (1920, 1080)
    S = synth_shape()
    U = S * (1, 1, -1)
    Rf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    out = []
    for x_off in (0.0, 1.2):                              # straight ahead vs ~40 deg to the side (facing the camera)
        t = np.array([x_off, 1.5, 1.0])
        yaw = math.degrees(math.atan2(x_off, 1.5))
        R = Rf @ rot(-yaw, 0, 0)                          # turned toward the camera
        Xw = 0.063 * U @ R.T + t
        px = ff.project(cam, Xw, size)
        P = np.c_[px, np.zeros(ff.N_ALL)]
        raw = fg.features_of(np.c_[px, np.zeros(ff.N_ALL)])
        rec = fg.features_of(ff.undistort_landmarks(P, cam, size))
        out.append((raw, rec))
    k = "face_height"                                     # vertical / IPD: the radial (horizontal) stretch shows
    assert abs(out[1][1][k] / out[0][1][k] - 1) < 0.01, (out[1][1][k], out[0][1][k])
    assert out[1][0][k] / out[0][0][k] < 0.9, (out[1][0][k], out[0][0][k])     # raw: IPD stretched, height not


# -- LODs + glTF ----------------------------------------------------------------------------------------------------
def test_loop_subdivision():
    S = synth_shape()
    V, T = S[:ff.N_SURF], ff.topology(True)
    V2, T2 = ff.loop_subdivide(V, T)
    assert len(T2) == 4 * len(T) and len(V2) == len(V) + len(ff.unique_edges(T))
    de = [(a, b) for t in T2 for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))]
    assert len(de) == len(set(de))
    V3, T3 = ff.loop_subdivide(V2, T2)
    assert len(T3) == 14368
    # a plane stays a plane; smoothing moves the original vertices inward (convex surface), not wildly
    Vp = V.copy()
    Vp[:, 2] = 0.3 * Vp[:, 0] - 0.2 * Vp[:, 1]
    Vq, _ = ff.loop_subdivide(Vp, T)
    assert np.allclose(Vq[:, 2], 0.3 * Vq[:, 0] - 0.2 * Vq[:, 1], atol=1e-9)
    assert np.abs(V2[:len(V)] - V).max() < 0.2
    n = ff.vertex_normals(V3, T3)
    assert np.mean(n[:, 2] > 0) > 0.9


def test_decimation_keeps_feature_landmarks():
    S = synth_shape()
    V, T = S[:ff.N_SURF], ff.topology(True)
    V0, T0, vid = ff.decimate(V, T, 250, keep=ff.FEATURE_VERTS)
    assert 200 <= len(T0) <= 260, len(T0)
    assert set(ff.FEATURE_VERTS) <= set(vid.tolist())
    assert np.allclose(V0, V[vid])
    de = [(a, b) for t in T0 for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))]
    assert len(de) == len(set(de)), "still manifold + oriented"
    n = np.cross(V0[T0[:, 1]] - V0[T0[:, 0]], V0[T0[:, 2]] - V0[T0[:, 0]])
    assert np.mean(n[:, 2] > 0) > 0.85
    # features on LOD0 equal features of the full shape (same vertices kept)
    full = np.zeros((ff.N_ALL, 3))
    full[vid] = V0
    full[ff.N_SURF:] = S[ff.N_SURF:]
    a, b = ff.features(S), ff.features(full)
    for k, (kind, _) in fg.FEATURES.items():
        if k not in ("eye_socket_depth",):          # uses only kept vertices + irises
            assert abs(a[k] - b[k]) < 1e-9, k


def test_gltf_lods_roundtrip_and_glb():
    S = synth_shape()
    lv = ff.lods(S)
    assert [x["name"] for x in lv] == ["LOD0", "LOD1", "LOD2", "LOD3"]
    assert [len(x["T"]) for x in lv][1:] == [898, 3592, 14368]
    sd = np.full(ff.N_ALL, 0.01)
    g = ff.gltf_lods(lv, unit_m=0.063, vertex_sd=sd)
    json.dumps(g)
    for x in lv:
        V, F = ff.gltf_mesh(g, x["name"])
        assert np.abs(V - x["V"] * (1, -1, 1) * 0.063).max() < 0.063 / 4000 + 1e-9
        assert np.array_equal(F, x["T"][:, ::-1])
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        assert np.mean(n[:, 2] > 0) > 0.85, "CCW seen from +z (the viewer)"
    assert g["scenes"][g["scene"]]["name"] == "LOD1"
    m1 = next(m for m in g["meshes"] if m["name"] == "LOD1")
    assert "_UNCERTAINTY" in m1["primitives"][0]["attributes"]
    glb = ff.gltf_to_glb(g)
    magic, ver, total = struct.unpack("<4sII", glb[:12])
    assert magic == b"glTF" and ver == 2 and total == len(glb) and total % 4 == 0
    jl, jt = struct.unpack("<I4s", glb[12:20])
    assert jt == b"JSON" and "uri" not in json.loads(glb[20:20 + jl])["buffers"][0]
    assert len(glb) < 260_000


# -- canonical prior from a .task bundle ------------------------------------------------------------------------------
def _pb(field, wt, payload: bytes) -> bytes:
    return ff._enc_varint(field << 3 | wt) + (ff._enc_varint(len(payload)) + payload if wt == 2 else payload)


def test_canonical_from_task_bundle():
    S = synth_shape()
    V = S[:ff.N_SURF] * (1, -1, 1) * 7.0                     # y up, z toward viewer, "cm"
    T = ff.topology(True)
    vb = np.c_[V, np.zeros((ff.N_SURF, 2))].astype("<f4").tobytes()
    ib = b"".join(ff._enc_varint(int(i)) for i in T.ravel())
    mesh = _pb(1, 0, ff._enc_varint(0)) + _pb(3, 2, vb) + _pb(4, 2, ib)
    meta = _pb(3, 0, ff._enc_varint(1)) + _pb(1, 2, mesh) + _pb(2, 2, _pb(1, 0, ff._enc_varint(4)))
    got = ff.parse_canonical_mesh(meta)
    assert got is not None and np.allclose(got[0], V, atol=1e-5) and np.array_equal(got[1], T)
    # unpacked index encoding is accepted too
    mesh2 = _pb(3, 2, vb) + b"".join(_pb(4, 0, ff._enc_varint(int(i))) for i in T.ravel()[:9])
    got2 = ff.parse_canonical_mesh(_pb(1, 2, mesh2))
    assert got2 is not None and np.array_equal(got2[1], T[:3])
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        z.writestr("face_detector.tflite", b"x")
        z.writestr("geometry_pipeline_metadata_landmarks.binarypb", meta)
    p = Path(__file__).resolve().parent / "_tmp_facefit_test.task"
    try:
        p.write_bytes(bio.getvalue())
        Q = ff.canonical_from_task(str(p))
    finally:
        p.unlink(missing_ok=True)
    assert Q is not None and Q.shape == (ff.N_SURF, 3)
    assert abs(Q[fg.NOSE_TIP, 2] - Q[fg.NASION, 2]) > 0.1          # z toward the camera: nose in front
    assert Q[fg.NOSE_TIP, 2] > Q[fg.CHEEK_R, 2]
    r = ff.fit_identity(frames_of(S, F=6)[0], np.ones(6), prior=Q, params=ff.FitParams(boot=0))
    assert r.prior == "canonical" and rms(r.S, S) < 0.05


# -- IR helpers, payload, policy ---------------------------------------------------------------------------------------
def test_ir_helpers():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    gray3 = np.stack([g, g, g], -1)
    col = np.stack([g, np.roll(g, 3, 0), np.roll(g, 7, 1)], -1)
    assert ff.is_grayscale(gray3) and ff.is_grayscale(g) and not ff.is_grayscale(col)
    out = ff.ir_prepare(gray3)
    assert out.shape == (64, 64, 3) and np.array_equal(out[..., 0], out[..., 1])


def test_fit_visit_payload_policy():
    S = synth_shape()
    X, _ = frames_of(S, F=12, seed=8)
    pl = ff.fit_visit(obs_of(X), params=ff.FitParams(boot=6), pairs=[])
    assert pl["explanatory"] is True and pl["n_used"] >= 10
    assert set(pl["features"]) == set(fg.FEATURES)
    assert pl["multiview"]["used"] is False
    assert "description only" in pl["apparent_expression"]["note"]
    s = json.dumps(ff.strip(pl))
    for bad in ("race", "ethnic", "gender", "identity_decision"):
        assert bad not in s.lower()
    assert ff.gltf_to_glb(pl["mesh"])[:4] == b"glTF"
    assert ff.fit_visit([], params=ff.FitParams(boot=0))["reason"] == "no_face"


def test_fit_collector_with_mock_landmarker():
    """FitCollector keeps landmarks only (no crops), bounded, and fits at the end (landmarker mocked)."""
    from types import SimpleNamespace
    S = synth_shape()
    X, _ = frames_of(S, F=10, seed=12, sigma_xy=0.2, sigma_z=0.5)
    cur = [0]

    def lm(crop):                                    # landmarks in crop px: the true face, wherever the crop is
        P = X[cur[0]].copy()
        P[:, :2] -= P[:ff.N_SURF, :2].min(0) - 30
        return P.astype(np.float32), None
    c = ff.FitCollector(lm, cam="v4", max_frames=8, params=ff.FitParams(boot=0))
    frame = np.zeros((600, 800, 3), np.uint8)
    for i in range(10):
        cur[0] = i
        c.offer(frame, SimpleNamespace(box=(300, 200, 400, 300), size=100.0, t=i), w=0.1 * (i + 1))
    assert len(c.obs) == 8 and min(w for w, _ in c.obs) > 0.25        # the two lowest-weight frames dropped
    r = c.result(mesh=False)
    assert r["n_frames"] == 8 and r["explanatory"] and "_fit" not in r and r["offered"] == 10


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e)[:300])
    sys.exit(1 if fails else 0)
