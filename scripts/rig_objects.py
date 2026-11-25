#!/usr/bin/env python
"""3D object models for the replay viewer from a multi-camera rig (scripts/multicam_rig.py output).

  ~/src/pixarmesh/.venv/bin/python scripts/rig_objects.py RIG_WORK --out deploy/recorder/replay/demo --name multicam
      [--prompt "desk. computer monitor. chair. ..."] [--replay deploy/recorder/replay/demo/multicam.json]

Every camera's (undistorted) rig frame is searched with Grounding DINO tiny + SAM ViT-B (open-vocabulary boxes ->
masks). The masked pixels are lifted to 3D with that camera's VGGT depth map (confident pixels only) and
RIG_WORK/calib/rig.json's world_from_vggt transform, so they land in the replay's metric world frame (Z up). The
mesh is the depth grid itself: neighbouring pixels of the mask become triangles unless the depth jumps (> 6 %)
between them, so each view contributes the surface it actually sees, coloured per vertex from the frame. The same
object seen by several cameras (same label, overlapping 10 cm voxels) is merged into one model, which fills in
sides one camera cannot see. People are never modelled (they move; the replay has its own figure).

Output: OUT/NAME_objects/<i>_<label>.glb (vertex-coloured triangle meshes, positions in WORLD metres, Z up: the viewer
adds them to its Z-up world group, so they deliberately skip glTF's Y-up convention) and OUT/NAME_objects.json
(label, score, views, centre, size, file); with --replay, also registers assets.objects in that replay JSON.
Limits: only surfaces some camera sees are modelled (backs and undersides stay open); thin parts (desk legs,
cables) are noisy; scale inherits the rig's assumed ceiling/desk heights.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scene_objects_pixarmesh import detect_and_segment  # noqa: E402

DEFAULT_PROMPT = ("desk. computer monitor. chair. shelf. cardboard box. cat tree. whiteboard. lamp. table. "
                  "cabinet. coat.")
SKIP = {"person", "man", "woman", "people"}


def grid_mesh(mask, depth, conf_ok, K, E, tf, rgb, jump=0.06):
    """Mask on the depth grid -> (world vertices, colours, triangles)."""
    h, w = depth.shape
    m = mask & conf_ok & (depth > 0)
    idx = -np.ones((h, w), np.int64)
    ys, xs = np.nonzero(m)
    idx[ys, xs] = np.arange(len(ys))
    d = depth[ys, xs]
    cam = np.stack([(xs - K[0, 2]) / K[0, 0] * d, (ys - K[1, 2]) / K[1, 1] * d, d], 1)
    pv = (cam - E[:, 3]) @ E[:, :3]
    world = (pv - tf["origin"]) @ tf["R"].T * tf["scale"]
    col = rgb[ys, xs]
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    D = depth

    def smooth(*pairs):
        ok = np.ones(a.shape, bool)
        for p, q in pairs:
            ok &= np.abs(p - q) <= jump * np.minimum(p, q)
        return ok

    d00, d01, d10, d11 = D[:-1, :-1], D[:-1, 1:], D[1:, :-1], D[1:, 1:]
    t1 = (a >= 0) & (b >= 0) & (c >= 0) & smooth((d00, d01), (d00, d10), (d01, d10))
    t2 = (b >= 0) & (e >= 0) & (c >= 0) & smooth((d01, d11), (d11, d10), (d01, d10))
    tris = np.concatenate([np.stack([a[t1], c[t1], b[t1]], 1), np.stack([b[t2], c[t2], e[t2]], 1)])
    return world, col, tris


def canonical(raw: str, classes: list[str]) -> str | None:
    """Grounding DINO's text label (often several phrases run together: "tree lamp", "shelf cabinet") -> the
    prompt class that matches most of it, longest phrase first, else None."""
    raw = raw.lower()
    best = None
    for c in sorted(classes, key=len, reverse=True):
        words = c.split()
        hit = sum(w in raw.split() for w in words) / len(words)
        if hit > 0.5 and (best is None or hit > best[0]):
            best = (hit, c)
    return best[1] if best else None


def vlm_label(url: str, img_rgb, box, classes: list[str]) -> str:
    """Ask a local OpenAI-compatible vision model (gpuserver Qwen3.8 27B + vision projector, llama-server) to name the object in the box.
    Returns one of classes, or "other"."""
    import base64
    import requests
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(t)) for t in box)
    mx, my = int(0.25 * (x2 - x1)) + 20, int(0.25 * (y2 - y1)) + 20
    cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
    crop = cv2.cvtColor(img_rgb[cy1:cy2, cx1:cx2].copy(), cv2.COLOR_RGB2BGR)
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 3)
    ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    opts = ", ".join(classes + ["person", "other"])
    msg = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}},
           {"type": "text", "text": f"What is the main object inside the red box? Answer with exactly one of: {opts}. "
                                    "Name only, no other words."}]
    try:
        r = requests.post(url.rstrip("/") + "/v1/chat/completions", timeout=60,
                          json={"messages": [{"role": "user", "content": msg}], "temperature": 0, "max_tokens": 8,
                                "chat_template_kwargs": {"enable_thinking": False}})
        ans = r.json()["choices"][0]["message"]["content"].strip().lower().strip(".")
    except Exception as e:  # noqa: BLE001
        print(f"  vlm failed ({e}); keeping the detector label")
        return canonical(" ".join(classes), classes) or "other"
    for c in sorted(classes + ["person"], key=len, reverse=True):
        if c in ans:
            return c
    return "other"


def voxels(P, size=0.1):
    return set(map(tuple, np.floor(P / size).astype(np.int64)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--replay", help="register assets.objects in this replay JSON")
    ap.add_argument("--box-thr", type=float, default=0.3)
    ap.add_argument("--conf-pct", type=float, default=75.0, help="keep this %% most confident depth pixels")
    ap.add_argument("--min-tris", type=int, default=150)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--max-bottom", type=float, default=1.3, help="drop objects whose bottom (2nd percentile Z) is "
                    "higher than this (m): furniture stands on something; floating parts are depth/label errors")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vlm-url", help="vision LLM (OpenAI API) that names each detection, e.g. http://203.0.113.20:5920 (Qwen3.8 27B + vision)")
    a = ap.parse_args()
    import trimesh

    classes = [c.strip() for c in a.prompt.split(".") if c.strip()]
    work = Path(a.work)
    z = np.load(work / "vggt.npz")
    rig = json.loads((work / "calib/rig.json").read_text())
    tf = {k: np.asarray(v, float) for k, v in rig["world_from_vggt"].items()}
    names = [str(n) for n in z["names"]]
    h, w = z["depth"].shape[1:]
    parts = []                                       # one entry per (view, detection)
    for i, nm in enumerate(names):
        img = cv2.imread(str(work / "images" / nm))[:, :, ::-1].copy()
        boxes, scores, labels, masks = detect_and_segment(img, a.prompt, a.device, box_thr=a.box_thr)
        depth, conf = z["depth"][i].astype(np.float32), z["conf"][i]
        conf_ok = conf >= np.percentile(conf, 100 - a.conf_pct)
        rgb_net = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        print(f"{nm}: {len(boxes)} detections: " + ", ".join(f"{l} {s:.2f}" for l, s in zip(labels, scores)))
        for b, s, lab, mk in zip(boxes, scores, labels, masks):
            dino = lab
            lab = canonical(lab, classes)
            if a.vlm_url:
                v = vlm_label(a.vlm_url, img, b, classes)
                print(f"  {nm} box {[int(t) for t in b]}: dino '{dino}' -> {lab}; vlm -> {v}")
                lab = v if v != "other" else None
            if lab is None or lab in SKIP:
                continue
            small = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            small = cv2.erode(small.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # drop edge bleed
            V, C, T = grid_mesh(small, depth, conf_ok, z["intrinsic"][i], z["extrinsic"][i], tf, rgb_net)
            if len(T) < a.min_tris:
                continue
            used = np.unique(T)
            parts.append({"label": lab, "score": float(s), "view": nm.rsplit(".", 1)[0], "V": V, "C": C, "T": T,
                          "vox": voxels(V[used])})

    # merge the same object across views: same label and >= 15 % voxel overlap (of the smaller part)
    groups: list[list[dict]] = []
    for p in sorted(parts, key=lambda p: -p["score"]):
        for g in groups:
            if g[0]["label"] != p["label"]:
                continue
            gv = set().union(*(q["vox"] for q in g))
            if len(gv & p["vox"]) >= 0.15 * min(len(gv), len(p["vox"])):
                g.append(p)
                break
        else:
            groups.append([p])

    out_dir = Path(a.out) / f"{a.name}_objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.glb"):
        f.unlink()
    items = []
    for g in groups:
        Vs, Cs, Ts, off = [], [], [], 0
        for p in g:
            Vs.append(p["V"]); Cs.append(p["C"]); Ts.append(p["T"] + off); off += len(p["V"])
        V, C, T = np.vstack(Vs), np.vstack(Cs), np.vstack(Ts)
        mesh = trimesh.Trimesh(V, T, vertex_colors=np.c_[C, np.full(len(C), 255)].astype(np.uint8), process=True)
        mesh.remove_unreferenced_vertices()
        # robust extent (2-98 %): stray depth pixels at mask borders must not inflate the box
        lo, hi = np.percentile(mesh.vertices, 2, 0), np.percentile(mesh.vertices, 98, 0)
        top = max(p["score"] for p in g)
        if top < a.min_score or lo[2] > a.max_bottom:
            print(f"skip {g[0]['label']} ({', '.join(p['view'] for p in g)}): score {top:.2f}, bottom {lo[2]:.2f} m")
            continue
        k = len(items)
        fn = f"{k:02d}_{re.sub('[^a-z0-9]+', '_', g[0]['label'])}.glb"
        mesh.export(out_dir / fn)
        items.append({"label": g[0]["label"], "score": round(max(p["score"] for p in g), 3),
                      "views": sorted({p["view"] for p in g}), "centre": ((lo + hi) / 2).round(3).tolist(),
                      "size": (hi - lo).round(3).tolist(), "triangles": int(len(mesh.faces)),
                      "file": f"{a.name}_objects/{fn}"})
        print(f"{fn}: {len(g)} view(s) {items[-1]['views']}, {len(mesh.faces)} triangles, size {items[-1]['size']} m")
    meta = {"source": "scripts/rig_objects.py: Grounding DINO + SAM masks on each rig view, lifted with VGGT depth, "
                      "merged across views", "frame": "replay world, metres, Z up", "prompt": a.prompt,
            "note": "only surfaces a camera sees are modelled; scale follows the rig's assumed ceiling/desk heights",
            "objects": items}
    (Path(a.out) / f"{a.name}_objects.json").write_text(json.dumps(meta, indent=1))
    if a.replay:
        rp = Path(a.replay)
        tr = json.loads(rp.read_text())
        tr.setdefault("assets", {})["objects"] = f"{a.name}_objects.json"
        rp.write_text(json.dumps(tr, separators=(",", ":")))
    print(f"{len(items)} objects -> {out_dir}")


if __name__ == "__main__":
    main()
