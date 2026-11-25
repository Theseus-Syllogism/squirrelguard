#!/usr/bin/env python3
"""One empty frame from the fixed camera -> per-object meshes + placements in a replay's world frame (PixARMesh).

  ~/src/pixarmesh/.venv/bin/python scripts/scene_objects_pixarmesh.py FRAME.png REPLAY.json \
      [--out ~/.cache/squirrel-guard-test/scene-objects] [--pixarmesh ~/src/pixarmesh/PixARMesh] \
      [--prompt "chair. desk. computer monitor. shelf. lamp. coat."] [--ceiling 2.5] [--max-objects 8]

Evaluation tool, not part of the runtime. Needs its own environment (docs/scene-objects.md): torch (CUDA), the
PixARMesh repo (CC BY-SA 4.0) and NVIDIA's EdgeRunner `meto` tokenizer (NVIDIA Source Code License: non-commercial
research/evaluation use on NVIDIA GPUs only). Models are pulled from Hugging Face on first use.

Pipeline (each step stands in for one of PixARMesh's own inputs):
  1. undistort FRAME with the replay camera's K/D (same K kept), so a pinhole back-projection is valid;
  2. metric depth: Depth Anything V2 Metric-Indoor Small (Apache-2.0) in place of their Depth Pro; optionally
     rescaled so the ceiling (world Z of the top-of-image pixels, median) sits at --ceiling metres;
  3. instance masks: Grounding DINO tiny boxes + SAM ViT-B masks (both Apache-2.0) = their Grounded-SAM;
  4. per object: a 4:3 crop (their renders are 648x484) around the box, resized to 648x484, K adjusted;
     point cloud back-projected, flipped to their Y-up camera frame and gravity-rectified (pitch removed using
     the replay R), normalised to [-0.95, 0.95] like their loader; PixARMesh-EdgeRunner predicts the
     gravity-aligned box (layout) and then the mesh;
  5. the mesh is placed by the predicted box (their recover_box_transform), un-normalised to metres and mapped
     to the replay world (Z up): p_world = C + A^T p_rect, A = rows (left, up, horizontal-forward) in world.
Output: OUT/objects/<i>_<label>.glb (glTF Y-up, metres, origin = centre of the mesh's world bounding box) and OUT/objects.json (schema in docs/scene-objects.md), plus OUT/debug/ (masks, depth).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

H_IN, W_IN = 484, 648          # PixARMesh input size (3D-FRONT renders from InstPIFu)
Y_UP = np.diag([-1.0, -1.0, 1.0])  # OpenCV camera (x right, y down, z fwd) -> their (x left, y up, z fwd)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rect_axes(R_wc: np.ndarray) -> np.ndarray:
    """Rows: left, up, forward(horizontal) of the gravity-rectified camera frame, in world (Z up) coordinates."""
    up = np.array([0.0, 0.0, 1.0])
    f = R_wc[:, 2].copy(); f[2] = 0; f /= np.linalg.norm(f)
    left = np.cross(up, f)
    return np.stack([left, up, f])


def depth_metric(img_rgb, dev):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForDepthEstimation.from_pretrained(name).to(dev).eval()
    with torch.no_grad():
        inp = proc(images=img_rgb, return_tensors="pt").to(dev)
        d = model(**inp).predicted_depth
        d = torch.nn.functional.interpolate(d[:, None], size=img_rgb.shape[:2], mode="bicubic")[0, 0]
    out = d.float().cpu().numpy()
    del model; torch.cuda.empty_cache()
    return out


def detect_and_segment(img_rgb, prompt, dev, box_thr=0.3, text_thr=0.25):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
    pil = Image.fromarray(img_rgb)
    gid = "IDEA-Research/grounding-dino-tiny"
    gp = AutoProcessor.from_pretrained(gid)
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(gid).to(dev).eval()
    with torch.no_grad():
        inp = gp(images=pil, text=prompt, return_tensors="pt").to(dev)
        out = gm(**inp)
    res = gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
                                                    target_sizes=[img_rgb.shape[:2]])[0]
    del gm; torch.cuda.empty_cache()
    boxes = res["boxes"].cpu().numpy(); scores = res["scores"].cpu().numpy()
    labels = [str(x).strip() or "object" for x in res.get("text_labels", res["labels"])]
    keep = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                            scores.tolist(), box_thr, 0.6) if len(boxes) else []
    keep = sorted(np.array(keep).reshape(-1).tolist(), key=lambda i: -scores[i])
    boxes, scores, labels = boxes[keep], scores[keep], [labels[i] for i in keep]
    if not len(boxes):
        return boxes, scores, labels, np.zeros((0, *img_rgb.shape[:2]), bool)
    sp = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dt = torch.float16 if dev == "cuda" else torch.float32
    sm = SamModel.from_pretrained("facebook/sam-vit-base", torch_dtype=dt).to(dev).eval()
    masks = []
    with torch.no_grad():
        emb = sm.get_image_embeddings(sp(pil, return_tensors="pt")["pixel_values"].to(dev, dt))   # image encoded once
        for b in boxes:
            inp = sp(pil, input_boxes=[[b.tolist()]], return_tensors="pt").to(dev)
            o = sm(image_embeddings=emb, input_boxes=inp["input_boxes"].to(dt), multimask_output=False)
            m = sp.image_processor.post_process_masks(o.pred_masks.float().cpu(), inp["original_sizes"].cpu(),
                                                      inp["reshaped_input_sizes"].cpu())[0][0, 0].numpy()
            masks.append(m.astype(bool))
    del sm; torch.cuda.empty_cache()
    return boxes, scores, labels, np.stack(masks)


def crop_for(box, W, H):
    """4:3 crop (full height) whose centre is as close as possible to the box centre."""
    cw = int(round(H * W_IN / H_IN))
    cx = 0.5 * (box[0] + box[2])
    x0 = int(np.clip(round(cx - cw / 2), 0, max(W - cw, 0)))
    return x0, cw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame"); ap.add_argument("replay")
    ap.add_argument("--out", default=str(Path.home() / ".cache/squirrel-guard-test/scene-objects"))
    ap.add_argument("--pixarmesh", default=str(Path.home() / "src/pixarmesh/PixARMesh"))
    ap.add_argument("--checkpoint", default="zx1239856/PixARMesh-EdgeRunner")
    ap.add_argument("--prompt", default="chair. desk. table. computer monitor. shelf. lamp. coat. cat tree.")
    ap.add_argument("--ceiling", type=float, default=0.0, help="rescale depth so the ceiling is at this Z (0 = off)")
    ap.add_argument("--max-objects", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=0.004, help="min mask area as a fraction of the frame")
    ap.add_argument("--max-new-tokens", type=int, default=12000,
                    help="mesh token cap (their limit 40960; the KV cache of long meshes does not fit in ~3 GB)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import torch
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out).expanduser(); (out / "objects").mkdir(parents=True, exist_ok=True); (out / "debug").mkdir(exist_ok=True)
    tr = json.loads(Path(a.replay).read_text()); c = tr["camera"]
    K = np.asarray(c["K"], float); D = np.asarray(c["D"], float); R = np.asarray(c["R"], float); C = np.asarray(c["C"], float)
    bgr = cv2.imread(a.frame)
    if (bgr.shape[1], bgr.shape[0]) != tuple(c["size"]):
        bgr = cv2.resize(bgr, tuple(c["size"]), interpolation=cv2.INTER_AREA)
    und = cv2.undistort(bgr, K, D, None, K)
    rgb = und[:, :, ::-1].copy(); H, W = rgb.shape[:2]
    cv2.imwrite(str(out / "debug/undistorted.jpg"), und)
    A = rect_axes(R)
    timings = {}

    # --- depth ---------------------------------------------------------------------------------------------------
    t0 = time.time(); torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    depth = depth_metric(rgb, dev)
    ys, xs = np.mgrid[0:H, 0:W]
    rays_cam = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    Pw = C + (rays_cam * depth[..., None]) @ R.T
    top = Pw[: H // 6, W // 4: 3 * W // 4, 2]
    ceil_raw = float(np.median(top)); scale = 1.0
    if a.ceiling > 0:
        scale = (a.ceiling - C[2]) / max(ceil_raw - C[2], 1e-3)
        depth = depth * scale
    log(f"depth: ceiling (median world Z, top-centre band) {ceil_raw:.2f} m raw, scale x{scale:.3f}")
    timings["depth_s"] = time.time() - t0
    np.save(out / "debug/depth.npy", depth.astype(np.float32))
    cv2.imwrite(str(out / "debug/depth.png"), cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / max(depth.max(), 1)), cv2.COLORMAP_TURBO))

    # --- masks ---------------------------------------------------------------------------------------------------
    t0 = time.time()
    boxes, scores, labels, masks = detect_and_segment(rgb, a.prompt, dev)
    timings["masks_s"] = time.time() - t0
    keep = [i for i in range(len(boxes)) if masks[i].mean() >= a.min_area][: a.max_objects]
    vis = und.copy()
    for j, i in enumerate(keep):
        col = np.array(cv2.applyColorMap(np.uint8([[37 * j % 255]]), cv2.COLORMAP_HSV)[0, 0], float)
        vis[masks[i]] = (0.5 * vis[masks[i]] + 0.5 * col).astype(np.uint8)
        x0, y0, x1, y1 = boxes[i].astype(int)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col.tolist(), 2)
        cv2.putText(vis, f"{j}:{labels[i]} {scores[i]:.2f}", (x0 + 4, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col.tolist(), 2)
    cv2.imwrite(str(out / "debug/masks.jpg"), vis)
    log("objects:", [(labels[i], round(float(scores[i]), 2)) for i in keep])

    # --- PixARMesh -----------------------------------------------------------------------------------------------
    sys.path.insert(0, str(Path(a.pixarmesh).expanduser()))
    os.chdir(Path(a.pixarmesh).expanduser())
    from transformers import AutoImageProcessor
    from src.utils.inference import (prepare_model_for_inference, get_prefix_allowed_tokens_fn_edgerunner,
                                     decode_mesh_edgerunner, recover_box_transform)
    from src.data.collator import get_mesh_data_collator
    from src.data import utils as du
    import trimesh

    import src.models.pc_edgerunner.attention as pa  # their no-flash-attn fallback builds the full N x M matrix (OOM)
    _att = pa.attention

    def _sdpa(q, k, v, mask_q=None, mask_kv=None, dropout=0, causal=False, **kw):
        if mask_q is not None or mask_kv is not None:
            return _att(q, k, v, mask_q, mask_kv, dropout, causal, **kw)
        o = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                             is_causal=causal and q.shape[1] > 1)
        return o.transpose(1, 2).contiguous()
    pa.attention = _sdpa
    _ms = torch.Tensor.masked_scatter_                # bf16 weights + autocast: cond features come back fp32
    torch.Tensor.masked_scatter_ = lambda self, m, src: _ms(self, m, src.to(self.dtype))
    import src.models.utils as mu                    # their loader hard-codes flash_attention_2; use torch SDPA instead
    _orig = mu.ShapeOPT.from_pretrained.__func__
    mu.ShapeOPT.from_pretrained = classmethod(lambda cls, *p, **k: _orig(cls, *p, **{**k, "attn_implementation": "sdpa"}))
    t0 = time.time()
    model, mcfg, dcfg = prepare_model_for_inference(False, a.checkpoint)
    model.to(torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()   # cast on CPU first: fp32 does not fit next to llama
    imgproc = AutoImageProcessor.from_pretrained(dcfg.image_preprocessor, size_divisor=dcfg.image_size_divisor)
    coll = get_mesh_data_collator(dcfg, mcfg)
    timings["load_model_s"] = time.time() - t0
    prefix = [coll.pc_token_id] * coll.prefix_len + [coll.bos_token_id]
    kern = np.ones((3, 3), np.uint8)

    results = []
    for j, i in enumerate(keep):
        t0 = time.time()
        x0, cw = crop_for(boxes[i], W, H)
        sx, sy = W_IN / cw, H_IN / H
        img_c = cv2.resize(rgb[:, x0:x0 + cw], (W_IN, H_IN), interpolation=cv2.INTER_AREA)
        dep_c = cv2.resize(depth[:, x0:x0 + cw], (W_IN, H_IN), interpolation=cv2.INTER_NEAREST)
        m_c = cv2.resize(masks[i, :, x0:x0 + cw].astype(np.uint8), (W_IN, H_IN), interpolation=cv2.INTER_NEAREST)
        m_c = cv2.erode(m_c, kern, iterations=dcfg.mask_erosion_size).astype(bool)   # their GT masks are eroded by 3
        Kc = np.array([[K[0, 0] * sx, 0, (K[0, 2] - x0 + 0.5) * sx - 0.5], [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]])
        valid = (dep_c > 1e-6) & (dep_c < 10.0)                  # their GT depth tops out at 10 m
        dep_c = np.where(valid, dep_c, 0.0)
        pc, pix = du.back_project_depth(dep_c.astype(np.float32), np.linalg.inv(Kc).astype(np.float32), return_pix_coords=True)
        pix = (pix + 0.5) / np.array([W_IN, H_IN]) * 2 - 1
        R_rect = A @ R @ Y_UP                                    # their camera (y up) -> gravity-rectified frame
        pc = (pc.reshape(-1, 3) @ Y_UP.T) @ R_rect.T
        pc[~valid.reshape(-1)] = 1e-6
        _, center, nscale = du.normalize_vertices(np.concatenate([np.zeros((8, 3)), pc]), bound=dcfg.norm_bound, return_all=True)
        pcn = ((pc - center) * nscale).astype(np.float32)
        obj = pcn[m_c.reshape(-1)]
        if len(obj) < 50:
            log(f"skip {j}: {len(obj)} points"); continue
        s_obj, inds = du.random_sample_point_clouds(obj, dcfg.num_points, return_inds=True)
        s_ctx, cinds = du.random_sample_point_clouds(pcn, dcfg.num_ctx_points, return_inds=True)
        pix_f = pix.reshape(-1, 2)
        T = lambda x: torch.as_tensor(np.asarray(x, np.float32)[None]).to(dev)
        ids = torch.as_tensor([prefix], dtype=torch.long, device=dev)
        pv = imgproc([img_c], return_tensors="pt")["pixel_values"].to(dev)
        cond = dict(cond_pcs=T(s_obj), cond_pcs_2d=T(pix_f[m_c.reshape(-1)][inds]), ctx_pcs=T(s_ctx), ctx_pcs_2d=T(pix_f[cinds]))
        try:
            with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                emb = model.get_inputs_with_cond(ids, pixel_values=pv, **cond)
                lay = model.generate(inputs_embeds=emb, max_new_tokens=25, use_cache=True, do_sample=False)
                lay_np = lay.cpu().numpy()
                pose = du.dequantize_points((lay_np[:, :-1] - mcfg.pos_token_offset).reshape(-1, 8, 3), mcfg.num_pos_tokens)[0]
                ids2 = torch.cat([ids, lay], 1)
                emb = model.get_inputs_with_cond(ids2, pixel_values=pv, **cond)
                toks = model.generate(inputs_embeds=emb, max_new_tokens=min(coll.max_seq_length - ids2.shape[1], a.max_new_tokens),
                                      use_cache=True, do_sample=False,
                                      prefix_allowed_tokens_fn=get_prefix_allowed_tokens_fn_edgerunner(model, batch_size=1))
        except torch.OutOfMemoryError:
            log(f"skip {j}: CUDA out of memory (mesh sequence too long for the free VRAM; try --max-new-tokens)")
            emb = None; torch.cuda.empty_cache(); continue
        toks = toks.cpu().numpy()[0]
        toks = toks[toks != coll.pad_token_id]
        eos = (toks == model.config.eos_token_id).nonzero()[0]
        if len(eos):
            toks = toks[: eos[0]]
        try:
            mesh = decode_mesh_edgerunner(toks, coll.tokenizer, clean=True)
        except Exception as e:  # noqa: BLE001
            log(f"skip {j}: decode failed ({e})"); continue
        if len(mesh.faces) == 0:
            log(f"skip {j}: empty mesh"); continue
        Tbox = recover_box_transform(du.make_3d_bbox(*mesh.bounds), pose)[-1]
        mesh.apply_transform(Tbox)                               # normalised rectified frame (y up)
        v = mesh.vertices / nscale + center                      # metres, rectified camera frame
        vw = C + v @ A                                           # world, Z up
        boxw = C + (pose / nscale + center) @ A
        pos = 0.5 * (vw.min(0) + vw.max(0))
        local = vw - pos
        g = trimesh.Trimesh(np.c_[local[:, 0], local[:, 2], -local[:, 1]], mesh.faces, process=False)   # glTF Y-up
        lab = labels[i].replace(" ", "_")
        fn = f"{j:02d}_{lab}.glb"
        g.export(out / "objects" / fn)
        dt = time.time() - t0
        ext = (vw.max(0) - vw.min(0)).tolist()
        results.append(dict(file=f"objects/{fn}", label=labels[i], score=round(float(scores[i]), 3),
                            position=[round(x, 4) for x in pos.tolist()], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                            size_m=[round(x, 3) for x in ext], box_world=np.round(boxw, 4).tolist(),
                            faces=int(len(mesh.faces)), tokens=int(len(toks)), seconds=round(dt, 1),
                            crop_x0=x0, mask_px=int(masks[i].sum())))
        log(f"{j} {labels[i]}: {len(mesh.faces)} faces, {len(toks)} tokens, size {np.round(ext, 2)} m, at {np.round(pos, 2)}, {dt:.0f}s")
        torch.cuda.empty_cache() if dev == "cuda" else None

    peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0
    meta = dict(source=dict(frame=str(a.frame), replay=str(a.replay), model=a.checkpoint, device=dev,
                            depth="Depth-Anything-V2-Metric-Indoor-Small", masks="grounding-dino-tiny + sam-vit-base",
                            prompt=a.prompt, ceiling_raw_m=round(ceil_raw, 3), depth_scale=round(scale, 4),
                            peak_vram_mib=round(peak), timings_s={k: round(v, 1) for k, v in timings.items()}),
                frame="replay world: X right, Y forward, Z up, metres; glb is glTF Y-up (rotate +90 deg about X)",
                objects=results)
    (out / "objects.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {len(results)} objects -> {out / 'objects.json'}; peak torch VRAM {peak:.0f} MiB")


if __name__ == "__main__":
    main()
