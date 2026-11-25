"""Export the re-ID v2 candidates to ONNX for squirrel_guard.reid_v2 (optionally int8 dynamic-quantised).

Usage:
    .venv/bin/python scripts/export_reid_v2.py clipreid CKPT [--int8] [--out models/]
    .venv/bin/python scripts/export_reid_v2.py solider CKPT --size s|t --solider-src DIR [--int8]
    .venv/bin/python scripts/export_reid_v2.py osnet_ain CKPT --torchreid-src FILE

Checkpoints (official release locations; sha256 of the files used in docs/reid-gait.md). Put each download in an
empty directory of its own; they are loaded with torch.load(weights_only=True), never unpickled as code.
  clipreid   MSMT17_clipreid_ViT-B-16_60.pth, "ViT-CLIP-ReID / MSMT17 / model" in the README of
             https://github.com/Syliz517/CLIP-ReID (Google Drive id 1BVaZo93kOksYLjFNH3Gf7JxIbPlWSkcO). MIT.
  solider    swin_{small,tiny}_msmt17.pth, README of https://github.com/tinyvision/SOLIDER-REID (Drive ids
             1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b / 10YLhMbwvmxZl3gTVo2BN_828SKZHdCjr). MIT. The Swin definition is
             loaded from a clone of that repo (--solider-src; model/backbones/swin_transformer.py, its mmcv import is
             stubbed) because it is ~1400 lines.
  osnet_ain  osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_..._jitter.pth, torchreid MODEL_ZOO "cross-domain,
             msmt17 (all) ->" table, https://github.com/KaiyangZhou/deep-person-reid (Drive id
             1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal). MIT. --torchreid-src = torchreid/models/osnet_ain.py from that repo.

Writes models/<name>.onnx (and <name>.int8.onnx with --int8): input "images" (N, 3, H, W) float32 RGB normalised,
output "features" (N, D), the tensor each project's own test code ranks with (TEST.NECK_FEAT 'before', i.e. before
the BNNeck; L2 normalisation is done by reid_v2.Embedder). ONNX metadata sg_input_h/w, sg_mean, sg_std, sg_model
carry the preprocessing.
    clipreid_vitb16_msmt17   256x128, mean/std 0.5, D = 1280 = [CLS 768 after ln_post, its 512-d CLIP projection]
    solider_swin_{s,t}_msmt17 384x128, mean/std 0.5, D = 768, semantic weight 0.2 (the repo's test setting)
    osnet_ain_x1_0_msmt17    256x128, ImageNet mean/std, D = 512

Weights were trained on MSMT17 (research dataset): LOCAL use only, never publish the files or the embeddings.

------------------------------------------------------------------------------------------------------------
The CLIP-ReID image encoder below is condensed from https://github.com/Syliz517/CLIP-ReID (commit eb1898b7,
model/clip/model.py VisionTransformer + model/make_model_clipreid.py build_transformer), MIT licence, which itself
derives from OpenAI CLIP (MIT). Same parameter names after stripping "image_encoder." and mapping
transformer.resblocks -> blocks; SIE camera embeddings are not used (the plain ViT-CLIP-ReID checkpoint has none).
------------------------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODELS = Path(__file__).resolve().parent.parent / "models"
HALF = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# --- CLIP-ReID ViT-B/16 (vendored, MIT; see above) -----------------------------------------------------------
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h)
        self.ln_1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(d, d * 4)), ("gelu", QuickGELU()),
                                              ("c_proj", nn.Linear(d * 4, d))]))
        self.ln_2 = nn.LayerNorm(d)

    def forward(self, x):                                      # x: (L, N, D), as in CLIP
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class ClipReIDImage(nn.Module):
    """256x128 input, 16x16 patches with stride 16 -> 16x8 grid + CLS."""

    def __init__(self, grid=(16, 8), width=768, layers=12, heads=12, out=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 16, 16, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros(grid[0] * grid[1] + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(*[Block(width, heads) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.zeros(width, out))

    def forward(self, x):
        x = self.conv1(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.expand(x.shape[0], 1, -1), x], 1) + self.positional_embedding
        x = self.blocks(self.ln_pre(x).permute(1, 0, 2)).permute(1, 0, 2)
        cls = self.ln_post(x[:, 0])
        return torch.cat([cls, cls @ self.proj], 1)


def build_clipreid(ckpt: str) -> nn.Module:
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k[len("image_encoder."):].replace("transformer.resblocks.", "blocks."): v
          for k, v in sd.items() if k.startswith("image_encoder.")}
    m = ClipReIDImage()
    m.load_state_dict(sd, strict=True)
    return m.eval()


# --- SOLIDER-REID Swin (definition loaded from the official repo clone) ---------------------------------------
def build_solider(ckpt: str, size: str, src: str, semantic_weight: float = 0.2) -> nn.Module:
    mm, mr = types.ModuleType("mmcv"), types.ModuleType("mmcv.runner")
    mr.load_checkpoint = None                                  # only used for loading ImageNet pretraining
    mm.runner = mr
    sys.modules.setdefault("mmcv", mm)
    sys.modules.setdefault("mmcv.runner", mr)
    spec = importlib.util.spec_from_file_location("solider_swin", Path(src) / "model/backbones/swin_transformer.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)

    def window_reverse(self, windows, H, W):
        """Upstream computes B = int(...), which a traced ONNX export freezes to the export batch size (silently
        wrong embeddings for any other batch size, including 1). Same reshape with -1 for the batch."""
        ws = self.window_size
        x = windows.view(-1, H // ws, W // ws, ws, ws, windows.shape[-1])
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, windows.shape[-1])
    st.ShiftWindowMSA.window_reverse = window_reverse
    fac = {"t": st.swin_tiny_patch4_window7_224, "s": st.swin_small_patch4_window7_224}[size]
    base = fac(img_size=(384, 128), drop_path_rate=0.0, drop_rate=0.0, attn_drop_rate=0.0, pretrained=None,
               convert_weights=False, semantic_weight=semantic_weight)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    base.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("base.")}, strict=True)

    class Solider(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base

        def forward(self, x):
            w = torch.full((x.shape[0], 1), semantic_weight, dtype=x.dtype)
            feat, _ = self.base(x, semantic_weight=torch.cat([w, 1 - w], -1))   # SwinTransformer.forward: [w, 1-w]
            return feat
    return Solider().eval()


# --- torchreid OSNet-AIN x1.0 ---------------------------------------------------------------------------------
def build_osnet_ain(ckpt: str, src: str) -> nn.Module:
    spec = importlib.util.spec_from_file_location("osnet_ain", src)
    oa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oa)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module."): v for k, v in sd.get("state_dict", sd).items()}
    m = oa.osnet_ain_x1_0(num_classes=sd["classifier.weight"].shape[0], pretrained=False, loss="softmax")
    m.load_state_dict(sd, strict=True)
    return m.eval()                                            # eval forward = 512-d feature before the classifier


def export(m: nn.Module, name: str, hw: tuple, norm: tuple, out: Path, int8: bool) -> None:
    import onnx
    import onnxruntime as ort
    out.mkdir(parents=True, exist_ok=True)
    x = torch.randn(2, 3, *hw)
    p = out / f"{name}.onnx"
    torch.onnx.export(m, (x,), str(p), input_names=["images"], output_names=["features"],
                      dynamic_axes={"images": {0: "n"}, "features": {0: "n"}}, opset_version=17, dynamo=False)
    paths = [p]
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre, q = out / f"{name}.pre.onnx", out / f"{name}.int8.onnx"
        quant_pre_process(str(p), str(pre), skip_symbolic_shape=True)
        quantize_dynamic(str(pre), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        pre.unlink()
        paths.append(q)
    for path in paths:
        mp = onnx.load(str(path))
        meta = {"sg_input_h": str(hw[0]), "sg_input_w": str(hw[1]), "sg_mean": json.dumps(norm[0]),
                "sg_std": json.dumps(norm[1]), "sg_model": name}
        del mp.metadata_props[:]
        for k, v in meta.items():
            mp.metadata_props.add(key=k, value=v)
        onnx.save(mp, str(path))
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        worst = 1.0
        for xb in (x, x[:1], torch.cat([x, x[:1]])):          # traced batch 2, and batch 1 / 3 (frozen-shape bugs)
            with torch.no_grad():
                rb = m(xb).numpy()
            o = sess.run(None, {"images": xb.numpy()})[0]
            worst = min(worst, float(((o * rb).sum(1) / (np.linalg.norm(o, axis=1) * np.linalg.norm(rb, axis=1))).min()))
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB, out {o.shape}, min cos vs torch over batch 1/2/3 "
              f"{worst:.5f}")
        if worst < (0.95 if path.name.endswith(".int8.onnx") else 0.9999):   # int8 on random noise: ~0.97-0.99
            raise SystemExit(f"{path.name}: ONNX output does not match PyTorch")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", choices=["clipreid", "solider", "osnet_ain"])
    ap.add_argument("ckpt")
    ap.add_argument("--size", choices=["s", "t"], default="s", help="SOLIDER Swin size")
    ap.add_argument("--solider-src", help="clone of github.com/tinyvision/SOLIDER-REID")
    ap.add_argument("--torchreid-src", help="torchreid/models/osnet_ain.py from github.com/KaiyangZhou/deep-person-reid")
    ap.add_argument("--int8", action="store_true", help="also write a dynamic int8 (MatMul/Gemm) model")
    ap.add_argument("--out", default=str(MODELS))
    a = ap.parse_args()
    out = Path(a.out)
    if a.model == "clipreid":
        export(build_clipreid(a.ckpt), "clipreid_vitb16_msmt17", (256, 128), HALF, out, a.int8)
    elif a.model == "solider":
        if not a.solider_src:
            ap.error("solider needs --solider-src")
        export(build_solider(a.ckpt, a.size, a.solider_src), f"solider_swin_{a.size}_msmt17", (384, 128), HALF, out, a.int8)
    else:
        if not a.torchreid_src:
            ap.error("osnet_ain needs --torchreid-src")
        export(build_osnet_ain(a.ckpt, a.torchreid_src), "osnet_ain_x1_0_msmt17", (256, 128), IMAGENET, out, a.int8)


if __name__ == "__main__":
    main()
