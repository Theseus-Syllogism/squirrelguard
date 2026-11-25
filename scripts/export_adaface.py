"""Export AdaFace IR-50 / IR-101 (CVLFace release) to ONNX for squirrel_guard.faceid_v2.

Usage:
    .venv/bin/python scripts/export_adaface.py SAFETENSORS --arch ir50|ir101 --out models/adaface_ir50_webface4m.onnx

Weights (download into an empty directory; read with safetensors, never executed as code; the repo's
`trust_remote_code` wrapper is NOT used):
  ir50   https://huggingface.co/minchul/cvlface_adaface_ir50_webface4m   model.safetensors (167 MB)
  ir101  https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m model.safetensors (249 MB)
Code licence: MIT (github.com/mk-minchul/AdaFace and github.com/mk-minchul/CVLface). The model cards give no
weight licence and say "follow the license of the training dataset": WebFace4M / WebFace12M are for non-commercial
research only. LOCAL use only: never commit, publish or bake the .onnx into an image (models/ and *.onnx are
git-ignored). The original Google Drive checkpoints are the same networks (BGR input there; CVLFace uses RGB).

The backbone below is written from the published IR architecture (InsightFace-style improved ResNet: BN-conv-BN-
PReLU-conv-BN basic blocks, 7x7x512 -> Linear 512 -> BN1d without affine); parameter names match the checkpoint.
Input "images" (N, 3, 112, 112) float32 RGB, (x/255 - 0.5)/0.5, on the standard ArcFace 5-point aligned crop.
Output "features" (N, 512), NOT normalised: the norm is AdaFace's feature-quality proxy, so faceid_v2 keeps it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

UNITS = {"ir50": (3, 4, 14, 3), "ir101": (3, 13, 30, 3)}


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.shortcut_layer = nn.MaxPool2d(1, stride) if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.PReLU(cout),
            nn.Conv2d(cout, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class IR(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        mods, cin = [], 64
        for depth, n in zip((64, 128, 256, 512), UNITS[arch]):
            for i in range(n):
                mods.append(Block(cin, depth, 2 if i == 0 else 1))
                cin = depth
        self.body = nn.Sequential(*mods)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(0.4), nn.Flatten(),
                                          nn.Linear(512 * 7 * 7, 512), nn.BatchNorm1d(512, affine=False))

    def forward(self, x):
        return self.output_layer(self.body(self.input_layer(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("--arch", choices=sorted(UNITS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from safetensors.torch import load_file
    sd = {k.removeprefix("model.net."): v for k, v in load_file(str(a.weights)).items()}
    # IR-101 WebFace12M carries ~200k denormal weights, which made CPU inference ~5x slower than IR-50: zero them
    sd = {k: (v.masked_fill(v.abs() < 1.18e-38, 0.0) if v.is_floating_point() else v) for k, v in sd.items()}
    net = IR(a.arch).eval()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        sys.exit(f"key mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    x = torch.randn(2, 3, 112, 112)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, x, str(a.out), input_names=["images"], output_names=["features"],
                      dynamic_axes={"images": {0: "n"}, "features": {0: "n"}}, opset_version=17, dynamo=False)
    import onnx
    m = onnx.load(str(a.out))
    for k, v in {"sg_model": f"adaface_{a.arch}", "sg_input": "RGB 112x112 (x/255-0.5)/0.5, ArcFace 5-pt aligned",
                 "sg_licence": "code MIT; weights WebFace research-only, local use"}.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(a.out))
    import onnxruntime as ort
    s = ort.InferenceSession(str(a.out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = net(x).numpy()
    got = s.run(None, {"images": x.numpy()})[0]
    print(a.out, "max abs diff vs torch", float(np.abs(ref - got).max()))


if __name__ == "__main__":
    main()
