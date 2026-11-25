"""Export OpenGait's GaitBase (trained on GREW) and a YOLOv8 segmentation model to ONNX for squirrel_guard.gait_embed.

Usage:
    .venv/bin/python scripts/export_gaitbase.py CKPT [--seg s|n] [--skip-seg]

CKPT is GREW/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-180000.pt from the OpenGait release asset
pretrained_grew_gaitbase.zip (https://github.com/ShiqiYu/OpenGait/releases/tag/v2.0). Unzip it into an empty
directory of its own; it is loaded with torch.load(weights_only=True).

Writes:
    models/gaitbase_grew.onnx   input "sils" (1, T, 64, 44) float32 in [0, 1], T dynamic
                                output "embeddings" (1, 256, 16): 16 horizontal parts x 256 channels
    models/yolov8{s,n}-seg.onnx  Ultralytics YOLOv8 instance segmentation (COCO), 640x640 input

The exported output is the tensor OpenGait's evaluator ranks with (Baseline.forward -> inference_feat
'embeddings' = embed_1, the SeparateFCs output BEFORE the BNNeck); the BNNeck and its 20000-class GREW
classifier are training-only and are left out, which is why the ONNX is ~28 MB and the checkpoint 356 MB.

Network config (configs/gaitbase/gaitbase_da_grew.yaml): model Baseline, backbone ResNet9 with BasicBlock,
channels [64,128,256,512], layers [1,1,1,1], strides [1,2,2,1], no maxpool; SeparateFCs 512->256 x 16 parts;
bin_num [16]. Temporal pooling is a max over all frames (evaluator sample_type all_ordered: whole sequence).

------------------------------------------------------------------------------------------------------------
The network definition below is vendored and condensed from OpenGait (https://github.com/ShiqiYu/OpenGait,
commit 4d228e15), files opengait/modeling/models/baseline.py, opengait/modeling/backbones/resnet.py
(which subclasses torchvision's ResNet/BasicBlock) and opengait/modeling/modules.py.
OpenGait ships no LICENSE file; its README states: "This code is only used for academic purposes, people
cannot use this code for anything that might be considered commercial use." The GREW-trained weights inherit
the GREW dataset's research-only terms. Both are used here only in a private, non-commercial home system.
Ultralytics YOLOv8 code and weights are AGPL-3.0.
------------------------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = Path(__file__).resolve().parent.parent / "models"


# --- vendored from OpenGait (see licence note above) -------------------------------------------------------
class BasicBlock(nn.Module):
    """torchvision.models.resnet.BasicBlock (expansion 1), same parameter names."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class BasicConv2d(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)

    def forward(self, x):
        return self.conv(x)


class ResNet9(nn.Module):
    """OpenGait ResNet9: conv1(3x3,s1) -> bn1 -> relu -> (no maxpool) -> layer1..4, one BasicBlock each."""

    def __init__(self, channels=(64, 128, 256, 512), strides=(1, 2, 2, 1), in_channel=1):
        super().__init__()
        self.conv1 = BasicConv2d(in_channel, channels[0], 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        inp = channels[0]
        for i, (c, s) in enumerate(zip(channels, strides), 1):
            setattr(self, f"layer{i}", nn.Sequential(BasicBlock(inp, c, s)))
            inp = c

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class SetBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.forward_block = block


class SeparateFCs(nn.Module):
    def __init__(self, parts_num=16, in_channels=512, out_channels=256):
        super().__init__()
        self.fc_bin = nn.Parameter(torch.zeros(parts_num, in_channels, out_channels))

    def forward(self, x):                               # [n, c_in, p] -> [n, c_out, p]
        x = x.permute(2, 0, 1).contiguous()
        return x.matmul(self.fc_bin).permute(1, 2, 0).contiguous()


class GaitBaseInfer(nn.Module):
    """Baseline.forward's inference path for ONE sequence: sils [1, T, 64, 44] -> embeddings [1, 256, 16]."""

    def __init__(self):
        super().__init__()
        self.Backbone = SetBlockWrapper(ResNet9())
        self.FCs = SeparateFCs()
        self.bin_num = 16

    def forward(self, sils):
        n, s, h, w = sils.shape
        x = sils.reshape(n * s, 1, h, w)                # SetBlockWrapper: frames through the 2-D backbone
        x = self.Backbone.forward_block(x)              # [n*s, 512, 16, 11]
        x = x.reshape(n, s, *x.shape[1:]).transpose(1, 2)
        x = x.max(dim=2)[0]                             # TP = PackSequenceWrapper(torch.max), seqL=None
        z = x.reshape(n, x.shape[1], self.bin_num, -1)  # HPP, bin_num [16]
        feat = z.mean(-1) + z.max(-1)[0]                # [n, 512, 16]
        return self.FCs(feat)                           # embed_1 = inference_feat['embeddings']
# --- end of vendored code ------------------------------------------------------------------------------------


def export_gait(ckpt: Path, dst: Path) -> None:
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("BNNecks.") and not k.endswith("num_batches_tracked")}
    net = GaitBaseInfer().eval()
    net.load_state_dict(sd, strict=True)
    print(f"loaded {len(sd)} tensors from {ckpt.name}")
    dummy = torch.rand(1, 30, 64, 44)
    torch.onnx.export(net, (dummy,), str(dst), input_names=["sils"], output_names=["embeddings"],
                      dynamic_axes={"sils": {1: "frames"}}, opset_version=17, dynamo=False)
    # verify against torch on real-ish binary inputs of several lengths
    import onnxruntime as ort
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in (1, 7, 30, 90):
        x = (rng.random((1, t, 64, 44)) > 0.6).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"sils": x})[0]
        d = float(np.abs(ref - got).max())
        worst = max(worst, d)
        print(f"  T={t:3d}: max|onnx-torch| = {d:.2e}  (|ref| max {np.abs(ref).max():.2f})")
    print(f"exported {dst} ({dst.stat().st_size / 1e6:.1f} MB), worst diff {worst:.2e}")


def export_seg(size: str, dst: Path) -> None:
    from ultralytics import YOLO
    name = f"yolov8{size}-seg"
    model = YOLO(f"{name}.pt")
    onnx = Path(model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False))
    shutil.move(onnx, dst)
    pt = Path(f"{name}.pt")
    if pt.exists():
        shutil.move(pt, MODELS / pt.name)
    print(f"exported {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--seg", default="s", choices=["n", "s"])
    ap.add_argument("--skip-seg", action="store_true")
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    export_gait(a.ckpt, MODELS / "gaitbase_grew.onnx")
    if not a.skip_seg:
        export_seg(a.seg, MODELS / f"yolov8{a.seg}-seg.onnx")
