"""Export PYSKL's ST-GCN++ (NTU RGB+D 120 XSub, HRNet 2D COCO-17 keypoints, joint stream) to ONNX.

Usage: .venv/bin/python scripts/export_stgcnpp.py [path/to/j.pth]
  Without an argument the checkpoint is downloaded from CKPT_URL into a temp dir (it is ~6 MB) and loaded with
  torch.load(weights_only=True): a plain state dict, no pickled code.
Output: models/stgcnpp_ntu120_hrnet_j.onnx
  input  "keypoint": float32 (N clips, M=2 persons, T=100 frames, V=17 joints, C=3 = x, y, score), already
         preprocessed as in squirrel_guard/action_skeleton.py (pyskl test pipeline)
  output "logits":   float32 (N, 120) class scores (pyskl averages softmax over clips: 'prob' mode)

Source: https://github.com/kennymckormick/pyskl (configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py; reported top-1
84.4 on NTU120 XSub). The model definition below is a trimmed copy of pyskl's
pyskl/models/gcns/stgcn.py, pyskl/models/gcns/utils/{gcn,tcn}.py, pyskl/utils/graph.py and
pyskl/models/heads/simple_head.py (GCNHead), with the mmcv builders replaced by plain torch.nn modules and
the unused options (other graph layouts/modes, aagcn, dgmstcn, training code) removed.

    Copyright (c) OpenMMLab / Haodong Duan (pyskl contributors).
    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
    compliance with the License. You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software distributed under the License is
    distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    Modified for squirrel-guard: mmcv dependencies removed, ONNX export added.

The checkpoint itself is distributed by OpenMMLab with pyskl (Apache-2.0 code), but it was trained on NTU
RGB+D 120, whose terms allow non-commercial research use only.
"""
from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CKPT_URL = "http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth"
EPS = 1e-4


# ---- pyskl/utils/graph.py (layout 'coco', mode 'spatial') ----------------------------------------------------
def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


def coco_spatial_graph() -> np.ndarray:
    num_node = 17
    inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
              (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
    self_link = [(i, i) for i in range(num_node)]
    outward = [(j, i) for (i, j) in inward]
    Iden = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((Iden, In, Out))


# ---- pyskl/models/gcns/utils/gcn.py: unit_gcn (conv_pos='pre', norm BN, act ReLU) -----------------------------
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive="importance", with_res=False):
        super().__init__()
        self.num_subsets = A.size(0)
        assert adaptive in [None, "init"], "only the variants used by this checkpoint are kept"
        self.adaptive = adaptive
        self.with_res = with_res
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        if adaptive == "init":
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer("A", A)
        self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        if with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            else:
                self.down = lambda x: x

    def forward(self, x):
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0
        x = self.conv(x)
        x = x.view(n, self.num_subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, self.A)).contiguous()
        return self.act(self.bn(x) + res)


# ---- pyskl/models/gcns/utils/tcn.py: unit_tcn, mstcn ---------------------------------------------------------
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm="BN", dropout=0):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class mstcn(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.,
                 ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"), stride=1):
        super().__init__()
        num_branches = len(ms_cfg)
        self.act = nn.ReLU()
        assert mid_channels is None
        mid_channels = out_channels // num_branches
        rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            if cfg[0] == "max":
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                    nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None)))
        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(nn.BatchNorm2d(tin_channels), self.act,
                                       nn.Conv2d(tin_channels, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        feat = torch.cat([b(x) for b in self.branches], dim=1)
        return self.drop(self.bn(self.transform(feat)))


# ---- pyskl/models/gcns/stgcn.py: STGCNBlock, STGCN -----------------------------------------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == "gcn_"}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == "tcn_"}
        tcn_type = tcn_kwargs.pop("type", "unit_tcn")
        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        if tcn_type == "unit_tcn":
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        else:
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        x = self.tcn(self.gcn(x)) + res
        return self.relu(x)


class STGCN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, ch_ratio=2, num_stages=10, inflate_stages=(5, 8),
                 down_stages=(5, 8), **kwargs):
        super().__init__()
        A = torch.tensor(coco_spatial_graph(), dtype=torch.float32, requires_grad=False)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))       # data_bn_type 'VC'
        base = base_channels
        modules = []
        if in_channels != base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **kwargs)]
        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            cin = base_channels
            if i in inflate_stages:
                inflate_times += 1
            cout = int(base * ch_ratio ** inflate_times + EPS)
            base_channels = cout
            modules.append(STGCNBlock(cin, cout, A.clone(), stride, **kwargs))
        self.gcn = nn.ModuleList(modules)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for g in self.gcn:
            x = g(x)
        return x.reshape((N, M) + x.shape[1:])


# ---- pyskl/models/heads/simple_head.py: GCNHead (mode 'GCN', dropout 0) -------------------------------------
class GCNHead(nn.Module):
    def __init__(self, num_classes=120, in_channels=256):
        super().__init__()
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        N, M, C, T, V = x.shape
        x = x.reshape(N, M, C, T * V).mean(-1)       # AdaptiveAvgPool2d(1) over (T, V)
        return self.fc_cls(x.mean(dim=1))            # mean over persons


class STGCNPP(nn.Module):
    """RecognizerGCN(backbone=STGCN(gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn', coco/spatial),
    cls_head=GCNHead(120, 256)): keypoint (N, M, T, V, C) -> logits (N, 120)."""

    def __init__(self):
        super().__init__()
        self.backbone = STGCN(gcn_adaptive="init", gcn_with_res=True, tcn_type="mstcn")
        self.cls_head = GCNHead(120, 256)

    def forward(self, keypoint):
        return self.cls_head(self.backbone(keypoint))


def main():
    root = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        ckpt = Path(sys.argv[1])
    else:
        ckpt = Path(tempfile.mkdtemp()) / "j.pth"
        urllib.request.urlretrieve(CKPT_URL, ckpt)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    model = STGCNPP().eval()
    model.load_state_dict(sd, strict=True)

    dst = root / "models" / "stgcnpp_ntu120_hrnet_j.onnx"
    dummy = torch.randn(2, 2, 100, 17, 3)
    torch.onnx.export(model, (dummy,), str(dst), input_names=["keypoint"], output_names=["logits"],
                      dynamic_axes={"keypoint": {0: "clips"}, "logits": {0: "clips"}}, opset_version=17,
                      dynamo=False)

    # torch vs onnxruntime on random input and on a realistic-looking one (one person, second slot zero)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(4, 2, 100, 17, 3, generator=g)
    x2 = torch.zeros(3, 2, 100, 17, 3)
    x2[:, 0, :, :, :2] = torch.rand(3, 100, 17, 2, generator=g) * 0.6 - 0.3
    x2[:, 0, :, :, 2] = 0.9
    for x in (x1, x2):
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {"keypoint": x.numpy()})[0]
        print(f"max |torch - onnx| logits = {np.abs(ref - out).max():.2e}, "
              f"argmax equal = {bool((ref.argmax(1) == out.argmax(1)).all())}")
    print(f"exported {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
