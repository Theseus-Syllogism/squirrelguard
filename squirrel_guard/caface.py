"""CAFace (Cluster and Aggregate, Kim et al., NeurIPS 2022; code MIT, github.com/mk-minchul/caface) as a learned
alternative to the quality-weighted mean in faceseq_agg.py.

CAFace has its OWN backbone (AdaFace IR-101 trained on WebFace4M, bundled in its checkpoint): its fused vectors live
in that model's embedding space, NOT in our live AdaFace IR-101 WebFace12M space. Galleries must be embedded with
the same CAFace backbone (embed_singles) before comparing. Weights: WebFace4M-derived = non-commercial research:
local only, never committed or published. Checkpoints are loaded with torch.load(weights_only=True).

Needs a CUDA PyTorch env with omegaconf/pyrootutils/einops (~/src/splat/.venv) and the repo at CAFACE_DIR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

CAFACE_DIR = Path(os.environ.get("CAFACE_DIR", Path.home() / "src/caface"))
CKPT = CAFACE_DIR / "pretrained_models/CAFace_AdaFaceWebFace4M.ckpt"


class CAFace:
    def __init__(self, device: str = "cuda:0"):
        import torch
        demo = CAFACE_DIR / "demo"
        sys.path[:0] = [str(demo), str(CAFACE_DIR)]
        cwd = os.getcwd()
        os.chdir(demo)                                    # pyrootutils finds the demo's pyproject.toml
        try:
            from model_loader import load_caface
            from inference import fuse_feature
            self.agg, self.backbone, self.hp = load_caface(str(CKPT), device=device)
        finally:
            os.chdir(cwd)
        self._fuse, self.torch, self.device = fuse_feature, torch, device

    def _features(self, crops_bgr: np.ndarray):
        t = self.torch
        x = t.from_numpy(crops_bgr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
        x = (x - 0.5) / 0.5
        feats, norms, inters = [], [], []
        with t.no_grad():
            for i in range(0, len(x), 64):
                f, n, inter = self.backbone(x[i:i + 64].to(self.device), return_style=self.hp.style_index)
                ci = self.agg.compress_intermediate(inter)
                feats.append((f * n).cpu().numpy())
                norms.append(n.cpu().numpy())
                inters.append(ci.cpu().numpy())
        return np.concatenate(feats), np.concatenate(norms).reshape(-1), np.concatenate(inters)

    def embed_singles(self, crops_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-crop L2-normalised CAFace-backbone embeddings + AdaFace norms (for galleries / per-frame checks)."""
        f, n, _ = self._features(crops_bgr)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9), n

    def fuse(self, crops_bgr: np.ndarray) -> np.ndarray:
        """One L2-normalised fused template from a sequence of aligned crops."""
        f, _, inter = self._features(crops_bgr)
        fused, _w = self._fuse(f, aggregator=self.agg, intermediates=inter, method="cluster_and_aggregate",
                               device=self.device)
        v = np.asarray(fused, np.float32).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)
