"""TensorRT (NVIDIA, RTX 4070) engines for our ONNX models: build once in fp16, cache, run with torch CUDA buffers.

  eng = TRTModel(onnx_path, fp16=True, max_batch=64)       # builds or loads ~/.cache/squirrel-guard/trt/<name>-<hash>
  outs = eng(np_input)                                      # list of numpy outputs, same as onnxruntime's run()

Needs ~/src/splat/.venv (torch CUDA + tensorrt-cu12 10.7). On Debian 13 the pip wheel's
libnvinfer_builder_resource.so needs its executable-stack flag cleared (glibc 2.41 refuses it); see docs/trt.md.
Engines are tied to this GPU + TensorRT version and are local files (never committed).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

CACHE = Path.home() / ".cache/squirrel-guard/trt"
LOG = trt.Logger(trt.Logger.WARNING)
_DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64}


def _hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


class TRTModel:
    def __init__(self, onnx_path: str | Path, fp16: bool = True, max_batch: int = 32, workspace_gb: float = 2.0):
        onnx_path = Path(onnx_path)
        CACHE.mkdir(parents=True, exist_ok=True)
        tag = f"{onnx_path.stem}-{_hash(onnx_path)}-{'fp16' if fp16 else 'fp32'}-b{max_batch}-trt{trt.__version__}"
        eng_path = CACHE / f"{tag}.engine"
        if not eng_path.exists():
            eng_path.write_bytes(self._build(onnx_path, fp16, max_batch, workspace_gb))
        self.engine = trt.Runtime(LOG).deserialize_cuda_engine(eng_path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.stream = torch.cuda.Stream()
        self.path = eng_path

    @staticmethod
    def _build(onnx_path: Path, fp16: bool, max_batch: int, workspace_gb: float) -> bytes:
        b = trt.Builder(LOG)
        net = b.create_network(0)
        parser = trt.OnnxParser(net, LOG)
        if not parser.parse(onnx_path.read_bytes()):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        cfg = b.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
        if fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)
        prof = b.create_optimization_profile()
        dyn = False
        for i in range(net.num_inputs):
            t = net.get_input(i)
            shp = list(t.shape)
            if shp[0] == -1:
                dyn = True
                rest = [d if d > 0 else 1 for d in shp[1:]]
                prof.set_shape(t.name, [1] + rest, [max(1, max_batch // 2)] + rest, [max_batch] + rest)
        if dyn:
            cfg.add_optimization_profile(prof)
        ser = b.build_serialized_network(net, cfg)
        if ser is None:
            raise RuntimeError(f"TensorRT build failed for {onnx_path}")
        return bytes(ser)

    def __call__(self, *arrays: np.ndarray) -> list[np.ndarray]:
        with torch.cuda.stream(self.stream):
            keep = []
            for n, a in zip(self.inputs, arrays):
                x = torch.from_numpy(np.ascontiguousarray(a)).to("cuda", _DT[self.engine.get_tensor_dtype(n)],
                                                                    non_blocking=True)
                self.ctx.set_input_shape(n, tuple(x.shape))
                self.ctx.set_tensor_address(n, x.data_ptr())
                keep.append(x)
            outs = []
            for n in self.outputs:
                shp = tuple(self.ctx.get_tensor_shape(n))
                y = torch.empty(shp, dtype=_DT[self.engine.get_tensor_dtype(n)], device="cuda")
                self.ctx.set_tensor_address(n, y.data_ptr())
                outs.append(y)
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return [o.float().cpu().numpy() for o in outs]
