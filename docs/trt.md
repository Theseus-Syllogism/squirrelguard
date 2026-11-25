# TensorRT on the laptop RTX 4070

`squirrel_guard/trt.py` builds fp16 TensorRT engines from our ONNX models (cached in
`~/.cache/squirrel-guard/trt/`, tied to this GPU + TensorRT version) and runs them with torch CUDA buffers.
`AdaFace(..., backend="trt")` and `PoseEstimator(..., backend="trt")` use it; the default stays onnxruntime CPU
(gpuserver's profiler container has no NVIDIA GPU). Env: `~/src/splat/.venv` + `tensorrt-cu12{,-bindings,-libs}==10.7.0`.

Debian 13 trap: glibc 2.41 refuses `libnvinfer_builder_resource.so.10.7.0` ("cannot enable executable stack").
Fixed by clearing PF_X on its PT_GNU_STACK program header (backup `*.orig-execstack` next to it); redo after
reinstalling the wheel.

`scripts/trt_bench.py` (2026-11-10, 128 real face crops, 40 v4 frames):

| model | onnxruntime CPU (4 threads) | TensorRT fp16 | agreement |
|---|---|---|---|
| AdaFace IR-101 | 73.5 ms / face | 0.75 ms / face | cosine >= 0.99987, norm within 0.24 % |
| YOLOv8s-pose | 108.7 ms / frame | 9.7 ms / frame | box IoU >= 0.987, keypoints 0.32 px, same counts |

BasicVSR++ (`vsr.py`) uses fp16 autocast instead (109 -> 75 ms/frame, same PSNR).
`scripts/enhance_event.py EVENT` = face still (lucky stack for faces <= 40 px, else best camera frame) + BasicVSR++
close-up, on the laptop, ~50 s per 13 s event.
