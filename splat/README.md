# Yard Gaussian splat (laptop, RTX 4070 8 GB)

Phone walk-around video of the yard → camera poses (pycolmap) → Gaussian splat (gsplat) → `splat.ply`. The 3D replay
viewer loads it as its scene (`/replay/?splat=...`).

## Environment (one time)
gsplat only ships prebuilt CUDA wheels for Python 3.10 + torch 2.4, so the env is separate from the project `.venv`:

```
mkdir -p ~/src/splat && cd ~/src/splat
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python "gsplat==1.5.3+pt24cu124" --extra-index-url https://docs.gsplat.studio/whl --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python pycolmap==4.2.1 "numpy<2" opencv-python-headless==4.10.0.84 "imageio[ffmpeg]==2.36.1" plyfile==1.1 tqdm packaging "setuptools<70" ninja
```
`packaging` and `setuptools` are undeclared runtime dependencies of the gsplat wheel (its CUDA backend loader
imports them): without them training fails with ModuleNotFoundError.

## Capture (on mounting day)
Walk slowly around the whole yard, 2-3 min, phone in landscape, overlapping views, no fast turns; include the
spot where the Wyze camera hangs. Avoid people/cars moving through the shot.

## Run
```
~/src/splat/.venv/bin/python make_splat.py all yard.mp4 ~/splat-work/yard --fps 2
```
Add `--method vggt --refine` for the fast pose path (below). Close games / local LLMs that hold VRAM first. COLMAP SfM runs on CPU (pycolmap wheel has no CUDA SIFT): ~10 min for 40
frames, expect 30-60 min for ~300.

## Test result (2026-11-05, 8 s stock clip panning past a statue, 39 frames)
SfM: 39/39 registered, 16,621 points, 0.66 px reprojection, 572 s. Training (7,000 iters, factor 2, cap 300k):
1.5 min, 190k Gaussians, **peak VRAM 0.26 GiB**, held-out PSNR 18.5: the building behind is reconstructed well,
the statue in front is smeared (seen from a narrow arc only). A walk-around that circles the yard avoids that.
Defaults are now full resolution, 30,000 iterations, 1M Gaussians.
(Correction 2026-11-06, see below: the smearing is mostly COLMAP's bad self-calibration on this clip, not the arc.)

## Fast path: VGGT instead of COLMAP
[VGGT](https://github.com/facebookresearch/vggt) (Meta, CVPR 2025) predicts camera poses, intrinsics and per-pixel
depth for all frames in one feed-forward pass. `sfm --method vggt` runs it and writes the same `sparse/` +
`undistorted/{sparse,images}` layout as the COLMAP path, so `train` is unchanged:
```
~/src/splat/.venv/bin/python make_splat.py sfm WORK --method vggt --refine      # ~1 min for 39 frames
~/src/splat/.venv/bin/python make_splat.py all yard.mp4 WORK --method vggt --refine --fps 2
~/src/splat/.venv/bin/python make_splat.py compare WORK_A WORK_B --out DIR       # held-out views side by side
```
- One shared SIMPLE_PINHOLE camera (median VGGT focal, principal point at the centre; VGGT models no lens
  distortion, fine for phone main cameras). Initial points: 150k pixels sampled from the 50 % most confident depth
  pixels, unprojected with VGGT's cameras (`--vggt-points`, `--vggt-conf`).
- `--refine` (recommended): SIFT + exhaustive/sequential matching (~22 s CPU), triangulation on VGGT's poses, one
  pycolmap bundle adjustment of poses, focal and points (~13 s), then a similarity transform back into VGGT's
  frame so the depth points are re-unprojected with the refined cameras. Adds ~35 s, +4 dB held-out PSNR here.
  The slow part of COLMAP (incremental mapping, 9 of its 9.5 min) is skipped.
- VRAM: the user's llama-server usually leaves ~3 GB free, which plain VGGT does not fit in (1.26B parameters).
  `vggt_infer.py` therefore drops the point-map and tracking heads, runs the aggregator in bf16 with its weights in
  pinned CPU RAM streamed in block by block, moves the cached layer outputs to CPU, and runs the heads in fp32 two
  frames at a time. Result matches upstream fp32 within 0.3 % depth (median) and 1e-3 in pose. Peak 0.83 GiB for
  8 frames, 1.31 GiB for 39; global attention grows with the frame count, so more than `--vggt-chunk` (80) frames
  are processed in chunks that share chunk/4 evenly spaced anchor frames and are aligned to the first chunk by a
  similarity fit on the anchors' depth points. Test (39 frames forced into 3 chunks of 20): peak 0.85 GiB, camera
  centres within 3 % (median) of the trajectory extent of the single pass.
- Runs as a subprocess (`splat/vggt_infer.py`, `--vggt-python` selects the interpreter), so its memory is gone
  before training. `--vggt-device cpu` works for testing (fp32, ~60-75 s for 8 frames).

Setup (one time; everything else is already in `.venv`, no separate venv was needed):
```
mkdir -p ~/src/vggt && curl -sSL https://github.com/facebookresearch/vggt/archive/refs/heads/main.tar.gz | tar xz -C ~/src/vggt --strip-components=1
uv pip install --python ~/src/splat/.venv/bin/python einops==0.8.1        # imported by vggt's (unused) track head
```
Tested with vggt commit a288dd0f (2026-05-19). Weights download on first use into the Hugging Face cache (4.7 GB).

### Licences
- VGGT code: "VGGT License" (Meta, v1 2025-07-29): use, modification and redistribution allowed incl. commercial,
  subject to Meta's Acceptable Use Policy (no military/warfare, ITAR etc.).
- Weights `facebook/VGGT-1B` (the default here): **CC BY-NC 4.0, non-commercial only**. Fine for this personal
  home project; do not use it for anything commercial (e.g. the PC-repair business).
- Weights `facebook/VGGT-1B-Commercial`: same VGGT License as the code (commercial use allowed), reportedly the same
  accuracy, but gated on Hugging Face (application form, approval). After approval and `huggingface-cli login`,
  `--vggt-weights facebook/VGGT-1B-Commercial` should work (that repo also has a `model.safetensors`; untested).

### Benchmark (2026-11-06, statue clip, 39 frames at 1600x900, RTX 4070 Laptop with llama-server resident)
Training identical for all: 7,000 iterations, factor 2, cap 300k, SH 2; held-out PSNR on frames 0, 8, 16, 24, 32,
three seeds each (`--seed`, new; before it, unseeded reruns of the COLMAP case varied 17.7-18.9).

| | COLMAP | VGGT | VGGT + `--refine` |
|---|---|---|---|
| pose step wall time | 572 s (CPU) | **24 s** (13 s weight load, 8.4 s inference) | 59 s |
| pose step peak VRAM | none (CPU) | 1.31 GiB | 1.31 GiB |
| initial points | 16,621 | 150,000 | 178,515 (28,515 SIFT + 150k depth) |
| reprojection error | 0.66 px | (1.58 px on SIFT tracks) | 0.64 px |
| focal (1600 px wide) | fx 1879 / fy 1290 (!) | 3354 | 3267 |
| training time / peak VRAM | 1.2-1.3 min / 0.26 GiB | 2.2-2.6 min / 0.65 GiB | 2.9-3.6 min / 0.69 GiB |
| Gaussians after training | 190,519 | 300,000 (cap) | 300,000 (cap) |
| held-out PSNR (3 seeds) | 18.73 / 18.62 / 18.87 | 24.52 / 24.49 / 24.48 | **28.61 / 28.59 / 28.68** |
| video frames to splat | ~11 min | ~3 min | ~4 min |

Comparison renders: `~/.cache/squirrel-guard-test/splat/statue-vggt-compare/compare_view{000,016,032}.png`
(one row per method: real held-out frame | render). Caveats: COLMAP's frames are its undistorted versions
(cropped to 1396x805, trained at 698x402) while VGGT trains on the original frames (800x450), so the PSNRs are on
slightly different images; training takes longer with VGGT because it starts from 9x more points and reaches the
Gaussian cap. The COLMAP result on this clip is itself poor: a narrow-arc pan gave it a non-physical
self-calibration (fx/fy = 1.46, strong tangential distortion), so its held-out renders are misregistered and the
statue smears, while VGGT's renders line up. On a real walk-around COLMAP will calibrate better and the gap should
shrink; the speed difference (incremental mapping is the slow part) remains.
