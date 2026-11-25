# 4D Gaussian-splat avatar (squirrel_guard/avatar4d.py)

Dynamic Gaussian-splat avatar of the person in the two-camera demo walk (v4 2560x1440 + frontyard 1920x1080,
frontyard shift -15 decoded frames = -1.80 s), bound to the rig-driven SMPL body. Method, limits and licence are
in the module docstring. **All outputs are SMPL-derived (non-commercial): local only, under
`~/.cache/squirrel-guard-test/avatar/`, never in git** (`.gitignore` covers `deploy/recorder/replay/demo/*_avatar*`).

## Pipeline

```
.venv/bin/python scripts/avatar4d_train.py prep --clip v4=… --clip frontyard=… --calib v4=… --calib frontyard=… \
    --replay deploy/recorder/replay/demo/multicam.json --out AV/prep.npz            # 17 s CPU
~/src/splat/.venv/bin/python scripts/avatar4d_train.py train --prep AV/prep.npz --out AV/run_frames --iters 15000
~/src/splat/.venv/bin/python scripts/avatar4d_train.py train … --mode smplcolor                    # baseline
~/src/splat/.venv/bin/python scripts/avatar4d_train.py train … --out AV/run_cam --holdout-cam 1   # v4 only
~/src/splat/.venv/bin/python scripts/avatar4d_train.py export --prep AV/prep.npz --model AV/run_frames/model_full.pt --out AV/multicam_avatar
~/src/splat/.venv/bin/python scripts/avatar4d_train.py renders --prep AV/prep.npz --run AV/run_frames --out AV/renders
```

* SMPL refit on the rig markers (`smplfit.fit_to_rig`): 110 of 256 replay frames have markers; marker error
  1.1 cm median, 2.6 cm p90.
* Views: each frame undistorted (K, D -> pinhole K) and cropped to 384x512 around the projected body: 220 views
  (110 per camera). Pixels outside the source image are masked out of the loss.
* Masks: YOLOv8s-seg 320. SAM ViT-B (`masks` subcommand) was tried with the body box prompt and was worse here (it
  grabbed background and the black out-of-image border); IoU with YOLO 0.77. Training uses the YOLO masks.

## Results (held out, never trained on)

Metrics on the 384x512 crop with the ground truth composited on black. "Person PSNR" counts only the dilated
union of GT mask and rendered alpha (the honest number; the whole-crop PSNR is inflated by black background).
No LPIPS (not installed in the splat venv).

| Split | Model | PSNR crop | PSNR person | SSIM | mask IoU |
|---|---|---|---|---|---|
| every 8th frame, both cams (28 views) | avatar (40k G, 15k it) | **20.47** | **14.41** | **0.832** | **0.658** |
| | coloured SMPL baseline | 19.52 | 13.86 | 0.815 | 0.608 |
| frontyard held out entirely (110 views), train v4 only | avatar | 18.81 | 12.86 | 0.806 | 0.465 |
| | coloured SMPL baseline | 18.80 | 12.87 | 0.809 | 0.457 |

Per camera (frame hold-out, avatar): v4 14.31 dB person / IoU 0.75; frontyard 14.51 dB / IoU 0.57.

Reading: on held-out frames the avatar beats the coloured-SMPL baseline by ~0.55 dB person-PSNR and +0.05 mask IoU
(the coat silhouette). On the held-out camera it is **no better than the baseline**: with one training viewpoint
the extra geometry/offsets do not generalise to a view from the other side, and the unseen side stays dark/blotchy;
the camera's colour gain is also unknown (identity). Absolute numbers are low: limited by YOLO-320 mask quality
(holes, wrong edges), SMPL-to-image misalignment of a few cm, and the low view count (2 cameras).

Training: 15k iterations in 4.0 min, peak VRAM 0.64 GiB (RTX 4070); baseline 0.6 min, 0.54 GiB.
An earlier 8k-iteration variant (person-weighted loss, weaker offset prior) gave 14.19 dB: more iterations or the
loss tweak do not move the number, so the limit is the data/masks, not optimisation.

## Renders (local only)

* `~/.cache/squirrel-guard-test/avatar/renders/contact_sheet.jpg`: GT | avatar | baseline for 8 held-out views.
* `~/.cache/squirrel-guard-test/avatar/renders/turntable.mp4` (+ `turntable/*.png`): 48-frame turntable at the
  middle frame, then the walk from an orbiting camera.

## Browser export

`multicam_avatar.bin/.json` (11.8 MB: 30,870 Gaussians after dropping opacity < 0.05, 110 frames): canonical xyz,
canonical covariance (6 floats), RGBA (SH dc only), top-4 joint indices/weights (u8), per-frame 24 world bone
3x4 transforms (per-frame translation fix folded in), per-frame non-rigid offsets int8 x `offset_scale`.
`deploy/recorder/replay/avatar4d.html` (three.js 0.160) skins on the CPU per frame, depth-sorts (counting sort), and
splats with an EWA vertex shader; orbit, play/scrub, follow. Loads `?src=demo/multicam_avatar` (symlink the two
files into the git-ignored demo dir) or via the file picker. Checked in headless Chromium (SwiftShader).

## Known artefacts

* Coat hem and open coat flaps: blurred, partly fused with the legs; the front of the coat is blotchy/dark.
* Body regions seen by neither camera (back of head/one flank at some frames) stay at the initial mean colour.
* Held-out frames use interpolated latents: smoother than training frames.
* SH degree 1 is evaluated in the world frame; the browser uses only the dc colour.
* Frames without rig markers (146 of 256) have no avatar.

## Integrating into the replay viewer

viewer.js would need: a loader for `multicam_avatar.*` next to `body3d` (keyed by `rig_frame` -> replay index),
the same CPU skin + sort + instanced splat material (move it out of avatar4d.html into a module), a toggle
between SMPL mesh / avatar, hiding outside the 110 fitted frames, and the replay's time base (rig_frame is the
replay frame index). The export must stay off the recorder host deploy unless the SMPL licence question is settled.

## Filling the unseen side (2026-11-08): measured, no gain

Held-out camera split (train v4 only, evaluate frontyard, 110 views), person PSNR / mask IoU:

| model | person PSNR | IoU |
|---|---|---|
| avatar (before) | 12.855 | 0.465 |
| + mirror fill (post-hoc) | 12.854 | 0.465 |
| + neighbour (harmonic) colour diffusion | 12.848 | 0.465 |
| + mirror then diffusion | 12.850 | 0.465 |
| avatar trained with the visibility-weighted mirror/neighbour/offset prior (`--fill-reg 1`) | 12.838 | 0.466 |
| coloured-SMPL baseline (before) | 12.866 | 0.457 |
| baseline + mirror (+ diffusion) | 12.870 | 0.457 |

Why nothing moves: the premise was wrong. Visibility (depth-tested, training views only) says only 945 of 40,000
surface Gaussians (2.4 %) are seen in fewer than 2 v4 views (the person turns over 110 frames); in the trained avatar
9,648, mostly ones that drifted or went transparent. Rendered into the held-out views, unseen Gaussians draw ~0 % of
the overlapping person pixels. 74-81 % of the held-out squared error is **silhouette mismatch** (GT mask vs rendered
alpha: YOLO-320 masks, SMPL/coat misalignment), the rest is colour on SEEN regions; even an oracle per-channel colour
map fitted on the held-out views adds only 0.3-0.4 dB on the overlap. Next levers: better masks (a stronger
segmenter or matting) and pose/garment alignment, not completing unseen regions.
Code: `avatar4d.visibility`, `fill_colours`, `fill_eval`, `train(fill_reg=)`; CLI `avatar4d_train.py fill` and
`train --fill-reg W --tag NAME`.

Generative priors (licence survey, 2026-11-08): LHM++ (code Apache-2.0 + NVIDIA non-commercial EG3D parts, weights
CC BY-NC 4.0, SMPL-X-based assets, stated 7.3-8 GB) is the only multi-view human model that is licence-compatible
with local non-commercial use and near the 8 GB budget, but it needs the whole GPU (llama services hold ~1.2-1.4 GB)
and some checkpoints were "pending". Zero123++ (CC BY-NC weights, ~5-5.7 GB) fits but is a generic object model.
HumanSplat (no weights), GHG / IDOL / Human4DiT / AniGS (no licence file), MagicMan / PSHuman / SIFU / LHM v1 (>= 14 GB)
are out. Not tried: given the measurement above, hallucinated back-side colour cannot improve this metric, and it
would be invented appearance, not the person.
