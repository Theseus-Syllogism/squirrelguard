# 3D body for the trajectory replay (WHAM, offline)

`scripts/body3d_wham.py` fits a full SMPL body to the person in a replay clip with
[WHAM](https://github.com/yohanshin/WHAM) (world-grounded human motion from video) and writes it into the replay
JSON as an optional `"body3d"` block. The replay viewer (`deploy/recorder/replay/viewer.js`) then draws a body mesh
(and/or its joints) instead of, or together with, the 2D-keypoint stick figure. It is an offline, GPU-heavy
extra for demos and investigations; nothing in the runtime (`squirrel_guard/`, the profiler, the recorder host services)
depends on it.

## Licences: read before copying anything anywhere

* **WHAM code**: MIT (Soyong Shin, 2023). The checkpoints are downloaded by its `fetch_demo_data.sh` URLs (Google
  Drive); they come from the WHAM release and are research artefacts. HMR2.0 (`hmr2a.ckpt`, 4D-Humans) is MIT,
  ViTPose (`vitpose-h-multi-coco.pth`) is Apache-2.0.
* **SMPL** (Max Planck / Meshcapade): the user's own registration, **non-commercial use only**. The model files
  (`models/smpl/SMPL_*.npz`, the converted `~/src/wham/dataset/body_models/smpl/*.pkl`) and everything that
  contains SMPL data (the triangle list `smpl_faces_u16.bin`, the per-frame vertex buffers `*_body.bin`, and the
  `*_body3d.json` files that hold fitted SMPL parameters) must **never be committed, published or deployed
  outside the user's own machines**. `.gitignore` covers `models/` and the three output patterns under
  `deploy/recorder/replay/demo/`. The committed demo JSONs (`basement.json`, `demo.json`) do not contain a body3d
  block; the viewer works without it.
* **Neutral model only.** `--setup-smpl` converts only `SMPL_NEUTRAL.npz`; WHAM is built with `gender="neutral"`
  (asserted), and the fit, the exported joints, mesh and SMPL parameters all come from that model (`body3d.model` =
  "SMPL neutral (v1.1, first 10 betas)"). No MALE or FEMALE file is used anywhere.

## Setup (once, outside the project)

```
~/src/wham                     WHAM source (GitHub tarball of main), MIT
~/src/wham/.venv               Python 3.10 (uv): torch 2.5.1+cu124, torchvision 0.20.1, numpy<2, smplx, yacs,
                               joblib, scipy, scikit-image, opencv-python-headless, loguru, progress, einops,
                               timm 0.9.16, onnxruntime, gdown
~/src/wham/checkpoints         wham_vit_bedlam_w_3dpw.pth.tar (used), wham_vit_w_3dpw.pth.tar, hmr2a.ckpt,
                               vitpose-h-multi-coco.pth   (~6 GB; no yolov8x.pt, no dpvo.pth)
~/src/wham/dataset/body_models WHAM's auxiliary files (body_models.tar.gz) + smpl/SMPL_NEUTRAL.pkl
```

Not installed: mmcv / mmpose / ViTPose repo (the script has a ~40-line plain-torch ViTPose-H: WHAM's own ViT
backbone + the 2-deconv heatmap head, UDP crop, DARK decoding), DPVO / pytorch3d / torch-scatter (the camera is
static, so there is no SLAM step), chumpy (the SMPL `.pkl` files are written from the `.npz` as plain arrays).

```
~/src/wham/.venv/bin/python scripts/body3d_wham.py --setup-smpl models/smpl      # SMPL .npz -> WHAM .pkl
```

## Usage

```
~/src/wham/.venv/bin/python scripts/body3d_wham.py CLIP.mp4 --replay deploy/recorder/replay/demo/basement.json \
    [--out deploy/recorder/replay/demo/basement_body3d.json] [--kp2d vitpose|yolo] [--flip] [--device cuda|cpu] \
    [--align gravity|camera] [--ground snap|constant] [--root camera|image|world] [--tracker kinematics|simple] \
    [--mesh-step 1] [--cache DIR] [--hmr2-fp16]
```

Open `index.html?data=demo/basement_body3d.json` (the file sits next to the replay, so its assets resolve).
`?body=mesh|both|joints|stick` picks the drawing; the choice is also in the top bar ("Body") and remembered. If the
mesh files are missing (e.g. a copy of the JSON without the `.bin` files) the viewer draws the 3D joints only and
says so in the status line.

The pose model for the person track is `models/yolov8s-pose.onnx` relative to the checkout (a worktree has no
`models/`: pass `--pose-model /home/operator/src/squirrel-guard/models/yolov8s-pose.onnx`).

## What it does

1. **Track**: `kinematics.track` (the replay's own single-person tracker, YOLOv8s-pose ONNX on CPU) on the same
   ~8 fps subsampled frames as the replay (its weak-torso/re-acquire rules count frames, so it is not run at
   25/30 fps); boxes in between are interpolated. If it keeps < 30 % of the frames, a plain IoU tracker
   (`--tracker simple`) is used instead: the current `kinematics.track` keeps only 3 of 211 frames of the fog
   stock walk (its torso-quality gate came after demo.json was made).
2. **2D keypoints**: ViTPose-H at the full frame rate (25/30 fps; WHAM is trained at 30 fps) on the track boxes,
   fp16 on the GPU (matches fp32 to 0.01 px median). `--kp2d yolo` reuses the tracker's keypoints (only on the
   subsampled frames; the rest are masked).
3. **Features**: HMR2.0 ViT-H tokens + the first frame's SMPL estimate (WHAM's FeatureExtractor), **fp32**. fp16
   features differ from fp32 by only ~0.2 % (cosine 0.99999) but systematically, and WHAM's shape follows them:
   basement stature 1.47 m from fp16 features vs 1.76 m from fp32, fog walk 1.48 vs 1.73 m (0.2 % *random* noise
   on the fp32 features changes the stature by < 1 mm). fp32 ViT-H needs ~3.1 GB free VRAM; next to the
   llama-server (~4.8 GB used) only ~2.4 GB is free, so it runs on the CPU (~0.7 s per frame) automatically.
   `--hmr2-fp16` forces fp16 on the GPU (fast, wrong shape).
4. **WHAM** on every run of >= 30 frames (gaps up to 8 frames are bridged with masked keypoints), with the camera
   angular velocity set to zero (static camera: WHAM's "local only" path without DPVO) and the replay's focal
   length as the camera intrinsics (keypoints in image coordinates; the lens is applied in the fit, step 5).
5. **Alignment** into the replay world and sampling at the replay's analysed frames.

### Alignment: WHAM's world vs our camera

WHAM gives the body twice: in **camera coordinates** (root orientation from the image, translation from a full
perspective camera with our focal length), and in a **world frame** built by integrating the predicted root
velocity. For a static camera without SLAM that world frame is not truly gravity-aligned: it is level with the
*camera* (its "up" is the camera's -y axis, measured below as `wham_pitch_deg_median` ~ 0). The body is upright
in it, because WHAM's motion prior is learned from upright people.

* `--align gravity` (default): WHAM's world up is mapped to our +Z (the body stands upright on our floor). Yaw and
  XY translation are then fitted (2D Procrustes, no scale) so WHAM's pelvis path matches WHAM's own camera
  placement (`R @ x_cam + C`): every pelvis stays on the pixel ray the camera saw it on, at WHAM's depth.
* `--align camera`: WHAM's world goes through WHAM's camera rotation and then our `R`. This trusts the camera-frame
  root orientation, which monocular HMR cannot resolve for the pitch: with the basement camera looking 38.5 deg up
  the body ends up leaning ~40 deg backwards. Kept for level cameras and for comparison.
* `--root fit` (default since the follow-up) is described in its own section below. `--root camera`: after that, the pelvis XY follows the camera placement smoothed over 0.3 s, not WHAM's
  integrated path (which under-walks badly on the basement clip: ~0.5 m of travel for ~2 m in the image).
  `--root world` keeps WHAM's path; `--root image` fits the horizontal position so the grounded body's joints
  re-project onto the 2D keypoints through our camera (exact overlay in Camera view, but on the basement clip it
  puts the person at 1.2 m, see below).
* `--ground snap` (default): the lowest body vertex during WHAM's predicted foot contact is put on Z = 0 frame by
  frame (offsets interpolated between contacts and smoothed over 0.2 s). This removes the vertical drift of WHAM's
  integrated path, which appears exactly when the real camera is not level (`vertical_drift_m` in the stats).
  `--ground constant` uses one offset.

### Scale and stature

WHAM has no metric cue but the human body: the SMPL betas fix the body size, and the camera depth follows from
body size x focal length / size in pixels. The stature of the fitted body (zero pose, mean betas over the clip)
is reported as `stature_m` and compared with the 1.75 m that the replay assumes. The replay's distances from the
camera come from the camera calibration and that 1.75 m (and for the basement clip from an assumed hfov of 95 deg,
camera height 0.8 m and pitch 38.5 deg up), WHAM's from its fitted body and our focal length; their ratio is in
`vs_trajectory.median_ground_distance_from_camera_m`.

## Output

`body3d` in the replay JSON (arrays aligned with `t`; `null` where there is no body):

| key | content |
|---|---|
| `joints` | COCO-17 joints (WHAM's regressor), world metres, same order as `skeleton` |
| `root` | pelvis (hip midpoint), world metres |
| `contact` | `[left, right]` foot contact probability |
| `smpl` | `betas` (10, mean over the clip), per frame `global_orient` (axis-angle, replay world), `body_pose` (69), `transl`: `vertices = SMPL_neutral(global_orient, body_pose, betas) + transl` |
| `mesh` | `vertices` file: float16 LE, `(len(index), 6890, 3)` relative to `root[index[i]]`; `faces`: uint16 triangle list (13776 x 3); `index`: entries of `t` that have a mesh |
| `stats` | timing, VRAM, contact, stature, comparison with the trajectory (below) |

Size: 41 KB of mesh per replay frame (6890 vertices x 3 x float16).

## Image-aligned body (`--root fit`, default) and the camera re-solve

The first version placed WHAM's body with WHAM's own depth and gravity alignment and snapped the feet to the floor.
On the basement clip the mesh came out larger, lower and closer than the person in Camera view, with crouched legs.
Three causes, three fixes:

1. **Lens model.** The replay's assumed basement lens (hfov 95 deg, k1 -0.25, k2 0.025) folds over: d(r) = r (1 +
   k1 r^2 + k2 r^4) peaks at 68 % of the image-corner radius, so pixels near the edge (where the person walks for
   most of the clip) have no valid ray. `cv2.undistortPoints` returned points 500+ px outside the image, which made
   WHAM's input box ~5x too big (body "closer"). WHAM now gets the keypoints in image coordinates (`--undistort`
   restores the old behaviour); the lens is only used where it belongs, in the projection of the fit.
2. **Legs out of frame.** Where knees or ankles are not seen (ViTPose conf < 0.3), WHAM's guessed legs (often
   seated: knee flexion median 55 deg) are replaced by a straight-leg walking prior (hip +-22 deg and knee 5-40 deg,
   phased by the cadence, amplitude from the trajectory speed; standing straight when the person stops), the torso
   is made upright, and the feet are **not** forced onto the floor. `body3d.legs_prior` marks those frames.
3. **Placement by reprojection.** Every frame is fitted to the ViTPose keypoints through the replay camera (K, D
   with distortion, R, C): pelvis translation + a small global-orientation correction (prior 8 deg), Cauchy-robust,
   confidence-weighted, visible joints only, smooth path (acceleration prior 3 m/s^2) and smooth correction, coarse
   to fine (LBFGS). It runs twice: first in camera coordinates starting from WHAM's camera-frame body (orientation
   free within 25 deg), then mapped into the world with (R, C) and refined there. Depth along each ray comes from
   the body size (betas) and the focal length, not from the camera height or pitch. WHAM still provides pose,
   shape and foot contacts; its integrated world path is no longer used for placement.

**Camera re-solve** (`--solve-camera --solve-hfov 75,80,85,90,95 --solve-k1=-0.1,-0.15,-0.2,-0.25,-0.3`): for each
candidate lens the camera-frame fit is run; lenses that fold over inside the image (`lens_reach` < 1) are rejected;
the lens with the lowest median reprojection error wins; then pitch (roll 0) and camera height are solved so that
the fitted pelvis and nose positions sit at their standing heights above a level floor. Basement result: **hfov 85
deg, k1 -0.2, pitch -25.6 deg (looking up), height 0.869 m** (floor rms 0.06 m; camera-frame fit 5.9 px median).
The desk height (~0.8 m) agrees; the old pitch (-38.5 deg) was too steep. Every valid lens in the grid put the
camera at 0.85-0.93 m and -19 to -34 deg; the error surface over hfov is shallow (6-8 px), so hfov is only loosely
constrained. The basement replay was regenerated with these values (`replay_demo.py ... --cam-height 0.869
--pitch -25.6 --hfov 85 --k1 -0.2 --fps 8 --max-s 12.5`, head plane 1.75 m; the gaze comes from replay_demo, the
splat was rebuilt with `splat/replay_splat.py ... --far 5.0`). `basement.json` records the old and the solved
values under `calibration`.

| basement | before (as merged) | after |
|---|---|---|
| joint reprojection error, visible joints (px) | median 111, mean 178 | median 5.9, mean 12.8 |
| knee flexion where the legs are unseen (median) | 55 deg (crouched/seated) | 9 deg |
| frames with the leg prior | - | 237 of 278 |
| fitted stature (neutral, mean betas) | 1.76 m | 1.68 m |
| soles above the floor where the legs are seen (median) | snapped to 0 | 0.12 m (not forced) |
| body path vs trajectory RMSE (observed frames) | 1.31 m | 0.23 m (0.17 m after a rigid fit) |
| median distance from the camera, body / trajectory | 2.11 / 3.30 m (64 %) | 4.45 / 4.43 m (100 %) |

"Before" is the merged version's body (undistorted input, gravity + contact placement) on the old camera. The
remaining trajectory disagreement is at the start (t < 2.5 s), where the walker is 5-7 m away behind furniture
and the trajectory uses the box bottom as the foot (lower legs hidden), which puts them 9-12 m away; the body puts
them at 6-7 m. A head plane of 1.68 m (the fitted stature) instead of 1.75 m was tried and was worse (RMSE 1.84 m:
it pushes those foot-anchored frames further out), so the replay keeps 1.75 m.

Stock walk (fog.mp4, its calibrated demo camera, `--root fit`): on the same 633 frames as before, reprojection
median 7.6 -> 3.3 px (mean 23.8 -> 12.6), stature 1.73 m, path RMSE vs trajectory 0.65 m (0.61 m with WHAM's own
placement), distance 10.07 / 10.29 m (98 %, was 96 %): not worse. With main's fixed `kinematics.track` (no
fallback needed any more; the IoU fallback stays for clips where the tracker keeps < 30 %), the body covers 837
frames instead of 633: reprojection median 6.2 -> 2.7 px (mean 19.0 -> 9.9), stature 1.725 m, median distance
12.97 / 13.0 m. The overall RMSE is 1.82 m only because the extra frames are the walker standing ~17-18 m away
(t 21-28 s): by trajectory range it is 0.99 m (< 8 m), 0.22 m (8-12 m), 0.18 m (12-16 m), 2.9 m (> 16 m, where 15 %
of range is 2.6 m and the trajectory itself is weakest).

The stock sample `demo.json` was then rebuilt with today's code and its original settings (`replay_demo.py fog.mp4
--name demo --height 1.75 --hfov 55 --tilt 25 --fps 8 --max-s 40`, walk calibration: pitch -9.81 deg, roll 4.02,
height 0.335 m, residual 7.2 cm median; gaze recomputed by replay_demo): tracked 210 of 211 analysed frames; 209
trajectory frames, observed 201 -> 200, path length 18.06 -> 18.59 m, mean speed 0.65 -> 0.67 m/s, cadence 33.3 ->
56.2 steps/min (gait now measures it), stop 21.2-25.0 s -> 21.4-23.5 s, path moved by 0.17 m RMS; 99 cut-outs, gaze
"facing camera" 0 % (walking away) as before. No visible difference in the viewer apart from the shorter stop and
more footprints (24 vs 14). Its body3d (local, git-ignored): 837 WHAM frames, reprojection median 6.3 -> 2.7 px
(mean 18.9 -> 9.9), stature 1.725 m, contacts L/R 64 / 71 %, distance 13.04 / 13.02 m, RMSE by range 0.93 / 0.22 /
0.20 / 2.65 m (< 8, 8-12, 12-16, > 16 m), 1.66 m overall. `11_stock_walk_mesh_on_video.png` shows the mesh on the
real frames at 2, 8, 14 and 24 s.

Screenshots: `~/.cache/squirrel-guard-test/shots/body3d/05_basement_camera_view_fit.png`,
`05b_basement_camera_view_fit_close.png` (Camera view, mesh over the camera projection),
`06_basement_overview_fit.png` (mesh + stick figure on the path), `07_stock_walk_camera_view_fit.png`, and
`08_basement_mesh_on_video.png` (the mesh projected with the lens distortion onto the real frames at 1.0, 5.0, 8.4
and 10.4 s, which is the actual acceptance check).

## Results of the first version (2026-11-06, RTX 4070 Laptop 8 GB, shared with llama-server)

Runtime per clip (`timing_s`; models load in a few seconds each, not included):

| | basement (walk.mp4, 16 s, 403 frames @ 25 fps) | stock walk (fog.mp4, 28 s, 841 frames @ 30 fps) |
|---|---|---|
| track (CPU, ~8 fps) | 14 s | 27 s (fallback tracker) |
| ViTPose-H fp16, GPU | 9 s (299 frames) | 17 s (638 frames) |
| HMR2 fp32, CPU (GPU too full) | 193 s (278 frames) | 451 s (633 frames) |
| WHAM | 1.2 s (2 runs: 0-57, 72-291) | 1.0 s (frames 4-636) |
| peak VRAM (nvidia-smi, process incl. CUDA context) | 1.7 GB | 1.7 GB |

Torch peak reserved per stage: ViTPose fp16 1.5 GB, WHAM 0.8-1.4 GB. HMR2 fp16 on the GPU takes 9 s / 42 s and
1.6 GB but gives the wrong shape (above). Everything on the CPU: ~0.75 s per frame per ViT model.

| | basement | stock walk |
|---|---|---|
| fitted stature (zero pose, mean betas) | 1.755 m (runs 1.69 / 1.77) | 1.728 m |
| WHAM's world pitch vs our camera | 26-43 deg apart (our camera looks 38.5 deg up; WHAM assumes level) | 8 deg |
| vertical drift of WHAM's own path (removed by snap) | 0.25 / 0.14 m | 0.29 m |
| frames with foot contact, L / R | 98 % / 83 % (reads as standing) | 62 % / 68 % (walking) |
| foot slide in contact, median / p90 | 0.44 / 0.84 m/s | 0.52 / 0.91 m/s |
| median ground distance from camera, WHAM / trajectory | 2.11 / 3.30 m | 9.89 / 10.29 m |
| RMSE pelvis vs trajectory mean (observed frames) | 1.31 m (camera placement 1.32) | 0.61 m (0.66) |
| same after best rigid 2D fit (path shape only) | 0.81 m | 0.55 m |
| bearing from the camera, WHAM vs trajectory (median) | 1.7 deg | 0.4 deg |

Reading:

* **Stock walk**: plausible. Stature 1.73 m against the assumed 1.75 m; the range from the camera agrees within
  4 % (9.9 vs 10.3 m at ~10 m), the path within 0.6 m RMSE, the bearing within half a degree. The difference is
  mostly along the line of sight, where both methods are weakest at 10 m. Contacts alternate like walking.
* **Basement**: the body is placed on the right bearing (1.7 deg) but at **64 % of the trajectory's range**
  (2.1 vs 3.3 m), although WHAM's stature (1.76 m) matches the 1.75 m assumption. WHAM's range uses only our
  focal length; the trajectory's head-plane range also uses the assumed camera height (0.8 m) and pitch (38.5 deg
  up, solved from the same 1.75 m). So the disagreement points at the basement camera model (hfov 95 deg assumed,
  pitch/height), not at the body size: worth a tape-measured check of one position. `--root image` (grounded
  body re-projected through our camera) gives 1.24 m, i.e. the camera model cannot make a 1.75 m upright person
  on the floor look the way the image shows at any single range.
* Failure modes seen: **feet out of frame** (basement: the camera on the desk sees the legs only near the
  start) -> WHAM guesses the legs, often bent or seated (screenshot 01), predicts near-constant contact, and its
  integrated root path barely moves (hence `--root camera`). **Close-up exit** (the person walks past the lens,
  frames 292-403): the track/keypoints drop below 6 confident joints and the run ends; frames 58-72 are another
  gap, so the clip is two WHAM runs with independent shapes (1.69 / 1.77 m). **Tracker**: the current
  `kinematics.track` loses the fog walker (fallback above). **fp16 features** silently shrink the body.
  **Non-level static camera**: WHAM's static-camera world is level with the camera; gravity has to come from our
  calibration (`--align gravity` + snap), and the camera-frame body orientation alone leans ~40 deg (`--align
  camera`).

Screenshots (headless Chromium, SwiftShader; not committed, they show SMPL renders):
`~/.cache/squirrel-guard-test/shots/body3d/01_basement_mesh_and_stick.png` (mesh + 2D stick figure, basement,
t = 8.4 s), `02_basement_camera_view.png`, `03_stock_walk_mesh.png`, `04_stock_walk_camera_view.png` (mesh over
the camera projection, seen from the camera).
