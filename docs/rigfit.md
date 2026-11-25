# Procedural IK rig: whole-body keypoints -> one consistent skeleton per visit (2026-11-06)

| piece | file |
|---|---|
| 133-point whole-body keypoints (RTMW-x, Apache-2.0, ONNX, CPU 118 ms per person) | `squirrel_guard/wholebody.py` |
| rig + IK fit + gait parameters | `squirrel_guard/rig.py` |
| keypoints for every camera of a visit | `scripts/rigfit_keypoints.py OUT.npz --cam v4=V4.mp4 --cam frontyard=FY.mp4` |
| fit, held-out check, write into a replay | `scripts/rigfit.py OUT.npz REPLAY.json --calib v4=... --calib frontyard=... [--use v4] [--heldout frontyard] [--write]` |
| synthetic recovery test | `tests/test_rig.py` |

Model files: `models/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122/end2end.onnx` (and the smaller
`rtmw-dw-l-m ... 256x192`), from download.openmmlab.com (`mmpose/v1/projects/rtmw/onnx_sdk/`), git-ignored.

## The rig

Kinematic tree pelvis -> spine -> neck -> head; shoulders -> elbows -> wrists; hips -> knees -> ankles -> heel/toes.
Segment lengths are stature x anthropometric ratios (Drillis & Contini) x a per-segment multiplier, left = right,
**constant over the visit**; joint angles move per frame within soft anatomical limits (knees and elbows bend one way
only, 0-150 deg). The fit minimises Huber reprojection error in every calibrated camera (lens distortion included,
error measured in fractions of the person's box height so near and far cameras count alike, weighted by keypoint
confidence), plus smoothness over time and heels/toes not below the floor. Confidence = clip((RTMW SimCC peak - 1) /
2.5, 0, 1): raw peaks are ~0.5-1 for unsure/occluded joints and 3-6 for confident ones; the low scores also reject
the coat that YOLO calls a person.

## Measured (18:03 basement walk, v4 + frontyard, rig0 calibration)

| check | old (17-point YOLO + flat card / triangulation) | rig |
|---|---|---|
| **held-out camera** (fit on v4 only, error in frontyard, which the fit never saw), same joints and frames | 1.06 box heights (743 px): feet hidden by the desk put the person metres away | **0.029 box heights (22 px)** |
| joints usable in the held-out camera | 208 | 1469 |
| stature from ONE camera | (assumed 1.70) | 1.68 m (two-camera triangulation said 1.70) |
| reprojection, both cameras fitted | | v4 0.0125 (14 px), frontyard 0.0143 (10 px) |
| synthetic walker, 2 distorted cameras, 2 px noise, 15 % joints missing | | stature 1.767 vs 1.78; worst segment 8.9 %; joints 1.5 cm median; no hyperextension |
| knee / hip flexion range, L vs R (this clip) | | knee 76 / 80 deg (symmetry 0.95), hip 47 / 46 deg (0.98) |

Stature differs between fits (1.62 two cameras, 1.68 v4 only): it is the rig's scale parameter, and the world scale
itself rests on the rig calibration's ASSUMED ceiling/desk heights (docs/multicam.md). One tape measurement fixes it.

**Gait parameters** (cadence, step length, step-time variability) need >= 3 alternating heel strikes 0.3-1.2 s apart.
This clip is mostly shuffling near the desk, so they are honestly not reported. At 8 fps a step is ~4 frames:
for gait, analyse at the camera's full rate (the v4 records 25 fps). Measured 2026-11-08 (docs/fusion.md, "Rig frame
rate vs gait"): heel strikes are whole frames, so cadence is quantised: 10.9 / 6.2 / 2.4 / 0-2 spm error at
8 / 12 / 16 / 20-25 fps on a synthetic walker; fused visits run at `fusion_fps = 16`.

## Infrared

Grayscale (night IR) frames are detected and fed as 3-channel grey. On daylight frames converted to grey, 96-98 % of
joints stay within 3 % of body height of the colour result and 3-4 % of confident joints are lost; CLAHE contrast
equalisation did not help there, so it is opt-in (`WholeBody(..., clahe=True)`). Real night IR (blooming, low
contrast, IR-reflective clothing) still needs a recorded night walk to measure.

## Where the rig runs (2026-11-06, second pass)

- **Demo builders:** `scripts/replay_demo.py ... --rig` and `scripts/multicam_demo.py ... --rig` run the whole-body
  keypoints and the rig fit in one go (shared code: `squirrel_guard/rigpipe.py`). All four basement/multicam samples
  carry `rig`; the stick figure is the rig.
- **Viewer:** "Body & gait" tab in the inspect pane (stature, per-camera fit error, segment lengths, knee/hip/elbow
  angle curves for L and R with a playhead that follows playback, gait with caveats); "Rig feet" toggle draws heels
  and toes. Deep link: `?pane=1&pane_tab=body`.
- **Live profiler (gpuserver):** `rig_enabled = true`: for every replay of a CALIBRATED camera, RTMW l-m keypoints at
  12 fps on the first 20 s of the event clip, rig fit (torch CPU, 2 threads), resampled onto the replay's 8 fps
  timeline; gait parameters come from the 12 fps fit. Cost measured on the laptop: 220 ms per frame for YOLO + RTMW
  l-m, ~10 s fit, ~1 CPU-minute per 20 s event, in the background after the event. Nothing happens for a camera
  without `/calib/<camera>.json`.
- **Single camera:** a ground-contact term (the lowest observed foot on the floor) plus a tighter stature prior fix
  the single-view size/distance ambiguity: v4-only stature 1.735 m, held-out camera error unchanged (0.029).
- **Scores:** `WholeBody` returns a 0..1 confidence calibrated per model (x-l raw peaks 1.0-3.5, l-m 0.13-0.55, matched
  by quantiles on the same frames; the two models' positions agree to 1.1 % of the box height).

## Handedness fix (2026-11-06)

The first rig put the subject's LEFT at +x with forward +y and z up: a mirror image. Fits still matched the 2D
keypoints by turning the figure around (nose behind the head, knees bending forward), so reprojection error could not
reveal it. Found by checking facing on frames where the face is clearly visible: rig nose toward the camera cos -0.88
before, +0.89 after; two-camera stature 1.70 m (matches the triangulation). tests/test_rig.py::test_rig_is_right_handed
pins it. All samples and the live profiler were refitted. The live v4's remaining 2.04-2.06 m was the calibration's
assumed scale: corrected x0.830 from the enrolled owner as a known-height walker (calibration `scale_correction`).

## Rig-driven path and SMPL body (2026-11-06)

- **The rig drives the path.** The rig and the replay path disagreed by 0.31-0.38 m median (up to 2-3.6 m): the path
  came from torso triangulation or head-plane placement, the rig from every camera with constant bone lengths. Now
  `rigpipe.fit_for_replay` -> `trajectory.estimate(ground_override=rigpipe.rig_ground(...))` -> `attach_rig(res=...)`
  in replay_demo, multicam_demo (rig first, triangulation where no rig) and the profiler: median gap 3.5-3.7 cm.
  The viewer no longer slides a rig's stick figure onto the path.
- **3D body for every rig replay** (`squirrel_guard/smplfit.py`, `scripts/body3d_rig.py`, profiler `body_enabled`):
  SMPL (plain-torch forward pass from the official .npz) dressed onto the rig's 23 markers: shared shape, per-frame
  heading/root/23 joint rotations/translation, Huber marker loss + pose/shape priors + smoothness. Marker error
  1.1-1.2 cm median; 8-46 s offline, 2 s in the profiler for a 33-frame visit; ~1.3 MB float16 vertices (every 2nd
  frame). Replaces WHAM's single-view placement on the basement and multicam samples. SMPL outputs stay local
  (`*_body.bin`, `*_body_faces.bin`, `*_body3d.json` git-ignored; the model is mounted at run time on gpuserver).
- **v4 scale:** the first known-height correction (x0.830) used single-camera statures already pulled toward 1.75 m
  by the rig's stature prior. Unbiased (prior off) the v4 still read 1.95-2.02 m, so the total factor is x0.711
  (camera 0.79 m, sitting on a table).

## Not done yet

- ~~Multi-camera fusion for LIVE visits~~: done 2026-11-08, docs/fusion.md (overlapping events of cameras sharing a
  calibration frame -> one rig over all views; frame rate vs gait measured there).
- Hands and face are in the rig as an opt-in third stage (below), not yet in rigpipe/profiler; face geometry
  (identity-free proportions) is a separate track.
- SMPL/WHAM meshes can be driven from the rig instead of WHAM's own single-view placement (now anchored to the path).
- Real night-IR measurement.

## Hands and face in the rig (2026-11-08, opt-in)

`rig.fit(views, ..., parts=("face", "hands"))` adds a third stage after the body (the body result is unchanged:
body markers moved 0.0 cm median on both real clips; without `parts` the output is exactly as before, and
`rig.forward` still returns 23 markers). Code: `rig.forward_parts`, `rig._fit_parts`, `rig.PART_GATE`.

* **Face:** a fixed 68-point mean adult face (hand-built from craniofacial anthropometry, in the head frame) moves
  with the existing 3-dof head joint; one jaw-opening dof (rotation about the temporomandibular axis, lower lip and
  chin weighted). Outputs: `head_R` (world head frame), `head_ypr_deg` (yaw/pitch/roll), `jaw_deg`.
  The face is used for head orientation only, never for identity.
* **Hands:** per side wrist pronation/flexion/deviation (3 dof) + one curl per finger (MCP:PIP:DIP = 1:1:0.7),
  constant bone lengths (hand length 0.108 stature, one free per-visit hand and face scale with a prior), soft limits
  (`LIMITS_X`), coarse per-frame wrist start (palm vs back trap). Outputs: `hand_R`, `palm_normal`, `finger_dir`,
  `angles_x_deg` (`ANGX_IDX`).
* **Gates (hard), per view and frame:** >= 34 face / 14 hand points at confidence >= 0.5; expected part size >= 24 px
  (focal x metric size / depth from the body); hand wrist point within 0.35 hand lengths of the body wrist; after the
  fit, view-frames whose median residual exceeds 0.12 face / 0.20 hand sizes are dropped and the stage refitted.
  `res["parts"]["face_usable"]`, `["hand_usable"]` and `["usable_rate"]` say where the result is data, not prior.
* **SMPL** (SMPL has no fingers/jaw; SMPL-X / SMPL+H are not on this machine, below):
  `smplfit.fit_to_rig(..., head_R=, head_ok=, hand_R=, hand_ok=)` pulls SMPL's head joint and hand (fingertip
  vertices, SMPL-H topology ids) toward the rig's frames on usable frames.
* Tests: `tests/test_rig.py` synthetic recovery (two cameras, face ~70 px, hands ~46 px, 1 px noise, 15 % dropped):
  head 1.1 deg median, jaw 0.7 deg, palm normal / finger direction 2-3 deg, curls 1.5-2 deg; 9 m cameras (hands ~12 px)
  -> hands never used.

### Measured

| clip | face usable | L hand | R hand | head vs MediaPipe (residual median after a constant offset, deg): body-only -> with face |
|---|---|---|---|---|
| demo0246 testcam+v4 (seated, frontal face IPD 69 px, head nearly still: MediaPipe yaw range 4 deg) | 62 % of frames (testcam 138/223; v4 0: back of head) | 0 % (below the testcam frame; ~10 px on the v4) | 0 % | 3.1 -> **2.3** (138 frames; p90 5.9 -> 11.1); constant offset 9.6 -> 5.8 |
| 18:03 walk v4+frontyard (faces 23-33 px IPD, head turning: MediaPipe yaw 5-67 deg) | 46 % of body frames (v4 27, frontyard 41 view-frames) | 57 % | 53 % | v4 11.9 -> 12.2 (29 frames); frontyard 12.5 -> 15.1 (19; p90 38 -> 27) |

Reading: at desk distance (face >= ~60 px) the face points tighten head orientation; at walking distance
(faces 25-35 px) they do not beat the 5-point (nose/eyes/ears) head, and MediaPipe itself is no reference there
(its yaw vs the rig correlates only 0.3-0.6 either way). Hands pass the gate on ~half the walk's body frames (hand
extents 40-80 px); their orientation is checked only synthetically (no independent hand reference on these clips).
SMPL driven by the rig: head 4.5 -> 0.2 deg, hand finger direction 25 -> 0.0 deg, across-hand 50 -> 0.0 deg from
the rig's frames on usable frames (18:03), marker error unchanged 1.1 cm; on the seated clip 1.9 -> 2.2 cm.
Cost (18:03, 267 frames, 2 torch threads): body 15 s, with parts 34 s; with torch's default thread count on a
loaded laptop the same fit took 41 min (oversubscription): set torch threads explicitly.
Scripts: `scripts/rig_handface_eval.py` (fit both ways, usability, comparison), `scripts/mp_headpose.py`
(MediaPipe legacy face mesh, run in ~/.cache/squirrel-guard-test/facefit-venv).

Body models: SMPL-X (smpl-x.is.tue.mpg.de) and SMPL+H (via MANO, mano.is.tue.mpg.de) both need a registered
account and are licensed for non-commercial scientific research, like SMPL: usable here only if the user downloads
them and the outputs stay local and git-ignored. Neither is on this machine (only WHAM's smplx2smpl.pkl mapping).
