# One fitted face per visit (explanatory evidence)

`squirrel_guard/facefit.py` fits **one consistent 3D face shape per visit** to every usable frame, from one or two
calibrated cameras. It replaces `facegeom.py`'s approach (MediaPipe meshes of the 8 best near-frontal crops,
measured one by one, then per-feature medians). The output keeps facegeom's feature names and units (IPD = 1), so
`facegeom.compare` / `merge` and the persons UI table work unchanged. Where two calibrated cameras see the face, it
adds millimetres. **It never decides who someone is**: the 512-d face embedding does that (`faceid.py` ->
`facematch.decide`).

Policy (same as `docs/faceid.md` and `docs/facegeom.md`): no ethnicity/race or any group inference. Geometry is only
compared with the same person's own enrolled geometry. No outside lookups. Expression is never classified: blendshape
scores or geometric proxies only *down-weight* frames and regions, and the summary is a description-only
"apparent expression" share. FLAME, DECA, EMOCA, MICA and SMIRK are not used (see Licences). Nothing from FASR++ is
used.

## Method

1. **Per frame** (`landmarks_in_window`). The head window comes from the YOLOv8-pose head keypoints
   (`faceid.head_window`); the profiler can pass InspireFace's face box instead (`FitCollector`). It is cropped
   square and resized to 256 px. If no face is found, the window is retried at x0.7 and x1.4. MediaPipe's 478-point
   face mesh then runs twice: the second pass uses a crop re-centred on the first pass's landmarks (face extent
   x 1.6), as MediaPipe's own tracker does. Kept per frame: landmarks in frame px (z relative), IPD px, crop size,
   sharpness (Laplacian variance at 128 px), quick yaw/pitch, blendshape scores (Tasks landmarker only), and an
   IR/grayscale flag. Only the landmarks are kept (about 6 KB per frame), never the crops.
2. **Lens and viewing direction** (`undistort_landmarks`). With a calibration, the landmarks are undistorted (K, D)
   and **re-projected into a virtual pinhole camera aimed at the face**. Plain undistortion to a rectilinear image
   stretches an off-axis face radially by about 1/cos of its off-axis angle (15 % at 30 deg). A per-frame
   similarity cannot absorb that stretch, and it showed up as a 9 % shorter face and the worst time-half drift
   (see Evaluation). Without a calibration, the raw landmarks are used.
3. **Identity fit** (`fit_identity`). The shape is fitted on the MediaPipe canonical topology: 468 surface vertices +
   10 iris points, with 898 triangles once the eyes and mouth are closed. The 852 open triangles are embedded in the
   module, derived from mediapipe's `FACEMESH_TESSELATION`, Apache-2.0. Model: `X_f ~= s_f R_f U + t_f` for every
   frame. It alternates weighted Umeyama poses with a linear shape step, solved by block-Jacobi conjugate gradient
   (numpy only, a few ms per iteration). The shape step weights:
   - x/y residuals at full weight; MediaPipe's z at 0.35 (it is a learned prior, not a measurement);
   - Huber weights per frame x vertex (k = 2 robust sigma) and per frame (k = 2.5, catches a wrong face or a bad crop);
   - face size `min(1, IPD/30 px)^2` (frames under 8 px IPD are dropped) x sharpness vs the visit median;
   - a pose gate at |yaw| 50 deg, |pitch| 40 deg, and per-vertex visibility: `0.15 + 0.85 * max(0, n . view)`,
     because the hidden cheek of a turned face is only guessed;
   - expression, by region (mouth/jaw, eyes, brows: x(1 - 0.85 score)) and per frame (x(1 - 0.4 max score)). The
     scores come from blendshapes when the Tasks landmarker gives them (jawOpen, smiles, blinks, brow...), otherwise
     from geometric proxies relative to the visit's resting state: inner-lip gap, mouth width, eye opening, brow
     height;
   - a Laplacian regulariser `lam ||L (U - M)||^2` (lam = 1.5 full-weight frames) toward a prior M. M is
     MediaPipe's canonical face when the `face_landmarker.task` bundle carries `geometry_pipeline_metadata`
     (`canonical_from_task`, parsed with a 30-line protobuf reader). Otherwise it is the visit's per-frame median
     shape after Taubin smoothing. On synthetic faces with 3-6 noisy frames, it cut the shape error by 21-33 % vs
     no regulariser (35-40 % vs the plain per-frame median), and changed it by < 2 % with 30 frames.
   Gauge: nasion at the origin, eye line = x, iris IPD = 1 (`facegeom.frontalise`). Uncertainty: 24 bootstrap
   resamples of the frames (3 warm-started iterations each) give the per-vertex RMS and a per-feature sd.
4. **Two calibrated views** (`multiview`). Frames from two cameras at the same instant: each landmark's two rays
   (`Camera.rays`) are triangulated (midpoint), and the ray gap checks sync and calibration.
   - **Metric scale**: the face's distance from each camera (10 %-trimmed mean of the triangulated landmarks)
     x the fit's px-per-IPD in that camera's face-centred view / focal length, giving mm per IPD. It is robust
     because it needs only the face's distance, not its relief. The median over pairs and both cameras is used when
     its sem is under 5 %.
   - **Depth stretch** k in `Xw ~= s R diag(1, 1, k) U + t` measures how much MediaPipe's relative depth must be
     stretched. It is applied only when consistent between pairs (sd <= 0.25). The expected depth error
     `Z^2 / (f B)` per px of landmark error is reported, because a short baseline cannot resolve a face's 3-4 cm
     relief.
5. **Outputs** (`fit_visit` payload; `strip()` removes the mesh and Python objects for storage):
   - `features`: the facegeom features of the fitted shape, each with its bootstrap sd;
   - `metric`: mm with sd, when the scale was used. Depth features stay relative unless the depth was resolved;
   - `multiview`: per pair IPD, depth stretch, ray gap, reprojection, distance, and why a step was not used;
   - per-frame reprojection error (median px and % of IPD), yaw range, the apparent-expression shares, and the
     bootstrap vertex sd;
   - `mesh`: one glTF with four LODs (one scene each, default LOD1).
6. **LODs** (`lods`, `gltf_lods`, `gltf_to_glb`; numpy only):
   - LOD0: greedy QEM edge collapse onto existing vertices, with boundary constraint planes, link-condition and
     flip checks. Every landmark `facegeom.features_of` reads survives exactly, so features on LOD0 equal those on
     LOD1.
   - LOD1: the native 898 triangles.
   - LOD2 / LOD3: Loop subdivision (Warren weights, B-spline rules on the face outline).
   - Encoding: KHR_mesh_quantization int16 positions, int8 normals, uint16 indices, and the bootstrap uncertainty
     as a `_UNCERTAINTY` vertex attribute on LOD1. The node scale makes the file metric (metres) when the two-view
     scale was used, else IPD units.

## Evaluation

`scripts/facefit_eval.py`, laptop CPU, 2026-11-06. Data: the consenting owner's two-camera walk `demo1803`: Wyze v4
2560x1440 and frontyard 1920x1080, joint calibration `rig0`. The rig's metric scale rests on an assumed 2.3 m
ceiling. Frames decoded at 8.3 fps, faces found by the pose model + head window. Landmarker: mediapipe 0.10.21's
legacy `face_mesh` with `refine_landmarks` (same 478-point topology, no blendshapes), because `face_landmarker.task`
is not on the laptop and no model was downloaded. The production image runs the Tasks model (`TaskLandmarker`),
which adds blendshapes and the canonical prior; the numbers below use the geometric expression proxies and the
smoothed visit-median prior. The old method is evaluated on the **same per-frame landmarks**, so the comparison
isolates the aggregation:
- `facegeom best-8`: production's choice, |yaw| < 25, |pitch| < 30, best 8 by size x pose x sharpness, medians;
- `facegeom all-frontal`: median over every near-frontal frame. This is a stronger baseline that production cannot
  run today, because it buffers only 8 crops.

Visit: v4 has 45 faces (t = 9.6-19.9 s, IPD 16-59 px, median 34), frontyard 28 (IPD 10-28 px, median 24). The
fit used 73/73 frames, yaw -36..+50 deg.

### Repeatability (split the visit's frames in two, fit each half, compare)

Mean |A - B| in % of the value, averaged over facegeom's `STABLE_FEATURES` proportions (eye spacing, nose width,
nose length, face height, brow-eye) and over all 11 proportions. Random = mean of 20 random half-splits (sampling
noise). Halves = first vs second half in time; these differ in distance and pose, so the comparison also tests
robustness to viewing conditions. Cross-camera = v4-only vs frontyard-only (another lens, distance and angle).

| split | facegeom best-8 | facegeom all-frontal | **facefit** |
|---|---|---|---|
| v4, random halves (stable / all props) | 4.3 / 5.1 % | 3.1 / 3.4 % | **2.2 / 3.0 %** |
| v4, odd/even | 3.0 / 3.2 % | 3.1 / 3.3 % | **1.5 / 1.9 %** |
| v4, time halves | 3.5 / 6.6 % | 4.0 / 5.6 % | **2.5** / 6.4 % |
| frontyard, random halves | 4.5 / 4.8 % | 2.9 / 3.6 % | **2.6 / 3.3 %** |
| frontyard, odd/even | 1.8 / 3.3 % | 1.8 / 2.2 % | **1.0 / 1.7 %** |
| frontyard, time halves | 2.5 / 2.6 % | 2.0 / **2.4 %** | **1.9** / 4.0 % |
| both cameras, random halves | 3.8 / 4.4 % | **2.1** / 2.7 % | 2.4 / 2.8 % |
| **cross-camera** (v4 vs frontyard) | 5.5 / 4.7 % | 3.5 / 4.6 % | **2.4 / 2.7 %** |
| shape RMS, random halves v4 / frontyard (IPD) | 0.039 / 0.033 | 0.031 / 0.031 | **0.023 / 0.023** |
| shape RMS, cross-camera (IPD) | 0.046 | 0.033 | **0.028** |
| nose protrusion (depth), cross-camera (IPD) | 0.037 | 0.034 | 0.035 |

Per feature, random splits, v4 (best-8 / all-frontal / facefit, %): eye spacing 2.9 / 1.6 / 2.8, nose width 3.2 / 2.8 / 1.8, nose length 5.8 / 4.0 / 1.9, face height 4.2 / 3.5 / 2.2, brow-eye 5.5 / 3.6 / 2.3, mouth width 5.9 / 3.9 / 3.0, jaw width 2.3 / 1.6 / 2.3, cheek width 2.4 / 1.9 / 2.3, philtrum 12 / 6.3 / 7.8. Cross-camera (same order): eye spacing 3.2 / 1.9 / 0.6, face height 7.0 / 4.2 / 0.9, nose length 5.9 / 3.2 / 1.4, nose width 3.3 / 2.5 / 4.1 (the one stable feature where facefit is worse).

Findings:
- Against production's best-8 aggregate, facefit **halves the sampling noise** of the stable proportions (v4
  random splits 4.3 % -> 2.2 %, frontyard 4.5 % -> 2.6 %). It agrees **twice as well between the two cameras**
  (5.5 % -> 2.4 %, shape RMS 0.046 -> 0.028 IPD), which is the closest proxy here for "another visit": different
  lens, distance and angle.
- The stronger, hypothetical baseline (median of every near-frontal frame) is about as good as facefit on random
  splits that pool both cameras (2.1 vs 2.4 %). It is worse across cameras (3.5 vs 2.4 %) and on single-camera
  splits.
- Time halves (approach vs close-up) still move the face outline: jaw and cheek width change by 4-8 % in v4. The
  outline landmarks of a turned or near face are partly guessed. Inner-face proportions are stable (2.5 %).
- Philtrum and single-eye widths stay noisy (blinks, speech); depth features are no better than facegeom's, as
  expected for a learned relative depth.
- **Lens**: plain undistortion to a rectilinear image (the face up to ~35 deg off-axis in the wide Wyze lenses)
  shortened the fitted face by 9 % and gave the most time-half drift (stable proportions 3.9 / 3.8 % on v4 /
  frontyard). The face-centred view fixes both: 2.5 / 1.9 %, against 2.9 / 1.0 % for raw (distorted) landmarks.
  Its face height (1.907 IPD) matches the median of the 27 near-frontal frames (1.914); raw gives 1.974 and
  facegeom's best-8 1.977. Raw and face-centred are equally repeatable here, but only the face-centred view is
  geometrically consistent with the calibration (the two-view step needs that), so it is the default when a
  calibration exists.
- On synthetic faces (`tests/test_facefit.py`), the fit beats the per-frame median (0.017 vs 0.026 IPD RMS with 30
  frames) and recovers yaw within 2 deg. Two "other face" frames and six open-mouth frames are down-weighted.
- Bootstrap: median per-vertex sd 0.018 IPD (p90 0.027); feature sd 0.3-1.5 % for the stable proportions. This is
  sampling noise only, so it is smaller than the cross-camera differences (systematic, 1-4 %).

### Two cameras

- **Sync**: the ray-gap scan of face landmarks at 25 fps (`--mv-fps`) puts frontyard at **-47 frames = -1.88 s**
  (median gap 4.5 mm; +-1 frame: 4.7 / 5.2 mm; +-8 frames: 14-19 mm). At 8.3 fps the same scan picked -16 frames
  (-1.92 s, gap 5.2 mm) over the body-joint estimate of -15 (-1.80 s, 7.3 mm). The body-joint value from
  `multicam_demo.py` is about one 25 fps frame late for the face.
- **71 simultaneous pairs** (v4 IPD ~32 px, frontyard ~24 px, face 1.7-3.3 m from v4, median 3.0 m).
- **Metric scale: used.** IPD 61.3 mm (pair sd 5.9, sem 0.9; v4 62.4, frontyard 61.1 separately). This is within
  the usual adult range, which suggests the interim rig scale is right to a few %. The 3D similarity fit of the
  noisy triangulated points says 74 mm (biased by depth noise), which is why the scale comes from distance x
  angular size.
- **Depth: not resolved** -> single-view relative depth kept. Expected depth error 22 mm per px of landmark error (0.24 m
  baseline at 3 m); per-pair depth stretch 0.2-5.0 (sd 0.88). The triangulated face is a 26 mm (median) noisy
  cloud around the fitted one. Measuring the ~40 mm nose/cheek relief needs a baseline of >= 1 m at this distance,
  or the face within ~1.2 m of both cameras.
- Metric reprojection of the scaled fitted face into both cameras: 5.6 / 4.4 px median (the depth noise of the
  per-pair 3D fit); the single-view fit's own reprojection is below.

### Reprojection

Median 2D distance between the fitted face (per-frame similarity) and the frame's landmarks: v4 **1.6 px**
(p90 3.4 px, 4.5 % of IPD), frontyard **0.8 px** (p90 1.6 px, 3.6 % of IPD); all 73 frames used (none rejected
by the frame Huber weight). The residual is the per-frame landmark noise + expression + what a rigid shape cannot
explain.

### IR / night

The Wyze cameras switch to grayscale IR at night. The test turned the colour frames to grayscale (3 equal
channels, which is what the IR stream delivers: `is_grayscale` flags the real IR frame). The pose model, head
windows and landmarks were then re-run per mode. `irsim` adds a rough night look (contrast x0.6, 1.2 px blur, noise
2 grey levels). The real IR snapshot is so denoised by the camera (noise 0.21 grey levels, same as the daylight
frames' 0.23) that noise matching was pointless. Detection = frames with a face / frames where the colour pose
model saw the nose and both eyes (v4 39, frontyard 33). Shift = median per-frame landmark difference vs colour
(includes the different head windows on gray frames). Jitter = the fit's median reprojection residual.
Features = mean |change| of the 5 stable proportions of the fitted face vs the colour fit.

| mode | detection v4 / frontyard | landmark shift vs colour (% IPD) | fit jitter (% IPD) | features vs colour | shape RMS vs colour (IPD) |
|---|---|---|---|---|---|
| colour | 0.79 / 0.73 | - | 4.5 / 3.6 | - | - |
| gray, plain | 0.77 / 0.70 | 9.9 / 6.7 | 4.6 / 3.9 | 1.9 / 2.1 % | 0.023 / 0.016 |
| gray + CLAHE always | 0.69 / 0.76 | 11.7 / 8.7 | 6.0 / 5.4 | 1.4 / 0.9 % | 0.024 / 0.024 |
| **gray, CLAHE retry (default `ir="auto"`)** | **0.82 / 0.79** | 10.7 / 6.8 | 4.9 / 4.5 | 1.9 / 1.8 % | 0.021 / 0.018 |
| irsim (CLAHE retry) | 0.67 / 0.55 | 8.0 / 11.9 | 4.7 / 4.1 | 3.1 / 2.4 % | 0.027 / 0.040 |

- Grayscale costs little: the fitted face moves by about as much as between two random halves of the colour visit
  (0.02 IPD).
- CLAHE on every crop finds more of the small frontyard faces but makes landmarks jitter ~30 % more. As a retry
  only (when the plain grayscale crop gives no face), it has the best detection rate of all modes and near-plain
  jitter, so that is what `ir="auto"` does.
- The low-contrast, blurred `irsim` loses 15-25 % of detections (mostly the small frontyard faces); the fitted
  shape still stays within ~3 % of colour.
- Real IR frame (`multicam/20261106-1752/frontyard.jpg`, no person near): grayscale detected. The pose model
  returned one low-confidence "person" without head keypoints, so no face was attempted. A brute-force sweep of the
  mesh model over 63 overlapping windows of the empty IR scene produced 1 spurious "face". The production path only
  meshes inside a person's head window, so that matters little, but the Huber frame weight is the backstop.

### LODs and runtime

| LOD | triangles | vertices | .glb bytes | glTF JSON (base64) bytes |
|---|---|---|---|---|
| LOD0 (QEM, feature landmarks kept) | 250 | 144 | 4 316 | 5 407 |
| LOD1 (native) | 898 | 468 | 12 092 | 15 778 |
| LOD2 (Loop x1) | 3 592 | 1 833 | 44 644 | 59 178 |
| LOD3 (Loop x2) | 14 368 | 7 257 | 174 392 | 232 173 |
| all four in one file (scenes) | | | 234 916 | 312 112 |

LOD1 is about the size of facegeom's mesh (13 KB) and adds normals and the uncertainty attribute. LOD2/LOD3 are
deterministic subdivisions of LOD1, so storing LOD0+LOD1 (16 KB) and subdividing in the viewer (or on demand)
avoids keeping 170 KB per visit.

Runtime (i7-14700HX laptop, CPU):

| step | single-thread BLAS | 4 threads |
|---|---|---|
| YOLOv8s-pose per 2560x1440 frame (already run by the profiler) | 242 ms | 98 ms |
| face mesh, 2 passes, per frame (legacy model, XNNPACK) | 14.6 ms | 19.7 ms |
| identity fit, 73 frames, no bootstrap | **95 ms** (1.3 ms/frame) | 133 ms |
| + 8 / 24 bootstrap resamples | 448 / 1179 ms | 606 / 1635 ms |
| LODs (decimation + 2 subdivisions) + glTF | 165 + 3 ms | 218 + 3 ms |

The fit is faster with one BLAS thread (small matrices; thread start-up dominates), so set
`OPENBLAS_NUM_THREADS=1` (or threadpoolctl) around it in the profiler. Per visit: ~15-20 ms x frames for
landmarks, plus about 1.5 s for fit + bootstrap + LODs. Single-threaded on the server-class CPU, expect 2-3x that.

## Wiring (not done)

- **Profiler**: `profiler.facegeom_attach` creates a `facegeom.GeomCollector` on `fs.geom`; `FaceSampler.on_box`
  calls `offer(frame, o, w)` for every accepted face, and `result()` lands in `faceid["facegeom"]`.
  `facefit.FitCollector` has the same interface. Wiring it needs:
  1. a second attribute (e.g. `fs.fit`) fed next to `fs.geom` (or replacing it), with its result under
     `faceid["facefit"]`;
  2. `[profiler] facefit_enabled`, `facefit_model` (the same `face_landmarker.task`; use `TaskLandmarker` for
     blendshapes and pass `canonical_from_task(model)` as `prior`), `facefit_max_frames` (96) and `facefit_max_ms`
     (4000 ms of landmarker time per visit);
  3. the camera calibration (`geometry.Camera` + frame size) passed as `calib` when the camera has one. Without it
     the fit still works, but off-axis faces from the wide Wyze lenses are stretched (see Evaluation).
  Cost: ~15-20 ms landmarks per offered face (two passes), plus ~0.1 s fit, ~1.2 s with the 24 bootstrap
  resamples and 0.17 s LODs (single-thread BLAS). Use `boot=8` on the server-class CPU, or run the fit after the
  event is published.
- **Two-camera metric**: it needs simultaneous frames from two calibrated cameras. That lives in the multicam flow
  (`scripts/multicam_*.py`, `rig*.py`; another engineer's area). It must hand over a list of `(FrameObs_a,
  FrameObs_b)` pairs at a common time base. Use 25 fps, not 8 fps: at 8 fps a half-frame offset of a walking
  person is 6 cm, which the ray gap does not see along the epipolar line. Then call
  `fit_visit(obs, cams, pairs=...)`.
- **Bridge / storage**: store `strip(payload)` (features, metric, multiview summary, sd; ~4-6 KB) next to
  `sightings.face_geom`, and the mesh as a `.glb` evidence file (LOD0+LOD1 ~16 KB; LOD2/LOD3 can be regenerated
  client-side by Loop subdivision or kept server-side). Enrolment copies the payload like `face_geom`.
  `facegeom.compare` / `merge` accept the `features` block once each entry carries `sem` (use the bootstrap sd as
  sem) and `n` (frames used).
- **Persons UI**: `facegeom.render_html` can render the feature table as is. The 3D view should read the glTF
  scene LOD1 (or LOD0 on slow clients) and can colour vertices by `_UNCERTAINTY`. Show `metric` only when
  `multiview.scale_used`, with the "as accurate as the rig calibration" note.

## Limits

- One person, one visit, two cameras: the repeatability numbers measure the pipeline's stability, not how faces
  differ between people. The between-visit spread (`facegeom.WITHIN_SD`) still has to be re-measured on more
  visits before `compare()` floors change.
- MediaPipe's z is a learned prior. With these cameras (0.24 m apart, face ~3 m away), the depth relief cannot be
  measured; depth features remain relative, exactly as in facegeom.
- The metric scale is only as good as the rig calibration (interim, scale from an assumed ceiling height). The
  measured IPD (61 mm, within the usual adult range) is a plausibility check of that scale, not a measurement to
  rely on.
- The evaluation used the legacy face mesh without blendshapes and without the canonical prior; with the Tasks
  model both switch on, and the numbers should be re-run (`--model face_landmarker.task`).
- The canonical-mesh reader is tested on synthetic protobuf bytes in the documented field layout; it has not seen
  a real `face_landmarker.task` (none on the laptop). It returns None on anything unexpected, and the fit then uses
  the smoothed visit median.
- Grayscale IR: only grayscale-converted daylight frames and one real IR snapshot without a close face were
  available (no Frigate night recordings were fetched). The real Wyze night look (IR skin reflectance, glowing or
  dark pupils, ISP denoise) is not covered by the simulation.
- Very turned faces (|yaw| > 40 deg): the hidden half is down-weighted, not measured; outline widths (cheek/jaw) move
  with the time-half split (distance/pose change) by up to 6-8 %.

## Licences

- MediaPipe (`mediapipe`), the face mesh topology embedded here and the Face Landmarker / legacy face mesh models:
  Apache-2.0 (Google).
- Not used, possible future options, all **non-commercial research licences** (not compatible with this project's
  use without a separate licence): FLAME (MPI-IS FLAME licence), DECA, EMOCA, MICA, SMIRK (built on FLAME, MPI-IS /
  authors' non-commercial terms). A metric face prior with a 3DMM basis would need one of these or a permissively
  licensed alternative.
- FASR++: no licence published; no code from it is used or read.
- InspireFace model packs (face boxes for `FitCollector` in production): InsightFace non-commercial terms, kept local.


## In the persons UI and the replay pane (2026-11-07)

- **Prior 3D model index** (`deploy/recorder/persons-ui/facemodels.py`, numpy only): an identity's fitted faces are read
  from their GLBs (LOD1 = the 468 MediaPipe surface points in a fixed order, so models correspond point by point),
  normalised to the eye spacing, aligned by generalised Procrustes (weights = frames used) into the identity's mean face;
  each model's per-point deviation from the mean (or from another visit of the SAME identity) is reported as RMS / p90 /
  max in % of the eye spacing. Owner on 2026-11-06: 3 models over both cameras agree within 2.05 % (median); the
  39-frame visit sits 1.18 % from the mean, a 12-frame one 3.74 %.
- **API**: `/api/identity/<id>/faces` (index, mean, per-feature spread across visits), `/api/identity/<id>/face_mean.glb`,
  `/api/face/<event>/deviation?ref=mean|<event of the same identity>` (cross-identity comparison is refused: 404),
  visit API `face_fit` block.
- **Viewer** `replay/face3d.html?identity=<id>[&event=..&ref=mean|..&color=deviation|uncertainty]`: model picker (visits +
  identity mean), compare-with, colour by deviation (green 0 -> red >= 5 % of the eye spacing) or by the fit's bootstrap
  uncertainty, LOD0-3, wireframe, `&embed=1` for frames.
- **Replay pane, Face tab**: "3D face" section (embedded viewer for the visit, fit summary, the identity's prior models
  with "View vs this"); **identity page**: "Face models (3D)" (viewer + table with deviation from the mean); **Why? page**:
  the 3D link opens the identity view.
- Policy unchanged: shapes are compared only within one identity, never used to search, rank or name anyone.
