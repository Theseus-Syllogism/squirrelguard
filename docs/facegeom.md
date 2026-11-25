# Face geometry and relative depth (explanatory evidence)

`squirrel_guard/facegeom.py` adds a "how is this face shaped" section to the face pipeline: proportions of facial
features, left/right symmetry and relative depth (nose protrusion, eye sockets, cheek and jaw contour), with
uncertainties, a landmark overlay, a depth heatmap and a small 3D mesh. **It never decides who someone is**: the
512-d embedding (`faceid.py` -> `facematch.decide`) does that. Geometry only explains, feature by feature, how a
visit's face compares with the enrolled geometry of the person the embedding matched (or nearly matched).

Policy (same as `docs/faceid.md`): no ethnicity/race inference, no mapping of geometry to any group, no outside
lookups; geometry is compared only with the same person's own enrolled geometry. Expression stays description-only.

## Method

1. **Face selection** (`GeomCollector`, fed by `faceid.FaceSampler` for every accepted face): faces >= 32 px with
   InspireFace |yaw| < 25 deg and |pitch| < 30 deg. The best 8 by the faceid weight (quality x size x pose x
   sharpness) keep a 256 px crop (face box x 1.7) until the end of the visit (~1.5 MB bounded). Nothing else is
   buffered; the per-frame cost is one crop copy.
2. **Dense mesh**: MediaPipe Face Landmarker (`face_landmarker.task`, float16 v1) on each kept crop at the end of
   the visit: 478 landmarks (468 surface + 10 iris) with a relative z. ~2-5 ms per crop on CPU (2.2 ms measured on the laptop, multi-threaded XNNPACK). InspireFace's 106
   dense 2D landmarks are recorded too (overlay only).
3. **Frontalisation**: x = the eye-corner axis, y = nasion -> chin made orthogonal, z = toward the camera; origin at
   the nasion; all lengths divided by the inter-pupil distance (iris centres, IPD = 1). Width features are measured
   in this face plane, so a residual yaw of 10-25 deg does not shrink them the way a plain 2D measurement would
   (as far as the mesh's own z is right). A face whose mesh pose disagrees with InspireFace by > 10 deg beyond the
   limit is dropped.
4. **Features** (`FEATURES`): nose width (alae 129-358), nose length (nasion 168 - subnasale 2), mouth width
   (61-291), eye widths (33-133, 362-263), inner eye spacing (133-362), face width at the cheekbones (234-454), jaw
   width (172-397), face height brow-chin (9-152), brow-eye height, philtrum (2-0), asymmetry (mean mirror error of
   10 left/right landmark pairs about the fitted midline, IPD units); depth: nose-tip protrusion vs the inner eye
   corners, eye socket depth (brow + cheek surface vs iris centres), cheek and jaw contour set-back (nasion vs
   234/454, chin vs 172/397).
5. **Per visit**: per feature the median over the meshed faces, spread = 1.4826 x MAD, sem = 1.2533 x spread /
   sqrt(n), n; plus the face pixel size (median and range) and |yaw|/|pitch| used, and the per-face values.
6. **Comparison** (`compare`, copy in `deploy/recorder/bridge/facegeomcmp.py`): per feature
   z = (visit - enrolled) / sqrt(sem_visit^2 + sem_enrolled^2 + floor^2), where floor (`WITHIN_SD`) is the measured
   between-visit spread of one person. |z| < 2 "consistent", < 3 "differs somewhat", else "differs". Marked
   `explanatory: true` with a note that the identity was decided by the embedding. Enrolled geometry of an
   identity = `merge()` of its templates' geometries (n-weighted mean of visit medians, spread = pooled within +
   between visit spread).

## Data flow and storage

- Profiler (`[profiler] facegeom_enabled = false` until deployed; `facegeom_model`, `facegeom_faces`,
  `facegeom_max_yaw`, `facegeom_min_px`): `faceid.facegeom` = features + uncertainties + per-face values + overlay
  and depth JPEGs (base64, ~15-25 KB + ~5 KB) + `mesh` (glTF 2.0, KHR_mesh_quantization int16 positions, uint16
  indices, one embedded buffer, ~13 KB of JSON). A missing mediapipe/model only switches the geometry off.
- Image: `mediapipe==1.2.2` installed `--no-deps` (its `opencv-contrib-python` requirement would replace
  `opencv-python-headless`; `matplotlib` is pulled because its drawing utils import it at load), the model is
  `ADD`ed with a pinned sha256 to `/opt/mediapipe/face_landmarker.task`. Checked on Python 3.12 + numpy 2.1.3 +
  opencv-python-headless 4.10 + inspireface 1.2.4.post3 (the image's versions).
- Bridge: `sightings.face_geom` (jsonb: everything except the images/mesh, plus `files` and `compare`); files
  `snapshots/evidence/<event>_facegeom_{overlay,depth}.jpg` and `_mesh.gltf`, removed with the sighting by the
  retention janitor (`evidence/<event>_*`) and as orphans by `facegeom_sweep` (hourly, > 1 h old).
  `face_templates.geom`: when the persons UI enrols a template from a visit, the bridge copies that visit's
  geometry onto the template (`sync_enrolled_geometry`, hourly and before each comparison); it is deleted with
  the template. Schema: `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in `deploy/recorder/persons_schema.sql`.
- UI: `facegeom.render_html(face_geom, compare, img_urls)` returns a self-contained fragment (table with value +-
  spread, stable features first and noisy ones hidden unless `show_all`, the two images, a drag-to-rotate canvas
  mesh view; no external scripts). The persons UI is not wired yet (it is being changed elsewhere): it needs to
  load `face_geom`, serve the three evidence files and call `render_html` (or port it).

## Evaluation

`scripts/facegeom_eval.py` (laptop, Python 3.12 + mediapipe 1.2.2, Megatron, 8 fps, frames downscaled to 2560 px
wide, 2026-11-06). Segments = "visits"; the visit value is the production aggregate (median of the 8 best faces).
Clips: `suit` (369 px faces), `model` (two visits: 44 px and 89 px), the basement `walk` (61 / 112 px, but only
2 + 1 faces pass the |yaw| < 25 gate), `talk` (Pixabay 92662, 100 px talking head, three 10 s visits). Faces in
`hiker`, `mall`, `road`, `couple` are too small or never frontal; `selfie` (Pixabay 61298) and the second
half of `suit` / middle of `model` were rejected by the pose gate in every frame. So: **5 people, 9 usable visits**.

within = CV% of the visit values of the same person (pooled sd / mean); between = CV% of the person means;
ratio = between / within. Depth features in IPD units (sd), since they are relative offsets.

| feature | within (visits) | between | ratio | within, pseudo-visits* | per-face CV <48 / 48-96 / >=96 px |
|---|---|---|---|---|---|
| eye spacing (inner) | 1.3 % | 4.3 % | **3.3** | 0.7 % | 1.5 / 1.2 / 1.5 |
| brow-eye height | 9.3 % | 19.0 % | **2.0** | 1.3 % | 4.5 / 5.0 / 2.4 |
| nose length | 4.2 % | 8.0 % | **1.9** | 2.3 % | 2.6 / 1.4 / 4.8 |
| face height | 2.2 % | 4.2 % | **1.9** | 0.6 % | 2.7 / 1.2 / 1.1 |
| nose width | 3.1 % | 5.6 % | **1.8** | 1.0 % | 3.0 / 2.3 / 4.2 |
| nose protrusion (depth) | 0.018 | 0.029 | 1.6 | 0.007 | 0.02 / 0.01 / 0.01 |
| jaw width | 3.1 % | 4.5 % | 1.5 | 1.1 % | 3.0 / 1.5 / 1.4 |
| mouth width | 5.4 % | 6.6 % | 1.2 | 3.0 % | 3.7 / 3.0 / 15.1 (talking) |
| eye width right / left | 6.7 / 0.9 % | 7.3 / 7.2 % | 1.1 / 7.8 | 0.8 % | ~2 |
| philtrum | 21 % | 23 % | 1.1 | 3.9 % | 7.3 / 3.5 / 5.0 |
| cheek width | 3.3 % | 3.1 % | 0.9 | 0.7 % | 2.3 / 1.9 / 1.6 |
| eye socket depth | 0.022 | 0.022 | 1.0 | 0.003 | |
| cheek contour depth | 0.069 | 0.043 | 0.6 | 0.011 | |
| jaw contour depth | 0.058 | 0.026 | 0.5 | 0.010 | |
| asymmetry | 8.9 % | 118 % | (13; one noisy 3-face visit dominates) | 24 % | 18 / 43 / 28 |

\* pseudo-visits (`--chunk`): consecutive runs of 8 faces inside one segment. They only measure frame sampling
noise, which is 2-10x smaller than the real between-visit spread: what moves a measurement between visits is
distance (face px), pose and expression, not the number of frames.

Findings:
- **Shown as stable** (`STABLE_FEATURES`): inner eye spacing, nose width, nose length, face height, brow-eye
  height, and nose protrusion as the one depth feature (ratio 1.6: indicative only). Everything else is stored
  and shown greyed only with `show_all`.
- Single-eye widths disagree with each other (the right eye's CV is 7x the left's): eyelid landmarks follow blinks
  and squints; mouth width and philtrum follow speech/smiles (15 % per-face CV while talking).
- Depth features other than nose protrusion separate people no better than they wobble: the relative depth mostly
  shows the model's prior face shape. Useful as a picture (heatmap, 3D view), not as a number to compare.
- Size: the same person at 44 px vs 89 px (`model`) differed by 1 % (nose width), 1.2 % (eye spacing), 1 % (face
  height), 0.013 IPD (nose protrusion). Per-face noise below 48 px is ~1.5-2x that of 48-96 px faces. With only
  5 people, ratios of ~2 mean "repeatable enough to explain", not "discriminative": none of this could identify
  anyone, which is the intended role.
- Within-person floors used by `compare()` (`WITHIN_SD`) are kept at or above the measured between-visit sd
  (e.g. nose width 0.035 vs 0.018 measured, eye spacing 0.03 vs 0.007) because the sample is so small.
- Cost: one mesh call ~2 ms per kept face on the laptop (8 per visit; the server-class gpuserver CPU will be slower), payload ~40 KB with the images (overlay ~15-21 KB, depth ~5 KB, glTF 13 KB), ~4-6 KB stored in
  `face_geom` without them.

## Limits

- **2D proportions from small faces are noisy.** Under ~48 px the mesh is fitted to a handful of pixels; the
  landmarks then follow the model's prior more than the face. The UI says so (`MIN_SHOW_PX`).
- **Depth is relative shape, not millimetres.** It is the mesh model's single-image estimate, regularised by its
  learned average face, so the depth features vary little between people and partly reflect the prior. Nothing
  here is a depth sensor.
- Pose: only |yaw| < 25 deg faces are used; many visits (walking past, looking down) have none and get
  `reason: no_frontal_face`. Expression (open mouth, smile) changes mouth width, philtrum and jaw.
- Lens distortion, low light/IR, motion blur and compression all bias landmarks. The stock clips are daylight
  4K->2560 px; the Wyze v4 night IR has not been tested.
- The evaluation set is tiny (few people, stock clips + one basement walk): treat the stability list as provisional
  and re-run `scripts/facegeom_eval.py` on the consenting Wyze v4 clips.

## Licences

- MediaPipe (`mediapipe` package) and the Face Landmarker model: Apache-2.0 (Google; see the Face Landmarker
  model card in the MediaPipe documentation); fine to bake into the image. Downloaded from
  storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  (sha256 64184e22...bc9ff).
- InspireFace model packs (106-point landmarks used for the overlay): InsightFace academic/non-commercial terms, as
  in `docs/faceid.md`.
- Test clips: Pixabay Content License (not redistributed).
