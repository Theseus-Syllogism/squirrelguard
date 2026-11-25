# Face identification v2: AdaFace on the same detections (opt-in)

`squirrel_guard/faceid_v2.py` adds an AdaFace embedding computed on the face the production pipeline already finds.
InspireFace still does detection, 5-point landmarks, quality, liveness and pose, and the result keeps the same
weighted aggregator. **`faceid.py` is unchanged and remains the default.** The privacy rules of
[faceid.md](faceid.md) apply unchanged: only enrolled, consenting people are named; there is no race or ethnicity
output and no outside lookup. Liveness still gates household. Face geometry is never an identity input, and no
super-resolution or face restoration is used.

**Short answer (2026-11-07):** switch the embedding to **AdaFace IR-101** (IR-50 if CPU is tight). Keep the
production aggregator, keep a single mean template per identity, and do not fuse with InspireFace. The win is
clear on public low-res pairs at 24-32 px and on side-on faces. On our own camera it is a wider margin, not a
measured accuracy gain: the camera data is one person on one afternoon.

## What was tried

| candidate | result | keep? |
|---|---|---|
| AdaFace IR-101 WebFace12M embedding | best at 24-32 px (CelebA TAR@0.1 % FAR 0.78 -> 0.95 at 24 px, CIs do not overlap); camera: weakest genuine visit 0.195 -> 0.401 with the stranger max unchanged (0.118 / 0.121); side-on faces (yaw >= 45) genuine median 0.30 -> 0.56 | **yes** |
| AdaFace IR-50 WebFace4M | nearly as good on CelebA (0.93 at 24 px); weaker on camera (genuine min 0.27); half the CPU | fallback |
| Score fusion InspireFace + AdaFace (impostor z-norm) | never better than IR-101 alone (CelebA 24 px 0.936 vs 0.950); costs both models | no |
| Multi-template gallery (pose bucket x colour/IR) and per-visit max | raises the stranger max (0.12 -> 0.15-0.18) more than the genuine min | no |
| Best-K faces (K = 1/3/5/10) instead of the production aggregator | worse or equal everywhere (camera IR-101 median 0.691 prod vs 0.597 / 0.653 / 0.681 / 0.684) | no |
| AdaFace feature-norm weighting in the aggregator | +-0.005, no change | no |
| Open-set threshold from impostor scores | implemented (`Calibration`); thresholds below | yes, as a procedure |
| Body re-ID / stature side evidence | 1 owner sighting with no usable face; stature not stored: **cannot conclude** | reported only |

## Data (sample sizes, plainly)

- **Camera, owner** (identity #1): 21 visits with >= 3 faces of >= 24 px. Tiers: A = 5 (the system's face
  decision), B = 11 (InspireFace cosine >= 0.40 to A; **selected by the baseline model, so biased in its
  favour**), R = 5 (identity #1 by body re-ID). All from 2026-11-06, 15:33-20:14, one or two outfits, two
  cameras. No labels.csv exists yet. There are **0 confirmed in-domain strangers.**
- **Camera, strangers:** 16 stock-clip visits (Pixabay, one uploader per clip; bright and out of domain). These
  stand-in impostors are optimistic.
- **Night IR:** 16 owner faces judged greyscale by `ir_score` and 0 stranger IR faces: **too few to conclude**.
  The CelebA greyscale rows are only a colour-loss proxy, not real NIR.
- **Public pairs:** CelebA, one test shard (315 identities, 2,118 probe photos), each photo downscaled so the face
  is 24/32/48/64 px, JPEG q=75, then the production detector plus alignment. The enrolment photo is native
  (~90 px).
- With 21 correlated visits from one afternoon, the camera AUC is 1.00 for every variant and all CIs are
  1.00-1.00, which says nothing. The margin (genuine min against stranger max) is the only informative camera
  number, and it is not significance-tested.

## Results

CelebA, one native enrolment photo against one low-res probe (TAR at FAR; 90 % CI from resampling identities):

| probe | InspireFace TAR@1 % / @0.1 % | AdaFace IR-50 | AdaFace IR-101 | fusion ISF+IR-101 |
|---|---|---|---|---|
| colour 24 px | 0.899 / 0.784 (0.76-0.80) | 0.956 / 0.927 | **0.966 / 0.950 (0.94-0.96)** | 0.962 / 0.936 |
| colour 32 px | 0.943 / 0.901 | 0.965 / 0.956 | **0.968 / 0.964** | 0.967 / 0.957 |
| colour 48 px | 0.960 / 0.936 | 0.971 / 0.962 | 0.970 / 0.966 | 0.970 / 0.963 |
| colour 64 px | 0.962 / 0.941 | 0.970 / 0.962 | 0.970 / 0.965 | 0.970 / 0.964 |
| grey 32 / 48 / 64 px (@0.1 %) | 0.906 / 0.940 / 0.945 | 0.955 / 0.960 / 0.962 | 0.964 / 0.966 / 0.967 | |

TAR tops out at ~0.97 for every model, which points to label noise or bad photos in the shard. Above 48 px the
models are within CI.
Visit-like CelebA templates (all low-res photos of an identity): every model reaches ~0.99 TAR@1 %, so aggregation
removes most of the difference. Best-1 is the worst for every model.

Camera, leave-one-visit-out (gallery = owner visits more than 15 min away; single mean template):

| model | genuine median | genuine min | stranger max | margin |
|---|---|---|---|---|
| InspireFace (prod) | 0.593 | 0.195 | 0.118 | 0.077 |
| AdaFace IR-50 | 0.653 | 0.274 | 0.082 | 0.192 |
| AdaFace IR-101 | 0.691 | 0.401 | 0.121 | **0.280** |

Per face (against the owner's other-visit template): IR-101 genuine medians are 0.52-0.58 at every size
(InspireFace 0.36-0.52; InspireFace's >= 96 px faces are often side-on and score lowest). At |yaw| >= 45, the
genuine median is 0.56 (InspireFace 0.30), with stranger p99 0.12 / 0.14.

The 27 v4 desk sightings (#47-#74, unlabelled) score 0.55-0.70 against the owner template with IR-101 and
0.20-0.39 with InspireFace. Every stranger stays at or below 0.12 for both models. Most likely these are the
owner from the side, which the production model could not match. **The user should confirm this on the
labelling sheet; it is not used as a label here.**

## Thresholds (if switched)

AdaFace cosines sit on a different scale from InspireFace's, so `FACE_MATCH_SIM = 0.40` must not be reused
blindly. For IR-101 visit templates, the impostor score at FAR 1 % against single CelebA 48 px photos is 0.14,
and the highest stock-stranger visit score is 0.12. The lowest owner visit score is 0.40. A starting point is
**0.30**, keeping `FACE_MARGIN` 0.08. Recalibrate once real in-domain strangers (deliveries, neighbours) are
labelled: `Calibration.from_impostors(scores).thr_by_far[1e-3]`.

## Cost

Laptop i7-14700HX, ONNX Runtime CPU, batch 16: IR-50 26-44 ms per face, IR-101 46-88 ms per face (the range
depends on load). A visit pool of at most 96 faces adds about 4-8 s CPU with IR-101, once at the end of the visit,
outside the 8 s face budget. ONNX files: 174 MB (IR-50) and 261 MB (IR-101); expect roughly +300-400 MB RSS for
IR-101.
The IR-101 WebFace12M checkpoint holds ~200k denormal weights, which made CPU inference ~5x slower than IR-50.
`export_adaface.py` zeroes them and `AdaFace` enables flush-to-zero; the output differs by < 1e-5.

## Files

- `squirrel_guard/faceid_v2.py`: `FaceEngineV2` (adds `pts5`, `aligned` 112x112 ArcFace crop and `ir` to each
  face; no InspireFace change), `AdaFace` (ONNX), `aggregate` / `best_k` / `multi_templates` / `Gallery`,
  `Calibration` / `fuse`, `side_evidence`, and `FaceSamplerV2` (the result gains a `v2` block:
  `{version, model, vec[512], n_faces, ada_norm_med, ir_frac}`; `vec` and all other fields are unchanged).
- `squirrel_guard/profiler.py`: opt-in only. When `[profiler] face_v2_adaface` is set, the engine is
  `FaceEngineV2` and the sampler is `FaceSamplerV2`. Unset, it behaves exactly as before.
- `scripts/export_adaface.py`: safetensors to ONNX, using our own IR backbone definition (never the repo's
  `trust_remote_code`).
- `scripts/faceid_eval.py`: `visits | celeba | embed | report`. Data lives in `~/.cache/squirrel-guard-test/faceid-v2/`
  (fetched read-only with `reid_gait_eval.py fetch --data ...`); the report is written to `report.md` there.
- `tests/test_faceid_v2.py`: needs no models.

## Deploying (done 2026-11-08: profiler, bridge decide_any, persons-ui enrolment, owner templates re-embedded)

1. Download `minchul/cvlface_adaface_ir101_webface12m/model.safetensors` and run `scripts/export_adaface.py ...
   --arch ir101`. Copy the `.onnx` to `deploy/gpuserver/config/model_cache/adaface/` and mount it read-only like the
   InspireFace pack (never put it in the image or in git).
2. `[profiler] face_v2_adaface = "/models/adaface/adaface_ir101_webface12m.onnx"`, `face_v2_threads = 2`; raise
   `mem_limit` by ~0.5 GB; rebuild the image (needs `faceid_v2.py`; no new pip packages, since onnxruntime is
   already present).
3. **Bridge (recorder host), required before v2 decides anything:** store `faceid.v2.vec` (a new `sightings.face_vec_v2`
   column and `face_templates.vec_v2` / model name), make `facematch.decide()` compare v2 against v2 with its
   own `FACE_MATCH_SIM_V2` (start at 0.30), and keep the InspireFace path as the fallback whenever `v2` is
   missing. Existing enrolments need a v2 template: re-enrol from a new visit, or re-embed the stored enrolment
   crops (detect, align, AdaFace) offline.
4. Run v2 in shadow mode for a few weeks (store and log it, do not decide with it), label strangers on the
   sheet, rerun `faceid_eval.py report` and set the threshold from real impostors.

## Auto-tracking unknown faces (bridge, `FACE_AUTO_TRACK`, default on)

Every usable face that is **not** an enrolled match is grouped into an **unnamed** identity, so repeat visitors
and drivers are recognised over time (the "familiar faces" idea). It never names anyone and never makes anyone
trusted: unnamed identities stay `kind = unknown` with no label, alert exactly as before, and only the user can
label one or enrol a face (persons UI, with consent).

**Order of evidence** (person sightings, `handle_sighting`): enrolled face template (`decide_any`) > Frigate face
name > unnamed face track (`facematch.decide_unknown`) > body re-ID. Rules for re-ID:
- A usable face that matches an unnamed identity files the visit there (`match_kind = face_unknown`); the body
  centroid of that identity is not updated (as for enrolled matches).
- If body re-ID picks an UNNAMED identity that has usable faces of its own and this face clearly disagrees with
  them (best cosine < `FACE_UNKNOWN_SPLIT_V2` 0.20 / InspireFace 0.25), the visit starts a new unnamed identity
  (`face_new`, `possible_id` = the re-ID pick, explain code `FACE_SPLIT`). Between the split value and the match
  threshold the face is ambiguous and body re-ID decides as before.
- Body re-ID picks of named / household / known identities are never overridden by an unknown face, and an
  unusable face (any gate failing) never splits or joins anything.
- Otherwise body re-ID works as before; a visit that founds a new identity carries its face, so that identity
  becomes trackable by face from then on. The end-of-visit template (`apply_late_faceid`) may move a sighting that
  founded its own identity onto an unnamed identity it matches (the empty identity is deleted).

**Gallery:** the last `FACE_UNKNOWN_LAST_N` (8) visit templates per unnamed identity (kind unknown, label NULL)
from person sightings (`face_vec_v2` / `face_vec`) and occupant faces, within the 90-day window, only visits that
pass every gate. Score = best cosine over those visits; AdaFace when both sides have it, InspireFace otherwise.
A user-labelled identity leaves the gallery (tracking never puts a name on anyone).

**Thresholds** (`scripts/face_unknown_calib.py`, 2026-11-08). Visit-vs-visit scores, not visit-vs-enrolled:

| set | genuine | impostor | false merges at the chosen threshold |
|---|---|---|---|
| CelebA visit-like halves, AdaFace IR-101 (315 ids) | median 0.956, min 0.668 | 197,820 pairs: p99.99 0.286, **max 0.323** | 0 at 0.40 |
| Stock stranger clips, AdaFace (16 visits) | 7 pairs, min 0.690 | 113 pairs, max 0.098 | 0 |
| Camera owner visit pairs > 15 min apart vs stock strangers, AdaFace | 289 pairs, median 0.516, p5 0.315 | 416 pairs, max 0.163 | 0; **82 % of owner pairs pass 0.40** |
| CelebA, InspireFace (fallback) | median 0.945 | max 0.421 | 0 at 0.50 |

`FACE_UNKNOWN_SIM_V2 = 0.40`, `FACE_UNKNOWN_SIM = 0.50`, `FACE_MARGIN = 0.08` over the next unnamed identity
(otherwise `FACE_UNKNOWN_AMBIGUOUS` -> new identity). Expect splits rather than merges: a returning visitor whose
single earlier visit was poor starts a second unnamed identity (merge them in the UI). There are still **no
labelled in-domain strangers**; the impostor numbers are CelebA and stock clips. Unlabelled camera identity #17
scores up to 0.865 against the owner mean (median 0.425): it is probably the owner and worth a look in the UI.
Of 55 stored camera visits with an AdaFace template, 28 pass every gate (19 fail liveness, 12 quality, 10 count),
so only those 28 kinds of visit are tracked.

**Storage and retention:** person face templates stay on their sightings as before (deleted with them).
Occupant faces of anyone not household/known are deleted after 90 days row by row (`occupant_retention`), even if
the vehicle is kept; unnamed identities without any sighting, occupant face or enrolled template are removed
hourly; crops of removed rows are swept. Switch off with `FACE_AUTO_TRACK=0` (bridge env): then only enrolled
matches are linked and no occupant vector is stored.

## Vehicle occupants (profiler `vehicle_occupant_faces`, bridge `occupant_faces`)

`squirrel_guard/occupants.py`: for a vehicle visit the profiler fetches full-resolution recording windows (the
first 20 s, plus the last 20 s of a long visit), finds the visit's vehicle (largest box of its class) in pass 1 at
2 fps, then runs face detection on 8 frames spread over the visit: the cabin region (box top - 5 % to 70 % of its
height, 4 % side margin) is cut into 320 px tiles upscaled bilinearly to the 640 px detector input of a second
InspireFace session (all faces per tile, not just the largest). Faces inside the box of a standing person (box
bottom below the cabin) are pedestrians and dropped. Faces of the visit are grouped into occupants (InspireFace
cosine >= 0.30 to a group's mean; horizontal position as a tie-break), each group goes through the production
aggregator (quality weights, outlier rejection) and gets InspireFace + AdaFace templates and liveness. No apparent
expression / age / gender is computed for occupants. Message: `occupants_detected`, `occupant_faces` (up to 3
templates, one crop each), `occupant_stats`.

Bridge (`store_occupants`): each template goes through `decide_any` against the enrolled templates (the gates and
thresholds of people), then, if not an enrolled match and usable, through `decide_unknown`. Stored on
`vehicle_sightings`: `occupants`, `occupant_face` (per-occupant status, sims, reasons; identity only for an enrolled
match or a tracked unnamed identity, label only for an enrolled match; never candidates or per-identity scores),
`driver_identity_id` (best enrolled match, else the first tracked occupant), `driver_sim`, `driver_kind`.
`occupant_faces`: one row per LINKED occupant (vectors only for unnamed tracked identities; one crop). Unusable
faces leave only the count. Occupant-only identities have `origin = 'vehicle_occupant'` and an all-zero body
centroid. Persons UI: "Driver: Robin (enrolled, face match 0.62)", "Driver: Unknown #N (unnamed, ...)",
"Occupant(s) detected (n), not recognised (reasons)"; the identity page lists the vehicles a face was seen in.

**Measured on v4 (2026-11-08, `scripts/occupant_eval.py`):** 18 car events (13:06-14:36, all of them; 3 lacked
some recordings, 2 long events sampled at arrival/departure). The camera sits ~30 cm high under the cars: frames
show bumpers, wheels and the underside, the cabins are mostly above the frame or behind dark, sunlit glass.
**0 of 18 visits produced a usable occupant face**: 11 raw detections in 5 visits, median 9 px (detector noise
below the 24 px floor), 2 >= 32 px; the one usable 60 px "face" was a pedestrian silhouetted next to the car (now
dropped by the standing-person rule). No clip shows the owner (identity 1) driving, so no match outcome on v4
exists; the owner's last face match that day was 12:10, before the first car event. CPU ~6-7 s per visit
(8 frames x ~11 tiles), wall 4-15 s. Positive control (same code on 71 owner person clips with the whole frame as
"cabin"): faces in 60 clips (median 274 px), 33 enrolled matches, the rest weak/liveness/no_match (mostly 1-2
faces in 8 frames: `FACE_TOO_FEW`). Liveness through a windscreen is untested and likely to fail, which blocks
both naming and tracking, by design. Hence `vehicle_occupant_faces = false` in the gpuserver config until a camera sees
windscreens or side windows at face height.

## Licences

| item | licence | use |
|---|---|---|
| AdaFace code (github.com/mk-minchul/AdaFace) | MIT | architecture reference |
| CVLFace code (github.com/mk-minchul/CVLface) | MIT | weight layout reference; its remote-code wrapper is not run |
| AdaFace weights (HF `minchul/cvlface_adaface_ir{50_webface4m,101_webface12m}`) | no weight licence; the model card says "follow the license of the training dataset"; WebFace4M/12M are non-commercial research only | **local only**, never commit or publish |
| InspireFace packs | InsightFace terms, academic / non-commercial | unchanged, local only |
| CelebA (flwrlabs/celeba mirror, one test shard) | "non-commercial research purposes only", no redistribution | local evaluation only, `~/.cache/...` |
| Pixabay stock clips | Pixabay Content License | as before |
| TinyFace | no licence stated on its site | **not used** |
| FASR++ code | no licence | not used (idea only, see faceid.md) |
| GFPGAN / CodeFormer / any SR | | not used, by policy |
