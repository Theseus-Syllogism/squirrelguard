# Face identification (enrolled people only)

`squirrel_guard/faceid.py` (profiler, gpuserver, CPU) builds one **face template per visit**; the recorder host bridge
(`deploy/recorder/bridge/facematch.py`, numpy only) compares it with the **enrolled** templates of household/known
people. Everyone else stays "Unknown #n": a face never names someone who has not been enrolled, and there are no
outside lookups of any kind.

## Privacy rules (the user's decisions)

| Rule | Where it is enforced |
|---|---|
| Only enrolled, consenting household/known people are named | `facematch.decide()` ignores templates of any other kind; the persons UI enrols only household/known identities and needs the consent box; changing an identity away from household/known deletes its templates |
| No ethnicity/race | InspireFace's attribute network (which also predicts race) is **off** (`face_attributes = false`). If switched on for apparent age/gender, `FaceEngine.faces()` overwrites the race field before anything reads it; `FaceObs` has no race field; tests check that no message contains it |
| Apparent age/gender only as estimates | labelled "apparent ... (estimate)" in the message and UI |
| Facial expression = "apparent expression", description only | stored in `face_meta.apparent`, shown on the visit page and in the "Looked" line; never an input to `facematch`, `intel` alerts or the identity decision |
| Liveness before household | a template can only decide the identity when passive RGB liveness passed (>= 60 % of the visit's faces, by weight, score >= 0.5); otherwise `FACE_LIVENESS_FAIL` and the visit stays with re-ID / unknown, so a photo held to the camera cannot silence alerts |
| Retention | `sightings.face_vec`/`face_meta` live on the sighting: unknown identities and their sightings are deleted after 90 days (existing janitor); face crops `snapshots/faces/<event>_face<k>.jpg` are deleted with them, and an hourly sweep deletes crops whose sighting is gone (the persons UI mounts snapshots read-only). Enrolled templates (`face_templates`, crop stored in the row) are deleted with the identity (`ON DELETE CASCADE`), from the identity page, or when the identity stops being household/known |
| Model licence | see below: InspireFace model packs are academic/non-commercial; never in git, images or anything published |

## Pipeline

1. **Frames**: the profiler's end-of-visit analysis decodes the clip at 8 fps (`kinematics.decode`, a generator)
   and follows one person (`kinematics.track`). `FaceSampler.on_frame(frame, pose)` runs inside the existing
   `on_frame` hook, so frames are never buffered. The ~10 s live sighting uses its 6 sampled frames and the
   detector's person boxes (`faceid_of_boxes`).
2. **Head window**: a square around the COCO head keypoints (nose/eyes/ears), or the top of the person box.
   Frames whose head is estimated below 0.6 x `face_min_px` are skipped without running anything.
3. **InspireFace** (`FaceEngine`): face detection inside the window (largest face), 5-point alignment, 512-d
   recognition embedding, face quality, passive RGB liveness, head pose (yaw/pitch/roll), apparent expression.
4. **Budget**: `face_max_attempts` detector calls and `face_max_cpu_s` CPU seconds per event; after misses the
   sampler backs off (every 2nd/4th/8th frame) but always tries a head that has grown by 20 %, because someone
   walking toward the camera shows the best face last.
5. **Aggregation** (`FaceAggregator`, below): bounded pool of the best `face_max_frames` faces -> one template.
6. **Messages**: `"faceid"` in the sighting message (live sample) and in the behaviour message (whole visit):
   `{vec[512], n_faces, n_rejected, faces_seen, quality, quality_max, weight_sum, consistency, size_px,
   max_size_px, yaw_abs_med, liveness{score, pass_frac, frames, ok}, best_crop_b64, crops_b64[<=3],
   faces[{t, size, q, yaw, pitch, live, w, sim, inlier}], apparent{expression, expression_share,
   expression_note[, age, gender]}, model, aggregator, attempts, cpu_s}` or `{vec: null, reason: no_face |
   too_small | low_quality, max_size_px}`.
7. **Bridge**: `handle_sighting` decides **face template > Frigate face name > re-ID**; `handle_behavior` stores
   the whole-visit template when it is better (`facematch.better`) and, if it matches an enrolled identity, moves
   the sighting there (a brand-new unlabelled identity left empty is deleted).

## Aggregation: the FASR / FASR++ idea, our implementation

References (the idea only; the FASR++ code has no licence and was **not** read, copied or downloaded):

- M. dos Santos, R. Laroca, J. C. R. Neves, D. Menotti, *Robust Face Super-Resolution and Recognition Through
  Multi-Feature Aggregation in Diffusion Models*, arXiv:2607.05702; Journal of the Brazilian Computer Society
  (2026), DOI 10.5753/jbcs.2026.5884 (FASR++).
- M. dos Santos et al., FASR, arXiv:2408.15386 (SIBGRAPI 2024).

The papers condition a diffusion super-resolution model on identity features aggregated over several
low-quality images of one subject (e.g. consecutive video frames): each image is encoded separately (512-d
recognition vector), vectors are L2-normalised and averaged (FASR: plain mean, which the authors found better than
max/sum on noisy inputs); FASR++ merges pairs as mean + a small learned correction delta([a; b]) (two linear layers,
triplet loss), L2-normalised, and averages the pair outputs.

**We do no super-resolution and generate no pixels.** We use only the aggregation idea to turn many small, noisy
surveillance faces into one identity template:

- weight per face = quality x size (smoothstep 24..80 px) x pose (cos^2 yaw x cos pitch) x sharpness (Laplacian
  variance at 112 px) x detection confidence;
- pool of the `face_max_frames` (96) best faces (memory-bounded; only the best 3 keep a 160 px JPEG);
- outlier rejection against a running consensus (weighted mean of the best half): drop faces whose cosine is
  below `max(0.20, median - max(3 x 1.4826 x MAD, 0.25))`, twice (a second person, a mis-detection, a poster);
- template = L2N(sum w_i v_i) over the inliers (the FASR mean, quality-weighted);
- `PairwiseCombiner(delta=...)` is the FASR++-shaped hook for a learned aggregator (delta = 0 today). Training one
  needs our own labelled visits (enrolled people, consenting clips); validate it with `scripts/face_eval.py`
  against the weighted mean before switching.

## Measured (laptop, i7-14700HX, Megatron pack, 8 fps, 2026-11-06)

Clips: `real/walk.mp4` (basement, 1920x1080, the user walking toward the camera, face visible ~7.6-11.6 s; split
at 10.3 s: "enrol" = 48 px faces looking down, "test" = 120 px faces from below, 54 deg median yaw, backlit) and six
Pixabay clips (4K downscaled to 2560x1440 like the Wyze v4; each a different person, first/second half = genuine
pair; `road` and `couple` never show a detectable face, `mall` only <=25 px faces). The original openvocab/clips
stock clips contain no detectable face at all. Full logs: `scripts/face_eval.py ... --json`.

Visit-template cosine (aggregated; `--no-budget`, every frame at 8 fps):

| pair | aggregated (ours) | plain mean, all faces | best single frame | mean of per-frame cosines |
|---|---|---|---|---|
| walk enrol -> walk test (genuine, hard) | **0.460** | 0.453 | 0.314 | 0.339 |
| suit 1st -> 2nd half | **0.928** | 0.921 | 0.906 | 0.870 |
| model 1st -> 2nd half | **0.887** | 0.885 | 0.567 | 0.643 |
| hiker (one 24-41 px face each side, fails the gates) | 0.417 | 0.417 | 0.179 | 0.298 |
| impostors, all 64 pairs | mean 0.046, max 0.294 | max 0.334 | max 0.295 | per-frame max 0.496 |
| impostors between templates that pass the gates (24 pairs) | **max 0.061** | | | |
| genuine pairs that pass the gates (6) | **min 0.460** | | | |

- Aggregation matters most where single frames are poor: walk 0.31 (best frame) -> 0.46, model 0.57 -> 0.89.
- Ablation: a pool of only 24 faces lost pose variety (model pair 0.73 vs 0.89 with all faces), so the pool is now
  96. Quality weighting vs the plain FASR mean: +0.007 on average here (single-person clips; it matters with mixed
  quality, see tests). Outlier rejection did not change these single-person pairs; it removed 0-9 faces per segment.
- Per-face statistics by face size (all 2,744 impostor / 328 genuine face-vs-template comparisons):

| face px | n | quality | liveness | genuine cos | impostor cos |
|---|---|---|---|---|---|
| < 24 | 432 | 0.03 | 0.05 | - | 0.07 |
| 24-32 | 32 | 0.03 | 0.27 | 0.42 | 0.08 |
| 32-48 | 304 | 0.28 | 0.93 | 0.59 | -0.01 |
| 48-64 | 344 | 0.41 | 0.89 | 0.54 | 0.01 |
| 64-96 | 464 | 0.50 | 0.90 | 0.68 | -0.02 |
| >= 96 | 1496 | 0.56 | 0.78 | 0.72 | 0.01 |

  Faces under ~32 px are useless for identity AND fail liveness (so they can never mark anyone household).
- Liveness: every real-person segment with >= 32 px faces passed (scores 0.78-0.98); the `mall` faces (<= 25 px)
  failed (0.04). **No spoof (photo/screen) was tested**: see open issues.
- With the production budget (backoff, 150 attempts, 8 CPU s): walk kept all 19 + 9 faces (41 + 5 frames skipped
  by backoff), the whole walk clip 28 faces in 7.2 CPU s; the 4K stock segments hit the 8 s cap after ~40 faces
  (50 ms each) and still gave genuine 0.87 / 0.81; gated impostor max 0.067.
- Time (laptop i7-14700HX, Megatron, 320 px detector): ~11 ms per head window without a face, 45-60 ms with a face.
  Pikachu pack: 6 / 14-18 ms, but weaker (walk genuine 0.375 vs 0.46; impostor max 0.23).
- Memory: Megatron engine 0.6 GB after load, ~1.0 GB after the first inference (Pikachu ~0.3 / 0.7 GB); the
  evaluation process (pose + decode + Megatron) peaked at 1.45-1.5 GB.

## Thresholds (bridge environment)

| Variable | Default | Meaning |
|---|---|---|
| `FACE_MATCH_SIM` | 0.40 | cosine of the visit template to the best enrolled template of an identity |
| `FACE_MARGIN` | 0.08 | best identity must beat the next enrolled identity by this much |
| `FACE_MIN_FACES` | 3 | inlier faces in the visit template |
| `FACE_MIN_QUALITY` | 0.10 | weighted InspireFace quality |
| `FACE_MIN_CONSISTENCY` | 0.35 | mean inlier cosine to the template (low = mixed faces) |
| `FACE_MIN_PX` | 32 | median face size |

`FACE_MIN_FACES` and `FACE_MIN_QUALITY` are read by the persons UI too (enrolment uses the same gates).
Re-calibrate with the Wyze v4 clips (below) before relying on it.

## Reason codes (explain.py)

`FACE_TEMPLATE_MATCH` (with the cosine), `FACE_NO_MATCH` (a face, not an enrolled one), `FACE_LIVENESS_FAIL`,
`FACE_TOO_SMALL`, `FACE_TOO_FEW`, `FACE_LOW_QUALITY`, `FACE_INCONSISTENT`. Scores: `face_template_cos`,
`face_faces`. Match kind `face_template` (headline "Recognised X by enrolled face"). The existing `FACE_MATCH` /
`FACE_SEEN` (Frigate's own face library) are unchanged.

## How to enroll someone

1. They agree to be recognised by face. Record a visit where they walk toward the camera (good light, face
   uncovered); the Wyze v4 at 2560x1440 gives 60-130 px faces at a few metres.
2. In the persons UI mark their identity **household** or **known** (and name it).
3. Open that visit's **Why?** page. The *Face* section shows the crops, face count, quality and liveness. If the
   gates pass, choose the identity in *Use this visit's face for ...*, tick the consent box and press **Enroll**.
   Two or three visits (different days/light, with and without glasses) make matching more robust; each is one
   template and the best one counts.
4. Enrolled templates are listed on the identity page with **Delete**. Setting the identity to unknown/watch
   deletes all of them.

## Configuration (`[profiler]`, deploy/gpuserver/squirrel-guard/config.toml, copied into the image at build)

`face_enabled` (false), `face_model_pack` (`/models/inspireface/Megatron`), `face_detect_level` (320),
`face_min_px` (24), `face_min_quality` (0.0), `face_max_frames` (96), `face_every` (1), `face_max_attempts` (150),
`face_max_cpu_s` (8), `face_expression` (true), `face_attributes` (false).

To enable on gpuserver (not done by this change): copy the pack (from `~/.inspireface/ms/models/tunmxy--InspireFace/
snapshots/master/Megatron` after `python -c "import inspireface as i; i.launch('Megatron')"`) to
`deploy/gpuserver/config/model_cache/inspireface/Megatron`, uncomment the volume in `compose.yaml`, raise the
profiler's `mem_limit` (Megatron needs ~1 GB more, see the numbers), set `face_enabled = true`, rebuild the image
(the Dockerfile installs `inspireface==1.2.4.post3`, SDK only) and apply `persons_schema.sql` on the recorder host.

## Licences

- **InspireFace SDK** (HyperInspire/InspireFace, pip `inspireface`): its repository does not state a code licence
  on its front page; check before redistributing the SDK itself.
- **InspireFace model packs** (Pikachu, Megatron, ...): per the InspireFace README the models follow InsightFace's
  terms: **academic/non-commercial use only**. Keep them out of git (`.gitignore` covers `inspireface/`,
  `Megatron`, `Pikachu`), out of container images (mounted read-only at run time) and out of anything published.
- **FASR/FASR++**: papers cited above; the authors' code has no licence and is not used.
- Test clips for the evaluation: the user's own camera, and Pixabay stock clips (Pixabay Content License) under
  `~/.cache/squirrel-guard-test/faces/` (not in git).

## Evaluation harness

```
.venv/bin/python scripts/face_eval.py --pose models/yolov8s-pose.onnx \
  --seg me=real/walk.mp4@0-10.3 --seg me=real/walk.mp4@10.3-99 \
  --seg suit=faces/suit_199623.mp4@0-6 --seg suit=faces/suit_199623.mp4@6-99 ... [--no-budget] [--json out.json]
```

Same label = same person (genuine pairs), different labels = impostor pairs. Reports per segment (faces, rejected,
quality, liveness, consistency, size, CPU), per pair the aggregated template cosine next to the unweighted mean,
the single best frame and per-frame cosines, the ablation variants (pool size, weighting, rejection), quality /
liveness / similarity by face size, and peak RSS. With the Wyze v4 consenting clips: one `--seg name=clip` per
recording, several per person, then set `FACE_MATCH_SIM` between the lowest genuine and the highest impostor
aggregated cosine.

## Open issues

- **Calibration data is thin**: one real genuine pair from our camera (the user, 0.46) and 5 stock people. Record
  the consenting Wyze v4 walk-toward-camera clips (several per person, day/night, glasses/hat) and re-run
  `face_eval.py`; adjust `FACE_MATCH_SIM` (0.40 now: gated genuine min 0.46, gated impostor max 0.07, ungated
  impostor max 0.29).
- **Liveness is not tested against spoofs.** Passive RGB liveness through a surveillance camera (re-encoded
  H.264, small faces) is weaker than on a phone. Test by holding a printed photo and a phone screen with an
  enrolled face to the camera; if any passes, raise `live_thr`/`live_frac` or require `FACE_MIN_FACES` frames
  with motion. Night IR frames are untested (InspireFace's RGB liveness on IR-greyscale may fail genuine faces).
- **Memory**: Megatron adds ~1 GB; the profiler's `mem_limit: 2g` already holds pose, re-ID, DINOv2, gait,
  actions and YOLO-World. Raise it to 3g when enabling (or use Pikachu).
- **Re-ID centroid contamination**: when the late face template moves a sighting that re-ID had matched to another
  identity, that identity's centroid keeps the moved embedding (minor; merge/re-review fixes it).
- **Spoof + clothing**: a liveness failure only stops the FACE from deciding; re-ID may still match the visit to a
  household identity by clothing (existing behaviour).
- **Outlier rejection** removed up to 9 of 41 genuine faces on one stock clip with strong pose changes; harmless
  there (template 0.81), but watch `n_rejected` on real visits.
- `face_attributes` (apparent age/gender from InspireFace) is off: InspireFace computes it in the same network as
  race. The wrapper discards race if it is switched on, but the user should decide whether running that network at
  all is acceptable. The existing Qwen description already gives apparent age/gender.
- The learned FASR++-style aggregator (`PairwiseCombiner` delta) is only an interface; training needs our own
  enrolled visits.
- InspireFace SDK licence unclear from its front page; the models are academic/non-commercial.
