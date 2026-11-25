# Small faces: face sequences, combined recognition, viewing composites (2026-11-10)

Modules: `faceseq.py` (front end), `faceseq_agg.py` (#1 recognition), `caface.py` (#1 learned option),
`ofiq.py` (quality evidence), `lucky.py` (#2 viewing composite). Evaluations: `scripts/faceseq_eval.py`,
`scripts/lucky_eval.py`. Not wired into the live profiler yet.

## Front end
Pose keypoints (YOLOv8s-pose, COCO) locate the head even when the face is ~8 px; the head window is upscaled so the
face is ~100 px for YuNet (OpenCV Zoo, MIT); landmarks are mapped back and the 112x112 ArcFace crop is warped from
CAMERA pixels (bilinear interpolation only). Face size = YuNet box in camera pixels (eye spacing collapses on
profile faces). It finds faces down to 5-8 px.

## #1 Recognition: combine fingerprints, not pixels
Test: the 12:35 v4 visitor, distance simulated by shrinking the frames (antialiased); reference = 4 best full-res
frames, held out. Strangers: 5 Pixabay clips (`~/.cache/squirrel-guard-test/faces/`), same scales.

| face px | one frame (by quality) | plain mean | **AdaFace-norm weighted** | CAFace (own 4M space) |
|---|---|---|---|---|
| 30-33 | 0.75 | 0.83 | **0.89** | 0.75 |
| 22-23 | 0.75 | 0.80 | **0.87** | 0.74 |
| 16-17 | 0.70 | 0.66 | **0.76** | 0.65 |
| 11-12 | 0.02 | 0.34 | 0.03 | 0.23 |
| 8 | refused (< 12 px) | | | |

Strangers (all sizes): highest single-frame score 0.25, weighted 0.19, CAFace 0.15; decision rule named nobody.
=> Weighted averaging beats any single frame by 0.06-0.14 at 16-33 px and LOWERS stranger scores. CAFace separates
about as well but is not better here, needs its own backbone + re-enrolment: kept as an option, not default.
Recognition breaks down between 12 and 16 px; the 12 px floor + "2 agreeing frames" rule held.
Caveats: one genuine person, 5 stock strangers, simulated distance (real distance adds lens blur + compression).
The walk test (owner walks away from v4; distance from the site ground model) replaces this.

OFIQ (built in ~/src/ofiq, 787/787 conformance): unified score ~0-2 on all camera faces; on aligned crops its
detector found 24/44 faces and its component quality did not correlate with the AdaFace norm (r = -0.05).
Used for explanations only, not weights.

## #2 Viewing composite (labelled ENHANCED, never used for recognition)
Tier A lucky imaging: best frames of similar pose, landmark + ECC sub-pixel alignment, per-pixel outlier rejection
(median/MAD), quality-weighted stack, drizzle (STScI, BSD-3) onto a 2x grid, uncertainty map.
Against the TRUE full-resolution face (face region):

| face px | best single frame, bicubic | lucky stack |
|---|---|---|
| 34 | 22.1 dB | 23.3 dB |
| 22 | 20.1 dB | **23.5 dB** |
| 17 | 21.3 dB | 22.1 dB |
| 15 | 23.2 dB | 17.1 dB (alignment failed) |

Drizzle adds nothing measurable over the aligned stack at these sizes. Failure detection: face-region
uncertainty > 13 or consistency < 19 dB -> `suspect`, show the single frame (provisional, from these 4 points).

Tier B Burstormer (~/src/burstormer, CLI by a helper agent): code MIT, weights unlicensed + non-commercial training
data -> local only. Beats bicubic only on rigid motion (+1.4-1.7 dB synthetic, +1.5-2 dB static scene with jitter);
on a moving car it was WORSE (20.1 vs 24.5 dB). Not used by default; its `suspect` flag falls back to bicubic.
Tier C BasicVSR++ (`vsr.py`) for moving close-up clips.
