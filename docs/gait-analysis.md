# Gait analysis

`squirrel_guard/gaitlab.py` measures how a person walks from the 2D body keypoints the profiler already tracks, keeps
a walk profile per identity, and compares a new walk with it measurement by measurement. It needs no gait model, so it
has no licence problem (the learned GaitBase signature does: see `docs/reid-gait.md`), and every result can be read
and checked.

## Measurements

| Measurement | What it is | Views |
|---|---|---|
| Cadence | steps per minute | any |
| Step / stride regularity | how alike consecutive steps / strides are (autocorrelation of the hips' up-and-down motion) | any |
| Left/right symmetry | step regularity ÷ stride regularity; a limp lowers it | any |
| Up-and-down motion | hip travel as a share of leg length | any |
| Step time difference | left-to-right against right-to-left step time | side, diagonal |
| Knee range, left and right | knee bend range through the cycle (degrees) | side, diagonal |
| Knee difference | difference between the two knee ranges | side, diagonal |
| Knee bend in swing | the largest bend while the leg swings through | side, diagonal |
| Arm swing, arm difference | wrist travel front-to-back | side, diagonal |
| Step length | in leg lengths; in metres with speed when the camera is calibrated | side, diagonal |
| Trunk lean | forward (+) or back (−) | side, diagonal |

Each walk also stores the knee angle through one averaged walking cycle for each leg (51 points from heel strike to
the next), drawn on the sighting page with the person's usual range as a band.

## Comparing

`compare(walk, profile)` puts each measurement against the person's usual value. The tolerance is the larger of the
spread seen in their own walks (from 3 walks on) and a fixed allowance for day-to-day variation plus keypoint noise
(`TOLERANCE` in the module). Each row is *within the usual range* (under 1.5 tolerances), *somewhat different* (under
3) or *different*. The overall result is **consistent**, **partly consistent** or **not consistent**, or **not
comparable** when the walks share fewer than three measurements (for example a side view against a walk straight at
the camera).

The wording is deliberate: gait can support or weaken other evidence; it does not identify anyone. People walk
differently when carrying something, hurrying, injured, or on ice; two people can walk alike.

## Where it shows

- **Sighting page:** the walk, the comparison table against the identity's other walks, the knee chart.
- **Identity page:** the walk profile (usual value, spread, number of walks) and the usual knee curves.
- **Evidence packages:** the comparison is in `visit.json` and in the report.

## Accuracy (synthetic walkers, `tests/test_gaitlab.py`)

Synthetic side-view walkers with a realistic knee cycle, keypoint noise of 2–5 px and 8–25 % of frames dropped:
cadence within 4 steps/min at 10 and 25 frames/s; knee range within about 10° (the swing peak is underestimated at 10
frames/s because the camera samples between peaks); a limp (one knee bending 45 % less) is flagged in the knee
measurements; a walk at 140 steps/min is *not consistent* with a 110 steps/min profile, a second walk at 112 is
*consistent*. A frontal walk measures rhythm only. **Not yet measured on real footage with known people**: that is the
next step before anyone relies on these numbers, as with every other matching feature here.

## Pipeline

`kinematics.gait()` (run by the profiler on gpuserver for each person visit) now adds these measurements to the stored
`gait` record (version 2). The earlier keys keep their meaning; `usable` now follows the view rules above (the old
value is kept as `usable_v1`). Walks stored before this version are not in profiles until they are measured again.
A copy of the module is in `deploy/recorder/persons-ui/` for the pages (a test keeps the two identical).
