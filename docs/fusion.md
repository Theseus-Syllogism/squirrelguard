# Live multi-camera fusion + rig frame rate (2026-11-08)

One rig over every camera that saw a visit, for LIVE Frigate events (until now only the offline demo builders fused
cameras; the profiler fitted each camera's event alone).

| piece | file |
|---|---|
| sync, calibration check, fused fit, replay, live planning | `squirrel_guard/fusion.py` |
| profiler hook (own thread, 2-deep queue) | `squirrel_guard/profiler.py` (`fusion_enabled`, `fusion_*`) |
| bridge: store + link from every member sighting | `deploy/recorder/bridge/app.py` `handle_fusion`, column `sightings.fused_trajectory_file` |
| UI | persons-ui sighting page "3D replay, all cameras"; replay pane "Open the all-camera replay" |
| synthetic tests | `tests/test_fusion.py` |
| real-clip measurement (fused vs single vs held-out, fps sweep, optional fused replay) | `scripts/fusion_eval.py` |
| frame rate vs gait on a walker with known cadence | `scripts/rig_fps_eval.py` |

## How it works

1. **Which events.** After an event's own analysis (sighting + behaviour published as before) the profiler asks Frigate
   for person events on the cameras whose calibration shares the event camera's **frame**: an explicit `"frame"` in
   the calibration JSON (`scripts/multicam_rig.py` now writes one per joint solve), else method + date for older
   `multicam-vggt` files; single-camera calibrations never fuse. Events overlapping it (1 s tolerance, transitively,
   several events per camera allowed) are the members. If one of them is still running the fusion waits: the LAST
   member to end triggers it, earlier ones see it in progress. Members are remembered, so a visit fuses once.
2. **Window.** Union of the members' intervals padded by 3 s (Frigate keeps 3-5 s pre-capture, and an event starts
   only after the detector is sure), capped at `fusion_max_s` (20 s) around the part the cameras saw together.
   Per camera, `/api/<cam>/recordings` gives the recorded range inside the window (largest run without holes > 1 s:
   a hole would silently shift every later frame of the concatenated clip); that range is fetched as
   `/api/<cam>/start/../end/../clip.mp4`.
3. **One time grid.** Every clip is sampled at `fusion_fps` on the SAME instants (t = window start + k / fps,
   nearest frame by container timestamp). The old demo decoder gave 8.33 fps for the 25 fps v4 and 7.5 fps for the
   15 fps testcam, so the 02:46 demo compared frames up to a second apart; `scripts/multicam_demo.py` now uses
   `fusion.grid_frames` too.
4. **Time offsets + calibration check.** Joint rays (body 17 + feet 6, confidence >= 0.5) of every camera pair are
   compared over shifts of +-3 s; the shift with the smallest median ray-to-ray gap wins, refined to a fraction of a
   frame by a parabola. The **anchor** is the pair that agrees best (gap <= `fusion_max_gap_m` = 15 cm). Every
   further camera must reproject onto the anchors' TRIANGULATED joints (<= 0.06 box heights) at its best shift.
   Why not just ray gaps: on the synthetic walker a camera turned by 8 deg still pairs with one other camera at a wrong
   time shift (-1.6 s) with an 11 cm gap, because the shift slides the person along the walk; against a fixed 3D
   triangulation it reprojects 0.13 box heights off and is dropped. A minimum at the edge of the searched shifts is
   rejected (no real match). A camera with too few confident frames is dropped (saw nothing useful).
5. **Fit.** `rig.fit` over every remaining view on the primary's time base (primary = the camera with the most
   confident torso/head joints; its cut-outs and background make the replay). If a view's median error stays above
   0.06 box heights the cameras disagree: with 3+ views the worst is dropped and the fit repeated, with 2 the
   fusion fails. Well-fitted views are 0.01-0.04.
6. **Replay.** `fusion.build_replay`: path from the rig (`trajectory.estimate(ground_override=rig_ground)`), stick
   figure + gait + angles from the rig, `cameras_extra` (frustums in the viewer), `fusion` (members, shifts, gaps,
   per-camera report incl. why a camera was dropped, window), `multicam` (as in the demos). Body (SMPL) and assets
   as for a per-camera replay, prefix `<primary event>_fused`. Published on `squirrelguard/<primary>/fusion`; the
   bridge stores `<primary event>_fused_traj.json` (+ assets) and sets `fused_trajectory_file` on EVERY member's
   sighting. Each sighting keeps its own per-camera replay (fallback, and what a failed fusion leaves).

## Measured

Laptop, RTMW l-m + YOLOv8s-pose (the profiler's models), keypoints once at 25 fps on the common grid and subsampled
(`scripts/fusion_eval.py`, cached keypoints in `~/.cache/squirrel-guard-test/multicam/fusion_eval/`). Error = median
reprojection error over confident joints in **box heights** (pixels in brackets). "Held-out" = a camera the fit never
saw. Single-camera fits start, like the profiler, from that camera's own replay path.

**02:41 (2026-11-07), live Frigate clips of two overlapping events, v4 + frontyard, 11-07 calibration** (the person
walks in and sits down; cameras ~0.8 m apart):

| rate | fused: v4 / frontyard error | v4 alone: train / held-out frontyard | frontyard alone: train / held-out v4 | stature fused / v4 / fy |
|---|---|---|---|---|
| 8 fps | 0.027 (28 px) / 0.039 (26 px) | 0.013 / **0.21** (152 px) | 0.014 / **0.18** (200 px) | 1.13 / 1.53 / 1.63 |
| 12 fps | 0.028 (28 px) / 0.034 (24 px) | 0.013 / **0.25** (185 px) | 0.015 / **0.14** (166 px) | 1.29 / 1.34 / 1.68 |
| 25 fps | 0.035 (36 px) / 0.040 (28 px) | 0.016 / **0.27** (212 px) | 0.016 / **0.14** (174 px) | 1.38 / 1.42 / 1.65 |

One camera fits its own keypoints well (0.013-0.016) and is 5-10x off in the other view: depth along its own rays is
free. The fused rig explains both views at 0.03-0.04. Time offset found: frontyard -2.05 s (8 fps) / -2.14 s (12 fps)
against v4, ray gap 2.5-2.6 cm. Gait: not available (sitting down, no 3 alternating steps) in any variant.
Fused statures are low because the person sits for most of the window (the rig's scale fitted to a folded body).

**18:03 (2026-11-06) basement walk, v4 + frontyard, rig0 calibration** (cameras 25 cm apart: nearly one viewpoint):
offset 1.85 s, ray gap 1.7 cm. Fused error frontyard 0.015 (10-11 px), v4 0.012-0.013 (14 px) at every rate; alone
each camera trains to 0.007-0.011 and is 0.029-0.033 in the other (held-out). Statures fused 1.71-1.74 m, alone
1.71-1.74 m. With a 25 cm baseline fusion adds little geometry; it adds coverage (frames seen by either camera).

**02:46 (2026-11-07) three cameras (recorder host continuous segments), seated at the desk:** frontyard saw nobody (dropped:
"person not seen well enough"); testcam + v4 synchronise consistently (1.14-1.15 s at 12 and 25 fps, ray gap 6.4 cm)
but the rig cannot explain both views (testcam 0.096, v4 0.061-0.068 box heights), so the fusion is **refused** and
the per-camera replays stay. Forced through (`--max-err 1`): stature 1.00 m. Alone: testcam held-out in v4 0.35 box
heights; v4 alone puts the rig behind the testcam. The old demo replay of this scene (stature 0.94 m, 37/46 px) was
built on the fps mismatch described above.

**11-08 03:17, live Frigate overlap, v4 + frontyard, after both cameras were re-aimed** (11-07 calibration stale):
both cameras see the person in the same seconds, but no joint ray pair ever meets in front of both cameras -> both
dropped, "camera moved since the calibration?", no fused replay (the live code path, `LiveFusion.run`).

**Synthetic (tests/test_fusion.py):** offsets recovered to < 0.35 frame (integer) and 2.5-frame fractional shifts to
< 0.4 frame; a camera turned 8 deg is dropped by the 3-camera reprojection check (0.069-0.13 box heights at its best
shift vs 0.01 for good cameras); a 35 deg turn with two cameras fails the fusion; an empty camera is dropped and the
two-camera fit recovers stature within 5 %.

### Rig frame rate vs gait

Synthetic walker with KNOWN gait (`scripts/rig_fps_eval.py`: 1.1 s stride = 109.1 steps/min, 8 s, keypoint noise 1 %
or 2 % of the box height, 10 % joints missing, 2 seeds), one camera:

| rig fps | cadence error (1 % / 2 % noise) | step-time CV (truth 0) | fit CPU s, 1 view (8 s) |
|---|---|---|---|
| 8 | 10.9 / 10.9 spm | 0.11 / 0.11-0.14 | 20 |
| 12 | 6.2 / 6.2 spm | 0.12 / 0.07-0.15 | 21 |
| 16 | 2.4 / 2.4 spm | 0.06-0.08 / 0.08-0.09 | 32 |
| 20 | 0.0 / 0.0 spm | 0.04-0.08 / 0.07-0.11 | 60 |
| 25 | 2.0 / 4.2 spm | 0.07 / 0.07-0.11 | 70 |

(CPU seconds measured while another job loaded the machine; relative values only.) Every rate found >= 8 alternating
steps; the error is **quantisation**: heel strikes are whole frames, so the median step time snaps to multiples of
1/fps (0.55 s at 8 fps -> 0.5 or 0.625 s = 120 or 96 spm). Two cameras: same cadence errors, 14 steps found instead
of 11-13. Real 18:03 walk (mostly shuffling near the desk): cadence was reported at every rate except v4 alone at
8-12 fps, but jumps between 92 and 125 spm with step-time CV 0.17-0.40, i.e. not a usable walk at any rate; the
lowest CV was fused at 20 fps (0.17). A real straight walk past the cameras is still needed to validate gait.

## Cost and where it runs

Fusion runs **in the profiler on gpuserver**, in its own thread with a 2-deep queue, after the members' own analyses
(their sightings/behaviour are never delayed). Nothing new runs on the recorder host except storing the message. Clips come
from Frigate's API on the same host; the window is capped at `fusion_max_s` = 20 s and `fusion_max_cameras` = 3.

| step | measured (laptop, unloaded) |
|---|---|
| end-to-end, live code path, real 02:41 overlap: 10.4 s window, 2 cameras, 12 fps | **44 s wall, 198 s CPU** (ONNX spin-wait included), peak RSS 1.09 GB for the whole process incl. all models |
| of which keypoints (YOLO pose + RTMW l-m, 250 grid frames) | 33 s wall (134 ms/frame) |
| rig fit, 2 views, torch 2 threads | 8 s (8-12 fps, 10-20 s), 11-14 s (25 fps, 20-31 s) |
| MQTT message (replay + assets, no SMPL body) | 0.40 MB; the SMPL body adds ~1-2 MB as for per-camera replays |

Keypoint cost is linear in cameras x seconds x fps: the worst case (3 cameras x 20 s x 16 fps = 960 frames) is
~2.2 min wall on the laptop plus a ~15 s fit; gpuserver's CPU is slower per core (budget about 2x). Fusions happen only
for overlapping events of calibrated cameras, so the steady-state load is small; a burst beyond 2 queued fusions is
dropped (logged, per-camera replays stay). Not done (would cut the cost by most of the keypoint time): reuse the
per-camera rig's keypoints for the window instead of recomputing them (they are on a different, shorter clip and
rate today).

**Recommended settings** (`deploy/gpuserver/squirrel-guard/config.toml`): `fusion_fps = 16` (replay every 2nd frame =
8 fps; cadence quantisation 2.4 spm vs 6.2 at 12; 33 % more keypoint time than 12). Per-camera `rig_fps`: keep 12 for
now. Note that `kinematics.decode` picks every round(src/fps)-th frame, so on the 25 fps cameras only 12.5 or 25 are
reachable (16 or 20 give 12.5); 25 doubles the per-event keypoint time (~2 CPU-min per 20 s event). The cheaper fix
for gait at 12 fps is sub-frame heel strikes in `rig.gait_metrics` (parabolic interpolation of the heel-height
minimum), which would remove most of the 6 spm quantisation without more keypoints.

## Limits

- **Two cameras can hide a small move.** With only two cameras the ray gap and the rig error are the only checks; a
  few degrees of turn while the person walks can pair at a wrong time shift (synthetic 8 deg: 3-11 cm gap, under the
  15 cm limit). Three cameras catch it (step 4). A gross move (11-08: v4 and frontyard re-aimed from the ceiling to
  level) fails the fusion and logs `FUSION none ... camera moved since the calibration?`.
- **One person.** Each camera picks its most plausible person (`rigpipe.single_person`). Two people in view can
  be paired across cameras wrongly; that shows up as large gaps and the camera is dropped, not fused.
- **A camera without its own event cannot join**: Frigate keeps recordings only around events
  (`record.continuous.days: 0`). With continuous retention (`continuous: {days: 1}`) the planner could also pull
  cameras without an event (not enabled: storage, and it would fuse every camera on every event).
- **Retention:** the fused files are named after the primary event, so the retention janitor removes them with that
  sighting; other members then keep a dead "all cameras" link (their own replay stays).
- **Stature while sitting/crouching** is the rig's scale parameter fitted to a folded body and reads low (02:41:
  fused 1.13-1.38 m for a person who sat down). Single-camera statures (1.34-1.68 m) are held near the prior, and
  their held-out errors (0.14-0.27 box heights) show the geometry behind them was wrong.
- **Gait** stays unvalidated on real data (no straight walk recorded under a valid calibration); see the frame-rate
  section for the quantisation limit and the sub-frame fix that belongs in `rig.gait_metrics`.
