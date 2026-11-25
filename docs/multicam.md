# Multi-camera rig: joint calibration, triangulated replays, splat, 3D objects (2026-11-06)

Three fixed cameras (v4, frontyard = Wyze v3, testcam = laptop webcam) in one metric world frame, from ONE
simultaneous frame each, then a replay where two of them watched the same walk.

| step | command | output |
|---|---|---|
| rig | `~/src/splat/.venv/bin/python scripts/multicam_rig.py WORK --raw FRAMES --cams v4=2560x1440 frontyard=1920x1080 testcam=1280x720 --lens frontyard=-0.3 v4=-0.3 testcam=-0.2 --vggt-python ~/src/splat/.venv/bin/python --splat-work WORK/splat` | `WORK/calib/<cam>.json` (geometry.Camera), `rig.json` (+ world_from_vggt) |
| splat | `splat/make_splat.py train WORK/splat --iters 3000 --sh-degree 1`, then `splat/ply_to_splat.py WORK/splat/splat.ply OUT.splat --bbox=...` | `.splat` in the world frame |
| replay | `.venv/bin/python scripts/multicam_demo.py --cam v4=V4.mp4,WORK/calib/v4.json --cam frontyard=FY.mp4,WORK/calib/frontyard.json --event v4=<frigate id> --splat OUT.splat` | `demo/<name>.json` (+ cameras_extra, multicam stats, event) |
| objects | `~/src/pixarmesh/.venv/bin/python scripts/rig_objects.py WORK --out demo --name multicam --vlm-url http://203.0.113.20:5920 --replay demo/multicam.json` | `demo/<name>_objects/*.glb` + `<name>_objects.json` |
| body | `~/src/wham/.venv/bin/python scripts/body3d_wham.py V4.mp4 --replay demo/multicam.json --tracker simple`, then `scripts/body3d_anchor.py demo/multicam_body3d.json demo/multicam.json` | `demo/multicam_body3d.json` (SMPL-derived: git-ignored, recorder host only) |

## What was learned

- **The cameras must not move between the rig frames and the walk.** The 17:52 rig was stale by 18:02 (frontyard
  turned ~25 deg, a coat swinging in front of both lenses): triangulated bodies came out 0.6 m tall at 0.6 m range.
  Re-solving from the walk clip's own first frame fixed it. Take rig frames right before a capture, room empty.
- **Lens.** VGGT fits a pinhole; the Wyze lenses are barrel-distorted. The rig undistorts with a one-parameter
  division model chosen by the plumb-line rule (joists straight) and stores the lens as OpenCV k1/k2/k3.
- **Baseline.** frontyard and v4 sit ~25 cm apart: joint rays meet to 1.4 cm, but depth along the viewing direction
  is the weak axis (shoulder width came out 0.60 m). Wider spacing (testcam is ~1 m away at 23 deg) is better; the
  testcam is not recorded yet, so it only contributes to the rig and the splat.
- **Indoors the feet are hidden** (desk, boxes), so single-camera foot points are useless: frames seen by one
  camera use the head plane at the *measured* stature (triangulated: 1.70 m) instead.
- **Time sync**: Frigate clips start on keyframes; the offset is found by minimising the joint ray gap (here
  -1.80 s, consistent with the clip-length difference of 1.73 s).
- **WHAM** single-camera placement was ~8 m off in this room; its pose/shape are kept and each frame is translated
  onto the triangulated path (body3d_anchor.py).
- **Objects**: Grounding DINO labels run together ("tree lamp", "cat tree coat"); the gpuserver vision model (now the Qwen3.8 27B) names
  each crop instead (and flags people, which are never modelled). Meshes are the depth grid of each view (visible
  surfaces only), merged across views by label + voxel overlap; floating parts (bottom > 1.3 m) are dropped.

## Assumptions to replace

Metric scale = floor-to-ceiling 2.30 m and desk 0.75 m (both ASSUMED). One tape measurement (e.g. the distance
between frontyard and v4, or the desk height) fixes every number in the replay, splat and objects.

## Splat spikes (needles)

`make_splat.py train --aniso-weight 0.1 --max-ratio 4 --max-scale 0.3` penalises Gaussians whose longest axis is
more than 4x their MIDDLE axis (needles; flat discs on walls stay allowed) and caps every axis at 0.3 m;
`ply_to_splat.py --max-ratio 6` prunes any needles left in an already trained model. On the 3-view basement splat
visible needles over 6:1 went from 29 % to 0 % at the same Gaussian count. What remains when orbiting far from the
cameras are smears of surfaces no camera saw from that side: only more viewpoints fix those (record the testcam, or
a slow phone walk-around of the empty room, added to the rig's frames).

## Vision model

Camera vision (Frigate descriptions, squirrel verifier, attributes, object naming) runs on the gpuserver Qwen3.8 27B
with its vision projector (`deploy/gpuserver/llm/llama-qwen27b-zzz-vision.conf`, live in
/etc/systemd/system/llama-qwen27b.service.d/). The drop-in must sort AFTER zz-vram.conf: both reset ExecStart and
the last one wins ("zz-vision" sorted before "zz-vram" and was silently ignored). llama-vlm (Qwen2.5-VL 7B, port
5925) is disabled, which freed the ~6 GB the projector needs; `systemctl enable --now llama-vlm` + reverting the
5920 URLs restores it. The 27B has one slot shared with opencode, so camera requests can queue (verifier timeout
60 s); our calls send `enable_thinking: false`.
