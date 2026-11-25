# Per-clip room scenes (2026-11-08)

Every replay on the recorder host (per-camera `<event>_traj.json` and fused `<event>_fused_traj.json`) gets a Gaussian splat
and 3D objects of the room **as it was during that visit**, with no manual step. The laptop builds them
(`scripts/scene_worker.py`): the splat needs the RTX 4070 (the recorder host has no GPU, gpuserver is AMD).

## Pipeline (per replay)

| step | what | where |
|---|---|---|
| discover | `*_traj.json` written within `--lookback-h` (48 h) without `replay/scenes/clip/<stem>.json` | `ssh db-host` |
| frame | `fusion.frame` / `calibration.frame` / `calibration_frame`; untagged replays: the installed calibration whose camera centre matches the replay's camera (< 5 mm). Installed calibrations are read from gpuserver `/mnt/media-extra/frigate/config/calib` and kept per frame in `~/.cache/scene-worker/calib/` | |
| cache | a scene of the same frame built for a visit within `--cache-h` (6 h) is reused (manifest only) if each camera's view still matches the build's frame (edge-map correlation >= `--view-ncc` 0.6; same view scores >= 0.8 even with a person in it, a turned camera ~0.2) | |
| frames | every calibrated camera's Frigate recording of the visit (+-8 s, <= 60 s; `docker exec frigate curl .../clip.mp4` on gpuserver); a camera with no recording (Frigate keeps motion/object segments only) uses its live frame if the visit ended < 30 min ago. `scripts/scene_frames.py`: a person-free sample, else the least covered sample with its person boxes (Grounding DINO) filled from samples showing the same view | GPU |
| solve | `scripts/multicam_rig.py --align-to <installed calib dir> [--align-ref <its VGGT run>]`: VGGT on those frames, then a similarity onto the installed cameras: rotation = chordal mean of the camera rotations; scale = median ratio of this run's depth maps to the calibration's own VGGT depth maps (`ref/`, when registered), else least squares on the camera centres; then translation. Per-camera residuals and all scale estimates in the manifest | GPU |
| moved? | residual > `--max-centre-m` 0.10 m or `--max-rot-deg` 3 deg: status `camera_moved`, no scene attached, one ntfy push per frame per 6 h | |
| splat | `make_splat.py train --iters 3000 --sh-degree 1 --aniso-weight 0.1 --max-ratio 4 --max-scale 0.3`, `ply_to_splat.py --max-ratio 6 --max-scale 0.3 --bbox` (camera centres +-6 m, Z -0.3..3) | GPU |
| objects | `rig_objects.py` (Grounding DINO + SAM, VLM naming on gpuserver :5920); a failure publishes the splat alone | GPU |
| publish | `publish_scene.py --name b-<time>-<hash> --url-base /replay/scenes/clip/` -> rsync to `db-host:/srv/camstack/replay/scenes/clip/`, then the manifest `clip/<stem>.json` | |

Every GPU step first waits (<= 5 min) for `--min-free-mib` 2500 MiB free on the card (the llama services share it);
otherwise the cycle ends and the queue is retried 2 min later. Nothing is ever stopped to make room.

### Scale reference (once per new calibration)

VGGT's unit differs per run, so the clip solve's metric scale must come from somewhere. The camera centres fix it
well only over a long baseline (v4-frontyard are 0.33 m apart); the ceiling/desk planes were off by up to 20 % on
clip frames. Register the calibration's own `multicam_rig.py` work dir when installing a new calibration:

```
python3 scripts/scene_worker.py --add-ref RIG_WORK      # copies RIG_WORK/vggt.npz + calib/rig.json to the cache
```

Measured 2026-11-08 (3 cameras): depth-ratio scale per camera within 3 % of each other and 2-3 % of the centres
estimate; 2 cameras: depth 3.54 vs centres 3.47 vs planes 2.85.

## Storage and lookup

```
replay/scenes/<frame slug>.{json,splat}, _objects.json, _objects/*.glb      calibration-frame scene (publish_scene.py)
replay/scenes/clip/<stem>.json                                              per-replay manifest (status ok | camera_moved)
replay/scenes/clip/b-<time>-<hash>.{json,splat}, _objects.json, _objects/*.glb   a build (shared by cached replays)
```

persons-ui `attach_scene(tr, stem)` (served with `/evidence/<stem>_traj.json`) prefers the clip manifest when its
status is `ok`, else the calibration-frame scene; a `camera_moved` clip manifest adds `scene.warning` to the frame
scene. All URLs are absolute (`/replay/scenes/clip/...`); the `/replay` allow-list takes `scenes/clip/<file>` and
`scenes/clip/<build>_objects/<file>.glb`.

## Retention

The bridge's retention janitor deletes `evidence/<event>_*` with the sighting (90 days, unconfirmed people). The
worker's hourly sweep then removes every `clip/<stem>.json` whose `<stem>_traj.json` is gone, and every build no
manifest uses that is older than `--cache-h`. `python3 scripts/scene_worker.py --sweep` runs it once. Locally,
`~/.cache/scene-worker` keeps builds' person-free frames for the cache check for <= 24 h.

## Running

```
deploy/laptop/scene-worker.sh                  # transient unit: systemd-run --user --unit=scene-worker -p Restart=always
journalctl --user -u scene-worker -f
systemctl --user stop scene-worker
python3 scripts/scene_worker.py --event <stem> [--frame F] [--force] [--no-publish]    # one replay by hand
```

A persistent install would need a `~/.config/systemd/user/scene-worker.service` (ExecStart as in the launcher,
`Restart=always`, `RestartSec=60`, `Nice=10`, `IOSchedulingClass=idle`, `Environment=SSH_AUTH_SOCK=...` or a
dedicated passwordless key for `db-host` and `gpuserver` in `~/.ssh/config`), `systemctl --user enable --now`, and
`loginctl enable-linger operator` so it runs without a login session.

## Limits

- A scene needs recordings (or a live frame) of 2+ calibrated cameras; replays that cannot get them keep the frame scene.
- Two cameras cannot tell which one moved: the residual is shared between them.
- Scale follows the installed cameras' centres (the calibration's assumed ceiling/desk heights).
- A person standing still through the whole clip stays in the frames (`person_px` in the manifest).
- The replay viewer does not display `scene.note` / `scene.warning` yet.
