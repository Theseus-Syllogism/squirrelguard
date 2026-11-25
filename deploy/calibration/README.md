# Height calibration (Phase 2) and gait (Phase 3)

The profiler runs a pose model (YOLOv8s-pose) over each finished person event and sends:

- **gait**: always, when someone walks roughly side-on for a few steps. It records cadence (steps per minute), relative step length, arm swing and torso lean. With a calibrated camera it also records step length in metres and walking speed.
- **height**: only once `/calib/<camera>.json` exists. It's the median over the frames where the person stands or walks upright with the whole body in view.

Both appear on the identity pages (http://198.51.100.20:8092) and in the LLM reports. Neither one decides identity. The identity page flags a sighting whose height differs from that person's usual height by more than 12 cm.

All commands run on gpuserver in `/mnt/media-extra/frigate`. Output goes to `config/calib/`, which the profiler mounts at `/calib` and reloads automatically.

## 1. Lens calibration (optional; can be done before mounting)

The Wyze v3 lens bends straight lines. A checkerboard calibration corrects for that, which brings simulated height error down from about 3–4 cm to about 1.6 cm. The lens doesn't change when the camera is moved, so you can do this step indoors.

1. Print `checkerboard.png` on letter paper, landscape, at **100 % scale** (no fit-to-page). Measure one square: it must be 25 mm. Tape the sheet flat to cardboard.
2. Start the capture. It runs for 3 minutes, or until it has 30 different views:

   ```
   set -a && . ./.env && set +a
   docker compose run --rm -T -e FRIGATE_RTSP_PASSWORD profiler sh -c \
     'python -m squirrel_guard.calibrate intrinsics --source rtsp://camstack:$FRIGATE_RTSP_PASSWORD@192.0.2.20:8554/frontyard'
   ```

3. While it runs, hold the board 0.5–1.5 m from the camera and move it slowly. Cover the corners, the edges and the centre, near and far, tilted up, down and sideways. Hold each pose for about a second. A good result has an rms below 1 px.

## 2. Camera pose (after mounting; repeat whenever the camera is moved)

**Walk method (recommended, no tape measure).** Someone whose height you know (shoes on) walks slowly around the **whole** yard for about a minute: close to and far from the camera, left and right, whole body in view. Find the event ID in Frigate (Explore, then the event, then copy the ID), then run:

```
docker compose exec profiler python -m squirrel_guard.calibrate walk --camera frontyard --height-cm 180 --event <event id>
```

`--event` can be repeated. The result's `residual_cm_median_abs` should be under about 4 cm.

**Points method (alternative).** Run `calibrate snapshot --out /calib/snap.jpg`. Mark at least 6 points on the image and measure where they are in metres from one origin, for example a house corner. Use ground points (z = 0) plus the tops of a few verticals of known height, such as a door frame or a post. Write them to a JSON file:

```json
{"frame_size": [1280, 720], "points": [{"px": [612, 540], "world": [0, 0, 0]}, {"px": [640, 210], "world": [0, 0, 2.03]}]}
```

Then run `calibrate points --json /calib/points.json`.

## 3. Check

Have someone else of known height walk through. Then run:

```
docker compose exec profiler python -m squirrel_guard.calibrate check --event <their event id>
```

Expect the result within about 5 cm. Hair, hats and shoes count as height. Posture and a partly hidden body make it worse.

## Limits

- Heights need the feet on the same ground plane as the calibration. Steps, slopes and hidden feet give wrong values, and those frames are mostly filtered out.
- Gait from a single fixed camera is weak. It's only marked `usable` for a side-on walk of four or more steps, and it's for review, not identification.
