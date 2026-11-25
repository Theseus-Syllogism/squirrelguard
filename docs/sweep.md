# Small-person sweeper

**Problem (2026-11-10 15:38, v4):** a walker on the far sidewalk (~40-45 m, ~65 px tall at 2560x1440) produced no
Frigate person event, so no profile and no face attempt. Frigate's detector (YOLOv8s Open Images V7, kept for its
squirrel class) scored the person 0.02-0.08 (Frigate needs 0.35), and only 0.1-0.24 even with a 2.5x zoomed region.
The COCO-trained YOLOv8s-pose already used by the profiler scored 0.53-0.78 on a 2x zoomed (320 px) region.

**Fix:** `squirrel_guard/sweep.py`, started by the profiler for `[profiler] sweep_cameras`:
1. once a second, Frigate's latest frame (detect resolution) -> frame difference -> small moving blobs;
2. a 320 px tile around each blob (up to 6) -> COCO person model at 640 input (2x zoom);
3. short tracks; a person seen in 2+ sweeps while Frigate has no person event in progress becomes a Frigate event
   (`POST /api/events/<camera>/person/create`, recording included, box drawn) and is queued for profiling;
4. `profile()` has a zoomed fallback (band of tiles around the event's box) when the whole-frame detector finds no
   person, so the swept person also gets re-ID, attributes, gait.

Dry run on the 15:38 recording: found at second 9, tracked 22 s (scores 0.52-0.87), one event; no events on a
people-free passing-car clip; ~0.36 s CPU per sweep.

**Limits:** faces at this distance are ~8 px: no face recognition is possible from v4 at 40 m (needs ~30+ px faces).
Re-ID/attributes on 65 px crops are weaker than on close-ups. Sweeps are 1 fps: someone crossing the view in under
~2 s can still be missed.
