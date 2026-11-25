# Vehicle analysis

Three additions to the vehicle lane: our own number-plate reading, a colour measurement that knows when it cannot
see colour, and a match decision that explains itself.

Code: `squirrel_guard/vehicle_read.py` (profiler, gpuserver: plates, colour, motion), `squirrel_guard/vehmatch.py`
(matching; copies in the recorder host bridge and persons-ui). Tests: `tests/test_vehicles.py`.

## Plate reading

Plate detector (open-image-models YOLOv9-t 640) and plate reader (fast-plate-ocr `cct-s-v2` global model, Latin
plates from about 60 countries including the US). Both are MIT-licensed code; the models are pinned by SHA-256 in
`deploy/gpuserver/squirrel-guard/Dockerfile` and switched on by `plate_models` in the profiler config.

Every vehicle crop of a visit is read, and the readings are combined character slot by character slot, so a frame
that misreads one character is outvoted by the others. The result keeps a confidence for each character; the vehicle
page shows unsure characters faded and underlined. A reading under 50 % overall confidence, or one the frames do not
agree on, is kept as a candidate and **not** used for matching. Frigate's own plate reader still comes first when it
read something; ours fills in when it did not.

Plates are used only to recognise repeat visits. There are no outside lookups.

## Colour

Infrared night images have no colour. They are recognised (no saturation anywhere in the frame), and the colour is
recorded as unknown rather than guessed; the language model's and PULC's colour words are dropped for those visits
too, because they guess colours on infrared images. In daylight the colour is white-balanced from the neutral
surfaces around the car (road, walls), not from the whole frame, so a car that fills the frame keeps its colour, and
low sun no longer turns a silver car red. Glare and deep shadow are ignored, and dark paint with sky reflections
counts as black.

On the stock test photos, after the fix: 9 of 11 sensible (red ×2, blue ×2, yellow, mint green ×2, beige, black; a
black-and-white photo correctly refused), one copper car read as red with low confidence, one odd crop read as mixed.

## Movement

From Frigate's path data: direction across the image (left to right, toward the camera…) and how long the vehicle
stood still. Shown on each vehicle sighting.

## Explained matching

`vehmatch.explain()` scores a new visit against each candidate vehicle and lists its reasons:

| Evidence | Effect |
|---|---|
| Plate | read the same: strong; differs only in characters that are easily misread (0/O, 8/B, 5/S…) or that the reader was unsure of: strong to moderate; clearly different: decides **different**, however alike the cars look |
| Appearance | DINOv2 similarity, from "looks very similar" (0.85+) to "looks different" |
| Colour | same / similar (lighting can shift grey↔white, red↔orange) / different; **not compared at night** |
| Body type | same / different |

The result is a band (strong, likely, possible, unlikely) and a decision: file under the vehicle, keep separate but
flag as possibly the same, or new vehicle. Look-alike cars with different plates are no longer merged, which the
earlier appearance-first rule could do. The bands are this system's own scale: **no labelled vehicle data has
calibrated them yet**. The reasons are stored with each sighting and shown on the vehicle page.

## Not done

- Make and model come from the local vision-language model's description only; no licensed make/model classifier
  was found (the common training sets are non-commercial).
- No calibration of the bands on real repeat visits; no speed in m/s (needs camera calibration for the vehicle lane).
