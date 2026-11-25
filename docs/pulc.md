# PULC attributes (PaddleClas): apparent clothing/bag tags for people, colour/type for vehicles

Two small classifiers from PaddleClas' PULC series ("Practical Ultra Lightweight Classification", PPLCNet_x1_0
backbone, ~7 MB each) run in the profiler next to the existing re-ID / Qwen description:

| model | trained on | outputs |
|---|---|---|
| `pulc_person_attribute.onnx` | **PA100K** | 26 sigmoid scores, see below |
| `pulc_vehicle_attribute.onnx` | **VeRi-776** | 10 colours + 9 body types |

They add a structured, cheap, deterministic summary to each sighting (`pulc` in the MQTT message, `pulc` jsonb column),
used as **weak** matching evidence and as `/search` filters. Qwen's free-text description stays as it was.

## Licences and datasets

* PaddleClas code and the released inference weights: **Apache-2.0** (github.com/PaddlePaddle/PaddleClas, mirror
  gitee.com/paddlepaddle/PaddleClas, release/2.6).
* The weights were trained on research datasets with their own terms:
  * **PA100K** (Liu et al., "HydraPlus-Net", ICCV 2017): 100k pedestrian crops from 598 outdoor surveillance cameras;
    released for academic / non-commercial research.
  * **VeRi-776** (Liu et al., ECCV/ICME 2016): ~50k images of 776 vehicles from 20 city cameras; academic /
    non-commercial research use, access on request.
  * PP-ShiTuV2 (evaluation only, not used in production): trained on a mix of public sets (Aliproduct, GLDv2,
    VeRi-Wild, LogoDet-3K, SOP, Inshop, iNat, ...), several research-only.
* This is a private, non-commercial home system, which fits those terms. Using the weights in a product or service
  would need a fresh look at the dataset terms (Apache-2.0 on the weights does not override them).
* The `.onnx` files live in `models/` (untracked) and on gpuserver in `config/model_cache/`; they are not committed.

## Export (reproducible)

```sh
# converter env: paddle2onnx 1.0.x converts the old .pdmodel format without PaddlePaddle; no 3.13 wheels, so 3.10
uv venv -p 3.10 ~/.cache/sg-paddle2onnx && VIRTUAL_ENV=~/.cache/sg-paddle2onnx uv pip install paddle2onnx==1.0.6 six
.venv/bin/python scripts/export_pulc.py --models person,vehicle --check      # + ,shitu for the re-ID comparison
```

The script downloads the PaddleClas inference tarballs (resume + retry: the Baidu CDN drops connections mid-transfer),
checks pinned sha256s, converts with opset 13 and writes metadata into each ONNX file: `sg_kind`, `sg_labels` (column
order), `sg_input_wh`, `sg_mean`/`sg_std`, `sg_thresholds`, `sg_dataset`, `sg_licence`, `sg_source`.
`squirrel_guard/pulc.py` reads labels and sizes from there. `--check` runs PaddleClas' own demo images and compares
with the outputs printed in the PaddleClas docs: all three match (yellow 0.9893 / hatchback 0.9734, red 0.9999 / sedan
1.0000, and the 26-bit vector for `090004.jpg`), so the conversion and preprocessing are faithful.

Preprocessing (PaddleClas `deploy/configs/PULC/*/inference_*.yaml`): plain resize, no letterbox, to 192x256 (person) or
256x192 (vehicle), BGR->RGB, /255, ImageNet mean/std, CHW. The models end in a sigmoid; dynamic batch.

## Labels

Person (column order, from `ppcls/data/postprocess/attr_rec.py`):
`Hat, Glasses, ShortSleeve, LongSleeve, UpperStride, UpperLogo, UpperPlaid, UpperSplice, LowerStripe, LowerPattern,
LongCoat, Trousers, Shorts, Skirt&Dress, Boots, HandBag, ShoulderBag, Backpack, HoldObjectsInFront, AgeLess18,
Age18-60, AgeOver60, Female, Front, Side, Back`. PaddleClas thresholds: 0.5, Glasses 0.3, HoldObjectsInFront 0.6;
sleeves = whichever of Long/Short is higher. (The PA100K paper orders the age columns the other way round; we follow
the PaddleClas code and docs. Age18-60 is in the middle either way, which is what almost every visitor gets.)

Vehicle: colours `yellow orange green gray red blue white golden brown black`, types `sedan suv van hatchback mpv
pickup bus truck estate`; a top class is reported only at >= 0.5.

**Policy.** Everything is labelled *apparent*. Age band and gender are a model's reproduction of PA100K annotators'
guesses from a low-resolution crop; they are never identity. PA100K has no ethnicity/skin attribute and nothing here
infers one.

## What the profiler sends

`[profiler]` keys (deploy/gpuserver/squirrel-guard/config.toml; that file is copied into the image at build):

| key | default | meaning |
|---|---|---|
| `pulc_person_model` | `/models/pulc_person_attribute.onnx` | "" = off |
| `pulc_vehicle_model` | `/models/pulc_vehicle_attribute.onnx` | "" = off |
| `pulc_max_crops` | 6 | best N crops per visit (detector score x height) |
| `pulc_min_aspect` | 1.5 | person crops must be upright (h/w) ... |
| `pulc_threads` | 2 | onnxruntime intra-op threads |

... and not cut off by the top/bottom of the frame (`profiler.full_body`). That gate matters: PA100K is full-body
pedestrians, and on close-ups PULC invents things (see below). A missing/broken model file logs once and turns the
feature off; the profiler keeps running.

Person sighting `pulc`:
```json
{"model": "pulc_person_attribute", "crops": 4, "considered": 6,
 "attributes": {"Hat": 0.04, "LongSleeve": 0.93, "Backpack": 0.81, ...},      // visit mean, no age/gender/orientation
 "orientation": {"front": 0.21, "side": 0.30, "back": 0.49},                   // mean of per-crop normalised scores
 "views": {"front": 1, "side": 1, "back": 2},                                  // crops by argmax
 "apparent": {"gender_female": 0.08, "age": {"under_18": 0.01, "18_60": 0.98, "over_60": 0.01}},
 "tags": ["long_sleeves", "trousers", "backpack"], "spread": 0.21}            // spread = max per-label std over crops
```
or `{"crops": 0, "skipped": "no full-body view", "considered": n}`. Vehicle sighting `pulc`:
`{"color": "blue", "type": null, "colors": {top 3}, "types": {top 3}, "crops": n}`.

## recorder host bridge, explain, UI

* `persons_schema.sql`: `sightings.pulc`, `vehicle_sightings.pulc` (jsonb, `ADD COLUMN IF NOT EXISTS`). The bridge
  stores them in a savepoint, so a bridge deployed before the schema keeps working.
* Vehicles: when Qwen gave no body type / colour, PULC's top class (if >= 0.5) is used for the existing "same look"
  check (mpv -> minivan, estate -> wagon, golden -> gold). Qwen's words win when present.
* `explain.py` `pulc_evidence()` / `pulc_adjust()`: weak evidence on top of an appearance match (and on the shown
  estimate for a "possible" match), compared with the identity's earlier visits:
  * backpack (any visit in 30 days, needs >= 2 earlier visits), hat / long sleeves / shorts / skirt-dress / long coat
    (same-day visits only, <= 12 h: people change clothes);
  * agree = current >= 0.7 and >= 60 % of earlier visits >= 0.5: odds x1.1-1.15 each -> `PULC_CONSISTENT`;
  * contradict = current >= 0.7 vs <= 10 % earlier, or current <= 0.15 vs >= 90 % earlier: x0.85-0.9 each ->
    `PULC_CONTRADICTS`;
  * apparent gender (>= 0.9 vs earlier mean <= 0.1 or the reverse, >= 2 earlier visits) x0.7 ->
    `PULC_APPARENT_GENDER_DIFFERS`; apparent under-18 likewise (0.8 / 0.1) -> `PULC_APPARENT_AGE_DIFFERS`;
  * the total is clamped to 0.6-1.4 so PULC can move a band edge but never decide a match. `scores.pulc_factor` records it.
* persons-ui: the Why? page has an "Apparent attributes" section (tags, view distribution, apparent gender/age,
  attribute table); identity and vehicle cards get one PULC line; `/search` has attribute checkboxes (backpack, hat,
  long/short sleeves, shorts, long coat, skirt/dress, handbag, shoulder bag). Alone they list the newest 60 matching
  sightings from Postgres; with a text query they filter the Frigate results (people only).

## Measurements (laptop i7-14700HX, onnxruntime 1.23 CPU; gpuserver's i9-12900K is the same generation)

Latency per crop, batch 1, median of 50: person **5.0 ms (1 thread) / 3.7 ms (4 threads)**, vehicle 5.0 / 3.7 ms.
Six crops per visit add ~25 ms (7 MB weights per model); nothing measurable next to pose/gait/YOLO-World.

Outputs (`scripts/pulc_eval.py`, 12 frames per clip, gate on):

| clip | what is there | PULC visit summary | verdict |
|---|---|---|---|
| hiker (stock) | big red hiking pack, walking away | short sleeves, trousers; **front** 56 %; backpack 0.50 (not tagged) | sleeves/trousers right; **missed the backpack and got the view backwards** (the pack covers the whole back, unlike PA100K's school bags) |
| walkman (stock) | white T-shirt, chinos, from behind | short sleeves, trousers, glasses (0.3 threshold); back 99 %, male | right except glasses (cannot be seen from behind) |
| street (stock, night) | dark, small, edge-cut | gated out (0 crops) | gate correct; ungated it said long sleeves + trousers, plausibly |
| ladder / hammer (stock) | torso / hands close-ups | gated out | ungated: "handbag, plaid top, long coat, **female 0.97**" for a hand with a drill: garbage, hence the gate |
| fog (stock) | walkers in fog | long sleeves, trousers, handbag; back 55 % | plausible; handbag over-fires in general |
| nyc crossing (stock) | pedestrians | short sleeves, trousers; back 56 % | plausible |
| basement walk.mp4 (Wyze) | jacket, holding a mug | long sleeves, long coat, trousers, shoulder bag (2 of 5 crops pass) | coat right, bag wrong (mug/strap) |
| 24 stored recorder host person crops | seated close-ups at the desk | (would be gated: edge-cut, not upright) | ungated: female 0.75-0.98 for two men, bags and skirts invented, view "side" while facing the camera: **useless on this camera position** |
| IR simulation (grey, gamma 1.8, noise) | walkman / hiker | walkman unchanged; hiker apparent gender 0.41 -> 0.73 | clothing roughly survives IR, apparent gender does not |

Vehicles (no vehicle sightings are stored on the recorder host yet, so stock clips): parked dark-blue car -> blue 0.98, type
below threshold; anime black car under a pink sunset -> "red pickup" (colour cast); tilt-shift highway -> no
confident colour/type. Colour is decent in neutral light, type is often unsure.

Honest impression: on upright full-body views PULC gets sleeves, trousers/shorts and coats mostly right and the
front/side/back split is usually right (but not with large backpacks). Bags are unreliable (handbag/shoulder bag fire
on hands and straps; a huge backpack was missed), glasses fire from behind, apparent gender is unstable (IR, hair,
close-ups). The basement camera's typical view (seated, close, cut by the frame) is out of domain and is gated out.
Hence: weak evidence with a clamp, only a handful of attributes in explain, and the gate.

## PP-ShiTuV2 vs DINOv2 for vehicle re-ID (offline only)

`scripts/vehicle_reid_compare.py` (stock clips in `~/.cache/squirrel-guard-test/vehicles/`, Pixabay licence): vehicles
detected at 3 fps and linked into tracklets by box overlap; positives = same tracklet >= 1.5 s apart, negatives =
different tracklets. 490 crops, 79 tracklets, 478 positive / 118k negative pairs:

| model | dim | ms/crop (4 thr) | cos same / diff | ROC AUC | rank-1 | same pairs below diff p99 |
|---|---|---|---|---|---|---|
| DINOv2 ViT-S/14 (production) | 384 | 49 | 0.80 / 0.63 | 0.757 | 0.14 | 65 % |
| PP-ShiTuV2 PPLCNetV2_base | 512 | 8 | 0.67 / 0.35 | **0.864** | **0.43** | 50 % |

Caveats: the highway clip breaks one car into several tracklets (scored as "different"), so absolute numbers are low
for both; same-minute footage is easier than a car seen on different days. Still, ShiTuV2 separated vehicles better
and is 6x cheaper. Worth a proper test on real vehicle sightings once the recorder host has some (`--crops DIR` with
`<vehicle-id>_*.jpg` files); production matching is unchanged (thresholds VEHICLE_MATCH_SIM 0.80 / POSSIBLE 0.65
are DINOv2-specific and would need recalibrating).

## Deploy checklist (not done; for when it is wanted)

1. `scripts/export_pulc.py`, then copy `models/pulc_{person,vehicle}_attribute.onnx` to gpuserver
   `config/model_cache/` (before `compose up`: a missing bind-mount source becomes an empty directory).
2. Rebuild the squirrel-guard image (config.toml is baked in), restart `profiler`.
3. Apply `deploy/recorder/persons_schema.sql`, rebuild/restart the bridge and persons-ui.
