# Open-vocabulary objects ("carrying a ladder")

The profiler looks for objects named in words on the frames of a person event and reports what the person carries.
Code: `squirrel_guard/openvocab.py` (detector, person association, clip aggregation), export:
`scripts/export_openvocab.py`, vocabulary: `scripts/openvocab_vocab.txt`, evaluation: `scripts/openvocab_eval.py`,
unit test (no model, no GPU): `tests/test_openvocab.py`.

## Model choice

**YOLO-World v2, small** (`yolov8s-worldv2.pt`, Ultralytics), exported to ONNX with the vocabulary baked in.
`set_classes()` runs the CLIP ViT-B/32 text encoder once at export time and stores the prompt embeddings in the
model, so the ONNX graph is an ordinary YOLO detector (input 1x3x480x480, output 1x(4+prompts)x4725) that
onnxruntime runs on CPU like the other models here. No CLIP, tokenizer or PyTorch at runtime.

OWLv2 (Apache-2.0) was not used: OWLv2-B/16 runs a ViT-B over a 960x960 image (3600 patches, roughly 15x the
compute of YOLO-World-s at 640 by FLOP count), i.e. seconds per frame on the i9's CPU, and it needs the text tower
or cached query embeddings at runtime. It was not measured here; the FLOP gap alone rules it out for a pass that
runs on ~60 frames per event next to pose, re-ID and gait on 4 cores.

## Licences

* Ultralytics code and the YOLO-World / YOLOE weights they distribute: **AGPL-3.0** (same as the YOLOv8 detector,
  pose and seg models already used). Fine for this private system; shipping it as a product or service to others
  needs an Ultralytics Enterprise licence or a different model.
* YOLO-World (Tencent AILab, github.com/AILab-CVC/YOLO-World): GPL-3.0.
* CLIP (OpenAI, text encoder used at export time only): MIT.

## How it runs (profiler)

`[profiler] openvocab_model` in `deploy/gpuserver/squirrel-guard/config.toml` turns it on (empty = off; a missing or
broken model file only logs an error). During the end-of-event analysis (`motion_of`, the same decoded frames
as pose/gait, nothing buffered) every `openvocab_every`-th analysed frame **with a tracked person** gets one pass
on a square window around the person (2.2x the person box, at least 320 px): a person 300 px tall in the 1080p
frame would be ~100 px in a whole-frame 640 letterbox, too small to see a crowbar. At most `openvocab_max_frames`
passes per event.

Per frame (`OpenVocabDetector.detect`): every box takes its single best prompt across the whole vocabulary,
**distractor** prompts included (`coat = -`, `lamp = -`, `person = -`, ...); boxes won by a distractor are dropped.
Then NMS per reported label, then across labels for near-identical boxes (IoU 0.7: one object is "backpack" or
"bag", not both). Boxes below `openvocab_min_score` are ignored.

Per object vs the tracked person (`relation`):
* `carried`: a confident wrist keypoint lies in the object box (grown by 10 % of the person's height), or the object
  lies mostly (>= 60 %) inside the person box and is not bigger than ~60 % of it (worn backpack, box held to the body);
* `near`: within half a person height of the person box;
* `scene`: anything else.

Per clip (`ObjectAggregator.summary`): one entry per label with max score, frames seen / sampled, carried frames and
fraction, `carried` (>= `openvocab_min_frames` carried frames and >= 30 % of the frames it was seen), the majority
relation, best time and normalised box, and which prompts fired. **Labels seen on fewer than `openvocab_min_frames`
(2) sampled frames are dropped.** The list goes into the behaviour message as `"objects"` and one line on the
evidence card.

## Bridge (recorder host)

* `sightings.objects jsonb` (`persons_schema.sql`; apply it before deploying the new bridge: the explanation query
  reads the column).
* `explain.py`: reason code `CARRYING_<LABEL>` (e.g. `CARRYING_LADDER`, `CARRYING_PRY_BAR`) and a "did" line per
  carried object.
* `intel.py`: optional alert `watched_object` (reason `WATCHED_OBJECT`) when an unknown / watch-list person carries a
  label from `WATCH_OBJECTS` (comma list, empty = off) with max score >= `WATCH_OBJECT_MIN` (default 0.3).

## Changing the vocabulary

1. Edit `scripts/openvocab_vocab.txt`. One prompt per line; `prompt = label` maps several prompts to one reported
   label; `prompt = -` is a distractor (detected, never reported). Short concrete nouns work best.
2. Re-export (about a minute, CPU is fine; the first run downloads ~25 MB weights + the 340 MB CLIP model, and
   Ultralytics pip-installs its `clip` package from git+https://github.com/ultralytics/CLIP into the venv):
   `.venv/bin/python scripts/export_openvocab.py --out models/yolov8s-worldv2-sg.onnx`
3. Check it on clips: `.venv/bin/python scripts/openvocab_eval.py --model models/yolov8s-worldv2-sg.onnx --sheet /tmp/ov clip.mp4`
   (prints raw hits and the clip summary; `--sheet` writes the best crop per label so you can see what each hit is).
4. Copy the ONNX file to gpuserver `config/model_cache/` and restart the profiler. `openvocab_classes` in the config can
   narrow the reported labels without a re-export, but cannot add new words.

More prompts cost almost nothing at runtime (one extra output row each); the per-frame cost is the backbone.

## Measurements (2026-11-06, laptop i7-14700HX, onnxruntime 1.31 CPU)

The laptop was busy with other work (load average 16-19 on 28 threads), so absolute numbers are pessimistic; the
ratios hold. Per frame, `detect()` on a 1080p frame, median of 30:

| model | 1 thread | 4 threads |
|---|---|---|
| yolov8s-worldv2, 480 (shipped) | 252 ms | 82 ms |
| yolov8s-worldv2, 640 | 431 ms | 147 ms |
| yolov8s-pose 640 (already in the profiler, for scale) | 247 ms | 86 ms |

So the object pass costs about what one pose pass costs; with `openvocab_every = 2` and `openvocab_max_frames = 60`
it adds at most ~5 s CPU per event at 4 threads (pose runs on every analysed frame, up to 320). Memory: +~130 MB RSS
(session + one inference), model file 50 MB. 480 vs 640 on the same clips: same objects found, scores within
+-0.1 (ladder 0.86 vs 0.72), 40 % less time; because the input is a window around the person, 480 is enough.

Clips (`scripts/openvocab_eval.py`, min score 0.25, every 2nd frame at 8 fps, final vocabulary with distractors).
"carried" is the summary verdict; the crops (`--sheet`) were checked by eye.

| clip | reported | truth |
|---|---|---|
| real/walk.mp4 (basement, person holding a mug) | ladder near 0.44, package near 0.29; nothing carried | both false: the ladder is a wooden chair, the package a book on a shelf |
| real/fy-1853.mp4 (basement, coat, lamp, monitors, shelves) | ladder near 0.29; nothing carried | false (same chair) |
| clips/fog.mp4, clips/forest.mp4 (legs walking) | fog: nothing; forest: backpack/duffel/pry bar/bag, none carried (2-4 frames each) | false hits on dark tree trunks/leaves, all "scene"/"near" |
| openvocab/hiker (hiker with a backpack, Pixabay 176780) | **carrying backpack 0.67, 80 of 85 frames** | correct |
| openvocab/ladder (electrician on a step ladder, Pixabay 4504) | **ladder 0.85** (scene: he stands on it), **carrying backpack 0.57** (his tool vest), carrying bag 0.36, gas can/plastic bag in view | ladder correct; vest = backpack reasonable; gas can is a bag on the floor |
| openvocab/street (feet, bicycles, Pixabay 21782) | bicycle 0.85, bag 0.53 (a red shopping bag), not carried | correct objects; carried by other people |
| openvocab/walkman (man walking, handheld tracking shot, Pixabay 45254) | carrying ladder 0.42 | **false**: a railing; the moving camera defeats the static-fixture rule |
| openvocab/hammer (POV close-up of drilling, Pixabay 106610) | carrying gas can 0.47, power drill near 0.57 | false/misplaced: the "gas can" is a log under a drill, the "drill" is a vise. Close-up POV is far from this camera's view |

Before tuning (min score 0.1-0.15, no extra distractors, no cross-label NMS, no static rule) the basement clips
reported "carrying gun 0.21" (the mug), "suitcase 0.54" (the coat), "gas can 0.41" (a heater on a stand) and the
hiker got both "backpack" and "bag" for one object. The changes that fixed those: min score 0.25, the distractor
prompts at the end of `openvocab_vocab.txt` (shoe, railing, fence, bench vise, space heater, bed, blanket, book),
cross-label NMS, and the static-fixture rule.

What this does NOT show yet: recall for the tools that matter (crowbar, bolt cutters, knife, a carried ladder) on
THIS camera; no stock clip had them in a fixed-camera view. Stage a few walks across the yard carrying a ladder, a
pry bar and a box, run `scripts/openvocab_eval.py` on the Frigate clips, and only then set `WATCH_OBJECTS`. YOLO-World
scores for thin tools are low (0.15-0.35 in the clips above); expect to tune `openvocab_min_score` per label if needed.
