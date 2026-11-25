# Re-ID and gait signatures: newer models vs the current ones (2026-11-06)

Question: do newer person re-ID and gait-signature models identify this household better than OSNet x0.25
(`squirrel_guard/reid.py`) and OpenGait GaitBase (`squirrel_guard/gait_embed.py`)?

**Short answer: not measurably, on the data that exists today.** Every re-ID candidate's AUC interval overlaps the
production model's, and no licensed gait model beats GaitBase. The candidates are integrated but switched OFF.
The numbers can only change once real other people are labelled (see "What the user should label").

## Data and labels

Read-only copy of the recorder host `sightings` table on 2026-11-06 20:28: **96 sightings in about 29 hours** (2026-11-05
15:32 to 2026-11-06 20:14), cameras `frontyard` and `v4`. Event clips were still in Frigate's retention for
72 of them; the other 24 (all from the afternoon of 11-05) only have the stored best crop.

Labels come only from evidence that does not depend on body re-ID:

| tier | rule | sightings |
|---|---|---|
| A | face decision: `match_kind` face_template / face, or `face_meta.match.status = match` | 5 (#30, #91-#94) |
| B | the visit's face template (`face_vec`) has cosine >= 0.40 (FACE_MATCH_SIM) to the mean template of the A visits, with >= 3 faces. Same face model, but without the liveness and count gates | 12 (#26 #33 #34 #36 #37 #38 #39 #42 #76 #78 #80 #81) |
| U | `labels.csv` filled in by the user | none yet |

- **No confirmed other person exists.** Different-person statistics therefore come from a stand-in: 11 Pixabay
  stock clips (one per uploader, so different clips are assumed to be different people, not verified), 23
  "visits" (first/second half of each clip). They are **out of domain**: bright, sharp, mostly outdoor footage,
  unlike a dim basement or a doorstep at night. Owner-vs-stock similarity is an optimistic stand-in for a real
  stranger, so every "TPR at 0 FP" and suggested threshold below is optimistic too.
- All owner labels fall on **one afternoon/evening** (2026-11-06, 16:17-20:14). A is #91-#94 within 11 minutes in
  one outfit plus #30 four hours earlier; A+B covers at least two outfits (jacket, then short sleeves). Nothing here tests days or weeks apart,
  night IR, or seasonal clothing.
- Tier B uses the face model without its gates, so it can be wrong. The 28 desk sightings on `v4` (#47-#74,
  ~40 side-on faces each) score only 0.10-0.26 against the A template: either another person or a side view the
  face model cannot handle. They are NOT labelled, and they are why the labelling sheet matters.
- Finding along the way: sighting **#81** (identity 17, 26 faces) has face cosine 0.76 to the owner's template, so
  identity 17 very probably holds the owner split off by an outfit change (OSNet scored it 0.81 to the owner centroid,
  above every other unknown identity). Not merged: that is the user's call in the persons UI.

## Candidates

| model | code licence | weights (source) | published MSMT17 | evaluated |
|---|---|---|---|---|
| OSNet x0.25 MSMT17 (production) | MIT (torchreid) | provenance not recorded in the repo | torchreid zoo, MSMT17-all -> Market: R1 59.9 / mAP 31.0 (if it is that checkpoint) | yes |
| OSNet-AIN x1.0 MSMT17 (all) | MIT, KaiyangZhou/deep-person-reid | torchreid MODEL_ZOO, Drive `1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal` | MSMT17-all -> Market: R1 70.1 / mAP 43.3 | yes |
| CLIP-ReID ViT-B/16 MSMT17 | MIT, Syliz517/CLIP-ReID | README table, Drive `1BVaZo93kOksYLjFNH3Gf7JxIbPlWSkcO` | paper: mAP 73.4 / R1 88.7 | yes (fp32, int8) |
| SOLIDER Swin-T MSMT17 | MIT, tinyvision/SOLIDER-REID | README, Drive `10YLhMbwvmxZl3gTVo2BN_828SKZHdCjr` | mAP 67.4 / R1 85.9 | yes (fp32, int8) |
| SOLIDER Swin-S MSMT17 | MIT, tinyvision/SOLIDER-REID | README, Drive `1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b` | mAP 76.9 / R1 90.8 | yes (fp32, int8) |

Not evaluated, and why:
- **OpenGait DeepGaitV2, BigGait, SkeletonGait** (and GaitGraph2): the OpenGait repository still has **no licence
  file** (checked 2026-11-06; its README says "only used for academic purposes"). The rule for this project is
  "never use models whose code has no licence", so they are out. **The production GaitBase is from the same
  repository** (`scripts/export_gaitbase.py` vendors its code), so it does not meet that rule either. Flagged
  here; it stays in place until the user decides.
- **GaitGraph** (MIT, tteepe/GaitGraph, CASIA-B weights in its v0.1 release): trained on CASIA-B (124 people, lab
  corridor, 25 fps HRNet keypoints in 320x240 frames). Skeleton models trained like this rank far below
  silhouette models in the wild (Gait3D / GREW rank-1 in the low tens of percent in the SkeletonGait paper's
  comparison, versus ~60 % for GaitBase). Not worth an evaluation here.
- **Keypoint-promptable re-ID (KPR)**: licence `NOASSERTION` on GitHub. **TransReID** (MIT): same ViT-B/16
  family as CLIP-ReID with lower published MSMT17 numbers; skipped for time. **Clothes-changing models**
  (Simple-CCReID, Apache-2.0): trained on small clothes-change sets (PRCC/LTCC), not checked further.

Weights are trained on research datasets (MSMT17, GREW): **local use only, never publish them or the embeddings**,
the same terms as the InspireFace models.

## Re-ID results

`scripts/reid_gait_eval.py report`. "Visit" = the profiler's own recipe (6 frames spread over the clip, YOLOv8-OIV7
largest person, crops averaged), for the 72 sightings with a clip; the stored best crop for the rest. Cosine
everywhere. AUC 90 % CI from 500 bootstrap resamples of sightings and stranger visits (not pairs).

**Labels A+B, visit embeddings: 17 owner sightings (136 owner pairs), 23 stranger visits**

| model | owner cos median | p10 | same cam | other cam | <30 min | >=30 min | stranger cos median | max | AUC | 90 % CI | rank-1 | mAP | TPR at 0 FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OSNet x0.25 (prod) | 0.73 | 0.54 | 0.73 | 0.73 | 0.76 | 0.71 | 0.51 | 0.71 | 0.91 | 0.84-0.99 | 1.00 | 0.92 | 0.54 |
| OSNet-AIN x1.0 | 0.71 | 0.56 | 0.72 | 0.70 | 0.76 | 0.70 | 0.52 | 0.67 | 0.95 | 0.90-0.99 | 1.00 | 0.93 | 0.68 |
| CLIP-ReID ViT-B/16 | 0.88 | 0.80 | 0.89 | 0.88 | 0.93 | 0.88 | 0.78 | 0.87 | 0.94 | 0.88-1.00 | 1.00 | 0.93 | 0.66 |
| CLIP-ReID int8 | 0.88 | 0.80 | 0.88 | 0.88 | 0.93 | 0.88 | 0.78 | 0.88 | 0.92 | 0.84-0.99 | 1.00 | 0.91 | 0.49 |
| SOLIDER Swin-T | 0.82 | 0.72 | 0.82 | 0.81 | 0.87 | 0.81 | 0.70 | 0.82 | 0.92 | 0.84-0.98 | 1.00 | 0.90 | 0.49 |
| SOLIDER Swin-T int8 | 0.82 | 0.72 | 0.82 | 0.82 | 0.87 | 0.82 | 0.70 | 0.82 | 0.91 | 0.83-0.98 | 1.00 | 0.90 | 0.49 |
| SOLIDER Swin-S | 0.96 | 0.94 | 0.96 | 0.96 | 0.97 | 0.96 | 0.92 | 0.97 | 0.96 | 0.92-0.99 | 1.00 | 0.94 | 0.29 |
| SOLIDER Swin-S int8 | 0.96 | 0.94 | 0.96 | 0.96 | 0.97 | 0.96 | 0.92 | 0.97 | 0.96 | 0.93-0.99 | 1.00 | 0.94 | 0.33 |

Rank-1 / mAP: each owner sighting queries the other labelled sightings plus the 23 stranger visits. "TPR at 0 FP" =
share of owner pairs above the highest owner-vs-stranger cosine.

Other cuts (full tables in `DATA/report.md`):
- **Labels A only** (5 sightings, 10 pairs): AUC OSNet 0.77 (0.55-1.00), OSNet-AIN 0.90, CLIP-ReID 0.88,
  Swin-T 0.83, Swin-S 0.98 (0.93-1.00); rank-1 0.80 for every model. Ten pairs decide nothing.
- **Stored best crop only** (one crop, no averaging), A+B: AUC OSNet 0.80, OSNet-AIN 0.85, CLIP-ReID 0.85, Swin-T
  0.79, Swin-S 0.80, all CIs overlapping. Averaging 6 crops per visit helps every model more than any model change.

What this shows:
- **No candidate is measurably better.** The point estimates move by a few hundredths and every interval overlaps
  OSNet's. With 17 sightings of one person on one day and no in-domain stranger, a difference would have to be
  large to show, and it is not.
- The newer models squeeze everything into a narrow, high cosine band (Swin-S: owner 0.96, strangers 0.92). They
  rank about as well, but their thresholds would sit on a knife edge: Swin-S has the best AUC and the worst
  TPR at 0 FP.
- **Production thresholds:** under OSNet only **57 %** of owner-owner visit pairs reach MATCH 0.70 (87 % reach
  POSSIBLE 0.55): this is why the owner keeps getting new identities (identity 17). At the same time the highest
  owner-vs-stock-stranger cosine is **0.71**, already above 0.70. Lowering MATCH to stop the splits would let
  strangers in; that trade-off needs real negatives, not a model swap.
- Not measured at all: clothing change across days, night IR, a real second household member, and a real
  stranger in the same lighting. These are exactly what the newer models claim to be better at.

Suggested thresholds IF a model were switched (do not apply; stock-stranger based, so optimistic):
CLIP-ReID MATCH ~0.88-0.90 (just above the stranger max 0.87; ~half of the owner pairs), POSSIBLE ~0.80 (owner
p10). OSNet-AIN MATCH ~0.68-0.70, POSSIBLE ~0.56. SOLIDER Swin-S: no usable margin (0.97 vs 0.96).

### CPU runtime and size

`scripts/reid_gait_eval.py bench --threads 4`, laptop i7-14700HX pinned to 2 P-cores (4 threads), median of 20
batch-1 runs; the machine had a background load of ~9 during the run. gpuserver's 4-core profiler is probably slower (not
measured there: gpuserver was off limits): treat these as lower bounds.

| model | ONNX MB | dim | ms / crop | ms / 6-crop visit |
|---|---|---|---|---|
| OSNet x0.25 (prod) | 0.9 | 512 | 4.6 | 23 |
| OSNet-AIN x1.0 | 8.7 | 512 | 51.9 | 311 |
| CLIP-ReID ViT-B/16 | 344.8 | 1280 | 84.9 | 502 |
| CLIP-ReID int8 | 88.9 | 1280 | 30.8 | 175 |
| SOLIDER Swin-T | 112.9 | 768 | 55.3 | 325 |
| SOLIDER Swin-T int8 | 30.2 | 768 | 31.8 | 177 |
| SOLIDER Swin-S | 200.9 | 768 | 113.8 | 654 |
| SOLIDER Swin-S int8 | 54.1 | 768 | 57.3 | 310 |

int8 = onnxruntime dynamic quantisation of MatMul/Gemm. On 32 real crops int8 vs fp32 cosine: CLIP-ReID median
0.976 (min 0.963), Swin-T 0.998, Swin-S 0.999. All candidates are under the 150 ms/crop budget here; fp32 Swin-S
(114 ms) would likely exceed it on gpuserver; OSNet-AIN is slow for its size (InstanceNorm on the onnxruntime CPU provider).

Export bug found and fixed: SOLIDER's Swin computes the batch size with `int(...)` in `window_reverse`, which a
traced ONNX export freezes to the export batch size. The exported model then returns **wrong embeddings for every
other batch size, including 1, without any error** (cosine 0.6 vs PyTorch). `scripts/export_reid_v2.py` patches it
and now checks batch sizes 1, 2 and 3 against PyTorch for every model.

## Gait results

Same clips, analysed like the profiler: 8 fps, up to 40 s, `kinematics.track` with YOLOv8s-pose, silhouettes cut
on the tracked boxes (YOLOv8s-seg-320) and GaitBase on >= 10 of them. 28 household sequences and 7 stranger
sequences got a GaitBase vector (the others had too few clean silhouettes: sitting, cut off, close-ups).
Candidate: a hand-built skeleton signature (percentiles of knee, thigh and shank angles, ankle spread, arm swing,
torso lean and torso:leg ratio from the 17 keypoints, z-scored over the population), the licence-clean stand-in
for a rig-angle signature.

| model, labels | owner pairs | owner cos median | p10 | stranger cos median | max | AUC | 90 % CI | rank-1 | mAP | TPR at 0 FP |
|---|---|---|---|---|---|---|---|---|---|---|
| GaitBase GREW (prod), A | 6 | 0.92 | 0.91 | 0.83 | 0.90 | 1.00 | 1.00-1.00 | 1.00 | 1.00 | 1.00 |
| GaitBase GREW (prod), A+B | 91 | 0.89 | 0.86 | 0.83 | 0.90 | 0.96 | 0.92-0.99 | 1.00 | 0.98 | 0.41 |
| pose-angle signature, A | 6 | 0.06 | -0.17 | 0.04 | 0.61 | 0.52 | 0.28-0.79 | 0.00 | 0.36 | 0.00 |
| pose-angle signature, A+B | 78 | -0.04 | -0.33 | -0.03 | 0.64 | 0.53 | 0.43-0.65 | 0.85 | 0.71 | 0.05 |

- The hand-built skeleton signature is at chance. Not integrated (it stays in the eval script). A fitted-rig version
  would need calibrated cameras and steady walking, which these clips rarely have; 8 fps is also too coarse for
  joint-angle curves.
- GaitBase separates the owner from the stock strangers reasonably (AUC 0.96), but the owner's own distances
  (median 1 - 0.89 = 0.11) sit between `GAIT_SIG_SAME` 0.05 and `GAIT_SIG_DIFF` 0.15, so the explain step mostly
  reports "neutral" for the owner. Thresholds left unchanged, as instructed.
- GaitBase costs ~0.7 s per 60 silhouettes here plus ~30 ms per silhouette frame on gpuserver.

## Recommendation

1. **Do not switch now.** Nothing is measurably better, and a switch costs a re-embedding and new thresholds.
2. **Label first.** Fill in `labels.csv` (below), then rerun `embed` + `report`. Real other people in the same
   lighting turn the different-person column from a stand-in into a measurement.
3. If a re-ID model is to be tried, use **CLIP-ReID ViT-B/16 int8** in **shadow mode** (`reid_shadow_model`): MIT,
   89 MB, ~31 ms/crop here, tied for the best AUC with a usable margin, and its CLIP backbone is the one most often
   reported as robust to viewpoint and outfit change (not verified here). Decide after a few weeks of dual vectors.
   OSNet-AIN is the cheap fallback (same 512 dims, still not comparable). SOLIDER Swin-S has no threshold margin.
4. Gait: keep GaitBase for now, but note its licence conflict (OpenGait has no licence file). There is no licensed,
   in-the-wild-trained replacement among the candidates.

## What the user should label

`~/.cache/squirrel-guard-test/reid-gait/labelling/index.html` (local only, never upload): every sighting's crop,
grouped by the identity the bridge filed it under, with sighting id, camera and time. Scores are deliberately left
out. Copy `labels_template.csv` to `labels.csv` and set `label` per sighting: `owner`, `other-a`, `other-b`, ...
(same tag = same person) or `unsure`. Most useful, in order:
- identities 2-8 (`frontyard`, 11-05 and 11-06 morning) and 13 (`v4`): the only possible in-domain other people;
- identity 1's desk run #47-#74 on `v4` (the face model cannot tell);
- identities 17, 18, 19 (11-06 evening).

## Integration (switched OFF)

- `squirrel_guard/reid_v2.py`: `Embedder` with reid.Embedder's interface (`__call__`, `one`), preprocessing from the
  ONNX metadata; `load_embedder(pc)` and `load_shadow(pc)` for the profiler.
- `squirrel_guard/profiler.py`: builds the sighting embedder with `reid_v2.load_embedder(pc)`. Default
  (`reid_backend` absent or `"osnet"`) = `reid.Embedder(reid_model)`, exactly as before. With `reid_shadow_model`
  set, the sighting message also carries `embedding_v2 = {"model", "vec"}`; the bridge ignores unknown keys.
- `scripts/export_reid_v2.py`: CLIP-ReID (vendored image encoder), SOLIDER (Swin definition from a clone of the
  official repo, batch fix applied) and OSNet-AIN (torchreid file) -> ONNX with metadata, optional `--int8`,
  checked against PyTorch.
- `tests/test_reid_v2.py`: tiny synthetic ONNX graphs; preprocessing, L2, visit averaging, the default staying OSNet.
- `models/clipreid_vitb16_msmt17.int8.onnx` is in the git-ignored `models/` on the laptop only; not on gpuserver.

```toml
[profiler]
reid_model = "/models/osnet_x0_25_msmt17.onnx"
# reid_backend = "osnet"                                     # default
# reid_shadow_model = "/models/clipreid_vitb16_msmt17.int8.onnx"   # step 1: dual vectors
# reid_backend = "v2"                                        # step 3, after migration
# reid_v2_model = "/models/clipreid_vitb16_msmt17.int8.onnx"
```

### Migration (old and new embeddings are NOT comparable)

Different models give vectors of different sizes in unrelated spaces; a dot product between them is meaningless
(and fails outright, 512 vs 1280). `identities.centroid`, `sightings.embedding`, the 0.70/0.55 thresholds,
`explain.py`'s `appearance_prob` and `VERSIONS["reid"]`, and `persons-ui/related.py` all assume OSNet.
1. Dual period: copy the ONNX to gpuserver `model_cache`, copy `reid_v2.py` into `deploy/gpuserver/squirrel-guard/`, set
   `reid_shadow_model`, rebuild. Add `sightings.embedding_v2 real[]` + `embedding_v2_model text` and store them in
   the bridge (not done here). Matching stays on OSNet.
2. With labels and a few weeks of dual vectors, rerun this evaluation on real pairs, pick thresholds.
3. Switch: compute v2 centroids per identity from the stored `embedding_v2` (or re-embed the stored crops, which
   are single crops and score lower than 6-crop visits, see above), swap the bridge to the new column and
   thresholds, update `VERSIONS`, then set `reid_backend = "v2"`. Keep the OSNet columns until the new matching
   has been reviewed.

## Reproduce

```
.venv/bin/python scripts/reid_gait_eval.py fetch      # read-only DB, crops, clips still in Frigate's retention
.venv/bin/python scripts/reid_gait_eval.py crops      # + Pixabay clips in DATA/stock (see DATA/stock/SOURCES.txt)
.venv/bin/python scripts/reid_gait_eval.py embed --model NAME=PATH ...
.venv/bin/python scripts/reid_gait_eval.py gait
.venv/bin/python scripts/reid_gait_eval.py report
.venv/bin/python scripts/reid_gait_eval.py sheet
taskset -c 0-3 .venv/bin/python scripts/reid_gait_eval.py bench --threads 4
```
DATA = `~/.cache/squirrel-guard-test/reid-gait` (person crops, clips and embeddings never go into the repo).
Exports: `.venv/bin/python scripts/export_reid_v2.py clipreid CKPT --int8` and so on (see the script's header);
Google Drive downloads with `gdown` from a scratch environment outside the project venv.

### Sources and checksums (sha256)

Checkpoints:
- `MSMT17_clipreid_ViT-B-16_60.pth` `9974ef57c418e8c88e59581f66f3f53fbe02f879311f7438009f84fd150b1c63`
- `swin_small_msmt17.pth` `5d65814904b9c8110d7e199947f366306432f2a87845e561692dbde875fb3836`
- `swin_tiny_msmt17.pth` `1501eeb2fdc6937da894bef6c45e567c74aed64e6e969252e256e959b53d9509`
- `osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth`
  `8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f`

Code: CLIP-ReID `eb1898b72c882875f478bebfc6d41644eece0a5d`, SOLIDER-REID `8c08e1c3255e8e1e51e006bf189e52cc57b009ed`,
deep-person-reid `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50` (`torchreid/models/osnet_ain.py`).

ONNX files evaluated (re-exports may differ byte-wise):
- `clipreid_vitb16_msmt17.onnx` `9e85c038c9e7a6d6e1e9c4e233efaa64ad30fd6abedb256ef99c37216d1e7cae`
- `clipreid_vitb16_msmt17.int8.onnx` `ee2d9ecfdeaa78b22589ab1a232829c8b3989bd03200abc15e58be938324d939`
- `solider_swin_s_msmt17.onnx` `371ef287c0ca7446e7341cc47e91485ed4abf18943c3c6b7bb82ffc5bad05298`
- `solider_swin_s_msmt17.int8.onnx` `cb6805c49e99f2530e861f85829e00e07738addc85888d99ba1b8d72650c454d`
- `solider_swin_t_msmt17.onnx` `1c03806d8745b6a64a5af627ecfc97ea08322dec480474b4d1027ce2f5c6cdb9`
- `solider_swin_t_msmt17.int8.onnx` `2d73656da954db59ac6a602ce8d78606f65fbc3d129fbec6d33defb9eddff4cd`
- `osnet_ain_x1_0_msmt17.onnx` `a0bbc90c4fad30dea541aa400bc3df5bd8ae7bf05ae18f8a410ad5850df95d29`
- production `osnet_x0_25_msmt17.onnx` `028a66ce99e7c6f8bb0b6c9549731eaf1ca4b6d2ddfa16600027d61ce161a995`,
  `gaitbase_grew.onnx` `b32865b921e8edc122eabecef5310d4071930b29aef8215c28af4c464aa65407`
