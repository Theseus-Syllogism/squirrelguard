# Scene objects from one frame: PixARMesh vs TripoSR (evaluation, 2026-11-06)

Goal: give the replay viewer an "Objects" layer (furniture as meshes placed in the replay world) built from the
empty basement frame. Evaluated offline on the laptop (RTX 4070 8 GB, ~3.2 GB free next to llama-server).
**Result: not good enough to ship. Do not integrate PixARMesh.** Details and the suggested schema are below, so a
better source (hand-placed CC0 assets) can drop in later.

## Scripts (evaluation only, own environment outside the project)

| script | does |
|---|---|
| `scripts/scene_objects_pixarmesh.py FRAME REPLAY.json --ceiling 2.5` | undistort, depth, masks, PixARMesh per object -> `OUT/objects/*.glb` + `OUT/objects.json` |
| `scripts/scene_objects_triposr.py FRAME REPLAY.json --objects 2,0,3 --mc-resolution 160` | same masks, TripoSR on the masked crop -> `OUT/triposr/*.glb` + `OUT/triposr.json` |
| `scripts/scene_objects_render.py OUT/objects.json OUT/triposr.json` | 8-view turntables per object (with a 1 m cube) + a scene sheet (floor, 2.5 m ceiling, camera) |

OUT defaults to `~/.cache/squirrel-guard-test/scene-objects/` (not in git). Environment: `~/src/pixarmesh/.venv`
(Python 3.12, torch 2.6.0+cu124; their Docker image and recommended torch 2.10 need CUDA 12.8 = driver >= 570, the
laptop has 550). Repos in `~/src/pixarmesh/{PixARMesh,EdgeRunner,TripoSR}`; `meto` is put on the path with a `.pth`
file (its setup.py builds only the C++ extension, not the Python package). No flash-attn: the script swaps in
torch SDPA at runtime (their loader hard-codes `flash_attention_2`, and their fallback attention materialises the
full matrix and runs out of memory), casts the model to bf16 on the CPU before moving it to the GPU, and casts the
cond features in `masked_scatter_`. Their repo is not modified.

## Pipeline and coordinate transform

1. Undistort the frame with the replay camera's K/D (same K kept).
2. Depth: Depth Anything V2 **Metric-Indoor** Small (their pipeline uses Depth Pro). It put the ceiling (median
   world Z of the top-centre band) at **3.80 m**; the real ceiling is ~2.5 m, so `--ceiling 2.5` multiplies the
   depth by **0.557**. The scale is therefore an assumption, anchored only on the ceiling height.
3. Masks: Grounding DINO tiny boxes + SAM ViT-B (= their Grounded-SAM), prompt
   `chair. desk. table. computer monitor. shelf. lamp. coat. cat tree.`
4. Per object: a full-height 4:3 crop around the box, resized to 648x484 (their render size), K scaled.
   Back-projected points go OpenCV camera -> their Y-up camera (`diag(-1,-1,1)`) -> gravity-rectified frame,
   then are normalised to +-0.95 as their loader does. The model predicts an 8-corner box, then the mesh.
5. Back to the replay world: `p_world = C + A^T p_rect`, where the rows of A are, in world coordinates, left =
   up x fwd, up = (0,0,1), and fwd = the optical axis R[:,2] with Z zeroed and normalised. The replay R is
   world_from_camera, as in viewer.js; this camera looks about 25 deg *up*. Metric = normalised / scale + centre.
6. `.glb` files are glTF Y-up, metres, origin = centre of the object's world bounding box; world = rotate +90 deg
   about X (glTF (x,y,z) -> world (x,-z,y)).

## Results (empty.png, --ceiling 2.5)

The detector found 6 objects, but **most labels are wrong**:

| idx | Grounding DINO label | actual object |
|---|---|---|
| 0 | computer monitor 0.73 | monitor (correct) |
| 1 | chair 0.43 | cat tree |
| 2 | chair 0.39 | chair at the left edge (correct) |
| 3 | cat tree 0.38 | hanging coat |
| 4 | lamp 0.34 | bulb |
| 5 | shelf 0.32 | cardboard box |

The masks themselves were clean (`debug/masks.jpg`).

**PixARMesh** (sequential per object, bf16, peak torch VRAM **3158 MiB**; depth 6 s, masks 5 s, model load 3 s):

| obj | what it produced | size (m, world x/y/z) | faces | time |
|---|---|---|---|---|
| 0 monitor | a clean **dining chair**, wrong class | 0.73 x 0.74 x 0.87 | 616 | 47 s |
| 1, 2 | out of GPU memory (mesh sequence > ~12k tokens) | - | - | ~4 min each, wasted |
| 3 coat | a tall flat cabinet/bookcase | 1.47 x 2.54 x 1.38 | 34 | 4 s |
| 4 bulb | a big staircase-like block, near the ceiling | 1.30 x 2.12 x 1.92 | 280 | 28 s |
| 5 box | shelf/cabinet, 4.9 m away | 1.04 x 0.60 x 1.78 | 2000 | 217 s |

PixARMesh produces tidy, artist-like low-poly meshes, but they are 3D-FRONT furniture that does not match the
object: the monitor came out as a chair and the coat as a 2.5 m cabinet. Sizes are implausible (the coat and
bulb are about room-height), and the scene sheet shows them overlapping the ceiling plane. This is expected from
the domain gap: training was synthetic 3D-FRONT rooms with level, eye-height cameras, and this is a real
ceiling-heavy basement seen from 0.87 m looking 25 deg up.

**TripoSR** (`--mc-resolution 160`; 256 ran out of memory in marching cubes; peak **2253 MiB**, load 6 s, about
1 s per object): vertex-coloured blobs with roughly the right silhouette (the monitor is a dark slab; the coat is
a shell open at the back). They have no pose and no metric scale: they are sized from the object's own depth
points. Chair 0.46 m tall: too short, partly cut by the frame edge. Monitor 0.35 m tall: plausible. Coat 0.58 m
tall: too short, because it is foreshortened close to the camera. 40-60k faces each.

Renders: `~/.cache/squirrel-guard-test/scene-objects/objects_0{0..3}.png`, `objects_scene.png`,
`triposr_0{0..2}.png`, `triposr_scene.png`, `debug/masks.jpg`, `debug/depth.png`.

## Recommendation

Neither route gives a usable Objects layer automatically. PixARMesh has the right output (posed, separate,
compact meshes), but on this camera it outputs the wrong furniture at the wrong size, and the larger objects
don't fit in the free VRAM. TripoSR is fast and roughly shape-faithful, but gives low-quality blobs with no pose.
For a handful of static objects in one room, place CC0 assets (Poly Haven / Poly Pizza chair, monitor, shelf) by
hand. Use the depth points (`debug/depth.npy`) or the existing splat to read off positions, and store them with
the schema below. Revisit PixARMesh only with a level, eye-height camera and >= 8 GB free VRAM.

## Suggested replay schema for a later "Objects" layer

```json
"assets": { "objects": [
  { "file": "objects/chair.glb", "label": "chair",
    "position": [x, y, z],          // world metres (Z up), centre of the object's bounding box
    "rotation": [qx, qy, qz, qw],   // world-frame quaternion, applied after the glTF Y-up -> Z-up fix
    "scale": [1, 1, 1],
    "source": "pixarmesh|triposr|cc0:<url>", "licence": "CC BY-SA 4.0 | MIT | CC0" } ] }
```

Viewer: GLTFLoader, `obj.rotation.x = Math.PI/2` (Y-up -> Z-up), then the quaternion, position and scale; add it
to the `world` group under an "Objects" toggle. The script's `objects.json` already has this shape, plus
diagnostics (`size_m`, `box_world`, `faces`, `tokens`, `seconds`).

## Licences

- **PixARMesh** code and weights (`zx1239856/PixARMesh-EdgeRunner`): CC BY-SA 4.0. Attribution if any mesh is
  shared: "Meshes generated with PixARMesh (Zhang, Yoo, Wu, Li, Xie, Tu, CVPR 2026,
  github.com/mlpc-ucsd/PixARMesh), CC BY-SA 4.0." Adapted material must stay CC BY-SA 4.0.
- **EdgeRunner / meto** (NVIDIA): NVIDIA Source Code License, section 3.3: use only "non-commercially and with
  NVIDIA Processors", where non-commercially means "for research or evaluation purposes only". This offline
  evaluation fits; permanent integration into squirrel-guard would not, which is one more reason not to adopt
  it. `meto/setup.py` also carries an "all rights reserved, no use without a licence agreement" header; the
  `license.md` shipped in the same folder is taken as the grant.
- **Training domain**: 3D-FRONT / 3D-FUTURE (Alibaba, synthetic indoor scenes, research licence). Outputs are
  biased toward that furniture catalogue, as seen above.
- TripoSR: MIT (Tripo AI + Stability AI). Depth Anything V2 Small, Grounding DINO tiny and SAM ViT-B: Apache-2.0.
