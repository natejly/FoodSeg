# Synthetic RGB-D plated-meal data — options and recipe

Compiled 2026-09-28. Verified against primary sources unless marked **[U]**.

## The gap
No public dataset combines rendered plated food + exact volume labels + LiDAR-like depth.
NutritionVerse-Synth (85k images, Isaac Sim) has mass but no volume; NutriMLLM's 1.1M diffusion
images have neither masks nor depth; nobody has rendered training data from MetaFood3D meshes.
And there is **no published iPhone/ARKit dToF noise model or code**. Building either is a contribution.

## Route A: 3D rendering (recommended)

| Tool | Out of the box | Physics plating | Renderer | License |
|---|---|---|---|---|
| **BlenderProc2** v2.8 | BOP writer: `scene_camera.json` (K), `scene_gt.json` (6DoF), 16-bit mm depth, COCO masks; `set_intrinsics_from_K_matrix` takes the iPad K | `simulate_physics_and_fix_final_poses` | Cycles (OptiX/CUDA only, no Eevee) | GPL-3, maintained |
| Kubric | RGB, depth, inst. seg, flow, normals, camera | PyBullet | Cycles | Apache-2, low maintenance |
| Isaac Sim Replicator 6.1 | rgb, seg, distance, camera params; no native COCO/BOP **[U]** | PhysX | RTX path tracer, needs RTX GPU | Omniverse terms |
| Infinigen | depth/normals/seg/K | constraint solver | Cycles, hours/frame | BSD-3; **no food** |

**Pick BlenderProc2**: only one that gives K + poses + masks + depth + physics plating with near-zero glue.

### Assets
- **MetaFood3D** (743 objs / 131 cats per paper, 637/108 on the download page; Revopoint scans, weight in g, mesh volume; CC BY-NC 4.0, Google Form) — primary, metric scale, real densities.
- NutritionVerse-3D (105 models, Polycam, weight; Kaggle `amytai/nutritionverse-3d`; license **[U]**).
- OmniObject3D food categories (fruit, vegetables, bread, cake, hot dog…, ~10–40 each; CC BY 4.0; real scale).
- Tableware + HDRIs: Poly Haven (CC0; ~24 food models too). Objaverse: no food count, scales unnatural. GSO/ABO/3D-FUTURE: no food.
- Generated meshes (TRELLIS.2 MIT; Hunyuan3D — license excludes EU/UK/KR and forbids using outputs to improve other models; SAM 3D Objects): **not safe as volume ground truth** (hallucinated backsides; PerBite still 33.9 % MAPE after plate-diameter scaling). Use only for appearance/category diversity, tag `generated=true`, ablate.

### Faking iPad LiDAR depth (own model; publishable)
Render clean 1920×1440 depth, then:
1. area-downsample to 256×192;
2. per-pixel shift σ 0.25 px + range-dependent Gaussian (BlenderProc `add_kinect_azure_noise` as template; fit σ to your own iPad-vs-calibration-object captures — Luetzenburg suggests ~1 cm on >10 cm objects, Kim 2023 ~0.5 mm σ / ≤3 mm bias at 30–40 cm);
3. edge bleeding: bilateral/guided upsample toward RGB edges (mimics Apple's RGB-guided fusion);
4. dropout on dark, specular, grazing-angle pixels;
5. 3-level confidence map from normal angle and edge distance.
Keep the clean depth too. Evidence that sensor-simulated depth is needed for transfer: DREDS (2208.03792), DepthSynth, SceneNet.

### Scene randomisation
3–7 items sampled by category co-occurrence, dropped onto randomised plates/bowls on a table; HDRI, key light, plate material/colour, table texture; camera 30–60 cm, 30–80° elevation; iPad K at 1920×1440; 8–12 poses per scene on a hand-held-like arc, poses exported so the fusion pipeline is testable end to end.

### Labels
BOP + COCO writers (K, 6DoF, visible + amodal masks, class) + per-item JSON `{name, fndds_code, volume_ml, mass_g, mesh_id, generated}`; JPEG-95 RGB, 16-bit PNG depth (mm) at both resolutions.

### Cost
Cycles 64–128 spp + OptiX denoise at 1920×1440: ~5–30 s/frame on L40S, 10–60 s on A40 (Blender Open Data: L40S ≈ 2.2× A40 ≈ 8× a 64-core CPU node). One process per GPU in a Slurm array (`--gpus=1 --cpus-per-task=8`, node-local `--temp-dir`, restart periodically: memory leak #1194; 5× slowdowns reported on clusters #950). Storage ≈ 0.5–1 MB/frame with JPEG RGB → 50k frames ≈ 30–60 GB.

**Target:** ~5k scenes × 10 views = 50k frames ≈ 250–500 L40S-hours. 40k for SAM 3 fine-tuning / regressor pretraining, 10k held out with known volumes for the geometric stress test.

## Route B: diffusion

- Image-space ControlNet/T2I-Adapter on rendered depth/seg is a **soft** constraint: silhouettes drift (ControlNet++ 2404.07987); FLUX depth ControlNets recommend strength 0.3–0.7, i.e. weak geometry lock; FLUX.1-Depth-dev is non-commercial. If used, dual depth+seg conditioning and **discard frames whose re-segmented mask IoU < 0.95** vs the render.
- Better: texture the **mesh** (SyncMVD 2311.12891, Paint3D, MVPaint, Hunyuan3D-Paint) then render — geometry, masks and depth stay exact by construction.
- Diffusion data does help segmentation elsewhere (DatasetDM VOC 65→78 mIoU, X-Paste LVIS +2.6 AP, FreeMask ADE20K +3.3, MosaicFusion APr 9.6→15.2); the closest analogue (Blender + constrained diffusion, mushrooms, 2512.08747) reached synth-only F1 0.86 on real. Food-specific diffusion work is classification-only.
- Joint RGB-D generation (LDM3D, JointNet) learns alignment but doesn't enforce it; not suitable for volume labels.

## Sim-to-real evidence
- Pretrain-on-synthetic then fine-tune-on-real beats mixing (1907.07061; Tremblay 2018 +2.1 AP; Hypersim NYUv2 45→52 mIoU). No fixed ratio prescribed; synthetic-heavy mixes underperform.
- NutritionVerse: synth-only kcal MAE 526, synth-pretrain + real-FT 476, real-only 297 → real still wins for regression when real data is decent.
- Park 2021 (Blender food, Mask R-CNN): synth-only 52.2 mask AP on real, +6.4 pp with real FT.
- **SAM 3 + synthetic fine-tuning works**: 68k UE5 renders, zero real images, fruit IoU 0.527 → 0.656 (2607.18576).

## What to report
synth-only / real-only / synth-pretrain + real-FT for (a) SAM 3 food segmentation on your labelled frames, (b) volume error on the 10k held-out synthetic set vs calibration objects. Publish the fitted LiDAR noise model.

## Sources
github.com/DLR-RM/BlenderProc · github.com/google-research/kubric · docs.isaacsim.omniverse.nvidia.com · 2409.01966 · 2304.05619 · github.com/omniobject3d/OmniObject3D · 2312.06192 · github.com/saeejithnair/nvsynth · 2309.07704 · 2606.08948 · 2404.12257 · 2604.12356 · 2107.07191 · 2606.02021 · 2404.07987 · 2311.12891 · 2512.08747 · 2208.03792 · 1907.07061 · 2607.18576 · 2009.07378 · github.com/ankurhanda/simkinect · Luetzenburg 2021 (10.1038/s41598-021-01763-9) · Tondo 2023 (10.3390/s23187832)

## Implementation (`pipeline/synth/`, started 2026-10-11)

Scenes are written in the **FoodScan capture format**, so `fuse`, `run_meal`, `segment`,
`depth_upsample` run on them unchanged; ground truth sits next to it in `gt.json` / `gt/`.

| Stage | Module | Env |
|---|---|---|
| asset index: units, exact volume, mass if known; procedural primitives | `synth/assets.py` | `synth` |
| scene spec: container (lathe bowl/plate), container-conforming fill, items, lights, hand-held orbit (protocol: ~45° ring + top-down), ARKit-like world | `synth/spec.py`, `synth/geometry.py` | `synth` |
| render: physics drop, RGB, z-depth, instance ids (one Blender per scene) | `synth/render.py` | Blender 4.2.1 via BlenderProc 2.8 |
| finalize: iPad LiDAR model (`synth/lidar.py`), FoodScan dirs, GT masks, exact container template | `synth/finalize.py` | `synth` |
| all of the above in a loop / Slurm array | `synth/batch.py`, `scripts/bouchet/synth_render.sbatch` | `synth` |
| volume error vs GT, plane vs container support, oracle masks | `synth/evaluate.py` | `foodseg` (open3d) |

```bash
cd pipeline
conda activate synth     # python 3.11: pip install -r synth/requirements.txt
python -m synth.assets --out ../data/synth/assets.jsonl --metafood3d ../data/datasets/MetaFood3D
python -m synth.batch --assets ../data/synth/assets.jsonl --root ../data/synth/v0 --n 20 --res-scale 0.5
conda activate foodseg
python -m synth.evaluate ../data/synth/v0/scans --out ../results/synth/v0
```

Per scan: `gt.json` (per-item `volume_ml`, `mass_g` when the asset has it, container
parameters + `capacity_ml`, table plane in ARKit coords, LiDAR params), `gt/masks/`
(oracle masks: `run_meal --masks SCAN/gt/masks`), `gt/container.npz` (exact template:
`run_meal --container SCAN/gt/container.npz`), `gt/container/` (dish masks),
`gt/depth_clean/` and optionally `gt/depth_hr/`.

Conventions: Blender world is Z up; `world_to_arkit` swaps to Y up with a random yaw and
offset (an ARKit session origin is arbitrary). Camera axes are the same in Blender and
ARKit (+X right, +Y up, −Z forward). ARKit K puts pixel centres at +0.5, BlenderProc at
integers, so `render.py` subtracts 0.5 from cx, cy.

Caveats / TODO
- iPad K is a placeholder (fx 1440 at 1920×1440); take it from a real `frames.jsonl`.
- `LidarParams` are literature placeholders; fit them on the `cal_*` scans and keep the
  fitted values (they are copied into every `gt.json`).
- v0 procedural food has flat colours (the hard preset adds noise colour + bump); real
  appearance comes from MetaFood3D / OmniObject3D textures. HDRIs (`--hdri-dir`, Poly
  Haven) and table textures are optional.
- No amodal masks yet (needs a per-item render pass).
- Physics: items that leave the dish or are visible in < 3 frames are flagged in `gt.json`
  (`off_container`, `frames_visible`) and excluded by `synth/evaluate.py`.
- Physics runs on a convex-hull proxy (≤ 20k sampled vertices) of each scan, whose pose is
  copied back: Bullet's hull shape walks every vertex per contact query, and with the
  full 0.3-1.6 M-vertex scans one scene did not finish physics in 10 min.

### Real online assets and physics (2026-10-13)

- `synth.fetch_assets` downloads real scans with no account: Poly Haven food (CC0, glTF), YCB
  (CC BY 4.0: cutlery, bowl, plate, mug, plastic fruit) and Google Scanned Objects (CC BY 4.0:
  plates, bowls, ramekins, mugs) into `data/datasets/online/` with `sources.jsonl` (licence,
  attribution). `assets.index_online` indexes them with a `role` (food / container / clutter);
  index with everything: `data/synth/assets_v3.jsonl`. GSO zips keep the texture away from the
  mesh, so the fetcher copies it next to `model.obj` (else dishes render magenta).
- Hard preset: 40 % of scenes use a real scanned dish (those scenes hold only dropped items,
  since procedural heaps are built on the analytic dish surface); clutter is 80 % real YCB
  cutlery / GSO mugs.
- **Compound colliders**: `synth.collision` decomposes each scan with CoACD (decimated to 20k
  faces, <= 12 convex parts, ~7-10 s per asset, cached in `data/synth/collision/`); render.py
  makes the scan a COMPOUND rigid body of hidden convex parts. Items now nest and lean on each
  other instead of resting on one hull, and physics got faster (139 s vs ~520 s for the same
  pile scene). Pile items are poured with random orientation; singles tilt up to ~0.3 rad.
- **Cloth leafy greens**: 35 % of hard scenes get a salad portion (lettuce, spinach, arugula,
  mixed greens, kale, radicchio): 6-16 leaf sheets dropped as one cloth object with
  self-collision onto the settled scene, frozen and given 0.6 mm thickness; volume = the
  thickened sheets. Stable settings came from a sweep (`simulate_cloth` docstring): mass
  3 g / vertex, air damping 0.5, contact distance 0.6 mm, quality 15, 150 frames, leaf tips
  >= 40 % of full width and leaves >= 3 cm (pointed tips or 2 cm leaves put vertices closer
  than the contact distance and the cloth exploded). Leaves that slide over a small bowl's
  rim are caught by `synth.manifest`'s `leaves_off_dish` flag (> 10 % of the salad's surface
  off the dish) and dropped.

### Hard preset: the 3D segmentation stress test (`--preset hard`, `synth/spec_hard.py`)

v0 scenes (and the `eval/mf3d_composite.py` scenes: separate scans on a flat plane, no
colour) are close to solved for 3D instance segmentation (DGCNN mIoU 0.945). Real meals
fail on what they lack, so the hard preset adds, keeping every volume exact:

| Case | How | Exactness |
|---|---|---|
| amorphous foods touching | one heap (`geometry.Mound`) split into 1-3 wedges with wavy boundaries (`split_angles`), each a different fill food with its own mound height; 30 % of splits pick the closest-coloured fills | wedges partition the heap: piece volumes sum to the whole (test) |
| stacked food | toppings (sauce, cheese, pesto…) lying on a piece's top surface, fully inside it (`topping_mesh`) | closed mesh on top of the surface, no overlap |
| same-class piles | 2-5 scans of one MetaFood3D category dropped together | physics; asset volumes |
| clutter | knife / fork / spoon / mug / napkin on the table, label 253 | not food; excluded from masks |
| appearance | noise colour + bump on procedural food (`render.textured_material`) | — |

Plates are 60 %, bowls 40 %; up to 8 food items. Each scan also gets `gt/points.npz`
(about one surface sample per mm² of every object in its final pose, ARKit frame,
labels 1..K / 254 container / 253 clutter), and `synth.seg_export` fuses the scan the way
`run_meal` does and labels each fused point by its nearest sample within 4 mm, giving
`gt/seg_points.npz` (xyz, rgb, normal, label) for training DGCNN / PTv3 on the fused
cloud.

```bash
cd pipeline
conda activate synth
python -m synth.batch --assets ../data/synth/assets.jsonl --root ../data/synth/hard --preset hard \
    --n 20 --res-scale 0.5 --spp 32
conda activate foodseg
python -m synth.seg_export ../data/synth/hard/scans
```

### MetaFood3D as downloaded (checked 2026-10-11)
- `3D_Mesh/<Category>/<object>/<Name>.obj` + `.mtl` + `.jpg` texture; 637 meshes, 108
  categories, 38 GB extracted. Units are **metres**, **Z up**, resting on z ≈ 0
  (Revopoint turntable), so `render.py` imports them with `up_axis="Z"` and spawns them
  upright (yaw + small tilt).
- `complete_dataset_nutrition_v2.xlsx`: Object_name (= category folder), Food_Type
  (= object folder; case/underscores differ, so keys are normalised), FNDDS code,
  Weight (g), kcal, macros, Volume (mL). 635/637 meshes join (no row: Carrot/carrot_8,
  Edamame/new_edamame_4).
- The xlsx `Volume` **is the mesh volume**: on the 133 closed meshes with a row, trimesh
  volume matches it with median difference 0.00 % (4 differ by > 5 %).
- Only 164 meshes are closed (+15 after trimesh hole filling). For open meshes the GT
  volume falls back to the xlsx value (`volume_source: ref`, 376 meshes); 97 with neither
  are `volume_ok: false` and are not used. 522 usable meshes in 100 categories.
- `X(bowl)` categories (Almond, Blueberry, Corn) are the **contents of a bowl with the
  bowl removed**: bowl-shaped underside, bulk density ≈ 0.64 g/mL for almonds. They are
  `container_shaped` and kept out of random placement; TODO fit a bowl to their underside
  and render them in it (the hardest volume case).
- Meshes are 0.3-1.6 M faces; decimating for physics/rendering speed is a later option.
- download.blender.org returns 403 from some networks; fetch Blender 4.2.1 from a mirror
  (ftp.nluug.nl) and check it against `blender-4.2.1.sha256`.
