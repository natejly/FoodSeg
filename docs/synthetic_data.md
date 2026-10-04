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
