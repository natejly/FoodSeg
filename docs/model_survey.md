# Model survey and test matrix — 3D Food Segmentation

Compiled 2026-09-28 from four literature/code sweeps. Everything here was checked
against a primary page (repo README, arXiv, HF card, vendor doc) on that date unless
marked **[U]** (unverified). Numbers move fast; re-check before quoting in the report.

Pipeline stages and where the accuracy is won or lost:

```
capture ─▶ (1) depth + poses ─▶ (2) reconstruction ─▶ (3) 2D masks ─▶ (4) 2D→3D fusion
        ─▶ (5) naming (VLM) ─▶ (6) volume + hidden surfaces ─▶ (7) density + nutrition
                                                                ─▶ (8) learned baselines
```

Hard floors to keep in mind:
- **Sensor.** ARKit depth is RGB-guided upsampling of ~576 dToF points. At 30–40 cm: σ ≈ 0.3–0.5 mm, bias ≤ 3 mm; below 20 cm bias jumps to +26–34 mm (Kim 2023). Edge pixels are interpolated, not measured.
- **Density.** 80 % of 35 foods differed significantly from USDA volume→weight values, −103 % to +39 % (Partridge 2018). Report volume error separately from mass/kcal error.
- **Best comparable systems.** VolE: 2.2–3.1 % volume MAPE with ARKit-scale multi-view mesh (code withheld). SNAQ: 12.7 % energy. Nutrition5k depth pipeline: 16.5 % kcal / 13.7 % mass. NuNet RGB-D regressor: 12.8 % / 8.7 %.

---

## 1. Depth upsampling and pose refinement

| Candidate | Takes metric prior? | Evidence | Cost | License | Link |
|---|---|---|---|---|---|
| **PromptDA** ViT-L (CVPR'25) | yes, native 192×256 ARKit prompt → up to 4K | ARKitScenes L1 13 mm; only model trained on the exact iPad format | 0.5 s/frame at 1920×1440, ~340M params | Apache-2.0 | github.com/DepthAnything/PromptDA |
| Prior Depth Anything (ICLR'26) | yes, any prior pattern | ARKitScenes AbsRel 1.95 % (PromptDA 1.34 %) | 160 ms | Apache-2.0 | github.com/SpatialVision/Prior-Depth-Anything |
| OMNI-DC / Marigold-DC | yes, sparse | AbsRel 2.1–2.2 % | 0.2 s / 4–10 s | BSD-3 / Apache | — |
| Depth Anything 3 | **no prior input** (issues #202, #231) | — | — | mixed | skip for this |
| **COLMAP ≥3.11 / pycolmap** known-poses triangulation + BA | K fixed, ARKit poses as init, Sim3 re-align to ARKit scale | MobileBrick: ARKit 0.46°/1.9 mm → **0.22°/1.5 mm** (arXiv 2608.21008) | CPU, minutes | BSD | colmap.github.io |
| MapAnything (Meta) | K + depth + poses + `is_metric_scale` | 0.16° / ATE 0.01 with all inputs | up to 2000 views, big VRAM | Apache ckpt | github.com/facebookresearch/map-anything |
| Open3D colored ICP + pose graph | init transforms | no ARKit-food numbers **[U]** | CPU | MIT | — |
| VGGT / π³ / MASt3R-SLAM | images only, ≤518 px, Sim3 | — | — | NC | init only, not final poses |

**Pick:** PromptDA for depth; COLMAP known-poses BA for poses; MapAnything as cross-check.

## 2. Surface reconstruction

| Config | Pipeline | Compute | Expected |
|---|---|---|---|
| **A. Baseline TSDF** (done: `pipeline/foodscan/fuse.py`) | ARKit poses + raw depth, conf ≥ high, edge erode, Open3D VoxelBlockGrid 1–3 mm, trunc 4 mm | < 1 min | volume ~5–10 %, surfaces limited by pose drift |
| **B. Refined** | COLMAP BA poses + PromptDA depth → TSDF 0.5–1 mm | ~10 min CPU + 4 min GPU | ~1.5 mm poses, sharper edges |
| **C. Depth-supervised splat** | DN-Splatter (2DGS branch, EdgeAwareLogL1 on LiDAR/PromptDA depth, pose_opt) → `gs-mesh o3dtsdf` | 15–40 min GPU, < 24 GB | best shot at 1–2 mm on textured food; glossy still fails |

RGB-only mesh methods (PGSR, GausSurf, NeuS2, Neuralangelo) ignore the sensor depth you paid for and are mostly non-commercial licences; use one at most as a comparison. MetaFood 2024 winner VolETA was NeuS2 + checkerboard (10.97 % MAPE); nobody in that challenge used sensor depth as the primary signal, which is the gap this thesis fills.

**Evaluation without a lab scanner:** MobileBrick (iPhone LiDAR at 20–50 cm, CAD GT to 0.2 mm, F-score at 2.5/5 mm; 18 eval sequences, github.com/ActiveVisionLab/MobileBrick). Plus SLA-printed or gauge-block reference shapes at 1–5 cm height (FDM prints are ±0.5 mm, too loose). Repeat scans 3–5× for a precision floor.

## 3. 2D segmentation (per frame / video)

| Candidate | Open-vocab quality | Masks | Video tracking | Weights | Notes |
|---|---|---|---|---|---|
| **SAM 3 / SAM 3.1** (Meta, Nov 2025 / Mar 2026) | LVIS 48.5 mask AP; ODinW13 71.8 | yes | yes, text-prompted; 3.1 Object Multiplex ~7× faster with many objects | gated HF `facebook/sam3`, `facebook/sam3.1`; SAM License | 848M params; PyTorch ≥2.7, CUDA ≥12.6; 24 GB comfortable **[U]** |
| MM-Grounding-DINO Swin-T/B + SAM 3 box prompts | LVIS 41.4 minival | via SAM | via SAM | Apache-2.0 (mmdet) | best open-weight GDINO; GDINO 1.6 Pro and DINO-X are API-only |
| YOLOE-v8-L | LVIS 35.9 box / 23.5 mask | yes | no | **AGPL-3.0** | speed baseline only |
| APE-D | LVIS 59.6 box / 53.0 mask | yes | no | Apache-2.0 | heavy EVA-02-L; trained on LVIS |
| FoodSAM / FoodMem | FoodSeg103 46.4 mIoU | yes | FoodMem yes | Apache-2.0 | old mmseg stack; 103 ingredient classes only |
| Semantic-SAM | — | multi-granularity per click | no | no licence | tool for splitting touching foods |

Food-specific SOTA for reference: FoodSeg103 (ingredient level) 55.0 mIoU (LIM-Q + Mask2Former Swin-L, Jul 2026); UECFoodPix Complete (dish level) 74.9 (Swin-TUNA). Ingredient-level is much harder than dish-level; expect the same.

**No food-finetuned SAM 2/3 exists.** That is a publishable gap.

### SAM 3 fine-tune plan
- `README_TRAIN.md` supports image-level text-prompted detection from COCO JSON (category names = prompts); configs `roboflow_v100_full_ft_100_images.yaml`, `odinw_text_only_train.yaml`; 1008 px, batch 1/GPU × 2 GPUs, 20–40 epochs × 1500 samples. Both examples have `enable_segmentation: False`; whether flipping it trains the mask head is **[U]** (issues #217/#304 unanswered). Fallback: LoRA on SAM 2.1 mask decoder.
- Recipe that worked elsewhere (remote sensing, arXiv 2512.15564): freeze vision encoder + DETR, train text encoder + geometry encoder + mask decoder; IoU 70 → 92 with 100–300 objects per class; most gain by FT100.
- Data: DishSeg24k (24k plated images, 112k instances, 278 dish classes, dense overlap) + FoodSeg103 (7.1k images, 103 ingredient classes, Apache-2.0, HF `EduardoPacheco/FoodSeg103`) + ~300 of your own frames; 10–20 % negative prompts.
- Compute: 2 × 48–80 GB GPUs, ~3–6 h **[estimate]**. On Bouchet: `--gpus=l40s:2` or `a40:2`.
- Eval: BenchSeg (Jan 2026, 55 dish scenes, 25k frames, 360° motion, flicker rate + IoU drift; CC BY-NC-ND) and held-out own frames.

## 4. 2D → 3D mask fusion

| Method | Principle | Code | Effort |
|---|---|---|---|
| **Tracker-first vote** (SAM3D Yang 2023 scheme) | SAM 3.1 video IDs → unproject → per-point majority vote weighted by view angle/confidence → DBSCAN cleanup | github.com/Pointcept/SegmentAnything3D (MIT), ScanNet-only scripts | ~200–300 lines, reimplement |
| **MaskClustering** (CVPR'24) | per-frame masks, view-consensus graph, iterative clustering | github.com/PKU-EPIC/MaskClustering (no licence); 3–15 min/scene | moderate, training-free |
| CDIS (Jul 2026, arXiv 2607.17778) | depth-warp masks, IoU > 0.8 match over 5-frame queue, superpoint merge | no code | easy to reimplement; fixes ID breaks |
| 3AM (Jan 2026) | SAM 2 + MUSt3R features, +15.9 IoU under viewpoint change | — | relevant for orbit videos |
| Gaussian Grouping / SAGA / OmniSeg3D / PanopticRecon | per-scene 3DGS with ID features | various | heavy; only if config C is adopted |

Metrics: render 3D labels back to held-out frames → 2D mIoU / mask AP; point-level AP on ~20 hand-labelled scans; BenchSeg flicker/drift.

## 5. Naming (VLM)

Caveat: DiningBench "classification" is 8-way multiple choice with same-menu distractors, not open naming. The 1→2 view jump is where multi-view helps; plateau at ≥3; small models get *worse* with more views on nutrition. No CoT for naming (hurts perception); CoT only for nutrient arithmetic.

| Model | DiningBench cls / nutrition MAPE | Other | VRAM | License |
|---|---|---|---|---|
| **Gemini 3 Flash / Pro** | 81.8 / 25.2 · 81.6 / 24.5 | best Nutrition5k zero-shot kcal (80.7 MAE), ingredient Jaccard 0.65 | API | — |
| GPT-5 | 70.2 / 32.2 | OmniFood cooking-method 79 % | API | — |
| **Qwen3-VL-30B-A3B-Instruct** | 65.4 / 37.4 | best open | 62 GB bf16 / 19 GB 4-bit (inference) | Apache-2.0 |
| **Qwen3-VL-8B-Instruct** | 64.2 / 39.2 (≈ GPT-4o) | best ingredient match on OmniFood-Bench | 18 GB bf16; LoRA ≈ 20 GB | Apache-2.0 |
| InternVL3.5-38B / 8B | 54.2 / 45.3 | — | — | Apache-2.0 |
| Gemma 3 12B | 48.6 | — | — | Gemma |
| Claude Sonnet 4.5 | 54.4 / 42.6 | VQA 83.6 | API | — |
| Qwen3.5-27B, Gemma 4, GLM-4.6V, Molmo 2, Llama 4 | **no food numbers** | — | — | open |

Food-specialised: **Food-R1** (Jun 2026; Qwen3-VL-8B, SFT → CoT → GRPO; Nutrition5k 27.6 kcal MAE as text, but N5k in training — verify split; Apache-2.0, github.com/hustvl/Food-R1, HF `zy12123/Food-R1`). FoodLMM (LISA-7B, 67.3 kcal via regression heads). NutriMLLM (Qwen3-VL LoRA on 1.1M synthetic, code promised). Geometry-Enhanced Portion Estimation (Gemini + DINOv2 head: per-food mass MAE 28.1 → 16.7 g, no code).

**Pick:** Gemini 3 Flash (closed), Qwen3-VL-8B (open, fine-tunable), Qwen3-VL-30B-A3B (open, inference). Crop each tracked item at 2 views, ask for JSON `{name, ingredients[]}`, canonicalise to FNDDS (BM25 + embedding + LLM rerank; FoodSEM-style).

### VLM SFT plan
- Model Qwen3-VL-8B-Instruct; LoRA r 16 / α 32 on all LLM linear layers; vision tower + merger frozen epoch 1; lr 1e-4 cosine; 2–3 epochs; bs 1 × GA 8–16; 0.5–0.8 Mpx/image; **2 images per sample**; loss on assistant tokens only; target JSON `{"items":[{"name","ingredients":[...],"mass_g","kcal"}]}` with plain digits.
- Framework: LLaMA-Factory (`qwen3_vl` template, sharegpt + images) or ms-swift (packing; supports `qwen3_vl_moe`). Needs transformers ≥ 4.57, qwen-vl-utils ≥ 0.0.14. Unsloth lacks InternVL3.5/GLM/Kimi/MiMo.
- Data (~60–80k): Nutrition5k via FoodDialogues frames (overhead + one side frame; ingredients + mass + kcal), FoodSeg103 names, Food-101 / FoodX-251 / UECFood256 (dish names, research use), MM-Food-100K (ingredients + weights, NC), OmniFood8K if labels check out. Hold out DiningBench, OmniFood-Bench, Nutrition5k `depth_test_ids`.
- Compute **[estimate]**: ~20–24 GB VRAM; ~1–1.5 h per 10k two-image samples on H100, 2–3× on L40S; 3 epochs × 70k ≈ 25–35 H100-h. GRPO phase only if kcal MAE plateaus (Food-R1 gained ~9 kcal).

## 6. Volume and hidden surfaces

Geometric estimators (implement all three, compare on calibration objects):
1. **Height-map integration** above the RANSAC plate plane on a 1–2 mm grid (Nutrition5k, Lu et al.) — primary.
2. **Plane-closed Poisson mesh** (FoodRiddle) — watertight, handles overhangs.
3. Convex hull — include only as the "what LiDARCalorieCam did" negative control (64 % MAPE on fried chicken).

Completion models for the underside (RQ3):

| Model | Conditioning | Scale | VRAM | License | Notes |
|---|---|---|---|---|---|
| **SAM 3D Objects** | image + mask + **optional pointmap** (`InferencePipelinePointMap`, `pointmap=`) | layout in pointmap frame; internal normalisation | ≥ 32 GB doc; 16 GB slow; 24 GB OOMs in mesh decode | SAM License, gated | issues #162/#174 report pointmap misalignment/NaNs → keep ICP + scale fit as safety net |
| **Hunyuan3D-Omni** | image + **partial point cloud** (trained with random-drop) | input normalised to [−0.98, 0.98]³, invert stored centre/scale | 10 GB | Hunyuan Omni Community (excluded EU/UK/KR) | partial-cloud bbox smaller than true → residual scale bias; align visible surface to fix |
| MV-SAM3D | multi-view img + mask + DA3 depth | as SAM 3D | — | MIT | early-stage |
| Test-time partial-observation guidance (arXiv 2609.10531) | SAM 3D + RGB-D partial view | metric via ray-consistent occupancy; GSO-30 Chamfer 9.6 → 1.0 | — | no code | reimplement if time |
| GenPC (CVPR'25) | zero-shot point completion, re-aligns to input | input frame | — | MIT | control for learned completion |
| AdaPoinTr / PointSea / P2C | supervised point completion | input frame after un-normalising | — | MIT | fine-tune on MetaFood3D renders (~750 meshes, unpublished) |

Food precedent: PerBite (MetaFood 2026 winner) scales Hunyuan3D/SAM 3D meshes by plate diameter → 33.9 % MAPE; nobody has used point conditioning on food.

## 7. Density and nutrition

- **FNDDS 2021-23 Portions & Weights**: 5,432 foods, 3,688 with a "cup" row → density = g / (cups × 236.59 mL); public domain (USDA warns it's approximate). Primary table.
- **FAO/INFOODS Density DB v2.0**: 638 entries, 20 groups, xlsx; backfill.
- **MetaFood3D** weight + mesh volume → ~743 empirical densities for validation (CC BY-NC, request form).
- FoodData Central API: `/foods/search` with `dataType=Survey (FNDDS)`, 1,000 req/h, CC0. Python: `usda-fdc`.
- Name → FNDDS code: BM25 + BGE-M3 + LLM rerank (Open-KNEAD, Nutri-ATLAS RR@5 80.7 %); FoodSEM LoRA Llama-3-8B F1 0.83 (code available).

## 8. Learned baselines to compare against the geometric pipeline

| Method | Input | Nutrition5k kcal / mass | Code |
|---|---|---|---|
| Nutrition5k volume-scalar | overhead RGB-D | 41.3 kcal (16.5 %) / 29.4 g (13.7 %) | reference only |
| DPF-Nutrition | RGB + DPT depth | 37.9 (14.7 %) / 21.2 (10.6 %) | none |
| **NuNet** (TMM'25) | real RGB-D dual Swin-B | 12.8 % / 8.7 % | none; reproduce |
| **MFP3D** | point cloud + RGB (CurveNet + ResNet50) | MetaFood3D 41 % vol | github.com/jingema99/MFP3D |
| **PortionNet** (Jan 2026) | PointNet → RGB distillation | MetaFood3D 17.4 % vol / 15.4 % kcal | github.com/darrinbright/PortionNet (no weights) |
| Food-R1 | RGB, VLM text output | 27.6 kcal (split **[U]**) | released |

Anything you report on Nutrition5k must use the official `depth_test_ids` (507 dishes) or `rgb_test_ids` (709) split.

---

## Test matrix

Fix the capture set (calibration objects + N weighed meals), then vary one stage at a time.

| Stage | Baseline | Alternative 1 | Alternative 2 |
|---|---|---|---|
| Geometry | A: TSDF on ARKit poses + raw depth | B: COLMAP BA + PromptDA + TSDF | C: DN-Splatter mesh |
| Input regime (RQ1) | single RGB (VLM direct) | single RGB-D frame (height map) | full scan |
| 2D masks (RQ2) | SAM 3.1 text prompts | MM-GDINO boxes → SAM 3 | SAM 3 food-finetuned |
| Fusion (RQ2) | tracker-first vote | MaskClustering | — |
| Naming | Gemini 3 Flash | Qwen3-VL-8B zero-shot | Qwen3-VL-8B SFT |
| Volume | height map | plane-closed Poisson | convex hull (control) |
| Hidden surface (RQ3) | plane closing | SAM 3D + pointmap | Hunyuan3D-Omni partial cloud |
| End-to-end learned | — | MFP3D / PortionNet fine-tuned on MetaFood3D | NuNet-style RGB-D on Nutrition5k |

Primary metrics: per-item volume MAPE (calibration objects and food), per-item mass MAPE, per-meal kcal MAE and MAE-as-%-of-mean, Bland–Altman for mass; segmentation 2D mIoU of re-projected 3D labels; reconstruction F-score at 2.5/5 mm on MobileBrick.

## Compute budget on Bouchet (estimates)

| Job | GPU | Time |
|---|---|---|
| SAM 3.1 video on 450 frames × 40 scans | L40S/A40 48 GB | ~1–3 min/scan |
| PromptDA on 450 frames | any | ~4 min/scan |
| COLMAP BA | CPU (`day`) | ~10 min/scan |
| DN-Splatter | L40S | 15–40 min/scan |
| SAM 3D Objects per item | A40/L40S (≥ 32 GB) | seconds/item |
| Hunyuan3D-Omni per item | any 16 GB+ | seconds/item |
| SAM 3 fine-tune | 2 × L40S | 3–6 h |
| Qwen3-VL-8B LoRA SFT | 1 × H100 (or L40S, 2–3× slower) | 25–35 H100-h for 3 epochs × 70k; ~4 h for a 10k pilot |
| MFP3D / PortionNet fine-tune | 1 × A40 | hours |
| Qwen3-VL-30B-A3B inference (vLLM) | 1 × H100 bf16 or L40S FP8 | — |

Weights to request now (all gated click-through): `facebook/sam3`, `facebook/sam3.1`, SAM 3D Objects checkpoints, MetaFood3D (Google Form), DishSeg24k, BenchSeg.

## Priority order given the Dec 17 deadline

1. Geometry A + calibration objects + height-map volume (this is the thesis core; everything else is comparison).
2. SAM 3.1 text prompts + tracker-first vote + Gemini naming → first end-to-end number.
3. Geometry B (COLMAP BA + PromptDA). Cheap, well-evidenced, likely the biggest accuracy gain.
4. RQ1 baselines (single RGB, single RGB-D) — same scans, near-zero extra capture.
5. SAM 3D + pointmap vs plane closing (RQ3).
6. One SFT: pick **either** SAM 3 on DishSeg24k/FoodSeg103 (novel, directly improves your pipeline) **or** Qwen3-VL-8B naming SFT (well-trodden). SAM 3 is the better thesis contribution; the VLM SFT is the safer one.
7. Geometry C and learned regressors only if 1–6 are done by mid-November.
