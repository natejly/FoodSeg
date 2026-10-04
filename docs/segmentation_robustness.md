# Segmentation robustness — literature and experiment plan

Compiled 2026-09-28. Verified against primary sources unless marked **[U]**.

## Why this is a defensible chapter
- No ImageNet-C-style corruption study of **SAM 3** exists (SAM 1 has two: arXiv 2306.07713, 2305.06422; SAM 2/3 only under occlusion in OccSAM-Bench, arXiv 2604.11711).
- No multi-view fusion paper reports an ablation against **frame count / stride**.
- No paper quantifies food-specific failures (rice, mash, sauces, peas, glossy); BenchSeg only names "similar colour/texture" and "thin parts".
- SAM 3 is documented as **prompt-sensitive** for semantically equivalent prompts (arXiv 2603.06384, 2609.27076, 2604.17126), and its confidence scores are miscalibrated (HYDRA, arXiv 2609.20283, +9.4 IoU by re-ranking suppressed candidates).

## Metrics (definitions)

Corruption protocol (Hendrycks & Dietterich; Michaelis et al. robust-detection-benchmark):
- 15 corruptions × 5 severities: noise (gaussian, shot, impulse), blur (defocus, glass, motion, zoom), weather (snow, frost, fog, brightness), digital (contrast, elastic, pixelate, JPEG). `pip install imagecorruptions` → `corrupt(img, corruption_name, severity)`. Test-only; don't train on them.
- **mPC** = (1/N_c N_s) Σ_c Σ_s P_{c,s} (severity 0 excluded); **rPC** = mPC / P_clean. Report P_clean, mPC, rPC together. Skip mCE (needs an arbitrary baseline).

SAM 3 own metrics: **cgF1** = 100 · pmF1 · IL_MCC (image, includes presence/absence); **pHOTA** (video). Report mask IoU *and* presence false-positive rate for granularity tests ("food" vs "rice" vs "white rice").

Video consistency (BenchSeg, arXiv 2601.07581):
- Continuity C_γ = fraction of consecutive frame pairs with IoU > γ (γ = 0.5)
- Flicker rate FR_δ = fraction of pairs where IoU drops by > δ (δ = 0.2)
- IoU drift = mean |IoU_t − IoU_{t−1}|; Volatility = std of per-frame IoU
- ID switches (count). HOTA = √(DetA · AssA) is standard but less interpretable for 3–8 items.

3D label consistency (own design, standard practice):
- Re-projection mIoU: render fused 3D labels to held-out labelled frames.
- Point-label agreement (Cohen's κ, majority margin) between fusions from disjoint frame subsets (odd/even, first/second half of orbit).
- Bootstrap over frame subsets (stride 1/2/4/8/16, or resample) → CI on per-item label and volume.

## Known failure modes to stratify by
- Viewpoint change on orbit videos: SAM 2 memory relies on appearance (3AM, arXiv 2601.08831: +15.9 IoU with geometric features); error accumulation in long videos (SAM2Long, 2410.16268).
- SAM 3: temporal hallucination when an object leaves FOV, spatial fragmentation at close-up, misclassification by visual similarity (ENEAS, 2609.03756); text prompts inject "misaligned semantic bias" vs exemplar prompts (2607.09583).
- Food: dense overlap and fine-grained similarity (DishSeg24k, 4.66 instances/img), thin/foreshortened parts, spurious non-food inclusion (BenchSeg), reflections (FoodMem).
- Occlusion by utensils/hands: OccSAM-Bench protocol (synthetic overlays at 3 severities; full/visible/invisible-region scoring) transfers directly.

## Improvement levers to ablate
| Lever | Source | Expectation |
|---|---|---|
| Prompt ensembling (synonyms, granularities) | 2604.17126 found it can *hurt* (drift to generic regions) | test, don't assume |
| Exemplar (box/mask) prompts instead of text after first frame | 2607.09583 | fewer semantic false positives |
| Test-time augmentation (flip/scale, `ttach`) | qubvel/ttach | small, cheap |
| HQ-SAM-style boundary refinement | 2306.01567 | sharper edges → volume |
| Semantic-SAM granularity for touching items | 2307.04767 | splits merged masks |
| SAM 3 → SAM 3.1 Object Multiplex | facebookresearch/sam3 README | crowded-scene accuracy |
| Depth-edge snapping / plane-aware splitting | own | uses the geometry you have |
| Multi-view voting (the pipeline itself) | MaskClustering +8.5 AP, Gaussian Grouping, SAM-V +5 IoU | **the headline claim: fusion raises rPC** |
| Uncertainty for user flagging | ensemble disagreement + vote margin (scores are miscalibrated: HYDRA, CalSAM 2509.23176, ConformalSAM 2507.15803) | flag low-margin items |

## Two-week experiment plan

**Days 1–3, ground truth.** 12–15 scans stratified by difficulty (touching, rice/mash, sauce, small items, glossy plate). Label every 30th frame (~15/scan, ~200 frames) in X-AnyLabeling (SAM 1/2/3 assisted, video propagation, COCO export; github.com/CVHub520/X-AnyLabeling) or CVAT. For 5 scans, hand-correct the fused 3D segmentation and render labels to all frames as dense "3D-consistent GT" (state its bias).

**Conditions.**
- RGB: 8 corruptions (gaussian noise, shot noise, motion blur, defocus, brightness, contrast, JPEG, fog) × severities {1, 3, 5} + clean = 25; fixed seed per frame so the video is consistent.
- Depth (own protocol, no standard exists): Gaussian noise σ ∈ {1, 2, 4} % of depth; edge-biased hole dropout {5, 15, 30} %; frame stride {1, 2, 4, 8, 16}.
- Prompts: 3 granularities × 3 synonyms per item; text vs exemplar after frame 1.
- Occlusion: synthetic utensil overlays at 3 severities (OccSAM style).

**Report per condition.** 2D: frame mIoU, presence FP rate, C_0.5, FR_0.2, drift, volatility, ID switches. 3D: re-projection mIoU, κ between odd/even fusions, per-item volume error. Aggregates: P_clean, mPC, rPC per metric with 1000× bootstrap CIs over scans. Plots: severity curves per corruption family; 2D-vs-3D rPC bars; stride curve; per-food-category breakdown.

**Compute [estimate].** SAM 3 ≈ 2–4 min per 450-frame video with tracker on an L40S. 15 scans × ~40 conditions × 3 prompt sets ≈ 1,800 runs ≈ 60–120 GPU-h → a Slurm job array over a weekend on `gpu`. Week 2: analysis + ablate HQ-SAM / TTA / exemplar prompts on the 3 worst conditions.

## Sources
1903.12261 · 1907.07484 · 1908.05005 · 2306.07713 · 2305.06422 · 2604.11711 · 2609.27076 · 2604.17126 · 2603.06384 · 2607.09583 · 2511.16719 · 2601.07581 · 2601.08831 · 2410.16268 · 2609.03756 · 2009.07736 · 2105.05409 · 2607.23070 · 2407.12121 · 2306.03908 · 2312.11557 · 2401.07745 · 2312.00732 · 2609.25490 · 2604.08916 · 2508.14036 · 2407.01349 · 2306.01567 · 2307.04767 · 2609.20283 · 2509.23176 · 2509.05809 · 2507.15803 · 2604.11585 · github.com/bethgelab/imagecorruptions · github.com/qubvel/ttach · github.com/CVHub520/X-AnyLabeling
