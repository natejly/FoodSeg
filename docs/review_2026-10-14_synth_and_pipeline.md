# Review: synthetic generator, thesis pipeline, segmentation robustness (2026-10-14)

Five independent read-only reviews by Sonnet agents. Nothing was edited or run. Code paths under
`pipeline/synth/` refer to the `worktree-synth-hard` branch (`.claude/worktrees/synth-hard`); data paths
refer to the main repo `data/synth/`.

---

## 1. Synth generator: appearance and physics realism

Viewed `hard_v8/contact_sheet.jpg` and a full-res `hard_v10` frame.

### (A) Correctness bugs or likely bugs

1. **Per-frame independent LiDAR bias breaks fusion realism.** `lidar.py:111-113` draws a fresh bias field and white noise per frame. Fused points from N frames average the bias down by about sqrt(N). Real iPad bias is temporally correlated per surface/viewpoint and material-dependent. Fix: one scan-level bias field in world/table coordinates (or slowly varying per frame) plus a small per-frame innovation; add per-frame pose error (B2).
2. **Dropout fill is not iPad-like.** `lidar.py:117-124` fills dropped pixels with nearest valid value + 5*sigma0 (about 4 mm). Real ARKit depth there is smeared and biased toward background. Check the model is not learning "conf 0 means ignore" from a too-clean signal.
3. **Dropout ignores specular and wet surfaces.** `lidar.py:114-116` triggers only on `lum < 0.06` and grazing angle. Glossy sauce, yogurt and glazed plates never drop out. Fix: dropout term from a specular pass or `lum > 0.95`, p about 0.3.
4. **MPM syncs the GPU every step.** `mpm.py:397` `vmax = self.g2p(...)` returns a scalar from a Taichi kernel on every one of ~10-12k steps, forcing a device sync. See C1.
5. **Uncertain: depth of field vs fstop 8-16.** `render.py:1003`, `spec_hard.py:430`. Effective blur depends on the sensor size `set_intrinsics_from_K_matrix` produces (36 mm default vs ~5 mm phone). Verify circle of confusion in pixels at the plate edge.

### (B) Realism and domain-gap gaps

1. **LiDAR noise too low and too white.** Defaults sigma0 0.8 mm, 1.2 mm bias at 25 px (`lidar.py:36-42`). ARKit depth at 256x192 is spatially correlated with a few mm error and lumpy 3-8 px structure. Proposed: sample sigma0 per scan 1-4 mm, blur white noise sigma 1.0-1.5 px, two bias scales (6 px and 30 px). Refit from `cal_*` captures.
2. **No camera pose error or drift.** `finalize.py:~85` writes exact `c2w`. Real ARKit VIO error is mm to cm and drifts across the orbit, producing ghost double surfaces at edges. Probably the largest geometric domain gap. Fix: smooth random-walk SE(3) perturbation per scan (0.5-3 mm/step, ~0.1 deg) applied to exported poses only; GT stays in the true frame.
3. **Saturated, unnatural palettes.** hard_v8 tables/plates include magenta, lime, lilac, red; some foods pink (eggs, "meat"). Restrict procedural plate/table HSV to realistic ranges; per-food colour-class priors. (v10 gingham PBR table looks good.)
4. **Food reads as waxy clay.** Smooth matte mounds with no micro-detail, specular break-up, juices or crumbs. Add micro-bump and oil sheen. Grains/chunks via `scatter_pieces` are good.
5. **No translucency in depth.** Rice, yogurt, soup are semi-translucent to 940 nm dToF; measured depth sits deeper. Fix: +1 to +3 mm bias for `puree`/`porridge` kinds (`looks.py:KIND`), keyed by seg mask.
6. **Motion blur and camera noise mild.** `camera.py:58,67`: 0.5-4 px blur at 1920, noise sigma 0.5-3. Widen to 6 px / sigma 5 for a minority of scans. Rolling shutter and ISP local tone mapping not modelled (low priority).
7. **Frame count.** v10 has 8 frames per scan vs hundreds in a real orbit. Fused density/coverage much lower than real. Consider 16-24 frames.

### (C) Speed wins

1. Remove per-step sync in MPM loop (`mpm.py:390-403`): accumulate vmax into a 1-element field in-kernel, read every 200-500 steps; early exit after ~60% when vmax < ~2 mm/s. Expected 20-40% saving.
2. Lower stiffness E 2-4x where moduli are already non-physical (dt scales 1/sqrt(E)). Test on sauce/yogurt first.
3. `mpm.py:383-387` pads to 1.5M particles per scene; upload only `[:n]` slices.
4. Bullet: `render.py:944` substeps 40 / solver iters 30; try 20/20 and compare rest poses; `check_object_interval=0.5`, `min_simulation_time` 1.5 s.
5. Cycles: with denoiser on, 16-24 spp with adaptive threshold ~0.05; skip full-res depth/seg passes when `--keep-hr` is off.

### Already good
Phone ISP chain in `camera.py`; `lidar.py` structure (magnitudes need work); CoACD compound colliders with caching; `scatter_pieces`; PBR tables; HDRI lighting; v10 frame is convincingly photographic apart from B4.

---

## 2. Synth generator: scene composition, assets, diversity

Data: 25 specs from hard_v9/hard_v10 with a `meal` field; `assets_v3.jsonl`; hard_v8 contact sheet. Small sample.

### (A) Correctness bugs

1. **Requested meal scans silently dropped.** 8 of 20 non-random scenes asked for scans and none were placed (mexican plate -> rice + salsa; pancakes -> blueberries only). Scans were 23 of 66 items. `spot()` (`spec_hard.py:318-330`) returns None when nothing fits and `add` is skipped without a log. Piles are placed first (`:381-390`) so garnishes take the space; for big scans `rr` clamps to 0 and collides with the pile. Fix: place scans largest-first, pile last; allow overlap and let physics resolve, or retry smaller spots; log `meal_unfulfilled` into spec and manifest.
2. **N5k toppings lost without a heap.** Toppings only generated in the heap branch (`:260-272`); n5k `heap` is `[]` when the dish has no fill, so `cheese`, `salsa`, `parmesan` (mapped to `TOP`) vanish. Fix: thin MPM puddle/sheet on plate when there is no heap.
3. **`composition.py` and generator disagree.** `"corn": GAP("small pile")` (`composition.py:213`) and `"ham"` reported as gaps but `meals.py:29` has a corn fill. `"cheese"` always -> melted cheese; `eggplant` -> roasted potato; `cherry tomatoes` -> tomato; `noodles` -> pasta scan though a spaghetti fill exists. The 62.6% renderable figure is wrong in both directions and was computed against `assets.jsonl`, not v3. `sample_dish` IS wired (`meals.py:146`) but uses `renderable_only=False` and drops gaps silently.
4. **N5k grams discarded.** Fill fractions from `dirichlet(4)`, pile size `randint(2,6)`. Fix: pile count = grams / asset `mass_g`; heap fractions = grams / density; dish size from total volume.
5. **No mass labels for procedural items.** `_procedural` sets `mass_g=None` (`spec_hard.py:62`): all 31 pieces, 8 toppings, 4 leaf items; primitives also none. Fix: `DENSITY` dict per fill/topping/leaf (~40 entries), `mass_g = volume_ml * density`.
6. Smaller: multiple leafy ingredients keep only the last (`chard`, `bok choy`, `caesar` -> mixed greens); `by_cat`/`cat_of` partly unused; `tiny_item` and `not_exported` not in `DROP` (`manifest.py:22`) contradicting the "<50 points" QC doc; `stacked` ignores leaves on a piece; `leaves_off_dish` in `DROP` biases surviving salads toward tight small piles.

### (B) Distribution gaps

1. **Sparse scenes.** 8 of 25 single-item, 16 of 25 at most 2 items; many nearly empty plates on the contact sheet. Causes: max 3 heap slots, dropped scans, `max_items=8` shared with pile pieces. Real dishes average ~2.7 visible components; PTv3 is only stressed by touching items. Fix: enforce minimum ~35% dish-area fill, redraw or add a scan/pile below that.
2. **Piles tiny.** 2-5 pieces vs 10-40 real (blueberries, fries, olives). Raise cap for same-category piles or split pile ids from the budget (uint8 labels, 253/254 reserved).
3. **Dish variety.** `sample_container` makes only round plates/bowls; 3 of 25 dishes scanned from 18 container assets. Missing: rectangular/square plates, compartment trays, takeout boxes, ramekins, sauce cups.
4. **Template balance.** Cuisine-first gives fruit and dessert (1 template each) the same share as three Western groups with 14 templates.
5. **Colours.** Saturated tables/dishes in hard_v8; weight PBR over procedural saturated colours.
6. **Clutter 70% of scenes** (18 of 25), heavy for lab/home captures. Toppings only on one wedge per heap, never on scans.

### (C) Asset coverage

v3 index: 795 rows, 637 MetaFood3D, 124 usable food categories (4-11 meshes each), only 27 non-food (18 containers, 9 clutter). All meal-template categories resolve. Cheapest wins in order: fix A1/A2 (code only); leafy greens (11% of N5k components) with higher-curvature shorter leaves and multiple greens per scene; procedural ham slices, radish rounds, corn kernels; OmniObject3D/Objaverse food (needs OK for ~19 GB); more dishes (only 4 GSO bowls are deep). Doc drift in `docs/dataset_curation.md`: "135 scanned categories" (index 124), "33 procedural foods" (recount), "617 usable scans" (index 637).

### Already good
Meal-first design with cuisine templates and bowl-only soups; `layout` options; shape/consistency by food type; wedge touching with MPM; manifest difficulty tags and diversity report; leaf and `off_dish` QC; HDRI/PBR realism.

---

## 3. Synth generator: GT export and data contract

Inspected `hard_v10` (16 scans x 8 frames), `hard_v9`, `hard_v7`.

### Already good
Per-scene RNG `[seed, i]` (`batch.py:93`) and post-physics `spec.json` rewrite; depth 256x192 f32 with confidence; `frames.jsonl` intrinsics/transforms match the iPad app layout; ARKit y-up frame, table plane and per-item `pose_arkit` stored; `gt.json` records LiDAR and phone params; `seg_points.npz` keeps per-point `dist`; manifest QC flags; rgb/mask/depth aligned from one render; labels in `seg_points` always item id, 253, 254 or 0.

### (A) Confirmed bugs / contract problems

1. **Non-atomic, mis-ordered finalize** (`finalize.py:136-139`): `gt.json` written before `meta.json`, and `batch.py:88` skips a scene once `gt.json` exists. A crash in between leaves a scan skipped forever with no `meta.json`. Fix: write to `scans/<sid>.tmp`, `gt.json` last, `os.replace`.
2. **Stale-output reuse on retry** (`batch.py:107-112`): `rdir` never cleared before GPU run or CPU retry; `render_meta.json` is the only success test; `finalize` never clears `out`, so old frames survive. Fix: rmtree `rdir`/`odir` per attempt; `render.py` writes `render_meta.json` last with `n_frames`, checked by finalize.
3. **Physics crash kills the batch**: `mpm.settle` (`batch.py:103`) has no try/except. Wrap per-scene steps, log to `failed/`.
4. **Labels include invisible items**: items with `frames_visible == 0` but hundreds of GT points (`hard_v9/s000001` arugula, `hard_v9/s000013` sour cream, `hard_v7/s000007` arugula). Drop from training or set volume target 0.
5. **Missing seed/provenance in `gt.json`**: `--seed`, `--views`, `--res-scale`, `--physics`, preset args, git commit not stored. MPM not bit-reproducible (Taichi).
6. **`seg_export.py:195` recomputes normals** with `estimate_normals()`, discarding TSDF normals. Confirm the real-iPad path does the same or the normal channel differs sim vs real.
7. **Existing `seg_points.npz` are stale**: only `hard` and `hard_v3` have them; hard_v7-v10 have none, so the claimed 0.7 mm median is not reproducible on current data. `manifest.py` flags `not_exported` but `ok` stays true.

### (B) Missing GT / metadata for PTv3, volume, mass

1. **No SAM 3 vote channel can be built.** Only crisp oracle instance masks saved; containers/clutter zeroed (`finalize.py:77`). Fix: save full seg-id map (items + 253 + 254) and either generate pseudo-SAM masks at export (erode/dilate, merge/drop, clutter as false-positive food) writing `vote.npy` (N, K), or better, run real SAM 3 on saved RGB and store its masks.
2. **Mass GT mostly missing**: 95 of 139 items `mass_g: None`. Derive from per-category density with a `mass_source` field.
3. **No per-item GT point index or visible/hidden volume**: store `volume_above_container_ml`, bbox, `n_gt_points`, `occluded_fraction`.
4. **Items tiny and skewed**: 34 of 139 under 5 mL with 20-60 fused points in hard_v3. Add `min_points` gate in `manifest.py` (currently flags <50 only); weight volume loss by volume.
5. **Table label conflated with 0** (`seg_export.py:205`): add GT table samples (label 252) so artefacts (0) are distinct.
6. **Perfect poses and per-frame independent bias** (`lidar.py:333`): see review 1. Both make 0.7 mm optimistic.
7. **Frame count tiny** (8 vs default 24 vs real hundreds). Record `n_views` in `gt.json`.
8. **Neighbour-label ambiguity at contacts**: nearest-GT labelling with 4 mm tol (`seg_export.py:205`) mislabels boundary points. Store second-nearest label or boundary flag to mask in the loss.
9. **`volume_mismatch` too strict and untraceable**: 1% threshold (`manifest.py:69`); `hard_v9/s000010` nugget 42.3 vs 48.8 mL. Store cause and choose the target volume.

### (C) Batch robustness and throughput

Strictly serial (hard_v10: ~140 s render for 8 frames + 33 s MPM); no `--shard i/n` (add with per-scene lock/claim file). Subprocess output discarded unless render fails (stream to `failed/<sid>.log`, write `batch_report.json`). CPU fallback has no timeout. `requirements.txt` lacks `open3d`, versions unpinned. `seg_export` and `evaluate.py` each re-fuse; cache the fused cloud once.

---

## 4. Thesis goals -> benchmarks -> candidate pipelines

Thesis PDF (3 pages, Oct 2): iPad scan -> fused metric 3D -> per-food segmentation -> volume vs plate -> USDA density -> calories. Target energy error <= ~13% (SNAQ). RQ1 scan vs single frame; RQ2 SAM 3 vs Search3D-style 3D open-vocab segmentation; RQ3 SAM 3D completion vs plate closing. Deliverables: code, weighed-meal dataset, demo, poster, report (Dec 17). Progress report due Oct 20.

### Status
| Goal | Status | Evidence |
|---|---|---|
| Capture + reconstruction | Partial | `pipeline/foodscan` (~2.3k lines); one real scan (a box) in `docs/reconstruction.md`; no `data/scans` locally |
| RQ1 | Not started on real data | no paired scan/overhead captures; collection milestone (Oct 4) passed |
| RQ2 | Partial | SAM 3 wired in `foodscan/segment.py`; no Search3D code |
| RQ3 | Not started | no SAM 3D code; MF3D shows +40% bias for one-view plane closing |
| Calibration volume | Not started on real data | `calibrate.py` exists, no results |
| <=13% energy | Met only on 2D proxy | N5k kcal 13.1%; no 3D real number |
| VLM naming + USDA | Partial | `naming.py`, `nutrition.py`; no accuracy numbers |

### Benchmarks (MAE % of mean, N5k `depth_test_ids`, 507 dishes)
| Benchmark | Baseline | Best published | Ours | Gap |
|---|---|---|---|---|
| N5k mass | 13.7% | NuNet 8.72% (IGSMNet 9.4% uses GT ingredient strings) | 8.97% (7-ens) / 9.0% (16-model mean) | +0.25 |
| N5k kcal | 16.5% | NuNet 12.80% | 13.05% / 13.1% | +0.25 |
| N5k fat/carb/protein | - | NuNet 19.7/18.7/18.4 | 21.2/19.7/20.4 | +1.5 to +2 |
| MF3D MFP3D photo split | vol 41.4%, energy 68.1% | PortionNet 17.4/15.4 (protocol unclear) | 28.6% / 50.3% | beat baseline |
| MF3D 16-view carving | - | - | vol 9.0% median, mass 25.1% | density floor 24% |
| Touching-food 3D seg | DBSCAN 0.63 | - | DGCNN offset voting 0.95 | synthetic, geometry only |
| Real weighed iPad scans | SNAQ 12.7% (48 meals) | - | no data | - |

Caveats: NuNet per-target numbers from `results/n5k/README.md`, confirm in the PDF (arXiv 2406.01938); NuNet's test set may differ slightly from `depth_test_ids`; no 2025-26 paper found beating NuNet on N5k; **`EXPERIMENTS.md` rounds 7-11 chose ensemble members and loss variants by reading test numbers**, so quote the pre-registered 16-model average only.

Sources: Nutrition5k arXiv 2103.03375; NuNet 2406.01938; MetaFood3D 2409.01966; MFP3D 2411.10492; implicit-scale benchmark 2602.13041; VolTex 2506.02895; Food-R1 2606.04986; PMC11990770; CVPRW 2025 MLLM decomposition; RDINet doi 10.3390/app16010454.

### Candidate pipelines
- **P1 Scaled 2D RGB-D regression** (current ConvNeXt ensemble). Near saturated; cannot answer RQ1-3; under-predicts heavy dishes; fits 6 GB.
- **P2 Scan -> 3D seg -> volume -> density.** Only path answering RQ1-3. Risks: carving 9% assumed 16 perfect views and clean depth; density floor ~24% mass; N5k shows area + per-ingredient density beats single-view volume (25.8 vs 29.8%); real LiDAR coarse; touching foods unsolved on real data. Needs the synthetic generator.
- **P3 Hybrid**: per-view 2D embedding + per-item height-above-container geometry + mask area -> learned per-item mass. `rgbd_thick` already approximates this (12.2 -> 10.7%). Could fix P2's density floor. Needs synthetic mass labels; sim-to-real risk.
- **P4 VLM / ingredient-aware**: oracle names cut mass error 9-12 pt (area 36.7 -> 27.2%); the biggest single lever, naming accuracy unmeasured.

### Recommendation: P3 as the thesis system (P2 geometry + learned per-item mass, ingredient-aware via P4), P1 kept as the 2D baseline for RQ1.
Queue (2-3 weeks):
1. Real data first (days 1-3): 15-20 weighed meals, orbit + overhead, calibration objects, empty plates per `docs/collection_protocol.md`. Go/no-go: >=15 meals by Oct 20.
2. Calibration-object volume (days 2-4): TSDF vs plane closing vs carving; go if median error <10%, else fix `refine_poses.py` / `depth_upsample.py` first.
3. Real-scan baseline (days 4-7): SAM 3 lift + container volume + N5k-fitted per-ingredient density; same meals single-overhead -> RQ1; kcal MAPE with bootstrap CI.
4. PTv3 refinement (days 7-14): go only if refined segments beat plain SAM 3 lift on real held-out volume error.
5. Learned per-item mass head (days 12-18): only if step 3 shows density dominates.
6. N5k side claim in parallel: per-target-weighted heads for fat/protein; count as beating NuNet only if all five targets (or their mean) beaten by the pre-registered 16-model average.

Fairness: official split, no oracle names/fractions, val-only selection, every test number reported, CIs on real scans.

### Risks in the thesis plan
Claim/evidence mismatch (2D regressor vs scan RQs; no real-scan number with 9 weeks left); 13% target vs SNAQ not comparable and 20-30 scans give multi-point CIs; density floor unbudgeted (thesis calls density "accepted error" yet targets 13%); single-frame volume worse than area on N5k; sim-to-real gap with no published iPad LiDAR noise model; missing ablations (scan vs single frame on same meals, SAM 3D, naming vs density, container template vs plane); N5k test-set leakage in rounds 7-11; SAM 3D (RQ3) at most risk given cluster unavailable.

---

## 5. Segmentation robustness and augmentations

### Current state
- `foodscan/segment.py`: SAM 3 video, one pass per text concept, prompt only at frame 0, forward only. Items appearing later are missed; earlier concepts win overlaps (`lab[m & (lab==0)]`) so results depend on prompt order.
- `foodscan/labels.py:lift_masks`: per-frame voting, visible if within 10 mm of LiDAR depth, >75 deg grazing dropped, cos-weighted; background votes counted; argmax + margin, no spatial smoothing, superpoints or cross-view ID repair. Fallback: cluster nearest orbit look-at (`run_meal.py`). Check whether `run_meal` passes `normals`.
- `eval/mf3d_composite.py`: geometry-only composites, 0.5 mm noise, 4096 pts. DBSCAN mIoU 0.63 full / 0.68 top, per-item volume MAPE 99.7% / 58.6%, exact count 12%; DGCNN offset voting 0.95 / 0.95, MAPE 9.8 / 8.6%, exact count 53 / 48%. Says nothing about sim-to-real.
- `n5k_train.py:107`: flips, any-angle rotation, +-6% shift, global brightness/contrast; never scales (area encodes volume); TTA 4 flips; masks never corrupted. Drop-path 0.2 worse; resolution, EMA, loss scale, seeds flat; regularisation route exhausted. SAM 3 masks cut empty masks 3.6% -> 0.2% for ~0.2 pt.
- `docs/segmentation_robustness.md` is entirely unimplemented (no corruption harness, no labelled scans, no PTv3/Sonata code in main). Its 60-120 GPU-hour estimate assumes an L40S.

### Failure taxonomy -> fixing stage
| Failure | Stage |
|---|---|
| Touching similar-colour items | 3D refiner (geometry/normals) + cross-view ID repair before lift |
| Sauce/topping on food (nested masks) | prompt design + refiner; "topping" class and layered masks in synth |
| Piles | decide evaluation unit (pile vs instance); merge by connected components after lift |
| Shiny/dark/transparent | sim dropout/specular/exposure; refiner channel dropout |
| Container confusion | prompt "plate/bowl", lift as label 254 |
| Clutter | class 253 in refiner, negative class in lift |
| Over-segmentation of textured food | superpoint / radius smoothing of votes |
| Missing depth / holes | confidence gating (`min_conf=2` exists); train with holes |
| Frame-0-only prompting | re-prompt every N frames or bidirectional |
| Prompt order | resolve overlaps by score |

### Augmentation plan
**C1 Synthetic generator (per scene):** refit `LidarParams` from `cal_*` scans, until then jitter sigma0 0.5-1.5 mm, bias 0.5-3 mm at 15-50 px, dropout x0.5-2; add quantisation and edge-flying pixels; ARKit pose drift 1-3 mm / 0.2-0.5 deg random walk with occasional jumps; orbit elevation 30-70 deg, radius 25-45 cm, 180-360 deg coverage, 10-30% frame dropout; appearance (plate colour/gloss, table, light colour, exposure, backlight, shadows); **run real SAM 3 on a few hundred synthetic renders and store its vote shares** rather than inventing a noise model.

**C2 PTv3 training-time:** start from Pointcept indoor recipe (z-rotation, scale [0.9,1.1], jitter, flips, sphere crop) rescaled for tabletop: jitter sigma 0.5-1 mm, **scale no more than +-5%** (scale is volume signal), full z-rotation, tilt a few degrees only; plate-radius crops 0.12-0.25 m; 20-50% cap/hemisphere removal for partial orbits; voxel 1.5-3 mm jittered per epoch. Colour: brightness +-30%, hue +-0.05, WB gain +-10%/channel, greyscale 10%, per-point noise. **Channel dropout**: RGB 15%, DINOv2 20%, vote share 15%, normals 10%, never all at once; a few percent xyz-only. **Vote-share corruption**: flip a 10-40 mm blob to a neighbour p=0.3; merge two touching items' votes in 20% of scenes; drop an item's votes 5%; erode/dilate 1-3 voxels; softmax-style margin noise; permute labels. Sonata: no published fine-tuning recipe found; check Pointcept configs; first run frozen encoder + head, then unfreeze. Ablate channel dropout and vote corruption.

**C3 ConvNeXt N5k:** only untried lever is mask corruption (dilate/erode +-5%, 3% empty/full masks, rim leakage); RGB-only jitter/blur; no cutmix/mixup. Low priority, judge once on test.

**C4 Test time:** z-rotation (8 angles) + flip TTA (small gains, e.g. MemorySeg 70.8 -> 71.5); voxel-offset TTA (3-4 shifts, most useful at tabletop scale); disjoint frame-half lift agreement as a per-point confidence flag; MaskClustering-style view-consensus graph for ID repair; stride/frame-dropout bootstrap for per-item volume CIs.

### Evaluation protocol
Sets: synthetic hard held-out split by seed and by food asset; MF3D composite (sanity only); real iPad scans (20-30). Metrics: mIoU, PQ, boundary IoU at a few mm, per-item volume APE (median/mean), item-count exactness, recall by item type, mass error from segmentation on real scans, flag rate for calibration. Baselines: SAM 3 lift, DGCNN xyz, DGCNN xyz+rgb, **plus PTv3 from scratch** and per-channel ablations. Stress: frame stride 1-16, coverage 90-360 deg, dropout 10-50%, depth noise x1-x4, exposure shifts; corruption grid test-side only. Real labelling: every 30th frame of ~12 scans, hand-corrected 3D labels on ~5, plus a few-click item identity on the fused cloud for all 20-30 (5-10 min/scan).

### Do this first
1. Refit `LidarParams` from `cal_*` scans and add ARKit pose-drift noise (1-2 days once scans exist).
2. Label the real scans (2-3 days).
3. Run SAM 3 on a synthetic subset, store real vote shares, train the refiner on those (1 day + GPU).
4. Fix `segment.py`: score-based overlap, plate/bowl prompts, periodic re-prompting (half day).
5. Train PTv3/Sonata with C2 vs DGCNN and from-scratch, with channel-dropout ablation (2-3 days).
6. Voxel-offset + z-rotation TTA and disjoint-half agreement flag (1 day).
7. Mask-corruption augmentation for N5k (half day, low priority).
8. Corruption/stride harness limited to what the 2060 affords (2 days).

Sources: PTv3 arXiv 2312.10035; Sonata github.com/facebookresearch/sonata; Vu et al. 2311.13152; MemorySeg 2311.01556; MaskClustering 2401.07745; ePointDA 2009.03456; 6IMPOSE 2208.14288; RealD2iff 2511.22505.
