# FoodSeg / FoodScan: project progress report, 2026-10-17

Nathaniel Ly. Advisor: Alex Wong. Written 2026-10-17 for the whole project to date, with emphasis on the 17 October session. Numbers cite the file they come from. `[unc]` marks anything uncertain or unverified; `[pending]` marks runs that had not finished when this was written. The earlier draft `docs/progress_report_2026-10-20.md` is the source for the segmentation, synthetic-volume and density numbers and is not repeated in full here.

## 1. Summary

FoodScan turns an iPad Pro LiDAR scan (RGB, depth, poses) into a fused metric 3D model, per-item segmentation, volume, then mass and calories via density and USDA lookup. The code for every stage exists. What is missing is real data: `data/meals.csv` and `data/calibration.csv` contain only their header row, so no weighed meal and no calibration object has been captured, and the thesis milestone "start collecting weighed meal scans" (Oct 4-17) is missed.

What changed today (Oct 17): the capture app was installed on the iPad and the first real food scans were taken (a kiwi, 95 g, and a sweet potato, 280 g). On the way I found that ARKit tracking can "run away" (metres of fake straight-line motion while `trackingState` still reads normal), which corrupts scans silently. The app now resets tracking per recording, gates readiness, shows a motion HUD and auto-stops on runaway. The pipeline now detects runaway scans and re-estimates their poses from depth and image features, cutting the end-to-end misalignment on the worst scan from 374 cm to about 1.3-1.8 cm (section 4.2). The Nutrition5k ensemble has now been run on the overhead frame of both real scans and underpredicts the weighed 375 g by about 2.2-2.6x (142 g and 172 g; section 4.6), so image models trained on Nutrition5k do not transfer to iPad scans, which supports the direct volumetric pipeline. The SAM 3 `run_meal` job on the two real scans returned: volumes are close, but the default pile density (0.613 g/mL) gave masses 22-41 % low; with whole-item densities hard-coded the totals are 414 g and 380 g against 375 g (section 4.3). The general density fix is still open.

Other headline numbers, all from existing files: the pre-registered 15-model Nutrition5k ensemble scores 9.51 % mass and 12.82 % kcal on the 507 official test dishes (NuNet 8.72 / 12.80), and the validation-to-test gap (11.2 vs 16.4 mean over five targets) is explained by capture-time sibling leakage in the val split, not by overfitting (section 4.6). On synthetic scans, an orbit beats one overhead frame by 26 pp item error against the envelope volume, and density, not segmentation, is the dominant mass-error source (section 4.5).

## 2. Goals and timeline versus the thesis plan

Source: thesis proposal PDF (Oct 2, 2026), Table 1 and Section 4. Status is my reading of the repo on 2026-10-17.

| Plan window | Milestone | Status | Evidence |
|---|---|---|---|
| Sep 14-19 | Proposal; set up compute cluster | Done / partial | Proposal submitted. Bouchet account works; env `thesis` and torch added 2026-10-17; project dir holds only a skeleton plus results (4.8). |
| Sep 20-Oct 3 | Capture app and 3D reconstruction | Done, hardened today | App on iPad; TSDF fusion, plane/container volume, pose refinement (4.1, 4.2). 10 real scans in `data/scans/`. |
| Oct 4-17 | VLM naming | Partial | Benchmarked on Nutrition5k and FoodSeg103 on the cluster (4.4); not run on real scans yet. |
| Oct 4-17 | Test segmentation approaches | Done on synthetic; real pending | SAM 3 lift validated on 40 synthetic scans; refiner paused (4.3). Real SAM 3 run done (4.3); masks clean, density fix open. |
| Oct 4-17 | Start collecting weighed meal scans | **Missed** | `data/meals.csv` has only a header. Two scans of weighed objects exist (4.2), but they are not part of a protocol session. |
| Oct 18-31 | Volume estimation checked on known-size objects | Not started on real data | `calibrate.py` exists; no calibration scans. Kiwi/sweet potato scans (known weights) are the first candidates. |
| Nov 1-14 | Nutrition lookup, first end-to-end results | Code complete; no real result yet | `run_meal` runs end to end on synthetic input; real run done with hard-coded whole-item densities (4.3); general density fix open. |
| Nov 15-28 | Evaluation and comparisons | Not started | Synthetic RQ1 comparison exists (4.5). |
| Dec 7 / 8 / 17 | Report, poster, submission | Not started | |

Research questions:

- RQ1 (scan versus single frame): synthetic only. Against the envelope target an orbit beats one overhead frame by 26 pp item APE [+20, +34] (`results/rq1/envelope.md`, via the 2026-10-20 draft). Real evidence is the 2D single-frame Nutrition5k baseline only.
- RQ2 (SAM 3 versus open-vocabulary 3D): SAM 3 lift implemented and tested on synthetic scans; no Search3D code.
- RQ3 (SAM 3D completion versus plate closing): started. `pipeline/eval/sam3d_test/` and `scripts/bouchet/sam3d_*.sbatch` (all untracked) install and run SAM 3D Objects on the cluster; `results/sam3d/first_test` on Bouchet was empty at the time of writing, so there are no results `[pending]`.

## 3. Schedule assessment

Segmentation, density and Nutrition5k work ran well ahead of the plan, but all of it is proxy evidence (synthetic scans, 2D benchmark). The critical path now is real data: calibration objects, weighed meals, and one end-to-end real result. Everything in section 6 is ordered by that.

## 4. Workstreams

### 4.1 Capture app

Files: `capture-app/Sources/{ScanRecorder,ContentView,PointCloud,FrameWriter}.swift`; commits 5e8eda6 and 16389bf (both 2026-10-17).

- Installed on the iPad Pro 11" (iPad14,3, 4th gen), iPadOS 26.6.1 (`data/scans/*/meta.json`: `device iPad14,3`, `systemVersion 26.6.1`).
- Black live point cloud fixed: the SceneKit geometry had no material, so it rendered unlit; an explicit `SCNMaterial` fixes it (5e8eda6, `PointCloud.swift`).
- Found ARKit VIO runaway: metres of straight-line motion with `trackingState == normal`. Mitigations in the app:
  - tracking is reset on each recording start, and only normal-tracking frames are kept;
  - readiness gate before record: tracking normal for 1 s, at least 100 feature points, ambient light at least 400, centre depth 0.25-0.8 m;
  - motion HUD: linear speed green up to 15 cm/s, yellow up to 25, red above; angular bands 30 / 60 deg/s;
  - runaway auto-stop (more than 0.6 m/s sustained for 0.3 s writes `trackingFailure` into `meta.json`) and per-frame diagnostics in `frames.jsonl`.
- 16389bf fixed stale speeds: `linearSpeed` and `angularSpeed` were never cleared when a new scan began, so frame 0 and the HUD carried the previous scan's last values. They are now zeroed in `beginRecording`, and the linear bands were tightened to 15 / 25 cm/s.
- The thresholds are empirical `[unc]`: Apple publishes no numeric limits (`docs/research_pose_estimation.md` 1.1).
- Possible OS cause: an Apple forum thread reports directional world-tracking drift on LiDAR devices in iOS 26.4+ with a fix reportedly in 26.6.1, secondhand. Our iPad is on 26.6.1 and still ran away, so the OS fix is not confirmed for this device `[unc]` (`docs/research_pose_estimation.md` 1.2).

### 4.2 Reconstruction, poses and the first real scans

Code: `pipeline/foodscan/{loader,fuse,volume,container,depth_odometry,refine_poses}.py`; commits 533e3d8 and 2ff40b2; eval scripts `pipeline/eval/pose_refine_{eval,colmap,table_eval}.py`.

**Methods added** (`depth_odometry.py`): `geometric`, `colored` and `features` ICP/odometry, each with table masking (`keep`, `drop`, `downweight` the table points). COLMAP is available through `refine_poses.py` (pycolmap pinned to 3.12.6; 4.x breaks the script).

**Metric.** There is no absolute ground truth. The decision (mine) was not to use fiducial boards, so validation is self-consistency: nearest-neighbour distance between the first and last frame's cloud, NN distance between clouds one second apart, and pose jitter. Known-volume objects are planned as the real check.

Results on the runaway scan 20261017_202402 (`results/pose_refine/metrics_methods.md`, `metrics_colmap.md`, `metrics_table.md`):

| Poses | First-vs-last NN (cm) | 1 s-gap NN (cm) | Path (m) |
|---|---|---|---|
| ARKit | 374.3 | 48.2 | 4.15 |
| COLMAP | 49.1 | 8.7 | 1.47 |
| geometric | 9.4 | 2.9 | 2.62 |
| colored (keep) | 3.6 (10.6 in the first table; run-to-run noisy) | 1.8-2.9 | 1.0-1.1 |
| features (keep) | 5.9 | 2.9 | 0.90 |
| features + table drop (2 runs) | 1.8, 1.3 | 2.6 | 0.70, 0.66 |

On healthy scans table drop is worse than ARKit (`metrics_table.md`, features_drop vs ARKit in `metrics_methods.md`):

| Scan | ARKit first-vs-last (cm) | features + drop (cm, 2 runs) |
|---|---|---|
| 20261002_210342 | 0.4 | 5.8, 5.7 |
| 20261017_202343 | 2.1 | 2.9, 2.8 |
| 20261017_203137 | 0.9 | 3.5, 3.3 |

Keeping ARKit rotation (`rot=arkit`) did not help: it was worse on every scan, for example 16.4 cm on 20261017_202343 and 48.5 cm (geometric) on the runaway scan.

**Auto pose source** (`loader.py`, `Scan(poses="auto")`): `arkit_runaway()` flags a scan if `meta.json` has `trackingFailure`, or at least 5 consecutive pose steps exceed 0.6 m/s (hand-held scans peak near 0.45 m/s per the docstring). It classified all 8 scans I checked correctly. Runaway scans get `features + table drop`; healthy scans keep ARKit. Reported by me from the session; the per-scan labels are not stored in a results file `[unc]`.

Caveats: the scan 20261017_203105 has a 40 cm first-vs-last NN for ARKit and every refinement (`metrics_methods.md`) while its 1 s-gap NN is about 1 cm, so the metric is probably not measuring drift there (the orbit may not return to its start) `[unc]`. Two runs of the same colored/features config differ by several cm (colored_drop 8.1 vs 15.3 on the runaway scan), so single-run numbers are noisy. The refinement is a fallback, not an improvement for good scans.

**Research report** (`docs/research_pose_estimation.md`, web research, nothing run): the table dominates the scene and makes depth-ICP slide; recommends a ChArUco board as ground truth and photometric/feature terms as the markerless route; feed-forward models (VGGT, MapAnything, MASt3R-SfM) are untested for mm-level tabletop geometry; a first-order estimate says about 1-1.5 mm inter-frame consistency is needed for 5-7 % volume error, and the report itself calls that estimate unsourced. COLMAP alone is weak on a textureless table (we measured 49 cm). Most claims are tagged `[U]` or `[S]` in the report.

**First real food scans** (`data/scans/20261017_212836_scan`, `..._213105_scan`; previews in `results/scan_previews/`): 292 and 207 frames. 212836 is a plate, 213105 a paper towel, each with a kiwi (95 g) and a sweet potato (280 g). Both are healthy (100 % and 99.5 % green linear motion, 0 flagged frames; from the scan previews, per the session). They need no pose refinement.

### 4.3 Segmentation (SAM 3, lift, CRF, seg3d)

Detailed numbers are in `docs/progress_report_2026-10-20.md` 3(d); summary:

| Item | Result | Source |
|---|---|---|
| Lift quality, 40 real-SAM-3-vote synthetic scans, 273 items | pooled instance mIoU 0.63; co-vote linker +0.047 [+0.022, +0.072] over the old one | `results/sam3_votes/summary.md`, `results/lift_sweep/summary.md` |
| Boundary CRF | pooled +0.001 [-0.002, +0.005]; off | `results/boundary_refine/summary.md` |
| Fragment merge (deployable arms) | all negative (heuristic -0.091) | `results/seg3d/merge/summary.md` |
| PTv3 refiner | semantic mIoU 0.75 vs lift 0.68, instance level or below; paused | `results/seg3d/ablations.md` |
| PTv3 as 4-class semantic cleaner | mIoU 0.831 vs 0.824, CIs overlap | `results/seg3d/semantic/summary.md` |
| SAM 3.1 / 3D-guided prompting | no instance mIoU gain | `results/sam3_variants/summary.md` |

Decision (Oct 16): the learned instance refiner is paused; the co-vote lift is the default.

**Real-scan SAM 3 `run_meal` (Oct 17, done).** `scripts/bouchet/real_meal.sbatch` ran `run_meal --segment --names "kiwi,sweet potato"` on both real scans on Bouchet; outputs and visuals are in `results/real_meals/<scan>/` (`result.json`, `viz/{masks_grid,labels3d,items,summary}.png`, made by `pipeline/eval/real_meal_viz.py`). Each scan yielded exactly two food instances. Volumes are close to truth (sweet potato 271-287 mL, kiwi 104-121 mL; weighed 280 g and 95 g, true volumes about 255-280 and 90 mL `[unc]`: the true volumes are my estimates, not measured).

Before the fix (default density path, `result_default_density.json`): both instances were named "fruit" and classed `kind=pile`, so mass = envelope x 0.613 g/mL (`envelope_bulk`).

| Scan | Item | Volume mL | Density g/mL | Mass g | True g | Error |
|---|---|---|---|---|---|---|
| 212836 | kiwi | 121 | 0.613 | 74 | 95 | -22 % |
| 212836 | sweet potato | 287 | 0.613 | 176 | 280 | -37 % |
| 213105 | kiwi | 104 | 0.613 | 64 | 95 | -33 % |
| 213105 | sweet potato | 271 | 0.613 | 166 | 280 | -41 % |

Diagnosis: the volumes are fine; the error is the density. (1) `SIMPLE_NAMES` in `pipeline/foodscan/segment.py` maps "kiwi" to "fruit" before the SAM 3 prompt, and the instance name is the prompt, so the specific name is gone before the density lookup. (2) The lookup then classes the item `kind=pile` and applies the bulk (loose-pile, air included) density of 0.613 from `data/density/bulk_density.csv`. A whole fruit or tuber has no air between pieces; its density is about 1.0-1.05 g/mL. Earlier runs without masks matched the "kiwi sliced" / "sweet potato baked" rows (`kind=pieces`, 0.761 / 0.845), still sliced-or-cubed bulk values. The bulk table is a pile/chopped-dish table (and the Nutrition5k fitter `pipeline/eval/n5k_density_fit.py` targets the same use).

After (hard-coded example override, `pipeline/eval/real_meal_density.py`: kiwi 1.05, sweet potato 1.0 g/mL applied to the saved volumes; the same volumes, no re-run):

| Scan | Item | Volume mL | Density g/mL | Mass g | True g | Error |
|---|---|---|---|---|---|---|
| 212836 | kiwi | 121 | 1.05 | 127 | 95 | +34 % |
| 212836 | sweet potato | 287 | 1.00 | 287 | 280 | +3 % |
| 213105 | kiwi | 104 | 1.05 | 109 | 95 | +15 % |
| 213105 | sweet potato | 271 | 1.00 | 271 | 280 | -3 % |

Totals: 414 g (212836) and 380 g (213105) against 375 g weighed. The remaining kiwi error is volume (104-121 mL against about 90 mL), not density. Two scans is an anecdote, not a benchmark.

Comparison with the Nutrition5k image models (4.6): they predicted 142-172 g total against 375 g (2.2-2.6x low, dish level only). The volumetric pipeline with a whole-item density is within 1-10 % on total mass, and gives per-item masses. With the default pile density it was 230-250 g total (-33 % to -39 %), still better than the image models but biased low by the density alone.

The general fix is still open: keep the specific user name for the density lookup (do not collapse it through `SIMPLE_NAMES`), and classify single compact connected instances of countable whole foods as a "whole" kind with a whole-object density (about 1.0-1.06 g/mL for fruit and vegetables), while piles keep bulk densities. The override above is hard-coded for these two scans and marked `ponytail:` in the script. `density_table.py` and `naming.py` have uncommitted user edits, so I did not touch them.

### 4.4 Naming (VLM)

`pipeline/foodscan/naming.py` (uncommitted changes add an "atomic" prompt, multi-view input, constrained JSON decoding, and Gemini / OpenAI Responses backends); benchmark `pipeline/eval/naming_bench.py`, `scripts/bouchet/naming_*.sbatch` (all uncommitted). The benchmark scores predicted item names against ground truth by embedding similarity (bge-large) and also kcal/g error.

Results are on Bouchet (`results/naming/*/*/n5k/summary.json`), not in the repo; only example images are local (`results/naming/side_examples/`). Nutrition5k test (507 dishes), items variant, similarity threshold 0.7:

| Model | Prompt | Precision | Recall | F1 | Mass-weighted recall |
|---|---|---|---|---|---|
| Qwen3-VL-8B | base | 0.81 | 0.57 | 0.60 | 0.718 |
| Qwen3-VL-8B | atomic | 0.83 | 0.62 | 0.65 | 0.764 |
| Qwen3-VL-32B-FP8 | atomic | 0.72 | 0.63 | 0.61 | 0.756 |
| Gemini 3 Flash | base | 0.83 | 0.60 | 0.63 | 0.770 |
| Gemini 3 Flash | atomic | 0.81 | 0.66 | 0.66 | 0.820 |
| gpt-6.1-sol | atomic | 0.81 | 0.66 | 0.66 | 0.814 |

Reading: the atomic prompt helps every model I looked at; model choice matters less than prompt (F1 0.57-0.66 across all configs). Adding "hidden" ingredients raises recall but lowers precision. Predicted-kcal/g error was about 0.38-0.50 kcal/g against a true mean of 1.44, versus 0.75 for a global-mean baseline (`summary.json` `cost.kcal_per_g`; the `pct_err` field is infinite in the files, so I report MAE only). Earlier result: with true ingredient names, mass-from-area error falls from 36.7 % to 25.8 % (`results/n5k/README.md`), so naming remains the largest lever. Not yet tested on real plates. Dish-level photos of one's own food are out of distribution for Nutrition5k, so these numbers are an upper bound `[unc]`.

### 4.5 Density, volume and mass

Details in `docs/progress_report_2026-10-20.md` 3(b), 3(f), 3(g); key numbers:

| Finding | Number | Source |
|---|---|---|
| Density floor, 548 MetaFood3D objects | 16-view carving: 9.0 % median volume error but 25.1 % mass error; perfect volume gives 24.0 % | `results/metafood3d/README.md` |
| One overhead frame vs 24-view orbit on hard_v12 (177 items) | item APE 48 % vs 24 % against the envelope; +26 pp [+20, +34] | `results/rq1/envelope.md` |
| USDA bulk-density table on 263 weighed assets | median USDA/real 1.16 (envelope path); IQR 0.77-1.67 | `results/mass_labels/summary.md` |
| Real mass step (envelope x bulk density), 26 items | item APE 44 % vs 29 % with the GT envelope | `results/mass_step/` |

**New today, open issue.** The density lookup (`foodscan/density_table.py`) chose bulk "pieces" densities for the real fruit: 0.761 g/mL for kiwi and 0.845 for sweet potato (`results/real_meals/*/result_*.json`). Those are bulk densities of cut or loose pieces with air gaps; a whole fruit is closer to 1.05 g/mL (my estimate, not from a file `[unc]`). Because mass is computed as envelope volume times this number, mass is underestimated by roughly 20 % for whole items. The table needs a "whole item" kind chosen by default for solid single objects. One uncommitted fix is a guard against an empty query in the matcher (`density_table.py`, 2-line diff).

An uncommitted fitter (`pipeline/eval/n5k_density_fit.py`, `scripts/bouchet/n5k_density_fit.sbatch`) learns per-ingredient effective top-view densities from Nutrition5k as a correction to the USDA table. No results are in the repo `[pending]`.

### 4.6 Nutrition5k models

Files: `results/n5k/README.md`, `EXPERIMENTS.md`, `docs/n5k_preregistration.md`, `results/n5k/phase3/`. 2D overhead RGB-D regression of dish mass and nutrition, metric = MAE as a percentage of the test-set mean, official `depth_test_ids` (507 dishes).

**Rounds 1-14** (README, EXPERIMENTS): backbone size, loss scale, SAM 3 masks, ingredient auxiliary head, a separate depth encoder, 100-150 epochs, per-target weighting (no gain). Rounds 7-11 selected ensembles on test, so only the fixed-rule average is quoted: the 17-model average gives mass 9.0 %, kcal 13.0 %, fat 21.1 %, carb 19.6 %, protein 20.3 %.

**Pre-registered phases** (`docs/n5k_preregistration.md`, written before any test evaluation, 2026-10-17):

- Phase 1 and 1b, validation only: ConvNeXt-XL (log loss) best at val mean5 12.08 at 40 epochs and 11.34 at 80 epochs; SigLIP2-L 14.43, DINOv3-L 15.03, ConvNeXt-XXL 16.18.
- Phase 2: set A = ConvNeXt-XL log (5 seeds) and ConvNeXt-L raw (5 seeds); set B = ConvNeXt-XL, 80 epochs (5 seeds). Primary result = equal-weight ensemble of all 15 models.
- Phase 3: test evaluated once.

| Phase 3 test result (`phase3/metrics.json`) | mass % | kcal % | fat % | carb % | protein % | mean5 |
|---|---|---|---|---|---|---|
| Pre-registered 15-model ensemble | 9.51 | 12.82 | 20.61 | 19.40 | 19.69 | 16.41 |
| NuNet (arXiv:2406.01938) | 8.72 | 12.80 | 19.67 | 18.66 | 18.42 | 15.65 |

The ensemble does not beat NuNet on any target; kcal is 0.02 points behind. NuNet's test set may differ slightly from `depth_test_ids` `[unc]`. The three five-seed groups score 16.63, 16.63 and 16.43 mean5, so seed and config choice are near noise.

**Why val (11.2) and test (16.4) differ** (`phase3/val_test_gap.md`, post-hoc, nothing retrained): Nutrition5k dishes were captured incrementally, so the same plate appears in several dish_ids seconds apart. The random val carve-out from the training split keeps its siblings in train (median gap to nearest train dish 36 s for val, 255 s for test). Reweighting val to test's gap distribution gives 17.4-17.9 vs the test 16.41, so at least 90 % of the gap is sibling leakage. Consequence: val scores are optimistic and the next val split should be grouped by capture time.

**Models on real scans (finished).** Job 29917459 on `gpu_devel` (8:49, `scripts/bouchet/real_n5k.sbatch`) ran the pre-registered Phase 2 ensemble unchanged: 15 full-data models (ConvNeXt-XL log, ConvNeXt-L raw, ConvNeXt-XL 80-epoch, 5 seeds each), equal weight, 4-flip TTA; checkpoints on Bouchet at `results/n5k/phase2/*_full_s*/best.pt`. Input is 448x336 RGB plus height-above-counter, thickness and food-mask channels. `pipeline/eval/real_n5k.py` picks the most overhead frame (6.2 and 9.8 degrees off vertical for the two scans), re-renders depth from a gravity-aligned virtual camera with N5k intrinsics at 0.319 m (to match N5k pixel scale), and masks food with SAM 3. A 0.359 m run checks sensitivity to the assumed height. Outputs: `results/real_alt/n5k/<scan>/prediction.json` and `<scan>_camh359/`; input check in `model_input_panels.png`.

| Scan | Pred @0.319 m | Pred @0.359 m | Pred kcal @0.319 m | Weighed truth |
|---|---|---|---|---|
| paper towel (20261017_213105) | 142 g | 115 g | 248 | 375 g (kiwi 95 + sweet potato 280) |
| plate (20261017_212836) | 172 g | 142 g | 251 | 375 g |

Reading:

- Mass is underpredicted by 2.2x (plate) to 2.6x (paper towel) at the matched pixel scale. Output is dish-level only; there is no per-item mass. No kcal truth was computed for these scans, so I do not report kcal error.
- Per-model spread is small (SD 9-10 g across the 15 models), so the error is a shared bias, not noise.
- The assumed camera height moves the estimate by about 18-19 % (142 to 115 g, 172 to 142 g) and is not known for an iPad scan, so any single number is soft `[unc]`.
- Training dishes average about 199 g, so 375 g is outside the range the models saw, which alone pushes predictions down.
- Domain gap: iPad 256x192 LiDAR depth versus RealSense, no-data strips, bare table, different lighting.
- SAM 3 masks of both items look clean (`model_input_panels.png`), so the input looks reasonable and the error is mostly model and domain, not a bad mask (visual check only).
- Implication: Nutrition5k-trained image models do not transfer to iPad real-meal scans, which supports the direct volumetric pipeline (RQ1 baseline). Two scans with one dish each is an anecdote, not a benchmark.
- SAM 3 therefore also runs in the cluster env (this job used it). The separate `run_meal` SAM 3 job on the real scans has finished (section 4.3).

### 4.7 Synthetic data

Files: `pipeline/synth/` (merged from `worktree-synth-hard`, fe1b24c), `docs/synthetic_data.md`, `docs/dataset_curation.md`, `results/synth/cluster_test/` (one scene, `gt_s000001.json`, four RGB/overlay views; a cluster smoke test of the generator).

Roughly 50 commits, Oct 11-16, build a BlenderProc generator: MPM physics for soft food, real MetaFood3D and Objaverse scans calibrated to photographs, phone ISP and LiDAR noise, per-scene pose drift, QC resampling, diffusion refinement, and per-item volume ground truth. Scale is small (hard_v12: 20 scans, 177 items, 146 valid by item-level QC); the 2,700-scene plan does not exist. Known gaps: clean LiDAR, waxy food, procedural leaf and topping masses off by large factors (`docs/progress_report_2026-10-20.md` 3(c), 3(g)). Synthetic work is paused at the co-vote lift until real scans exist.

### 4.8 Cluster infrastructure

Bouchet (Yale), account `cpsc4900`, SSH via Duo with a 24 h control master (memory note, checked 2026-10-17). Storage: project 4 TiB shared, scratch 10 TiB with a 30-day purge. Submitted to `gpu_devel`, `gpu_rtx6000`, `gpu_b200` successfully; `gpu`, `gpu_h100`, `gpu_h200` untested.

State today: FoodSeg103 complete; the memory note from 2026-10-17 records that the Nutrition5k overhead download had failed in the project dir (a `download_datasets.sbatch` destination bug, uncommitted fix), though the 15-model runs imply the imagery is available somewhere on the cluster `[unc]`; env `thesis` has open3d and torch 2.14.1+cu130; a SAM 3D build/install/weights chain exists (`sam3d_*.sbatch`) but its results are empty. The 15-model Nutrition5k training (Phase 2) ran as array jobs here. Scripts for naming, density fit, real_meal and real_n5k are uncommitted.

## 5. Open problems

1. **No weighed data.** No calibration objects, no weighed meals, so no real volume, mass or kcal error exists. Highest risk to the thesis.
2. **Pose validation has no ground truth.** Self-consistency metrics are noisy and, on 20261017_203105, ambiguous. Known-volume objects will be the first absolute check.
3. **Runaway recurrence.** The app mitigations reduce but do not remove runaway; the cause (OS regression, texture-poor table, motion) is unconfirmed.
4. **Density lookup picks bulk "pieces" densities for whole items** (0.761 kiwi, 0.845 sweet potato vs about 1.05 expected). Fix before any real mass number is quoted.
5. **Density floor.** Even with perfect volume the USDA table gives about 24-26 % mass error (4.5); recalibration needs weighed real meals.
6. **Segmentation of touching items** is limited by SAM 3's 2D masks; unknown on real plates.
7. **Local SAM 3 is unavailable** (6 GB RTX 2060), so `run_meal` run locally without masks falls back to clustering and produces meaningless numbers (for example 0.2 mL total on 212836, 687 mL on 213105; `results/real_meals/*/result_nomask_fallback.json`). A hand-made heuristic mask run (`results/real_meals/_heuristic_local/`) gives 234 g / 108 g (212836) and 235 g / 81 g (213105) for sweet potato / kiwi against weighed 280 g / 95 g, with the low densities of item 4. These are not valid results: the masks are heuristic and unvalidated `[unc]`.
8. **Nutrition5k val split leaks siblings.** Fix before further model selection.
10. **Nutrition5k models do not transfer to real scans.** Dish mass is underpredicted by 2.2-2.6x (142 / 172 g vs 375 g), with a further about 20 % swing from the assumed camera height (4.6). Not a path to a real result without retraining on iPad data.
9. **RQ3.** SAM 3D tests have not produced numbers.

## 6. Next steps

| When | Task | Done when |
|---|---|---|
| Now | Implement the general whole-item density fix (keep specific names, `kind=whole`; 4.3) and re-run the two scans without the hard-coded override | Both scans have a SAM 3 result and a per-item mass compared with 95 g and 280 g |
| Oct 18-20 | Fix whole-item density kind; rerun the two scans | Mass error on the two scans reported, whatever it is |
| Oct 18-31 | Calibration session (soda can 355 mL, box, measuring cup of water, 5 scans each, 5 empty plates) per `docs/collection_protocol.md`; use for the absolute check of poses and volume | Median volume error on known objects reported |
| Oct 18-31 | Weighed meals (target 15 by Oct 20 per the draft plan, 40-60 by mid-November) | Meals logged in `data/meals.csv` |
| Oct 24-31 | Real RQ1: orbit versus one overhead frame on the same meals | kcal MAPE with bootstrap CI |
| Early Nov | Per-kind density recalibration from weighed meals; naming on real plates | Recalibrated density beats raw USDA on held-out meals |
| Early Nov | Commit the uncommitted work (naming bench, density fit, n5k collect/predict, sam3d tests, real_meal/real_n5k scripts); group-by-time val split for Nutrition5k | Clean working tree |
| Nov | SAM 3D comparison (RQ3) only if the first run produces usable output; otherwise state the limitation | Decision by Nov 7 |
| Nov 15-Dec 17 | Evaluation, report (Dec 7), poster (Dec 8), submission (Dec 17) | |

## 7. Index of sources

- Proposal: `Senior_Thesis_FALL2026__Ly__Nate (1).pdf`; protocol `docs/collection_protocol.md`; earlier report `docs/progress_report_2026-10-20.md`.
- Poses: `docs/research_pose_estimation.md`, `results/pose_refine/{metrics,metrics_colmap,metrics_methods,metrics_table}.md`, `pipeline/foodscan/{depth_odometry,loader}.py`.
- Real scans: `results/scan_previews/`, `results/real_meals/`.
- Nutrition5k: `results/n5k/{README,EXPERIMENTS}.md`, `docs/n5k_preregistration.md`, `results/n5k/phase3/{metrics.json,val_test_gap.md}`; on real scans `results/real_alt/n5k/*/prediction.json` (and `*_camh359`), `pipeline/eval/real_n5k.py`.
- Naming: Bouchet `results/naming/*/*/n5k/summary.json`; code `pipeline/eval/naming_bench.py`.
- Git: commits 5e8eda6, 533e3d8, 2ff40b2, 777c5cf, 16389bf (all 2026-10-17); 80 commits in total, first on 2026-10-03.
