# Nutrition5k: pre-registration of Phase 2 configs and the ensemble rule

Written 2026-10-17 16:58 EDT, before any Phase 2 or Phase 1b result exists and before any evaluation on `depth_test_ids`.
No test image, prediction or score has been produced or inspected in Phase 1, Phase 1b or Phase 2. `--no-test` is set on every run;
runs that train on all 2,755 dishes (`--full --no-test`) do no test inference at all.

## Phase 1 validation results (train 2,480 / val 275, 40 epochs, bs 16, 448x336, SAM 3 cache, 2 seeds)

Val % MAE (best epoch, 4-flip TTA), mean over mass, kcal, fat, carb, protein:

| config | val mean5 | note |
|---|---|---|
| convnext_xlarge.fb_in22k_ft_in1k, log loss | 12.08 +- 0.02 | mass 7.22, kcal 11.48 |
| convnext_large.fb_in22k_ft_in1k, raw loss | 12.64 | |
| convnext_large.fb_in22k_ft_in1k, log loss | 12.77 | |
| convnext_base.fb_in22k_ft_in1k, log loss | 13.57 | |
| SigLIP2-L (lr 5e-5, head 5e-4, llrd 0.8) | 14.43 | |
| DINOv3-L (same settings) | 15.03 | |

## Phase 2 set A (fixed now)

Both configs trained with `--full` (all 2,755 depth-train dishes, no val split, last epoch kept), seeds 0-4, every other setting as in Phase 1:

- A1: `convnext_xlarge.fb_in22k_ft_in1k`, `--loss log`
- A2: `convnext_large.fb_in22k_ft_in1k`, `--loss raw`

10 models. Config file: `scripts/bouchet/n5k_phase2.txt`.

## Set B

Phase 1b (validation only: ConvNeXt-XXL, XL at 640x480, XL at 80 epochs, ViT lr retune) may add at most 2 more configs (set B),
chosen only by val mean5 (and a val-split ensemble check), before any test evaluation. Set B is appended to this file, dated and committed, before test is touched.

## Primary results

- **Primary ensemble:** the equal-weight average of the per-dish predictions of all Phase 2 full-data models (set A plus set B, all seeds).
- **Primary single-model result:** the config with the best val mean5, reported as mean +- sd over its 5 full-data seeds.
- **Metrics:** MAE as a percentage of the test-set mean, for mass, kcal, fat, carb, protein, plus the mass+kcal mean and the mean over all five, on the 507 `depth_test_ids` dishes.
- **Bar:** NuNet (arXiv:2406.01938) 8.72 mass / 12.80 kcal / 19.67 fat / 18.66 carb / 18.42 protein (mean5 15.65), and the rule "beat all five targets or their mean", both taken from `docs/progress_report_2026-10-20.md`.

## Test protocol

Test is evaluated exactly once, after set B is decided and all Phase 2 models are trained, with `eval.n5k_predict --split test --i-am-phase3` and only with the user's approval. Nothing is retuned after that evaluation.
