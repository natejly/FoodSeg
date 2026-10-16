# Nutrition5k overhead RGB-D benchmark (single-frame path)

Run 2026-10-05; dish surface + per-ingredient density added 2026-10-11. Code: `pipeline/eval/n5k_masks.py`, `pipeline/eval/nutrition5k.py`, `pipeline/eval/density.py`.

```bash
cd pipeline
python -m eval.n5k_masks ../data/datasets/Nutrition5k --out ../results/n5k/masks_v2      # GPU, ~45 min on RTX 2060
python -m eval.nutrition5k ../data/datasets/Nutrition5k --masks ../results/n5k/masks_v2 --out ../results/n5k   # CPU, ~20 min; cached per dish,
                                               # --redo-dish recomputes only the dish-surface volume (~4 min)
```

Data: Nutrition5k `depth_train_ids` (2,754 used) + `depth_test_ids` (507), overhead
`rgb.png` + `depth_raw.png` only (1.9 GB, public GCS bucket, CC BY 4.0) in
`data/datasets/Nutrition5k`. 3 dishes in the split lists have no depth in the bucket.

Masks: Grounding DINO-base ("plate. bowl. food.") boxes, then SAM 2.1-large. Food masks
that overlap a detected plate or bowl with IoU > 0.6 are dropped. This stands in for
`foodscan.segment` (SAM 3). 3.6 % of dishes get an empty mask.

Mass: either one density fitted on train (median mass/volume), or per-ingredient
density (`eval/density.py`): a specific volume per ingredient fitted on train from N5k's
per-ingredient grams, ridge toward the global value (lambda by 5-fold CV on train).
At test time it needs the dish's ingredient names, so those rows are an oracle on
food identity, not on total mass:
"names" = true names, fractions from each ingredient's typical grams on train
(what a VLM naming step could give); "frac" = true names and true mass fractions.
"kcal (oracle)" = predicted mass x the dish's true kcal/g. Test split, 507 dishes.

| estimator | one density | + ingr names | + ingr fractions | kcal (oracle), + names |
|---|---|---|---|---|
| predict train mean | 121.9 g / 61.4 % | — | — | 168.5 / 66.0 % |
| **mask area only (no depth)** | 72.8 g / 36.7 % | 54.0 g / 27.2 % | **51.2 g / 25.8 %** | 70.7 / 27.7 % |
| mask + container (bowl surface / plate ring) | 83.2 g / 41.9 % | 60.5 g / 30.5 % | 59.1 g / 29.8 % | 82.6 / 32.3 % |
| mask + dish surface model, all dishes | 85.5 g / 43.1 % | 70.0 g / 35.3 % | 68.8 g / 34.6 % | 90.4 / 35.4 % |
| mask + ring plane + height map | 98.7 g / 49.7 % | 74.9 g / 37.7 % | 73.9 g / 37.2 % | 103.9 / 40.7 % |
| mask + ring plane + convex hull | 105.4 g / 53.7 % | — | — | — |
| no mask, whole-image RANSAC + height map | 141.8 g / 71.4 % | — | — | — |
| no mask, centre RANSAC + height map | 164.2 g / 82.7 % | — | — | — |
| *published: N5k volume-scalar (learned, RGB + volume)* | *29.4 g / 13.7 %* | — | — | — |

Dish surface model (`dish_volume` in `eval/nutrition5k.py`): counter plane from the
frame border; dishes = regions > 25 mm above the counter (they sit on the scale)
touching the food; inner surface a + b rho^2 + c rho^4 fitted to the uncovered dish
pixels, with the floor pulled toward the lowest visible surface for plates (rim <= 80 mm)
or a 45 mm bowl floor for bowls (rim > 80 mm, where only wall above the food line shows).
Dishes off the scale (rim < 50 mm, a second rig layout; 142 of 3,261) fall back to the
ring plane. "Container" uses this surface for bowls and the ring plane for plates; the
split was chosen on train 5-fold CV (32.1 % vs 36.0 % dish-only, 36.6 % ring-only).

Per dish type, test, + ingredient fractions:

| | n | dish surface | ring plane | area |
|---|---|---|---|---|
| bowl (rim > 80 mm) | 212 | 27.3 % | 37.9 % | 20.9 % |
| plate on scale | 267 | 45.4 % | 35.3 % | 32.5 % |
| off scale / no dish | 28 | 42.0 % | 42.2 % | 33.2 % |

## Reading it

On one overhead frame, the system's plane-based volume is **worse than 2D mask area**.
The failures come from the support surface, not from depth noise:

- **Bowls.** Food inside a bowl is below the rim, so the ring of pixels around the
  mask lies on the rim and the food measures about 0 cm³. Oatmeal: 201 g gives 3 cm³.
  This is the "naive above-plane = −100 %" case `container.py` exists to fix.
- **Bowl or plate included in the mask.** When the dish itself is counted, the volume
  between rim and table is measured as food. A 115 g leafy salad gives 4,980 cm³.
- **Concave plates.** Without masks, the raised rim of a curved plate is counted as food
  (the `center` rows).
- **Low-density foods.** Leaves vary roughly 10× in g/cm³ against rice or meat, so
  one global density cannot fit them. Implied density spans 0.16–1.46 g/cm³ (5th–95th
  percentile).

So for RQ1, this is the single-RGB-D-frame baseline: it fails mostly on support-surface
estimation. That is the gap the multi-view scan + container templates are meant to close.

After the fixes (2026-10-11):

- **Per-ingredient density is the largest single gain**: 9-12 points on every estimator
  (area 36.7 -> 25.8 %, container volume 41.9 -> 29.8 %). Most of it survives with names
  only (27.2 %), so the true fractions add little.
- **A bowl surface fixes bowls**: 37.9 -> 27.3 % (oatmeal: 3 -> 164 cm^3 for 208 g).
  On plates the fitted surface is worse than the ring plane, so the container
  estimator keeps the ring there.
- **Area still beats volume**, on bowls and plates alike. One overhead frame cannot see
  the bowl floor or under-food plate shape, and mask spill onto the rim turns into
  height error; area with a per-ingredient g/cm^2 already encodes each food's typical
  thickness. Volume should pay off with the multi-view scan and container templates,
  not on this frame.

Files: `summary.json` (all metrics), `per_dish.jsonl` (every volume variant per dish),
`test_<estimator>.csv` (per-dish test predictions; `+ingr_names` / `+ingr_frac` suffixes for per-ingredient density), `masks_v2/` (food masks).

## Learned models (2026-10-11/12, local RTX 2060)

All on the official `depth_test_ids` (507 dishes), MAE and MAE as % of the mean, no
oracle names. Configs are chosen on a 10 % val split of `depth_train_ids` (seed 0);
every test number of every run is listed (none dropped).

**Frozen DINOv2-L + small head** (`eval/n5k_features.py`, `eval/n5k_regress.py`; model
per target chosen by 5-fold CV on train):

| features | mass % | kcal % | fat % | carb % | protein % |
|---|---|---|---|---|---|
| geometry only | 32.0 | 46.1 | 66.1 | 54.6 | 66.1 |
| image only | 15.5 | 20.1 | 29.7 | 26.8 | 27.1 |
| image + geometry, no container | 15.4 | 20.1 | 29.7 | 26.7 | 27.1 |
| image + geometry | 14.9 | 19.9 | 29.6 | 26.7 | 27.1 |

**Fine-tuned CNNs** (`eval/n5k_cache.py` -> `eval/n5k_train.py`; 40 epochs, AdamW, flips +
any-angle rotation, no scaling; 4-flip TTA). Channels: `rgb`; `rgbd` = + height above
the counter; `rgbd_thick` = + food height above its container (bowl surface / plate
plane, `vol_container` rule) + food mask. `raw` / `mix` loss = L1 on grams / kcal
(over train mean) instead of on log1p.

| run | backbone | input | loss | mass g / % | kcal / % | fat % | carb % | prot % | val score |
|---|---|---|---|---|---|---|---|---|---|
| rgb | ConvNeXt-T | rgb 448 | log | 24.1 / 12.2 | 39.0 / 15.3 | 25.1 | 21.8 | 24.5 | 12.0 |
| rgbd | ConvNeXt-T | rgbd 448 | log | 22.4 / 11.3 | 38.6 / 15.1 | 24.9 | 21.5 | 24.0 | 12.1 |
| rgbd_thick | ConvNeXt-T | rgbd_thick 448 | log | 21.3 / 10.7 | 38.8 / 15.2 | 25.6 | 21.6 | 23.6 | 11.7 |
| t640 | ConvNeXt-T | rgbd_thick 640 | log | 21.1 / 10.6 | 40.2 / 15.7 | 26.2 | 22.2 | 24.9 | 11.4 |
| t448_e60 | ConvNeXt-T | rgbd_thick 448, 60 ep + EMA | log | 21.3 / 10.7 | 39.5 / 15.4 | 25.1 | 21.1 | 23.3 | 11.2 |
| t448_raw | ConvNeXt-T | rgbd_thick 448 | raw | 21.7 / 10.9 | 39.1 / 15.3 | 25.3 | 22.9 | 23.8 | 11.2 |
| t448_mix | ConvNeXt-T | rgbd_thick 448 | mix | 21.5 / 10.8 | 38.3 / 15.0 | 24.8 | 22.3 | 23.2 | 11.1 |
| s448 | ConvNeXt-S | rgbd_thick 448 | log | 19.9 / 10.0 | 37.1 / 14.5 | 24.4 | 21.4 | 22.3 | 11.2 |
| s448_raw | ConvNeXt-S | rgbd_thick 448 | raw | 20.6 / 10.4 | 38.2 / 15.0 | 23.7 | 20.6 | 21.4 | 10.5 |
| b448_raw | ConvNeXt-B | rgbd_thick 448 | raw | 20.2 / 10.2 | 36.6 / 14.3 | 22.4 | 20.7 | 21.4 | 9.5 |
| vb448_raw | DINOv2 ViT-B | rgbd_thick 448, lr 5e-5 | raw | stopped at epoch 15 (val mass 16-18 % vs ~11 % for ConvNeXt) | | | | | |
| **ensemble s448 + s448_raw + b448_raw** (best on val, 7.2 %) | | | | **19.3 / 9.7** | **36.1 / 14.1** | | | | |

val score = mean of val mass % and kcal % at the kept epoch.

Published, same split: Nutrition5k 2021 29.4 g / 13.7 %, 41.3 kcal / 16.5 %; DPF-Nutrition
21.2 g / 10.6 %, 37.9 / 14.7 %; NuNet 8.7 % / 12.8 % (from docs/model_survey.md; check
the papers before quoting).

Reading: depth helps (rgb 12.2 -> rgbd 11.3 % mass) and container height a little more
(10.7 %); backbone size is the steadiest gain (T 10.7 -> S 10.0 -> B 10.2, ensemble 9.7).
Resolution, EMA/longer and the loss scale are all within noise (~ +-0.5 pt). Val runs
~2 pt easier than test throughout (val dishes share incremental-capture siblings with
train), but the ranking held.

### SAM 3 masks and final models (2026-10-12)

`eval/n5k_masks_sam3.py` (SAM 3, generic prompts "food" / "plate" / "bowl" only): empty
masks 3.6 % -> 0.2 %, no more plate rims in salad masks; median IoU with the old masks
0.97. Cache `data/cache/n5k_448_sam3b`.

| run | backbone | data | loss | mass g / % | kcal / % | fat % | carb % | prot % |
|---|---|---|---|---|---|---|---|---|
| s448_sam3b | ConvNeXt-S | train minus val, SAM 3 masks | log | 19.7 / 9.9 | 37.0 / 14.5 | 24.8 | 20.3 | 22.9 |
| s3_full | ConvNeXt-S | all train, SAM 3 masks | log | 19.0 / 9.6 | 35.2 / 13.8 | 23.6 | 20.2 | 21.4 |
| s3_raw_full | ConvNeXt-S | all train, SAM 3 masks | raw | 19.7 / 9.9 | 36.3 / 14.2 | 22.6 | 20.4 | 20.5 |
| b3_raw_full | ConvNeXt-B | all train, SAM 3 masks | raw | 18.8 / 9.4 | 34.8 / 13.6 | 21.5 | 20.2 | 20.8 |
| **ensemble of the three** | | | | **18.5 / 9.3** | **34.4 / 13.5** | 21.9 | 19.8 | 20.3 |

The three configs were fixed in advance from val (best val ensemble: S + S-raw + B-raw;
SAM 3 masks preferred on val 11.02 vs 11.21), then retrained on all of
`depth_train_ids` for 40 epochs, last epoch kept, scored once.
s448_full (old masks) was stopped at start when val preferred SAM 3 masks.

vs published on this split: Nutrition5k 13.7 % / 16.5 %, DPF-Nutrition 10.6 % / 14.7 %
(beaten on both); NuNet 8.7 % / 12.8 % (not yet: 0.6 / 0.7 pt behind).

### Round 12-13: chasing NuNet on all five targets (2026-10-13/14)

NuNet (Kwan et al., arXiv 2406.01938) reports the same metric (100 x MAE / mean) on the
Nutrition5k RGB-D split: kcal 12.80, mass 8.72, fat 19.67, carb 18.66, protein 18.42 (mean
15.65). IGSMNet (15.0) and NutriVision (13.6) feed ground-truth ingredient strings at test
time (and NutriVision early-stops on test), so they are not comparable to image-only results.

New options in `eval/n5k_train.py`: `--ingr-aux W` (auxiliary head predicting log grams of the
156 ingredients seen in >= 10 train dishes; train labels only, test input unchanged),
`--select all5` (pick the epoch by the mean of all five val %), `--arch-depth` (separate
depth/height/mask backbone, late fusion), `--comp` (predict per-gram densities x mass).
Val = mean of the five val %, ConvNeXt-S, SAM 3 cache:

| run | change | val mean5 | test mass / kcal / fat / carb / protein |
|---|---|---|---|
| s3_sel5 | baseline, 40 ep | 15.07 | 10.0 / 14.6 / 24.4 / 20.4 / 22.7 |
| s3_ingr | + ingredient aux | 14.91 | 9.8 / 13.8 / 24.4 / 20.8 / 21.8 |
| s3_ingr_comp | + composition head | 15.16 (dropped) | 10.5 / 14.8 / 23.9 / 20.9 / 23.2 |
| s3t_dual_ingr | + separate ConvNeXt-T depth encoder | 13.94 | 9.8 / 13.2 / 22.6 / 21.1 / 21.7 |
| s3_ingr_e100 | + 100 epochs | **12.50** | 9.7 / 13.3 / 22.5 / 19.9 / 21.1 |

Finals (all train data, 100 epochs, ingredient aux): s3_ie100_full 10.0 / 13.8 / 22.0 / 20.0 /
22.2; b3_ie100_full 9.6 / 13.3 / 21.1 / 19.6 / 21.1; s3t_dual_ie100_full 9.2 / 13.2 / 22.0 /
20.2 / 21.7.

**Average of all 16 full-data models** (rule fixed in advance, no test-based picking):
**mass 9.0 %, kcal 13.1 %, fat 21.2 %, carb 19.7 %, protein 20.4 % (mean 16.7)** vs NuNet
8.7 / 12.8 / 19.7 / 18.7 / 18.4 (mean 15.65). Still behind on every target; the gap is mostly
fat and protein. Note: NuNet's test set is "the split of [9] minus three test images
duplicated in training", not stated to be `depth_test_ids` (507), so the comparison is close
but not guaranteed identical.
