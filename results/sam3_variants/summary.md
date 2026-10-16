# SAM 3 variants: does a better 2D mask source or 3D-guided prompting raise the lifted-label ceiling?

## Reading (8 lines)

1. No variant moves the ceiling: on the 12 worst fragmented/merged scans (131 items) the reference lift is 0.498 instance mIoU; SAM 3.1 image mode gives 0.495 (d -0.006, CI [-0.018, +0.007]), 3D-guided prompting 0.487 (d -0.009, CI [-0.032, +0.016]), low-threshold union 0.532 (d +0.023, CI [+0.005, +0.048]).
2. SAM 3.1 loads in the image pipeline (its `detector.*` weights are shape-identical to SAM 3's; only the unused 4th neck scale `convs.3` is missing) but its masks are statistically the same as SAM 3's: single-frame hit 0.687 vs 0.682, mean best IoU 0.602 vs 0.598, merged-in-frame 0.436 vs 0.441; lift fragments 31 to 29, merges 75 to 70, no PQ or mIoU change. The 2D errors are not model-size/recipe errors.
3. 3D-guided second pass (box + up to 3 positives + up to 3 negatives per lifted instance per frame, SAM 3 interactive head) cleans the id set but not the item split: false-positive instances 97 to 47, fragmented items 31 to 20, merged 75 to 57, PQ +0.088 (CI [+0.046, +0.135]), but instance mIoU -0.009 and exact count 0.08 to 0.00. A merged lifted id gets one box and no negative inside it, so it stays merged; its single-frame mean best IoU (0.450 vs 0.598) shows the masks are only as good as the first lift. labels.repair_ids changed nothing (co-occurring ids are never merged), so C-norepair is identical.
4. Union of text detections down to 0.15 (D) raises the 2D hit rate (0.682 to 0.754 at IoU 0.3) and the lift on these 12 scans (+0.023, wins in 7 of 12 scans, v13 +0.034 [+0.002, +0.075], v12 +0.013 [-0.007, +0.038]) but does not hold: on 12 held-out scans (the first six per batch not selected, A = 0.769) D is 0.766 (d +0.000, CI [-0.018, +0.017]) and PQ -0.022 (CI [-0.047, -0.003]). Score weighting (D-score) is not better than plain weights on either set. Treat D as noise.
5. Why: the loss is geometric/semantic, not recall. 2D frames already merge touching items in 44 % of frame-items (up to 94 % on the worst scans) and every text source reproduces that, while prompting from a lifted id inherits its merges.
6. What would move it is separating touching same-appearance items inside a single mask (instance-aware 2D, depth/normal splits at prompt time, or the 3D refiner), not a better text detector or box/point prompting from the same lift.
7. Cost per scan (24 frames, RTX 2060 fp16, 2.8 GB peak): SAM 3 text at floor 0.15 about 46 s (SAM 3.1 the same), the 3D-guided pass adds about 2-3 s GPU (about 210 box prompts) plus 2.4 s CPU, replays 3-4 s CPU each; the 12-scan study took 28 GPU minutes in three jobs (SAM 3 + prompts 10, SAM 3.1 9.5, held-out SAM 3 text 8.8).
8. Caveats: 12 scans selected on A's failures (A' re-run reproduces A to 0.001, so no noise confound, but the selection favours any change that perturbs the masks); paired bootstrap over only 12 (6 per batch) scans; no multiple-comparison correction, so the D win on the selected set is already the weak result; C was run with SAM 3 only (the SAM 3.1 checkpoint carries a different interactive neck, `interactive_convs`, that the image-model builder does not load).

Setup: 12 scans (6 hard_v12 + 6 hard_v13) with the most fragmented/merged items in results/lift_errors/scans40/items.csv: v12 s000008 s000010 s000003 s000019 s000001 s000013; v13 s000004 s000017 s000018 s000010 s000015 s000006. A = stored masks_sam3_v2i raw maps replayed through the co-vote linker + cos+qual lift (reproduces items.csv exactly: inst IoU, fragmented, merged per scan). A' = this study's fresh SAM 3 detections through the same replay. B = SAM 3.1 detector (facebook/sam3.1 sam3.1_multiplex.pt, already in the HF cache: 3.3 GB, nothing downloaded) in the SAM 3 image model, same prompts and thresholds (food 0.3, names 0.2/0.3, container/clutter 0.5). C = 3D-guided second pass (see pipeline/eval/sam3_variants.py docstring). D = union of text detections to 0.15, D-score also weights votes by SAM score. Metrics: 2 mm voxel scene, instance mIoU pooled over items, PQ (IoU>0.5 matches), exact count = share of scans with the right number of instances, fragmented / merged GT items with the lift_errors definitions (>20 % of the item in 2+ ids / an id holding >20 % of two items); d = paired per-scan mean inst/PQ difference vs A with a 2000-resample bootstrap over scans. Match/copy/export stages of sam3_synth_votes were not used (they write into data/synth); the replay path is the one that produced the scans40 numbers.

# Selected 12 scans

## Lift metrics (2 mm canonical voxel scene, co-vote linker, same as results/lift_errors/scans40)

### All 12 scans (12 scans)

| variant | inst mIoU | PQ | exact count | fragmented | merged | items | d inst vs A (95% CI) | scans +/-/= | d PQ vs A (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.498 | 0.350 | 0.08 | 31 | 75 | 131 | +0.000 [+0.000, +0.000] | 0/0/12 | +0.000 [+0.000, +0.000] |
| A' | 0.498 | 0.350 | 0.08 | 31 | 75 | 131 | +0.000 [-0.000, +0.000] | 0/0/12 | -0.000 [-0.000, +0.000] |
| B | 0.495 | 0.359 | 0.00 | 29 | 70 | 131 | -0.006 [-0.018, +0.007] | 4/8/0 | +0.002 [-0.022, +0.034] |
| C | 0.487 | 0.439 | 0.00 | 20 | 57 | 131 | -0.009 [-0.032, +0.016] | 4/8/0 | +0.088 [+0.046, +0.135] |
| C-norepair | 0.487 | 0.439 | 0.00 | 20 | 57 | 131 | -0.009 [-0.032, +0.016] | 4/8/0 | +0.088 [+0.046, +0.135] |
| D | 0.532 | 0.361 | 0.08 | 30 | 64 | 131 | +0.023 [+0.005, +0.048] | 7/5/0 | +0.007 [-0.019, +0.047] |
| D-score | 0.512 | 0.363 | 0.17 | 30 | 67 | 131 | +0.005 [-0.007, +0.021] | 4/8/0 | +0.010 [-0.013, +0.045] |

### Batch hard_v12 (6 scans)

| variant | inst mIoU | PQ | exact count | fragmented | merged | items | d inst vs A (95% CI) | scans +/-/= | d PQ vs A (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.524 | 0.383 | 0.00 | 12 | 35 | 69 | +0.000 [+0.000, +0.000] | 0/0/6 | +0.000 [+0.000, +0.000] |
| A' | 0.524 | 0.383 | 0.00 | 12 | 35 | 69 | +0.000 [+0.000, +0.000] | 0/0/6 | +0.000 [+0.000, +0.000] |
| B | 0.514 | 0.387 | 0.00 | 12 | 34 | 69 | -0.007 [-0.018, +0.002] | 3/3/0 | -0.002 [-0.010, +0.005] |
| C | 0.495 | 0.475 | 0.00 | 6 | 25 | 69 | -0.017 [-0.050, +0.018] | 2/4/0 | +0.116 [+0.053, +0.178] |
| C-norepair | 0.495 | 0.475 | 0.00 | 6 | 25 | 69 | -0.017 [-0.050, +0.018] | 2/4/0 | +0.116 [+0.053, +0.178] |
| D | 0.549 | 0.368 | 0.00 | 13 | 31 | 69 | +0.013 [-0.007, +0.038] | 3/3/0 | -0.009 [-0.028, +0.002] |
| D-score | 0.537 | 0.368 | 0.17 | 14 | 30 | 69 | +0.005 [-0.010, +0.020] | 2/4/0 | -0.005 [-0.024, +0.013] |

### Batch hard_v13 (6 scans)

| variant | inst mIoU | PQ | exact count | fragmented | merged | items | d inst vs A (95% CI) | scans +/-/= | d PQ vs A (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.469 | 0.315 | 0.17 | 19 | 40 | 62 | +0.000 [+0.000, +0.000] | 0/0/6 | +0.000 [+0.000, +0.000] |
| A' | 0.469 | 0.315 | 0.17 | 19 | 40 | 62 | +0.000 [-0.000, +0.000] | 0/0/6 | -0.000 [-0.000, +0.000] |
| B | 0.473 | 0.330 | 0.00 | 17 | 36 | 62 | -0.004 [-0.026, +0.018] | 1/5/0 | +0.007 [-0.041, +0.064] |
| C | 0.477 | 0.401 | 0.00 | 14 | 32 | 62 | +0.000 [-0.028, +0.033] | 2/4/0 | +0.060 [+0.018, +0.120] |
| C-norepair | 0.477 | 0.401 | 0.00 | 14 | 32 | 62 | +0.000 [-0.028, +0.033] | 2/4/0 | +0.060 [+0.018, +0.120] |
| D | 0.513 | 0.354 | 0.17 | 17 | 33 | 62 | +0.034 [+0.002, +0.075] | 4/2/0 | +0.023 [-0.024, +0.097] |
| D-score | 0.484 | 0.358 | 0.17 | 16 | 37 | 62 | +0.006 [-0.014, +0.033] | 2/4/0 | +0.025 [-0.015, +0.086] |

## Single frames

### Single-frame item stats (raw per-frame id maps vs gt/seg_ids, items >= 150 px per frame)

| variant | frame-items | hit IoU>=0.3 | hit IoU>=0.5 | mean best IoU | fragmented in frame | merged in frame |
|---|---|---|---|---|---|---|
| A | 3255 | 0.682 | 0.621 | 0.598 | 0.101 | 0.441 |
| A' | 3255 | 0.682 | 0.621 | 0.598 | 0.101 | 0.441 |
| B | 3255 | 0.687 | 0.624 | 0.602 | 0.100 | 0.436 |
| C | 3255 | 0.606 | 0.477 | 0.450 | 0.047 | 0.258 |
| C-norepair | 3255 | 0.606 | 0.477 | 0.450 | 0.047 | 0.258 |
| D | 3255 | 0.754 | 0.678 | 0.648 | 0.123 | 0.389 |
| D-score | 3255 | 0.754 | 0.678 | 0.648 | 0.123 | 0.389 |

## Per scan inst mIoU

| scan | items | A | A' | B | C | C-norepair | D | D-score |
|---|---|---|---|---|---|---|---|---|
| v12/s000008 | 20 | 0.328 | 0.328 | 0.296 | 0.306 | 0.306 | 0.393 | 0.358 |
| v12/s000010 | 8 | 0.253 | 0.254 | 0.236 | 0.200 | 0.200 | 0.235 | 0.238 |
| v12/s000003 | 14 | 0.712 | 0.712 | 0.717 | 0.635 | 0.635 | 0.711 | 0.703 |
| v12/s000019 | 18 | 0.747 | 0.747 | 0.748 | 0.728 | 0.728 | 0.779 | 0.780 |
| v12/s000001 | 5 | 0.552 | 0.552 | 0.550 | 0.563 | 0.563 | 0.550 | 0.545 |
| v12/s000013 | 4 | 0.355 | 0.355 | 0.358 | 0.412 | 0.412 | 0.357 | 0.351 |
| v13/s000004 | 17 | 0.409 | 0.409 | 0.456 | 0.478 | 0.478 | 0.532 | 0.478 |
| v13/s000017 | 8 | 0.303 | 0.303 | 0.282 | 0.278 | 0.278 | 0.302 | 0.298 |
| v13/s000018 | 15 | 0.688 | 0.688 | 0.684 | 0.676 | 0.676 | 0.681 | 0.683 |
| v13/s000010 | 7 | 0.261 | 0.261 | 0.217 | 0.294 | 0.294 | 0.278 | 0.235 |
| v13/s000015 | 6 | 0.296 | 0.296 | 0.296 | 0.285 | 0.285 | 0.301 | 0.301 |
| v13/s000006 | 9 | 0.646 | 0.646 | 0.640 | 0.593 | 0.593 | 0.710 | 0.644 |

## GPU cost (RTX 2060, fp16, 24 frames per scan)

- sam3: mean 46 s/scan over 24 scans (all text detections at floor 0.15 + 3D-guided box/point prompts).
- sam31: mean 46 s/scan over 12 scans (all text detections at floor 0.15).
# Held-out confirmation (first six per batch not in the selected set; SAM 3 text detections only: A, A', D, D-score)

## Lift metrics (2 mm canonical voxel scene, co-vote linker, same as results/lift_errors/scans40)

### All 12 scans (12 scans)

| variant | inst mIoU | PQ | exact count | fragmented | merged | items | d inst vs A (95% CI) | scans +/-/= | d PQ vs A (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.769 | 0.619 | 0.25 | 9 | 12 | 73 | +0.000 [+0.000, +0.000] | 0/0/12 | +0.000 [+0.000, +0.000] |
| A' | 0.769 | 0.619 | 0.25 | 9 | 12 | 73 | +0.000 [-0.000, +0.000] | 2/0/10 | +0.000 [-0.000, +0.000] |
| D | 0.766 | 0.587 | 0.33 | 10 | 12 | 73 | +0.000 [-0.018, +0.017] | 7/5/0 | -0.022 [-0.047, -0.003] |
| D-score | 0.760 | 0.603 | 0.25 | 11 | 14 | 73 | -0.010 [-0.031, +0.009] | 5/7/0 | -0.011 [-0.037, +0.014] |

### Batch hard_v12 (6 scans)

| variant | inst mIoU | PQ | exact count | fragmented | merged | items | d inst vs A (95% CI) | scans +/-/= | d PQ vs A (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.802 | 0.644 | 0.33 | 7 | 8 | 43 | +0.000 [+0.000, +0.000] | 0/0/6 | +0.000 [+0.000, +0.000] |
| A' | 0.802 | 0.644 | 0.33 | 7 | 8 | 43 | +0.000 [+0.000, +0.000] | 0/0/6 | +0.000 [+0.000, +0.000] |
| D | 0.802 | 0.621 | 0.33 | 6 | 8 | 43 | -0.001 [-0.007, +0.005] | 3/3/0 | -0.016 [-0.030, -0.003] |
| D-score | 0.797 | 0.642 | 0.33 | 6 | 8 | 43 | -0.008 [-0.019, +0.002] | 2/4/0 | -0.004 [-0.017, +0.004] |

### Batch hard_v13 (6 scans)

| variant | inst mIoU | PQ | exact count | fragmented | merged | items | d inst vs A (95% CI) | scans +/-/= | d PQ vs A (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.723 | 0.585 | 0.17 | 2 | 4 | 30 | +0.000 [+0.000, +0.000] | 0/0/6 | +0.000 [+0.000, +0.000] |
| A' | 0.723 | 0.585 | 0.17 | 2 | 4 | 30 | +0.000 [-0.000, +0.000] | 2/0/4 | +0.000 [-0.000, +0.000] |
| D | 0.714 | 0.542 | 0.33 | 4 | 4 | 30 | +0.002 [-0.030, +0.032] | 4/2/0 | -0.029 [-0.071, +0.007] |
| D-score | 0.706 | 0.551 | 0.17 | 5 | 6 | 30 | -0.013 [-0.051, +0.025] | 3/3/0 | -0.017 [-0.062, +0.029] |

## Single frames

### Single-frame item stats (raw per-frame id maps vs gt/seg_ids, items >= 150 px per frame)

| variant | frame-items | hit IoU>=0.3 | hit IoU>=0.5 | mean best IoU | fragmented in frame | merged in frame |
|---|---|---|---|---|---|---|
| A | 1833 | 0.865 | 0.785 | 0.748 | 0.089 | 0.255 |
| A' | 1833 | 0.865 | 0.785 | 0.748 | 0.089 | 0.255 |
| D | 1833 | 0.912 | 0.830 | 0.782 | 0.111 | 0.203 |
| D-score | 1833 | 0.912 | 0.830 | 0.782 | 0.111 | 0.203 |

## Per scan inst mIoU

| scan | items | A | A' | D | D-score |
|---|---|---|---|---|---|
| v12/s000000 | 4 | 0.689 | 0.689 | 0.690 | 0.681 |
| v12/s000002 | 10 | 0.804 | 0.804 | 0.816 | 0.812 |
| v12/s000004 | 9 | 0.782 | 0.782 | 0.787 | 0.791 |
| v12/s000005 | 7 | 0.830 | 0.830 | 0.826 | 0.820 |
| v12/s000006 | 8 | 0.813 | 0.813 | 0.803 | 0.801 |
| v12/s000007 | 5 | 0.866 | 0.866 | 0.856 | 0.832 |
| v13/s000000 | 4 | 0.751 | 0.751 | 0.681 | 0.676 |
| v13/s000001 | 2 | 0.775 | 0.775 | 0.776 | 0.706 |
| v13/s000002 | 14 | 0.810 | 0.810 | 0.794 | 0.788 |
| v13/s000003 | 2 | 0.773 | 0.774 | 0.819 | 0.811 |
| v13/s000005 | 5 | 0.599 | 0.599 | 0.600 | 0.602 |
| v13/s000007 | 3 | 0.415 | 0.415 | 0.465 | 0.465 |

## GPU cost (RTX 2060, fp16, 24 frames per scan)

- sam3: mean 46 s/scan over 24 scans (all text detections at floor 0.15 + 3D-guided box/point prompts).
- sam31: mean 46 s/scan over 12 scans (all text detections at floor 0.15).