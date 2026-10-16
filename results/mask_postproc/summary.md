# 2D mask post-processing of stored SAM 3 raw maps (CPU replay, 11 hard_v12 scans)

Code: `pipeline/foodscan/segment.py` (`PostOpts`, `postprocess_raw`, `refine_frame(multi_name_cover=)`, CLI flags, all default off), `pipeline/foodscan/replay_lift.py` (`--post JSON`), tests `pipeline/tests/test_mask_postproc.py`.
Sweep `sweep.py`, tables/bootstrap `report.py`, raw numbers + every decision record `results.json`, extra configs `extra.json`/`extra2.json`.
Path per config: raw maps -> `postprocess_raw` -> co-vote linker (cluster 0.8 unless stated) -> `lift_for_meal` (cos+qual) -> seg3d metrics. Baseline reproduces 0.7406 inst mIoU / PQ 0.593 / 8 fragmented / 6 merged (primary error types of `eval/lift_errors.classify`).
CI = paired bootstrap over the 11 scans (10000 resamples, same resample for config and baseline), diff vs baseline. `+` in the CI column would mean the CI excludes 0 positively; none does, so no default changed.
exact-count = fraction of scans whose predicted instance count equals GT.
Decision precision (GT `seg_ids`, diagnostics only): split correct = the two largest parts have different GT food ids, each part >= 50 % pure; merge correct = both masks majority the same GT food id; drop correct = the dropped mask is majority non-food or < 50 % pure.
Context: 303 of 2574 raw food masks (12 %) hold >= 2 GT food items at > 20 % each (`mixed.json`).

Rules and parameters (the spec'd ones are the first row of each section; others are neighbours added because the spec'd value was poor):
- A2 depth: 4-neighbour graph in the mask at 256x192, cut at |dz| > dz_cut or normal angle > ang_cut, split when >= 2 components each hold > part_frac (0.2) of the mask; small bits/holes go to the nearest part. Default dz 8 mm / 40 deg.
- A3 colour: cv2 k-means (k=2) in Lab; accept if centroid dist > 12, each cluster's largest CC > 80 % of it, and (my addition) each cluster >= 15 % of the mask.
- B1: same frame, dilate 2 px touches, same top name in name_px (or both "food"), union area / hull > 0.8, Lab mean dist < 8; greedy by hull ratio.
- B2: drop by frame score / area share / area-to-hull ratio before linking.

## A1 multi-name split
Not replayable: the stored raw maps are the output of `refine_frame` (name masks are not stored per pixel); in `name_px` 0 of 2802 raw ids have two names > 25 %, i.e. every mask is already single-name. The rule is implemented (`refine_frame(multi_name_cover=0.25)`, CLI `--multi-name-cover`, default off, unit-tested) and needs a SAM 3 re-run to evaluate, so it is untested on data.

## A2 depth split
| config | inst mIoU | PQ | exact-count | fragmented | merged | d inst mIoU vs base [95% CI] | d PQ [95% CI] | decision precision (correct/total) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.741 | 0.593 | 0.364 | 8 | 6 | +0.000 [+0.000, +0.000] 0 | +0.000 [+0.000, +0.000] | - |
| A2_depth | 0.734 | 0.582 | 0.273 | 10 | 6 | -0.007 [-0.022, +0.003] 0 | -0.011 [-0.053, +0.010] | split:depth 10/95 |
| A2_depth_dz5_ang30 | 0.738 | 0.581 | 0.364 | 10 | 5 | -0.003 [-0.019, +0.017] 0 | -0.012 [-0.032, +0.005] | split:depth 6/98 |
| A2_depth_dz12_ang50 | 0.752 | 0.600 | 0.364 | 9 | 4 | +0.011 [-0.009, +0.046] 0 | +0.007 [-0.023, +0.051] | split:depth 11/80 |
| A2_depth_dz20_ang60 | 0.753 | 0.607 | 0.364 | 8 | 4 | +0.013 [-0.007, +0.042] 0 | +0.014 [-0.018, +0.051] | split:depth 17/82 |
| A2_depth_dz30_ang70 | 0.753 | 0.596 | 0.364 | 9 | 4 | +0.012 [-0.007, +0.041] 0 | +0.003 [-0.035, +0.045] | split:depth 18/83 |
| A2_depth_dz12_ang50_p30 | 0.742 | 0.600 | 0.455 | 8 | 6 | +0.002 [-0.000, +0.005] 0 | +0.007 [-0.000, +0.020] | split:depth 5/18 |

## A3 colour split
| config | inst mIoU | PQ | exact-count | fragmented | merged | d inst mIoU vs base [95% CI] | d PQ [95% CI] | decision precision (correct/total) |
|---|---|---|---|---|---|---|---|---|
| A3_colour | 0.704 | 0.474 | 0.091 | 12 | 5 | -0.037 [-0.061, -0.012] - | -0.119 [-0.181, -0.057] | split:colour 17/376 |
| A3_colour_d20_f25 | 0.713 | 0.515 | 0.182 | 12 | 5 | -0.027 [-0.042, -0.013] - | -0.078 [-0.119, -0.040] | split:colour 13/240 |

## B1 merge touching same-name masks
| config | inst mIoU | PQ | exact-count | fragmented | merged | d inst mIoU vs base [95% CI] | d PQ [95% CI] | decision precision (correct/total) |
|---|---|---|---|---|---|---|---|---|
| B1_merge | 0.699 | 0.578 | 0.545 | 9 | 10 | -0.041 [-0.102, +0.031] 0 | -0.015 [-0.114, +0.095] | merge:same-name 125/195 |
| B1_merge_strict | 0.717 | 0.572 | 0.364 | 8 | 8 | -0.023 [-0.066, +0.003] 0 | -0.021 [-0.069, +0.008] | merge:same-name 54/78 |

## B2 mask-quality filter
| config | inst mIoU | PQ | exact-count | fragmented | merged | d inst mIoU vs base [95% CI] | d PQ [95% CI] | decision precision (correct/total) |
|---|---|---|---|---|---|---|---|---|
| B2_score0.3 | 0.741 | 0.598 | 0.455 | 8 | 6 | +0.000 [-0.001, +0.001] 0 | +0.005 [-0.001, +0.018] | drop:score 29/182 |
| B2_score0.4 | 0.700 | 0.573 | 0.545 | 8 | 6 | -0.040 [-0.086, -0.002] - | -0.020 [-0.093, +0.036] | drop:score 129/615 |
| B2_score0.5 | 0.616 | 0.539 | 0.545 | 8 | 6 | -0.125 [-0.242, -0.024] - | -0.054 [-0.182, +0.063] | drop:score 185/1002 |
| B2_area0.3pct | 0.698 | 0.625 | 0.364 | 5 | 9 | -0.043 [-0.067, -0.013] - | +0.032 [-0.001, +0.068] | drop:area 290/1459 |
| B2_area0.1pct | 0.739 | 0.593 | 0.364 | 8 | 6 | -0.001 [-0.004, -0.000] - | -0.000 [-0.001, +0.000] | drop:area 279/1186 |
| B2_area0.05pct | 0.740 | 0.593 | 0.364 | 8 | 6 | -0.000 [-0.001, +0.000] 0 | -0.000 [-0.001, +0.000] | drop:area 270/1089 |
| B2_compact0.5 | 0.636 | 0.548 | 0.545 | 6 | 4 | -0.105 [-0.199, -0.023] - | -0.045 [-0.124, +0.017] | drop:compact 160/725 |
| B2_compact0.3 | 0.718 | 0.585 | 0.455 | 7 | 6 | -0.023 [-0.045, -0.003] - | -0.008 [-0.053, +0.026] | drop:compact 145/579 |
| B2_all_s0.4 | 0.567 | 0.523 | 0.455 | 3 | 7 | -0.173 [-0.298, -0.064] - | -0.070 [-0.234, +0.067] | drop:area 167/949, drop:score 129/615, drop:compact 21/128 |

## C combined and co-vote cluster threshold (distance cut; default 0.8)
| config | inst mIoU | PQ | exact-count | fragmented | merged | d inst mIoU vs base [95% CI] | d PQ [95% CI] | decision precision (correct/total) |
|---|---|---|---|---|---|---|---|---|
| C1_depth20_60+score0.3 | 0.753 | 0.612 | 0.455 | 8 | 4 | +0.013 [-0.007, +0.043] 0 | +0.019 [-0.015, +0.056] | drop:score 29/182, split:depth 15/78 |
| C2_depth20_60+score0.3+area0.3pct | 0.231 | 0.200 | 0.364 | 3 | 6 | -0.510 [-0.715, -0.277] - | -0.393 [-0.632, -0.152] | drop:area 261/1294, drop:score 29/182, split:depth 14/53 |
| C1_ct0.7 | 0.596 | 0.471 | 0.455 | 9 | 4 | -0.144 [-0.384, +0.037] 0 | -0.122 [-0.337, +0.040] | drop:score 29/182, split:depth 15/78 |
| C1_ct0.9 | 0.723 | 0.600 | 0.455 | 7 | 7 | -0.018 [-0.054, +0.031] 0 | +0.007 [-0.032, +0.056] | drop:score 29/182, split:depth 15/78 |
| baseline_ct0.7 | 0.658 | 0.510 | 0.364 | 7 | 6 | -0.083 [-0.268, +0.000] 0 | -0.083 [-0.254, -0.000] | - |
| baseline_ct0.9 | 0.732 | 0.595 | 0.455 | 7 | 7 | -0.009 [-0.026, +0.000] 0 | +0.002 [-0.001, +0.010] | - |
| A2best_ct0.7 | 0.749 | 0.593 | 0.364 | 9 | 4 | +0.008 [-0.016, +0.039] 0 | -0.000 [-0.033, +0.040] | split:depth 17/82 |
| A2best_ct0.9 | 0.624 | 0.501 | 0.455 | 7 | 2 | -0.116 [-0.300, -0.001] - | -0.092 [-0.250, +0.003] | split:depth 17/82 |

## Decision precision (correct / total, pooled over 11 scans)
| rule | decisions | correct |
|---|---|---|
| A2 depth, 8 mm / 40 deg | 95 splits | 10 (11 %) |
| A2 depth, 12 mm / 50 deg | 80 | 11 (14 %) |
| A2 depth, 20 mm / 60 deg | 82 | 17 (21 %) |
| A3 colour (spec) | 376 | 17 (5 %) |
| A3 colour (dist 20, min share 25 %) | 240 | 13 (5 %) |
| B1 merge (spec) | 195 | 125 (64 %) |
| B1 merge (hull 0.9, Lab 5) | 78 | 54 (69 %) |
| B2 drop score < 0.3 / 0.4 / 0.5 | 182 / 615 / 1002 | 29 / 129 / 185 (16 / 21 / 18 %) |
| B2 drop area < 0.3 % | 1459 | 290 (20 %) |
| B2 drop hull ratio < 0.5 | 725 | 160 (22 %) |

## Reading
1. No 2D rule has a CI excluding 0 in the positive direction on inst mIoU, so no default was changed (everything stays off).
2. Depth split helps only when loose (20 mm / 60 deg: +0.013 mIoU, +0.014 PQ, merged items 6 -> 4, CI [-0.007, +0.042]); at the spec'd 8 mm / 40 deg it hurts (-0.007) and creates fragments (8 -> 10). Precision is low (11-21 %): it fires on 80-95 of the 303 GT-mixed masks and on many single items with creases.
3. Colour split is clearly harmful (-0.037, CI excludes 0 negatively; PQ -0.12): 5 % precision, many wrong splits of textured single items.
4. Same-name merge fixes fragments only in 2D terms (64 % of merges join the same GT id) but the wrong 36 % join neighbouring same-category items, merged items rise 6 -> 10 and mIoU falls (-0.041); stricter thresholds still -0.023.
5. Quality filters do not help: score >= 0.4/0.5 and hull < 0.5/0.3 delete real, if poor, masks (mIoU -0.04 to -0.17, CI < 0); score 0.3 and tiny-area drops are no-ops on mIoU (area < 0.3 % raises PQ +0.032 but loses mIoU -0.043). The best-combined C1 (depth 20/60 + score 0.3) is +0.013 / PQ +0.019, still not significant.
6. After 2D changes the linker stays best at 0.8 (0.7 and 0.9 both worse, C1 0.9: -0.018); the C2 interaction with the area filter collapsed (-0.51), so filters and linking interact violently. Next: the depth split is the only lead worth a re-test with more scans or on top of a boundary/instance refiner; A1 needs a SAM 3 re-run.
