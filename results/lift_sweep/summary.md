# Lift sweep (CPU, 11 real-SAM 3 hard_v12 scans)

## Reading (8 lines)

1. Baselines: the stored lift is 0.632; today's `lift_scan` (repair + min_nvotes 2) is already 0.655 and is the `current` reference for every delta below.
2. What raises the ceiling is how mask ids are linked across views, not how votes are weighted: co-vote clustering (average-linkage, cosine, cut 0.8) gives 0.709 (+0.043, CI [-0.015, +0.089], better in most scans) and also lifts PQ 0.40 to 0.57 and exact count 0.09 to 0.27; mutual-best-overlap is a smaller +0.008.
3. Weighting: SAM score and edge-distance weights hurt or do nothing (score -0.011, CI excludes 0 on the wrong side); mask-quality weighting (solidity x area) is neutral alone (+0.005 inst) but adds about +0.03 on top of clustering (0.741, CI [-0.009, +0.120]).
4. Decision rule: margins always hurt (-0.04 at 0.1, -0.09 at 0.2); min_nvotes 1/2/3 and background-as-abstention are within +-0.001 because nearly every point has >= 3 votes; so keep plain argmax.
5. Smoothing radius 2-4 mm is the only change whose CI excludes 0 (+0.006) but it costs boundary IoU (0.297 to 0.285), so it stays off.
6. Superpoints (GT-free, 200-600) do not raise instance mIoU (colour-aware 200: 0.665, CI [-0.037, +0.033]; geometry-only is clearly worse, -0.05 to -0.09) but they merge fragments: PQ 0.40 to 0.51, exact count 0.09 to 0.27, fragmented items 20 to 14, and with clustering 7 / 52 fragmented, PQ 0.63.
7. Ceilings: perfect merge of whole ids is 0.735 for current, 0.745-0.778 after clustering (+quality); a perfect GT-majority label per superpoint would give 0.79-0.82, so superpoints do NOT make the refiner/merge question moot: even with clustering the best honest result (0.744) is 0.05-0.08 below the superpoint oracle and boundary IoU stays at 0.30-0.33.
8. Default: nothing is switched on, because no substantive gain has a CI excluding 0 on 11 scans; recommended next trial is `labels.cluster_ids` (+ `weighting="cos+qual"`) in the segment linker (not edited here; the per-scan win counts are 8-9 of 11), then confirm on the 47 train scans / a new real-SAM run.

Code `pipeline/eval/lift_sweep.py` (+ `pipeline/foodscan/labels.py`), raw numbers `results.json`. Total sweep runtime 87 s on CPU (11 scans in parallel). Union over the 11 scans, 52 valid GT items, evaluated on the 2 mm canonical voxel scene as in `results/seg3d/merge`. Deltas are the paired per-scan mean-inst-IoU difference vs `current` (2000-resample bootstrap over the 11 scans). `oracle-merge ceiling` = inst mIoU after merging every pair of lift ids whose GT-majority item (purity >= .5) is the same: what a perfect refiner/merge could add on top of that setting. fragmented = GT item with >= 2 lift ids holding >= 15 % each; no lift id = < 5 % of the item's points carry a food id.

Pipeline replayed from the stored RAW per-frame SAM 3 maps: projection once, instance linking (as `segment.link_instances`, one-frame clusters dropped), weighting, decision, optional smoothing/superpoints; the flag-free default (`current`) = linking repair_ids 0.5, cos weights, min_nvotes 2, background as a class, argmax, second repair_ids pass, no smoothing.

## Reference rows

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| stored lift (V.lift labels in seg_points.npz; the 0.63 of the ablations) | 0.632 | 0.371 | 0.00 | 24 / 52 | 0 | 0.321 | 0.727 | -0.024 [-0.047, -0.007] | scans +/-/=: 2/7/2 |
| labels.lift_scan on the stored linked masks (masks_sam3) | 0.656 | 0.399 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | +0.000 [+0.000, +0.001] | scans +/-/=: 0/0/11 |
| current (this script's replay of today's defaults) | 0.655 | 0.398 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/11 |

## weighting

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.655 | 0.398 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/11 |
| weighting=cos+score | 0.644 | 0.403 | 0.09 | 22 / 52 | 0 | 0.290 | 0.718 | -0.011 [-0.018, -0.004] | scans +/-/=: 0/6/5 |
| weighting=cos+qual | 0.660 | 0.421 | 0.09 | 21 / 52 | 0 | 0.299 | 0.740 | +0.000 [-0.009, +0.010] | scans +/-/=: 4/4/3 |
| edge_px=3.0 weighting=cos+edge | 0.654 | 0.401 | 0.09 | 20 / 52 | 0 | 0.293 | 0.737 | -0.001 [-0.003, +0.002] | scans +/-/=: 1/1/9 |
| edge_px=3.0 weighting=cos+score+edge | 0.645 | 0.405 | 0.09 | 22 / 52 | 0 | 0.290 | 0.722 | -0.009 [-0.016, -0.003] | scans +/-/=: 1/7/3 |
| edge_px=3.0 weighting=cos+score+edge+qual | 0.659 | 0.422 | 0.09 | 21 / 52 | 0 | 0.301 | 0.736 | -0.004 [-0.022, +0.014] | scans +/-/=: 3/5/3 |
| weighting=cos+edge | 0.656 | 0.412 | 0.09 | 20 / 52 | 0 | 0.299 | 0.739 | -0.000 [-0.003, +0.004] | scans +/-/=: 2/2/7 |
| weighting=cos+score+edge | 0.647 | 0.409 | 0.09 | 22 / 52 | 0 | 0.293 | 0.726 | -0.007 [-0.012, -0.003] | scans +/-/=: 1/8/2 |
| weighting=cos+score+edge+qual | 0.659 | 0.418 | 0.09 | 21 / 52 | 0 | 0.294 | 0.740 | -0.005 [-0.022, +0.014] | scans +/-/=: 2/6/3 |
| edge_px=12.0 weighting=cos+edge | 0.654 | 0.404 | 0.09 | 21 / 52 | 0 | 0.291 | 0.735 | -0.002 [-0.007, +0.002] | scans +/-/=: 2/6/3 |
| edge_px=12.0 weighting=cos+score+edge | 0.645 | 0.415 | 0.09 | 22 / 52 | 0 | 0.280 | 0.728 | -0.009 [-0.012, -0.005] | scans +/-/=: 1/9/1 |
| edge_px=12.0 weighting=cos+score+edge+qual | 0.656 | 0.426 | 0.09 | 21 / 52 | 0 | 0.290 | 0.738 | -0.006 [-0.019, +0.009] | scans +/-/=: 1/8/2 |
| **LOSO-selected within family** | 0.647 | 0.393 | 0.09 | 22 / 52 | 0 | 0.295 | 0.724 | -0.009 [-0.022, +0.001] | LOSO scans +/-/=: 2/5/4 |

LOSO picks: s000000: weighting=cos+qual; s000001: weighting=cos+qual; s000002: current; s000007: weighting=cos+qual; s000009: weighting=cos+qual; s000011: weighting=cos+qual; s000012: weighting=cos+qual; s000013: edge_px=3.0 weighting=cos+score+edge+qual; s000014: weighting=cos+score+edge+qual; s000016: weighting=cos+qual; s000017: edge_px=3.0 weighting=cos+score+edge+qual

## linking

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.655 | 0.398 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/11 |
| link=('repair', 0.3) | 0.649 | 0.391 | 0.00 | 21 / 52 | 0 | 0.297 | 0.736 | -0.009 [-0.028, +0.000] | scans +/-/=: 0/1/10 |
| link=('repair', 0.7) | 0.658 | 0.416 | 0.09 | 19 / 52 | 0 | 0.301 | 0.734 | +0.004 [+0.000, +0.010] | scans +/-/=: 1/0/10 |
| link=('mutual', 0.3) | 0.667 | 0.424 | 0.09 | 19 / 52 | 0 | 0.300 | 0.757 | +0.004 [-0.009, +0.021] | scans +/-/=: 2/1/8 |
| link=('mutual', 0.5) | 0.671 | 0.427 | 0.09 | 19 / 52 | 0 | 0.301 | 0.757 | +0.008 [+0.000, +0.022] | scans +/-/=: 3/0/8 |
| link=('cluster-cosine', 0.5) | 0.703 | 0.533 | 0.27 | 15 / 52 | 0 | 0.305 | 0.746 | +0.038 [-0.020, +0.085] | scans +/-/=: 8/1/2 |
| link=('cluster-cosine', 0.65) | 0.705 | 0.546 | 0.27 | 15 / 52 | 0 | 0.307 | 0.745 | +0.039 [-0.018, +0.086] | scans +/-/=: 8/1/2 |
| link=('cluster-cosine', 0.8) | 0.709 | 0.567 | 0.27 | 14 / 52 | 0 | 0.305 | 0.745 | +0.043 [-0.015, +0.089] | scans +/-/=: 8/1/2 |
| link=('cluster-cosine', 0.9) | 0.700 | 0.564 | 0.36 | 13 / 52 | 0 | 0.300 | 0.736 | +0.036 [-0.019, +0.082] | scans +/-/=: 7/2/2 |
| link=('cluster-cosine', 0.95) | 0.658 | 0.549 | 0.27 | 11 / 52 | 0 | 0.300 | 0.693 | +0.001 [-0.050, +0.043] | scans +/-/=: 6/3/2 |
| link=('cluster-min', 0.4) | 0.682 | 0.462 | 0.09 | 17 / 52 | 0 | 0.283 | 0.733 | +0.023 [-0.024, +0.068] | scans +/-/=: 6/2/3 |
| link=('cluster-min', 0.6) | 0.687 | 0.505 | 0.18 | 15 / 52 | 0 | 0.302 | 0.743 | +0.015 [-0.051, +0.071] | scans +/-/=: 7/2/2 |
| link=('cluster-min', 0.8) | 0.695 | 0.580 | 0.45 | 10 / 52 | 0 | 0.317 | 0.728 | +0.027 [-0.023, +0.067] | scans +/-/=: 7/1/3 |
| min_frames=1 | 0.655 | 0.396 | 0.09 | 20 / 52 | 0 | 0.297 | 0.736 | +0.000 [-0.000, +0.000] | scans +/-/=: 0/0/11 |
| min_frames=3 | 0.655 | 0.398 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/11 |
| link=('cluster-cosine', 0.9) min_frames=1 | 0.700 | 0.564 | 0.36 | 13 / 52 | 0 | 0.300 | 0.736 | +0.036 [-0.019, +0.082] | scans +/-/=: 7/2/2 |
| link=('cluster-cosine', 0.9) min_frames=3 | 0.700 | 0.574 | 0.36 | 13 / 52 | 0 | 0.300 | 0.736 | +0.036 [-0.019, +0.082] | scans +/-/=: 7/2/2 |
| repair2=False | 0.646 | 0.386 | 0.00 | 21 / 52 | 0 | 0.298 | 0.735 | -0.013 [-0.035, +0.000] | scans +/-/=: 0/2/9 |
| link=('cluster-cosine', 0.9) repair2=False | 0.700 | 0.564 | 0.36 | 13 / 52 | 0 | 0.300 | 0.736 | +0.036 [-0.019, +0.082] | scans +/-/=: 7/2/2 |
| **LOSO-selected within family** | 0.700 | 0.569 | 0.36 | 13 / 52 | 0 | 0.299 | 0.736 | +0.036 [-0.019, +0.083] | LOSO scans +/-/=: 7/2/2 |

LOSO picks: s000000: link=('cluster-cosine', 0.8); s000001: link=('cluster-cosine', 0.8); s000002: link=('cluster-cosine', 0.8); s000007: link=('cluster-cosine', 0.8); s000009: link=('cluster-cosine', 0.8); s000011: link=('cluster-cosine', 0.8); s000012: link=('cluster-cosine', 0.8); s000013: link=('cluster-cosine', 0.8); s000014: link=('cluster-cosine', 0.9) min_frames=1; s000016: link=('cluster-cosine', 0.8); s000017: link=('cluster-cosine', 0.8)

## decision

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| min_nvotes=1 | 0.655 | 0.398 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/11 |
| bg_abstain=True min_nvotes=1 | 0.656 | 0.391 | 0.09 | 20 / 52 | 0 | 0.303 | 0.738 | +0.001 [-0.000, +0.003] | scans +/-/=: 1/0/10 |
| margin=0.1 min_nvotes=1 | 0.607 | 0.371 | 0.09 | 15 / 52 | 0 | 0.269 | 0.671 | -0.043 [-0.061, -0.026] | scans +/-/=: 0/10/1 |
| bg_abstain=True margin=0.1 min_nvotes=1 | 0.609 | 0.367 | 0.09 | 15 / 52 | 0 | 0.277 | 0.675 | -0.040 [-0.059, -0.022] | scans +/-/=: 1/9/1 |
| margin=0.2 min_nvotes=1 | 0.559 | 0.343 | 0.09 | 10 / 52 | 0 | 0.242 | 0.614 | -0.088 [-0.123, -0.055] | scans +/-/=: 0/10/1 |
| bg_abstain=True margin=0.2 min_nvotes=1 | 0.561 | 0.334 | 0.09 | 10 / 52 | 0 | 0.251 | 0.619 | -0.085 [-0.121, -0.051] | scans +/-/=: 0/9/2 |
| bg_abstain=True | 0.656 | 0.391 | 0.09 | 20 / 52 | 0 | 0.303 | 0.738 | +0.001 [-0.000, +0.003] | scans +/-/=: 1/0/10 |
| margin=0.1 | 0.607 | 0.371 | 0.09 | 14 / 52 | 0 | 0.269 | 0.670 | -0.043 [-0.061, -0.026] | scans +/-/=: 0/10/1 |
| bg_abstain=True margin=0.1 | 0.609 | 0.367 | 0.09 | 15 / 52 | 0 | 0.277 | 0.675 | -0.040 [-0.059, -0.022] | scans +/-/=: 1/9/1 |
| margin=0.2 | 0.559 | 0.343 | 0.09 | 10 / 52 | 0 | 0.242 | 0.614 | -0.088 [-0.123, -0.055] | scans +/-/=: 0/10/1 |
| bg_abstain=True margin=0.2 | 0.561 | 0.334 | 0.09 | 10 / 52 | 0 | 0.251 | 0.619 | -0.085 [-0.121, -0.051] | scans +/-/=: 0/9/2 |
| min_nvotes=3 | 0.655 | 0.398 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | -0.000 [-0.000, +0.000] | scans +/-/=: 0/0/11 |
| bg_abstain=True min_nvotes=3 | 0.656 | 0.391 | 0.09 | 20 / 52 | 0 | 0.303 | 0.738 | +0.001 [-0.000, +0.003] | scans +/-/=: 1/0/10 |
| margin=0.1 min_nvotes=3 | 0.607 | 0.371 | 0.09 | 14 / 52 | 0 | 0.269 | 0.670 | -0.043 [-0.061, -0.026] | scans +/-/=: 0/10/1 |
| bg_abstain=True margin=0.1 min_nvotes=3 | 0.609 | 0.367 | 0.09 | 15 / 52 | 0 | 0.277 | 0.675 | -0.040 [-0.059, -0.022] | scans +/-/=: 1/9/1 |
| margin=0.2 min_nvotes=3 | 0.559 | 0.343 | 0.09 | 10 / 52 | 0 | 0.242 | 0.614 | -0.088 [-0.123, -0.055] | scans +/-/=: 0/10/1 |
| bg_abstain=True margin=0.2 min_nvotes=3 | 0.561 | 0.334 | 0.09 | 10 / 52 | 0 | 0.251 | 0.619 | -0.085 [-0.121, -0.051] | scans +/-/=: 0/9/2 |
| current | 0.655 | 0.398 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/11 |
| **LOSO-selected within family** | 0.656 | 0.391 | 0.09 | 20 / 52 | 0 | 0.303 | 0.738 | +0.001 [-0.000, +0.003] | LOSO scans +/-/=: 1/0/10 |

LOSO picks: s000000: bg_abstain=True min_nvotes=1; s000001: bg_abstain=True min_nvotes=1; s000002: bg_abstain=True min_nvotes=1; s000007: bg_abstain=True min_nvotes=1; s000009: bg_abstain=True min_nvotes=1; s000011: bg_abstain=True min_nvotes=1; s000012: bg_abstain=True min_nvotes=1; s000013: bg_abstain=True min_nvotes=1; s000014: bg_abstain=True min_nvotes=1; s000016: bg_abstain=True min_nvotes=1; s000017: bg_abstain=True min_nvotes=1

## geometry

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.655 | 0.398 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/11 |
| smooth=0.004 | 0.662 | 0.414 | 0.09 | 20 / 52 | 0 | 0.285 | 0.735 | +0.006 [+0.001, +0.013] | scans +/-/=: 5/2/4 |
| smooth=0.002 | 0.661 | 0.410 | 0.09 | 20 / 52 | 0 | 0.296 | 0.736 | +0.006 [+0.004, +0.008] | scans +/-/=: 5/0/6 |
| smooth=0.008 | 0.651 | 0.426 | 0.09 | 20 / 52 | 0 | 0.262 | 0.720 | -0.007 [-0.019, +0.006] | scans +/-/=: 4/5/2 |
| sp=(200, 'geom', 'hard') | 0.588 | 0.431 | 0.45 | 16 / 52 | 0 | 0.251 | 0.642 | -0.087 [-0.147, -0.033] | scans +/-/=: 1/10/0 |
| sp=(200, 'geom', 'soft') | 0.590 | 0.433 | 0.45 | 16 / 52 | 0 | 0.263 | 0.637 | -0.087 [-0.146, -0.031] | scans +/-/=: 2/9/0 |
| sp=(200, 'geom+colour', 'hard') | 0.665 | 0.512 | 0.27 | 14 / 52 | 0 | 0.309 | 0.728 | -0.003 [-0.037, +0.033] | scans +/-/=: 4/6/1 |
| sp=(200, 'geom+colour', 'soft') | 0.665 | 0.507 | 0.27 | 14 / 52 | 0 | 0.313 | 0.728 | -0.003 [-0.037, +0.034] | scans +/-/=: 5/6/0 |
| sp=(400, 'geom', 'hard') | 0.622 | 0.449 | 0.36 | 15 / 52 | 0 | 0.251 | 0.670 | -0.054 [-0.097, -0.008] | scans +/-/=: 2/8/1 |
| sp=(400, 'geom', 'soft') | 0.609 | 0.439 | 0.36 | 16 / 52 | 0 | 0.245 | 0.666 | -0.063 [-0.106, -0.021] | scans +/-/=: 2/9/0 |
| sp=(400, 'geom+colour', 'hard') | 0.661 | 0.481 | 0.18 | 19 / 52 | 0 | 0.308 | 0.722 | -0.002 [-0.030, +0.026] | scans +/-/=: 6/5/0 |
| sp=(400, 'geom+colour', 'soft') | 0.653 | 0.483 | 0.27 | 18 / 52 | 0 | 0.301 | 0.717 | -0.010 [-0.038, +0.018] | scans +/-/=: 5/6/0 |
| sp=(600, 'geom', 'hard') | 0.646 | 0.444 | 0.18 | 18 / 52 | 0 | 0.285 | 0.710 | -0.022 [-0.050, +0.005] | scans +/-/=: 4/7/0 |
| sp=(600, 'geom', 'soft') | 0.638 | 0.462 | 0.18 | 17 / 52 | 0 | 0.280 | 0.699 | -0.031 [-0.059, +0.002] | scans +/-/=: 2/9/0 |
| sp=(600, 'geom+colour', 'hard') | 0.658 | 0.452 | 0.18 | 18 / 52 | 0 | 0.302 | 0.720 | -0.000 [-0.018, +0.017] | scans +/-/=: 5/5/1 |
| sp=(600, 'geom+colour', 'soft') | 0.646 | 0.461 | 0.18 | 19 / 52 | 0 | 0.298 | 0.711 | -0.010 [-0.029, +0.010] | scans +/-/=: 4/6/1 |
| **LOSO-selected within family** | 0.638 | 0.447 | 0.27 | 17 / 52 | 0 | 0.304 | 0.702 | -0.018 [-0.044, +0.005] | LOSO scans +/-/=: 4/6/1 |

LOSO picks: s000000: sp=(200, 'geom+colour', 'hard'); s000001: sp=(200, 'geom+colour', 'soft'); s000002: smooth=0.002; s000007: sp=(200, 'geom+colour', 'hard'); s000009: sp=(200, 'geom+colour', 'hard'); s000011: sp=(200, 'geom+colour', 'hard'); s000012: sp=(200, 'geom+colour', 'hard'); s000013: sp=(200, 'geom+colour', 'hard'); s000014: sp=(200, 'geom+colour', 'hard'); s000016: sp=(200, 'geom+colour', 'hard'); s000017: sp=(200, 'geom+colour', 'hard')

## Superpoint ceiling diagnostic (GT-majority label per superpoint; uses GT, not a method)

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| sp_oracle 200 geom | 0.692 | 0.682 | 0.64 | 7 / 52 | 0 | 0.296 | 0.692 | +0.010 [-0.070, +0.093] | scans +/-/=: 5/6/0 |
| sp_oracle 200 geom+colour | 0.790 | 0.771 | 0.82 | 4 / 52 | 0 | 0.373 | 0.790 | +0.124 [+0.064, +0.189] | scans +/-/=: 10/1/0 |
| sp_oracle 400 geom | 0.743 | 0.720 | 0.82 | 7 / 52 | 0 | 0.320 | 0.743 | +0.064 [-0.015, +0.139] | scans +/-/=: 9/2/0 |
| sp_oracle 400 geom+colour | 0.796 | 0.800 | 0.82 | 6 / 52 | 0 | 0.369 | 0.796 | +0.127 [+0.067, +0.194] | scans +/-/=: 10/1/0 |
| sp_oracle 600 geom | 0.791 | 0.778 | 0.91 | 4 / 52 | 0 | 0.365 | 0.791 | +0.112 [+0.042, +0.184] | scans +/-/=: 9/1/1 |
| sp_oracle 600 geom+colour | 0.817 | 0.803 | 0.91 | 2 / 52 | 0 | 0.384 | 0.817 | +0.145 [+0.082, +0.212] | scans +/-/=: 10/1/0 |

Superpoints per scan (mean): 200 geom: 212, 200 geom+colour: 209, 400 geom: 410, 400 geom+colour: 406, 600 geom: 603, 600 geom+colour: 601.

## Combined (cross of the family winners nominated on all 11 scans; nomination is slightly optimistic, the LOSO row re-picks without the held-out scan)

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.655 | 0.398 | 0.09 | 20 / 52 | 0 | 0.297 | 0.735 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/11 |
| sp=(200, 'geom+colour', 'hard') | 0.665 | 0.512 | 0.27 | 14 / 52 | 0 | 0.309 | 0.728 | -0.003 [-0.037, +0.033] | scans +/-/=: 4/6/1 |
| bg_abstain=True min_nvotes=1 | 0.656 | 0.391 | 0.09 | 20 / 52 | 0 | 0.303 | 0.738 | +0.001 [-0.000, +0.003] | scans +/-/=: 1/0/10 |
| bg_abstain=True min_nvotes=1 sp=(200, 'geom+colour', 'hard') | 0.665 | 0.512 | 0.27 | 14 / 52 | 0 | 0.309 | 0.728 | -0.003 [-0.037, +0.033] | scans +/-/=: 4/6/1 |
| link=('cluster-cosine', 0.8) | 0.709 | 0.567 | 0.27 | 14 / 52 | 0 | 0.305 | 0.745 | +0.043 [-0.015, +0.089] | scans +/-/=: 8/1/2 |
| link=('cluster-cosine', 0.8) sp=(200, 'geom+colour', 'hard') | 0.687 | 0.632 | 0.36 | 7 / 52 | 0 | 0.307 | 0.717 | +0.026 [-0.032, +0.074] | scans +/-/=: 8/3/0 |
| bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 | 0.711 | 0.560 | 0.18 | 14 / 52 | 0 | 0.312 | 0.747 | +0.045 [-0.013, +0.091] | scans +/-/=: 9/1/1 |
| bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 sp=(200, 'geom+colour', 'hard') | 0.687 | 0.632 | 0.36 | 7 / 52 | 0 | 0.307 | 0.717 | +0.026 [-0.032, +0.074] | scans +/-/=: 8/3/0 |
| weighting=cos+qual | 0.660 | 0.421 | 0.09 | 21 / 52 | 0 | 0.299 | 0.740 | +0.000 [-0.009, +0.010] | scans +/-/=: 4/4/3 |
| sp=(200, 'geom+colour', 'hard') weighting=cos+qual | 0.662 | 0.498 | 0.27 | 13 / 52 | 0 | 0.316 | 0.722 | -0.004 [-0.037, +0.030] | scans +/-/=: 6/5/0 |
| bg_abstain=True min_nvotes=1 weighting=cos+qual | 0.662 | 0.417 | 0.09 | 21 / 52 | 0 | 0.309 | 0.744 | +0.003 [-0.006, +0.013] | scans +/-/=: 5/2/4 |
| bg_abstain=True min_nvotes=1 sp=(200, 'geom+colour', 'hard') weighting=cos+qual | 0.662 | 0.498 | 0.27 | 14 / 52 | 0 | 0.316 | 0.723 | -0.004 [-0.037, +0.030] | scans +/-/=: 6/5/0 |
| link=('cluster-cosine', 0.8) weighting=cos+qual | 0.741 | 0.593 | 0.36 | 10 / 52 | 0 | 0.324 | 0.774 | +0.058 [-0.009, +0.120] | scans +/-/=: 8/1/2 |
| link=('cluster-cosine', 0.8) sp=(200, 'geom+colour', 'hard') weighting=cos+qual | 0.727 | 0.640 | 0.45 | 8 / 52 | 0 | 0.331 | 0.757 | +0.049 [-0.017, +0.107] | scans +/-/=: 9/2/0 |
| bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual | 0.744 | 0.586 | 0.36 | 10 / 52 | 0 | 0.335 | 0.778 | +0.062 [-0.006, +0.123] | scans +/-/=: 9/1/1 |
| bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 sp=(200, 'geom+colour', 'hard') weighting=cos+qual | 0.727 | 0.640 | 0.45 | 8 / 52 | 0 | 0.331 | 0.757 | +0.049 [-0.017, +0.107] | scans +/-/=: 9/2/0 |
| **LOSO-selected among combined** | 0.744 | 0.586 | 0.36 | 10 / 52 | 0 | 0.335 | 0.778 | +0.062 [-0.006, +0.123] | LOSO scans +/-/=: 9/1/1 |

LOSO picks: s000000: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000001: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000002: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000007: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000009: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000011: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000012: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000013: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000014: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000016: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual; s000017: bg_abstain=True link=('cluster-cosine', 0.8) min_nvotes=1 weighting=cos+qual
