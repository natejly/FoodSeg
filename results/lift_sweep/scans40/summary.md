# Lift sweep (CPU, 40 real-SAM 3 scans)

Code `pipeline/eval/lift_sweep.py` (+ `pipeline/foodscan/labels.py`), raw numbers `results.json`. Total sweep runtime 210 s on CPU (40 scans in parallel). Union over the 40 scans, 273 valid GT items, evaluated on the 2 mm canonical voxel scene as in `results/seg3d/merge`. Deltas are the paired per-scan mean-inst-IoU difference vs `current` (2000-resample bootstrap over the 40 scans). `oracle-merge ceiling` = inst mIoU after merging every pair of lift ids whose GT-majority item (purity >= .5) is the same: what a perfect refiner/merge could add on top of that setting. fragmented = GT item with >= 2 lift ids holding >= 15 % each; no lift id = < 5 % of the item's points carry a food id.

Pipeline replayed from the stored RAW per-frame SAM 3 maps: projection once, instance linking (as `segment.link_instances`, one-frame clusters dropped), weighting, decision, optional smoothing/superpoints; the flag-free default (`current`) = linking repair_ids 0.5, cos weights, min_nvotes 2, background as a class, argmax, second repair_ids pass, no smoothing.

## Reference rows

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| stored lift (V.lift labels in seg_points.npz; the 0.63 of the ablations) | 0.601 | 0.399 | 0.10 | 93 / 273 | 2 | 0.323 | 0.637 | +0.021 [+0.000, +0.045] | scans +/-/=: 19/16/5 |
| labels.lift_scan on the stored linked masks (masks_sam3) | 0.614 | 0.443 | 0.15 | 80 / 273 | 3 | 0.325 | 0.642 | +0.035 [+0.017, +0.057] | scans +/-/=: 18/5/17 |
| current (this script's replay of today's defaults) | 0.582 | 0.362 | 0.10 | 115 / 273 | 3 | 0.303 | 0.632 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/40 |

## weighting

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.582 | 0.362 | 0.10 | 115 / 273 | 3 | 0.303 | 0.632 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/40 |
| weighting=cos+score | 0.573 | 0.367 | 0.10 | 108 / 273 | 4 | 0.293 | 0.618 | -0.008 [-0.013, -0.003] | scans +/-/=: 4/17/19 |
| weighting=cos+qual | 0.579 | 0.364 | 0.10 | 114 / 273 | 3 | 0.305 | 0.628 | -0.003 [-0.008, +0.003] | scans +/-/=: 13/15/12 |
| edge_px=3.0 weighting=cos+edge | 0.582 | 0.357 | 0.10 | 114 / 273 | 3 | 0.303 | 0.631 | +0.000 [-0.001, +0.001] | scans +/-/=: 3/4/33 |
| edge_px=3.0 weighting=cos+score+edge | 0.575 | 0.361 | 0.10 | 112 / 273 | 3 | 0.295 | 0.619 | -0.007 [-0.011, -0.002] | scans +/-/=: 8/20/12 |
| edge_px=3.0 weighting=cos+score+edge+qual | 0.572 | 0.362 | 0.10 | 114 / 273 | 3 | 0.301 | 0.619 | -0.009 [-0.018, +0.000] | scans +/-/=: 11/20/9 |
| weighting=cos+edge | 0.581 | 0.359 | 0.10 | 115 / 273 | 3 | 0.302 | 0.630 | -0.000 [-0.002, +0.002] | scans +/-/=: 8/9/23 |
| weighting=cos+score+edge | 0.574 | 0.362 | 0.10 | 114 / 273 | 3 | 0.294 | 0.620 | -0.006 [-0.011, -0.002] | scans +/-/=: 7/20/13 |
| weighting=cos+score+edge+qual | 0.573 | 0.361 | 0.10 | 115 / 273 | 3 | 0.299 | 0.621 | -0.008 [-0.018, +0.001] | scans +/-/=: 10/21/9 |
| edge_px=12.0 weighting=cos+edge | 0.575 | 0.353 | 0.12 | 120 / 273 | 3 | 0.296 | 0.627 | -0.003 [-0.007, +0.000] | scans +/-/=: 9/17/14 |
| edge_px=12.0 weighting=cos+score+edge | 0.570 | 0.357 | 0.12 | 120 / 273 | 3 | 0.288 | 0.618 | -0.008 [-0.014, -0.003] | scans +/-/=: 5/26/9 |
| edge_px=12.0 weighting=cos+score+edge+qual | 0.570 | 0.354 | 0.10 | 116 / 273 | 3 | 0.296 | 0.620 | -0.010 [-0.019, -0.001] | scans +/-/=: 8/24/8 |
| **LOSO-selected within family** | 0.577 | 0.353 | 0.10 | 113 / 273 | 3 | 0.299 | 0.628 | -0.002 [-0.005, +0.000] | LOSO scans +/-/=: 2/5/33 |

LOSO picks: s000000: edge_px=3.0 weighting=cos+edge; s000001: edge_px=3.0 weighting=cos+edge; s000002: current; s000007: edge_px=3.0 weighting=cos+edge; s000009: edge_px=3.0 weighting=cos+edge; s000011: edge_px=3.0 weighting=cos+edge; s000012: edge_px=3.0 weighting=cos+edge; s000013: edge_px=3.0 weighting=cos+edge; s000014: edge_px=3.0 weighting=cos+edge; s000016: edge_px=3.0 weighting=cos+edge; s000017: edge_px=3.0 weighting=cos+edge; v13/s000000: edge_px=3.0 weighting=cos+edge; v13/s000001: edge_px=3.0 weighting=cos+edge; v13/s000002: edge_px=3.0 weighting=cos+edge; v13/s000003: edge_px=3.0 weighting=cos+edge; v13/s000004: edge_px=3.0 weighting=cos+edge; v13/s000005: edge_px=3.0 weighting=cos+edge; v13/s000006: current; v13/s000007: edge_px=3.0 weighting=cos+edge; v13/s000008: edge_px=3.0 weighting=cos+edge; v13/s000009: edge_px=3.0 weighting=cos+edge; v13/s000010: edge_px=3.0 weighting=cos+edge; v13/s000011: edge_px=3.0 weighting=cos+edge; v13/s000012: edge_px=3.0 weighting=cos+edge; v13/s000013: edge_px=3.0 weighting=cos+edge; v13/s000014: edge_px=3.0 weighting=cos+edge; v13/s000015: edge_px=3.0 weighting=cos+edge; v13/s000016: edge_px=3.0 weighting=cos+edge; v13/s000017: current; v13/s000018: edge_px=3.0 weighting=cos+edge; v13/s000019: current; v12/s000003: edge_px=3.0 weighting=cos+edge; v12/s000004: current; v12/s000005: current; v12/s000006: current; v12/s000008: current; v12/s000010: current; v12/s000015: edge_px=3.0 weighting=cos+edge; v12/s000018: edge_px=3.0 weighting=cos+edge; v12/s000019: weighting=cos+qual

## linking

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.582 | 0.362 | 0.10 | 115 / 273 | 3 | 0.303 | 0.632 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/40 |
| link=('repair', 0.3) | 0.580 | 0.360 | 0.10 | 117 / 273 | 3 | 0.302 | 0.632 | -0.005 [-0.010, +0.000] | scans +/-/=: 1/4/35 |
| link=('repair', 0.7) | 0.580 | 0.369 | 0.12 | 113 / 273 | 3 | 0.305 | 0.627 | +0.001 [-0.006, +0.008] | scans +/-/=: 5/5/30 |
| link=('mutual', 0.3) | 0.586 | 0.370 | 0.10 | 116 / 273 | 3 | 0.304 | 0.640 | +0.001 [-0.005, +0.007] | scans +/-/=: 6/5/29 |
| link=('mutual', 0.5) | 0.589 | 0.373 | 0.10 | 116 / 273 | 3 | 0.306 | 0.639 | +0.005 [+0.001, +0.009] | scans +/-/=: 8/2/30 |
| link=('cluster-cosine', 0.5) | 0.614 | 0.464 | 0.20 | 80 / 273 | 3 | 0.318 | 0.636 | +0.035 [+0.012, +0.058] | scans +/-/=: 23/10/7 |
| link=('cluster-cosine', 0.65) | 0.616 | 0.457 | 0.20 | 83 / 273 | 3 | 0.320 | 0.639 | +0.043 [+0.019, +0.069] | scans +/-/=: 23/6/11 |
| link=('cluster-cosine', 0.8) | 0.624 | 0.474 | 0.20 | 73 / 273 | 3 | 0.327 | 0.644 | +0.047 [+0.022, +0.072] | scans +/-/=: 25/6/9 |
| link=('cluster-cosine', 0.9) | 0.616 | 0.488 | 0.25 | 67 / 273 | 3 | 0.322 | 0.634 | +0.038 [+0.011, +0.065] | scans +/-/=: 23/6/11 |
| link=('cluster-cosine', 0.95) | 0.598 | 0.488 | 0.23 | 64 / 273 | 3 | 0.314 | 0.612 | +0.023 [-0.005, +0.051] | scans +/-/=: 22/10/8 |
| link=('cluster-min', 0.4) | 0.596 | 0.372 | 0.12 | 100 / 273 | 3 | 0.305 | 0.633 | +0.013 [-0.008, +0.035] | scans +/-/=: 18/14/8 |
| link=('cluster-min', 0.6) | 0.623 | 0.433 | 0.17 | 90 / 273 | 3 | 0.325 | 0.659 | +0.032 [+0.008, +0.054] | scans +/-/=: 25/8/7 |
| link=('cluster-min', 0.8) | 0.635 | 0.502 | 0.33 | 68 / 273 | 3 | 0.336 | 0.655 | +0.051 [+0.026, +0.076] | scans +/-/=: 27/5/8 |
| min_frames=1 | 0.582 | 0.352 | 0.10 | 115 / 273 | 3 | 0.304 | 0.633 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/40 |
| min_frames=3 | 0.582 | 0.368 | 0.12 | 115 / 273 | 3 | 0.303 | 0.632 | -0.000 [-0.000, +0.000] | scans +/-/=: 0/0/40 |
| link=('cluster-cosine', 0.9) min_frames=1 | 0.616 | 0.483 | 0.25 | 67 / 273 | 3 | 0.322 | 0.634 | +0.038 [+0.011, +0.065] | scans +/-/=: 23/6/11 |
| link=('cluster-cosine', 0.9) min_frames=3 | 0.616 | 0.494 | 0.25 | 67 / 273 | 3 | 0.322 | 0.634 | +0.038 [+0.011, +0.065] | scans +/-/=: 23/6/11 |
| repair2=False | 0.573 | 0.343 | 0.10 | 123 / 273 | 3 | 0.299 | 0.632 | -0.015 [-0.027, -0.005] | scans +/-/=: 1/10/29 |
| link=('cluster-cosine', 0.9) repair2=False | 0.616 | 0.485 | 0.25 | 67 / 273 | 3 | 0.320 | 0.635 | +0.038 [+0.011, +0.066] | scans +/-/=: 23/7/10 |
| **LOSO-selected within family** | 0.623 | 0.487 | 0.33 | 72 / 273 | 3 | 0.330 | 0.643 | +0.046 [+0.023, +0.071] | LOSO scans +/-/=: 26/6/8 |

LOSO picks: s000000: link=('cluster-min', 0.8); s000001: link=('cluster-min', 0.8); s000002: link=('cluster-min', 0.8); s000007: link=('cluster-min', 0.8); s000009: link=('cluster-min', 0.8); s000011: link=('cluster-min', 0.8); s000012: link=('cluster-min', 0.8); s000013: link=('cluster-min', 0.8); s000014: link=('cluster-min', 0.8); s000016: link=('cluster-min', 0.8); s000017: link=('cluster-min', 0.8); v13/s000000: link=('cluster-min', 0.8); v13/s000001: link=('cluster-min', 0.8); v13/s000002: link=('cluster-min', 0.8); v13/s000003: link=('cluster-min', 0.8); v13/s000004: link=('cluster-cosine', 0.8); v13/s000005: link=('cluster-min', 0.8); v13/s000006: link=('cluster-min', 0.8); v13/s000007: link=('cluster-min', 0.8); v13/s000008: link=('cluster-min', 0.8); v13/s000009: link=('cluster-min', 0.8); v13/s000010: link=('cluster-min', 0.8); v13/s000011: link=('cluster-min', 0.8); v13/s000012: link=('cluster-min', 0.8); v13/s000013: link=('cluster-min', 0.8); v13/s000014: link=('cluster-min', 0.8); v13/s000015: link=('cluster-min', 0.8); v13/s000016: link=('cluster-min', 0.8); v13/s000017: link=('cluster-min', 0.8); v13/s000018: link=('cluster-min', 0.8); v13/s000019: link=('cluster-min', 0.8); v12/s000003: link=('cluster-min', 0.8); v12/s000004: link=('cluster-min', 0.8); v12/s000005: link=('cluster-min', 0.8); v12/s000006: link=('cluster-min', 0.8); v12/s000008: link=('cluster-min', 0.8); v12/s000010: link=('cluster-min', 0.8); v12/s000015: link=('cluster-min', 0.8); v12/s000018: link=('cluster-min', 0.8); v12/s000019: link=('cluster-min', 0.8)

## decision

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| min_nvotes=1 | 0.582 | 0.362 | 0.10 | 115 / 273 | 3 | 0.303 | 0.632 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/40 |
| bg_abstain=True min_nvotes=1 | 0.583 | 0.358 | 0.10 | 115 / 273 | 3 | 0.307 | 0.634 | +0.002 [+0.001, +0.004] | scans +/-/=: 5/0/35 |
| margin=0.1 min_nvotes=1 | 0.543 | 0.343 | 0.10 | 70 / 273 | 3 | 0.275 | 0.576 | -0.038 [-0.056, -0.026] | scans +/-/=: 1/36/3 |
| bg_abstain=True margin=0.1 min_nvotes=1 | 0.546 | 0.339 | 0.12 | 70 / 273 | 3 | 0.280 | 0.578 | -0.035 [-0.052, -0.022] | scans +/-/=: 3/32/5 |
| margin=0.2 min_nvotes=1 | 0.504 | 0.325 | 0.12 | 47 / 273 | 9 | 0.245 | 0.529 | -0.073 [-0.095, -0.054] | scans +/-/=: 1/38/1 |
| bg_abstain=True margin=0.2 min_nvotes=1 | 0.507 | 0.322 | 0.12 | 47 / 273 | 6 | 0.251 | 0.533 | -0.068 [-0.091, -0.048] | scans +/-/=: 2/34/4 |
| bg_abstain=True | 0.583 | 0.356 | 0.10 | 115 / 273 | 3 | 0.307 | 0.634 | +0.002 [+0.001, +0.004] | scans +/-/=: 4/0/36 |
| margin=0.1 | 0.543 | 0.342 | 0.10 | 69 / 273 | 3 | 0.275 | 0.576 | -0.038 [-0.056, -0.026] | scans +/-/=: 1/36/3 |
| bg_abstain=True margin=0.1 | 0.545 | 0.338 | 0.12 | 70 / 273 | 3 | 0.280 | 0.578 | -0.035 [-0.053, -0.022] | scans +/-/=: 3/32/5 |
| margin=0.2 | 0.503 | 0.324 | 0.12 | 47 / 273 | 9 | 0.244 | 0.528 | -0.073 [-0.095, -0.054] | scans +/-/=: 1/38/1 |
| bg_abstain=True margin=0.2 | 0.507 | 0.321 | 0.12 | 47 / 273 | 6 | 0.251 | 0.533 | -0.069 [-0.091, -0.048] | scans +/-/=: 2/35/3 |
| min_nvotes=3 | 0.581 | 0.362 | 0.10 | 114 / 273 | 3 | 0.303 | 0.632 | -0.000 [-0.001, -0.000] | scans +/-/=: 0/1/39 |
| bg_abstain=True min_nvotes=3 | 0.583 | 0.356 | 0.10 | 114 / 273 | 3 | 0.306 | 0.634 | +0.002 [+0.000, +0.004] | scans +/-/=: 4/0/36 |
| margin=0.1 min_nvotes=3 | 0.542 | 0.341 | 0.10 | 69 / 273 | 3 | 0.275 | 0.575 | -0.039 [-0.056, -0.026] | scans +/-/=: 1/36/3 |
| bg_abstain=True margin=0.1 min_nvotes=3 | 0.545 | 0.337 | 0.12 | 70 / 273 | 3 | 0.280 | 0.578 | -0.035 [-0.053, -0.022] | scans +/-/=: 3/33/4 |
| margin=0.2 min_nvotes=3 | 0.503 | 0.324 | 0.12 | 47 / 273 | 9 | 0.244 | 0.528 | -0.074 [-0.096, -0.055] | scans +/-/=: 1/38/1 |
| bg_abstain=True margin=0.2 min_nvotes=3 | 0.507 | 0.321 | 0.12 | 47 / 273 | 6 | 0.250 | 0.533 | -0.069 [-0.091, -0.049] | scans +/-/=: 2/35/3 |
| current | 0.582 | 0.362 | 0.10 | 115 / 273 | 3 | 0.303 | 0.632 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/40 |
| **LOSO-selected within family** | 0.583 | 0.358 | 0.10 | 115 / 273 | 3 | 0.307 | 0.634 | +0.002 [+0.001, +0.004] | LOSO scans +/-/=: 5/0/35 |

LOSO picks: s000000: bg_abstain=True min_nvotes=1; s000001: bg_abstain=True min_nvotes=1; s000002: bg_abstain=True min_nvotes=1; s000007: bg_abstain=True min_nvotes=1; s000009: bg_abstain=True min_nvotes=1; s000011: bg_abstain=True min_nvotes=1; s000012: bg_abstain=True min_nvotes=1; s000013: bg_abstain=True min_nvotes=1; s000014: bg_abstain=True min_nvotes=1; s000016: bg_abstain=True min_nvotes=1; s000017: bg_abstain=True min_nvotes=1; v13/s000000: bg_abstain=True min_nvotes=1; v13/s000001: bg_abstain=True min_nvotes=1; v13/s000002: bg_abstain=True min_nvotes=1; v13/s000003: bg_abstain=True min_nvotes=1; v13/s000004: bg_abstain=True min_nvotes=1; v13/s000005: bg_abstain=True min_nvotes=1; v13/s000006: bg_abstain=True min_nvotes=1; v13/s000007: bg_abstain=True min_nvotes=1; v13/s000008: bg_abstain=True min_nvotes=1; v13/s000009: bg_abstain=True min_nvotes=1; v13/s000010: bg_abstain=True min_nvotes=1; v13/s000011: bg_abstain=True min_nvotes=1; v13/s000012: bg_abstain=True min_nvotes=1; v13/s000013: bg_abstain=True min_nvotes=1; v13/s000014: bg_abstain=True min_nvotes=1; v13/s000015: bg_abstain=True min_nvotes=1; v13/s000016: bg_abstain=True min_nvotes=1; v13/s000017: bg_abstain=True min_nvotes=1; v13/s000018: bg_abstain=True min_nvotes=1; v13/s000019: bg_abstain=True min_nvotes=1; v12/s000003: bg_abstain=True min_nvotes=1; v12/s000004: bg_abstain=True min_nvotes=1; v12/s000005: bg_abstain=True min_nvotes=1; v12/s000006: bg_abstain=True min_nvotes=1; v12/s000008: bg_abstain=True min_nvotes=1; v12/s000010: bg_abstain=True min_nvotes=1; v12/s000015: bg_abstain=True min_nvotes=1; v12/s000018: bg_abstain=True min_nvotes=1; v12/s000019: bg_abstain=True min_nvotes=1

## geometry

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.582 | 0.362 | 0.10 | 115 / 273 | 3 | 0.303 | 0.632 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/40 |
| smooth=0.004 | 0.588 | 0.392 | 0.12 | 108 / 273 | 3 | 0.295 | 0.634 | +0.007 [+0.003, +0.011] | scans +/-/=: 19/6/15 |
| smooth=0.002 | 0.587 | 0.378 | 0.12 | 109 / 273 | 3 | 0.305 | 0.634 | +0.005 [+0.003, +0.007] | scans +/-/=: 17/0/23 |
| smooth=0.008 | 0.581 | 0.398 | 0.10 | 100 / 273 | 3 | 0.281 | 0.619 | -0.002 [-0.009, +0.005] | scans +/-/=: 15/17/8 |
| sp=(200, 'geom', 'hard') | 0.492 | 0.369 | 0.25 | 116 / 273 | 11 | 0.254 | 0.513 | -0.097 [-0.126, -0.069] | scans +/-/=: 1/37/2 |
| sp=(200, 'geom', 'soft') | 0.492 | 0.381 | 0.30 | 109 / 273 | 12 | 0.252 | 0.513 | -0.096 [-0.128, -0.067] | scans +/-/=: 3/33/4 |
| sp=(200, 'geom+colour', 'hard') | 0.548 | 0.412 | 0.25 | 105 / 273 | 3 | 0.284 | 0.577 | -0.028 [-0.054, -0.007] | scans +/-/=: 11/24/5 |
| sp=(200, 'geom+colour', 'soft') | 0.542 | 0.416 | 0.20 | 101 / 273 | 3 | 0.282 | 0.571 | -0.034 [-0.059, -0.012] | scans +/-/=: 11/24/5 |
| sp=(400, 'geom', 'hard') | 0.534 | 0.393 | 0.20 | 101 / 273 | 3 | 0.280 | 0.563 | -0.047 [-0.063, -0.032] | scans +/-/=: 6/33/1 |
| sp=(400, 'geom', 'soft') | 0.534 | 0.396 | 0.25 | 96 / 273 | 4 | 0.277 | 0.564 | -0.046 [-0.063, -0.029] | scans +/-/=: 7/30/3 |
| sp=(400, 'geom+colour', 'hard') | 0.574 | 0.430 | 0.20 | 106 / 273 | 3 | 0.304 | 0.609 | -0.008 [-0.021, +0.006] | scans +/-/=: 12/20/8 |
| sp=(400, 'geom+colour', 'soft') | 0.571 | 0.429 | 0.20 | 107 / 273 | 3 | 0.303 | 0.607 | -0.014 [-0.027, -0.002] | scans +/-/=: 11/23/6 |
| sp=(600, 'geom', 'hard') | 0.546 | 0.381 | 0.17 | 116 / 273 | 4 | 0.275 | 0.578 | -0.036 [-0.049, -0.022] | scans +/-/=: 5/33/2 |
| sp=(600, 'geom', 'soft') | 0.545 | 0.391 | 0.17 | 114 / 273 | 5 | 0.278 | 0.579 | -0.039 [-0.053, -0.024] | scans +/-/=: 5/34/1 |
| sp=(600, 'geom+colour', 'hard') | 0.569 | 0.399 | 0.23 | 107 / 273 | 2 | 0.294 | 0.610 | -0.013 [-0.026, -0.003] | scans +/-/=: 12/22/6 |
| sp=(600, 'geom+colour', 'soft') | 0.570 | 0.404 | 0.23 | 111 / 273 | 3 | 0.296 | 0.610 | -0.013 [-0.025, -0.003] | scans +/-/=: 10/23/7 |
| **LOSO-selected within family** | 0.588 | 0.392 | 0.12 | 108 / 273 | 3 | 0.295 | 0.634 | +0.007 [+0.003, +0.011] | LOSO scans +/-/=: 19/6/15 |

LOSO picks: s000000: smooth=0.004; s000001: smooth=0.004; s000002: smooth=0.004; s000007: smooth=0.004; s000009: smooth=0.004; s000011: smooth=0.004; s000012: smooth=0.004; s000013: smooth=0.004; s000014: smooth=0.004; s000016: smooth=0.004; s000017: smooth=0.004; v13/s000000: smooth=0.004; v13/s000001: smooth=0.004; v13/s000002: smooth=0.004; v13/s000003: smooth=0.004; v13/s000004: smooth=0.004; v13/s000005: smooth=0.004; v13/s000006: smooth=0.004; v13/s000007: smooth=0.004; v13/s000008: smooth=0.004; v13/s000009: smooth=0.004; v13/s000010: smooth=0.004; v13/s000011: smooth=0.004; v13/s000012: smooth=0.004; v13/s000013: smooth=0.004; v13/s000014: smooth=0.004; v13/s000015: smooth=0.004; v13/s000016: smooth=0.004; v13/s000017: smooth=0.004; v13/s000018: smooth=0.004; v13/s000019: smooth=0.004; v12/s000003: smooth=0.004; v12/s000004: smooth=0.004; v12/s000005: smooth=0.004; v12/s000006: smooth=0.004; v12/s000008: smooth=0.004; v12/s000010: smooth=0.004; v12/s000015: smooth=0.004; v12/s000018: smooth=0.004; v12/s000019: smooth=0.004

## Superpoint ceiling diagnostic (GT-majority label per superpoint; uses GT, not a method)

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| sp_oracle 200 geom | 0.570 | 0.534 | 0.60 | 78 / 273 | 3 | 0.290 | 0.570 | -0.013 [-0.061, +0.037] | scans +/-/=: 19/19/2 |
| sp_oracle 200 geom+colour | 0.629 | 0.612 | 0.70 | 57 / 273 | 3 | 0.320 | 0.629 | +0.065 [+0.024, +0.107] | scans +/-/=: 29/9/2 |
| sp_oracle 400 geom | 0.657 | 0.614 | 0.80 | 66 / 273 | 3 | 0.339 | 0.657 | +0.071 [+0.029, +0.119] | scans +/-/=: 27/12/1 |
| sp_oracle 400 geom+colour | 0.699 | 0.673 | 0.80 | 50 / 273 | 1 | 0.367 | 0.699 | +0.114 [+0.078, +0.154] | scans +/-/=: 30/5/5 |
| sp_oracle 600 geom | 0.699 | 0.649 | 0.82 | 54 / 273 | 2 | 0.359 | 0.699 | +0.108 [+0.067, +0.152] | scans +/-/=: 30/10/0 |
| sp_oracle 600 geom+colour | 0.726 | 0.682 | 0.85 | 47 / 273 | 1 | 0.383 | 0.726 | +0.138 [+0.101, +0.178] | scans +/-/=: 34/4/2 |

Superpoints per scan (mean): 200 geom: 222, 200 geom+colour: 220, 400 geom: 419, 400 geom+colour: 416, 600 geom: 614, 600 geom+colour: 612.

## Combined (cross of the family winners nominated on all scans; nomination is slightly optimistic, the LOSO row re-picks without the held-out scan)

| setting | inst mIoU | PQ | exact count | fragmented GT items | items w/o lift id | boundary IoU | oracle-merge ceiling | d inst vs current, per scan (95% CI) | note |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.582 | 0.362 | 0.10 | 115 / 273 | 3 | 0.303 | 0.632 | +0.000 [+0.000, +0.000] | scans +/-/=: 0/0/40 |
| smooth=0.004 | 0.588 | 0.392 | 0.12 | 108 / 273 | 3 | 0.295 | 0.634 | +0.007 [+0.003, +0.011] | scans +/-/=: 19/6/15 |
| bg_abstain=True min_nvotes=1 | 0.583 | 0.358 | 0.10 | 115 / 273 | 3 | 0.307 | 0.634 | +0.002 [+0.001, +0.004] | scans +/-/=: 5/0/35 |
| bg_abstain=True min_nvotes=1 smooth=0.004 | 0.589 | 0.393 | 0.12 | 108 / 273 | 3 | 0.297 | 0.636 | +0.008 [+0.004, +0.013] | scans +/-/=: 19/4/17 |
| link=('cluster-min', 0.8) | 0.635 | 0.502 | 0.33 | 68 / 273 | 3 | 0.336 | 0.655 | +0.051 [+0.026, +0.076] | scans +/-/=: 27/5/8 |
| link=('cluster-min', 0.8) smooth=0.004 | 0.638 | 0.517 | 0.38 | 68 / 273 | 3 | 0.326 | 0.656 | +0.054 [+0.029, +0.080] | scans +/-/=: 32/6/2 |
| bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 | 0.637 | 0.496 | 0.30 | 68 / 273 | 3 | 0.340 | 0.657 | +0.053 [+0.029, +0.078] | scans +/-/=: 30/5/5 |
| bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004 | 0.639 | 0.515 | 0.35 | 68 / 273 | 3 | 0.328 | 0.658 | +0.056 [+0.031, +0.082] | scans +/-/=: 32/6/2 |
| edge_px=3.0 weighting=cos+edge | 0.582 | 0.357 | 0.10 | 114 / 273 | 3 | 0.303 | 0.631 | +0.000 [-0.001, +0.001] | scans +/-/=: 3/4/33 |
| edge_px=3.0 smooth=0.004 weighting=cos+edge | 0.587 | 0.389 | 0.10 | 111 / 273 | 3 | 0.295 | 0.634 | +0.006 [+0.003, +0.011] | scans +/-/=: 17/5/18 |
| bg_abstain=True edge_px=3.0 min_nvotes=1 weighting=cos+edge | 0.583 | 0.356 | 0.10 | 114 / 273 | 3 | 0.304 | 0.633 | +0.002 [+0.000, +0.005] | scans +/-/=: 7/4/29 |
| bg_abstain=True edge_px=3.0 min_nvotes=1 smooth=0.004 weighting=cos+edge | 0.588 | 0.388 | 0.10 | 111 / 273 | 3 | 0.296 | 0.635 | +0.008 [+0.003, +0.013] | scans +/-/=: 17/4/19 |
| edge_px=3.0 link=('cluster-min', 0.8) weighting=cos+edge | 0.635 | 0.499 | 0.30 | 67 / 273 | 3 | 0.335 | 0.655 | +0.051 [+0.026, +0.077] | scans +/-/=: 28/5/7 |
| edge_px=3.0 link=('cluster-min', 0.8) smooth=0.004 weighting=cos+edge | 0.637 | 0.515 | 0.38 | 67 / 273 | 3 | 0.325 | 0.656 | +0.054 [+0.029, +0.079] | scans +/-/=: 32/6/2 |
| bg_abstain=True edge_px=3.0 link=('cluster-min', 0.8) min_nvotes=1 weighting=cos+edge | 0.636 | 0.494 | 0.28 | 67 / 273 | 3 | 0.336 | 0.655 | +0.053 [+0.028, +0.078] | scans +/-/=: 29/6/5 |
| bg_abstain=True edge_px=3.0 link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004 weighting=cos+edge | 0.638 | 0.514 | 0.35 | 67 / 273 | 3 | 0.325 | 0.657 | +0.055 [+0.030, +0.081] | scans +/-/=: 32/6/2 |
| **LOSO-selected among combined** | 0.639 | 0.515 | 0.35 | 68 / 273 | 3 | 0.328 | 0.658 | +0.056 [+0.031, +0.082] | LOSO scans +/-/=: 32/6/2 |

LOSO picks: s000000: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000001: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000002: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000007: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000009: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000011: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000012: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000013: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000014: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000016: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; s000017: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000000: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000001: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000002: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000003: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000004: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000005: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000006: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000007: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000008: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000009: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000010: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000011: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000012: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000013: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000014: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000015: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000016: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000017: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000018: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v13/s000019: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v12/s000003: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v12/s000004: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v12/s000005: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v12/s000006: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v12/s000008: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v12/s000010: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v12/s000015: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v12/s000018: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004; v12/s000019: bg_abstain=True link=('cluster-min', 0.8) min_nvotes=1 smooth=0.004
