## 40-scan validation (hard_v12 + hard_v13, real SAM 3 votes, co-vote lift), 2026-10-16

**Decision: no merge arm is default-on; the merge stays an off-by-default tool (`python -m seg3d.merge apply`, not wired into run_meal).** Rule: pooled paired per-scan CI excludes 0 (positive), no batch negative, broken < 0.2 % of points. Every deployable arm has a negative pooled point estimate and the shipped heuristic and the heuristic/HGB retrained leave-one-batch-out all have a pooled CI that excludes 0 on the wrong side. Only the oracle (not deployable) is positive. Fragment merge does not generalise from the 11-scan study (heuristic 0.632 -> 0.608 there; here 0.631 -> 0.527 shipped thresholds, 0.611 retuned).

Setup: co-vote lift regenerated for all 40 scans (`python -m foodscan.boundary_refine replay40`; the live seg_points lift is the older repair-linked one), 273 items, pooled lift inst mIoU 0.631, PQ 0.487, exact count 0.25, 72 fragmented items (66 after CRF). `python -m seg3d.merge val40 20` (CPU, ~15 min): per scan fragment graph + the 18 edge features; "retrained leave-one-batch-out": thresholds picked on the other batch's 20 scans from a coarser grid than the 11-scan study (GRID40, 192 configs, objective pooled item IoU minus 1e-6 per merge), HGB fit on 4435 pseudo edges (split_v3 train x 8 seeds) plus the other batch's oracle-collected real edges (weight 5; {354, 345} edges without CRF, {305, 203} with). Picked thresholds: {"False|v12": {"t1": 0.6, "t2": 5, "t3": 45, "t4": 0.0015}, "False|v13": {"t1": 0.6, "t2": 5, "t3": 45, "t4": 0.003}, "True|v12": {"t1": 0.6, "t2": 5, "t3": 45, "t4": 0.0015}, "True|v13": {"t1": 0.6, "t2": 5, "t3": 45, "t4": 0.0015}}. The shipped HEUR_FINAL was fit on the v12 scans, so its v12 row is partly in-sample (and still -0.088). "With CRF" feeds the fixed-parameter boundary CRF lift into the merge (merge learning/threshold edges recomputed on that lift). Deltas are paired per-scan vs the raw co-vote lift (2000-resample bootstrap); values in brackets are the raw lift for the same set; broken pts % = lift-right points that become wrong. Raw numbers `val40.json`.

| CRF before | arm | set | inst mIoU (lift) | PQ (lift) | exact count (lift) | boundary IoU | d scan-mIoU vs raw lift (95% CI) | scans +/- | fragments made whole | correct items broken | items fixed | ids merged | broken pts % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no | heuristic | v12 | 0.534 (0.657) | 0.462 (0.524) | 0.30 (0.30) | 0.264 | -0.088 [-0.158, -0.031] | 1/9 | 0/34 | 21/99 | 0 | 36 | 0.276 |
| no | heuristic | v13 | 0.519 (0.600) | 0.433 (0.447) | 0.15 (0.20) | 0.282 | -0.094 [-0.155, -0.045] | 3/12 | 0/38 | 13/86 | 1 | 35 | 0.862 |
| no | heuristic | pooled | 0.527 (0.631) | 0.448 (0.487) | 0.23 (0.25) | 0.272 | -0.091 [-0.136, -0.053] | 4/21 | 0/72 | 34/185 | 1 | 71 | 0.529 |
| no | heuristic LOBO | v12 | 0.645 (0.657) | 0.532 (0.524) | 0.35 (0.30) | 0.328 | -0.008 [-0.019, -0.000] | 1/4 | 0/34 | 1/99 | 0 | 9 | 0.012 |
| no | heuristic LOBO | v13 | 0.572 (0.600) | 0.444 (0.447) | 0.20 (0.20) | 0.315 | -0.013 [-0.030, +0.001] | 7/3 | 0/38 | 6/86 | 0 | 16 | 0.229 |
| no | heuristic LOBO | pooled | 0.611 (0.631) | 0.490 (0.487) | 0.28 (0.25) | 0.322 | -0.010 [-0.020, -0.002] | 8/7 | 0/72 | 7/185 | 0 | 25 | 0.106 |
| no | learned HGB (pseudo edges + real edges of the other batch | v12 | 0.637 (0.657) | 0.521 (0.524) | 0.35 (0.30) | 0.325 | -0.013 [-0.030, -0.001] | 0/6 | 1/34 | 4/99 | 0 | 9 | 0.056 |
| no | learned HGB (pseudo edges + real edges of the other batch | v13 | 0.567 (0.600) | 0.453 (0.447) | 0.25 (0.20) | 0.309 | -0.021 [-0.044, -0.004] | 4/8 | 1/38 | 6/86 | 0 | 24 | 0.180 |
| no | learned HGB (pseudo edges + real edges of the other batch | pooled | 0.605 (0.631) | 0.489 (0.487) | 0.30 (0.25) | 0.318 | -0.017 [-0.031, -0.006] | 4/14 | 2/72 | 10/185 | 0 | 33 | 0.110 |
| no | oracle merge (ceiling) | v12 | 0.676 (0.657) | 0.619 (0.524) | 0.55 (0.30) | 0.354 | +0.019 [+0.005, +0.038] | 9/1 | 5/34 | 0/99 | 6 | 33 | 0.001 |
| no | oracle merge (ceiling) | v13 | 0.608 (0.600) | 0.520 (0.447) | 0.40 (0.20) | 0.334 | +0.009 [-0.005, +0.024] | 10/4 | 5/38 | 2/86 | 2 | 37 | 0.054 |
| no | oracle merge (ceiling) | pooled | 0.645 (0.631) | 0.572 (0.487) | 0.47 (0.25) | 0.344 | +0.014 [+0.003, +0.025] | 19/5 | 10/72 | 2/185 | 8 | 70 | 0.024 |
| yes (fixed params) | heuristic | v12 | 0.575 (0.657) | 0.503 (0.524) | 0.40 (0.30) | 0.288 | -0.057 [-0.123, -0.007] | 10/10 | 0/30 | 15/101 | 0 | 17 | 0.292 |
| yes (fixed params) | heuristic | v13 | 0.538 (0.600) | 0.437 (0.447) | 0.20 (0.20) | 0.291 | -0.050 [-0.086, -0.021] | 5/15 | 0/36 | 10/84 | 1 | 22 | 0.703 |
| yes (fixed params) | heuristic | pooled | 0.558 (0.631) | 0.471 (0.487) | 0.30 (0.25) | 0.289 | -0.053 [-0.087, -0.023] | 15/25 | 0/66 | 25/185 | 1 | 39 | 0.469 |
| yes (fixed params) | heuristic LOBO | v12 | 0.659 (0.657) | 0.561 (0.524) | 0.40 (0.30) | 0.342 | +0.003 [-0.004, +0.010] | 11/9 | 0/30 | 0/101 | 0 | 1 | 0.095 |
| yes (fixed params) | heuristic LOBO | v13 | 0.600 (0.600) | 0.461 (0.447) | 0.25 (0.20) | 0.322 | -0.002 [-0.005, +0.002] | 10/10 | 0/36 | 0/84 | 0 | 2 | 0.267 |
| yes (fixed params) | heuristic LOBO | pooled | 0.632 (0.631) | 0.512 (0.487) | 0.33 (0.25) | 0.333 | +0.001 [-0.003, +0.005] | 21/19 | 0/66 | 0/185 | 0 | 3 | 0.169 |
| yes (fixed params) | learned HGB (pseudo edges + real edges of the other batch | v12 | 0.625 (0.657) | 0.537 (0.524) | 0.40 (0.30) | 0.328 | -0.017 [-0.038, +0.001] | 9/11 | 1/30 | 7/101 | 0 | 10 | 0.152 |
| yes (fixed params) | learned HGB (pseudo edges + real edges of the other batch | v13 | 0.594 (0.600) | 0.473 (0.447) | 0.35 (0.20) | 0.316 | -0.007 [-0.013, -0.001] | 9/11 | 0/36 | 0/84 | 1 | 8 | 0.313 |
| yes (fixed params) | learned HGB (pseudo edges + real edges of the other batch | pooled | 0.611 (0.631) | 0.506 (0.487) | 0.38 (0.25) | 0.323 | -0.012 [-0.023, -0.002] | 18/22 | 1/66 | 7/185 | 1 | 18 | 0.221 |
| yes (fixed params) | oracle merge (ceiling) | v12 | 0.678 (0.657) | 0.634 (0.524) | 0.50 (0.30) | 0.354 | +0.022 [+0.007, +0.041] | 15/5 | 5/30 | 0/101 | 7 | 19 | 0.089 |
| yes (fixed params) | oracle merge (ceiling) | v13 | 0.607 (0.600) | 0.522 (0.447) | 0.40 (0.20) | 0.325 | +0.006 [-0.008, +0.021] | 11/9 | 5/36 | 1/84 | 2 | 29 | 0.308 |
| yes (fixed params) | oracle merge (ceiling) | pooled | 0.645 (0.631) | 0.581 (0.487) | 0.45 (0.25) | 0.341 | +0.014 [+0.002, +0.026] | 26/14 | 10/66 | 1/185 | 9 | 48 | 0.184 |

Reading: (1) the shipped heuristic breaks 34 of 185 correct items (0.53 % of points) and costs -0.091 [-0.136, -0.053]; thresholds do not transfer to the co-vote lift. (2) Retuned thresholds / HGB break 7 / 10 items (0.11 % of points) and still lose -0.010 [-0.020, -0.002] / -0.017 [-0.031, -0.006]; HGB makes 2 of 72 fragments whole, the heuristic 0. They raise exact count (0.25 -> 0.28 / 0.30) and PQ marginally (0.487 -> 0.490 / 0.489), i.e. count/PQ gains are small and cost mIoU. (3) The CRF does not rescue the merge: with CRF the retuned heuristic merges 3 ids (about the CRF-only lift, +0.001), HGB -0.012 [-0.023, -0.002]. (4) Oracle merging of whole lift ids gives only +0.014 [+0.003, +0.025] (10 of 72 fragments whole), so the ceiling of an id-level merge on the co-vote lift is small; the earlier fragment problem is largely absorbed by the co-vote linker.

---



## Fragment merge (CPU baseline)

Code `pipeline/seg3d/merge.py`, tests `pipeline/tests/test_seg3d_merge.py` (4 pass), raw numbers `results/seg3d/merge/results.json` (+ `real_merged_lifts.npz`, `run.log`, `summary.md`).
No GPU, no backbone: fragment graph over the lift instances (nodes = food ids 1..251; edges = pairs within 4 mm), 18 edge features (contact ratio = seam points of the smaller side / its
boundary points, seam point count, Lab dE and RGB-histogram chi2 across the seam (points within 6 mm of the seam), normal angle, median |dz| across the seam, size ratio and log sizes, mean
lift_share / lift_nvotes of each side, hull convexity (V_a+V_b)/V_union, fraction of the seam touching the container), agglomerative merge (best edge first, features recomputed for the merged node).
Same-SAM-name flag skipped (instance names are not stored per lift id). Safety rules (all arms except `no safety`): convexity >= 0.6; seam container-fraction <= 0.5 (the
"no merge across the container" rule: a seam that mostly lies on container points is food-wall-food); no merge of a side with lift_share < 0.35 into a side with >= 400 points.
Learned arms train on edges from oracle agglomeration (every edge state met while merging GT-same pairs), labelled "same GT item" (majority GT item of each node, purity >= .5): 4435 pseudo edges
(1867 positive; 47 split_v3 train scans x 8 seeds of corrupt_lift + 1-3 `fragment_lift`) plus 216 real edges (86 positive, weight 5); probability > 0.5 merges.

**Real SAM 3 votes: 11 hard_v12 scans, 52 valid items, leave-one-scan-out** (thresholds / classifiers refit without the held-out scan; grid {"t1": [0.1, 0.2, 0.3, 0.45, 0.6, 0.8], "t2": [3, 5, 8, 12, 18, 1000000000.0], "t3": [30, 45, 60, 90, 181], "t4": [0.0015, 0.003, 0.006, 0.012, 1000000000.0]}; objective = pooled item IoU; d = paired per-scan bootstrap, 2000 resamples; "made whole" = IoU>0.5 and >=85 % of the item in one id, among items the lift splits into >=2 ids holding >=15 % each).

| arm | inst mIoU | PQ | exact count | d vs lift per scan (95% CI) | fragments made whole | correct items broken | items fixed | ids merged |
|---|---|---|---|---|---|---|---|---|
| lift | 0.632 | 0.371 | 0.00 | - | 0 / 24 | 0 / 35 | 0 | 0 |
| heuristic, untuned defaults (t1=.15,t2=15,t3=40,t4=6mm) | 0.489 | 0.460 | 0.18 | -0.100 [-0.203, +0.001] | 6 / 24 | 11 / 35 | 3 | 49 |
| heuristic, thresholds tuned leave-one-scan-out | 0.608 | 0.465 | 0.45 | -0.006 [-0.084, +0.061] | 5 / 24 | 4 / 35 | 2 | 30 |
| heuristic tuned, no safety rules | 0.600 | 0.474 | 0.36 | -0.014 [-0.103, +0.061] | 5 / 24 | 4 / 35 | 2 | 33 |
| learned HGB (pseudo + other real scans, LOSO) | 0.630 | 0.487 | 0.09 | +0.010 [-0.036, +0.052] | 5 / 24 | 3 / 35 | 3 | 28 |
| learned HGB, pseudo edges only | 0.640 | 0.391 | 0.00 | +0.007 [+0.000, +0.021] | 1 / 24 | 0 / 35 | 0 | 5 |
| learned logistic regression (pseudo + real LOSO) | 0.620 | 0.481 | 0.18 | +0.000 [-0.061, +0.065] | 6 / 24 | 5 / 35 | 3 | 31 |
| oracle merge (GT-same edges; ceiling of merging only) | 0.713 | 0.626 | 0.55 | +0.087 [+0.046, +0.135] | 9 / 24 | 0 / 35 | 4 | 41 |

Thresholds picked per fold (LOSO): {"t1": 0.2, "t2": 8, "t3": 45, "t4": 0.0015}; {"t1": 0.3, "t2": 8, "t3": 60, "t4": 0.0015}; {"t1": 0.45, "t2": 8, "t3": 60, "t4": 0.003}; {"t1": 0.6, "t2": 12, "t3": 60, "t4": 0.003}; {"t1": 0.6, "t2": 8, "t3": 60, "t4": 0.0015}; fit on all 11 scans: {"t1": 0.3, "t2": 8, "t3": 60, "t4": 0.0015} (t1 contact ratio >, t2 Lab dE <, t3 normal angle deg <, t4 height step m <). Selection is unstable across folds (5 distinct picks). A narrower first grid
(t1<=.45, t4>=3 mm) gave LOSO inst 0.653 / d +0.029 [-0.017, +0.081]; the wider grid above is what is reported (grid widened after seeing a boundary pick, so neither number is a clean estimate).

Edge classifier on the 117 labelled real edges of the unmodified lift (57 same-item), LOSO, decision including safety rules (heuristic row = tuned LOSO thresholds):

| arm | predicted merge | true merges | precision | recall |
|---|---|---|---|---|
| heur | 29 | 23 | 0.79 | 0.40 |
| hgb | 36 | 27 | 0.75 | 0.47 |
| lr | 45 | 32 | 0.71 | 0.56 |
| hgb_pseudo | 6 | 6 | 1.00 | 0.11 |

HGB permutation importance (average precision, real edges): chi2 0.060, share_small 0.021, dE 0.014, hstep 0.010, ang 0.009, contact_ratio 0.009.
Logistic-regression standardized coefficients (largest |coef|): chi2 -1.17, dE -0.94, ang -0.52, log_n_small -0.46, hstep -0.45, size_ratio +0.43.

**Second check: split_v3 val pseudo-votes (27 scans x 3 seeds, R1 corruption + 1-3 fragment_lift; seeds averaged per scan for the CI).** Heuristic thresholds {"t1": 0.45, "t2": 12, "t3": 60, "t4": 0.003}; thresholds and real-edge training exclude the real scans that are copies of val scans.

| arm | inst mIoU | PQ | exact count | d vs lift per scan (95% CI) | fragments made whole | correct items broken | items fixed | ids merged |
|---|---|---|---|---|---|---|---|---|
| lift | 0.499 | 0.268 | 0.04 | - | 0 / 84 | 0 / 94 | 0 | 0 |
| heuristic, thresholds tuned leave-one-scan-out | 0.454 | 0.358 | 0.20 | +0.051 [-0.013, +0.120] | 21 / 84 | 21 / 94 | 23 | 135 |
| heuristic tuned, no safety rules | 0.461 | 0.401 | 0.25 | +0.081 [+0.006, +0.155] | 27 / 84 | 22 / 94 | 30 | 160 |
| learned HGB (pseudo + other real scans, LOSO) | 0.558 | 0.450 | 0.17 | +0.101 [+0.048, +0.161] | 26 / 84 | 1 / 94 | 28 | 120 |
| learned HGB, pseudo edges only | 0.562 | 0.433 | 0.22 | +0.109 [+0.056, +0.168] | 27 / 84 | 2 / 94 | 26 | 112 |
| learned logistic regression (pseudo + real LOSO) | 0.506 | 0.419 | 0.20 | +0.087 [+0.021, +0.155] | 27 / 84 | 12 / 94 | 26 | 139 |
| oracle merge (GT-same edges; ceiling of merging only) | 0.590 | 0.524 | 0.44 | +0.168 [+0.103, +0.238] | 40 / 84 | 5 / 94 | 39 | 152 |

**Reading (6 lines).**
1. On real votes an explicit merge does not beat the lift beyond noise: tuned heuristic -0.006 [-0.084, +0.061], learned HGB +0.010 [-0.036, +0.052]; untuned thresholds are harmful (-0.100 [-0.203, +0.001]), so thresholds matter and are unstable across 11 scans.
2. Only 5-6 of 24 fragmented items become whole (oracle merging of whole lift ids reaches 9), so even a perfect merge lifts inst mIoU only from 0.632 to 0.713; exact count goes 0.00 -> 0.45 (heuristic) / 0.55 (oracle).
3. Cost: tuned heuristic breaks 4 of 35 correct items, HGB 3, LR 5; every method fixes only 2-3 items. Safety rules matter little at tuned thresholds (33 vs 30 merges) but the untuned heuristic breaks 11.
4. Learned vs heuristic on real edges: LOSO precision/recall HGB 0.75/0.47, heuristic 0.79/0.40, LR 0.71/0.56; no learned arm is better in instance mIoU (CIs overlap); the pseudo-only classifier merges almost nothing on real edges (recall 0.11), so the fragmentation augmentation does not transfer. Colour (chi2, dE) dominates, share_small, hstep and normal angle follow.
5. On pseudo val the learned classifier does beat the heuristic (+0.101 [+0.048, +0.161] vs +0.051 [-0.013, +0.120]) with 1 broken items vs 21, but plane / k-means fragments are easier than SAM 3's.
6. Verdict: the CPU merge is an interpretable baseline that roughly equals the lift on real votes (no significant gain; 11 scans / 52 items); a learned refiner must beat about 0 +-0.06 per scan with < 4 broken items to be worth it. Limits: tiny real set, grid-selection noise, same-name flag unused, no per-item CI.
