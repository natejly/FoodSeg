

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
