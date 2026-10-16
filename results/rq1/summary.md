# RQ1 (synthetic): single overhead RGB-D frame vs multi-frame / full orbit scan

Reproduce: `conda run -n foodseg python -m pipeline.eval.rq1_frame_vs_scan` from the repo root. Segmentation = GT labels in every arm (oracle per-frame ids for frame arms, GT labels on the fused cloud for `full`). Methods: `plane` = height map vs the GT table plane; `container` = vs a support fitted from the arm's visible container points (GT container if < 300 points); `hull` = convex hull + drops onto that support; `gtcont` = diagnostic, vs the GT container surface. APE = |est - true| / true per item (mean over draws/phases of an arm); target is `volume_ml` (whole item incl. hidden parts).

## Reading: RQ1 on synthetic data (hard_v12, 20 scans, 177 items; GT labels everywhere)
1. Honest answer: on this data a full orbit scan does NOT estimate per-item volume better than one overhead frame once the volume is closed to the container (container method: median APE 48% overhead vs 51% K=24 (48% vs 48% on paired items), paired difference +0 pp, 95% CI [-10, +9]; same with ICP-refined poses, -3 [-15, +11]). Dish totals: overhead 30% vs K=24 35% (CI [-25, +10]).
2. What the scan changes is the bias, not the error: signed bias goes -33% (overhead) -> +3% (K=4) -> +25% (K=24); visibility of the GT surface rises 15% -> 30% -> 42% (noise-free top envelope: 60%). The negative bias of the single frame (occluded sides) and the positive bias of the visible envelope (voids in piles/leaves, stacked items; gt_top container: +25% bias, 32% APE) cancel at K=2-4.
3. Where extra views stop helping: per-item median APE is flat from K=2 (46%) through K=24 (51%); dish APE bottoms out at K=4 (18%) and then worsens (K=8 32%, K=24 35%) as bias turns positive. K=4 vs overhead: +6 pp [-4, +12] items, +8 pp [-6, +20] dish, i.e. not distinguishable. Fusing more than ~4 views buys visibility, not accuracy, with this estimator.
4. Choice of single frame matters more than the scan: the most top-down frame beats a random oblique one (48% vs 64% APE, bias -33% vs -58%; oblique - K24: +17 pp [+10, +22]); one oblique frame is clearly worse than the scan.
5. Plane closing (no container model) flips the story: overhead 39% APE (-19% bias) vs K=24 72% (+60%); a multi-view cloud closed to the table over-counts (hull even more: 88%), so the container support is what makes extra views usable. This is the same direction as the MF3D prior (one overhead frame + plane closing over-estimates there, +40%) but here the single frame is biased low because 85% of the GT surface is out of sight.
6. Caveat on the prior: MF3D's 9% carving needs 16 clean views and full-object coverage; here the target is whole-item volume incl. hidden parts, the clouds are noisy (LiDAR noise, persistent bias, pose drift) and items are piles/leaves/toppings, so no arm beats ~30% item-level (gt_top floor 32%), and means are dominated by loose leaves/toppings (+1000% or more).
7. Pose noise is not the reason: replacing exported poses by oracle point-to-plane ICP onto the GT surface changes K=24 container APE only 51% -> 54% and bias 25% -> 17%.
8. Limitations: segmentation is oracle in every arm (per-frame GT ids, minus a 1-px label/depth-edge gate), depth is noisy-but-synthetic LiDAR, 19% of overhead item rows have no estimate (<20 points; fully occluded items) and count as 0 in dish totals; only 5 of 20 hard_v12 scans are QC-ok (ok-only table, 26 items: container overhead 42% vs K=4 41% vs K=24 52%, same picture); no TSDF mesh, no shape completion (RQ3), no depth upsampling (smooth_depth/ is also 256x192 so no higher-res depth exists).
9. hard_v10/hard_v9 (8 clean-pose views, 51 / 68 items) give the same container-method picture (overhead 56% / 60% vs K=8 55% / 53%; CIs span 0) and the same plane-closing ranking, but their `full` arm (seg_points.npz) is unusable (61-68% of items missing), so only the K arms are comparable there.
10. Implication for the thesis: do not claim "scan beats single frame" from synthetic geometry alone. Claim instead (a) top-down frame + container model is a strong single-frame baseline, (b) a scan gives unbiased totals at K~4 but not lower per-item error, (c) the remaining error is hidden/void volume, which is RQ3 (completion), not more views.


# hard_v12
20 scans (5 ok-flagged), 177 items, 3 max draws per arm; overhead frame tilt from vertical: median 6 deg (max 10); single-frame container fit fell back to the GT container in 0% of item rows.

## Per-item volume error vs `volume_ml` (all scans); items with no estimate are 'missing'
| arm | method | n items | missing | med APE % | mean APE % | med bias % | mean bias % | GT-surface visibility % |
|---|---|---|---|---|---|---|---|---|
| overhead | plane | 177 | 19% | 39 | 155 | -19 | 106 | 15 |
| overhead | container | 177 | 19% | 48 | 100 | -33 | 29 | 15 |
| overhead | hull | 177 | 19% | 39 | 255 | -13 | 209 | 15 |
| overhead | gtcont | 177 | 19% | 51 | 113 | -35 | 40 | 15 |
| oblique | plane | 177 | 16% | 54 | 131 | -46 | 44 | 12 |
| oblique | container | 177 | 16% | 64 | 92 | -58 | -15 | 12 |
| oblique | hull | 177 | 16% | 56 | 214 | -42 | 135 | 12 |
| oblique | gtcont | 177 | 16% | 65 | 101 | -59 | -5 | 12 |
| K1 | plane | 177 | 15% | 57 | 123 | -45 | 35 | 13 |
| K1 | container | 177 | 15% | 64 | 85 | -60 | -23 | 13 |
| K1 | hull | 177 | 15% | 56 | 171 | -39 | 91 | 13 |
| K1 | gtcont | 177 | 15% | 65 | 97 | -59 | -10 | 13 |
| K2 | plane | 177 | 11% | 42 | 162 | -8 | 111 | 21 |
| K2 | container | 177 | 11% | 46 | 103 | -28 | 34 | 21 |
| K2 | hull | 177 | 11% | 50 | 251 | 3 | 208 | 21 |
| K2 | gtcont | 177 | 11% | 48 | 117 | -29 | 48 | 21 |
| K4 | plane | 177 | 11% | 57 | 201 | 29 | 169 | 30 |
| K4 | container | 177 | 11% | 48 | 127 | 3 | 81 | 30 |
| K4 | hull | 177 | 11% | 57 | 298 | 36 | 272 | 30 |
| K4 | gtcont | 177 | 11% | 47 | 138 | 0 | 90 | 30 |
| K8 | plane | 177 | 11% | 59 | 234 | 50 | 208 | 36 |
| K8 | container | 177 | 11% | 53 | 146 | 16 | 108 | 36 |
| K8 | hull | 177 | 11% | 72 | 343 | 60 | 323 | 36 |
| K8 | gtcont | 177 | 11% | 45 | 155 | 12 | 116 | 36 |
| K12 | plane | 177 | 10% | 65 | 243 | 54 | 217 | 38 |
| K12 | container | 177 | 10% | 50 | 152 | 20 | 114 | 38 |
| K12 | hull | 177 | 10% | 82 | 358 | 65 | 338 | 38 |
| K12 | gtcont | 177 | 10% | 48 | 160 | 19 | 123 | 38 |
| K24 | plane | 177 | 10% | 72 | 257 | 60 | 234 | 42 |
| K24 | container | 177 | 10% | 51 | 163 | 25 | 128 | 42 |
| K24 | hull | 177 | 10% | 88 | 386 | 79 | 369 | 42 |
| K24 | gtcont | 177 | 10% | 46 | 169 | 23 | 136 | 42 |
| full | plane | 177 | 13% | 40 | 164 | -18 | 109 | 20 |
| full | container | 177 | 13% | 62 | 101 | -49 | 7 | 20 |
| full | hull | 177 | 13% | 45 | 216 | -17 | 160 | 20 |
| full | gtcont | 177 | 13% | 59 | 113 | -48 | 23 | 20 |
| gt_top | plane | 177 | 0% | 77 | 229 | 77 | 227 | 60 |
| gt_top | container | 177 | 0% | 32 | 124 | 25 | 111 | 60 |
| gt_top | hull | 177 | 5% | 75 | 329 | 75 | 325 | 60 |
| gt_top | gtcont | 177 | 0% | 34 | 121 | 20 | 103 | 60 |

## Per-item volume error vs `volume_ml` (ok-flagged scans only); items with no estimate are 'missing'
| arm | method | n items | missing | med APE % | mean APE % | med bias % | mean bias % | GT-surface visibility % |
|---|---|---|---|---|---|---|---|---|
| overhead | plane | 26 | 0% | 45 | 81 | -10 | 41 | 20 |
| overhead | container | 26 | 0% | 42 | 60 | -36 | -11 | 20 |
| overhead | hull | 26 | 0% | 38 | 61 | -1 | 20 | 20 |
| overhead | gtcont | 26 | 0% | 39 | 61 | -30 | -2 | 20 |
| oblique | plane | 26 | 0% | 53 | 79 | -35 | 16 | 18 |
| oblique | container | 26 | 0% | 54 | 59 | -50 | -32 | 18 |
| oblique | hull | 26 | 0% | 42 | 55 | -19 | -2 | 18 |
| oblique | gtcont | 26 | 0% | 52 | 61 | -44 | -18 | 18 |
| K1 | plane | 26 | 0% | 55 | 82 | -36 | 13 | 19 |
| K1 | container | 26 | 0% | 61 | 64 | -59 | -33 | 19 |
| K1 | hull | 26 | 0% | 42 | 58 | -29 | -2 | 19 |
| K1 | gtcont | 26 | 0% | 55 | 67 | -44 | -20 | 19 |
| K2 | plane | 26 | 0% | 47 | 105 | 8 | 74 | 29 |
| K2 | container | 26 | 0% | 45 | 70 | -31 | 9 | 29 |
| K2 | hull | 26 | 0% | 32 | 81 | 11 | 53 | 29 |
| K2 | gtcont | 26 | 0% | 40 | 75 | -10 | 25 | 29 |
| K4 | plane | 26 | 0% | 67 | 140 | 46 | 126 | 39 |
| K4 | container | 26 | 0% | 41 | 86 | 1 | 52 | 39 |
| K4 | hull | 26 | 0% | 48 | 115 | 38 | 102 | 39 |
| K4 | gtcont | 26 | 0% | 31 | 93 | 5 | 65 | 39 |
| K8 | plane | 26 | 0% | 80 | 168 | 71 | 158 | 46 |
| K8 | container | 26 | 0% | 48 | 102 | 9 | 74 | 46 |
| K8 | hull | 26 | 0% | 62 | 142 | 59 | 133 | 46 |
| K8 | gtcont | 26 | 0% | 32 | 109 | 12 | 88 | 46 |
| K12 | plane | 26 | 0% | 82 | 178 | 76 | 168 | 49 |
| K12 | container | 26 | 0% | 50 | 106 | 11 | 79 | 49 |
| K12 | hull | 26 | 0% | 65 | 151 | 65 | 143 | 49 |
| K12 | gtcont | 26 | 0% | 30 | 115 | 15 | 96 | 49 |
| K24 | plane | 26 | 0% | 81 | 188 | 80 | 178 | 52 |
| K24 | container | 26 | 0% | 52 | 117 | 17 | 91 | 52 |
| K24 | hull | 26 | 0% | 73 | 166 | 73 | 159 | 52 |
| K24 | gtcont | 26 | 0% | 30 | 123 | 19 | 106 | 52 |
| full | plane | 26 | 0% | 47 | 119 | -14 | 71 | 26 |
| full | container | 26 | 0% | 52 | 80 | -49 | -10 | 26 |
| full | hull | 26 | 0% | 45 | 77 | -14 | 30 | 26 |
| full | gtcont | 26 | 0% | 49 | 84 | -43 | 5 | 26 |
| gt_top | plane | 26 | 0% | 89 | 152 | 89 | 152 | 58 |
| gt_top | container | 26 | 0% | 19 | 78 | 16 | 75 | 58 |
| gt_top | hull | 26 | 4% | 73 | 145 | 73 | 145 | 58 |
| gt_top | gtcont | 26 | 0% | 15 | 73 | 9 | 69 | 58 |

## By size: median APE % / median bias % (n items in the overhead arm)

plane
| arm | <10 mL | 10-50 mL | >50 mL |
|---|---|---|---|
| overhead | 55 / 14 (n=41) | 39 / -23 (n=91) | 37 / -16 (n=45) |
| oblique | 58 / -28 (n=41) | 55 / -50 (n=91) | 52 / -46 (n=45) |
| K4 | 80 / 48 (n=41) | 47 / 29 (n=91) | 50 / 19 (n=45) |
| K8 | 90 / 82 (n=41) | 55 / 50 (n=91) | 58 / 32 (n=45) |
| K24 | 112 / 112 (n=41) | 67 / 58 (n=91) | 57 / 43 (n=45) |
| full | 49 / -10 (n=41) | 40 / -21 (n=91) | 39 / -16 (n=45) |
| gt_top | 120 / 120 (n=41) | 76 / 76 (n=91) | 56 / 56 (n=45) |

container
| arm | <10 mL | 10-50 mL | >50 mL |
|---|---|---|---|
| overhead | 52 / -13 (n=41) | 47 / -33 (n=91) | 48 / -44 (n=45) |
| oblique | 70 / -44 (n=41) | 64 / -63 (n=91) | 63 / -59 (n=45) |
| K4 | 81 / 30 (n=41) | 40 / 4 (n=91) | 48 / -15 (n=45) |
| K8 | 93 / 63 (n=41) | 39 / 17 (n=91) | 48 / -6 (n=45) |
| K24 | 95 / 88 (n=41) | 44 / 26 (n=91) | 48 / -2 (n=45) |
| full | 92 / -55 (n=41) | 59 / -48 (n=91) | 51 / -49 (n=45) |
| gt_top | 39 / 27 (n=41) | 37 / 26 (n=91) | 28 / 16 (n=45) |

hull
| arm | <10 mL | 10-50 mL | >50 mL |
|---|---|---|---|
| overhead | 55 / -8 (n=41) | 42 / -20 (n=91) | 35 / -3 (n=45) |
| oblique | 71 / -47 (n=41) | 58 / -44 (n=91) | 42 / -25 (n=45) |
| K4 | 84 / 49 (n=41) | 47 / 36 (n=91) | 64 / 35 (n=45) |
| K8 | 98 / 91 (n=41) | 62 / 56 (n=91) | 78 / 54 (n=45) |
| K24 | 137 / 137 (n=41) | 80 / 73 (n=91) | 89 / 66 (n=45) |
| full | 78 / -32 (n=41) | 44 / -19 (n=91) | 35 / -3 (n=45) |
| gt_top | 77 / 77 (n=41) | 73 / 73 (n=91) | 71 / 71 (n=45) |

## By kind: median APE % / median bias % (n items in the overhead arm)

plane
| arm | pile | single | piece | topping | leaves |
|---|---|---|---|---|---|
| overhead | 32 / -20 (n=96) | 42 / -15 (n=59) | 48 / -37 (n=11) | 310 / 310 (n=6) | 2096 / 2096 (n=5) |
| oblique | 50 / -47 (n=96) | 54 / -47 (n=59) | 75 / -51 (n=11) | 255 / 255 (n=6) | 1677 / 1677 (n=5) |
| K4 | 55 / 29 (n=96) | 52 / 17 (n=59) | 38 / 6 (n=11) | 536 / 536 (n=6) | 2463 / 2463 (n=5) |
| K8 | 59 / 50 (n=96) | 57 / 34 (n=59) | 41 / 28 (n=11) | 695 / 695 (n=6) | 2834 / 2834 (n=5) |
| K24 | 74 / 63 (n=96) | 58 / 45 (n=59) | 44 / 44 (n=11) | 808 / 808 (n=6) | 3144 / 3144 (n=5) |
| full | 40 / -27 (n=96) | 39 / -18 (n=59) | 42 / -16 (n=11) | 536 / 536 (n=6) | 2274 / 2274 (n=5) |
| gt_top | 54 / 54 (n=96) | 91 / 91 (n=59) | 93 / 93 (n=11) | 570 / 570 (n=6) | 3099 / 3099 (n=5) |

container
| arm | pile | single | piece | topping | leaves |
|---|---|---|---|---|---|
| overhead | 40 / -34 (n=96) | 47 / -31 (n=59) | 63 / -63 (n=11) | 157 / 157 (n=6) | 1084 / 1084 (n=5) |
| oblique | 62 / -61 (n=96) | 61 / -56 (n=59) | 80 / -80 (n=11) | 79 / 71 (n=6) | 674 / 674 (n=5) |
| K4 | 45 / 6 (n=96) | 47 / -3 (n=59) | 42 / -42 (n=11) | 372 / 372 (n=6) | 1314 / 1314 (n=5) |
| K8 | 51 / 20 (n=96) | 46 / 4 (n=59) | 34 / -34 (n=11) | 525 / 525 (n=6) | 1400 / 1400 (n=5) |
| K24 | 53 / 30 (n=96) | 47 / 13 (n=59) | 42 / -32 (n=11) | 755 / 755 (n=6) | 1474 / 1474 (n=5) |
| full | 59 / -56 (n=96) | 53 / -44 (n=59) | 62 / -62 (n=11) | 134 / 134 (n=6) | 904 / 904 (n=5) |
| gt_top | 35 / 17 (n=96) | 31 / 28 (n=59) | 15 / 15 (n=11) | 299 / 299 (n=6) | 1466 / 1466 (n=5) |

hull
| arm | pile | single | piece | topping | leaves |
|---|---|---|---|---|---|
| overhead | 36 / -26 (n=96) | 40 / 19 (n=59) | 32 / -31 (n=11) | 159 / 159 (n=6) | 2392 / 2392 (n=5) |
| oblique | 56 / -53 (n=96) | 52 / -12 (n=59) | 55 / -48 (n=11) | 79 / 74 (n=6) | 1631 / 1631 (n=5) |
| K4 | 49 / 28 (n=96) | 75 / 58 (n=59) | 22 / 6 (n=11) | 504 / 504 (n=6) | 3222 / 3222 (n=5) |
| K8 | 64 / 50 (n=96) | 86 / 72 (n=59) | 24 / 24 (n=11) | 749 / 749 (n=6) | 3941 / 3941 (n=5) |
| K24 | 81 / 67 (n=96) | 92 / 89 (n=59) | 38 / 38 (n=11) | 1200 / 1200 (n=6) | 4898 / 4898 (n=5) |
| full | 42 / -34 (n=96) | 45 / 2 (n=59) | 27 / -14 (n=11) | 229 / 229 (n=6) | 1690 / 1690 (n=5) |
| gt_top | 52 / 49 (n=96) | 90 / 90 (n=59) | 84 / 84 (n=11) | 385 / 385 (n=6) | 3021 / 3021 (n=5) |

## Dish totals: median APE % / median bias % over scans (missing items count as 0)
| arm | plane | container | hull | gtcont |
|---|---|---|---|---|
| overhead | 18 / -13 | 30 / -27 | 24 / 8 | 30 / -27 |
| oblique | 46 / -42 | 54 / -54 | 31 / -18 | 54 / -49 |
| K1 | 44 / -43 | 51 / -51 | 24 / -16 | 51 / -50 |
| K2 | 13 / 1 | 23 / -20 | 32 / 18 | 20 / -16 |
| K4 | 29 / 29 | 18 / 3 | 59 / 59 | 15 / 5 |
| K8 | 42 / 42 | 32 / 14 | 80 / 80 | 23 / 14 |
| K12 | 46 / 46 | 32 / 19 | 85 / 85 | 23 / 18 |
| K24 | 51 / 51 | 35 / 22 | 93 / 93 | 27 / 25 |
| full | 24 / -6 | 38 / -37 | 18 / 3 | 35 / -35 |
| gt_top | 74 / 74 | 38 / 38 | 85 / 85 | 30 / 30 |

## APE vs number of frames K (evenly spaced, fused by 2 mm voxel; averaged over phases)
per-item median APE % / median bias %
| arm | plane | container | hull | visibility % (container rows) | dish APE % (container) |
|---|---|---|---|---|---|
| overhead | 39 / -19 | 48 / -33 | 39 / -13 | 15 | 30 |
| oblique | 54 / -46 | 64 / -58 | 56 / -42 | 12 | 54 |
| K1 | 57 / -45 | 64 / -60 | 56 / -39 | 13 | 51 |
| K2 | 42 / -8 | 46 / -28 | 50 / 3 | 21 | 23 |
| K4 | 57 / 29 | 48 / 3 | 57 / 36 | 30 | 18 |
| K8 | 59 / 50 | 53 / 16 | 72 / 60 | 36 | 32 |
| K12 | 65 / 54 | 50 / 20 | 82 / 65 | 38 | 32 |
| K24 | 72 / 60 | 51 / 25 | 88 / 79 | 42 | 35 |
| full | 40 / -18 | 62 / -49 | 45 / -17 | 20 | 38 |
| gt_top | 77 / 77 | 32 / 25 | 75 / 75 | 60 | 38 |

## Paired cluster bootstrap (scans resampled, 2000x): median APE(A) - median APE(B), percentage points
| A - B | method | paired items | med APE A / B | d med APE [95% CI] | d dish APE (mean over scans) [95% CI] |
|---|---|---|---|---|---|
| overhead - K24 | plane | 143 | 39 / 69 | -31 [-56, -19] | -42 [-58, -24] |
| overhead - K24 | container | 143 | 48 / 48 | +0 [-10, +9] | -6 [-25, +10] |
| overhead - K24 | hull | 143 | 39 / 92 | -52 [-70, -28] | -71 [-97, -45] |
| overhead - K4 | plane | 143 | 39 / 48 | -9 [-19, +0] | -16 [-28, -4] |
| overhead - K4 | container | 143 | 48 / 42 | +6 [-4, +12] | +8 [-6, +20] |
| overhead - K4 | hull | 143 | 39 / 51 | -12 [-29, -3] | -36 [-54, -18] |
| overhead - K8 | plane | 143 | 39 / 59 | -20 [-30, -10] | -30 [-44, -16] |
| overhead - K8 | container | 143 | 48 / 46 | +2 [-10, +11] | +1 [-15, +15] |
| overhead - K8 | hull | 143 | 39 / 71 | -31 [-56, -16] | -53 [-76, -31] |
| overhead - full | plane | 141 | 39 / 40 | -1 [-6, +6] | -6 [-14, +1] |
| overhead - full | container | 141 | 47 / 58 | -11 [-20, -0] | -4 [-11, +2] |
| overhead - full | hull | 141 | 39 / 42 | -3 [-9, +7] | +2 [-3, +7] |
| oblique - K24 | plane | 148 | 54 / 68 | -14 [-35, -1] | -28 [-54, -3] |
| oblique - K24 | container | 148 | 64 / 47 | +17 [+10, +22] | +11 [-10, +28] |
| oblique - K24 | hull | 148 | 56 / 89 | -32 [-51, -12] | -74 [-108, -41] |
| K4 - K24 | plane | 158 | 57 / 71 | -14 [-34, -1] | -25 [-32, -18] |
| K4 - K24 | container | 158 | 48 / 50 | -2 [-10, +3] | -14 [-21, -7] |
| K4 - K24 | hull | 158 | 57 / 87 | -30 [-41, -21] | -35 [-44, -26] |
| K8 - K24 | plane | 158 | 59 / 71 | -12 [-25, -3] | -11 [-14, -8] |
| K8 - K24 | container | 158 | 53 / 50 | +3 [-4, +7] | -7 [-11, -4] |
| K8 - K24 | hull | 158 | 72 / 87 | -14 [-21, -5] | -18 [-23, -13] |

## Visibility explains the error (container method; median signed bias % by GT-surface visibility)
| arm | <20% | 20-50% | >=50% |
|---|---|---|---|
| overhead | -49 (APE 52, n=123) | -11 (APE 37, n=51) | 1084 (APE 1084, n=3) |
| K4 | -68 (APE 72, n=55) | 12 (APE 37, n=100) | 212 (APE 212, n=22) |
| K24 | -78 (APE 78, n=37) | 6 (APE 41, n=67) | 55 (APE 62, n=73) |
| full | -64 (APE 65, n=100) | -36 (APE 52, n=69) | 167 (APE 167, n=8) |

# hard_v12+icp
20 scans (5 ok-flagged), 177 items, 3 max draws per arm; overhead frame tilt from vertical: median 6 deg (max 10); single-frame container fit fell back to the GT container in 0% of item rows.

## Per-item volume error vs `volume_ml` (all scans); items with no estimate are 'missing'
| arm | method | n items | missing | med APE % | mean APE % | med bias % | mean bias % | GT-surface visibility % |
|---|---|---|---|---|---|---|---|---|
| overhead | plane | 177 | 19% | 38 | 158 | -16 | 113 | 18 |
| overhead | container | 177 | 19% | 48 | 100 | -33 | 29 | 18 |
| overhead | hull | 177 | 19% | 39 | 255 | -12 | 208 | 18 |
| overhead | gtcont | 177 | 19% | 47 | 112 | -32 | 44 | 18 |
| oblique | plane | 177 | 16% | 55 | 132 | -47 | 46 | 14 |
| oblique | container | 177 | 16% | 64 | 92 | -58 | -16 | 14 |
| oblique | hull | 177 | 16% | 57 | 213 | -42 | 133 | 14 |
| oblique | gtcont | 177 | 16% | 64 | 101 | -58 | -3 | 14 |
| K1 | plane | 177 | 15% | 56 | 123 | -46 | 35 | 14 |
| K1 | container | 177 | 15% | 64 | 85 | -60 | -24 | 14 |
| K1 | hull | 177 | 15% | 56 | 168 | -39 | 88 | 14 |
| K1 | gtcont | 177 | 15% | 65 | 96 | -60 | -10 | 14 |
| K2 | plane | 177 | 11% | 40 | 157 | -9 | 104 | 23 |
| K2 | container | 177 | 11% | 45 | 100 | -29 | 27 | 23 |
| K2 | hull | 177 | 11% | 44 | 248 | -5 | 202 | 23 |
| K2 | gtcont | 177 | 11% | 44 | 112 | -28 | 40 | 23 |
| K4 | plane | 177 | 11% | 49 | 194 | 21 | 159 | 32 |
| K4 | container | 177 | 11% | 49 | 117 | -7 | 65 | 32 |
| K4 | hull | 177 | 11% | 51 | 290 | 28 | 261 | 32 |
| K4 | gtcont | 177 | 11% | 47 | 128 | -9 | 76 | 32 |
| K8 | plane | 177 | 11% | 59 | 225 | 36 | 196 | 37 |
| K8 | container | 177 | 11% | 49 | 129 | 7 | 84 | 37 |
| K8 | hull | 177 | 11% | 63 | 313 | 51 | 291 | 37 |
| K8 | gtcont | 177 | 11% | 46 | 143 | 4 | 100 | 37 |
| K12 | plane | 177 | 10% | 62 | 240 | 42 | 214 | 40 |
| K12 | container | 177 | 10% | 49 | 136 | 10 | 93 | 40 |
| K12 | hull | 177 | 10% | 68 | 335 | 53 | 314 | 40 |
| K12 | gtcont | 177 | 10% | 48 | 153 | 8 | 112 | 40 |
| K24 | plane | 177 | 10% | 70 | 260 | 50 | 236 | 43 |
| K24 | container | 177 | 10% | 54 | 148 | 17 | 109 | 43 |
| K24 | hull | 177 | 10% | 82 | 370 | 70 | 351 | 43 |
| K24 | gtcont | 177 | 10% | 49 | 165 | 18 | 128 | 43 |
| full | plane | 177 | 13% | 40 | 164 | -18 | 109 | 20 |
| full | container | 177 | 13% | 62 | 101 | -49 | 7 | 20 |
| full | hull | 177 | 13% | 45 | 216 | -17 | 160 | 20 |
| full | gtcont | 177 | 13% | 59 | 113 | -48 | 23 | 20 |
| gt_top | plane | 177 | 0% | 77 | 229 | 77 | 227 | 60 |
| gt_top | container | 177 | 0% | 32 | 124 | 25 | 111 | 60 |
| gt_top | hull | 177 | 5% | 75 | 329 | 75 | 325 | 60 |
| gt_top | gtcont | 177 | 0% | 34 | 121 | 20 | 103 | 60 |

## Dish totals: median APE % / median bias % over scans (missing items count as 0)
| arm | plane | container | hull | gtcont |
|---|---|---|---|---|
| overhead | 15 / -10 | 31 / -27 | 23 / 8 | 25 / -25 |
| oblique | 45 / -41 | 54 / -54 | 31 / -18 | 53 / -51 |
| K1 | 45 / -43 | 51 / -51 | 23 / -16 | 50 / -49 |
| K2 | 12 / -7 | 25 / -23 | 28 / 19 | 22 / -21 |
| K4 | 18 / 16 | 10 / -3 | 49 / 49 | 11 / -4 |
| K8 | 28 / 28 | 15 / 11 | 62 / 62 | 12 / 3 |
| K12 | 36 / 36 | 19 / 14 | 66 / 66 | 11 / 7 |
| K24 | 44 / 44 | 27 / 19 | 75 / 75 | 15 / 14 |
| full | 24 / -6 | 38 / -37 | 18 / 3 | 35 / -35 |
| gt_top | 74 / 74 | 38 / 38 | 85 / 85 | 30 / 30 |

## APE vs number of frames K (evenly spaced, fused by 2 mm voxel; averaged over phases)
per-item median APE % / median bias %
| arm | plane | container | hull | visibility % (container rows) | dish APE % (container) |
|---|---|---|---|---|---|
| overhead | 38 / -16 | 48 / -33 | 39 / -12 | 18 | 31 |
| oblique | 55 / -47 | 64 / -58 | 57 / -42 | 14 | 54 |
| K1 | 56 / -46 | 64 / -60 | 56 / -39 | 14 | 51 |
| K2 | 40 / -9 | 45 / -29 | 44 / -5 | 23 | 25 |
| K4 | 49 / 21 | 49 / -7 | 51 / 28 | 32 | 10 |
| K8 | 59 / 36 | 49 / 7 | 63 / 51 | 37 | 15 |
| K12 | 62 / 42 | 49 / 10 | 68 / 53 | 40 | 19 |
| K24 | 70 / 50 | 54 / 17 | 82 / 70 | 43 | 27 |
| full | 40 / -18 | 62 / -49 | 45 / -17 | 20 | 38 |
| gt_top | 77 / 77 | 32 / 25 | 75 / 75 | 60 | 38 |

## Paired cluster bootstrap (scans resampled, 2000x): median APE(A) - median APE(B), percentage points
| A - B | method | paired items | med APE A / B | d med APE [95% CI] | d dish APE (mean over scans) [95% CI] |
|---|---|---|---|---|---|
| overhead - K24 | plane | 143 | 38 / 69 | -31 [-46, -11] | -40 [-59, -23] |
| overhead - K24 | container | 143 | 48 / 51 | -3 [-15, +11] | -1 [-18, +13] |
| overhead - K24 | hull | 143 | 39 / 79 | -40 [-70, -20] | -64 [-87, -42] |
| overhead - K4 | plane | 143 | 38 / 44 | -6 [-16, +3] | -10 [-21, -1] |
| overhead - K4 | container | 143 | 48 / 45 | +4 [-4, +15] | +11 [-1, +21] |
| overhead - K4 | hull | 143 | 39 / 47 | -8 [-22, +5] | -28 [-42, -13] |
| overhead - K8 | plane | 143 | 38 / 56 | -18 [-31, -6] | -24 [-38, -12] |
| overhead - K8 | container | 143 | 48 / 44 | +5 [-7, +11] | +8 [-6, +19] |
| overhead - K8 | hull | 143 | 39 / 59 | -20 [-38, -8] | -43 [-61, -25] |
| overhead - full | plane | 141 | 37 / 40 | -2 [-10, +5] | -7 [-13, -1] |
| overhead - full | container | 141 | 47 / 58 | -11 [-20, -0] | -4 [-10, +2] |
| overhead - full | hull | 141 | 39 / 42 | -3 [-9, +7] | +2 [-4, +7] |
| oblique - K24 | plane | 148 | 55 / 68 | -13 [-22, +4] | -26 [-54, -1] |
| oblique - K24 | container | 148 | 64 / 51 | +13 [+5, +22] | +16 [-5, +32] |
| oblique - K24 | hull | 148 | 57 / 79 | -22 [-46, -4] | -66 [-95, -38] |
| K4 - K24 | plane | 158 | 49 / 69 | -21 [-32, -4] | -30 [-40, -21] |
| K4 - K24 | container | 158 | 49 / 54 | -5 [-14, +2] | -13 [-21, -4] |
| K4 - K24 | hull | 158 | 51 / 80 | -29 [-43, -16] | -37 [-47, -26] |
| K8 - K24 | plane | 158 | 59 / 69 | -11 [-17, +2] | -16 [-21, -11] |
| K8 - K24 | container | 158 | 49 / 54 | -5 [-10, +2] | -9 [-13, -5] |
| K8 - K24 | hull | 158 | 63 / 80 | -17 [-26, -8] | -21 [-27, -15] |

## Visibility explains the error (container method; median signed bias % by GT-surface visibility)
| arm | <20% | 20-50% | >=50% |
|---|---|---|---|
| overhead | -54 (APE 55, n=108) | -16 (APE 34, n=61) | 444 (APE 444, n=8) |
| K4 | -77 (APE 77, n=51) | -2 (APE 32, n=102) | 143 (APE 143, n=24) |
| K24 | -80 (APE 80, n=36) | 1 (APE 41, n=67) | 55 (APE 65, n=74) |
| full | -64 (APE 65, n=100) | -36 (APE 52, n=69) | 167 (APE 167, n=8) |

# hard_v10
20 scans (8 ok-flagged), 51 items, 3 max draws per arm; overhead frame tilt from vertical: median 8 deg (max 18); single-frame container fit fell back to the GT container in 0% of item rows.

## Per-item volume error vs `volume_ml` (all scans); items with no estimate are 'missing'
| arm | method | n items | missing | med APE % | mean APE % | med bias % | mean bias % | GT-surface visibility % |
|---|---|---|---|---|---|---|---|---|
| overhead | plane | 51 | 6% | 53 | 291 | 50 | 282 | 37 |
| overhead | container | 51 | 6% | 56 | 114 | -25 | 49 | 37 |
| overhead | hull | 51 | 6% | 49 | 226 | 17 | 186 | 37 |
| overhead | gtcont | 51 | 6% | 36 | 178 | 9 | 146 | 37 |
| oblique | plane | 51 | 4% | 45 | 223 | 20 | 196 | 30 |
| oblique | container | 51 | 4% | 61 | 99 | -51 | 8 | 30 |
| oblique | hull | 51 | 4% | 56 | 202 | -19 | 142 | 30 |
| oblique | gtcont | 51 | 4% | 51 | 144 | -19 | 87 | 30 |
| K1 | plane | 51 | 6% | 40 | 228 | 16 | 201 | 30 |
| K1 | container | 51 | 6% | 62 | 99 | -45 | 10 | 30 |
| K1 | hull | 51 | 6% | 51 | 203 | -17 | 148 | 30 |
| K1 | gtcont | 51 | 6% | 52 | 148 | -23 | 89 | 30 |
| K2 | plane | 51 | 2% | 58 | 284 | 53 | 275 | 41 |
| K2 | container | 51 | 2% | 59 | 113 | -26 | 48 | 41 |
| K2 | hull | 51 | 2% | 58 | 224 | 12 | 184 | 41 |
| K2 | gtcont | 51 | 2% | 39 | 173 | 3 | 144 | 41 |
| K4 | plane | 51 | 2% | 76 | 338 | 76 | 334 | 50 |
| K4 | container | 51 | 2% | 58 | 131 | -2 | 80 | 50 |
| K4 | hull | 51 | 2% | 56 | 254 | 33 | 223 | 50 |
| K4 | gtcont | 51 | 2% | 38 | 200 | 17 | 182 | 50 |
| K8 | plane | 51 | 2% | 84 | 362 | 84 | 359 | 54 |
| K8 | container | 51 | 2% | 55 | 137 | 4 | 92 | 54 |
| K8 | hull | 51 | 2% | 60 | 269 | 51 | 246 | 54 |
| K8 | gtcont | 51 | 2% | 47 | 212 | 18 | 198 | 54 |
| full | plane | 51 | 61% | 88 | 208 | -68 | 113 | 6 |
| full | container | 51 | 61% | 90 | 110 | -83 | -18 | 6 |
| full | hull | 51 | 61% | 75 | 319 | -13 | 256 | 6 |
| full | gtcont | 51 | 61% | 90 | 132 | -83 | 6 | 6 |
| gt_top | plane | 51 | 0% | 135 | 436 | 135 | 436 | 62 |
| gt_top | container | 51 | 0% | 21 | 220 | 21 | 218 | 62 |
| gt_top | hull | 51 | 0% | 97 | 402 | 97 | 402 | 62 |
| gt_top | gtcont | 51 | 0% | 20 | 217 | 18 | 215 | 62 |

## Per-item volume error vs `volume_ml` (ok-flagged scans only); items with no estimate are 'missing'
| arm | method | n items | missing | med APE % | mean APE % | med bias % | mean bias % | GT-surface visibility % |
|---|---|---|---|---|---|---|---|---|
| overhead | plane | 13 | 0% | 69 | 658 | 69 | 648 | 42 |
| overhead | container | 13 | 0% | 85 | 225 | -31 | 141 | 42 |
| overhead | hull | 13 | 0% | 61 | 548 | 8 | 485 | 42 |
| overhead | gtcont | 13 | 0% | 51 | 388 | 21 | 362 | 42 |
| oblique | plane | 13 | 0% | 68 | 559 | 58 | 542 | 36 |
| oblique | container | 13 | 0% | 89 | 199 | -37 | 109 | 36 |
| oblique | hull | 13 | 0% | 68 | 531 | -9 | 464 | 36 |
| oblique | gtcont | 13 | 0% | 68 | 350 | 4 | 305 | 36 |
| K1 | plane | 13 | 0% | 66 | 572 | 54 | 552 | 36 |
| K1 | container | 13 | 0% | 84 | 198 | -28 | 114 | 36 |
| K1 | hull | 13 | 0% | 66 | 525 | -4 | 473 | 36 |
| K1 | gtcont | 13 | 0% | 77 | 355 | 0 | 310 | 36 |
| K2 | plane | 13 | 0% | 70 | 657 | 70 | 648 | 44 |
| K2 | container | 13 | 0% | 82 | 231 | -25 | 153 | 44 |
| K2 | hull | 13 | 0% | 63 | 561 | 15 | 505 | 44 |
| K2 | gtcont | 13 | 0% | 37 | 395 | 11 | 371 | 44 |
| K4 | plane | 13 | 0% | 87 | 697 | 87 | 690 | 50 |
| K4 | container | 13 | 0% | 76 | 244 | -1 | 175 | 50 |
| K4 | hull | 13 | 0% | 62 | 597 | 32 | 548 | 50 |
| K4 | gtcont | 13 | 0% | 54 | 408 | 18 | 392 | 50 |
| K8 | plane | 13 | 0% | 90 | 711 | 90 | 704 | 53 |
| K8 | container | 13 | 0% | 80 | 247 | -8 | 189 | 53 |
| K8 | hull | 13 | 0% | 63 | 618 | 63 | 582 | 53 |
| K8 | gtcont | 13 | 0% | 59 | 419 | 20 | 404 | 53 |
| full | plane | 13 | 0% | 84 | 270 | -68 | 170 | 15 |
| full | container | 13 | 0% | 93 | 132 | -83 | 7 | 15 |
| full | hull | 13 | 0% | 73 | 434 | -9 | 379 | 15 |
| full | gtcont | 13 | 0% | 93 | 167 | -83 | 43 | 15 |
| gt_top | plane | 13 | 0% | 115 | 774 | 115 | 774 | 58 |
| gt_top | container | 13 | 0% | 11 | 410 | 11 | 410 | 58 |
| gt_top | hull | 13 | 0% | 72 | 817 | 72 | 817 | 58 |
| gt_top | gtcont | 13 | 0% | 11 | 409 | 11 | 409 | 58 |

## By size: median APE % / median bias % (n items in the overhead arm)

plane
| arm | <10 mL | 10-50 mL | >50 mL |
|---|---|---|---|
| overhead | 371 / 371 (n=12) | 34 / 17 (n=19) | 46 / 42 (n=20) |
| oblique | 132 / 132 (n=12) | 45 / -8 (n=19) | 30 / 4 (n=20) |
| K4 | 364 / 364 (n=12) | 47 / 47 (n=19) | 65 / 65 (n=20) |
| K8 | 446 / 446 (n=12) | 51 / 51 (n=19) | 68 / 68 (n=20) |
| full | 157 / 157 (n=12) | 175 / 175 (n=19) | 83 / -83 (n=20) |
| gt_top | 518 / 518 (n=12) | 135 / 135 (n=19) | 87 / 87 (n=20) |

container
| arm | <10 mL | 10-50 mL | >50 mL |
|---|---|---|---|
| overhead | 100 / 100 (n=12) | 42 / -15 (n=19) | 65 / -65 (n=20) |
| oblique | 54 / 5 (n=12) | 60 / -48 (n=19) | 74 / -74 (n=20) |
| K4 | 168 / 168 (n=12) | 54 / 3 (n=19) | 58 / -54 (n=20) |
| K8 | 192 / 192 (n=12) | 54 / 9 (n=19) | 52 / -36 (n=20) |
| full | 37 / -22 (n=12) | 96 / 58 (n=19) | 88 / -88 (n=20) |
| gt_top | 272 / 272 (n=12) | 21 / 21 (n=19) | 6 / 6 (n=20) |

hull
| arm | <10 mL | 10-50 mL | >50 mL |
|---|---|---|---|
| overhead | 208 / 208 (n=12) | 47 / 27 (n=19) | 49 / -42 (n=20) |
| oblique | 89 / 54 (n=12) | 53 / -4 (n=19) | 53 / -47 (n=20) |
| K4 | 149 / 149 (n=12) | 56 / 56 (n=19) | 53 / -7 (n=20) |
| K8 | 182 / 182 (n=12) | 63 / 63 (n=19) | 57 / 20 (n=20) |
| full | 222 / 222 (n=12) | 277 / 277 (n=19) | 33 / -33 (n=20) |
| gt_top | 516 / 516 (n=12) | 160 / 160 (n=19) | 58 / 58 (n=20) |

## By kind: median APE % / median bias % (n items in the overhead arm)

plane
| arm | pile | single | piece | topping | leaves |
|---|---|---|---|---|---|
| overhead | 37 / 32 (n=12) | 90 / 90 (n=8) | 42 / 37 (n=23) | 566 / 566 (n=4) | 1506 / 1506 (n=4) |
| oblique | 39 / 13 (n=12) | 54 / 54 (n=8) | 39 / -7 (n=23) | 400 / 400 (n=4) | 1054 / 1054 (n=4) |
| K4 | 109 / 109 (n=12) | 153 / 153 (n=8) | 51 / 51 (n=23) | 683 / 683 (n=4) | 1740 / 1740 (n=4) |
| K8 | 149 / 149 (n=12) | 165 / 165 (n=8) | 56 / 56 (n=23) | 724 / 724 (n=4) | 1922 / 1922 (n=4) |
| full | - / - (n=12) | 79 / -79 (n=8) | 84 / -83 (n=23) | 184 / 184 (n=4) | 205 / 205 (n=4) |
| gt_top | 180 / 180 (n=12) | 193 / 193 (n=8) | 90 / 90 (n=23) | 784 / 784 (n=4) | 2435 / 2435 (n=4) |

container
| arm | pile | single | piece | topping | leaves |
|---|---|---|---|---|---|
| overhead | 39 / -12 (n=12) | 36 / 6 (n=8) | 66 / -66 (n=23) | 111 / 111 (n=4) | 676 / 676 (n=4) |
| oblique | 56 / -55 (n=12) | 22 / -21 (n=8) | 76 / -76 (n=23) | 51 / 11 (n=4) | 449 / 449 (n=4) |
| K4 | 38 / 5 (n=12) | 63 / 58 (n=8) | 55 / -55 (n=23) | 146 / 146 (n=4) | 712 / 712 (n=4) |
| K8 | 29 / 14 (n=12) | 71 / 71 (n=8) | 50 / -50 (n=23) | 170 / 170 (n=4) | 738 / 738 (n=4) |
| full | - / - (n=12) | 90 / -90 (n=8) | 93 / -88 (n=23) | 16 / 16 (n=4) | 77 / -0 (n=4) |
| gt_top | 109 / 109 (n=12) | 106 / 106 (n=8) | 7 / 7 (n=23) | 495 / 495 (n=4) | 1200 / 1200 (n=4) |

hull
| arm | pile | single | piece | topping | leaves |
|---|---|---|---|---|---|
| overhead | 41 / -12 (n=12) | 54 / 28 (n=8) | 47 / -6 (n=23) | 100 / 100 (n=4) | 1290 / 1290 (n=4) |
| oblique | 59 / -52 (n=12) | 52 / 17 (n=8) | 51 / -34 (n=23) | 41 / 4 (n=4) | 1114 / 1114 (n=4) |
| K4 | 37 / 8 (n=12) | 83 / 82 (n=8) | 54 / 19 (n=23) | 149 / 149 (n=4) | 1430 / 1430 (n=4) |
| K8 | 20 / 18 (n=12) | 100 / 100 (n=8) | 57 / 26 (n=23) | 182 / 182 (n=4) | 1528 / 1528 (n=4) |
| full | - / - (n=12) | 39 / -33 (n=8) | 49 / -33 (n=23) | 77 / 77 (n=4) | 514 / 514 (n=4) |
| gt_top | 128 / 128 (n=12) | 164 / 164 (n=8) | 61 / 61 (n=23) | 531 / 531 (n=4) | 2370 / 2370 (n=4) |

## Dish totals: median APE % / median bias % over scans (missing items count as 0)
| arm | plane | container | hull | gtcont |
|---|---|---|---|---|
| overhead | 64 / 64 | 35 / -20 | 46 / 22 | 17 / 4 |
| oblique | 34 / 26 | 60 / -41 | 48 / -1 | 27 / -19 |
| K1 | 26 / 24 | 59 / -41 | 44 / -4 | 26 / -22 |
| K2 | 64 / 64 | 45 / -17 | 50 / 29 | 20 / 14 |
| K4 | 81 / 81 | 47 / -5 | 53 / 51 | 24 / 24 |
| K8 | 84 / 84 | 46 / -2 | 62 / 60 | 28 / 28 |
| full | 91 / -88 | 97 / -97 | 90 / -73 | 96 / -96 |
| gt_top | 127 / 127 | 29 / 29 | 94 / 94 | 27 / 27 |

## APE vs number of frames K (evenly spaced, fused by 2 mm voxel; averaged over phases)
per-item median APE % / median bias %
| arm | plane | container | hull | visibility % (container rows) | dish APE % (container) |
|---|---|---|---|---|---|
| overhead | 53 / 50 | 56 / -25 | 49 / 17 | 37 | 35 |
| oblique | 45 / 20 | 61 / -51 | 56 / -19 | 30 | 60 |
| K1 | 40 / 16 | 62 / -45 | 51 / -17 | 30 | 59 |
| K2 | 58 / 53 | 59 / -26 | 58 / 12 | 41 | 45 |
| K4 | 76 / 76 | 58 / -2 | 56 / 33 | 50 | 47 |
| K8 | 84 / 84 | 55 / 4 | 60 / 51 | 54 | 46 |
| full | 88 / -68 | 90 / -83 | 75 / -13 | 6 | 97 |
| gt_top | 135 / 135 | 21 / 21 | 97 / 97 | 62 | 29 |

## Paired cluster bootstrap (scans resampled, 2000x): median APE(A) - median APE(B), percentage points
| A - B | method | paired items | med APE A / B | d med APE [95% CI] | d dish APE (mean over scans) [95% CI] |
|---|---|---|---|---|---|
| overhead - K4 | plane | 48 | 53 / 76 | -23 [-62, -6] | -32 [-47, -20] |
| overhead - K4 | container | 48 | 56 / 58 | -2 [-25, +12] | -8 [-25, +5] |
| overhead - K4 | hull | 48 | 49 / 56 | -7 [-29, +4] | -27 [-52, -8] |
| overhead - K8 | plane | 48 | 53 / 78 | -25 [-86, -9] | -43 [-63, -27] |
| overhead - K8 | container | 48 | 56 / 56 | -0 [-29, +15] | -12 [-32, +5] |
| overhead - K8 | hull | 48 | 49 / 61 | -12 [-40, +2] | -45 [-82, -16] |
| overhead - full | plane | 20 | 68 / 88 | -20 [-43, +89] | +172 [+2, +415] |
| overhead - full | container | 20 | 79 / 90 | -11 [-61, +19] | +67 [-37, +212] |
| overhead - full | hull | 20 | 59 / 75 | -16 [-45, +34] | +139 [-1, +353] |

## Visibility explains the error (container method; median signed bias % by GT-surface visibility)
| arm | <20% | 20-50% | >=50% |
|---|---|---|---|
| overhead | -71 (APE 71, n=8) | -27 (APE 39, n=36) | 444 (APE 444, n=7) |
| K4 | -93 (APE 93, n=2) | -54 (APE 54, n=23) | 63 (APE 70, n=26) |
| full | -95 (APE 95, n=44) | 16 (APE 47, n=7) | - (APE -, n=0) |

# hard_v9
20 scans (5 ok-flagged), 68 items, 3 max draws per arm; overhead frame tilt from vertical: median 10 deg (max 19); single-frame container fit fell back to the GT container in 0% of item rows.

## Per-item volume error vs `volume_ml` (all scans); items with no estimate are 'missing'
| arm | method | n items | missing | med APE % | mean APE % | med bias % | mean bias % | GT-surface visibility % |
|---|---|---|---|---|---|---|---|---|
| overhead | plane | 68 | 7% | 50 | 343 | 47 | 332 | 36 |
| overhead | container | 68 | 7% | 60 | 139 | -11 | 77 | 36 |
| overhead | hull | 68 | 7% | 43 | 314 | 7 | 271 | 36 |
| overhead | gtcont | 68 | 7% | 32 | 185 | 1 | 153 | 36 |
| oblique | plane | 68 | 7% | 42 | 244 | 9 | 213 | 30 |
| oblique | container | 68 | 7% | 56 | 138 | -41 | 57 | 30 |
| oblique | hull | 68 | 7% | 38 | 299 | -11 | 249 | 30 |
| oblique | gtcont | 68 | 7% | 50 | 145 | -27 | 84 | 30 |
| K1 | plane | 68 | 7% | 43 | 233 | 9 | 203 | 29 |
| K1 | container | 68 | 7% | 58 | 127 | -42 | 45 | 29 |
| K1 | hull | 68 | 7% | 39 | 282 | -13 | 227 | 29 |
| K1 | gtcont | 68 | 7% | 44 | 135 | -19 | 79 | 29 |
| K2 | plane | 68 | 4% | 54 | 309 | 44 | 298 | 42 |
| K2 | container | 68 | 4% | 45 | 145 | -6 | 96 | 42 |
| K2 | hull | 68 | 4% | 40 | 314 | 26 | 287 | 42 |
| K2 | gtcont | 68 | 4% | 35 | 174 | 3 | 144 | 42 |
| K4 | plane | 68 | 4% | 66 | 386 | 66 | 380 | 51 |
| K4 | container | 68 | 4% | 49 | 188 | 12 | 151 | 51 |
| K4 | hull | 68 | 4% | 60 | 379 | 48 | 358 | 51 |
| K4 | gtcont | 68 | 4% | 41 | 217 | 10 | 198 | 51 |
| K8 | plane | 68 | 4% | 69 | 408 | 69 | 404 | 55 |
| K8 | container | 68 | 4% | 53 | 210 | 20 | 174 | 55 |
| K8 | hull | 68 | 4% | 70 | 416 | 57 | 397 | 55 |
| K8 | gtcont | 68 | 4% | 47 | 231 | 15 | 214 | 55 |
| full | plane | 68 | 68% | 86 | 84 | -60 | -18 | 4 |
| full | container | 68 | 68% | 88 | 78 | -82 | -66 | 4 |
| full | hull | 68 | 68% | 81 | 210 | -67 | 106 | 4 |
| full | gtcont | 68 | 68% | 89 | 80 | -81 | -63 | 4 |
| gt_top | plane | 68 | 0% | 104 | 564 | 104 | 564 | 64 |
| gt_top | container | 68 | 0% | 33 | 307 | 33 | 307 | 64 |
| gt_top | hull | 68 | 4% | 76 | 5897 | 76 | 5897 | 64 |
| gt_top | gtcont | 68 | 0% | 30 | 305 | 30 | 304 | 64 |

## Per-item volume error vs `volume_ml` (ok-flagged scans only); items with no estimate are 'missing'
| arm | method | n items | missing | med APE % | mean APE % | med bias % | mean bias % | GT-surface visibility % |
|---|---|---|---|---|---|---|---|---|
| overhead | plane | 8 | 0% | 131 | 470 | 131 | 470 | 48 |
| overhead | container | 8 | 0% | 71 | 241 | 3 | 204 | 48 |
| overhead | hull | 8 | 0% | 42 | 822 | 21 | 801 | 48 |
| overhead | gtcont | 8 | 0% | 47 | 306 | 1 | 282 | 48 |
| oblique | plane | 8 | 0% | 128 | 364 | 110 | 344 | 39 |
| oblique | container | 8 | 0% | 81 | 195 | -14 | 139 | 39 |
| oblique | hull | 8 | 0% | 51 | 790 | 18 | 758 | 39 |
| oblique | gtcont | 8 | 0% | 65 | 246 | -16 | 198 | 39 |
| K1 | plane | 8 | 0% | 129 | 393 | 109 | 372 | 39 |
| K1 | container | 8 | 0% | 81 | 208 | -15 | 152 | 39 |
| K1 | hull | 8 | 0% | 51 | 775 | 12 | 742 | 39 |
| K1 | gtcont | 8 | 0% | 63 | 261 | -15 | 213 | 39 |
| K2 | plane | 8 | 0% | 148 | 494 | 148 | 494 | 51 |
| K2 | container | 8 | 0% | 70 | 274 | 9 | 238 | 51 |
| K2 | hull | 8 | 0% | 48 | 857 | 35 | 837 | 51 |
| K2 | gtcont | 8 | 0% | 44 | 326 | 8 | 304 | 51 |
| K4 | plane | 8 | 0% | 177 | 547 | 177 | 547 | 60 |
| K4 | container | 8 | 0% | 68 | 322 | 22 | 288 | 60 |
| K4 | hull | 8 | 0% | 56 | 905 | 50 | 889 | 60 |
| K4 | gtcont | 8 | 0% | 39 | 371 | 18 | 352 | 60 |
| K8 | plane | 8 | 0% | 181 | 570 | 181 | 570 | 63 |
| K8 | container | 8 | 0% | 68 | 335 | 25 | 301 | 63 |
| K8 | hull | 8 | 0% | 56 | 938 | 52 | 923 | 63 |
| K8 | gtcont | 8 | 0% | 37 | 381 | 20 | 363 | 63 |
| full | plane | 8 | 0% | 91 | 116 | -47 | 22 | 10 |
| full | container | 8 | 0% | 92 | 81 | -85 | -55 | 10 |
| full | hull | 8 | 0% | 93 | 470 | -31 | 389 | 10 |
| full | gtcont | 8 | 0% | 96 | 87 | -84 | -48 | 10 |
| gt_top | plane | 8 | 0% | 194 | 657 | 194 | 657 | 63 |
| gt_top | container | 8 | 0% | 18 | 392 | 18 | 392 | 63 |
| gt_top | hull | 8 | 0% | 137 | 1128 | 137 | 1128 | 63 |
| gt_top | gtcont | 8 | 0% | 15 | 385 | 15 | 385 | 63 |

## By size: median APE % / median bias % (n items in the overhead arm)

plane
| arm | <10 mL | 10-50 mL | >50 mL |
|---|---|---|---|
| overhead | 329 / 329 (n=24) | 51 / 28 (n=20) | 32 / 28 (n=24) |
| oblique | 232 / 232 (n=24) | 41 / 1 (n=20) | 30 / -17 (n=24) |
| K4 | 497 / 497 (n=24) | 70 / 70 (n=20) | 34 / 34 (n=24) |
| K8 | 697 / 697 (n=24) | 74 / 74 (n=20) | 35 / 35 (n=24) |
| full | 67 / 49 (n=24) | 28 / -28 (n=20) | 88 / -88 (n=24) |
| gt_top | 955 / 955 (n=24) | 80 / 80 (n=20) | 75 / 75 (n=24) |

container
| arm | <10 mL | 10-50 mL | >50 mL |
|---|---|---|---|
| overhead | 158 / 158 (n=24) | 21 / -11 (n=20) | 49 / -49 (n=24) |
| oblique | 124 / 106 (n=24) | 42 / -27 (n=20) | 52 / -52 (n=24) |
| K4 | 409 / 409 (n=24) | 37 / 21 (n=20) | 20 / -6 (n=24) |
| K8 | 473 / 473 (n=24) | 38 / 26 (n=20) | 35 / -1 (n=24) |
| full | 71 / 14 (n=24) | 78 / -78 (n=20) | 94 / -94 (n=24) |
| gt_top | 667 / 667 (n=24) | 39 / 39 (n=20) | 6 / 6 (n=24) |

hull
| arm | <10 mL | 10-50 mL | >50 mL |
|---|---|---|---|
| overhead | 158 / 158 (n=24) | 18 / 10 (n=20) | 28 / -14 (n=24) |
| oblique | 285 / 285 (n=24) | 25 / -6 (n=20) | 29 / -29 (n=24) |
| K4 | 533 / 447 (n=24) | 53 / 53 (n=20) | 33 / 22 (n=24) |
| K8 | 692 / 692 (n=24) | 59 / 59 (n=20) | 49 / 30 (n=24) |
| full | 82 / 36 (n=24) | 93 / -49 (n=20) | 78 / -78 (n=24) |
| gt_top | 1082 / 1082 (n=24) | 79 / 79 (n=20) | 42 / 42 (n=24) |

## By kind: median APE % / median bias % (n items in the overhead arm)

plane
| arm | pile | single | piece | topping | leaves |
|---|---|---|---|---|---|
| overhead | 45 / 35 (n=14) | 178 / 178 (n=10) | 32 / 24 (n=30) | 758 / 758 (n=8) | 1813 / 1813 (n=6) |
| oblique | 26 / 9 (n=14) | 98 / 98 (n=10) | 32 / -20 (n=30) | 392 / 392 (n=8) | 1174 / 1174 (n=6) |
| K4 | 95 / 95 (n=14) | 250 / 250 (n=10) | 37 / 32 (n=30) | 656 / 656 (n=8) | 1914 / 1914 (n=6) |
| K8 | 103 / 103 (n=14) | 282 / 282 (n=10) | 35 / 34 (n=30) | 731 / 731 (n=8) | 1973 / 1973 (n=6) |
| full | 90 / -90 (n=14) | 8 / -3 (n=10) | 88 / -88 (n=30) | 144 / 144 (n=8) | 49 / 49 (n=6) |
| gt_top | 124 / 124 (n=14) | 251 / 251 (n=10) | 69 / 69 (n=30) | 1215 / 1215 (n=8) | 3080 / 3080 (n=6) |

container
| arm | pile | single | piece | topping | leaves |
|---|---|---|---|---|---|
| overhead | 20 / -7 (n=14) | 37 / 7 (n=10) | 42 / -42 (n=30) | 229 / 229 (n=8) | 818 / 818 (n=6) |
| oblique | 26 / -24 (n=14) | 55 / -17 (n=10) | 57 / -57 (n=30) | 213 / 213 (n=8) | 518 / 518 (n=6) |
| K4 | 58 / 45 (n=14) | 46 / 38 (n=10) | 20 / -11 (n=30) | 473 / 473 (n=8) | 872 / 872 (n=6) |
| K8 | 63 / 57 (n=14) | 43 / 38 (n=10) | 26 / -12 (n=30) | 600 / 600 (n=8) | 882 / 882 (n=6) |
| full | 95 / -95 (n=14) | 76 / -76 (n=10) | 94 / -94 (n=30) | 25 / 25 (n=8) | 71 / 14 (n=6) |
| gt_top | 56 / 56 (n=14) | 142 / 142 (n=10) | 8 / 8 (n=30) | 862 / 862 (n=8) | 1302 / 1302 (n=6) |

hull
| arm | pile | single | piece | topping | leaves |
|---|---|---|---|---|---|
| overhead | 13 / -0 (n=14) | 34 / 18 (n=10) | 32 / -2 (n=30) | 330 / 330 (n=8) | 2743 / 2743 (n=6) |
| oblique | 13 / -11 (n=14) | 50 / -8 (n=10) | 29 / -27 (n=30) | 287 / 287 (n=8) | 2571 / 2571 (n=6) |
| K4 | 60 / 57 (n=14) | 59 / 59 (n=10) | 46 / 26 (n=30) | 600 / 600 (n=8) | 2966 / 2966 (n=6) |
| K8 | 72 / 72 (n=14) | 61 / 61 (n=10) | 51 / 32 (n=30) | 791 / 791 (n=8) | 3082 / 3082 (n=6) |
| full | 93 / -93 (n=14) | 82 / -2 (n=10) | 78 / -78 (n=30) | 36 / 36 (n=8) | 478 / 478 (n=6) |
| gt_top | 96 / 96 (n=14) | 186 / 186 (n=10) | 48 / 48 (n=30) | 1088 / 1088 (n=8) | 4462 / 4462 (n=6) |

## Dish totals: median APE % / median bias % over scans (missing items count as 0)
| arm | plane | container | hull | gtcont |
|---|---|---|---|---|
| overhead | 56 / 55 | 24 / -17 | 52 / 6 | 12 / 2 |
| oblique | 19 / 8 | 45 / -42 | 30 / -11 | 32 / -30 |
| K1 | 23 / 5 | 45 / -41 | 31 / -17 | 35 / -30 |
| K2 | 50 / 48 | 19 / -11 | 37 / 20 | 10 / -0 |
| K4 | 74 / 70 | 30 / 8 | 53 / 45 | 13 / 9 |
| K8 | 79 / 75 | 40 / 14 | 68 / 56 | 16 / 12 |
| full | 90 / -90 | 98 / -98 | 94 / -93 | 95 / -95 |
| gt_top | 127 / 127 | 36 / 36 | 84 / 84 | 32 / 32 |

## APE vs number of frames K (evenly spaced, fused by 2 mm voxel; averaged over phases)
per-item median APE % / median bias %
| arm | plane | container | hull | visibility % (container rows) | dish APE % (container) |
|---|---|---|---|---|---|
| overhead | 50 / 47 | 60 / -11 | 43 / 7 | 36 | 24 |
| oblique | 42 / 9 | 56 / -41 | 38 / -11 | 30 | 45 |
| K1 | 43 / 9 | 58 / -42 | 39 / -13 | 29 | 45 |
| K2 | 54 / 44 | 45 / -6 | 40 / 26 | 42 | 19 |
| K4 | 66 / 66 | 49 / 12 | 60 / 48 | 51 | 30 |
| K8 | 69 / 69 | 53 / 20 | 70 / 57 | 55 | 40 |
| full | 86 / -60 | 88 / -82 | 81 / -67 | 4 | 98 |
| gt_top | 104 / 104 | 33 / 33 | 76 / 76 | 64 | 36 |

## Paired cluster bootstrap (scans resampled, 2000x): median APE(A) - median APE(B), percentage points
| A - B | method | paired items | med APE A / B | d med APE [95% CI] | d dish APE (mean over scans) [95% CI] |
|---|---|---|---|---|---|
| overhead - K4 | plane | 63 | 50 / 63 | -14 [-36, -0] | -42 [-71, -19] |
| overhead - K4 | container | 63 | 60 / 49 | +10 [-18, +18] | -53 [-145, +10] |
| overhead - K4 | hull | 63 | 43 / 59 | -16 [-32, +2] | -52 [-113, -11] |
| overhead - K8 | plane | 63 | 50 / 69 | -20 [-43, -6] | -57 [-98, -24] |
| overhead - K8 | container | 63 | 60 / 53 | +7 [-24, +13] | -73 [-195, +2] |
| overhead - K8 | hull | 63 | 43 / 69 | -26 [-41, -4] | -80 [-174, -21] |
| overhead - full | plane | 22 | 54 / 86 | -33 [-67, +86] | +51 [-42, +195] |
| overhead - full | container | 22 | 76 / 88 | -12 [-77, +21] | -17 [-70, +62] |
| overhead - full | hull | 22 | 71 / 81 | -10 [-71, +39] | +139 [-47, +476] |

## Visibility explains the error (container method; median signed bias % by GT-surface visibility)
| arm | <20% | 20-50% | >=50% |
|---|---|---|---|
| overhead | -77 (APE 77, n=13) | -12 (APE 35, n=44) | 637 (APE 637, n=11) |
| K4 | -88 (APE 88, n=5) | -21 (APE 31, n=31) | 76 (APE 76, n=32) |
| full | -94 (APE 94, n=63) | -37 (APE 46, n=5) | - (APE -, n=0) |

## Envelope vs solid targets, and mass

Reproduce: `conda run -n foodseg python -m pipeline.eval.rq1_envelope` (`--score-only` reuses results/rq1/envelope_items.csv). Per-item numbers: results/rq1/envelope_items.csv, all results: results/rq1/envelope.json. Envelope target = `volume_envelope_gt`; solid target = `volume_ml`; RQ1 container method, GT labels, ok+not-ok scans.

### Reading: envelope vs solid targets, and mass (hard_v12 primary, hard_v10 second point)
1. The K=24 over-count is definitional. The GT envelope (full noise-free GT top surface closed to the GT container, 2 mm column fill) is 1.21x the solid `volume_ml` (median; IQR 0.95-1.71; pooled 1.38x) on hard_v12 and 1.20x (pooled 1.44x) on hard_v10: piece 1.08/1.06, pile 1.18/1.93, single 1.28/2.08, topping 3.5/6.0, leaves 15/13 (v12/v10). Scored against the envelope, K=24 container-method bias goes +25% -> 0% and median item APE 51% -> 24% (dish 35% -> 14%); the noise-free `gt_top` floor goes 32% -> 3%.
2. K=4 being "unbiased" vs solid was a cancellation (occluded sides under-count, voids over-count). Vs the envelope the real picture is visible: overhead -48%, K2 -42%, K4 -16%, K8 -7%, K12 -4%, K24 0%; APE plateaus from K=4 (21%) to K=24 (24%).
3. The RQ1 answer changes for the envelope target: overhead - K24 = +26 pp [+20, +34] item APE and +30 pp [+25, +35] dish (hard_v12, 143 paired items; solid target: +0 [-10, +9]); overhead - K4 = +30 [+24, +34]; K4 vs K24 not distinguishable (-2 [-10, +5]). hard_v10 (K=8 max): overhead - K4 +20 [+4, +27]. The sparse `full` fused cloud (seg_points.npz) stays at 61% either way: its problem is coverage, not the target.
4. Target for the pipeline: use the envelope for anything measured by a scan (visible volume + completion), it is the quantity the geometry can actually reproduce (gt_top vs envelope 1-3% APE), and convert to mass with a BULK density; solid volume needs a void/solid-fraction model on top and a MATERIAL density. The two densities are different tables: material density is tight within a kind (CV 0.14 pile/piece, 0.08 leaves, 0.03 topping; single 0.33 because it mixes categories), bulk density is wide and batch-dependent (pile median 0.54 / 0.53 / 0.74 g/mL on v10 / v11 / v12; CV 0.4-1.9).
5. Mass, out-of-sample (per-kind medians fitted on hard_v10, scored on hard_v12; complete-mass dishes for dish rows): envelope x bulk gives K24 40% item APE [35, 50] (bias -23%), dish 26% [19, 45]; overhead 64% / 60%. Solid x material gives overhead 50% [42, 60] / dish 28% [12, 43], K4 50% / 35%, K24 63% / 53% (bias +42%). With bulk densities fitted on v12 itself (in-sample diagnostic) K24 envelope bias is -4% and dish 24%, so ~20 pp of the out-of-sample bias is density transfer, not geometry.
6. In mass terms scan-vs-single-frame is therefore target-dependent: via the envelope path overhead - K24 = +27 pp [+20, +33] item, +29 [+25, +33] dish (scan wins, plateau from K=8: item 42%, dish 32%); via the solid path no item-level difference (-11 [-28, +3]) and the single frame is better at dish level (-34 [-59, -10]) because its -33% volume under-count is offset by the 1.2-1.4x solid-vs-envelope gap and the +25% K24 over-count is not. That offset is luck, not a method: the solid-path numbers get worse as views are added.
7. The mass floor is the density table: the noise-free `gt_top` envelope with the same per-kind bulk density is 37% item / 19% dish APE. Envelope K>=8 (item 40-42%) is already at that floor, so more views or better geometry will not help mass; a better density model (per category, fill/stacking aware) will.
8. Hidden-volume correction (results/volume_head): it currently regresses log(V_true/V_vis) with V_true = solid `volume_ml`, so it has to learn two unrelated things at once, the occluded fraction and the void/solid fraction (which explains its failures on toppings and leaves: envelope/solid 3.5-15x there). It should predict the ENVELOPE (hidden-part completion only, the K24-vs-envelope bias of 0% shows the visible envelope is the right anchor) and treat solid fraction as a separate per-kind factor, only if solid volume is wanted at all. Retraining the head on `volume_envelope_gt` (results/rq1/envelope_items.csv) is the next step; it was not done here.
9. synth_volume headline (57 items / 26 ok v12 items, GT labels, sparse fused cloud): the target change barely helps there (foodscan median APE vs solid 54% / 43% -> vs envelope 76% / 38%; bias stays -41% / -38%) because occlusion of the sparse cloud, not the definition, dominates; `gt_top` is 28% / 15% vs solid and 1% vs envelope (estimator = envelope by construction).
10. Caveats: GT envelope depends on the closing radius (1.5x sample spacing, 1.5-5 mm, so gaps up to ~5-10 mm in a pile count as filled) and on the container-surface base (stacked items are charged for what is below, as the scan does); ratios < 1 (25% quantile 0.95; 0.79 for <10 mL) come from the 1 mm above-support cut and thin/small items; hull/solid is 1.65x and alpha-shape 1.2x (watertight on only 95/177 items). Densities are oracle per-kind medians from 51 items (v10); mass_g is integer-rounded for small items; bulk density medians per kind assume scans of the same food mix.


### GT envelope / solid ratio, hard (envelope = 2 mm column fill of the full GT top surface closed to the GT container)

| group | n | envelope/solid median | IQR | pooled (sum/sum) | hull/solid median | alpha/solid median |
|---|---|---|---|---|---|---|
| all | 17 | 1.77 | 1.09-3.35 | 1.34 | 2.33 | 1.59 (n=6) |
| kind=piece | 6 | 1.07 | 1.04-1.09 | 1.04 | 1.29 | 1.07 (n=1) |
| kind=pile | 8 | 2.96 | 1.91-3.47 | 3.28 | 3.42 | 2.06 (n=5) |
| kind=single | 2 | 1.79 | 1.78-1.80 | 1.80 | 2.75 | nan (n=0) |
| kind=topping | 1 | 10.02 | 10.02-10.02 | 10.02 | 10.60 | nan (n=0) |
| size <10 mL | 9 | 3.35 | 1.39-3.77 | 3.84 | 3.31 | 1.59 (n=4) |
| size 10-50 mL | 4 | 1.43 | 1.09-1.85 | 1.36 | 3.46 | 1.94 (n=2) |
| size >50 mL | 4 | 1.05 | 1.03-1.25 | 1.24 | 1.26 | nan (n=0) |


### GT envelope / solid ratio, hard_v10 (envelope = 2 mm column fill of the full GT top surface closed to the GT container)

| group | n | envelope/solid median | IQR | pooled (sum/sum) | hull/solid median | alpha/solid median |
|---|---|---|---|---|---|---|
| all | 51 | 1.20 | 1.05-3.18 | 1.44 | 1.83 | 2.21 (n=14) |
| kind=leaves | 4 | 12.98 | 8.59-20.95 | 16.57 | 21.48 | 10.61 (n=2) |
| kind=piece | 23 | 1.06 | 1.03-1.10 | 1.10 | 1.40 | 1.10 (n=2) |
| kind=pile | 12 | 1.93 | 1.06-3.79 | 1.22 | 2.30 | 2.09 (n=5) |
| kind=single | 8 | 2.08 | 1.61-2.30 | 1.84 | 2.41 | 1.94 (n=3) |
| kind=topping | 4 | 6.00 | 4.94-6.08 | 4.96 | 6.19 | 3.86 (n=2) |
| size <10 mL | 12 | 3.91 | 2.61-5.98 | 6.00 | 5.88 | 2.33 (n=7) |
| size 10-50 mL | 19 | 1.20 | 1.10-2.98 | 2.90 | 2.31 | 2.23 (n=6) |
| size >50 mL | 20 | 1.05 | 1.03-1.08 | 1.10 | 1.40 | 1.08 (n=1) |


### GT envelope / solid ratio, hard_v11 (envelope = 2 mm column fill of the full GT top surface closed to the GT container)

| group | n | envelope/solid median | IQR | pooled (sum/sum) | hull/solid median | alpha/solid median |
|---|---|---|---|---|---|---|
| all | 143 | 1.49 | 1.09-2.03 | 1.57 | 2.06 | 1.30 (n=65) |
| kind=leaves | 2 | 20.23 | 16.93-23.53 | 25.13 | 93.65 | nan (n=0) |
| kind=piece | 19 | 1.07 | 1.04-1.12 | 1.13 | 1.79 | nan (n=0) |
| kind=pile | 58 | 1.51 | 1.12-1.86 | 1.74 | 1.70 | 1.26 (n=31) |
| kind=single | 60 | 1.59 | 1.15-2.11 | 1.90 | 2.61 | 1.35 (n=31) |
| kind=topping | 4 | 5.05 | 3.75-81.49 | 4.56 | 7.18 | 3.90 (n=3) |
| size <10 mL | 33 | 1.25 | 0.93-2.24 | 1.59 | 1.72 | 1.22 (n=26) |
| size 10-50 mL | 75 | 1.61 | 1.22-2.04 | 2.15 | 2.34 | 1.36 (n=32) |
| size >50 mL | 35 | 1.10 | 1.04-1.67 | 1.41 | 1.80 | 1.39 (n=7) |


### GT envelope / solid ratio, hard_v12 (envelope = 2 mm column fill of the full GT top surface closed to the GT container)

| group | n | envelope/solid median | IQR | pooled (sum/sum) | hull/solid median | alpha/solid median |
|---|---|---|---|---|---|---|
| all | 177 | 1.21 | 0.95-1.71 | 1.38 | 1.65 | 1.19 (n=95) |
| kind=leaves | 5 | 15.49 | 14.31-32.81 | 18.65 | 30.76 | 5.70 (n=1) |
| kind=piece | 11 | 1.08 | 1.05-1.15 | 1.09 | 1.82 | 1.05 (n=4) |
| kind=pile | 96 | 1.18 | 0.88-1.65 | 1.41 | 1.43 | 1.13 (n=61) |
| kind=single | 59 | 1.28 | 0.97-1.72 | 1.28 | 1.86 | 1.26 (n=23) |
| kind=topping | 6 | 3.47 | 2.02-4.60 | 3.16 | 4.33 | 3.44 (n=6) |
| size <10 mL | 41 | 1.27 | 0.79-2.56 | 2.81 | 1.60 | 1.08 (n=35) |
| size 10-50 mL | 91 | 1.20 | 1.04-1.69 | 1.57 | 1.65 | 1.24 (n=42) |
| size >50 mL | 45 | 1.15 | 1.01-1.49 | 1.27 | 1.71 | 1.14 (n=18) |


### GT envelope / solid ratio, hard_v3 (envelope = 2 mm column fill of the full GT top surface closed to the GT container)

| group | n | envelope/solid median | IQR | pooled (sum/sum) | hull/solid median | alpha/solid median |
|---|---|---|---|---|---|---|
| all | 17 | 1.77 | 1.08-2.16 | 1.33 | 2.10 | 1.33 (n=6) |
| kind=piece | 6 | 1.06 | 1.04-1.08 | 1.04 | 1.21 | 1.08 (n=1) |
| kind=pile | 8 | 2.12 | 1.81-2.59 | 2.31 | 2.92 | 1.54 (n=5) |
| kind=single | 2 | 1.90 | 1.83-1.96 | 1.99 | 2.86 | nan (n=0) |
| kind=topping | 1 | 10.50 | 10.50-10.50 | 10.50 | 11.11 | nan (n=0) |
| size <10 mL | 8 | 2.32 | 1.81-3.07 | 4.08 | 3.18 | 1.33 (n=4) |
| size 10-50 mL | 5 | 1.11 | 1.08-1.77 | 1.33 | 2.40 | 1.95 (n=2) |
| size >50 mL | 4 | 1.04 | 1.03-1.29 | 1.26 | 1.20 | nan (n=0) |


### GT envelope / solid ratio, hard_v9 (envelope = 2 mm column fill of the full GT top surface closed to the GT container)

| group | n | envelope/solid median | IQR | pooled (sum/sum) | hull/solid median | alpha/solid median |
|---|---|---|---|---|---|---|
| all | 68 | 1.32 | 1.07-3.47 | 12.12 | 1.80 | 1.63 (n=11) |
| kind=leaves | 6 | 23.16 | 13.43-2551.18 | 1129.08 | 43.40 | 1.38 (n=1) |
| kind=piece | 30 | 1.07 | 1.04-1.09 | 1.06 | 1.41 | 1.09 (n=2) |
| kind=pile | 14 | 1.61 | 1.50-2.66 | 1.63 | 1.92 | 1.69 (n=3) |
| kind=single | 10 | 2.48 | 1.60-2.99 | 2.30 | 2.60 | 1.95 (n=3) |
| kind=topping | 8 | 9.44 | 6.95-11.35 | 8.62 | 11.67 | 5.27 (n=2) |
| size <10 mL | 24 | 8.39 | 2.99-12.92 | 462.60 | 10.81 | 1.95 (n=7) |
| size 10-50 mL | 20 | 1.41 | 1.11-1.61 | 1.63 | 1.80 | 1.30 (n=4) |
| size >50 mL | 24 | 1.05 | 1.03-1.08 | 1.06 | 1.33 | nan (n=0) |


### Re-scored RQ1 arms, hard_v12, container method: median APE % [95% scan-cluster CI] / median bias %; dish APE %

| arm | item APE vs solid | dish APE vs solid | item APE vs envelope | dish APE vs envelope |
|---|---|---|---|---|
| overhead | 48 [40, 55] / -33 | 30 | 48 [43, 54] / -48 | 46 |
| K2 | 46 [39, 55] / -28 | 23 | 43 [36, 47] / -42 | 37 |
| K4 | 48 [39, 54] / 3 | 18 | 21 [16, 31] / -16 | 23 |
| K8 | 53 [44, 58] / 16 | 32 | 22 [16, 28] / -7 | 17 |
| K12 | 50 [43, 59] / 20 | 32 | 22 [18, 32] / -4 | 14 |
| K24 | 51 [45, 60] / 25 | 35 | 24 [20, 30] / -0 | 14 |
| full | 62 [49, 69] / -49 | 38 | 61 [50, 70] / -61 | 49 |
| gt_top | 32 [26, 43] / 25 | 38 | 3 [2, 5] / 2 | 2 |


Paired bootstrap (container method), difference in median item APE, pp [95% CI]; dish APE difference

| target | A - B | paired items | med APE A / B | d med APE | d dish APE |
|---|---|---|---|---|---|
| solid | overhead-K24 | 143 | 48 / 48 | +0 [-10, +9] | -6 [-25, +10] |
| solid | overhead-K4 | 143 | 48 / 42 | +6 [-4, +12] | +8 [-6, +20] |
| solid | K4-K24 | 158 | 48 / 50 | -2 [-10, +3] | -14 [-21, -7] |
| solid | overhead-full | 141 | 47 / 58 | -11 [-20, -0] | -4 [-11, +2] |
| solid | K2-K24 | 157 | 46 / 50 | -4 [-12, +5] | -12 [-26, +1] |
| envelope_gt | overhead-K24 | 143 | 48 / 22 | +26 [+20, +34] | +30 [+25, +35] |
| envelope_gt | overhead-K4 | 143 | 48 / 18 | +30 [+24, +34] | +26 [+22, +30] |
| envelope_gt | K4-K24 | 158 | 21 / 23 | -2 [-10, +5] | +4 [-2, +9] |
| envelope_gt | overhead-full | 141 | 47 / 58 | -11 [-20, -1] | -2 [-8, +3] |
| envelope_gt | K2-K24 | 157 | 43 / 23 | +20 [+12, +25] | +20 [+14, +25] |


### Re-scored RQ1 arms, hard_v10, container method: median APE % [95% scan-cluster CI] / median bias %; dish APE %

| arm | item APE vs solid | dish APE vs solid | item APE vs envelope | dish APE vs envelope |
|---|---|---|---|---|
| overhead | 56 [36, 73] / -25 | 35 | 48 [34, 70] / -48 | 37 |
| K2 | 59 [37, 75] / -26 | 45 | 53 [29, 72] / -53 | 34 |
| K4 | 58 [47, 76] / -2 | 47 | 33 [14, 60] / -33 | 21 |
| K8 | 55 [38, 80] / 4 | 46 | 25 [10, 58] / -20 | 15 |
| full | 90 [59, 97] / -83 | 97 | 89 [80, 98] / -89 | 99 |
| gt_top | 21 [15, 99] / 21 | 29 | 2 [2, 3] / 1 | 2 |


Paired bootstrap (container method), difference in median item APE, pp [95% CI]; dish APE difference

| target | A - B | paired items | med APE A / B | d med APE | d dish APE |
|---|---|---|---|---|---|
| solid | overhead-K4 | 48 | 56 / 58 | -2 [-25, +12] | -8 [-25, +5] |
| solid | overhead-full | 20 | 79 / 90 | -11 [-61, +19] | +67 [-37, +212] |
| envelope_gt | overhead-K4 | 48 | 48 / 29 | +20 [+4, +27] | +12 [+8, +17] |
| envelope_gt | overhead-full | 20 | 75 / 89 | -13 [-60, +0] | -32 [-53, -14] |


### Densities (g/mL): bulk = mass / envelope, material = mass / solid (per kind; median and IQR)

| batch | kind | n | bulk density | material density |
|---|---|---|---|---|
| hard | pile | 8 | 0.44 (0.37-0.55), CV 0.53 | 1.23 (1.07-1.44), CV 0.23 |
| hard | single | 2 | 0.30 (0.28-0.32), CV 0.16 | 0.54 (0.50-0.57), CV 0.15 |
| hard | _all | 10 | 0.41 (0.32-0.49), CV 0.55 | 1.14 (0.71-1.41), CV 0.34 |
| hard_v10 | leaves | 4 | 0.01 (0.01-0.02), CV 0.53 | 0.15 (0.15-0.16), CV 0.08 |
| hard_v10 | piece | 23 | 0.84 (0.71-0.96), CV 0.25 | 0.90 (0.80-1.00), CV 0.18 |
| hard_v10 | pile | 12 | 0.54 (0.25-0.83), CV 0.59 | 0.90 (0.88-0.97), CV 0.30 |
| hard_v10 | single | 8 | 0.42 (0.27-0.67), CV 0.54 | 1.02 (0.62-1.07), CV 0.32 |
| hard_v10 | topping | 4 | 0.17 (0.17-0.26), CV 0.58 | 1.02 (1.02-1.04), CV 0.02 |
| hard_v10 | _all | 51 | 0.69 (0.24-0.89), CV 0.57 | 0.90 (0.77-1.02), CV 0.34 |
| hard_v11 | leaves | 2 | 0.01 (0.01-0.01), CV 0.33 | 0.15 (0.15-0.15), CV 0.00 |
| hard_v11 | piece | 19 | 0.80 (0.73-0.85), CV 0.20 | 0.90 (0.79-1.00), CV 0.14 |
| hard_v11 | pile | 58 | 0.53 (0.44-0.77), CV 0.42 | 0.85 (0.75-0.90), CV 0.14 |
| hard_v11 | single | 60 | 0.49 (0.35-0.70), CV 2.60 | 0.80 (0.63-1.07), CV 0.53 |
| hard_v11 | topping | 4 | 0.21 (0.12-0.27), CV 0.63 | 1.01 (1.01-1.02), CV 0.02 |
| hard_v11 | _all | 143 | 0.54 (0.38-0.79), CV 2.17 | 0.83 (0.73-0.95), CV 0.40 |
| hard_v12 | leaves | 5 | 0.01 (0.01-0.01), CV 0.86 | 0.17 (0.15-0.18), CV 0.08 |
| hard_v12 | piece | 11 | 0.86 (0.74-0.90), CV 0.13 | 0.90 (0.78-1.03), CV 0.14 |
| hard_v12 | pile | 96 | 0.74 (0.56-1.03), CV 1.56 | 0.90 (0.88-1.01), CV 0.14 |
| hard_v12 | single | 59 | 0.59 (0.36-0.88), CV 1.85 | 0.83 (0.59-0.95), CV 0.33 |
| hard_v12 | topping | 6 | 0.31 (0.22-0.55), CV 0.58 | 1.02 (1.01-1.04), CV 0.03 |
| hard_v12 | _all | 177 | 0.71 (0.45-0.95), CV 1.68 | 0.90 (0.75-1.01), CV 0.26 |
| hard_v3 | pile | 8 | 0.52 (0.50-0.67), CV 0.37 | 1.23 (1.07-1.44), CV 0.23 |
| hard_v3 | single | 2 | 0.29 (0.26-0.32), CV 0.21 | 0.54 (0.50-0.57), CV 0.15 |
| hard_v3 | _all | 10 | 0.51 (0.38-0.63), CV 0.44 | 1.14 (0.71-1.41), CV 0.34 |
| hard_v9 | leaves | 6 | 0.01 (0.00-0.01), CV 0.87 | 0.15 (0.15-0.15), CV 0.12 |
| hard_v9 | piece | 30 | 0.88 (0.76-0.95), CV 0.12 | 0.95 (0.80-1.00), CV 0.11 |
| hard_v9 | pile | 14 | 0.45 (0.25-0.57), CV 0.48 | 0.87 (0.67-0.90), CV 0.28 |
| hard_v9 | single | 10 | 0.39 (0.34-0.42), CV 0.81 | 0.93 (0.88-1.10), CV 0.27 |
| hard_v9 | topping | 8 | 0.10 (0.09-0.17), CV 0.93 | 0.98 (0.95-1.02), CV 0.04 |
| hard_v9 | _all | 68 | 0.63 (0.21-0.87), CV 0.63 | 0.93 (0.78-1.00), CV 0.32 |


### Mass, hard_v12 (container-method volume x per-kind median density fitted on hard_v10): median APE % [95% CI] / median bias % [CI]; dish APE % [CI] / dish bias %

| arm | item: envelope x bulk | dish: envelope x bulk | item: solid x material | dish: solid x material |
|---|---|---|---|---|
| overhead | 64 [59, 68] / -64 [-67, -56] | 60 [52, 70] / -60 | 50 [42, 60] / -30 [-41, -18] | 28 [12, 43] / -15 |
| K2 | 59 [54, 64] / -57 [-61, -52] | 56 [44, 60] / -56 | 48 [39, 63] / -19 [-31, -11] | 23 [9, 41] / 0 |
| K4 | 48 [43, 53] / -40 [-48, -36] | 42 [30, 51] / -42 | 50 [42, 65] / 14 [-3, 24] | 35 [20, 56] / 25 |
| K8 | 42 [38, 50] / -31 [-39, -22] | 32 [25, 48] / -32 | 58 [53, 66] / 31 [10, 53] | 45 [30, 63] / 35 |
| K12 | 40 [37, 51] / -31 [-37, -20] | 29 [21, 47] / -29 | 60 [54, 70] / 35 [13, 57] | 47 [35, 68] / 43 |
| K24 | 40 [35, 50] / -23 [-35, -13] | 26 [19, 45] / -26 | 63 [55, 81] / 42 [20, 64] | 53 [41, 77] / 49 |
| full | 70 [66, 77] / -69 [-75, -64] | 66 [57, 70] / -66 | 61 [52, 68] / -45 [-59, -30] | 34 [17, 44] / -26 |
| gt_top | 37 [31, 44] / -23 [-29, -11] | 19 [15, 26] / -17 | 44 [34, 67] / 37 [25, 60] | 68 [37, 84] / 68 |


Paired bootstrap in mass, pp [95% CI]

| density | A - B | paired items | med APE A / B | d med item APE | d dish APE |
|---|---|---|---|---|---|
| envelope_x_bulk | overhead-K24 | 143 | 64 / 37 | +27 [+20, +33] | +29 [+25, +33] |
| envelope_x_bulk | overhead-K4 | 143 | 64 / 45 | +20 [+15, +24] | +21 [+18, +25] |
| envelope_x_bulk | K4-K24 | 158 | 48 / 39 | +8 [+1, +12] | +8 [+4, +11] |
| envelope_x_bulk | overhead-full | 141 | 64 / 69 | -4 [-11, +1] | -2 [-7, +2] |
| solid_x_material | overhead-K24 | 143 | 50 / 61 | -11 [-28, +3] | -34 [-59, -10] |
| solid_x_material | overhead-K4 | 143 | 50 / 45 | +5 [-2, +15] | -15 [-33, +2] |
| solid_x_material | K4-K24 | 158 | 50 / 62 | -12 [-29, -2] | -19 [-27, -11] |
| solid_x_material | overhead-full | 141 | 50 / 59 | -8 [-16, +3] | -3 [-11, +5] |


### Mass, hard_v10 (container-method volume x per-kind median density fitted on hard_v10; IN-SAMPLE): median APE % [95% CI] / median bias % [CI]; dish APE % [CI] / dish bias %

| arm | item: envelope x bulk | dish: envelope x bulk | item: solid x material | dish: solid x material |
|---|---|---|---|---|
| overhead | 55 [41, 65] / -54 [-64, -39] | 55 [39, 69] / -55 | 56 [40, 74] / -20 [-40, -6] | 41 [21, 77] / -25 |
| K2 | 55 [35, 65] / -54 [-63, -22] | 50 [38, 67] / -50 | 55 [42, 84] / -19 [-36, 11] | 41 [18, 74] / -20 |
| K4 | 52 [34, 61] / -38 [-59, -14] | 42 [23, 55] / -42 | 60 [42, 86] / -8 [-30, 42] | 46 [20, 75] / -13 |
| K8 | 48 [30, 56] / -31 [-50, -11] | 32 [22, 52] / -32 | 58 [38, 85] / -2 [-19, 48] | 53 [19, 77] / -2 |
| K12 | - | - | - | - |
| K24 | - | - | - | - |
| full | 91 [81, 96] / -91 [-96, -81] | 99 [90, 100] / -99 | 87 [59, 95] / -82 [-94, -52] | 97 [87, 100] / -97 |
| gt_top | 27 [12, 38] / 2 [-12, 11] | 23 [11, 36] / 3 | 35 [19, 150] / 35 [16, 150] | 41 [19, 60] / 41 |


Paired bootstrap in mass, pp [95% CI]

| density | A - B | paired items | med APE A / B | d med item APE | d dish APE |
|---|---|---|---|---|---|
| envelope_x_bulk | overhead-K4 | 48 | 55 / 49 | +6 [-4, +17] | +11 [+6, +18] |
| envelope_x_bulk | overhead-full | 20 | 61 / 91 | -31 [-52, -4] | -29 [-47, -12] |
| solid_x_material | overhead-K4 | 48 | 56 / 61 | -6 [-28, +8] | -6 [-21, +8] |
| solid_x_material | overhead-full | 20 | 78 / 87 | -9 [-55, +18] | +57 [-37, +192] |


### synth_volume headline re-scored, results/synth_volume (ok scans, GT labels): median APE % [95% CI] / median bias %

| source/method | n items | vs solid | vs envelope |
|---|---|---|---|
| gt/foodscan | 57 | 54 [35, 82] / -41 | 76 [45, 88] / -76 |
| gt/voxel | 57 | 54 [32, 80] / -3 | 54 [38, 82] / -49 |
| gt/hull | 57 | 58 [35, 79] / +26 | 38 [17, 57] / -13 |
| gt_top/foodscan | 57 | 28 [10, 120] / +28 | 1 [1, 2] / -0 |
| gt_top/voxel | 57 | 35 [13, 122] / +35 | 2 [1, 3] / +2 |
| gt_top/hull | 57 | 74 [40, 149] / +74 | 23 [15, 37] / +23 |


### synth_volume headline re-scored, results/synth_volume/v12 (ok scans, GT labels): median APE % [95% CI] / median bias %

| source/method | n items | vs solid | vs envelope |
|---|---|---|---|
| gt/foodscan | 26 | 43 [21, 57] / -38 | 38 [16, 66] / -38 |
| gt/voxel | 26 | 39 [13, 53] / -19 | 33 [13, 53] / -32 |
| gt/hull | 26 | 45 [39, 51] / +39 | 23 [12, 33] / +1 |
| gt_top/foodscan | 26 | 15 [6, 28] / +9 | 1 [0, 1] / -0 |
| gt_top/voxel | 26 | 15 [8, 28] / +10 | 1 [1, 2] / +1 |
| gt_top/hull | 26 | 56 [50, 79] / +56 | 38 [32, 44] / +38 |


### Mass, hard_v12, densities fitted on hard_v12 itself (IN-SAMPLE diagnostic): item APE % / bias %; dish APE %

| arm | item: envelope x bulk | dish | item: solid x material | dish |
|---|---|---|---|---|
| overhead | 52 [48, 58] / -50 | 45 [37, 59] | 49 [42, 59] / -32 | 30 [21, 39] |
| K2 | 48 [43, 54] / -42 | 41 [26, 45] | 47 [38, 62] / -24 | 24 [11, 32] |
| K4 | 44 [36, 55] / -21 | 26 [20, 36] | 49 [41, 62] / 5 | 19 [13, 41] |
| K8 | 44 [34, 52] / -10 | 28 [11, 40] | 56 [45, 66] / 21 | 29 [16, 46] |
| K12 | 44 [35, 53] / -7 | 27 [9, 41] | 58 [50, 69] / 26 | 34 [19, 49] |
| K24 | 46 [38, 52] / -4 | 24 [10, 43] | 61 [51, 74] / 32 | 42 [19, 53] |
| full | 62 [56, 71] / -59 | 53 [41, 61] | 61 [48, 68] / -46 | 38 [27, 45] |
