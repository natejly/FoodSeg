# Synthetic per-item volume evaluation
Tables use **all** scans; table plane = **gt**; items with fewer than 20 points are 'missing'. Reproduce: `conda run -n foodseg python -m pipeline.eval.synth_volume --all` from the repo root.
## Reading
1. Segmentation is not the bottleneck; scan visibility and the volume step are. On the 76 items / 59 scans that have lift labels, foodscan's median item APE is 86% with GT labels and 86% with lift labels (segmentation adds almost nothing); per-dish totals 95% vs 95%. The noise-free GT surface (gt_top, oracle container pose) still gives 15% median / +10% bias on the same items.
2. Refined predictions (ptv3_short, only 3 scans / 4 paired items; no results/seg3d/v2_*/preds exist, only this run dumped predictions): foodscan median APE 34% with GT labels on the same voxel subset vs 33% with the predictions (dish 19% vs 20%); far too few scans to rank segmentation sources.
3. With GT labels (ok scans, n=285): median APE / median bias foodscan 76% / -71%, voxel 76% / -58%, hull 71% / -4% (mean APE 177%: the hull blows up on non-convex items), flat-table 77% / -26%. 61% of items are missing (<20 points) and 0% non-positive.
4. Where it fails 1, occlusion: items hidden >= 70% in the scan (n=237) come out -90% (median bias) with GT labels, yet the same items from the full GT surface give 50% APE: the integral is fine, the visible-only cloud has no height to integrate (column fill sums observed cells only, no completion).
5. Where it fails 2, stacking and voids: even from noise-free surfaces the container-relative column fill over-counts, +52% median bias in stacked scans (n=208) vs +33% in unstacked ones (n=88); an item sitting on another is charged for the item below, and loose piles / leaves / toppings are charged for their air gaps (gt_top median bias by kind: topping +725%, leaves +1471%, pile +70%, piece +4%).
6. By size (GT labels, foodscan): <10 mL median APE 55% (bias -26%), 10-50 mL 75%, >50 mL 87% (bias -87%): large items are the ones under-seen, small ones carry the 2 mm cell footprint bias (tests: +13% on a 1 cm cylinder, +4% on 3 cm) and the hull's +52% bias.
7. Registration: template ICP lands 1.5 mm median from the true pose but fails (>= 10 mm) on the sparse scans (94 of 285 ok items; run_meal's init is the centroid of all off-plane points, no multi-start, no fitness gate). Replacing ICP by the true pose changes the median APE only 76% -> 77%, so for these scans occlusion, not registration, is what hurts.
8. run_meal's item_volume crops the whole mesh by the item's bounding box, not by label; emulated on points (foodscan_bbox) the median APE goes 76% -> 81% (bias -71% -> -61%) because neighbours, cutlery and other items inside the box are integrated too.
9. Dish totals (what Nutrition5k-style scoring sees), foodscan: volume median APE 93% (bias -93%, n=71 dishes) with GT labels, 95% vs 95% GT vs lift labels on the 59 lift scans; under-seeing and over-counting partly cancel in the sum, so dish error is lower than item error. Mass with oracle density, 63 dishes with complete mass labels (volume APE 94% on them): 95% with the oracle category-median density, 94% with the oracle per-item density. Densities here are oracle, never predicted; mass_g is integer-rounded for small items.
10. Fix first: (1) get the hidden volume (shape completion / footprint-and-height prior per item, or more views) because visible-only height maps lose 90%+ on occluded items; (2) make stacked and loose items volume-consistent (subtract the supporting item's top, solid-fill factor for piles; compute dish totals as one union column map to avoid double counting); (3) crop by label instead of bbox in run_meal.item_volume; (4) give ContainerTemplate.register a container-label init and a fitness gate so failed registrations are flagged instead of silently used.

## Method definitions
- **foodscan**: GT container samples -> top-down envelope, densified to 2 mm (stand-in for an empty-container scan) -> `ContainerTemplate.from_points` -> `register` (ICP on the scene cloud) -> `Support` height field -> `points_above(1 mm)` -> `Support.height_map_volume` (point analogue of run_meal's mesh-prism integral; no TSDF mesh in the harness).
- **foodscan_gtreg**: same with the template at its true pose (isolates ICP). **foodscan_bbox**: integrates all above-support scene points inside the item's padded bbox, emulating `crop_mesh_to_points` (a bbox crop of the whole mesh, not a label crop).
- **hull**: convex hull of the above-support item points plus their drops onto the container surface. **voxel**: 2 mm column fill (closed + hole-filled footprint, mean top height, bottom = container surface). **table / hull_table**: same idea against the flat table plane (the no-container branch of run_meal / `volume.convex_hull_volume`).
- **Sources**: `gt_top` = noise-free GT surface samples, top surface per item (estimator floor); `gt` = GT labels on the fused scan cloud (sparse, noisy, occluded); `lift` = `lift_label` (SAM-mask votes), IoU-matched; `pred:<run>` = refiner predictions, `gtsub:<run>` = GT labels on the same voxel subset. Targets: `volume_ml` (whole item, including hidden parts); `volume_above_container_ml` is the volume above the container *rim plane* (itemstats.py), not above the container surface.
- APE = |est - gt| / gt per item; bias = signed (est - gt) / gt. 'missing' = no instance matched or fewer than the point minimum; '<=0' = estimate not positive. Mass uses **oracle density** (label/category known), never a predicted one.

## Data and counts (scans / items; ok-flag vs all)
| batch | scans | ok | items (all) | items (ok) | empty cloud | with lift |
|---|---|---|---|---|---|---|
| hard | 6 | 6 | 17 | 17 | 0 | 0 |
| hard_v3 | 6 | 6 | 17 | 17 | 0 | 0 |
| hard_v9 | 20 | 5 | 68 | 8 | 0 | 20 |
| hard_v10 | 20 | 8 | 51 | 13 | 0 | 20 |
| hard_v11 | 20 | 1 | 143 | 2 | 1 | 20 |
| **total ok** | 26 |  |  | 57 | 0 | 14 |
| **total all** | 72 |  | 296 |  | 1 | 60 |

Container registration (71 scans): ICP pose error vs truth median 1.48 mm (max 412.80); support height-field RMS vs oracle-pose support median 3.56 mm (max 1197.40). Scans with no support: 0.

## 1. Per-item volume error vs `volume_ml`
| source | method | n items | missing | <=0 | med APE % | mean APE % | med bias % | mean bias % |
|---|---|---|---|---|---|---|---|---|
| gt | foodscan | 285 | 61% | 0% | 76 | 92 | -71 | -21 |
| gt | voxel | 285 | 61% | 0% | 76 | 96 | -58 | -7 |
| gt | hull | 285 | 61% | 0% | 71 | 177 | -4 | 116 |
| gt | foodscan_cinit | 285 | 70% | 0% | 74 | 96 | -64 | -9 |
| gt | foodscan_gtreg | 285 | 61% | 0% | 77 | 93 | -67 | -17 |
| gt | foodscan_bbox | 285 | 61% | 0% | 81 | 126 | -61 | 23 |
| gt | table | 285 | 61% | 0% | 77 | 137 | -26 | 58 |
| gt | hull_table | 285 | 61% | 0% | 78 | 280 | 43 | 235 |
| gt_top | foodscan | 296 | 4% | 0% | 51 | 314 | 47 | 312 |
| gt_top | voxel | 296 | 4% | 0% | 51 | 322 | 50 | 320 |
| gt_top | hull | 296 | 5% | 0% | 97 | 1694 | 97 | 1694 |
| gt_top | foodscan_cinit | 296 | 4% | 0% | 51 | 314 | 47 | 312 |
| gt_top | foodscan_gtreg | 296 | 4% | 0% | 51 | 314 | 47 | 312 |
| gt_top | foodscan_bbox | 296 | 4% | 0% | 100 | 3189 | 100 | 3189 |
| gt_top | table | 296 | 0% | 0% | 126 | 498 | 126 | 498 |
| gt_top | hull_table | 296 | 2% | 0% | 146 | 1763 | 146 | 1763 |
| lift | foodscan | 251 | 70% | 0% | 86 | 93 | -85 | -51 |
| lift | voxel | 251 | 70% | 0% | 84 | 98 | -81 | -38 |
| lift | hull | 251 | 70% | 0% | 75 | 196 | -20 | 119 |
| lift | foodscan_cinit | 251 | 79% | 0% | 87 | 101 | -86 | -47 |
| lift | foodscan_gtreg | 251 | 70% | 0% | 86 | 93 | -82 | -47 |
| lift | foodscan_bbox | 251 | 70% | 0% | 86 | 96 | -82 | -43 |
| lift | table | 251 | 70% | 0% | 83 | 109 | -70 | -0 |
| lift | hull_table | 251 | 70% | 0% | 74 | 292 | -6 | 231 |
| gtsub:ptv3_short | foodscan | 5 | 0% | 0% | 43 | 172 | -24 | 125 |
| gtsub:ptv3_short | voxel | 5 | 0% | 0% | 41 | 209 | 1 | 191 |
| gtsub:ptv3_short | hull | 5 | 0% | 0% | 73 | 237 | 73 | 235 |
| gtsub:ptv3_short | foodscan_cinit | 5 | 0% | 0% | 44 | 174 | -25 | 126 |
| gtsub:ptv3_short | foodscan_gtreg | 5 | 0% | 0% | 44 | 174 | -26 | 125 |
| gtsub:ptv3_short | foodscan_bbox | 5 | 0% | 0% | 25 | 272 | -18 | 234 |
| gtsub:ptv3_short | table | 5 | 0% | 0% | 70 | 351 | 70 | 343 |
| gtsub:ptv3_short | hull_table | 5 | 0% | 0% | 153 | 395 | 153 | 395 |
| pred:ptv3_short | foodscan | 5 | 20% | 0% | 33 | 34 | -30 | -22 |
| pred:ptv3_short | voxel | 5 | 20% | 0% | 36 | 33 | -14 | -3 |
| pred:ptv3_short | hull | 5 | 20% | 0% | 46 | 58 | 46 | 54 |
| pred:ptv3_short | foodscan_cinit | 5 | 20% | 0% | 33 | 34 | -31 | -22 |
| pred:ptv3_short | foodscan_gtreg | 5 | 20% | 0% | 34 | 35 | -31 | -23 |
| pred:ptv3_short | foodscan_bbox | 5 | 20% | 0% | 25 | 30 | -22 | -18 |
| pred:ptv3_short | table | 5 | 20% | 0% | 43 | 62 | 37 | 53 |
| pred:ptv3_short | hull_table | 5 | 20% | 0% | 119 | 119 | 119 | 119 |

## 2. By item size, kind, container (primary methods)

**by size**

| source | size | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | <10 mL | foodscan | 85 | 69% | 55 | 127 | -26 |
| gt | <10 mL | hull | 85 | 69% | 74 | 294 | 52 |
| gt | <10 mL | voxel | 85 | 69% | 54 | 132 | -23 |
| gt | <10 mL | table | 85 | 69% | 63 | 264 | 59 |
| gt | 10-50 mL | foodscan | 113 | 76% | 75 | 92 | -49 |
| gt | 10-50 mL | hull | 113 | 76% | 92 | 306 | 32 |
| gt | 10-50 mL | voxel | 113 | 76% | 58 | 107 | -2 |
| gt | 10-50 mL | table | 113 | 76% | 60 | 154 | 17 |
| gt | >50 mL | foodscan | 87 | 34% | 87 | 75 | -87 |
| gt | >50 mL | hull | 87 | 34% | 66 | 62 | -41 |
| gt | >50 mL | voxel | 87 | 34% | 85 | 74 | -84 |
| gt | >50 mL | table | 87 | 34% | 84 | 72 | -82 |
| gt_top | <10 mL | foodscan | 86 | 1% | 190 | 815 | 190 |
| gt_top | <10 mL | hull | 86 | 6% | 317 | 5342 | 317 |
| gt_top | <10 mL | voxel | 86 | 1% | 203 | 827 | 203 |
| gt_top | <10 mL | table | 86 | 0% | 441 | 1221 | 441 |
| gt_top | 10-50 mL | foodscan | 123 | 8% | 56 | 156 | 56 |
| gt_top | 10-50 mL | hull | 123 | 9% | 129 | 312 | 129 |
| gt_top | 10-50 mL | voxel | 123 | 8% | 63 | 166 | 63 |
| gt_top | 10-50 mL | table | 123 | 0% | 123 | 269 | 123 |
| gt_top | >50 mL | foodscan | 87 | 0% | 4 | 30 | 4 |
| gt_top | >50 mL | hull | 87 | 0% | 50 | 77 | 50 |
| gt_top | >50 mL | voxel | 87 | 0% | 6 | 32 | 5 |
| gt_top | >50 mL | table | 87 | 0% | 85 | 108 | 85 |
| lift | <10 mL | foodscan | 68 | 85% | 70 | 77 | -49 |
| lift | <10 mL | hull | 68 | 85% | 79 | 452 | 26 |
| lift | <10 mL | voxel | 68 | 85% | 68 | 80 | -41 |
| lift | <10 mL | table | 68 | 85% | 72 | 114 | 60 |
| lift | 10-50 mL | foodscan | 104 | 83% | 84 | 132 | -81 |
| lift | 10-50 mL | hull | 104 | 83% | 93 | 407 | -7 |
| lift | 10-50 mL | voxel | 104 | 83% | 82 | 152 | -73 |
| lift | 10-50 mL | table | 104 | 83% | 79 | 205 | -44 |
| lift | >50 mL | foodscan | 79 | 39% | 91 | 82 | -91 |
| lift | >50 mL | hull | 79 | 39% | 70 | 64 | -54 |
| lift | >50 mL | voxel | 79 | 39% | 89 | 81 | -89 |
| lift | >50 mL | table | 79 | 39% | 84 | 72 | -83 |
| pred:ptv3_short | <10 mL | foodscan | 1 | 100% | - | - | - |
| pred:ptv3_short | <10 mL | hull | 1 | 100% | - | - | - |
| pred:ptv3_short | <10 mL | voxel | 1 | 100% | - | - | - |
| pred:ptv3_short | <10 mL | table | 1 | 100% | - | - | - |
| pred:ptv3_short | 10-50 mL | foodscan | 1 | 0% | 42 | 42 | -42 |
| pred:ptv3_short | 10-50 mL | hull | 1 | 0% | 132 | 132 | 132 |
| pred:ptv3_short | 10-50 mL | voxel | 1 | 0% | 32 | 32 | -32 |
| pred:ptv3_short | 10-50 mL | table | 1 | 0% | 6 | 6 | 6 |
| pred:ptv3_short | >50 mL | foodscan | 3 | 0% | 25 | 31 | -17 |
| pred:ptv3_short | >50 mL | hull | 3 | 0% | 20 | 34 | 20 |
| pred:ptv3_short | >50 mL | voxel | 3 | 0% | 41 | 34 | 4 |
| pred:ptv3_short | >50 mL | table | 3 | 0% | 68 | 80 | 68 |

**by kind**

| source | kind | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | leaves | foodscan | 12 | 25% | 80 | 187 | -16 |
| gt | leaves | hull | 12 | 25% | 385 | 1151 | 385 |
| gt | leaves | voxel | 12 | 25% | 82 | 246 | -8 |
| gt | piece | foodscan | 84 | 37% | 80 | 67 | -80 |
| gt | piece | hull | 84 | 37% | 58 | 56 | -14 |
| gt | piece | voxel | 84 | 37% | 80 | 64 | -77 |
| gt | pile | foodscan | 90 | 78% | 49 | 56 | -36 |
| gt | pile | hull | 90 | 78% | 74 | 95 | -37 |
| gt | pile | voxel | 90 | 78% | 54 | 60 | -39 |
| gt | single | foodscan | 81 | 70% | 78 | 69 | -78 |
| gt | single | hull | 81 | 70% | 72 | 90 | 15 |
| gt | single | voxel | 81 | 70% | 77 | 68 | -74 |
| gt | topping | foodscan | 18 | 78% | 547 | 525 | 547 |
| gt | topping | hull | 18 | 78% | 514 | 505 | 514 |
| gt | topping | voxel | 18 | 78% | 554 | 527 | 554 |
| gt_top | leaves | foodscan | 12 | 0% | 1471 | 1793 | 1471 |
| gt_top | leaves | hull | 12 | 8% | 3834 | 36075 | 3834 |
| gt_top | leaves | voxel | 12 | 0% | 1494 | 1845 | 1494 |
| gt_top | piece | foodscan | 84 | 0% | 4 | 13 | 4 |
| gt_top | piece | hull | 84 | 0% | 40 | 69 | 40 |
| gt_top | piece | voxel | 84 | 0% | 6 | 15 | 6 |
| gt_top | pile | foodscan | 100 | 10% | 70 | 126 | 70 |
| gt_top | pile | hull | 100 | 12% | 89 | 168 | 89 |
| gt_top | pile | voxel | 100 | 10% | 67 | 132 | 67 |
| gt_top | single | foodscan | 82 | 1% | 70 | 134 | 68 |
| gt_top | single | hull | 82 | 4% | 158 | 308 | 158 |
| gt_top | single | voxel | 82 | 1% | 75 | 144 | 75 |
| gt_top | topping | foodscan | 18 | 0% | 725 | 2482 | 725 |
| gt_top | topping | hull | 18 | 0% | 769 | 1808 | 769 |
| gt_top | topping | voxel | 18 | 0% | 738 | 2497 | 738 |
| lift | leaves | foodscan | 12 | 25% | 80 | 184 | -18 |
| lift | leaves | hull | 12 | 25% | 385 | 1141 | 385 |
| lift | leaves | voxel | 12 | 25% | 82 | 239 | -10 |
| lift | piece | foodscan | 72 | 44% | 91 | 81 | -91 |
| lift | piece | hull | 72 | 44% | 67 | 63 | -37 |
| lift | piece | voxel | 72 | 44% | 89 | 79 | -89 |
| lift | pile | foodscan | 74 | 92% | 89 | 83 | -89 |
| lift | pile | hull | 74 | 92% | 84 | 75 | -84 |
| lift | pile | voxel | 74 | 92% | 87 | 79 | -87 |
| lift | single | foodscan | 77 | 75% | 83 | 77 | -83 |
| lift | single | hull | 77 | 75% | 66 | 81 | -18 |
| lift | single | voxel | 77 | 75% | 79 | 75 | -79 |
| lift | topping | foodscan | 16 | 88% | 97 | 97 | 41 |
| lift | topping | hull | 16 | 88% | 63 | 63 | 7 |
| lift | topping | voxel | 16 | 88% | 104 | 104 | 49 |
| pred:ptv3_short | piece | foodscan | 2 | 0% | 34 | 34 | -34 |
| pred:ptv3_short | piece | hull | 2 | 0% | 14 | 14 | 6 |
| pred:ptv3_short | piece | voxel | 2 | 0% | 22 | 22 | -18 |
| pred:ptv3_short | single | foodscan | 2 | 0% | 33 | 33 | -9 |
| pred:ptv3_short | single | hull | 2 | 0% | 102 | 102 | 102 |
| pred:ptv3_short | single | voxel | 2 | 0% | 44 | 44 | 13 |
| pred:ptv3_short | topping | foodscan | 1 | 100% | - | - | - |
| pred:ptv3_short | topping | hull | 1 | 100% | - | - | - |
| pred:ptv3_short | topping | voxel | 1 | 100% | - | - | - |

**by container**

| source | container | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | bowl | foodscan | 105 | 43% | 63 | 107 | -46 |
| gt | bowl | hull | 105 | 43% | 84 | 197 | 14 |
| gt | bowl | voxel | 105 | 43% | 64 | 115 | -39 |
| gt | plate | foodscan | 180 | 72% | 82 | 74 | -81 |
| gt | plate | hull | 180 | 72% | 65 | 153 | -24 |
| gt | plate | voxel | 180 | 72% | 79 | 73 | -77 |
| gt_top | bowl | foodscan | 116 | 9% | 101 | 337 | 101 |
| gt_top | bowl | hull | 116 | 11% | 149 | 3861 | 149 |
| gt_top | bowl | voxel | 116 | 9% | 110 | 348 | 110 |
| gt_top | plate | foodscan | 180 | 0% | 35 | 301 | 31 |
| gt_top | plate | hull | 180 | 2% | 82 | 433 | 82 |
| gt_top | plate | voxel | 180 | 0% | 38 | 307 | 33 |
| lift | bowl | foodscan | 81 | 54% | 85 | 100 | -78 |
| lift | bowl | hull | 81 | 54% | 86 | 209 | 7 |
| lift | bowl | voxel | 81 | 54% | 84 | 110 | -76 |
| lift | plate | foodscan | 170 | 77% | 86 | 87 | -86 |
| lift | plate | hull | 170 | 77% | 70 | 184 | -58 |
| lift | plate | voxel | 170 | 77% | 84 | 86 | -83 |
| pred:ptv3_short | bowl | foodscan | 3 | 33% | 30 | 30 | -30 |
| pred:ptv3_short | bowl | hull | 3 | 33% | 76 | 76 | 76 |
| pred:ptv3_short | bowl | voxel | 3 | 33% | 18 | 18 | -14 |
| pred:ptv3_short | plate | foodscan | 2 | 0% | 38 | 38 | -13 |
| pred:ptv3_short | plate | hull | 2 | 0% | 40 | 40 | 32 |
| pred:ptv3_short | plate | voxel | 2 | 0% | 49 | 49 | 8 |

**by asset_source**

| source | asset_source | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | metafood3d | foodscan | 144 | 71% | 67 | 62 | -64 |
| gt | metafood3d | hull | 144 | 71% | 73 | 93 | 1 |
| gt | metafood3d | voxel | 144 | 71% | 67 | 63 | -51 |
| gt | primitive | foodscan | 23 | 100% | - | - | - |
| gt | primitive | hull | 23 | 100% | - | - | - |
| gt | primitive | voxel | 23 | 100% | - | - | - |
| gt | procedural_leaves | foodscan | 12 | 25% | 80 | 187 | -16 |
| gt | procedural_leaves | hull | 12 | 25% | 385 | 1151 | 385 |
| gt | procedural_leaves | voxel | 12 | 25% | 82 | 246 | -8 |
| gt | procedural_piece | foodscan | 84 | 37% | 80 | 67 | -80 |
| gt | procedural_piece | hull | 84 | 37% | 58 | 56 | -14 |
| gt | procedural_piece | voxel | 84 | 37% | 80 | 64 | -77 |
| gt | procedural_topping | foodscan | 18 | 78% | 547 | 525 | 547 |
| gt | procedural_topping | hull | 18 | 78% | 514 | 505 | 514 |
| gt | procedural_topping | voxel | 18 | 78% | 554 | 527 | 554 |
| gt | ycb | foodscan | 4 | 50% | 88 | 88 | -88 |
| gt | ycb | hull | 4 | 50% | 87 | 87 | -87 |
| gt | ycb | voxel | 4 | 50% | 88 | 88 | -88 |
| gt_top | metafood3d | foodscan | 154 | 6% | 80 | 143 | 76 |
| gt_top | metafood3d | hull | 154 | 6% | 143 | 263 | 143 |
| gt_top | metafood3d | voxel | 154 | 6% | 84 | 151 | 84 |
| gt_top | primitive | foodscan | 23 | 0% | 31 | 55 | 31 |
| gt_top | primitive | hull | 23 | 17% | 36 | 44 | 36 |
| gt_top | primitive | voxel | 23 | 0% | 36 | 56 | 36 |
| gt_top | procedural_leaves | foodscan | 12 | 0% | 1471 | 1793 | 1471 |
| gt_top | procedural_leaves | hull | 12 | 8% | 3834 | 36075 | 3834 |
| gt_top | procedural_leaves | voxel | 12 | 0% | 1494 | 1845 | 1494 |
| gt_top | procedural_piece | foodscan | 84 | 0% | 4 | 13 | 4 |
| gt_top | procedural_piece | hull | 84 | 0% | 40 | 69 | 40 |
| gt_top | procedural_piece | voxel | 84 | 0% | 6 | 15 | 6 |
| gt_top | procedural_topping | foodscan | 18 | 0% | 725 | 2482 | 725 |
| gt_top | procedural_topping | hull | 18 | 0% | 769 | 1808 | 769 |
| gt_top | procedural_topping | voxel | 18 | 0% | 738 | 2497 | 738 |
| gt_top | ycb | foodscan | 5 | 20% | 67 | 98 | 67 |
| gt_top | ycb | hull | 5 | 20% | 72 | 102 | 72 |
| gt_top | ycb | voxel | 5 | 20% | 69 | 101 | 69 |
| lift | metafood3d | foodscan | 124 | 81% | 85 | 78 | -85 |
| lift | metafood3d | hull | 124 | 81% | 70 | 78 | -38 |
| lift | metafood3d | voxel | 124 | 81% | 79 | 75 | -79 |
| lift | primitive | foodscan | 23 | 100% | - | - | - |
| lift | primitive | hull | 23 | 100% | - | - | - |
| lift | primitive | voxel | 23 | 100% | - | - | - |
| lift | procedural_leaves | foodscan | 12 | 25% | 80 | 184 | -18 |
| lift | procedural_leaves | hull | 12 | 25% | 385 | 1141 | 385 |
| lift | procedural_leaves | voxel | 12 | 25% | 82 | 239 | -10 |
| lift | procedural_piece | foodscan | 72 | 44% | 91 | 81 | -91 |
| lift | procedural_piece | hull | 72 | 44% | 67 | 63 | -37 |
| lift | procedural_piece | voxel | 72 | 44% | 89 | 79 | -89 |
| lift | procedural_topping | foodscan | 16 | 88% | 97 | 97 | 41 |
| lift | procedural_topping | hull | 16 | 88% | 63 | 63 | 7 |
| lift | procedural_topping | voxel | 16 | 88% | 104 | 104 | 49 |
| lift | ycb | foodscan | 4 | 50% | 88 | 88 | -88 |
| lift | ycb | hull | 4 | 50% | 87 | 87 | -87 |
| lift | ycb | voxel | 4 | 50% | 88 | 88 | -88 |
| pred:ptv3_short | metafood3d | foodscan | 2 | 0% | 33 | 33 | -9 |
| pred:ptv3_short | metafood3d | hull | 2 | 0% | 102 | 102 | 102 |
| pred:ptv3_short | metafood3d | voxel | 2 | 0% | 44 | 44 | 13 |
| pred:ptv3_short | procedural_piece | foodscan | 2 | 0% | 34 | 34 | -34 |
| pred:ptv3_short | procedural_piece | hull | 2 | 0% | 14 | 14 | 6 |
| pred:ptv3_short | procedural_piece | voxel | 2 | 0% | 22 | 22 | -18 |
| pred:ptv3_short | procedural_topping | foodscan | 1 | 100% | - | - | - |
| pred:ptv3_short | procedural_topping | hull | 1 | 100% | - | - | - |
| pred:ptv3_short | procedural_topping | voxel | 1 | 100% | - | - | - |

**by batch**

| source | batch | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | hard | foodscan | 17 | 0% | 32 | 85 | -12 |
| gt | hard | hull | 17 | 0% | 46 | 122 | 26 |
| gt | hard | voxel | 17 | 0% | 39 | 90 | 4 |
| gt | hard_v10 | foodscan | 51 | 61% | 86 | 126 | -78 |
| gt | hard_v10 | hull | 51 | 61% | 74 | 355 | -4 |
| gt | hard_v10 | voxel | 51 | 61% | 89 | 148 | -69 |
| gt | hard_v11 | foodscan | 132 | 73% | 86 | 82 | -86 |
| gt | hard_v11 | hull | 132 | 73% | 75 | 85 | -43 |
| gt | hard_v11 | voxel | 132 | 73% | 83 | 79 | -83 |
| gt | hard_v3 | foodscan | 17 | 12% | 31 | 92 | -4 |
| gt | hard_v3 | hull | 17 | 12% | 34 | 136 | 32 |
| gt | hard_v3 | voxel | 17 | 12% | 34 | 92 | 5 |
| gt | hard_v9 | foodscan | 68 | 68% | 91 | 82 | -80 |
| gt | hard_v9 | hull | 68 | 68% | 80 | 234 | -19 |
| gt | hard_v9 | voxel | 68 | 68% | 88 | 83 | -79 |
| gt_top | hard | foodscan | 17 | 0% | 80 | 197 | 80 |
| gt_top | hard | hull | 17 | 0% | 104 | 296 | 104 |
| gt_top | hard | voxel | 17 | 0% | 82 | 209 | 82 |
| gt_top | hard_v10 | foodscan | 51 | 0% | 20 | 225 | 20 |
| gt_top | hard_v10 | hull | 51 | 0% | 82 | 371 | 82 |
| gt_top | hard_v10 | voxel | 51 | 0% | 21 | 235 | 20 |
| gt_top | hard_v11 | foodscan | 143 | 8% | 55 | 374 | 54 |
| gt_top | hard_v11 | hull | 143 | 9% | 116 | 501 | 116 |
| gt_top | hard_v11 | voxel | 143 | 8% | 56 | 380 | 55 |
| gt_top | hard_v3 | foodscan | 17 | 0% | 86 | 143 | 86 |
| gt_top | hard_v3 | hull | 17 | 0% | 74 | 230 | 74 |
| gt_top | hard_v3 | voxel | 17 | 0% | 90 | 146 | 90 |
| gt_top | hard_v9 | foodscan | 68 | 0% | 34 | 336 | 34 |
| gt_top | hard_v9 | hull | 68 | 4% | 67 | 5866 | 67 |
| gt_top | hard_v9 | voxel | 68 | 0% | 34 | 347 | 34 |
| lift | hard_v10 | foodscan | 51 | 61% | 86 | 125 | -78 |
| lift | hard_v10 | hull | 51 | 61% | 74 | 352 | -6 |
| lift | hard_v10 | voxel | 51 | 61% | 90 | 146 | -69 |
| lift | hard_v11 | foodscan | 132 | 74% | 86 | 82 | -86 |
| lift | hard_v11 | hull | 132 | 74% | 74 | 82 | -40 |
| lift | hard_v11 | voxel | 132 | 74% | 83 | 79 | -83 |
| lift | hard_v9 | foodscan | 68 | 68% | 91 | 82 | -80 |
| lift | hard_v9 | hull | 68 | 68% | 80 | 230 | -19 |
| lift | hard_v9 | voxel | 68 | 68% | 89 | 82 | -79 |
| pred:ptv3_short | hard | foodscan | 2 | 0% | 38 | 38 | -13 |
| pred:ptv3_short | hard | hull | 2 | 0% | 40 | 40 | 32 |
| pred:ptv3_short | hard | voxel | 2 | 0% | 49 | 49 | 8 |
| pred:ptv3_short | hard_v3 | foodscan | 3 | 33% | 30 | 30 | -30 |
| pred:ptv3_short | hard_v3 | hull | 3 | 33% | 76 | 76 | 76 |
| pred:ptv3_short | hard_v3 | voxel | 3 | 33% | 18 | 18 | -14 |

**by occluded_fraction**

| source | occluded_fraction | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | 0.3-0.7 | foodscan | 14 | 7% | 77 | 159 | -47 |
| gt | 0.3-0.7 | hull | 14 | 7% | 70 | 697 | 70 |
| gt | 0.3-0.7 | voxel | 14 | 7% | 58 | 197 | -37 |
| gt | >=0.7 | foodscan | 237 | 73% | 90 | 80 | -90 |
| gt | >=0.7 | hull | 237 | 73% | 78 | 96 | -61 |
| gt | >=0.7 | voxel | 237 | 73% | 88 | 78 | -85 |
| gt | n/a | foodscan | 34 | 6% | 32 | 89 | -7 |
| gt | n/a | hull | 34 | 6% | 41 | 129 | 30 |
| gt | n/a | voxel | 34 | 6% | 36 | 91 | 4 |
| gt_top | 0.3-0.7 | foodscan | 14 | 0% | 306 | 593 | 306 |
| gt_top | 0.3-0.7 | hull | 14 | 0% | 453 | 1170 | 453 |
| gt_top | 0.3-0.7 | voxel | 14 | 0% | 309 | 609 | 309 |
| gt_top | >=0.7 | foodscan | 248 | 4% | 50 | 318 | 46 |
| gt_top | >=0.7 | hull | 248 | 6% | 90 | 1935 | 90 |
| gt_top | >=0.7 | voxel | 248 | 4% | 50 | 326 | 49 |
| gt_top | n/a | foodscan | 34 | 0% | 83 | 170 | 83 |
| gt_top | n/a | hull | 34 | 0% | 89 | 263 | 89 |
| gt_top | n/a | voxel | 34 | 0% | 86 | 177 | 86 |
| lift | 0.3-0.7 | foodscan | 14 | 14% | 79 | 165 | -52 |
| lift | 0.3-0.7 | hull | 14 | 14% | 63 | 743 | 57 |
| lift | 0.3-0.7 | voxel | 14 | 14% | 68 | 205 | -37 |
| lift | >=0.7 | foodscan | 237 | 73% | 89 | 80 | -89 |
| lift | >=0.7 | hull | 237 | 73% | 77 | 93 | -47 |
| lift | >=0.7 | voxel | 237 | 73% | 87 | 78 | -85 |
| lift | n/a | foodscan | 0 | - | - | - | - |
| lift | n/a | hull | 0 | - | - | - | - |
| lift | n/a | voxel | 0 | - | - | - | - |
| pred:ptv3_short | n/a | foodscan | 5 | 20% | 33 | 34 | -30 |
| pred:ptv3_short | n/a | hull | 5 | 20% | 46 | 58 | 46 |
| pred:ptv3_short | n/a | voxel | 5 | 20% | 36 | 33 | -14 |

**by stacked_scan**

| source | stacked_scan | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | False | foodscan | 77 | 49% | 73 | 86 | -41 |
| gt | False | hull | 77 | 49% | 58 | 309 | 23 |
| gt | False | voxel | 77 | 49% | 64 | 98 | -34 |
| gt | True | foodscan | 208 | 66% | 80 | 95 | -75 |
| gt | True | hull | 208 | 66% | 74 | 104 | -19 |
| gt | True | voxel | 208 | 66% | 80 | 95 | -67 |
| gt_top | False | foodscan | 88 | 12% | 51 | 210 | 33 |
| gt_top | False | hull | 88 | 12% | 88 | 5066 | 88 |
| gt_top | False | voxel | 88 | 12% | 50 | 222 | 35 |
| gt_top | True | foodscan | 208 | 0% | 52 | 352 | 52 |
| gt_top | True | hull | 208 | 2% | 101 | 415 | 101 |
| gt_top | True | voxel | 208 | 0% | 54 | 359 | 54 |
| lift | False | foodscan | 63 | 60% | 86 | 120 | -80 |
| lift | False | hull | 63 | 60% | 58 | 432 | -2 |
| lift | False | voxel | 63 | 60% | 91 | 140 | -77 |
| lift | True | foodscan | 188 | 73% | 86 | 80 | -86 |
| lift | True | hull | 188 | 73% | 76 | 81 | -61 |
| lift | True | voxel | 188 | 73% | 84 | 77 | -84 |
| pred:ptv3_short | False | foodscan | 1 | 0% | 42 | 42 | -42 |
| pred:ptv3_short | False | hull | 1 | 0% | 132 | 132 | 132 |
| pred:ptv3_short | False | voxel | 1 | 0% | 32 | 32 | -32 |
| pred:ptv3_short | True | foodscan | 4 | 25% | 25 | 31 | -17 |
| pred:ptv3_short | True | hull | 4 | 25% | 20 | 34 | 20 |
| pred:ptv3_short | True | voxel | 4 | 25% | 41 | 34 | 4 |

**by icp_pose_error**

| source | icp_pose_error | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | <10 mm | foodscan | 191 | 61% | 74 | 99 | -59 |
| gt | <10 mm | hull | 191 | 61% | 66 | 212 | -2 |
| gt | <10 mm | voxel | 191 | 61% | 72 | 106 | -50 |
| gt | <10 mm | foodscan_gtreg | 191 | 61% | 75 | 101 | -56 |
| gt | <10 mm | foodscan_cinit | 191 | 61% | 74 | 100 | -58 |
| gt | >=10 mm (failed) | foodscan | 94 | 63% | 85 | 76 | -80 |
| gt | >=10 mm (failed) | hull | 94 | 63% | 88 | 102 | -9 |
| gt | >=10 mm (failed) | voxel | 94 | 63% | 84 | 74 | -77 |
| gt | >=10 mm (failed) | foodscan_gtreg | 94 | 63% | 81 | 75 | -73 |
| gt | >=10 mm (failed) | foodscan_cinit | 94 | 88% | 77 | 74 | -77 |
| gt_top | <10 mm | foodscan | 191 | 0% | 46 | 332 | 37 |
| gt_top | <10 mm | hull | 191 | 1% | 89 | 487 | 89 |
| gt_top | <10 mm | voxel | 191 | 0% | 48 | 340 | 40 |
| gt_top | <10 mm | foodscan_gtreg | 191 | 0% | 46 | 332 | 37 |
| gt_top | <10 mm | foodscan_cinit | 191 | 0% | 46 | 332 | 37 |
| gt_top | >=10 mm (failed) | foodscan | 94 | 0% | 70 | 277 | 70 |
| gt_top | >=10 mm (failed) | hull | 94 | 4% | 132 | 4242 | 132 |
| gt_top | >=10 mm (failed) | voxel | 94 | 0% | 70 | 286 | 70 |
| gt_top | >=10 mm (failed) | foodscan_gtreg | 94 | 0% | 70 | 277 | 70 |
| gt_top | >=10 mm (failed) | foodscan_cinit | 94 | 0% | 70 | 277 | 70 |
| gt_top | n/a | foodscan | 11 | 100% | - | - | - |
| gt_top | n/a | hull | 11 | 100% | - | - | - |
| gt_top | n/a | voxel | 11 | 100% | - | - | - |
| gt_top | n/a | foodscan_gtreg | 11 | 100% | - | - | - |
| gt_top | n/a | foodscan_cinit | 11 | 100% | - | - | - |
| lift | <10 mm | foodscan | 157 | 73% | 87 | 107 | -86 |
| lift | <10 mm | hull | 157 | 73% | 72 | 275 | -47 |
| lift | <10 mm | voxel | 157 | 73% | 85 | 117 | -82 |
| lift | <10 mm | foodscan_gtreg | 157 | 73% | 90 | 109 | -86 |
| lift | <10 mm | foodscan_cinit | 157 | 73% | 88 | 108 | -86 |
| lift | >=10 mm (failed) | foodscan | 94 | 64% | 83 | 75 | -79 |
| lift | >=10 mm (failed) | hull | 94 | 64% | 84 | 98 | 6 |
| lift | >=10 mm (failed) | voxel | 94 | 64% | 83 | 74 | -77 |
| lift | >=10 mm (failed) | foodscan_gtreg | 94 | 64% | 81 | 72 | -72 |
| lift | >=10 mm (failed) | foodscan_cinit | 94 | 88% | 77 | 74 | -77 |
| pred:ptv3_short | <10 mm | foodscan | 5 | 20% | 33 | 34 | -30 |
| pred:ptv3_short | <10 mm | hull | 5 | 20% | 46 | 58 | 46 |
| pred:ptv3_short | <10 mm | voxel | 5 | 20% | 36 | 33 | -14 |
| pred:ptv3_short | <10 mm | foodscan_gtreg | 5 | 20% | 34 | 35 | -31 |
| pred:ptv3_short | <10 mm | foodscan_cinit | 5 | 20% | 33 | 34 | -31 |

## 3. Against `volume_above_container_ml` (rim-plane definition; items with >= 1 mL above the rim)
| source | method | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|
| gt | foodscan | 142 | 74% | 83 | 170 | -56 |
| gt | table | 142 | 74% | 88 | 362 | -15 |
| gt | hull_table | 142 | 74% | 86 | 1196 | 85 |
| gt | (foodscan vs total volume_ml, same items) | 142 | 74% | 90 | 111 | -87 |
| gt_top | foodscan | 142 | 0% | 301 | 717 | 301 |
| gt_top | table | 142 | 0% | 443 | 1202 | 443 |
| gt_top | hull_table | 142 | 2% | 480 | 2560 | 480 |
| gt_top | (foodscan vs total volume_ml, same items) | 142 | 0% | 56 | 248 | 56 |

## 4. Error decomposition (paired: same items valid in every listed source)
Ladder gt_top -> gt (+ sensor sparsity / noise / occlusion) -> lift or pred (+ segmentation). `gtsub` = GT labels on the prediction subset.

| ladder | source | method | items | scans | item med APE % | item mean APE % | item med bias % | dish med APE % | dish mean APE % | dish med bias % |
|---|---|---|---|---|---|---|---|---|---|---|
| lift | gt_top | foodscan | 76 | 59 | 15 | 298 | 10 | 33 | 109 | 33 |
| lift | gt | foodscan | 76 | 59 | 86 | 93 | -84 | 95 | 86 | -95 |
| lift | lift | foodscan | 76 | 59 | 86 | 93 | -85 | 95 | 89 | -94 |
| lift | gt_top | hull | 76 | 59 | 77 | 715 | 77 | 89 | 6234 | 89 |
| lift | gt | hull | 76 | 59 | 75 | 199 | -24 | 87 | 103 | -80 |
| lift | lift | hull | 76 | 59 | 75 | 196 | -20 | 87 | 253 | -76 |
| lift | gt_top | voxel | 76 | 59 | 14 | 307 | 13 | 36 | 116 | 36 |
| lift | gt | voxel | 76 | 59 | 84 | 98 | -81 | 93 | 85 | -92 |
| lift | lift | voxel | 76 | 59 | 84 | 98 | -81 | 92 | 90 | -92 |
| pred:ptv3_short | gt_top | foodscan | 4 | 3 | 42 | 43 | 42 | 48 | 56 | 48 |
| pred:ptv3_short | gt | foodscan | 4 | 3 | 33 | 31 | -16 | 7 | 9 | 6 |
| pred:ptv3_short | gtsub:ptv3_short | foodscan | 4 | 3 | 34 | 36 | -34 | 19 | 17 | -19 |
| pred:ptv3_short | pred:ptv3_short | foodscan | 4 | 3 | 33 | 34 | -30 | 20 | 23 | -20 |
| pred:ptv3_short | gt_top | hull | 4 | 3 | 68 | 99 | 68 | 74 | 122 | 74 |
| pred:ptv3_short | gt | hull | 4 | 3 | 44 | 68 | 44 | 44 | 88 | 44 |
| pred:ptv3_short | gtsub:ptv3_short | hull | 4 | 3 | 43 | 66 | 43 | 41 | 84 | 41 |
| pred:ptv3_short | pred:ptv3_short | hull | 4 | 3 | 46 | 58 | 46 | 39 | 62 | 39 |
| pred:ptv3_short | gt_top | voxel | 4 | 3 | 44 | 45 | 44 | 50 | 59 | 50 |
| pred:ptv3_short | gt | voxel | 4 | 3 | 23 | 26 | 4 | 18 | 19 | 18 |
| pred:ptv3_short | gtsub:ptv3_short | voxel | 4 | 3 | 23 | 26 | -2 | 17 | 17 | 17 |
| pred:ptv3_short | pred:ptv3_short | voxel | 4 | 3 | 36 | 33 | -14 | 17 | 16 | 1 |

## 5. Dish totals (sum over all instances; items without an estimate contribute 0)
Mass columns use only dishes whose every item has a positive `mass_g` (the first presets left it null on ~40% of items).

| source | method | dishes | vol med APE % | vol mean APE % | vol med bias % | mass dishes | vol med APE % (mass dishes) | mass APE med % (cat density) | mass APE mean % | mass bias med % | mass APE med % (item density) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gt | foodscan | 71 | 93 | 74 | -93 | 63 | 94 | 95 | 88 | -94 | 94 |
| gt | voxel | 71 | 91 | 73 | -90 | 63 | 92 | 93 | 87 | -93 | 93 |
| gt | hull | 71 | 84 | 98 | -68 | 63 | 89 | 89 | 177 | -80 | 91 |
| gt | table | 71 | 88 | 77 | -86 | 63 | 89 | 92 | 89 | -90 | 91 |
| gt_top | foodscan | 72 | 35 | 99 | 35 | 64 | 40 | 33 | 210 | 33 | 29 |
| gt_top | voxel | 72 | 37 | 106 | 36 | 64 | 43 | 36 | 225 | 35 | 31 |
| gt_top | hull | 72 | 84 | 5127 | 82 | 64 | 95 | 84 | 6062 | 82 | 86 |
| gt_top | table | 72 | 126 | 274 | 126 | 64 | 131 | 114 | 464 | 114 | 116 |
| lift | foodscan | 59 | 95 | 89 | -94 | 59 | 95 | 95 | 94 | -94 | 95 |
| lift | voxel | 59 | 92 | 90 | -92 | 59 | 92 | 94 | 96 | -93 | 93 |
| lift | hull | 59 | 87 | 253 | -76 | 59 | 87 | 85 | 321 | -76 | 87 |
| lift | table | 59 | 89 | 150 | -88 | 59 | 89 | 90 | 160 | -89 | 89 |
| gtsub:ptv3_short | foodscan | 3 | 19 | 17 | -19 | 1 | 24 | 9 | 9 | 9 | 24 |
| gtsub:ptv3_short | voxel | 3 | 17 | 17 | 17 | 1 | 5 | 37 | 37 | 37 | 5 |
| gtsub:ptv3_short | hull | 3 | 41 | 84 | 41 | 1 | 173 | 293 | 293 | 293 | 173 |
| gtsub:ptv3_short | table | 3 | 36 | 75 | 36 | 1 | 36 | 96 | 96 | 96 | 36 |
| pred:ptv3_short | foodscan | 3 | 20 | 23 | -20 | 1 | 42 | 17 | 17 | -17 | 42 |
| pred:ptv3_short | voxel | 3 | 17 | 16 | 1 | 1 | 32 | 1 | 1 | -1 | 32 |
| pred:ptv3_short | hull | 3 | 39 | 62 | 39 | 1 | 132 | 234 | 234 | 234 | 132 |
| pred:ptv3_short | table | 3 | 32 | 62 | 32 | 1 | 6 | 52 | 52 | 52 | 6 |

## 6. Per-item mass with oracle density
`oracle per-item` density = gt mass / gt volume of the same item, so its mass APE equals the volume APE (shown as a check); `oracle category median` is the non-tautological one. Neither is a deployable density.

| source | method | density | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | foodscan | oracle per-item | 271 | 65% | 81 | 83 | -76 |
| gt | foodscan | oracle category median | 271 | 65% | 82 | 89 | -75 |
| gt | hull | oracle per-item | 271 | 65% | 75 | 180 | -18 |
| gt | hull | oracle category median | 271 | 65% | 75 | 232 | -14 |
| gt | voxel | oracle per-item | 271 | 65% | 79 | 88 | -71 |
| gt | voxel | oracle category median | 271 | 65% | 82 | 96 | -69 |
| gt_top | foodscan | oracle per-item | 282 | 4% | 54 | 323 | 54 |
| gt_top | foodscan | oracle category median | 282 | 4% | 58 | 357 | 57 |
| gt_top | hull | oracle per-item | 282 | 6% | 102 | 1775 | 102 |
| gt_top | hull | oracle category median | 282 | 6% | 113 | 1864 | 113 |
| gt_top | voxel | oracle per-item | 282 | 4% | 57 | 331 | 55 |
| gt_top | voxel | oracle category median | 282 | 4% | 62 | 367 | 61 |
| lift | foodscan | oracle per-item | 251 | 70% | 86 | 93 | -85 |
| lift | foodscan | oracle category median | 251 | 70% | 87 | 96 | -85 |
| lift | hull | oracle per-item | 251 | 70% | 75 | 196 | -20 |
| lift | hull | oracle category median | 251 | 70% | 75 | 250 | -22 |
| lift | voxel | oracle per-item | 251 | 70% | 84 | 98 | -81 |
| lift | voxel | oracle category median | 251 | 70% | 85 | 102 | -81 |
| gtsub:ptv3_short | foodscan | oracle per-item | 2 | 0% | 25 | 25 | 0 |
| gtsub:ptv3_short | foodscan | oracle category median | 2 | 0% | 75 | 75 | 75 |
| gtsub:ptv3_short | hull | oracle per-item | 2 | 0% | 123 | 123 | 123 |
| gtsub:ptv3_short | hull | oracle category median | 2 | 0% | 264 | 264 | 264 |
| gtsub:ptv3_short | voxel | oracle per-item | 2 | 0% | 31 | 31 | 27 |
| gtsub:ptv3_short | voxel | oracle category median | 2 | 0% | 122 | 122 | 122 |
| pred:ptv3_short | foodscan | oracle per-item | 2 | 0% | 33 | 33 | -9 |
| pred:ptv3_short | foodscan | oracle category median | 2 | 0% | 79 | 79 | 62 |
| pred:ptv3_short | hull | oracle per-item | 2 | 0% | 102 | 102 | 102 |
| pred:ptv3_short | hull | oracle category median | 2 | 0% | 234 | 234 | 234 |
| pred:ptv3_short | voxel | oracle per-item | 2 | 0% | 44 | 44 | 13 |
| pred:ptv3_short | voxel | oracle category median | 2 | 0% | 103 | 103 | 101 |

## 7. ok-flag vs all scans (item APE)
| subset | source | method | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| ok | gt_top | foodscan | 57 | 0% | 33 | 252 | 33 |
| ok | gt_top | hull | 57 | 0% | 74 | 499 | 74 |
| ok | gt_top | voxel | 57 | 0% | 35 | 263 | 35 |
| ok | gt | foodscan | 57 | 4% | 49 | 104 | -27 |
| ok | gt | hull | 57 | 4% | 62 | 271 | 26 |
| ok | gt | voxel | 57 | 4% | 52 | 115 | -2 |
| ok | lift | foodscan | 23 | 0% | 93 | 124 | -80 |
| ok | lift | hull | 23 | 0% | 87 | 463 | 15 |
| ok | lift | voxel | 23 | 0% | 92 | 144 | -78 |
| all | gt_top | foodscan | 296 | 4% | 51 | 314 | 47 |
| all | gt_top | hull | 296 | 5% | 97 | 1694 | 97 |
| all | gt_top | voxel | 296 | 4% | 51 | 322 | 50 |
| all | gt | foodscan | 285 | 61% | 76 | 92 | -71 |
| all | gt | hull | 285 | 61% | 71 | 177 | -4 |
| all | gt | voxel | 285 | 61% | 76 | 96 | -58 |
| all | lift | foodscan | 251 | 70% | 86 | 93 | -85 |
| all | lift | hull | 251 | 70% | 75 | 196 | -20 |
| all | lift | voxel | 251 | 70% | 84 | 98 | -81 |
