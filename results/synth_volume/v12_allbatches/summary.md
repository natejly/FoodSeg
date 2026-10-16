# Synthetic per-item volume evaluation
Tables use **ok-flagged** scans; table plane = **gt**; items with fewer than 20 points are 'missing'. Reproduce: `conda run -n foodseg python -m pipeline.eval.synth_volume` from the repo root.
## Reading
1. Segmentation is not the bottleneck; scan visibility and the volume step are. On the 49 items / 19 scans that have lift labels, foodscan's median item APE is 69% with GT labels and 68% with lift labels (segmentation adds almost nothing); per-dish totals 75% vs 75%. The noise-free GT surface (gt_top, oracle container pose) still gives 12% median / +11% bias on the same items.
2. Refined predictions (ptv3_short, only 3 scans / 4 paired items; no results/seg3d/v2_*/preds exist, only this run dumped predictions): foodscan median APE 26% with GT labels on the same voxel subset vs 36% with the predictions (dish 15% vs 15%); far too few scans to rank segmentation sources.
3. Note: after the foodscan fix (Support.height_map_volume hole-fills and rasterises the footprint at 0.5 mm) the per-item median APE with GT labels moved from 49% to ~54%: the old estimator over-counted sparse items by accident, which partly cancelled under-seeing; occlusion dominates either way.
4. With GT labels (ok scans, n=83): median APE / median bias foodscan 53% / -40%, voxel 44% / -9%, hull 52% / +29% (mean APE 214%: the hull blows up on non-convex items), flat-table 68% / +20%. 2% of items are missing (<20 points) and 0% non-positive.
5. Where it fails 1, occlusion: items hidden >= 70% in the scan (n=30) come out -84% (median bias) with GT labels, yet the same items from the full GT surface give 10% APE: the integral is fine, the visible-only cloud has no height to integrate (column fill sums observed cells only, no completion).
6. Where it fails 2, stacking and voids: even from noise-free surfaces the container-relative column fill over-counts, +62% median bias in stacked scans (n=35) vs +14% in unstacked ones (n=48); an item sitting on another is charged for the item below, and loose piles / leaves / toppings are charged for their air gaps (gt_top median bias by kind: topping +782%, leaves +1034%, pile +132%, piece +6%).
7. By size (GT labels, foodscan): <10 mL median APE 53% (bias -15%), 10-50 mL 52%, >50 mL 59% (bias -42%): large items are the ones under-seen, small ones carry the 2 mm cell footprint bias (tests: +13% on a 1 cm cylinder, +4% on 3 cm) and the hull's +75% bias.
8. Registration: template ICP lands 2.3 mm median from the true pose but fails (>= 10 mm) on the sparse scans (16 of 83 ok items; run_meal's init is the centroid of all off-plane points, no multi-start, no fitness gate). Replacing ICP by the true pose changes the median APE only 53% -> 53%, so for these scans occlusion, not registration, is what hurts.
9. run_meal's item_volume crops the whole mesh by the item's bounding box, not by label; emulated on points (foodscan_bbox) the median APE goes 53% -> 70% (bias -40% -> -6%) because neighbours, cutlery and other items inside the box are integrated too.
10. Dish totals (what Nutrition5k-style scoring sees), foodscan: volume median APE 34% (bias -31%, n=31 dishes) with GT labels, 75% vs 75% GT vs lift labels on the 19 lift scans; under-seeing and over-counting partly cancel in the sum, so dish error is lower than item error. Mass with oracle density, 23 dishes with complete mass labels (volume APE 56% on them): 79% with the oracle category-median density, 56% with the oracle per-item density. Densities here are oracle, never predicted; mass_g is integer-rounded for small items.
11. Fix first: (1) get the hidden volume (shape completion / footprint-and-height prior per item, or more views) because visible-only height maps lose 90%+ on occluded items; (2) make stacked and loose items volume-consistent (subtract the supporting item's top, solid-fill factor for piles; compute dish totals as one union column map to avoid double counting); (3) crop by label instead of bbox in run_meal.item_volume; (4) give ContainerTemplate.register a container-label init and a fitness gate so failed registrations are flagged instead of silently used.

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
| hard_v12 | 20 | 5 | 177 | 26 | 0 | 20 |
| **total ok** | 31 |  |  | 83 | 0 | 19 |
| **total all** | 92 |  | 473 |  | 1 | 80 |

Container registration (91 scans): ICP pose error vs truth median 2.29 mm (max 412.80); support height-field RMS vs oracle-pose support median 3.58 mm (max 1197.40). Scans with no support: 0.

## 1. Per-item volume error vs `volume_ml`
| source | method | n items | missing | <=0 | med APE % | mean APE % | med bias % | mean bias % |
|---|---|---|---|---|---|---|---|---|
| gt | foodscan | 83 | 2% | 0% | 53 | 98 | -40 | 18 |
| gt | voxel | 83 | 2% | 0% | 44 | 107 | -9 | 47 |
| gt | hull | 83 | 2% | 0% | 52 | 214 | 29 | 183 |
| gt | foodscan_cinit | 83 | 11% | 0% | 48 | 99 | -30 | 28 |
| gt | foodscan_gtreg | 83 | 2% | 0% | 53 | 99 | -42 | 19 |
| gt | foodscan_bbox | 83 | 2% | 0% | 70 | 176 | -6 | 117 |
| gt | table | 83 | 2% | 0% | 68 | 176 | 20 | 131 |
| gt | hull_table | 83 | 2% | 0% | 81 | 327 | 62 | 309 |
| gt_top | foodscan | 83 | 0% | 0% | 21 | 187 | 19 | 186 |
| gt_top | voxel | 83 | 0% | 0% | 28 | 206 | 28 | 205 |
| gt_top | hull | 83 | 1% | 0% | 61 | 386 | 61 | 386 |
| gt_top | foodscan_cinit | 83 | 0% | 0% | 21 | 187 | 19 | 186 |
| gt_top | foodscan_gtreg | 83 | 0% | 0% | 21 | 187 | 19 | 186 |
| gt_top | foodscan_bbox | 83 | 0% | 0% | 82 | 379 | 82 | 379 |
| gt_top | table | 83 | 0% | 0% | 127 | 421 | 127 | 421 |
| gt_top | hull_table | 83 | 1% | 0% | 141 | 562 | 141 | 562 |
| lift | foodscan | 49 | 0% | 0% | 68 | 97 | -58 | -6 |
| lift | voxel | 49 | 0% | 0% | 62 | 106 | -42 | 22 |
| lift | hull | 49 | 0% | 0% | 62 | 261 | 21 | 220 |
| lift | foodscan_cinit | 49 | 14% | 0% | 63 | 98 | -54 | 6 |
| lift | foodscan_gtreg | 49 | 0% | 0% | 71 | 97 | -57 | -4 |
| lift | foodscan_bbox | 49 | 0% | 0% | 79 | 145 | -33 | 62 |
| lift | table | 49 | 0% | 0% | 75 | 145 | -27 | 73 |
| lift | hull_table | 49 | 2% | 0% | 81 | 375 | 46 | 345 |
| gtsub:ptv3_short | foodscan | 5 | 0% | 0% | 40 | 204 | 2 | 183 |
| gtsub:ptv3_short | voxel | 5 | 0% | 0% | 41 | 209 | 1 | 191 |
| gtsub:ptv3_short | hull | 5 | 0% | 0% | 73 | 237 | 73 | 235 |
| gtsub:ptv3_short | foodscan_cinit | 5 | 0% | 0% | 40 | 206 | 2 | 185 |
| gtsub:ptv3_short | foodscan_gtreg | 5 | 0% | 0% | 41 | 206 | 2 | 184 |
| gtsub:ptv3_short | foodscan_bbox | 5 | 0% | 0% | 39 | 345 | 10 | 327 |
| gtsub:ptv3_short | table | 5 | 0% | 0% | 70 | 351 | 70 | 343 |
| gtsub:ptv3_short | hull_table | 5 | 0% | 0% | 153 | 395 | 153 | 395 |
| pred:ptv3_short | foodscan | 5 | 20% | 0% | 36 | 34 | -12 | -3 |
| pred:ptv3_short | voxel | 5 | 20% | 0% | 36 | 33 | -14 | -3 |
| pred:ptv3_short | hull | 5 | 20% | 0% | 46 | 58 | 46 | 54 |
| pred:ptv3_short | foodscan_cinit | 5 | 20% | 0% | 36 | 34 | -12 | -3 |
| pred:ptv3_short | foodscan_gtreg | 5 | 20% | 0% | 37 | 34 | -13 | -4 |
| pred:ptv3_short | foodscan_bbox | 5 | 20% | 0% | 25 | 28 | 1 | 4 |
| pred:ptv3_short | table | 5 | 20% | 0% | 43 | 62 | 37 | 53 |
| pred:ptv3_short | hull_table | 5 | 20% | 0% | 119 | 119 | 119 | 119 |

## 2. By item size, kind, container (primary methods)

**by size**

| source | size | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | <10 mL | foodscan | 21 | 10% | 53 | 196 | -15 |
| gt | <10 mL | hull | 21 | 10% | 75 | 437 | 75 |
| gt | <10 mL | voxel | 21 | 10% | 57 | 224 | 42 |
| gt | <10 mL | table | 21 | 10% | 163 | 421 | 163 |
| gt | 10-50 mL | foodscan | 28 | 0% | 52 | 80 | -36 |
| gt | 10-50 mL | hull | 28 | 0% | 51 | 265 | 49 |
| gt | 10-50 mL | voxel | 28 | 0% | 24 | 89 | 7 |
| gt | 10-50 mL | table | 28 | 0% | 47 | 144 | 18 |
| gt | >50 mL | foodscan | 34 | 0% | 59 | 58 | -42 |
| gt | >50 mL | hull | 34 | 0% | 37 | 48 | 2 |
| gt | >50 mL | voxel | 34 | 0% | 56 | 56 | -39 |
| gt | >50 mL | table | 34 | 0% | 74 | 66 | -16 |
| gt_top | <10 mL | foodscan | 21 | 0% | 225 | 370 | 225 |
| gt_top | <10 mL | hull | 21 | 0% | 314 | 768 | 314 |
| gt_top | <10 mL | voxel | 21 | 0% | 265 | 430 | 265 |
| gt_top | <10 mL | table | 21 | 0% | 904 | 891 | 904 |
| gt_top | 10-50 mL | foodscan | 28 | 0% | 33 | 244 | 29 |
| gt_top | 10-50 mL | hull | 28 | 0% | 123 | 476 | 123 |
| gt_top | 10-50 mL | voxel | 28 | 0% | 39 | 256 | 35 |
| gt_top | 10-50 mL | table | 28 | 0% | 134 | 456 | 134 |
| gt_top | >50 mL | foodscan | 34 | 0% | 6 | 27 | 5 |
| gt_top | >50 mL | hull | 34 | 3% | 44 | 66 | 44 |
| gt_top | >50 mL | voxel | 34 | 0% | 5 | 28 | 4 |
| gt_top | >50 mL | table | 34 | 0% | 82 | 101 | 82 |
| lift | <10 mL | foodscan | 4 | 0% | 243 | 231 | 243 |
| lift | <10 mL | hull | 4 | 0% | 478 | 1170 | 478 |
| lift | <10 mL | voxel | 4 | 0% | 298 | 285 | 298 |
| lift | <10 mL | table | 4 | 0% | 529 | 428 | 529 |
| lift | 10-50 mL | foodscan | 19 | 0% | 63 | 109 | -58 |
| lift | 10-50 mL | hull | 19 | 0% | 57 | 339 | 43 |
| lift | 10-50 mL | voxel | 19 | 0% | 51 | 125 | -23 |
| lift | 10-50 mL | table | 19 | 0% | 57 | 192 | -19 |
| lift | >50 mL | foodscan | 26 | 0% | 79 | 67 | -79 |
| lift | >50 mL | hull | 26 | 0% | 51 | 64 | 2 |
| lift | >50 mL | voxel | 26 | 0% | 77 | 65 | -70 |
| lift | >50 mL | table | 26 | 0% | 77 | 68 | -57 |
| pred:ptv3_short | <10 mL | foodscan | 1 | 100% | - | - | - |
| pred:ptv3_short | <10 mL | hull | 1 | 100% | - | - | - |
| pred:ptv3_short | <10 mL | voxel | 1 | 100% | - | - | - |
| pred:ptv3_short | <10 mL | table | 1 | 100% | - | - | - |
| pred:ptv3_short | 10-50 mL | foodscan | 1 | 0% | 33 | 33 | -33 |
| pred:ptv3_short | 10-50 mL | hull | 1 | 0% | 132 | 132 | 132 |
| pred:ptv3_short | 10-50 mL | voxel | 1 | 0% | 32 | 32 | -32 |
| pred:ptv3_short | 10-50 mL | table | 1 | 0% | 6 | 6 | 6 |
| pred:ptv3_short | >50 mL | foodscan | 3 | 0% | 40 | 34 | 8 |
| pred:ptv3_short | >50 mL | hull | 3 | 0% | 20 | 34 | 20 |
| pred:ptv3_short | >50 mL | voxel | 3 | 0% | 41 | 34 | 4 |
| pred:ptv3_short | >50 mL | table | 3 | 0% | 68 | 80 | 68 |

**by kind**

| source | kind | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | leaves | foodscan | 5 | 0% | 93 | 289 | 79 |
| gt | leaves | hull | 5 | 0% | 925 | 1852 | 925 |
| gt | leaves | voxel | 5 | 0% | 132 | 406 | 132 |
| gt | piece | foodscan | 31 | 0% | 41 | 47 | -31 |
| gt | piece | hull | 31 | 0% | 32 | 41 | 8 |
| gt | piece | voxel | 31 | 0% | 32 | 45 | -26 |
| gt | pile | foodscan | 22 | 9% | 52 | 47 | -51 |
| gt | pile | hull | 22 | 9% | 51 | 86 | 42 |
| gt | pile | voxel | 22 | 9% | 46 | 50 | -13 |
| gt | single | foodscan | 21 | 0% | 48 | 52 | -42 |
| gt | single | hull | 21 | 0% | 72 | 78 | 39 |
| gt | single | voxel | 21 | 0% | 39 | 45 | -30 |
| gt | topping | foodscan | 4 | 0% | 864 | 741 | 864 |
| gt | topping | hull | 4 | 0% | 966 | 861 | 966 |
| gt | topping | voxel | 4 | 0% | 960 | 824 | 960 |
| gt_top | leaves | foodscan | 5 | 0% | 1034 | 1436 | 1034 |
| gt_top | leaves | hull | 5 | 0% | 3210 | 3393 | 3210 |
| gt_top | leaves | voxel | 5 | 0% | 1060 | 1487 | 1060 |
| gt_top | piece | foodscan | 31 | 0% | 6 | 25 | 6 |
| gt_top | piece | hull | 31 | 0% | 38 | 61 | 38 |
| gt_top | piece | voxel | 31 | 0% | 6 | 26 | 6 |
| gt_top | pile | foodscan | 22 | 0% | 132 | 142 | 132 |
| gt_top | pile | hull | 22 | 0% | 158 | 300 | 158 |
| gt_top | pile | voxel | 22 | 0% | 147 | 182 | 147 |
| gt_top | single | foodscan | 21 | 0% | 36 | 73 | 36 |
| gt_top | single | hull | 21 | 5% | 91 | 152 | 91 |
| gt_top | single | voxel | 21 | 0% | 37 | 80 | 37 |
| gt_top | topping | foodscan | 4 | 0% | 782 | 727 | 782 |
| gt_top | topping | hull | 4 | 0% | 874 | 792 | 874 |
| gt_top | topping | voxel | 4 | 0% | 880 | 806 | 880 |
| lift | leaves | foodscan | 5 | 0% | 93 | 280 | 79 |
| lift | leaves | hull | 5 | 0% | 925 | 1835 | 925 |
| lift | leaves | voxel | 5 | 0% | 132 | 392 | 132 |
| lift | piece | foodscan | 19 | 0% | 89 | 70 | -89 |
| lift | piece | hull | 19 | 0% | 42 | 49 | -6 |
| lift | piece | voxel | 19 | 0% | 91 | 66 | -84 |
| lift | pile | foodscan | 6 | 0% | 61 | 54 | -58 |
| lift | pile | hull | 6 | 0% | 48 | 55 | 1 |
| lift | pile | voxel | 6 | 0% | 54 | 48 | -25 |
| lift | single | foodscan | 17 | 0% | 59 | 56 | -59 |
| lift | single | hull | 17 | 0% | 75 | 82 | 26 |
| lift | single | voxel | 17 | 0% | 43 | 49 | -42 |
| lift | topping | foodscan | 2 | 0% | 358 | 358 | 358 |
| lift | topping | hull | 2 | 0% | 472 | 472 | 472 |
| lift | topping | voxel | 2 | 0% | 441 | 441 | 441 |
| pred:ptv3_short | piece | foodscan | 2 | 0% | 24 | 24 | -16 |
| pred:ptv3_short | piece | hull | 2 | 0% | 14 | 14 | 6 |
| pred:ptv3_short | piece | voxel | 2 | 0% | 22 | 22 | -18 |
| pred:ptv3_short | single | foodscan | 2 | 0% | 43 | 43 | 10 |
| pred:ptv3_short | single | hull | 2 | 0% | 102 | 102 | 102 |
| pred:ptv3_short | single | voxel | 2 | 0% | 44 | 44 | 13 |
| pred:ptv3_short | topping | foodscan | 1 | 100% | - | - | - |
| pred:ptv3_short | topping | hull | 1 | 100% | - | - | - |
| pred:ptv3_short | topping | voxel | 1 | 100% | - | - | - |

**by container**

| source | container | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | bowl | foodscan | 43 | 5% | 54 | 124 | -46 |
| gt | bowl | hull | 43 | 5% | 53 | 238 | 35 |
| gt | bowl | voxel | 43 | 5% | 52 | 138 | -3 |
| gt | plate | foodscan | 40 | 0% | 46 | 71 | -38 |
| gt | plate | hull | 40 | 0% | 45 | 189 | 24 |
| gt | plate | voxel | 40 | 0% | 43 | 75 | -27 |
| gt_top | bowl | foodscan | 43 | 0% | 72 | 252 | 72 |
| gt_top | bowl | hull | 43 | 0% | 135 | 476 | 135 |
| gt_top | bowl | voxel | 43 | 0% | 90 | 282 | 90 |
| gt_top | plate | foodscan | 40 | 0% | 11 | 117 | 10 |
| gt_top | plate | hull | 40 | 2% | 54 | 286 | 54 |
| gt_top | plate | voxel | 40 | 0% | 15 | 125 | 13 |
| lift | bowl | foodscan | 19 | 0% | 94 | 134 | -75 |
| lift | bowl | hull | 19 | 0% | 43 | 306 | 26 |
| lift | bowl | voxel | 19 | 0% | 91 | 151 | -54 |
| lift | plate | foodscan | 30 | 0% | 61 | 73 | -54 |
| lift | plate | hull | 30 | 0% | 72 | 232 | 13 |
| lift | plate | voxel | 30 | 0% | 55 | 78 | -40 |
| pred:ptv3_short | bowl | foodscan | 3 | 33% | 21 | 21 | -12 |
| pred:ptv3_short | bowl | hull | 3 | 33% | 76 | 76 | 76 |
| pred:ptv3_short | bowl | voxel | 3 | 33% | 18 | 18 | -14 |
| pred:ptv3_short | plate | foodscan | 2 | 0% | 47 | 47 | 7 |
| pred:ptv3_short | plate | hull | 2 | 0% | 40 | 40 | 32 |
| pred:ptv3_short | plate | voxel | 2 | 0% | 49 | 49 | 8 |

**by asset_source**

| source | asset_source | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | metafood3d | foodscan | 37 | 5% | 51 | 49 | -48 |
| gt | metafood3d | hull | 37 | 5% | 53 | 85 | 39 |
| gt | metafood3d | voxel | 37 | 5% | 42 | 46 | -33 |
| gt | objaverse | foodscan | 3 | 0% | 59 | 48 | -14 |
| gt | objaverse | hull | 3 | 0% | 64 | 68 | 64 |
| gt | objaverse | voxel | 3 | 0% | 54 | 55 | 22 |
| gt | primitive | foodscan | 3 | 0% | 37 | 56 | -36 |
| gt | primitive | hull | 3 | 0% | 85 | 65 | -18 |
| gt | primitive | voxel | 3 | 0% | 39 | 54 | -30 |
| gt | procedural_leaves | foodscan | 5 | 0% | 93 | 289 | 79 |
| gt | procedural_leaves | hull | 5 | 0% | 925 | 1852 | 925 |
| gt | procedural_leaves | voxel | 5 | 0% | 132 | 406 | 132 |
| gt | procedural_piece | foodscan | 31 | 0% | 41 | 47 | -31 |
| gt | procedural_piece | hull | 31 | 0% | 32 | 41 | 8 |
| gt | procedural_piece | voxel | 31 | 0% | 32 | 45 | -26 |
| gt | procedural_topping | foodscan | 4 | 0% | 864 | 741 | 864 |
| gt | procedural_topping | hull | 4 | 0% | 966 | 861 | 966 |
| gt | procedural_topping | voxel | 4 | 0% | 960 | 824 | 960 |
| gt_top | metafood3d | foodscan | 37 | 0% | 72 | 116 | 72 |
| gt_top | metafood3d | hull | 37 | 0% | 128 | 246 | 128 |
| gt_top | metafood3d | voxel | 37 | 0% | 90 | 142 | 90 |
| gt_top | objaverse | foodscan | 3 | 0% | 125 | 107 | 125 |
| gt_top | objaverse | hull | 3 | 0% | 166 | 145 | 166 |
| gt_top | objaverse | voxel | 3 | 0% | 138 | 117 | 138 |
| gt_top | primitive | foodscan | 3 | 0% | 27 | 23 | -6 |
| gt_top | primitive | hull | 3 | 33% | 40 | 40 | 30 |
| gt_top | primitive | voxel | 3 | 0% | 26 | 23 | -4 |
| gt_top | procedural_leaves | foodscan | 5 | 0% | 1034 | 1436 | 1034 |
| gt_top | procedural_leaves | hull | 5 | 0% | 3210 | 3393 | 3210 |
| gt_top | procedural_leaves | voxel | 5 | 0% | 1060 | 1487 | 1060 |
| gt_top | procedural_piece | foodscan | 31 | 0% | 6 | 25 | 6 |
| gt_top | procedural_piece | hull | 31 | 0% | 38 | 61 | 38 |
| gt_top | procedural_piece | voxel | 31 | 0% | 6 | 26 | 6 |
| gt_top | procedural_topping | foodscan | 4 | 0% | 782 | 727 | 782 |
| gt_top | procedural_topping | hull | 4 | 0% | 874 | 792 | 874 |
| gt_top | procedural_topping | voxel | 4 | 0% | 880 | 806 | 880 |
| lift | metafood3d | foodscan | 17 | 0% | 59 | 57 | -59 |
| lift | metafood3d | hull | 17 | 0% | 57 | 77 | 25 |
| lift | metafood3d | voxel | 17 | 0% | 51 | 46 | -42 |
| lift | objaverse | foodscan | 3 | 0% | 63 | 48 | -12 |
| lift | objaverse | hull | 3 | 0% | 63 | 76 | 63 |
| lift | objaverse | voxel | 3 | 0% | 57 | 58 | 19 |
| lift | primitive | foodscan | 3 | 0% | 48 | 60 | -48 |
| lift | primitive | hull | 3 | 0% | 79 | 62 | -16 |
| lift | primitive | voxel | 3 | 0% | 43 | 57 | -43 |
| lift | procedural_leaves | foodscan | 5 | 0% | 93 | 280 | 79 |
| lift | procedural_leaves | hull | 5 | 0% | 925 | 1835 | 925 |
| lift | procedural_leaves | voxel | 5 | 0% | 132 | 392 | 132 |
| lift | procedural_piece | foodscan | 19 | 0% | 89 | 70 | -89 |
| lift | procedural_piece | hull | 19 | 0% | 42 | 49 | -6 |
| lift | procedural_piece | voxel | 19 | 0% | 91 | 66 | -84 |
| lift | procedural_topping | foodscan | 2 | 0% | 358 | 358 | 358 |
| lift | procedural_topping | hull | 2 | 0% | 472 | 472 | 472 |
| lift | procedural_topping | voxel | 2 | 0% | 441 | 441 | 441 |
| pred:ptv3_short | metafood3d | foodscan | 2 | 0% | 43 | 43 | 10 |
| pred:ptv3_short | metafood3d | hull | 2 | 0% | 102 | 102 | 102 |
| pred:ptv3_short | metafood3d | voxel | 2 | 0% | 44 | 44 | 13 |
| pred:ptv3_short | procedural_piece | foodscan | 2 | 0% | 24 | 24 | -16 |
| pred:ptv3_short | procedural_piece | hull | 2 | 0% | 14 | 14 | 6 |
| pred:ptv3_short | procedural_piece | voxel | 2 | 0% | 22 | 22 | -18 |
| pred:ptv3_short | procedural_topping | foodscan | 1 | 100% | - | - | - |
| pred:ptv3_short | procedural_topping | hull | 1 | 100% | - | - | - |
| pred:ptv3_short | procedural_topping | voxel | 1 | 100% | - | - | - |

**by batch**

| source | batch | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | hard | foodscan | 17 | 0% | 26 | 80 | -15 |
| gt | hard | hull | 17 | 0% | 44 | 122 | 26 |
| gt | hard | voxel | 17 | 0% | 34 | 90 | 4 |
| gt | hard_v10 | foodscan | 13 | 0% | 93 | 160 | -89 |
| gt | hard_v10 | hull | 13 | 0% | 70 | 466 | 0 |
| gt | hard_v10 | voxel | 13 | 0% | 89 | 195 | -82 |
| gt | hard_v11 | foodscan | 2 | 0% | 85 | 85 | -85 |
| gt | hard_v11 | hull | 2 | 0% | 65 | 65 | -30 |
| gt | hard_v11 | voxel | 2 | 0% | 84 | 84 | -84 |
| gt | hard_v12 | foodscan | 26 | 0% | 43 | 87 | -38 |
| gt | hard_v12 | hull | 26 | 0% | 45 | 105 | 39 |
| gt | hard_v12 | voxel | 26 | 0% | 39 | 90 | -19 |
| gt | hard_v3 | foodscan | 17 | 12% | 39 | 89 | -4 |
| gt | hard_v3 | hull | 17 | 12% | 34 | 136 | 32 |
| gt | hard_v3 | voxel | 17 | 12% | 34 | 92 | 5 |
| gt | hard_v9 | foodscan | 8 | 0% | 96 | 91 | -84 |
| gt | hard_v9 | hull | 8 | 0% | 90 | 539 | -12 |
| gt | hard_v9 | voxel | 8 | 0% | 93 | 88 | -73 |
| gt_top | hard | foodscan | 17 | 0% | 72 | 169 | 72 |
| gt_top | hard | hull | 17 | 0% | 104 | 296 | 104 |
| gt_top | hard | voxel | 17 | 0% | 82 | 209 | 82 |
| gt_top | hard_v10 | foodscan | 13 | 0% | 11 | 409 | 11 |
| gt_top | hard_v10 | hull | 13 | 0% | 58 | 801 | 58 |
| gt_top | hard_v10 | voxel | 13 | 0% | 13 | 421 | 13 |
| gt_top | hard_v11 | foodscan | 2 | 0% | 127 | 127 | 127 |
| gt_top | hard_v11 | hull | 2 | 0% | 218 | 218 | 218 |
| gt_top | hard_v11 | voxel | 2 | 0% | 127 | 127 | 127 |
| gt_top | hard_v12 | foodscan | 26 | 0% | 15 | 73 | 9 |
| gt_top | hard_v12 | hull | 26 | 4% | 56 | 128 | 56 |
| gt_top | hard_v12 | voxel | 26 | 0% | 15 | 82 | 10 |
| gt_top | hard_v3 | foodscan | 17 | 0% | 62 | 124 | 62 |
| gt_top | hard_v3 | hull | 17 | 0% | 74 | 230 | 74 |
| gt_top | hard_v3 | voxel | 17 | 0% | 90 | 146 | 90 |
| gt_top | hard_v9 | foodscan | 8 | 0% | 15 | 385 | 15 |
| gt_top | hard_v9 | hull | 8 | 0% | 116 | 1083 | 116 |
| gt_top | hard_v9 | voxel | 8 | 0% | 15 | 407 | 15 |
| lift | hard_v10 | foodscan | 13 | 0% | 93 | 157 | -89 |
| lift | hard_v10 | hull | 13 | 0% | 70 | 460 | 0 |
| lift | hard_v10 | voxel | 13 | 0% | 91 | 190 | -82 |
| lift | hard_v11 | foodscan | 2 | 0% | 85 | 85 | -85 |
| lift | hard_v11 | hull | 2 | 0% | 46 | 46 | -16 |
| lift | hard_v11 | voxel | 2 | 0% | 84 | 84 | -84 |
| lift | hard_v12 | foodscan | 26 | 0% | 54 | 69 | -34 |
| lift | hard_v12 | hull | 26 | 0% | 55 | 95 | 41 |
| lift | hard_v12 | voxel | 26 | 0% | 42 | 72 | -16 |
| lift | hard_v9 | foodscan | 8 | 0% | 96 | 92 | -84 |
| lift | hard_v9 | hull | 8 | 0% | 89 | 529 | 5 |
| lift | hard_v9 | voxel | 8 | 0% | 92 | 86 | -73 |
| pred:ptv3_short | hard | foodscan | 2 | 0% | 47 | 47 | 7 |
| pred:ptv3_short | hard | hull | 2 | 0% | 40 | 40 | 32 |
| pred:ptv3_short | hard | voxel | 2 | 0% | 49 | 49 | 8 |
| pred:ptv3_short | hard_v3 | foodscan | 3 | 33% | 21 | 21 | -12 |
| pred:ptv3_short | hard_v3 | hull | 3 | 33% | 76 | 76 | 76 |
| pred:ptv3_short | hard_v3 | voxel | 3 | 33% | 18 | 18 | -14 |

**by occluded_fraction**

| source | occluded_fraction | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | 0.3-0.7 | foodscan | 17 | 0% | 40 | 109 | -10 |
| gt | 0.3-0.7 | hull | 17 | 0% | 43 | 538 | 43 |
| gt | 0.3-0.7 | voxel | 17 | 0% | 37 | 143 | 8 |
| gt | <0.3 | foodscan | 2 | 0% | 600 | 600 | 600 |
| gt | <0.3 | hull | 2 | 0% | 757 | 757 | 757 |
| gt | <0.3 | voxel | 2 | 0% | 689 | 689 | 689 |
| gt | >=0.7 | foodscan | 30 | 0% | 84 | 73 | -84 |
| gt | >=0.7 | hull | 30 | 0% | 60 | 85 | -16 |
| gt | >=0.7 | voxel | 30 | 0% | 80 | 64 | -74 |
| gt | n/a | foodscan | 34 | 6% | 33 | 84 | -6 |
| gt | n/a | hull | 34 | 6% | 40 | 128 | 30 |
| gt | n/a | voxel | 34 | 6% | 34 | 91 | 4 |
| gt_top | 0.3-0.7 | foodscan | 17 | 0% | 19 | 376 | 19 |
| gt_top | 0.3-0.7 | hull | 17 | 0% | 71 | 798 | 71 |
| gt_top | 0.3-0.7 | voxel | 17 | 0% | 18 | 392 | 18 |
| gt_top | <0.3 | foodscan | 2 | 0% | 520 | 520 | 520 |
| gt_top | <0.3 | hull | 2 | 0% | 601 | 601 | 601 |
| gt_top | <0.3 | voxel | 2 | 0% | 610 | 610 | 610 |
| gt_top | >=0.7 | foodscan | 30 | 0% | 10 | 104 | 9 |
| gt_top | >=0.7 | hull | 30 | 3% | 52 | 274 | 52 |
| gt_top | >=0.7 | voxel | 30 | 0% | 10 | 108 | 9 |
| gt_top | n/a | foodscan | 34 | 0% | 67 | 147 | 67 |
| gt_top | n/a | hull | 34 | 0% | 89 | 263 | 89 |
| gt_top | n/a | voxel | 34 | 0% | 86 | 177 | 86 |
| lift | 0.3-0.7 | foodscan | 17 | 0% | 36 | 104 | -4 |
| lift | 0.3-0.7 | hull | 17 | 0% | 43 | 552 | 43 |
| lift | 0.3-0.7 | voxel | 17 | 0% | 37 | 139 | 18 |
| lift | <0.3 | foodscan | 2 | 0% | 358 | 358 | 358 |
| lift | <0.3 | hull | 2 | 0% | 472 | 472 | 472 |
| lift | <0.3 | voxel | 2 | 0% | 441 | 441 | 441 |
| lift | >=0.7 | foodscan | 30 | 0% | 85 | 75 | -85 |
| lift | >=0.7 | hull | 30 | 0% | 60 | 82 | -17 |
| lift | >=0.7 | voxel | 30 | 0% | 81 | 65 | -75 |
| lift | n/a | foodscan | 0 | - | - | - | - |
| lift | n/a | hull | 0 | - | - | - | - |
| lift | n/a | voxel | 0 | - | - | - | - |
| pred:ptv3_short | n/a | foodscan | 5 | 20% | 36 | 34 | -12 |
| pred:ptv3_short | n/a | hull | 5 | 20% | 46 | 58 | 46 |
| pred:ptv3_short | n/a | voxel | 5 | 20% | 36 | 33 | -14 |

**by stacked_scan**

| source | stacked_scan | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | False | foodscan | 48 | 2% | 54 | 77 | -44 |
| gt | False | hull | 48 | 2% | 50 | 253 | 23 |
| gt | False | voxel | 48 | 2% | 41 | 83 | -28 |
| gt | True | foodscan | 35 | 3% | 51 | 127 | -12 |
| gt | True | hull | 35 | 3% | 53 | 159 | 37 |
| gt | True | voxel | 35 | 3% | 56 | 140 | -3 |
| gt_top | False | foodscan | 48 | 0% | 16 | 192 | 14 |
| gt_top | False | hull | 48 | 2% | 53 | 449 | 53 |
| gt_top | False | voxel | 48 | 0% | 21 | 201 | 19 |
| gt_top | True | foodscan | 35 | 0% | 62 | 181 | 62 |
| gt_top | True | hull | 35 | 0% | 104 | 301 | 104 |
| gt_top | True | voxel | 35 | 0% | 82 | 214 | 82 |
| lift | False | foodscan | 34 | 0% | 68 | 96 | -59 |
| lift | False | hull | 34 | 0% | 63 | 326 | 4 |
| lift | False | voxel | 34 | 0% | 61 | 108 | -42 |
| lift | True | foodscan | 15 | 0% | 82 | 99 | -54 |
| lift | True | hull | 15 | 0% | 43 | 114 | 26 |
| lift | True | voxel | 15 | 0% | 70 | 102 | -23 |
| pred:ptv3_short | False | foodscan | 1 | 0% | 33 | 33 | -33 |
| pred:ptv3_short | False | hull | 1 | 0% | 132 | 132 | 132 |
| pred:ptv3_short | False | voxel | 1 | 0% | 32 | 32 | -32 |
| pred:ptv3_short | True | foodscan | 4 | 25% | 40 | 34 | 8 |
| pred:ptv3_short | True | hull | 4 | 25% | 20 | 34 | 20 |
| pred:ptv3_short | True | voxel | 4 | 25% | 41 | 34 | 4 |

**by icp_pose_error**

| source | icp_pose_error | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | <10 mm | foodscan | 67 | 3% | 46 | 103 | -19 |
| gt | <10 mm | hull | 67 | 3% | 53 | 249 | 32 |
| gt | <10 mm | voxel | 67 | 3% | 41 | 117 | -2 |
| gt | <10 mm | foodscan_gtreg | 67 | 3% | 49 | 105 | -21 |
| gt | <10 mm | foodscan_cinit | 67 | 4% | 44 | 104 | -18 |
| gt | >=10 mm (failed) | foodscan | 16 | 0% | 88 | 76 | -88 |
| gt | >=10 mm (failed) | hull | 16 | 0% | 43 | 71 | 15 |
| gt | >=10 mm (failed) | voxel | 16 | 0% | 83 | 65 | -73 |
| gt | >=10 mm (failed) | foodscan_gtreg | 16 | 0% | 87 | 76 | -73 |
| gt | >=10 mm (failed) | foodscan_cinit | 16 | 38% | 65 | 66 | -65 |
| gt_top | <10 mm | foodscan | 67 | 0% | 27 | 209 | 21 |
| gt_top | <10 mm | hull | 67 | 1% | 67 | 439 | 67 |
| gt_top | <10 mm | voxel | 67 | 0% | 30 | 231 | 30 |
| gt_top | <10 mm | foodscan_gtreg | 67 | 0% | 27 | 209 | 21 |
| gt_top | <10 mm | foodscan_cinit | 67 | 0% | 27 | 209 | 21 |
| gt_top | >=10 mm (failed) | foodscan | 16 | 0% | 10 | 98 | 10 |
| gt_top | >=10 mm (failed) | hull | 16 | 0% | 55 | 168 | 55 |
| gt_top | >=10 mm (failed) | voxel | 16 | 0% | 12 | 103 | 12 |
| gt_top | >=10 mm (failed) | foodscan_gtreg | 16 | 0% | 10 | 98 | 10 |
| gt_top | >=10 mm (failed) | foodscan_cinit | 16 | 0% | 10 | 98 | 10 |
| lift | <10 mm | foodscan | 33 | 0% | 63 | 106 | -31 |
| lift | <10 mm | hull | 33 | 0% | 70 | 355 | 25 |
| lift | <10 mm | voxel | 33 | 0% | 57 | 126 | -23 |
| lift | <10 mm | foodscan_gtreg | 33 | 0% | 68 | 107 | -39 |
| lift | <10 mm | foodscan_cinit | 33 | 3% | 61 | 107 | -29 |
| lift | >=10 mm (failed) | foodscan | 16 | 0% | 89 | 78 | -89 |
| lift | >=10 mm (failed) | hull | 16 | 0% | 47 | 66 | 15 |
| lift | >=10 mm (failed) | voxel | 16 | 0% | 85 | 66 | -75 |
| lift | >=10 mm (failed) | foodscan_gtreg | 16 | 0% | 86 | 77 | -76 |
| lift | >=10 mm (failed) | foodscan_cinit | 16 | 38% | 70 | 69 | -70 |
| pred:ptv3_short | <10 mm | foodscan | 5 | 20% | 36 | 34 | -12 |
| pred:ptv3_short | <10 mm | hull | 5 | 20% | 46 | 58 | 46 |
| pred:ptv3_short | <10 mm | voxel | 5 | 20% | 36 | 33 | -14 |
| pred:ptv3_short | <10 mm | foodscan_gtreg | 5 | 20% | 37 | 34 | -13 |
| pred:ptv3_short | <10 mm | foodscan_cinit | 5 | 20% | 36 | 34 | -12 |

## 3. Against `volume_above_container_ml` (rim-plane definition; items with >= 1 mL above the rim)
| source | method | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|
| gt | foodscan | 30 | 0% | 61 | 212 | -0 |
| gt | table | 30 | 0% | 73 | 433 | 43 |
| gt | hull_table | 30 | 0% | 107 | 974 | 107 |
| gt | (foodscan vs total volume_ml, same items) | 30 | 0% | 60 | 97 | -51 |
| gt_top | foodscan | 30 | 0% | 185 | 936 | 185 |
| gt_top | table | 30 | 0% | 287 | 1530 | 287 |
| gt_top | hull_table | 30 | 3% | 356 | 1784 | 356 |
| gt_top | (foodscan vs total volume_ml, same items) | 30 | 0% | 20 | 237 | 20 |

## 4. Error decomposition (paired: same items valid in every listed source)
Ladder gt_top -> gt (+ sensor sparsity / noise / occlusion) -> lift or pred (+ segmentation). `gtsub` = GT labels on the prediction subset.

| ladder | source | method | items | scans | item med APE % | item mean APE % | item med bias % | dish med APE % | dish mean APE % | dish med bias % |
|---|---|---|---|---|---|---|---|---|---|---|
| lift | gt_top | foodscan | 49 | 19 | 12 | 215 | 11 | 34 | 158 | 34 |
| lift | gt | foodscan | 49 | 19 | 69 | 107 | -54 | 75 | 63 | -61 |
| lift | lift | foodscan | 49 | 19 | 68 | 97 | -58 | 75 | 84 | -61 |
| lift | gt_top | hull | 48 | 19 | 59 | 473 | 59 | 79 | 428 | 79 |
| lift | gt | hull | 48 | 19 | 60 | 275 | 30 | 55 | 121 | 7 |
| lift | lift | hull | 48 | 19 | 63 | 266 | 23 | 59 | 573 | 18 |
| lift | gt_top | voxel | 49 | 19 | 13 | 227 | 10 | 33 | 160 | 33 |
| lift | gt | voxel | 49 | 19 | 56 | 117 | -37 | 54 | 57 | -45 |
| lift | lift | voxel | 49 | 19 | 62 | 106 | -42 | 54 | 76 | -44 |
| pred:ptv3_short | gt_top | foodscan | 4 | 3 | 39 | 41 | 39 | 49 | 52 | 49 |
| pred:ptv3_short | gt | foodscan | 4 | 3 | 30 | 28 | -9 | 20 | 21 | 13 |
| pred:ptv3_short | gtsub:ptv3_short | foodscan | 4 | 3 | 26 | 27 | -5 | 15 | 20 | 15 |
| pred:ptv3_short | pred:ptv3_short | foodscan | 4 | 3 | 36 | 34 | -12 | 15 | 18 | 5 |
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
| gt | foodscan | 31 | 34 | 46 | -31 | 23 | 56 | 79 | 68 | -52 | 56 |
| gt | voxel | 31 | 28 | 40 | -8 | 23 | 28 | 65 | 68 | -28 | 28 |
| gt | hull | 31 | 45 | 103 | 41 | 23 | 67 | 62 | 214 | 22 | 66 |
| gt | table | 31 | 43 | 60 | 8 | 23 | 49 | 75 | 89 | -25 | 51 |
| gt_top | foodscan | 31 | 36 | 116 | 36 | 23 | 41 | 21 | 322 | 21 | 28 |
| gt_top | voxel | 31 | 36 | 120 | 36 | 23 | 44 | 22 | 331 | 22 | 31 |
| gt_top | hull | 31 | 74 | 303 | 74 | 23 | 97 | 81 | 898 | 81 | 80 |
| gt_top | table | 31 | 121 | 257 | 121 | 23 | 149 | 137 | 559 | 137 | 127 |
| lift | foodscan | 19 | 75 | 84 | -61 | 19 | 75 | 88 | 95 | -81 | 86 |
| lift | voxel | 19 | 54 | 76 | -44 | 19 | 54 | 80 | 93 | -65 | 74 |
| lift | hull | 19 | 59 | 573 | 18 | 19 | 59 | 49 | 656 | 6 | 61 |
| lift | table | 19 | 75 | 315 | -19 | 19 | 75 | 78 | 343 | -29 | 76 |
| gtsub:ptv3_short | foodscan | 3 | 15 | 20 | 15 | 1 | 12 | 29 | 29 | 29 | 12 |
| gtsub:ptv3_short | voxel | 3 | 17 | 17 | 17 | 1 | 5 | 40 | 40 | 40 | 5 |
| gtsub:ptv3_short | hull | 3 | 41 | 84 | 41 | 1 | 173 | 300 | 300 | 300 | 173 |
| gtsub:ptv3_short | table | 3 | 36 | 75 | 36 | 1 | 36 | 100 | 100 | 100 | 36 |
| pred:ptv3_short | foodscan | 3 | 15 | 18 | 5 | 1 | 33 | 1 | 1 | -1 | 33 |
| pred:ptv3_short | voxel | 3 | 17 | 16 | 1 | 1 | 32 | 0 | 0 | 0 | 32 |
| pred:ptv3_short | hull | 3 | 39 | 62 | 39 | 1 | 132 | 240 | 240 | 240 | 132 |
| pred:ptv3_short | table | 3 | 32 | 62 | 32 | 1 | 6 | 55 | 55 | 55 | 6 |

## 6. Per-item mass with oracle density
`oracle per-item` density = gt mass / gt volume of the same item, so its mass APE equals the volume APE (shown as a check); `oracle category median` is the non-tautological one. Neither is a deployable density.

| source | method | density | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | foodscan | oracle per-item | 69 | 3% | 59 | 90 | -48 |
| gt | foodscan | oracle category median | 69 | 3% | 59 | 96 | -41 |
| gt | hull | oracle per-item | 69 | 3% | 62 | 227 | 35 |
| gt | hull | oracle category median | 69 | 3% | 58 | 266 | 38 |
| gt | voxel | oracle per-item | 69 | 3% | 54 | 99 | -30 |
| gt | voxel | oracle category median | 69 | 3% | 59 | 110 | -13 |
| gt_top | foodscan | oracle per-item | 69 | 0% | 36 | 197 | 36 |
| gt_top | foodscan | oracle category median | 69 | 0% | 51 | 275 | 51 |
| gt_top | hull | oracle per-item | 69 | 1% | 91 | 432 | 91 |
| gt_top | hull | oracle category median | 69 | 1% | 122 | 637 | 122 |
| gt_top | voxel | oracle per-item | 69 | 0% | 37 | 218 | 37 |
| gt_top | voxel | oracle category median | 69 | 0% | 57 | 300 | 57 |
| lift | foodscan | oracle per-item | 49 | 0% | 68 | 97 | -58 |
| lift | foodscan | oracle category median | 49 | 0% | 71 | 100 | -54 |
| lift | hull | oracle per-item | 49 | 0% | 62 | 261 | 21 |
| lift | hull | oracle category median | 49 | 0% | 58 | 300 | 20 |
| lift | voxel | oracle per-item | 49 | 0% | 62 | 106 | -42 |
| lift | voxel | oracle category median | 49 | 0% | 70 | 114 | -43 |
| gtsub:ptv3_short | foodscan | oracle per-item | 2 | 0% | 33 | 33 | 21 |
| gtsub:ptv3_short | foodscan | oracle category median | 2 | 0% | 116 | 116 | 116 |
| gtsub:ptv3_short | hull | oracle per-item | 2 | 0% | 123 | 123 | 123 |
| gtsub:ptv3_short | hull | oracle category median | 2 | 0% | 270 | 270 | 270 |
| gtsub:ptv3_short | voxel | oracle per-item | 2 | 0% | 31 | 31 | 27 |
| gtsub:ptv3_short | voxel | oracle category median | 2 | 0% | 126 | 126 | 126 |
| pred:ptv3_short | foodscan | oracle per-item | 2 | 0% | 43 | 43 | 10 |
| pred:ptv3_short | foodscan | oracle category median | 2 | 0% | 102 | 102 | 100 |
| pred:ptv3_short | hull | oracle per-item | 2 | 0% | 102 | 102 | 102 |
| pred:ptv3_short | hull | oracle category median | 2 | 0% | 240 | 240 | 240 |
| pred:ptv3_short | voxel | oracle per-item | 2 | 0% | 44 | 44 | 13 |
| pred:ptv3_short | voxel | oracle category median | 2 | 0% | 105 | 105 | 105 |

## 7. ok-flag vs all scans (item APE)
| subset | source | method | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| ok | gt_top | foodscan | 83 | 0% | 21 | 187 | 19 |
| ok | gt_top | hull | 83 | 1% | 61 | 386 | 61 |
| ok | gt_top | voxel | 83 | 0% | 28 | 206 | 28 |
| ok | gt | foodscan | 83 | 2% | 53 | 98 | -40 |
| ok | gt | hull | 83 | 2% | 52 | 214 | 29 |
| ok | gt | voxel | 83 | 2% | 44 | 107 | -9 |
| ok | lift | foodscan | 49 | 0% | 68 | 97 | -58 |
| ok | lift | hull | 49 | 0% | 62 | 261 | 21 |
| ok | lift | voxel | 49 | 0% | 62 | 106 | -42 |
| all | gt_top | foodscan | 473 | 2% | 39 | 222 | 28 |
| all | gt_top | hull | 473 | 5% | 81 | 1176 | 81 |
| all | gt_top | voxel | 473 | 2% | 44 | 249 | 33 |
| all | gt | foodscan | 462 | 43% | 61 | 104 | -52 |
| all | gt | hull | 462 | 43% | 52 | 221 | 3 |
| all | gt | voxel | 462 | 43% | 54 | 107 | -37 |
| all | lift | foodscan | 428 | 49% | 67 | 107 | -58 |
| all | lift | hull | 428 | 49% | 58 | 246 | -0 |
| all | lift | voxel | 428 | 49% | 60 | 112 | -45 |
