# Synthetic per-item volume evaluation
Tables use **ok-flagged** scans; table plane = **gt**; items with fewer than 20 points are 'missing'. Reproduce: `conda run -n foodseg python -m pipeline.eval.synth_volume` from the repo root.
## Reading
1. Segmentation is not the bottleneck; scan visibility and the volume step are. On the 23 items / 14 scans that have lift labels, foodscan's median item APE is 94% with GT labels and 94% with lift labels (segmentation adds almost nothing); per-dish totals 86% vs 90%. The noise-free GT surface (gt_top, oracle container pose) still gives 11% median / +11% bias on the same items.
2. Refined predictions (ptv3_short, only 3 scans / 4 paired items; no results/seg3d/v2_*/preds exist, only this run dumped predictions): foodscan median APE 26% with GT labels on the same voxel subset vs 36% with the predictions (dish 15% vs 15%); far too few scans to rank segmentation sources.
3. Note: after the foodscan fix (Support.height_map_volume hole-fills and rasterises the footprint at 0.5 mm) the per-item median APE with GT labels moved from 49% to ~54%: the old estimator over-counted sparse items by accident, which partly cancelled under-seeing; occlusion dominates either way.
4. With GT labels (ok scans, n=57): median APE / median bias foodscan 54% / -41%, voxel 54% / -3%, hull 58% / +26% (mean APE 270%: the hull blows up on non-convex items), flat-table 72% / +38%. 4% of items are missing (<20 points) and 0% non-positive.
5. Where it fails 1, occlusion: items hidden >= 70% in the scan (n=17) come out -96% (median bias) with GT labels, yet the same items from the full GT surface give 5% APE: the integral is fine, the visible-only cloud has no height to integrate (column fill sums observed cells only, no completion).
6. Where it fails 2, stacking and voids: even from noise-free surfaces the container-relative column fill over-counts, +100% median bias in stacked scans (n=27) vs +16% in unstacked ones (n=30); an item sitting on another is charged for the item below, and loose piles / leaves / toppings are charged for their air gaps (gt_top median bias by kind: topping +933%, leaves +1034%, pile +140%, piece +5%).
7. By size (GT labels, foodscan): <10 mL median APE 50% (bias -19%), 10-50 mL 35%, >50 mL 79% (bias -79%): large items are the ones under-seen, small ones carry the 2 mm cell footprint bias (tests: +13% on a 1 cm cylinder, +4% on 3 cm) and the hull's +61% bias.
8. Registration: template ICP lands 1.4 mm median from the true pose but fails (>= 10 mm) on the sparse scans (12 of 57 ok items; run_meal's init is the centroid of all off-plane points, no multi-start, no fitness gate). Replacing ICP by the true pose changes the median APE only 54% -> 58%, so for these scans occlusion, not registration, is what hurts.
9. run_meal's item_volume crops the whole mesh by the item's bounding box, not by label; emulated on points (foodscan_bbox) the median APE goes 54% -> 79% (bias -41% -> -15%) because neighbours, cutlery and other items inside the box are integrated too.
10. Dish totals (what Nutrition5k-style scoring sees), foodscan: volume median APE 47% (bias -35%, n=26 dishes) with GT labels, 86% vs 90% GT vs lift labels on the 14 lift scans; under-seeing and over-counting partly cancel in the sum, so dish error is lower than item error. Mass with oracle density, 18 dishes with complete mass labels (volume APE 77% on them): 84% with the oracle category-median density, 82% with the oracle per-item density. Densities here are oracle, never predicted; mass_g is integer-rounded for small items.
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
| **total ok** | 26 |  |  | 57 | 0 | 14 |
| **total all** | 72 |  | 296 |  | 1 | 60 |

Container registration (71 scans): ICP pose error vs truth median 1.37 mm (max 412.80); support height-field RMS vs oracle-pose support median 3.49 mm (max 1197.40). Scans with no support: 0.

## 1. Per-item volume error vs `volume_ml`
| source | method | n items | missing | <=0 | med APE % | mean APE % | med bias % | mean bias % |
|---|---|---|---|---|---|---|---|---|
| gt | foodscan | 57 | 4% | 0% | 54 | 103 | -41 | 19 |
| gt | voxel | 57 | 4% | 0% | 54 | 115 | -3 | 53 |
| gt | hull | 57 | 4% | 0% | 58 | 270 | 26 | 236 |
| gt | foodscan_cinit | 57 | 16% | 0% | 50 | 106 | -22 | 34 |
| gt | foodscan_gtreg | 57 | 4% | 0% | 58 | 106 | -29 | 26 |
| gt | foodscan_bbox | 57 | 4% | 0% | 79 | 176 | -15 | 109 |
| gt | table | 57 | 4% | 0% | 72 | 204 | 38 | 160 |
| gt | hull_table | 57 | 4% | 0% | 85 | 421 | 81 | 405 |
| gt_top | foodscan | 57 | 0% | 0% | 28 | 239 | 28 | 239 |
| gt_top | voxel | 57 | 0% | 0% | 35 | 263 | 35 | 263 |
| gt_top | hull | 57 | 0% | 0% | 74 | 499 | 74 | 499 |
| gt_top | foodscan_cinit | 57 | 0% | 0% | 28 | 239 | 28 | 239 |
| gt_top | foodscan_gtreg | 57 | 0% | 0% | 28 | 239 | 28 | 239 |
| gt_top | foodscan_bbox | 57 | 0% | 0% | 83 | 462 | 83 | 462 |
| gt_top | table | 57 | 0% | 0% | 168 | 543 | 168 | 543 |
| gt_top | hull_table | 57 | 0% | 0% | 156 | 733 | 156 | 733 |
| lift | foodscan | 23 | 0% | 0% | 94 | 128 | -87 | -11 |
| lift | voxel | 23 | 0% | 0% | 92 | 144 | -70 | 29 |
| lift | hull | 23 | 0% | 0% | 70 | 458 | 0 | 404 |
| lift | foodscan_cinit | 23 | 30% | 0% | 91 | 146 | -81 | 18 |
| lift | foodscan_gtreg | 23 | 0% | 0% | 92 | 129 | -84 | 5 |
| lift | foodscan_bbox | 23 | 0% | 0% | 87 | 138 | -82 | 8 |
| lift | table | 23 | 0% | 0% | 89 | 199 | -69 | 103 |
| lift | hull_table | 23 | 4% | 0% | 87 | 676 | 40 | 641 |
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
| gt | <10 mL | foodscan | 19 | 11% | 50 | 148 | -19 |
| gt | <10 mL | hull | 19 | 11% | 61 | 399 | 61 |
| gt | <10 mL | voxel | 19 | 11% | 52 | 169 | -8 |
| gt | <10 mL | table | 19 | 11% | 157 | 362 | 157 |
| gt | 10-50 mL | foodscan | 14 | 0% | 35 | 110 | -12 |
| gt | 10-50 mL | hull | 14 | 0% | 178 | 489 | 178 |
| gt | 10-50 mL | voxel | 14 | 0% | 8 | 134 | 8 |
| gt | 10-50 mL | table | 14 | 0% | 47 | 234 | 41 |
| gt | >50 mL | foodscan | 24 | 0% | 79 | 67 | -79 |
| gt | >50 mL | hull | 24 | 0% | 45 | 50 | -6 |
| gt | >50 mL | voxel | 24 | 0% | 82 | 65 | -69 |
| gt | >50 mL | table | 24 | 0% | 83 | 74 | -47 |
| gt_top | <10 mL | foodscan | 19 | 0% | 197 | 354 | 197 |
| gt_top | <10 mL | hull | 19 | 0% | 292 | 786 | 292 |
| gt_top | <10 mL | voxel | 19 | 0% | 218 | 411 | 218 |
| gt_top | <10 mL | table | 19 | 0% | 904 | 890 | 904 |
| gt_top | 10-50 mL | foodscan | 14 | 0% | 106 | 443 | 106 |
| gt_top | 10-50 mL | hull | 14 | 0% | 246 | 853 | 246 |
| gt_top | 10-50 mL | voxel | 14 | 0% | 123 | 463 | 123 |
| gt_top | 10-50 mL | table | 14 | 0% | 230 | 805 | 230 |
| gt_top | >50 mL | foodscan | 24 | 0% | 5 | 29 | 5 |
| gt_top | >50 mL | hull | 24 | 0% | 42 | 66 | 42 |
| gt_top | >50 mL | voxel | 24 | 0% | 4 | 29 | 4 |
| gt_top | >50 mL | table | 24 | 0% | 87 | 116 | 87 |
| lift | <10 mL | foodscan | 2 | 0% | 103 | 103 | 74 |
| lift | <10 mL | hull | 2 | 0% | 1868 | 1868 | 1868 |
| lift | <10 mL | voxel | 2 | 0% | 129 | 129 | 119 |
| lift | <10 mL | table | 2 | 0% | 263 | 263 | 263 |
| lift | 10-50 mL | foodscan | 5 | 0% | 87 | 269 | -67 |
| lift | 10-50 mL | hull | 5 | 0% | 550 | 1182 | 550 |
| lift | 10-50 mL | voxel | 5 | 0% | 68 | 344 | 31 |
| lift | 10-50 mL | table | 5 | 0% | 165 | 561 | 165 |
| lift | >50 mL | foodscan | 16 | 0% | 95 | 87 | -95 |
| lift | >50 mL | hull | 16 | 0% | 51 | 55 | -31 |
| lift | >50 mL | voxel | 16 | 0% | 92 | 84 | -91 |
| lift | >50 mL | table | 16 | 0% | 86 | 78 | -83 |
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
| gt | leaves | foodscan | 5 | 0% | 87 | 288 | 79 |
| gt | leaves | hull | 5 | 0% | 934 | 1896 | 934 |
| gt | leaves | voxel | 5 | 0% | 135 | 403 | 135 |
| gt | piece | foodscan | 26 | 0% | 54 | 52 | -41 |
| gt | piece | hull | 26 | 0% | 27 | 42 | 1 |
| gt | piece | voxel | 26 | 0% | 40 | 49 | -34 |
| gt | pile | foodscan | 16 | 12% | 51 | 46 | -51 |
| gt | pile | hull | 16 | 12% | 57 | 102 | 47 |
| gt | pile | voxel | 16 | 12% | 42 | 51 | -13 |
| gt | single | foodscan | 8 | 0% | 65 | 59 | -40 |
| gt | single | hull | 8 | 0% | 85 | 111 | 85 |
| gt | single | voxel | 8 | 0% | 56 | 47 | 8 |
| gt | topping | foodscan | 2 | 0% | 882 | 882 | 882 |
| gt | topping | hull | 2 | 0% | 966 | 966 | 966 |
| gt | topping | voxel | 2 | 0% | 960 | 960 | 960 |
| gt_top | leaves | foodscan | 5 | 0% | 1034 | 1436 | 1034 |
| gt_top | leaves | hull | 5 | 0% | 3210 | 3393 | 3210 |
| gt_top | leaves | voxel | 5 | 0% | 1060 | 1487 | 1060 |
| gt_top | piece | foodscan | 26 | 0% | 5 | 29 | 5 |
| gt_top | piece | hull | 26 | 0% | 31 | 60 | 31 |
| gt_top | piece | voxel | 26 | 0% | 4 | 29 | 4 |
| gt_top | pile | foodscan | 16 | 0% | 140 | 170 | 140 |
| gt_top | pile | hull | 16 | 0% | 234 | 373 | 234 |
| gt_top | pile | voxel | 16 | 0% | 167 | 222 | 167 |
| gt_top | single | foodscan | 8 | 0% | 76 | 141 | 76 |
| gt_top | single | hull | 8 | 0% | 158 | 249 | 158 |
| gt_top | single | voxel | 8 | 0% | 90 | 155 | 90 |
| gt_top | topping | foodscan | 2 | 0% | 933 | 933 | 933 |
| gt_top | topping | hull | 2 | 0% | 982 | 982 | 982 |
| gt_top | topping | voxel | 2 | 0% | 1002 | 1002 | 1002 |
| lift | leaves | foodscan | 5 | 0% | 87 | 279 | 79 |
| lift | leaves | hull | 5 | 0% | 934 | 1878 | 934 |
| lift | leaves | voxel | 5 | 0% | 135 | 390 | 135 |
| lift | piece | foodscan | 14 | 0% | 96 | 87 | -96 |
| lift | piece | hull | 14 | 0% | 51 | 57 | -31 |
| lift | piece | voxel | 14 | 0% | 92 | 83 | -92 |
| lift | single | foodscan | 4 | 0% | 79 | 80 | -79 |
| lift | single | hull | 4 | 0% | 48 | 86 | 10 |
| lift | single | voxel | 4 | 0% | 62 | 53 | -62 |
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
| gt | bowl | foodscan | 39 | 5% | 61 | 127 | -51 |
| gt | bowl | hull | 39 | 5% | 61 | 251 | 26 |
| gt | bowl | voxel | 39 | 5% | 54 | 141 | -3 |
| gt | plate | foodscan | 18 | 0% | 47 | 54 | -11 |
| gt | plate | hull | 18 | 0% | 47 | 309 | 24 |
| gt | plate | voxel | 18 | 0% | 50 | 60 | -6 |
| gt_top | bowl | foodscan | 39 | 0% | 72 | 268 | 72 |
| gt_top | bowl | hull | 39 | 0% | 149 | 511 | 149 |
| gt_top | bowl | voxel | 39 | 0% | 101 | 300 | 101 |
| gt_top | plate | foodscan | 18 | 0% | 9 | 177 | 9 |
| gt_top | plate | hull | 18 | 0% | 42 | 472 | 42 |
| gt_top | plate | voxel | 18 | 0% | 13 | 184 | 13 |
| lift | bowl | foodscan | 15 | 0% | 96 | 147 | -94 |
| lift | bowl | hull | 15 | 0% | 44 | 357 | 15 |
| lift | bowl | voxel | 15 | 0% | 92 | 167 | -89 |
| lift | plate | foodscan | 8 | 0% | 87 | 92 | -85 |
| lift | plate | hull | 8 | 0% | 85 | 646 | -29 |
| lift | plate | voxel | 8 | 0% | 88 | 103 | -69 |
| pred:ptv3_short | bowl | foodscan | 3 | 33% | 21 | 21 | -12 |
| pred:ptv3_short | bowl | hull | 3 | 33% | 76 | 76 | 76 |
| pred:ptv3_short | bowl | voxel | 3 | 33% | 18 | 18 | -14 |
| pred:ptv3_short | plate | foodscan | 2 | 0% | 47 | 47 | 7 |
| pred:ptv3_short | plate | hull | 2 | 0% | 40 | 40 | 32 |
| pred:ptv3_short | plate | voxel | 2 | 0% | 49 | 49 | 8 |

**by asset_source**

| source | asset_source | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | metafood3d | foodscan | 24 | 8% | 52 | 51 | -51 |
| gt | metafood3d | hull | 24 | 8% | 72 | 105 | 68 |
| gt | metafood3d | voxel | 24 | 8% | 47 | 50 | 7 |
| gt | procedural_leaves | foodscan | 5 | 0% | 87 | 288 | 79 |
| gt | procedural_leaves | hull | 5 | 0% | 934 | 1896 | 934 |
| gt | procedural_leaves | voxel | 5 | 0% | 135 | 403 | 135 |
| gt | procedural_piece | foodscan | 26 | 0% | 54 | 52 | -41 |
| gt | procedural_piece | hull | 26 | 0% | 27 | 42 | 1 |
| gt | procedural_piece | voxel | 26 | 0% | 40 | 49 | -34 |
| gt | procedural_topping | foodscan | 2 | 0% | 882 | 882 | 882 |
| gt | procedural_topping | hull | 2 | 0% | 966 | 966 | 966 |
| gt | procedural_topping | voxel | 2 | 0% | 960 | 960 | 960 |
| gt_top | metafood3d | foodscan | 24 | 0% | 116 | 160 | 116 |
| gt_top | metafood3d | hull | 24 | 0% | 200 | 332 | 200 |
| gt_top | metafood3d | voxel | 24 | 0% | 120 | 200 | 120 |
| gt_top | procedural_leaves | foodscan | 5 | 0% | 1034 | 1436 | 1034 |
| gt_top | procedural_leaves | hull | 5 | 0% | 3210 | 3393 | 3210 |
| gt_top | procedural_leaves | voxel | 5 | 0% | 1060 | 1487 | 1060 |
| gt_top | procedural_piece | foodscan | 26 | 0% | 5 | 29 | 5 |
| gt_top | procedural_piece | hull | 26 | 0% | 31 | 60 | 31 |
| gt_top | procedural_piece | voxel | 26 | 0% | 4 | 29 | 4 |
| gt_top | procedural_topping | foodscan | 2 | 0% | 933 | 933 | 933 |
| gt_top | procedural_topping | hull | 2 | 0% | 982 | 982 | 982 |
| gt_top | procedural_topping | voxel | 2 | 0% | 1002 | 1002 | 1002 |
| lift | metafood3d | foodscan | 4 | 0% | 79 | 80 | -79 |
| lift | metafood3d | hull | 4 | 0% | 48 | 86 | 10 |
| lift | metafood3d | voxel | 4 | 0% | 62 | 53 | -62 |
| lift | procedural_leaves | foodscan | 5 | 0% | 87 | 279 | 79 |
| lift | procedural_leaves | hull | 5 | 0% | 934 | 1878 | 934 |
| lift | procedural_leaves | voxel | 5 | 0% | 135 | 390 | 135 |
| lift | procedural_piece | foodscan | 14 | 0% | 96 | 87 | -96 |
| lift | procedural_piece | hull | 14 | 0% | 51 | 57 | -31 |
| lift | procedural_piece | voxel | 14 | 0% | 92 | 83 | -92 |
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
| gt | hard_v10 | foodscan | 13 | 0% | 89 | 160 | -87 |
| gt | hard_v10 | hull | 13 | 0% | 70 | 482 | 0 |
| gt | hard_v10 | voxel | 13 | 0% | 89 | 194 | -70 |
| gt | hard_v11 | foodscan | 2 | 0% | 85 | 85 | -85 |
| gt | hard_v11 | hull | 2 | 0% | 67 | 67 | -30 |
| gt | hard_v11 | voxel | 2 | 0% | 84 | 84 | -84 |
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
| gt_top | hard_v3 | foodscan | 17 | 0% | 62 | 124 | 62 |
| gt_top | hard_v3 | hull | 17 | 0% | 74 | 230 | 74 |
| gt_top | hard_v3 | voxel | 17 | 0% | 90 | 146 | 90 |
| gt_top | hard_v9 | foodscan | 8 | 0% | 15 | 385 | 15 |
| gt_top | hard_v9 | hull | 8 | 0% | 116 | 1083 | 116 |
| gt_top | hard_v9 | voxel | 8 | 0% | 15 | 407 | 15 |
| lift | hard_v10 | foodscan | 13 | 0% | 89 | 156 | -87 |
| lift | hard_v10 | hull | 13 | 0% | 70 | 476 | 0 |
| lift | hard_v10 | voxel | 13 | 0% | 91 | 190 | -70 |
| lift | hard_v11 | foodscan | 2 | 0% | 85 | 85 | -85 |
| lift | hard_v11 | hull | 2 | 0% | 49 | 49 | -12 |
| lift | hard_v11 | voxel | 2 | 0% | 85 | 85 | -85 |
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
| gt | 0.3-0.7 | foodscan | 6 | 0% | 84 | 253 | 24 |
| gt | 0.3-0.7 | hull | 6 | 0% | 485 | 1434 | 485 |
| gt | 0.3-0.7 | voxel | 6 | 0% | 102 | 345 | 83 |
| gt | >=0.7 | foodscan | 17 | 0% | 96 | 86 | -96 |
| gt | >=0.7 | hull | 17 | 0% | 79 | 124 | -18 |
| gt | >=0.7 | voxel | 17 | 0% | 92 | 78 | -89 |
| gt | n/a | foodscan | 34 | 6% | 33 | 84 | -6 |
| gt | n/a | hull | 34 | 6% | 40 | 128 | 30 |
| gt | n/a | voxel | 34 | 6% | 34 | 91 | 4 |
| gt_top | 0.3-0.7 | foodscan | 6 | 0% | 573 | 1011 | 573 |
| gt_top | 0.3-0.7 | hull | 6 | 0% | 1021 | 2115 | 1021 |
| gt_top | 0.3-0.7 | voxel | 6 | 0% | 581 | 1052 | 581 |
| gt_top | >=0.7 | foodscan | 17 | 0% | 5 | 152 | 5 |
| gt_top | >=0.7 | hull | 17 | 0% | 49 | 401 | 49 |
| gt_top | >=0.7 | voxel | 17 | 0% | 6 | 157 | 6 |
| gt_top | n/a | foodscan | 34 | 0% | 67 | 147 | 67 |
| gt_top | n/a | hull | 34 | 0% | 89 | 263 | 89 |
| gt_top | n/a | voxel | 34 | 0% | 86 | 177 | 86 |
| lift | 0.3-0.7 | foodscan | 6 | 0% | 84 | 245 | 24 |
| lift | 0.3-0.7 | hull | 6 | 0% | 485 | 1423 | 485 |
| lift | 0.3-0.7 | voxel | 6 | 0% | 102 | 333 | 83 |
| lift | >=0.7 | foodscan | 17 | 0% | 94 | 86 | -94 |
| lift | >=0.7 | hull | 17 | 0% | 70 | 117 | -17 |
| lift | >=0.7 | voxel | 17 | 0% | 92 | 78 | -91 |
| lift | n/a | foodscan | 0 | - | - | - | - |
| lift | n/a | hull | 0 | - | - | - | - |
| lift | n/a | voxel | 0 | - | - | - | - |
| pred:ptv3_short | n/a | foodscan | 5 | 20% | 36 | 34 | -12 |
| pred:ptv3_short | n/a | hull | 5 | 20% | 46 | 58 | 46 |
| pred:ptv3_short | n/a | voxel | 5 | 20% | 36 | 33 | -14 |

**by stacked_scan**

| source | stacked_scan | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | False | foodscan | 30 | 3% | 66 | 93 | -46 |
| gt | False | hull | 30 | 3% | 58 | 385 | 26 |
| gt | False | voxel | 30 | 3% | 41 | 106 | -8 |
| gt | True | foodscan | 27 | 4% | 52 | 114 | -30 |
| gt | True | hull | 27 | 4% | 57 | 140 | 21 |
| gt | True | voxel | 27 | 4% | 58 | 125 | -3 |
| gt_top | False | foodscan | 30 | 0% | 16 | 280 | 16 |
| gt_top | False | hull | 30 | 0% | 47 | 650 | 47 |
| gt_top | False | voxel | 30 | 0% | 25 | 294 | 25 |
| gt_top | True | foodscan | 27 | 0% | 100 | 193 | 100 |
| gt_top | True | hull | 27 | 0% | 125 | 331 | 125 |
| gt_top | True | voxel | 27 | 0% | 106 | 229 | 106 |
| lift | False | foodscan | 16 | 0% | 91 | 145 | -81 |
| lift | False | hull | 16 | 0% | 74 | 625 | -3 |
| lift | False | voxel | 16 | 0% | 92 | 175 | -61 |
| lift | True | foodscan | 7 | 0% | 96 | 90 | -96 |
| lift | True | hull | 7 | 0% | 70 | 76 | 15 |
| lift | True | voxel | 7 | 0% | 89 | 75 | -89 |
| pred:ptv3_short | False | foodscan | 1 | 0% | 33 | 33 | -33 |
| pred:ptv3_short | False | hull | 1 | 0% | 132 | 132 | 132 |
| pred:ptv3_short | False | voxel | 1 | 0% | 32 | 32 | -32 |
| pred:ptv3_short | True | foodscan | 4 | 25% | 40 | 34 | 8 |
| pred:ptv3_short | True | hull | 4 | 25% | 20 | 34 | 20 |
| pred:ptv3_short | True | voxel | 4 | 25% | 41 | 34 | 4 |

**by icp_pose_error**

| source | icp_pose_error | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | <10 mm | foodscan | 45 | 4% | 46 | 107 | -18 |
| gt | <10 mm | hull | 45 | 4% | 58 | 312 | 26 |
| gt | <10 mm | voxel | 45 | 4% | 41 | 125 | 1 |
| gt | <10 mm | foodscan_gtreg | 45 | 4% | 51 | 112 | -20 |
| gt | <10 mm | foodscan_cinit | 45 | 7% | 44 | 109 | -16 |
| gt | >=10 mm (failed) | foodscan | 12 | 0% | 97 | 88 | -97 |
| gt | >=10 mm (failed) | hull | 12 | 0% | 70 | 119 | 8 |
| gt | >=10 mm (failed) | voxel | 12 | 0% | 91 | 77 | -89 |
| gt | >=10 mm (failed) | foodscan_gtreg | 12 | 0% | 93 | 86 | -92 |
| gt | >=10 mm (failed) | foodscan_cinit | 12 | 50% | 89 | 84 | -89 |
| gt_top | <10 mm | foodscan | 45 | 0% | 51 | 257 | 51 |
| gt_top | <10 mm | hull | 45 | 0% | 78 | 543 | 78 |
| gt_top | <10 mm | voxel | 45 | 0% | 57 | 286 | 57 |
| gt_top | <10 mm | foodscan_gtreg | 45 | 0% | 51 | 257 | 51 |
| gt_top | <10 mm | foodscan_cinit | 45 | 0% | 51 | 257 | 51 |
| gt_top | >=10 mm (failed) | foodscan | 12 | 0% | 8 | 172 | 8 |
| gt_top | >=10 mm (failed) | hull | 12 | 0% | 55 | 334 | 55 |
| gt_top | >=10 mm (failed) | voxel | 12 | 0% | 9 | 177 | 9 |
| gt_top | >=10 mm (failed) | foodscan_gtreg | 12 | 0% | 8 | 172 | 8 |
| gt_top | >=10 mm (failed) | foodscan_cinit | 12 | 0% | 8 | 172 | 8 |
| lift | <10 mm | foodscan | 11 | 0% | 87 | 170 | -75 |
| lift | <10 mm | hull | 11 | 0% | 79 | 835 | -6 |
| lift | <10 mm | voxel | 11 | 0% | 91 | 218 | -54 |
| lift | <10 mm | foodscan_gtreg | 11 | 0% | 88 | 177 | -75 |
| lift | <10 mm | foodscan_cinit | 11 | 9% | 91 | 182 | -56 |
| lift | >=10 mm (failed) | foodscan | 12 | 0% | 97 | 89 | -97 |
| lift | >=10 mm (failed) | hull | 12 | 0% | 49 | 111 | 15 |
| lift | >=10 mm (failed) | voxel | 12 | 0% | 92 | 77 | -91 |
| lift | >=10 mm (failed) | foodscan_gtreg | 12 | 0% | 93 | 86 | -91 |
| lift | >=10 mm (failed) | foodscan_cinit | 12 | 50% | 89 | 85 | -89 |
| pred:ptv3_short | <10 mm | foodscan | 5 | 20% | 36 | 34 | -12 |
| pred:ptv3_short | <10 mm | hull | 5 | 20% | 46 | 58 | 46 |
| pred:ptv3_short | <10 mm | voxel | 5 | 20% | 36 | 33 | -14 |
| pred:ptv3_short | <10 mm | foodscan_gtreg | 5 | 20% | 37 | 34 | -13 |
| pred:ptv3_short | <10 mm | foodscan_cinit | 5 | 20% | 36 | 34 | -12 |

## 3. Against `volume_above_container_ml` (rim-plane definition; items with >= 1 mL above the rim)
| source | method | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|
| gt | foodscan | 12 | 0% | 75 | 111 | -21 |
| gt | table | 12 | 0% | 84 | 506 | 52 |
| gt | hull_table | 12 | 0% | 498 | 1610 | 498 |
| gt | (foodscan vs total volume_ml, same items) | 12 | 0% | 91 | 175 | -85 |
| gt_top | foodscan | 12 | 0% | 794 | 1672 | 794 |
| gt_top | table | 12 | 0% | 1367 | 2789 | 1367 |
| gt_top | hull_table | 12 | 0% | 2572 | 3185 | 2572 |
| gt_top | (foodscan vs total volume_ml, same items) | 12 | 0% | 35 | 527 | 35 |

## 4. Error decomposition (paired: same items valid in every listed source)
Ladder gt_top -> gt (+ sensor sparsity / noise / occlusion) -> lift or pred (+ segmentation). `gtsub` = GT labels on the prediction subset.

| ladder | source | method | items | scans | item med APE % | item mean APE % | item med bias % | dish med APE % | dish mean APE % | dish med bias % |
|---|---|---|---|---|---|---|---|---|---|---|
| lift | gt_top | foodscan | 23 | 14 | 11 | 376 | 11 | 41 | 207 | 41 |
| lift | gt | foodscan | 23 | 14 | 94 | 129 | -87 | 86 | 76 | -86 |
| lift | lift | foodscan | 23 | 14 | 94 | 128 | -87 | 90 | 108 | -86 |
| lift | gt_top | hull | 23 | 14 | 59 | 848 | 59 | 108 | 556 | 108 |
| lift | gt | hull | 23 | 14 | 79 | 466 | -6 | 67 | 169 | 0 |
| lift | lift | hull | 23 | 14 | 70 | 458 | 0 | 66 | 718 | -0 |
| lift | gt_top | voxel | 23 | 14 | 13 | 390 | 13 | 41 | 209 | 41 |
| lift | gt | voxel | 23 | 14 | 92 | 148 | -70 | 77 | 70 | -71 |
| lift | lift | voxel | 23 | 14 | 92 | 144 | -70 | 78 | 94 | -73 |
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
| gt | foodscan | 26 | 47 | 50 | -35 | 18 | 77 | 84 | 77 | -67 | 82 |
| gt | voxel | 26 | 30 | 43 | -8 | 18 | 61 | 76 | 81 | -52 | 70 |
| gt | hull | 26 | 51 | 125 | 42 | 18 | 80 | 79 | 416 | 10 | 79 |
| gt | table | 26 | 46 | 64 | 9 | 18 | 52 | 79 | 99 | -17 | 56 |
| gt_top | foodscan | 26 | 41 | 134 | 41 | 18 | 51 | 48 | 572 | 48 | 41 |
| gt_top | voxel | 26 | 41 | 138 | 41 | 18 | 52 | 50 | 582 | 50 | 43 |
| gt_top | hull | 26 | 73 | 348 | 73 | 18 | 154 | 103 | 1550 | 103 | 110 |
| gt_top | table | 26 | 155 | 292 | 155 | 18 | 191 | 166 | 978 | 166 | 167 |
| lift | foodscan | 14 | 90 | 108 | -86 | 14 | 90 | 93 | 120 | -87 | 90 |
| lift | voxel | 14 | 78 | 94 | -73 | 14 | 78 | 84 | 117 | -78 | 81 |
| lift | hull | 14 | 66 | 718 | -0 | 14 | 66 | 58 | 1011 | -17 | 58 |
| lift | table | 14 | 65 | 377 | -46 | 14 | 65 | 78 | 413 | -41 | 76 |
| gtsub:ptv3_short | foodscan | 3 | 15 | 20 | 15 | 1 | 12 | 26 | 26 | 26 | 12 |
| gtsub:ptv3_short | voxel | 3 | 17 | 17 | 17 | 1 | 5 | 37 | 37 | 37 | 5 |
| gtsub:ptv3_short | hull | 3 | 41 | 84 | 41 | 1 | 173 | 293 | 293 | 293 | 173 |
| gtsub:ptv3_short | table | 3 | 36 | 75 | 36 | 1 | 36 | 96 | 96 | 96 | 36 |
| pred:ptv3_short | foodscan | 3 | 15 | 18 | 5 | 1 | 33 | 3 | 3 | -3 | 33 |
| pred:ptv3_short | voxel | 3 | 17 | 16 | 1 | 1 | 32 | 1 | 1 | -1 | 32 |
| pred:ptv3_short | hull | 3 | 39 | 62 | 39 | 1 | 132 | 234 | 234 | 234 | 132 |
| pred:ptv3_short | table | 3 | 32 | 62 | 32 | 1 | 6 | 52 | 52 | 52 | 6 |

## 6. Per-item mass with oracle density
`oracle per-item` density = gt mass / gt volume of the same item, so its mass APE equals the volume APE (shown as a check); `oracle category median` is the non-tautological one. Neither is a deployable density.

| source | method | density | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | foodscan | oracle per-item | 43 | 5% | 73 | 92 | -54 |
| gt | foodscan | oracle category median | 43 | 5% | 79 | 100 | -51 |
| gt | hull | oracle per-item | 43 | 5% | 75 | 309 | 34 |
| gt | hull | oracle category median | 43 | 5% | 79 | 433 | 37 |
| gt | voxel | oracle per-item | 43 | 5% | 68 | 104 | -34 |
| gt | voxel | oracle category median | 43 | 5% | 75 | 122 | 7 |
| gt_top | foodscan | oracle per-item | 43 | 0% | 80 | 272 | 80 |
| gt_top | foodscan | oracle category median | 43 | 0% | 140 | 465 | 140 |
| gt_top | hull | oracle per-item | 43 | 0% | 135 | 609 | 135 |
| gt_top | hull | oracle category median | 43 | 0% | 158 | 1108 | 158 |
| gt_top | voxel | oracle per-item | 43 | 0% | 101 | 300 | 101 |
| gt_top | voxel | oracle category median | 43 | 0% | 156 | 499 | 156 |
| lift | foodscan | oracle per-item | 23 | 0% | 94 | 128 | -87 |
| lift | foodscan | oracle category median | 23 | 0% | 94 | 134 | -87 |
| lift | hull | oracle per-item | 23 | 0% | 70 | 458 | 0 |
| lift | hull | oracle category median | 23 | 0% | 75 | 644 | -11 |
| lift | voxel | oracle per-item | 23 | 0% | 92 | 144 | -70 |
| lift | voxel | oracle category median | 23 | 0% | 92 | 160 | -75 |
| gtsub:ptv3_short | foodscan | oracle per-item | 2 | 0% | 33 | 33 | 21 |
| gtsub:ptv3_short | foodscan | oracle category median | 2 | 0% | 112 | 112 | 112 |
| gtsub:ptv3_short | hull | oracle per-item | 2 | 0% | 123 | 123 | 123 |
| gtsub:ptv3_short | hull | oracle category median | 2 | 0% | 264 | 264 | 264 |
| gtsub:ptv3_short | voxel | oracle per-item | 2 | 0% | 31 | 31 | 27 |
| gtsub:ptv3_short | voxel | oracle category median | 2 | 0% | 122 | 122 | 122 |
| pred:ptv3_short | foodscan | oracle per-item | 2 | 0% | 43 | 43 | 10 |
| pred:ptv3_short | foodscan | oracle category median | 2 | 0% | 100 | 100 | 97 |
| pred:ptv3_short | hull | oracle per-item | 2 | 0% | 102 | 102 | 102 |
| pred:ptv3_short | hull | oracle category median | 2 | 0% | 234 | 234 | 234 |
| pred:ptv3_short | voxel | oracle per-item | 2 | 0% | 44 | 44 | 13 |
| pred:ptv3_short | voxel | oracle category median | 2 | 0% | 103 | 103 | 101 |

## 7. ok-flag vs all scans (item APE)
| subset | source | method | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| ok | gt_top | foodscan | 57 | 0% | 28 | 239 | 28 |
| ok | gt_top | hull | 57 | 0% | 74 | 499 | 74 |
| ok | gt_top | voxel | 57 | 0% | 35 | 263 | 35 |
| ok | gt | foodscan | 57 | 4% | 54 | 103 | -41 |
| ok | gt | hull | 57 | 4% | 58 | 270 | 26 |
| ok | gt | voxel | 57 | 4% | 54 | 115 | -3 |
| ok | lift | foodscan | 23 | 0% | 94 | 128 | -87 |
| ok | lift | hull | 23 | 0% | 70 | 458 | 0 |
| ok | lift | voxel | 23 | 0% | 92 | 144 | -70 |
| all | gt_top | foodscan | 296 | 4% | 46 | 285 | 43 |
| all | gt_top | hull | 296 | 5% | 97 | 1694 | 97 |
| all | gt_top | voxel | 296 | 4% | 51 | 322 | 50 |
| all | gt | foodscan | 285 | 61% | 79 | 91 | -74 |
| all | gt | hull | 285 | 61% | 70 | 176 | -4 |
| all | gt | voxel | 285 | 61% | 74 | 96 | -61 |
| all | lift | foodscan | 251 | 70% | 88 | 94 | -87 |
| all | lift | hull | 251 | 70% | 74 | 193 | -21 |
| all | lift | voxel | 251 | 70% | 84 | 97 | -81 |

## Note: solid vs envelope target

`gt_volume_ml` is the solid mesh volume; a scan measures the envelope (visible top surface closed to the container, voids included). GT envelope = 2 mm column fill of the full GT top surface, results/rq1/envelope_items.csv. Re-scored headline (ok scans, GT labels, median APE / bias vs solid -> vs envelope): see the 'synth_volume headline re-scored' tables in results/rq1/envelope.md and results/rq1/summary.md. Short version: `gt_top` 28% -> 1% (v12 subset 15% -> 1%), foodscan on the sparse fused cloud 54% -> 76% (v12 subset 43% -> 38%), i.e. occlusion, not the target definition, dominates here; per-kind envelope/solid ratios: results/rq1/summary.md.
