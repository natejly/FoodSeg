# Synthetic per-item volume evaluation
Tables use **all** scans; table plane = **gt**; items with fewer than 20 points are 'missing'. Reproduce: `conda run -n foodseg python -m pipeline.eval.synth_volume --all` from the repo root.
## Reading
1. Segmentation is not the bottleneck; scan visibility and the volume step are. On the 141 items / 20 scans that have lift labels, foodscan's median item APE is 50% with GT labels and 54% with lift labels (segmentation adds almost nothing); per-dish totals 30% vs 31%. The noise-free GT surface (gt_top, oracle container pose) still gives 32% median / +28% bias on the same items.
2. Note: after the foodscan fix (Support.height_map_volume hole-fills and rasterises the footprint at 0.5 mm) the per-item median APE with GT labels moved from 49% to ~54%: the old estimator over-counted sparse items by accident, which partly cancelled under-seeing; occlusion dominates either way.
3. With GT labels (ok scans, n=177): median APE / median bias foodscan 53% / -40%, voxel 46% / -26%, hull 40% / +5% (mean APE 254%: the hull blows up on non-convex items), flat-table 40% / -18%. 13% of items are missing (<20 points) and 0% non-positive.
4. Where it fails 1, occlusion: items hidden >= 70% in the scan (n=98) come out -57% (median bias) with GT labels, yet the same items from the full GT surface give 31% APE: the integral is fine, the visible-only cloud has no height to integrate (column fill sums observed cells only, no completion).
5. Where it fails 2, stacking and voids: even from noise-free surfaces the container-relative column fill over-counts, +24% median bias in stacked scans (n=62) vs +16% in unstacked ones (n=115); an item sitting on another is charged for the item below, and loose piles / leaves / toppings are charged for their air gaps (gt_top median bias by kind: topping +261%, leaves +1448%, pile +14%, piece +8%).
6. By size (GT labels, foodscan): <10 mL median APE 77% (bias -35%), 10-50 mL 52%, >50 mL 40% (bias -39%): large items are the ones under-seen, small ones carry the 2 mm cell footprint bias (tests: +13% on a 1 cm cylinder, +4% on 3 cm) and the hull's +8% bias.
7. Registration: template ICP lands 3.7 mm median from the true pose but fails (>= 10 mm) on the sparse scans (31 of 177 ok items; run_meal's init is the centroid of all off-plane points, no multi-start, no fitness gate). Replacing ICP by the true pose changes the median APE only 53% -> 59%, so for these scans occlusion, not registration, is what hurts.
8. run_meal's item_volume crops the whole mesh by the item's bounding box, not by label; emulated on points (foodscan_bbox) the median APE goes 53% -> 41% (bias -40% -> -5%) because neighbours, cutlery and other items inside the box are integrated too.
9. Dish totals (what Nutrition5k-style scoring sees), foodscan: volume median APE 30% (bias -30%, n=20 dishes) with GT labels, 30% vs 31% GT vs lift labels on the 20 lift scans; under-seeing and over-counting partly cancel in the sum, so dish error is lower than item error. Mass with oracle density, 20 dishes with complete mass labels (volume APE 30% on them): 30% with the oracle category-median density, 30% with the oracle per-item density. Densities here are oracle, never predicted; mass_g is integer-rounded for small items.
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
| hard_v12 | 20 | 5 | 177 | 26 | 0 | 20 |
| **total ok** | 5 |  |  | 26 | 0 | 5 |
| **total all** | 20 |  | 177 |  | 0 | 20 |

Container registration (20 scans): ICP pose error vs truth median 3.74 mm (max 33.27); support height-field RMS vs oracle-pose support median 4.60 mm (max 9.08). Scans with no support: 0.

## 1. Per-item volume error vs `volume_ml`
| source | method | n items | missing | <=0 | med APE % | mean APE % | med bias % | mean bias % |
|---|---|---|---|---|---|---|---|---|
| gt | foodscan | 177 | 13% | 0% | 53 | 113 | -40 | 31 |
| gt | voxel | 177 | 13% | 0% | 46 | 115 | -26 | 49 |
| gt | hull | 177 | 14% | 0% | 40 | 254 | 5 | 217 |
| gt | foodscan_cinit | 177 | 13% | 0% | 53 | 113 | -40 | 31 |
| gt | foodscan_gtreg | 177 | 13% | 0% | 59 | 113 | -48 | 23 |
| gt | foodscan_bbox | 177 | 13% | 0% | 41 | 226 | -5 | 184 |
| gt | table | 177 | 13% | 0% | 40 | 164 | -18 | 109 |
| gt | hull_table | 177 | 13% | 0% | 44 | 316 | 19 | 285 |
| gt_top | foodscan | 177 | 0% | 0% | 34 | 121 | 20 | 103 |
| gt_top | voxel | 177 | 0% | 0% | 35 | 131 | 23 | 114 |
| gt_top | hull | 177 | 5% | 0% | 68 | 317 | 63 | 310 |
| gt_top | foodscan_cinit | 177 | 0% | 0% | 34 | 121 | 20 | 103 |
| gt_top | foodscan_gtreg | 177 | 0% | 0% | 34 | 121 | 20 | 103 |
| gt_top | foodscan_bbox | 177 | 0% | 0% | 94 | 291 | 94 | 289 |
| gt_top | table | 177 | 0% | 0% | 77 | 229 | 77 | 227 |
| gt_top | hull_table | 177 | 5% | 0% | 87 | 386 | 87 | 384 |
| lift | foodscan | 177 | 20% | 0% | 54 | 114 | -42 | 34 |
| lift | voxel | 177 | 20% | 0% | 47 | 120 | -28 | 56 |
| lift | hull | 177 | 20% | 0% | 46 | 276 | 7 | 238 |
| lift | foodscan_cinit | 177 | 20% | 0% | 54 | 114 | -42 | 34 |
| lift | foodscan_gtreg | 177 | 20% | 0% | 57 | 115 | -50 | 27 |
| lift | foodscan_bbox | 177 | 20% | 0% | 47 | 238 | -1 | 197 |
| lift | table | 177 | 20% | 0% | 45 | 170 | -19 | 113 |
| lift | hull_table | 177 | 20% | 0% | 53 | 338 | 20 | 306 |

## 2. By item size, kind, container (primary methods)

**by size**

| source | size | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | <10 mL | foodscan | 41 | 20% | 77 | 277 | -35 |
| gt | <10 mL | hull | 41 | 20% | 55 | 857 | 8 |
| gt | <10 mL | voxel | 41 | 20% | 75 | 300 | -19 |
| gt | <10 mL | table | 41 | 20% | 49 | 507 | -10 |
| gt | 10-50 mL | foodscan | 91 | 11% | 52 | 77 | -44 |
| gt | 10-50 mL | hull | 91 | 12% | 40 | 105 | -1 |
| gt | 10-50 mL | voxel | 91 | 11% | 38 | 72 | -26 |
| gt | 10-50 mL | table | 91 | 11% | 40 | 82 | -21 |
| gt | >50 mL | foodscan | 45 | 11% | 40 | 48 | -39 |
| gt | >50 mL | hull | 45 | 11% | 38 | 56 | 25 |
| gt | >50 mL | voxel | 45 | 11% | 39 | 48 | -31 |
| gt | >50 mL | table | 45 | 11% | 39 | 48 | -16 |
| gt_top | <10 mL | foodscan | 41 | 0% | 45 | 288 | 25 |
| gt_top | <10 mL | hull | 41 | 0% | 55 | 896 | 55 |
| gt_top | <10 mL | voxel | 41 | 0% | 46 | 317 | 29 |
| gt_top | <10 mL | table | 41 | 0% | 120 | 580 | 120 |
| gt_top | 10-50 mL | foodscan | 91 | 0% | 32 | 84 | 20 |
| gt_top | 10-50 mL | hull | 91 | 7% | 68 | 158 | 61 |
| gt_top | 10-50 mL | voxel | 91 | 0% | 37 | 89 | 23 |
| gt_top | 10-50 mL | table | 91 | 0% | 76 | 144 | 76 |
| gt_top | >50 mL | foodscan | 45 | 0% | 25 | 46 | 16 |
| gt_top | >50 mL | hull | 45 | 4% | 70 | 78 | 70 |
| gt_top | >50 mL | voxel | 45 | 0% | 26 | 48 | 13 |
| gt_top | >50 mL | table | 45 | 0% | 56 | 82 | 56 |
| lift | <10 mL | foodscan | 41 | 34% | 83 | 308 | 7 |
| lift | <10 mL | hull | 41 | 34% | 89 | 1075 | 89 |
| lift | <10 mL | voxel | 41 | 34% | 85 | 352 | 53 |
| lift | <10 mL | table | 41 | 34% | 86 | 585 | 49 |
| lift | 10-50 mL | foodscan | 91 | 18% | 55 | 80 | -48 |
| lift | 10-50 mL | hull | 91 | 18% | 43 | 101 | -5 |
| lift | 10-50 mL | voxel | 91 | 18% | 41 | 75 | -33 |
| lift | 10-50 mL | table | 91 | 18% | 43 | 84 | -31 |
| lift | >50 mL | foodscan | 45 | 11% | 41 | 49 | -39 |
| lift | >50 mL | hull | 45 | 11% | 45 | 64 | 30 |
| lift | >50 mL | voxel | 45 | 11% | 41 | 48 | -33 |
| lift | >50 mL | table | 45 | 11% | 44 | 49 | -18 |

**by kind**

| source | kind | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | leaves | foodscan | 5 | 0% | 1286 | 1328 | 1286 |
| gt | leaves | hull | 5 | 0% | 2461 | 5314 | 2461 |
| gt | leaves | voxel | 5 | 0% | 1338 | 1453 | 1338 |
| gt | piece | foodscan | 11 | 9% | 49 | 41 | -49 |
| gt | piece | hull | 11 | 9% | 37 | 42 | 37 |
| gt | piece | voxel | 11 | 9% | 41 | 37 | -41 |
| gt | pile | foodscan | 96 | 16% | 52 | 54 | -46 |
| gt | pile | hull | 96 | 17% | 31 | 46 | -16 |
| gt | pile | voxel | 96 | 16% | 40 | 48 | -31 |
| gt | single | foodscan | 59 | 8% | 48 | 82 | -40 |
| gt | single | hull | 59 | 8% | 48 | 112 | 33 |
| gt | single | voxel | 59 | 8% | 44 | 80 | -24 |
| gt | topping | foodscan | 6 | 33% | 265 | 381 | 265 |
| gt | topping | hull | 6 | 33% | 391 | 540 | 391 |
| gt | topping | voxel | 6 | 33% | 312 | 447 | 312 |
| gt_top | leaves | foodscan | 5 | 0% | 1448 | 1932 | 1448 |
| gt_top | leaves | hull | 5 | 0% | 2974 | 6797 | 2974 |
| gt_top | leaves | voxel | 5 | 0% | 1486 | 2073 | 1486 |
| gt_top | piece | foodscan | 11 | 0% | 8 | 10 | 8 |
| gt_top | piece | hull | 11 | 0% | 82 | 97 | 82 |
| gt_top | piece | voxel | 11 | 0% | 8 | 9 | 8 |
| gt_top | pile | foodscan | 96 | 0% | 36 | 52 | 14 |
| gt_top | pile | hull | 96 | 7% | 47 | 64 | 41 |
| gt_top | pile | voxel | 96 | 0% | 38 | 56 | 18 |
| gt_top | single | foodscan | 59 | 0% | 34 | 87 | 26 |
| gt_top | single | hull | 59 | 2% | 85 | 167 | 85 |
| gt_top | single | voxel | 59 | 0% | 35 | 91 | 28 |
| gt_top | topping | foodscan | 6 | 0% | 261 | 276 | 261 |
| gt_top | topping | hull | 6 | 0% | 332 | 516 | 332 |
| gt_top | topping | voxel | 6 | 0% | 327 | 337 | 327 |
| lift | leaves | foodscan | 5 | 0% | 1299 | 1357 | 1299 |
| lift | leaves | hull | 5 | 0% | 2047 | 5528 | 2047 |
| lift | leaves | voxel | 5 | 0% | 1352 | 1530 | 1352 |
| lift | piece | foodscan | 11 | 0% | 54 | 46 | -54 |
| lift | piece | hull | 11 | 0% | 40 | 48 | 21 |
| lift | piece | voxel | 11 | 0% | 50 | 41 | -50 |
| lift | pile | foodscan | 96 | 23% | 51 | 54 | -46 |
| lift | pile | hull | 96 | 23% | 41 | 52 | -14 |
| lift | pile | voxel | 96 | 23% | 41 | 50 | -32 |
| lift | single | foodscan | 59 | 19% | 50 | 82 | -40 |
| lift | single | hull | 59 | 19% | 59 | 115 | 36 |
| lift | single | voxel | 59 | 19% | 42 | 83 | -27 |
| lift | topping | foodscan | 6 | 33% | 269 | 257 | 269 |
| lift | topping | hull | 6 | 33% | 421 | 398 | 421 |
| lift | topping | voxel | 6 | 33% | 309 | 316 | 309 |

**by container**

| source | container | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | bowl | foodscan | 14 | 14% | 76 | 389 | -1 |
| gt | bowl | hull | 14 | 14% | 63 | 542 | 46 |
| gt | bowl | voxel | 14 | 14% | 57 | 399 | 2 |
| gt | plate | foodscan | 163 | 13% | 52 | 89 | -41 |
| gt | plate | hull | 163 | 13% | 40 | 230 | 3 |
| gt | plate | voxel | 163 | 13% | 44 | 91 | -27 |
| gt_top | bowl | foodscan | 14 | 0% | 47 | 400 | 47 |
| gt_top | bowl | hull | 14 | 0% | 113 | 723 | 113 |
| gt_top | bowl | voxel | 14 | 0% | 60 | 433 | 60 |
| gt_top | plate | foodscan | 163 | 0% | 34 | 97 | 17 |
| gt_top | plate | hull | 163 | 5% | 60 | 280 | 57 |
| gt_top | plate | voxel | 163 | 0% | 34 | 105 | 20 |
| lift | bowl | foodscan | 14 | 14% | 82 | 397 | -4 |
| lift | bowl | hull | 14 | 14% | 78 | 673 | 47 |
| lift | bowl | voxel | 14 | 14% | 71 | 422 | -1 |
| lift | plate | foodscan | 163 | 20% | 53 | 88 | -43 |
| lift | plate | hull | 163 | 20% | 46 | 239 | 4 |
| lift | plate | voxel | 163 | 20% | 43 | 92 | -28 |

**by asset_source**

| source | asset_source | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | metafood3d | foodscan | 109 | 14% | 50 | 56 | -44 |
| gt | metafood3d | hull | 109 | 14% | 39 | 63 | 3 |
| gt | metafood3d | voxel | 109 | 14% | 42 | 52 | -31 |
| gt | objaverse | foodscan | 9 | 0% | 59 | 213 | 24 |
| gt | objaverse | hull | 9 | 11% | 104 | 333 | 104 |
| gt | objaverse | voxel | 9 | 0% | 75 | 232 | 75 |
| gt | primitive | foodscan | 36 | 14% | 56 | 51 | -56 |
| gt | primitive | hull | 36 | 14% | 29 | 37 | -23 |
| gt | primitive | voxel | 36 | 14% | 34 | 42 | -31 |
| gt | procedural_leaves | foodscan | 5 | 0% | 1286 | 1328 | 1286 |
| gt | procedural_leaves | hull | 5 | 0% | 2461 | 5314 | 2461 |
| gt | procedural_leaves | voxel | 5 | 0% | 1338 | 1453 | 1338 |
| gt | procedural_piece | foodscan | 11 | 9% | 49 | 41 | -49 |
| gt | procedural_piece | hull | 11 | 9% | 37 | 42 | 37 |
| gt | procedural_piece | voxel | 11 | 9% | 41 | 37 | -41 |
| gt | procedural_topping | foodscan | 6 | 33% | 265 | 381 | 265 |
| gt | procedural_topping | hull | 6 | 33% | 391 | 540 | 391 |
| gt | procedural_topping | voxel | 6 | 33% | 312 | 447 | 312 |
| gt | ycb | foodscan | 1 | 0% | 20 | 20 | -20 |
| gt | ycb | hull | 1 | 0% | 5 | 5 | -5 |
| gt | ycb | voxel | 1 | 0% | 8 | 8 | -8 |
| gt_top | metafood3d | foodscan | 109 | 0% | 41 | 60 | 23 |
| gt_top | metafood3d | hull | 109 | 1% | 71 | 106 | 71 |
| gt_top | metafood3d | voxel | 109 | 0% | 43 | 65 | 26 |
| gt_top | objaverse | foodscan | 9 | 0% | 125 | 275 | 125 |
| gt_top | objaverse | hull | 9 | 0% | 168 | 345 | 168 |
| gt_top | objaverse | voxel | 9 | 0% | 156 | 290 | 156 |
| gt_top | primitive | foodscan | 36 | 0% | 26 | 28 | -3 |
| gt_top | primitive | hull | 36 | 19% | 23 | 28 | -1 |
| gt_top | primitive | voxel | 36 | 0% | 26 | 30 | -1 |
| gt_top | procedural_leaves | foodscan | 5 | 0% | 1448 | 1932 | 1448 |
| gt_top | procedural_leaves | hull | 5 | 0% | 2974 | 6797 | 2974 |
| gt_top | procedural_leaves | voxel | 5 | 0% | 1486 | 2073 | 1486 |
| gt_top | procedural_piece | foodscan | 11 | 0% | 8 | 10 | 8 |
| gt_top | procedural_piece | hull | 11 | 0% | 82 | 97 | 82 |
| gt_top | procedural_piece | voxel | 11 | 0% | 8 | 9 | 8 |
| gt_top | procedural_topping | foodscan | 6 | 0% | 261 | 276 | 261 |
| gt_top | procedural_topping | hull | 6 | 0% | 332 | 516 | 332 |
| gt_top | procedural_topping | voxel | 6 | 0% | 327 | 337 | 327 |
| gt_top | ycb | foodscan | 1 | 0% | 1 | 1 | 1 |
| gt_top | ycb | hull | 1 | 0% | 8 | 8 | 8 |
| gt_top | ycb | voxel | 1 | 0% | 7 | 7 | 7 |
| lift | metafood3d | foodscan | 109 | 21% | 55 | 55 | -50 |
| lift | metafood3d | hull | 109 | 21% | 46 | 61 | 3 |
| lift | metafood3d | voxel | 109 | 21% | 42 | 51 | -35 |
| lift | objaverse | foodscan | 9 | 0% | 63 | 211 | 25 |
| lift | objaverse | hull | 9 | 0% | 98 | 308 | 98 |
| lift | objaverse | voxel | 9 | 0% | 74 | 236 | 74 |
| lift | primitive | foodscan | 36 | 28% | 44 | 48 | -44 |
| lift | primitive | hull | 36 | 28% | 39 | 52 | -17 |
| lift | primitive | voxel | 36 | 28% | 36 | 45 | -28 |
| lift | procedural_leaves | foodscan | 5 | 0% | 1299 | 1357 | 1299 |
| lift | procedural_leaves | hull | 5 | 0% | 2047 | 5528 | 2047 |
| lift | procedural_leaves | voxel | 5 | 0% | 1352 | 1530 | 1352 |
| lift | procedural_piece | foodscan | 11 | 0% | 54 | 46 | -54 |
| lift | procedural_piece | hull | 11 | 0% | 40 | 48 | 21 |
| lift | procedural_piece | voxel | 11 | 0% | 50 | 41 | -50 |
| lift | procedural_topping | foodscan | 6 | 33% | 269 | 257 | 269 |
| lift | procedural_topping | hull | 6 | 33% | 421 | 398 | 421 |
| lift | procedural_topping | voxel | 6 | 33% | 309 | 316 | 309 |
| lift | ycb | foodscan | 1 | 0% | 20 | 20 | -20 |
| lift | ycb | hull | 1 | 0% | 5 | 5 | -5 |
| lift | ycb | voxel | 1 | 0% | 8 | 8 | -8 |

**by batch**

| source | batch | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | hard_v12 | foodscan | 177 | 13% | 53 | 113 | -40 |
| gt | hard_v12 | hull | 177 | 14% | 40 | 254 | 5 |
| gt | hard_v12 | voxel | 177 | 13% | 46 | 115 | -26 |
| gt_top | hard_v12 | foodscan | 177 | 0% | 34 | 121 | 20 |
| gt_top | hard_v12 | hull | 177 | 5% | 68 | 317 | 63 |
| gt_top | hard_v12 | voxel | 177 | 0% | 35 | 131 | 23 |
| lift | hard_v12 | foodscan | 177 | 20% | 54 | 114 | -42 |
| lift | hard_v12 | hull | 177 | 20% | 46 | 276 | 7 |
| lift | hard_v12 | voxel | 177 | 20% | 47 | 120 | -28 |

**by occluded_fraction**

| source | occluded_fraction | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | 0.3-0.7 | foodscan | 64 | 0% | 38 | 99 | -21 |
| gt | 0.3-0.7 | hull | 64 | 2% | 41 | 384 | 29 |
| gt | 0.3-0.7 | voxel | 64 | 0% | 31 | 109 | -10 |
| gt | <0.3 | foodscan | 15 | 0% | 177 | 428 | 177 |
| gt | <0.3 | hull | 15 | 0% | 243 | 726 | 243 |
| gt | <0.3 | voxel | 15 | 0% | 200 | 452 | 200 |
| gt | >=0.7 | foodscan | 98 | 23% | 60 | 61 | -57 |
| gt | >=0.7 | hull | 98 | 23% | 38 | 51 | -21 |
| gt | >=0.7 | voxel | 98 | 23% | 49 | 52 | -48 |
| gt_top | 0.3-0.7 | foodscan | 64 | 0% | 30 | 143 | 22 |
| gt_top | 0.3-0.7 | hull | 64 | 3% | 57 | 474 | 57 |
| gt_top | 0.3-0.7 | voxel | 64 | 0% | 32 | 154 | 25 |
| gt_top | <0.3 | foodscan | 15 | 0% | 222 | 492 | 222 |
| gt_top | <0.3 | hull | 15 | 0% | 253 | 1007 | 253 |
| gt_top | <0.3 | voxel | 15 | 0% | 245 | 544 | 245 |
| gt_top | >=0.7 | foodscan | 98 | 0% | 31 | 51 | 14 |
| gt_top | >=0.7 | hull | 98 | 6% | 61 | 98 | 57 |
| gt_top | >=0.7 | voxel | 98 | 0% | 34 | 54 | 15 |
| lift | 0.3-0.7 | foodscan | 64 | 6% | 34 | 103 | -19 |
| lift | 0.3-0.7 | hull | 64 | 6% | 46 | 411 | 41 |
| lift | 0.3-0.7 | voxel | 64 | 6% | 30 | 115 | -9 |
| lift | <0.3 | foodscan | 15 | 7% | 188 | 416 | 188 |
| lift | <0.3 | hull | 15 | 7% | 275 | 811 | 275 |
| lift | <0.3 | voxel | 15 | 7% | 222 | 466 | 222 |
| lift | >=0.7 | foodscan | 98 | 31% | 63 | 62 | -63 |
| lift | >=0.7 | hull | 98 | 31% | 41 | 46 | -26 |
| lift | >=0.7 | voxel | 98 | 31% | 50 | 53 | -49 |
| lift | n/a | foodscan | 0 | - | - | - | - |
| lift | n/a | hull | 0 | - | - | - | - |
| lift | n/a | voxel | 0 | - | - | - | - |

**by stacked_scan**

| source | stacked_scan | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | False | foodscan | 115 | 13% | 55 | 85 | -42 |
| gt | False | hull | 115 | 14% | 39 | 243 | 1 |
| gt | False | voxel | 115 | 13% | 49 | 85 | -31 |
| gt | True | foodscan | 62 | 13% | 52 | 164 | -33 |
| gt | True | hull | 62 | 13% | 47 | 275 | 20 |
| gt | True | voxel | 62 | 13% | 40 | 169 | -20 |
| gt_top | False | foodscan | 115 | 0% | 39 | 95 | 16 |
| gt_top | False | hull | 115 | 2% | 57 | 285 | 55 |
| gt_top | False | voxel | 115 | 0% | 42 | 102 | 20 |
| gt_top | True | foodscan | 62 | 0% | 25 | 171 | 24 |
| gt_top | True | hull | 62 | 10% | 80 | 380 | 80 |
| gt_top | True | voxel | 62 | 0% | 26 | 186 | 26 |
| lift | False | foodscan | 115 | 23% | 54 | 89 | -43 |
| lift | False | hull | 115 | 23% | 52 | 273 | 4 |
| lift | False | voxel | 115 | 23% | 49 | 93 | -33 |
| lift | True | foodscan | 62 | 15% | 54 | 157 | -35 |
| lift | True | hull | 62 | 15% | 45 | 281 | 10 |
| lift | True | voxel | 62 | 15% | 42 | 165 | -22 |

**by icp_pose_error**

| source | icp_pose_error | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | <10 mm | foodscan | 146 | 12% | 56 | 120 | -39 |
| gt | <10 mm | hull | 146 | 12% | 41 | 279 | 10 |
| gt | <10 mm | voxel | 146 | 12% | 49 | 123 | -25 |
| gt | <10 mm | foodscan_gtreg | 146 | 12% | 61 | 121 | -45 |
| gt | <10 mm | foodscan_cinit | 146 | 12% | 56 | 120 | -38 |
| gt | >=10 mm (failed) | foodscan | 31 | 19% | 51 | 76 | -50 |
| gt | >=10 mm (failed) | hull | 31 | 19% | 38 | 127 | -18 |
| gt | >=10 mm (failed) | voxel | 31 | 19% | 40 | 73 | -37 |
| gt | >=10 mm (failed) | foodscan_gtreg | 31 | 19% | 58 | 72 | -56 |
| gt | >=10 mm (failed) | foodscan_cinit | 31 | 19% | 51 | 76 | -50 |
| gt_top | <10 mm | foodscan | 146 | 0% | 37 | 129 | 20 |
| gt_top | <10 mm | hull | 146 | 5% | 70 | 354 | 70 |
| gt_top | <10 mm | voxel | 146 | 0% | 39 | 141 | 23 |
| gt_top | <10 mm | foodscan_gtreg | 146 | 0% | 37 | 129 | 20 |
| gt_top | <10 mm | foodscan_cinit | 146 | 0% | 37 | 129 | 20 |
| gt_top | >=10 mm (failed) | foodscan | 31 | 0% | 22 | 84 | 17 |
| gt_top | >=10 mm (failed) | hull | 31 | 0% | 43 | 153 | 43 |
| gt_top | >=10 mm (failed) | voxel | 31 | 0% | 21 | 89 | 19 |
| gt_top | >=10 mm (failed) | foodscan_gtreg | 31 | 0% | 22 | 84 | 17 |
| gt_top | >=10 mm (failed) | foodscan_cinit | 31 | 0% | 22 | 84 | 17 |
| lift | <10 mm | foodscan | 146 | 20% | 54 | 122 | -37 |
| lift | <10 mm | hull | 146 | 20% | 52 | 308 | 13 |
| lift | <10 mm | voxel | 146 | 20% | 48 | 130 | -26 |
| lift | <10 mm | foodscan_gtreg | 146 | 20% | 56 | 123 | -42 |
| lift | <10 mm | foodscan_cinit | 146 | 20% | 54 | 122 | -37 |
| lift | >=10 mm (failed) | foodscan | 31 | 19% | 55 | 79 | -54 |
| lift | >=10 mm (failed) | hull | 31 | 19% | 43 | 124 | -31 |
| lift | >=10 mm (failed) | voxel | 31 | 19% | 42 | 75 | -41 |
| lift | >=10 mm (failed) | foodscan_gtreg | 31 | 19% | 59 | 75 | -59 |
| lift | >=10 mm (failed) | foodscan_cinit | 31 | 19% | 56 | 78 | -54 |

## 3. Against `volume_above_container_ml` (rim-plane definition; items with >= 1 mL above the rim)
| source | method | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|
| gt | foodscan | 109 | 6% | 64 | 272 | 35 |
| gt | table | 109 | 6% | 82 | 382 | 76 |
| gt | hull_table | 109 | 6% | 117 | 584 | 117 |
| gt | (foodscan vs total volume_ml, same items) | 109 | 6% | 40 | 82 | -36 |
| gt_top | foodscan | 109 | 0% | 192 | 547 | 192 |
| gt_top | table | 109 | 0% | 280 | 837 | 280 |
| gt_top | hull_table | 109 | 3% | 281 | 1016 | 281 |
| gt_top | (foodscan vs total volume_ml, same items) | 109 | 0% | 36 | 110 | 32 |

## 4. Error decomposition (paired: same items valid in every listed source)
Ladder gt_top -> gt (+ sensor sparsity / noise / occlusion) -> lift or pred (+ segmentation). `gtsub` = GT labels on the prediction subset.

| ladder | source | method | items | scans | item med APE % | item mean APE % | item med bias % | dish med APE % | dish mean APE % | dish med bias % |
|---|---|---|---|---|---|---|---|---|---|---|
| lift | gt_top | foodscan | 141 | 20 | 32 | 139 | 28 | 30 | 42 | 30 |
| lift | gt | foodscan | 141 | 20 | 50 | 115 | -38 | 30 | 33 | -30 |
| lift | lift | foodscan | 141 | 20 | 54 | 114 | -41 | 31 | 34 | -27 |
| lift | gt_top | hull | 133 | 20 | 72 | 376 | 72 | 79 | 100 | 79 |
| lift | gt | hull | 133 | 20 | 39 | 279 | 11 | 35 | 53 | 32 |
| lift | lift | hull | 133 | 20 | 51 | 289 | 9 | 77 | 172 | 77 |
| lift | gt_top | voxel | 141 | 20 | 34 | 151 | 31 | 32 | 45 | 32 |
| lift | gt | voxel | 141 | 20 | 41 | 118 | -25 | 21 | 27 | -20 |
| lift | lift | voxel | 141 | 20 | 47 | 120 | -27 | 23 | 29 | -17 |

## 5. Dish totals (sum over all instances; items without an estimate contribute 0)
Mass columns use only dishes whose every item has a positive `mass_g` (the first presets left it null on ~40% of items).

| source | method | dishes | vol med APE % | vol mean APE % | vol med bias % | mass dishes | vol med APE % (mass dishes) | mass APE med % (cat density) | mass APE mean % | mass bias med % | mass APE med % (item density) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gt | foodscan | 20 | 30 | 33 | -30 | 20 | 30 | 30 | 37 | -30 | 30 |
| gt | voxel | 20 | 21 | 27 | -20 | 20 | 21 | 22 | 33 | -21 | 20 |
| gt | hull | 20 | 35 | 53 | 32 | 20 | 35 | 27 | 64 | 27 | 28 |
| gt | table | 20 | 24 | 39 | -6 | 20 | 24 | 26 | 48 | -4 | 20 |
| gt_top | foodscan | 20 | 30 | 42 | 30 | 20 | 30 | 43 | 55 | 43 | 32 |
| gt_top | voxel | 20 | 32 | 45 | 32 | 20 | 32 | 45 | 58 | 45 | 34 |
| gt_top | hull | 20 | 79 | 100 | 79 | 20 | 79 | 83 | 118 | 83 | 74 |
| gt_top | table | 20 | 74 | 103 | 74 | 20 | 74 | 83 | 121 | 83 | 75 |
| lift | foodscan | 20 | 31 | 34 | -27 | 20 | 31 | 31 | 39 | -26 | 27 |
| lift | voxel | 20 | 23 | 29 | -17 | 20 | 23 | 22 | 34 | -17 | 22 |
| lift | hull | 20 | 77 | 172 | 77 | 20 | 77 | 66 | 184 | 66 | 52 |
| lift | table | 20 | 37 | 72 | 6 | 20 | 37 | 41 | 82 | 6 | 35 |

## 6. Per-item mass with oracle density
`oracle per-item` density = gt mass / gt volume of the same item, so its mass APE equals the volume APE (shown as a check); `oracle category median` is the non-tautological one. Neither is a deployable density.

| source | method | density | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | foodscan | oracle per-item | 177 | 13% | 53 | 113 | -40 |
| gt | foodscan | oracle category median | 177 | 13% | 55 | 324 | -38 |
| gt | hull | oracle per-item | 177 | 14% | 40 | 254 | 5 |
| gt | hull | oracle category median | 177 | 14% | 41 | 1094 | 6 |
| gt | voxel | oracle per-item | 177 | 13% | 46 | 115 | -26 |
| gt | voxel | oracle category median | 177 | 13% | 43 | 346 | -26 |
| gt_top | foodscan | oracle per-item | 177 | 0% | 34 | 121 | 20 |
| gt_top | foodscan | oracle category median | 177 | 0% | 38 | 391 | 22 |
| gt_top | hull | oracle per-item | 177 | 5% | 68 | 317 | 63 |
| gt_top | hull | oracle category median | 177 | 5% | 70 | 1283 | 70 |
| gt_top | voxel | oracle per-item | 177 | 0% | 35 | 131 | 23 |
| gt_top | voxel | oracle category median | 177 | 0% | 40 | 419 | 27 |
| lift | foodscan | oracle per-item | 177 | 20% | 54 | 114 | -42 |
| lift | foodscan | oracle category median | 177 | 20% | 54 | 349 | -42 |
| lift | hull | oracle per-item | 177 | 20% | 46 | 276 | 7 |
| lift | hull | oracle category median | 177 | 20% | 53 | 1217 | 7 |
| lift | voxel | oracle per-item | 177 | 20% | 47 | 120 | -28 |
| lift | voxel | oracle category median | 177 | 20% | 46 | 384 | -28 |

## 7. ok-flag vs all scans (item APE)
| subset | source | method | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| ok | gt_top | foodscan | 26 | 0% | 15 | 73 | 9 |
| ok | gt_top | hull | 26 | 4% | 56 | 128 | 56 |
| ok | gt_top | voxel | 26 | 0% | 15 | 82 | 10 |
| ok | gt | foodscan | 26 | 0% | 43 | 87 | -38 |
| ok | gt | hull | 26 | 0% | 45 | 105 | 39 |
| ok | gt | voxel | 26 | 0% | 39 | 90 | -19 |
| ok | lift | foodscan | 26 | 0% | 54 | 69 | -34 |
| ok | lift | hull | 26 | 0% | 55 | 95 | 41 |
| ok | lift | voxel | 26 | 0% | 42 | 72 | -16 |
| all | gt_top | foodscan | 177 | 0% | 34 | 121 | 20 |
| all | gt_top | hull | 177 | 5% | 68 | 317 | 63 |
| all | gt_top | voxel | 177 | 0% | 35 | 131 | 23 |
| all | gt | foodscan | 177 | 13% | 53 | 113 | -40 |
| all | gt | hull | 177 | 14% | 40 | 254 | 5 |
| all | gt | voxel | 177 | 13% | 46 | 115 | -26 |
| all | lift | foodscan | 177 | 20% | 54 | 114 | -42 |
| all | lift | hull | 177 | 20% | 46 | 276 | 7 |
| all | lift | voxel | 177 | 20% | 47 | 120 | -28 |
