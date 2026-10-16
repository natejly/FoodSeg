# Synthetic per-item volume evaluation
Tables use **ok-flagged** scans; table plane = **gt**; items with fewer than 20 points are 'missing'. Reproduce: `conda run -n foodseg python -m pipeline.eval.synth_volume` from the repo root.
## Reading
1. Segmentation matters on top of the volume step. On the 26 items / 5 scans that have lift labels, foodscan's median item APE is 43% with GT labels and 54% with lift labels (segmentation adds 11 points); per-dish totals 30% vs 21%. The noise-free GT surface (gt_top, oracle container pose) still gives 15% median / +9% bias on the same items.
2. Note: after the foodscan fix (Support.height_map_volume hole-fills and rasterises the footprint at 0.5 mm) the per-item median APE with GT labels moved from 49% to ~54%: the old estimator over-counted sparse items by accident, which partly cancelled under-seeing; occlusion dominates either way.
3. With GT labels (ok scans, n=26): median APE / median bias foodscan 43% / -38%, voxel 39% / -19%, hull 45% / +39% (mean APE 105%: the hull blows up on non-convex items), flat-table 47% / -14%. 0% of items are missing (<20 points) and 0% non-positive.
4. Where it fails 1, occlusion: items hidden >= 70% in the scan (n=13) come out -54% (median bias) with GT labels, yet the same items from the full GT surface give 21% APE: the integral is fine, the visible-only cloud has no height to integrate (column fill sums observed cells only, no completion).
5. Where it fails 2, stacking and voids: even from noise-free surfaces the container-relative column fill over-counts, +13% median bias in stacked scans (n=8) vs +9% in unstacked ones (n=18); an item sitting on another is charged for the item below, and loose piles / leaves / toppings are charged for their air gaps (gt_top median bias by kind: topping +520%, leaves n/a%, pile +40%, piece +6%).
6. By size (GT labels, foodscan): <10 mL median APE 600% (bias +600%), 10-50 mL 56%, >50 mL 37% (bias -25%): large items are the ones under-seen, small ones carry the 2 mm cell footprint bias (tests: +13% on a 1 cm cylinder, +4% on 3 cm) and the hull's +757% bias.
7. Registration: template ICP lands 3.7 mm median from the true pose but fails (>= 10 mm) on the sparse scans (5 of 26 ok items; run_meal's init is the centroid of all off-plane points, no multi-start, no fitness gate). Replacing ICP by the true pose changes the median APE only 43% -> 49%, so for these scans occlusion, not registration, is what hurts.
8. run_meal's item_volume crops the whole mesh by the item's bounding box, not by label; emulated on points (foodscan_bbox) the median APE goes 43% -> 43% (bias -38% -> -2%) because neighbours, cutlery and other items inside the box are integrated too.
9. Dish totals (what Nutrition5k-style scoring sees), foodscan: volume median APE 30% (bias -30%, n=5 dishes) with GT labels, 30% vs 21% GT vs lift labels on the 5 lift scans; under-seeing and over-counting partly cancel in the sum, so dish error is lower than item error. Mass with oracle density, 5 dishes with complete mass labels (volume APE 30% on them): 30% with the oracle category-median density, 33% with the oracle per-item density. Densities here are oracle, never predicted; mass_g is integer-rounded for small items.
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
| gt | foodscan | 26 | 0% | 0% | 43 | 87 | -38 | 16 |
| gt | voxel | 26 | 0% | 0% | 39 | 90 | -19 | 34 |
| gt | hull | 26 | 0% | 0% | 45 | 105 | 39 | 80 |
| gt | foodscan_cinit | 26 | 0% | 0% | 43 | 87 | -38 | 16 |
| gt | foodscan_gtreg | 26 | 0% | 0% | 49 | 84 | -43 | 5 |
| gt | foodscan_bbox | 26 | 0% | 0% | 43 | 175 | -2 | 136 |
| gt | table | 26 | 0% | 0% | 47 | 119 | -14 | 71 |
| gt | hull_table | 26 | 0% | 0% | 60 | 127 | 44 | 105 |
| gt_top | foodscan | 26 | 0% | 0% | 15 | 73 | 9 | 69 |
| gt_top | voxel | 26 | 0% | 0% | 15 | 82 | 10 | 79 |
| gt_top | hull | 26 | 4% | 0% | 56 | 128 | 56 | 127 |
| gt_top | foodscan_cinit | 26 | 0% | 0% | 15 | 73 | 9 | 69 |
| gt_top | foodscan_gtreg | 26 | 0% | 0% | 15 | 73 | 9 | 69 |
| gt_top | foodscan_bbox | 26 | 0% | 0% | 72 | 196 | 72 | 196 |
| gt_top | table | 26 | 0% | 0% | 89 | 152 | 89 | 152 |
| gt_top | hull_table | 26 | 4% | 0% | 105 | 172 | 105 | 172 |
| lift | foodscan | 26 | 0% | 0% | 54 | 69 | -34 | -2 |
| lift | voxel | 26 | 0% | 0% | 42 | 72 | -16 | 16 |
| lift | hull | 26 | 0% | 0% | 55 | 95 | 41 | 66 |
| lift | foodscan_cinit | 26 | 0% | 0% | 54 | 69 | -34 | -2 |
| lift | foodscan_gtreg | 26 | 0% | 0% | 55 | 68 | -44 | -11 |
| lift | foodscan_bbox | 26 | 0% | 0% | 59 | 151 | 4 | 109 |
| lift | table | 26 | 0% | 0% | 49 | 98 | -12 | 47 |
| lift | hull_table | 26 | 0% | 0% | 74 | 121 | 57 | 95 |

## 2. By item size, kind, container (primary methods)

**by size**

| source | size | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | <10 mL | foodscan | 2 | 0% | 600 | 600 | 600 |
| gt | <10 mL | hull | 2 | 0% | 757 | 757 | 757 |
| gt | <10 mL | voxel | 2 | 0% | 689 | 689 | 689 |
| gt | <10 mL | table | 2 | 0% | 921 | 921 | 921 |
| gt | 10-50 mL | foodscan | 14 | 0% | 56 | 49 | -49 |
| gt | 10-50 mL | hull | 14 | 0% | 49 | 55 | 45 |
| gt | 10-50 mL | voxel | 14 | 0% | 47 | 44 | -35 |
| gt | 10-50 mL | table | 14 | 0% | 47 | 55 | -23 |
| gt | >50 mL | foodscan | 10 | 0% | 37 | 37 | -25 |
| gt | >50 mL | hull | 10 | 0% | 37 | 43 | 21 |
| gt | >50 mL | voxel | 10 | 0% | 36 | 35 | -22 |
| gt | >50 mL | table | 10 | 0% | 40 | 47 | -10 |
| gt_top | <10 mL | foodscan | 2 | 0% | 520 | 520 | 520 |
| gt_top | <10 mL | hull | 2 | 0% | 601 | 601 | 601 |
| gt_top | <10 mL | voxel | 2 | 0% | 610 | 610 | 610 |
| gt_top | <10 mL | table | 2 | 0% | 896 | 896 | 896 |
| gt_top | 10-50 mL | foodscan | 14 | 0% | 14 | 45 | 9 |
| gt_top | 10-50 mL | hull | 14 | 0% | 55 | 100 | 55 |
| gt_top | 10-50 mL | voxel | 14 | 0% | 14 | 49 | 10 |
| gt_top | 10-50 mL | table | 14 | 0% | 108 | 108 | 108 |
| gt_top | >50 mL | foodscan | 10 | 0% | 9 | 23 | 4 |
| gt_top | >50 mL | hull | 10 | 10% | 50 | 66 | 50 |
| gt_top | >50 mL | voxel | 10 | 0% | 9 | 23 | 5 |
| gt_top | >50 mL | table | 10 | 0% | 44 | 65 | 44 |
| lift | <10 mL | foodscan | 2 | 0% | 358 | 358 | 358 |
| lift | <10 mL | hull | 2 | 0% | 472 | 472 | 472 |
| lift | <10 mL | voxel | 2 | 0% | 441 | 441 | 441 |
| lift | <10 mL | table | 2 | 0% | 593 | 593 | 593 |
| lift | 10-50 mL | foodscan | 14 | 0% | 59 | 52 | -58 |
| lift | 10-50 mL | hull | 14 | 0% | 47 | 54 | 23 |
| lift | 10-50 mL | voxel | 14 | 0% | 51 | 46 | -44 |
| lift | 10-50 mL | table | 14 | 0% | 49 | 61 | -41 |
| lift | >50 mL | foodscan | 10 | 0% | 34 | 34 | -12 |
| lift | >50 mL | hull | 10 | 0% | 61 | 78 | 41 |
| lift | >50 mL | voxel | 10 | 0% | 38 | 36 | -7 |
| lift | >50 mL | table | 10 | 0% | 45 | 52 | 18 |

**by kind**

| source | kind | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | piece | foodscan | 5 | 0% | 14 | 24 | -14 |
| gt | piece | hull | 5 | 0% | 39 | 34 | 39 |
| gt | piece | voxel | 5 | 0% | 13 | 22 | -13 |
| gt | pile | foodscan | 6 | 0% | 58 | 51 | -50 |
| gt | pile | hull | 6 | 0% | 42 | 50 | 19 |
| gt | pile | voxel | 6 | 0% | 52 | 48 | -18 |
| gt | single | foodscan | 13 | 0% | 42 | 48 | -42 |
| gt | single | hull | 13 | 0% | 47 | 57 | 7 |
| gt | single | voxel | 13 | 0% | 39 | 43 | -34 |
| gt | topping | foodscan | 2 | 0% | 600 | 600 | 600 |
| gt | topping | hull | 2 | 0% | 757 | 757 | 757 |
| gt | topping | voxel | 2 | 0% | 689 | 689 | 689 |
| gt_top | piece | foodscan | 5 | 0% | 6 | 8 | 6 |
| gt_top | piece | hull | 5 | 0% | 50 | 65 | 50 |
| gt_top | piece | voxel | 5 | 0% | 6 | 8 | 6 |
| gt_top | pile | foodscan | 6 | 0% | 40 | 68 | 40 |
| gt_top | pile | hull | 6 | 0% | 79 | 105 | 79 |
| gt_top | pile | voxel | 6 | 0% | 43 | 74 | 43 |
| gt_top | single | foodscan | 13 | 0% | 21 | 32 | 9 |
| gt_top | single | hull | 13 | 8% | 60 | 87 | 60 |
| gt_top | single | voxel | 13 | 0% | 22 | 34 | 10 |
| gt_top | topping | foodscan | 2 | 0% | 520 | 520 | 520 |
| gt_top | topping | hull | 2 | 0% | 601 | 601 | 601 |
| gt_top | topping | voxel | 2 | 0% | 610 | 610 | 610 |
| lift | piece | foodscan | 5 | 0% | 12 | 21 | -12 |
| lift | piece | hull | 5 | 0% | 40 | 30 | 40 |
| lift | piece | voxel | 5 | 0% | 9 | 18 | -9 |
| lift | pile | foodscan | 6 | 0% | 61 | 54 | -58 |
| lift | pile | hull | 6 | 0% | 48 | 55 | 1 |
| lift | pile | voxel | 6 | 0% | 54 | 48 | -25 |
| lift | single | foodscan | 13 | 0% | 48 | 49 | -48 |
| lift | single | hull | 13 | 0% | 79 | 81 | 43 |
| lift | single | voxel | 13 | 0% | 42 | 48 | -38 |
| lift | topping | foodscan | 2 | 0% | 358 | 358 | 358 |
| lift | topping | hull | 2 | 0% | 472 | 472 | 472 |
| lift | topping | voxel | 2 | 0% | 441 | 441 | 441 |

**by container**

| source | container | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | bowl | foodscan | 4 | 0% | 12 | 96 | -1 |
| gt | bowl | hull | 4 | 0% | 46 | 126 | 46 |
| gt | bowl | voxel | 4 | 0% | 12 | 105 | 2 |
| gt | plate | foodscan | 22 | 0% | 46 | 85 | -42 |
| gt | plate | hull | 22 | 0% | 45 | 101 | 23 |
| gt | plate | voxel | 22 | 0% | 42 | 87 | -32 |
| gt_top | bowl | foodscan | 4 | 0% | 5 | 99 | 5 |
| gt_top | bowl | hull | 4 | 0% | 46 | 135 | 46 |
| gt_top | bowl | voxel | 4 | 0% | 7 | 111 | 7 |
| gt_top | plate | foodscan | 22 | 0% | 20 | 68 | 14 |
| gt_top | plate | hull | 22 | 5% | 63 | 127 | 63 |
| gt_top | plate | voxel | 22 | 0% | 20 | 77 | 14 |
| lift | bowl | foodscan | 4 | 0% | 10 | 83 | -4 |
| lift | bowl | hull | 4 | 0% | 43 | 116 | 43 |
| lift | bowl | voxel | 4 | 0% | 7 | 91 | -1 |
| lift | plate | foodscan | 22 | 0% | 56 | 66 | -51 |
| lift | plate | hull | 22 | 0% | 60 | 91 | 23 |
| lift | plate | voxel | 22 | 0% | 47 | 69 | -38 |

**by asset_source**

| source | asset_source | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | metafood3d | foodscan | 13 | 0% | 44 | 48 | -44 |
| gt | metafood3d | hull | 13 | 0% | 39 | 49 | 7 |
| gt | metafood3d | voxel | 13 | 0% | 39 | 40 | -37 |
| gt | objaverse | foodscan | 3 | 0% | 59 | 48 | -14 |
| gt | objaverse | hull | 3 | 0% | 64 | 68 | 64 |
| gt | objaverse | voxel | 3 | 0% | 54 | 55 | 22 |
| gt | primitive | foodscan | 3 | 0% | 37 | 56 | -36 |
| gt | primitive | hull | 3 | 0% | 85 | 65 | -18 |
| gt | primitive | voxel | 3 | 0% | 39 | 54 | -30 |
| gt | procedural_piece | foodscan | 5 | 0% | 14 | 24 | -14 |
| gt | procedural_piece | hull | 5 | 0% | 39 | 34 | 39 |
| gt | procedural_piece | voxel | 5 | 0% | 13 | 22 | -13 |
| gt | procedural_topping | foodscan | 2 | 0% | 600 | 600 | 600 |
| gt | procedural_topping | hull | 2 | 0% | 757 | 757 | 757 |
| gt | procedural_topping | voxel | 2 | 0% | 689 | 689 | 689 |
| gt_top | metafood3d | foodscan | 13 | 0% | 12 | 33 | 9 |
| gt_top | metafood3d | hull | 13 | 0% | 56 | 89 | 56 |
| gt_top | metafood3d | voxel | 13 | 0% | 12 | 36 | 10 |
| gt_top | objaverse | foodscan | 3 | 0% | 125 | 107 | 125 |
| gt_top | objaverse | hull | 3 | 0% | 166 | 145 | 166 |
| gt_top | objaverse | voxel | 3 | 0% | 138 | 117 | 138 |
| gt_top | primitive | foodscan | 3 | 0% | 27 | 23 | -6 |
| gt_top | primitive | hull | 3 | 33% | 40 | 40 | 30 |
| gt_top | primitive | voxel | 3 | 0% | 26 | 23 | -4 |
| gt_top | procedural_piece | foodscan | 5 | 0% | 6 | 8 | 6 |
| gt_top | procedural_piece | hull | 5 | 0% | 50 | 65 | 50 |
| gt_top | procedural_piece | voxel | 5 | 0% | 6 | 8 | 6 |
| gt_top | procedural_topping | foodscan | 2 | 0% | 520 | 520 | 520 |
| gt_top | procedural_topping | hull | 2 | 0% | 601 | 601 | 601 |
| gt_top | procedural_topping | voxel | 2 | 0% | 610 | 610 | 610 |
| lift | metafood3d | foodscan | 13 | 0% | 57 | 49 | -57 |
| lift | metafood3d | hull | 13 | 0% | 57 | 74 | 25 |
| lift | metafood3d | voxel | 13 | 0% | 42 | 43 | -38 |
| lift | objaverse | foodscan | 3 | 0% | 63 | 48 | -12 |
| lift | objaverse | hull | 3 | 0% | 63 | 76 | 63 |
| lift | objaverse | voxel | 3 | 0% | 57 | 58 | 19 |
| lift | primitive | foodscan | 3 | 0% | 48 | 60 | -48 |
| lift | primitive | hull | 3 | 0% | 79 | 62 | -16 |
| lift | primitive | voxel | 3 | 0% | 43 | 57 | -43 |
| lift | procedural_piece | foodscan | 5 | 0% | 12 | 21 | -12 |
| lift | procedural_piece | hull | 5 | 0% | 40 | 30 | 40 |
| lift | procedural_piece | voxel | 5 | 0% | 9 | 18 | -9 |
| lift | procedural_topping | foodscan | 2 | 0% | 358 | 358 | 358 |
| lift | procedural_topping | hull | 2 | 0% | 472 | 472 | 472 |
| lift | procedural_topping | voxel | 2 | 0% | 441 | 441 | 441 |

**by batch**

| source | batch | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | hard_v12 | foodscan | 26 | 0% | 43 | 87 | -38 |
| gt | hard_v12 | hull | 26 | 0% | 45 | 105 | 39 |
| gt | hard_v12 | voxel | 26 | 0% | 39 | 90 | -19 |
| gt_top | hard_v12 | foodscan | 26 | 0% | 15 | 73 | 9 |
| gt_top | hard_v12 | hull | 26 | 4% | 56 | 128 | 56 |
| gt_top | hard_v12 | voxel | 26 | 0% | 15 | 82 | 10 |
| lift | hard_v12 | foodscan | 26 | 0% | 54 | 69 | -34 |
| lift | hard_v12 | hull | 26 | 0% | 55 | 95 | 41 |
| lift | hard_v12 | voxel | 26 | 0% | 42 | 72 | -16 |

**by occluded_fraction**

| source | occluded_fraction | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | 0.3-0.7 | foodscan | 11 | 0% | 30 | 30 | -10 |
| gt | 0.3-0.7 | hull | 11 | 0% | 43 | 50 | 43 |
| gt | 0.3-0.7 | voxel | 11 | 0% | 34 | 34 | -6 |
| gt | <0.3 | foodscan | 2 | 0% | 600 | 600 | 600 |
| gt | <0.3 | hull | 2 | 0% | 757 | 757 | 757 |
| gt | <0.3 | voxel | 2 | 0% | 689 | 689 | 689 |
| gt | >=0.7 | foodscan | 13 | 0% | 54 | 55 | -54 |
| gt | >=0.7 | hull | 13 | 0% | 39 | 50 | -12 |
| gt | >=0.7 | voxel | 13 | 0% | 44 | 45 | -44 |
| gt_top | 0.3-0.7 | foodscan | 11 | 0% | 7 | 30 | 6 |
| gt_top | 0.3-0.7 | hull | 11 | 0% | 54 | 79 | 54 |
| gt_top | 0.3-0.7 | voxel | 11 | 0% | 9 | 33 | 8 |
| gt_top | <0.3 | foodscan | 2 | 0% | 520 | 520 | 520 |
| gt_top | <0.3 | hull | 2 | 0% | 601 | 601 | 601 |
| gt_top | <0.3 | voxel | 2 | 0% | 610 | 610 | 610 |
| gt_top | >=0.7 | foodscan | 13 | 0% | 21 | 41 | 9 |
| gt_top | >=0.7 | hull | 13 | 8% | 58 | 94 | 58 |
| gt_top | >=0.7 | voxel | 13 | 0% | 22 | 43 | 10 |
| lift | 0.3-0.7 | foodscan | 11 | 0% | 27 | 27 | -4 |
| lift | 0.3-0.7 | hull | 11 | 0% | 43 | 78 | 43 |
| lift | 0.3-0.7 | voxel | 11 | 0% | 28 | 34 | 3 |
| lift | <0.3 | foodscan | 2 | 0% | 358 | 358 | 358 |
| lift | <0.3 | hull | 2 | 0% | 472 | 472 | 472 |
| lift | <0.3 | voxel | 2 | 0% | 441 | 441 | 441 |
| lift | >=0.7 | foodscan | 13 | 0% | 59 | 60 | -59 |
| lift | >=0.7 | hull | 13 | 0% | 52 | 52 | -16 |
| lift | >=0.7 | voxel | 13 | 0% | 51 | 49 | -51 |
| lift | n/a | foodscan | 0 | - | - | - | - |
| lift | n/a | hull | 0 | - | - | - | - |
| lift | n/a | voxel | 0 | - | - | - | - |

**by stacked_scan**

| source | stacked_scan | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | False | foodscan | 18 | 0% | 46 | 50 | -43 |
| gt | False | hull | 18 | 0% | 43 | 53 | 6 |
| gt | False | voxel | 18 | 0% | 42 | 45 | -36 |
| gt | True | foodscan | 8 | 0% | 34 | 170 | -1 |
| gt | True | hull | 8 | 0% | 48 | 221 | 48 |
| gt | True | voxel | 8 | 0% | 32 | 191 | 2 |
| gt_top | False | foodscan | 18 | 0% | 16 | 44 | 9 |
| gt_top | False | hull | 18 | 6% | 56 | 94 | 56 |
| gt_top | False | voxel | 18 | 0% | 17 | 47 | 10 |
| gt_top | True | foodscan | 8 | 0% | 13 | 139 | 13 |
| gt_top | True | hull | 8 | 0% | 62 | 200 | 62 |
| gt_top | True | voxel | 8 | 0% | 13 | 162 | 13 |
| lift | False | foodscan | 18 | 0% | 58 | 52 | -55 |
| lift | False | hull | 18 | 0% | 60 | 72 | 15 |
| lift | False | voxel | 18 | 0% | 47 | 48 | -40 |
| lift | True | foodscan | 8 | 0% | 32 | 107 | -4 |
| lift | True | hull | 8 | 0% | 43 | 147 | 43 |
| lift | True | voxel | 8 | 0% | 30 | 126 | -1 |

**by icp_pose_error**

| source | icp_pose_error | method | n | missing+<=0 | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | <10 mm | foodscan | 21 | 0% | 42 | 96 | -30 |
| gt | <10 mm | hull | 21 | 0% | 50 | 117 | 39 |
| gt | <10 mm | voxel | 21 | 0% | 39 | 103 | -13 |
| gt | <10 mm | foodscan_gtreg | 21 | 0% | 46 | 91 | -36 |
| gt | <10 mm | foodscan_cinit | 21 | 0% | 42 | 96 | -30 |
| gt | >=10 mm (failed) | foodscan | 5 | 0% | 44 | 48 | -44 |
| gt | >=10 mm (failed) | hull | 5 | 0% | 39 | 52 | 39 |
| gt | >=10 mm (failed) | voxel | 5 | 0% | 37 | 35 | -37 |
| gt | >=10 mm (failed) | foodscan_gtreg | 5 | 0% | 53 | 56 | -53 |
| gt | >=10 mm (failed) | foodscan_cinit | 5 | 0% | 44 | 48 | -44 |
| gt_top | <10 mm | foodscan | 21 | 0% | 12 | 81 | 7 |
| gt_top | <10 mm | hull | 21 | 5% | 55 | 131 | 55 |
| gt_top | <10 mm | voxel | 21 | 0% | 12 | 92 | 9 |
| gt_top | <10 mm | foodscan_gtreg | 21 | 0% | 12 | 81 | 7 |
| gt_top | <10 mm | foodscan_cinit | 21 | 0% | 12 | 81 | 7 |
| gt_top | >=10 mm (failed) | foodscan | 5 | 0% | 28 | 39 | 28 |
| gt_top | >=10 mm (failed) | hull | 5 | 0% | 79 | 118 | 79 |
| gt_top | >=10 mm (failed) | voxel | 5 | 0% | 28 | 43 | 28 |
| gt_top | >=10 mm (failed) | foodscan_gtreg | 5 | 0% | 28 | 39 | 28 |
| gt_top | >=10 mm (failed) | foodscan_cinit | 5 | 0% | 28 | 39 | 28 |
| lift | <10 mm | foodscan | 21 | 0% | 53 | 73 | -12 |
| lift | <10 mm | hull | 21 | 0% | 57 | 105 | 40 |
| lift | <10 mm | voxel | 21 | 0% | 43 | 80 | -5 |
| lift | <10 mm | foodscan_gtreg | 21 | 0% | 55 | 71 | -16 |
| lift | <10 mm | foodscan_cinit | 21 | 0% | 53 | 72 | -12 |
| lift | >=10 mm (failed) | foodscan | 5 | 0% | 59 | 53 | -59 |
| lift | >=10 mm (failed) | hull | 5 | 0% | 52 | 55 | 43 |
| lift | >=10 mm (failed) | voxel | 5 | 0% | 38 | 42 | -38 |
| lift | >=10 mm (failed) | foodscan_gtreg | 5 | 0% | 65 | 59 | -65 |
| lift | >=10 mm (failed) | foodscan_cinit | 5 | 0% | 59 | 53 | -59 |

## 3. Against `volume_above_container_ml` (rim-plane definition; items with >= 1 mL above the rim)
| source | method | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|
| gt | foodscan | 18 | 0% | 38 | 279 | 12 |
| gt | table | 18 | 0% | 55 | 385 | 43 |
| gt | hull_table | 18 | 0% | 105 | 549 | 105 |
| gt | (foodscan vs total volume_ml, same items) | 18 | 0% | 42 | 44 | -41 |
| gt_top | foodscan | 18 | 0% | 117 | 446 | 117 |
| gt_top | table | 18 | 0% | 186 | 690 | 186 |
| gt_top | hull_table | 18 | 6% | 211 | 795 | 211 |
| gt_top | (foodscan vs total volume_ml, same items) | 18 | 0% | 16 | 44 | 15 |

## 4. Error decomposition (paired: same items valid in every listed source)
Ladder gt_top -> gt (+ sensor sparsity / noise / occlusion) -> lift or pred (+ segmentation). `gtsub` = GT labels on the prediction subset.

| ladder | source | method | items | scans | item med APE % | item mean APE % | item med bias % | dish med APE % | dish mean APE % | dish med bias % |
|---|---|---|---|---|---|---|---|---|---|---|
| lift | gt_top | foodscan | 26 | 5 | 15 | 73 | 9 | 21 | 22 | 21 |
| lift | gt | foodscan | 26 | 5 | 43 | 87 | -38 | 30 | 25 | -30 |
| lift | lift | foodscan | 26 | 5 | 54 | 69 | -34 | 21 | 21 | -0 |
| lift | gt_top | hull | 25 | 5 | 56 | 128 | 56 | 78 | 70 | 78 |
| lift | gt | hull | 25 | 5 | 47 | 108 | 39 | 29 | 31 | 29 |
| lift | lift | hull | 25 | 5 | 57 | 98 | 42 | 59 | 209 | 59 |
| lift | gt_top | voxel | 26 | 5 | 15 | 82 | 10 | 22 | 23 | 22 |
| lift | gt | voxel | 26 | 5 | 39 | 90 | -19 | 21 | 19 | -21 |
| lift | lift | voxel | 26 | 5 | 42 | 72 | -16 | 15 | 20 | 3 |

## 5. Dish totals (sum over all instances; items without an estimate contribute 0)
Mass columns use only dishes whose every item has a positive `mass_g` (the first presets left it null on ~40% of items).

| source | method | dishes | vol med APE % | vol mean APE % | vol med bias % | mass dishes | vol med APE % (mass dishes) | mass APE med % (cat density) | mass APE mean % | mass bias med % | mass APE med % (item density) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gt | foodscan | 5 | 30 | 25 | -30 | 5 | 30 | 30 | 23 | -30 | 33 |
| gt | voxel | 5 | 21 | 19 | -21 | 5 | 21 | 22 | 18 | -22 | 25 |
| gt | hull | 5 | 29 | 31 | 29 | 5 | 29 | 19 | 30 | 19 | 24 |
| gt | table | 5 | 32 | 41 | -6 | 5 | 32 | 27 | 42 | 0 | 30 |
| gt_top | foodscan | 5 | 21 | 22 | 21 | 5 | 21 | 17 | 21 | 17 | 15 |
| gt_top | voxel | 5 | 22 | 23 | 22 | 5 | 22 | 19 | 22 | 19 | 17 |
| gt_top | hull | 5 | 78 | 70 | 78 | 5 | 78 | 66 | 69 | 66 | 76 |
| gt_top | table | 5 | 72 | 75 | 72 | 5 | 72 | 80 | 78 | 80 | 76 |
| lift | foodscan | 5 | 21 | 21 | -0 | 5 | 21 | 17 | 18 | 5 | 23 |
| lift | voxel | 5 | 15 | 20 | 3 | 5 | 15 | 11 | 16 | 9 | 19 |
| lift | hull | 5 | 59 | 209 | 59 | 5 | 59 | 49 | 210 | 49 | 61 |
| lift | table | 5 | 98 | 140 | 98 | 5 | 98 | 91 | 140 | 91 | 92 |

## 6. Per-item mass with oracle density
`oracle per-item` density = gt mass / gt volume of the same item, so its mass APE equals the volume APE (shown as a check); `oracle category median` is the non-tautological one. Neither is a deployable density.

| source | method | density | n | missing | med APE % | mean APE % | med bias % |
|---|---|---|---|---|---|---|---|
| gt | foodscan | oracle per-item | 26 | 0% | 43 | 87 | -38 |
| gt | foodscan | oracle category median | 26 | 0% | 40 | 78 | -27 |
| gt | hull | oracle per-item | 26 | 0% | 45 | 105 | 39 |
| gt | hull | oracle category median | 26 | 0% | 51 | 101 | 39 |
| gt | voxel | oracle per-item | 26 | 0% | 39 | 90 | -19 |
| gt | voxel | oracle category median | 26 | 0% | 37 | 81 | -11 |
| gt_top | foodscan | oracle per-item | 26 | 0% | 15 | 73 | 9 |
| gt_top | foodscan | oracle category median | 26 | 0% | 25 | 71 | 19 |
| gt_top | hull | oracle per-item | 26 | 4% | 56 | 128 | 56 |
| gt_top | hull | oracle category median | 26 | 4% | 79 | 123 | 79 |
| gt_top | voxel | oracle per-item | 26 | 0% | 15 | 82 | 10 |
| gt_top | voxel | oracle category median | 26 | 0% | 26 | 79 | 19 |
| lift | foodscan | oracle per-item | 26 | 0% | 54 | 69 | -34 |
| lift | foodscan | oracle category median | 26 | 0% | 52 | 64 | -27 |
| lift | hull | oracle per-item | 26 | 0% | 55 | 95 | 41 |
| lift | hull | oracle category median | 26 | 0% | 57 | 97 | 44 |
| lift | voxel | oracle per-item | 26 | 0% | 42 | 72 | -16 |
| lift | voxel | oracle category median | 26 | 0% | 47 | 68 | -6 |

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

## Note: solid vs envelope target

`gt_volume_ml` is the solid mesh volume; a scan measures the envelope (visible top surface closed to the container, voids included). GT envelope = 2 mm column fill of the full GT top surface, results/rq1/envelope_items.csv. Re-scored headline (ok scans, GT labels, median APE / bias vs solid -> vs envelope): see the 'synth_volume headline re-scored' tables in results/rq1/envelope.md and results/rq1/summary.md. Short version: `gt_top` 28% -> 1% (v12 subset 15% -> 1%), foodscan on the sparse fused cloud 54% -> 76% (v12 subset 43% -> 38%), i.e. occlusion, not the target definition, dominates here; per-kind envelope/solid ratios: results/rq1/summary.md.
