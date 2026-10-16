## Reading (10 lines)

1. Of the 0.259 mIoU loss, fragmented items carry 32 % (8 items, IoU 0.45) and merged items 32 % (6 items, IoU 0.28); residual error on already-ok items is 23 % and plain boundary items 9 %; missed and under-covered items are negligible (0 and 1 item). Nothing is invisible to SAM: 89-100 % of frames carry a mask over every error type's items.
2. Fragmentation is a 2D (SAM) problem, not a linking one: 8 of 10 fragment pairs are separate raw masks in > 50 % of co-visible frames (mean 0.74; 3 of the pairs in 100 %), only 1 is a single mask; SAM splits the item consistently, so no cross-view linker can re-merge them from votes alone. Fragmented items have 61 % of frames with a 2D split vs 4 % for ok items, lower SAM score (0.64 vs 0.74).
3. Fragment seams are weak cues: only 6 / 10 pairs touch in 3D (the rest are spatially separate pieces: leaves, grapes), and where they touch the seam differs only moderately from a within-fragment control (normal angle 22 vs 11 deg, colour 10.5 vs 6.1, 48 % within 3 px of a mask edge vs 25 %). Geometry/colour merging of touching ids is therefore unreliable; same-category / appearance similarity is the usable signal for leaf / pile kinds.
4. Merges are 2D too: on the points carrying the merged id, one SAM mask spans both items in > 50 % of co-visible frames for 7 / 7 pairs (mean 0.82); 6 / 7 pairs touch and 3 / 7 are same-category (Broccoli x Broccoli, etc.). It is mask quality (a mask covering two foods), not 3D linking.
5. Hard cases: small items (< 50 ml: 31 of 52 items, 82 % of the loss), stacked/touching items (mIoU 0.68-0.70 vs 0.82 isolated), items near the container rim (4 items, all fragmented), leaves (0.60).
6. Boundary: 44 % of all error voxels lie within 3 mm of the true boundary and 72 % within 6 mm; perfect snapping would lift mIoU 0.741 -> 0.839 (3 mm) / 0.904 (6 mm). This is an oracle upper bound, but it is 3-5x what an oracle merge of whole ids can give (0.774, results/lift_sweep).
7. Fixable in 2D (SAM prompt / threshold / mask quality): the merges (filter or split masks spanning two foods) and the SAM-split fragments (fragmented items score lower and are split in 61 % of frames; raising the threshold or dropping low-quality sub-masks is untested here).
8. Fixable in linking: little (1 of 10 fragment pairs shows cross-view inconsistency; merges are not linking errors); a same-category merge rule for non-touching fragments is the only linker-side lever.
9. Fixable by a point-level boundary step: the ok / boundary items' residual (32 % of the loss, 58-67 % of their errors within 3 mm) and a part of the fragmented/merged items' errors (24-27 % within 3 mm, ~50 % within 6 mm).
10. Single most valuable next change: a point-level boundary/instance refinement over the lift (geometry- and colour-aware, e.g. superpoint graph with lift votes as unary) because its bound (+0.10 at 3 mm, +0.16 at 6 mm) dwarfs any linking change; the 2D merge / split fix is the runner-up but cannot be bounded from this analysis.

# Lift error analysis (52 GT items, 11 hard_v12 scans, co-vote + cos+qual replay)

Pooled instance mIoU 0.741 (loss 1 - mIoU = 0.259). Code `pipeline/eval/lift_errors.py`; per-item table `items.csv`, everything raw `results.json`. Error type rules in the module docstring (ok = IoU >= .75; fragmented = >= 2 lift ids each > 20 % of the item; merged = the id holding > 20 % of the item also holds > 20 % of another item). Touch = >= 5 % of the item's voxels have another item's voxel within 4 mm; stacked = touching and > 50 % of contact voxels sit > 1 mm above the neighbour (or a topping placed on an item); rim = >= 2 % of voxels within 8 mm of the container's top 15 % height band.

## 1. Error types

| error type | n items | share | mean IoU | mIoU loss contributed (sum(1-IoU)/N) | share of total loss |
|---|---|---|---|---|---|
| ok | 33 | 0.63 | 0.908 | 0.059 | 0.23 |
| boundary | 4 | 0.08 | 0.699 | 0.023 | 0.09 |
| fragmented | 8 | 0.15 | 0.454 | 0.084 | 0.32 |
| merged | 6 | 0.12 | 0.279 | 0.083 | 0.32 |
| under-covered | 1 | 0.02 | 0.466 | 0.010 | 0.04 |
| missed | 0 | 0 | - | 0 | 0 |

Secondary flags (not exclusive): fragmented 8, merged 14, both 4; items with IoU >= .75 that are still fragmented 0.

### By kind

| kind | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| leaf | 4 | 0.599 | 1 | 0 | 3 | 0 | 0 | 0 | 0.12 |
| pile | 11 | 0.776 | 8 | 0 | 1 | 1 | 1 | 0 | 0.18 |
| procedural | 10 | 0.770 | 6 | 3 | 1 | 0 | 0 | 0 | 0.17 |
| scan | 26 | 0.746 | 18 | 1 | 2 | 5 | 0 | 0 | 0.49 |
| topping | 1 | 0.472 | 0 | 0 | 1 | 0 | 0 | 0 | 0.04 |

### By size (item volume)

| size | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| <15 ml | 9 | 0.596 | 3 | 1 | 4 | 1 | 0 | 0 | 0.27 |
| 15-50 ml | 22 | 0.660 | 12 | 1 | 3 | 5 | 1 | 0 | 0.55 |
| 50-150 ml | 11 | 0.860 | 9 | 1 | 1 | 0 | 0 | 0 | 0.11 |
| >=150 ml | 10 | 0.917 | 9 | 1 | 0 | 0 | 0 | 0 | 0.06 |

### By touching / stacked / rim

| contact | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| isolated | 20 | 0.815 | 15 | 1 | 2 | 1 | 1 | 0 | 0.27 |
| touching | 23 | 0.698 | 13 | 3 | 2 | 5 | 0 | 0 | 0.51 |
| stacked | 9 | 0.682 | 5 | 0 | 4 | 0 | 0 | 0 | 0.21 |

| container rim | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| no rim | 48 | 0.765 | 33 | 4 | 4 | 6 | 1 | 0 | 0.84 |
| rim nearby | 4 | 0.450 | 0 | 0 | 4 | 0 | 0 | 0 | 0.16 |

| same-category neighbour | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| no same-category neighbour | 43 | 0.745 | 28 | 4 | 7 | 3 | 1 | 0 | 0.81 |
| same category as a neighbour | 9 | 0.718 | 5 | 0 | 1 | 3 | 0 | 0 | 0.19 |

### SAM presence per error type

| error type | frames with a SAM food mask over >= 50 % of the item | frames where SAM splits the item (>= 2 raw masks each > 20 %) | mean SAM score |
|---|---|---|---|
| ok | 1.00 | 0.04 | 0.74 |
| boundary | 0.97 | 0.10 | 0.67 |
| fragmented | 0.89 | 0.61 | 0.64 |
| merged | 0.96 | 0.21 | 0.81 |
| under-covered | 1.00 | 0.08 | 0.62 |

## 2. Fragmented items: where do the fragments come from

10 fragment pairs in 8 items. Per pair, over frames where both fragments (> 20 % of the item each) have >= 20 visible points: SAM `split` = the dominant raw mask under each fragment differs (2D split), `same` = one raw mask covers both, `miss` = one fragment has no food mask.

| sid | item | category | share a/b | co-visible frames | split | same mask | missing | seam pairs |
|---|---|---|---|---|---|---|---|---|
| s000000 | 1 | lettuce | 0.46/0.23 | 21 | 0.81 | 0.10 | 0.10 | 0 |
| s000000 | 7 | Cucumber | 0.36/0.32 | 19 | 1.00 | 0.00 | 0.00 | 91 |
| s000000 | 7 | Cucumber | 0.36/0.30 | 17 | 1.00 | 0.00 | 0.00 | 71 |
| s000000 | 7 | Cucumber | 0.32/0.30 | 16 | 1.00 | 0.00 | 0.00 | 18 |
| s000001 | 2 | Grapes | 0.18/0.40 | 19 | 0.53 | 0.00 | 0.47 | 0 |
| s000007 | 5 | Broccoli | 0.30/0.65 | 12 | 0.17 | 0.83 | 0.00 | 41 |
| s000011 | 2 | mixed greens | 0.51/0.36 | 24 | 1.00 | 0.00 | 0.00 | 0 |
| s000014 | 4 | boiled egg | 0.37/0.39 | 9 | 0.44 | 0.44 | 0.11 | 0 |
| s000016 | 3 | radicchio | 0.52/0.44 | 23 | 0.83 | 0.17 | 0.00 | 35 |
| s000017 | 3 | gravy | 0.41/0.46 | 24 | 0.62 | 0.38 | 0.00 | 172 |

Pairs split as separate masks in > 50 % of co-visible frames: **8 / 10**; one SAM mask over both in > 50 %: **1 / 10**; neither (split in some views only, i.e. cross-view inconsistency / linking, or a fragment lacks a mask): **1 / 10**. Mean over pairs: split 0.74, same 0.19, missing 0.07.

### Seam properties (6 pairs with >= 10 seam voxel pairs within 4 mm), seam vs control (neighbour pairs inside one fragment)

| quantity | seam | control |
|---|---|---|
| raw-mask id differs between the two sides (mean fraction of frames) | 0.260 | 0.077 |
| seam within 3 px of a raw-mask edge | 0.477 | 0.245 |
| height step |dz| (mm) | 1.320 | 0.806 |
| fraction with |dz| > 3 mm | 0.065 | 0.016 |
| normal angle (deg) | 21.710 | 10.602 |
| fraction with normal angle > 30 deg | 0.219 | 0.038 |
| colour difference (mean |dRGB|, 0-255) | 10.520 | 6.107 |
| fraction of seam pairs with colour difference above the control's 90th percentile | 0.29 | 0.10 |

## 3. Merged items: 2D or 3D?

For each (lift id holding > 20 % of both items g and h): over frames where both items have >= 20 visible points and a food mask, the fraction in which ONE raw SAM mask is dominant under both (2D merge). `id pts` restricts to the points that carry the merged lift id.

| sid | g (category) | h (category) | share of g / h | touching | co-visible | one SAM mask | co-vis (id pts) | one mask (id pts) |
|---|---|---|---|---|---|---|---|---|
| s000001 | 1 (Pork_Chop) | 2 (Grapes) | 0.97/0.22 | yes | 14 | 0.21 | 22 | 0.86 |
| s000001 | 4 (Broccoli) | 5 (Broccoli) | 1.00/0.83 | yes | 22 | 1.00 | 22 | 1.00 |
| s000007 | 4 (Broccoli) | 5 (Broccoli) | 0.99/0.23 | yes | 19 | 0.42 | 14 | 0.64 |
| s000013 | 1 (lettuce) | 5 (Cucumber) | 0.99/0.50 | yes | 21 | 0.95 | 19 | 1.00 |
| s000013 | 2 (Broccoli) | 3 (Broccoli) | 0.79/0.69 | no | 19 | 0.68 | 18 | 0.67 |
| s000014 | 4 (boiled egg) | 7 (Cauliflower) | 0.31/0.83 | yes | 12 | 0.50 | 16 | 0.88 |
| s000017 | 2 (corn) | 3 (gravy) | 0.84/0.41 | yes | 24 | 0.17 | 24 | 0.71 |

Merge pairs where one SAM mask spans both items in > 50 % of co-visible frames (2D), whole items: **3 / 7**; restricted to the points carrying the merged id: **7 / 7** (mean 0.82); mean one-mask fraction 0.56; same category 3 / 7; touching 6 / 7.


## 4. Boundary errors

Per-point errors of the matched lift id (false negatives + false positives, 5664 voxels, 2 mm grid; distance to the item's true boundary):

- within 3 mm: **0.44**, within 6 mm: **0.72**, within 12 mm: 0.86 of all error voxels.

| error type | error voxels | within 3 mm | within 6 mm |
|---|---|---|---|
| ok | 2302 | 0.58 | 0.89 |
| boundary | 606 | 0.67 | 0.95 |
| fragmented | 2052 | 0.24 | 0.49 |
| merged | 577 | 0.27 | 0.51 |
| under-covered | 127 | 0.68 | 1.00 |
| missed | 0 | nan | nan |

If every misassigned voxel within the band were snapped to GT (matching and ids unchanged): instance mIoU 0.741 -> **0.839** (3 mm, +0.098), **0.904** (6 mm, +0.163), 0.933 (12 mm, +0.193). Gain by the item's primary error type (6 mm band, mIoU points):

| error type | gain at 3 mm | gain at 6 mm | gain at 12 mm |
|---|---|---|---|
| ok | 0.033 | 0.052 | 0.057 |
| boundary | 0.015 | 0.023 | 0.023 |
| fragmented | 0.031 | 0.056 | 0.072 |
| merged | 0.012 | 0.023 | 0.031 |
| under-covered | 0.006 | 0.010 | 0.010 |
| missed | 0.000 | 0.000 | 0.000 |
