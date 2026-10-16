# Lift error analysis (273 GT items, 40 real-vote scans, co-vote + cos+qual replay)

Pooled instance mIoU 0.631 (loss 1 - mIoU = 0.369). Code `pipeline/eval/lift_errors.py`; per-item table `items.csv`, everything raw `results.json`. Error type rules in the module docstring (ok = IoU >= .75; fragmented = >= 2 lift ids each > 20 % of the item; merged = the id holding > 20 % of the item also holds > 20 % of another item). Touch = >= 5 % of the item's voxels have another item's voxel within 4 mm; stacked = touching and > 50 % of contact voxels sit > 1 mm above the neighbour (or a topping placed on an item); rim = >= 2 % of voxels within 8 mm of the container's top 15 % height band.

## 1. Error types

| error type | n items | share | mean IoU | mIoU loss contributed (sum(1-IoU)/N) | share of total loss |
|---|---|---|---|---|---|
| ok | 131 | 0.48 | 0.895 | 0.050 | 0.14 |
| boundary | 30 | 0.11 | 0.658 | 0.038 | 0.10 |
| fragmented | 50 | 0.18 | 0.374 | 0.115 | 0.31 |
| merged | 46 | 0.17 | 0.266 | 0.124 | 0.33 |
| under-covered | 13 | 0.05 | 0.323 | 0.032 | 0.09 |
| missed | 3 | 0.01 | 0.001 | 0.011 | 0.03 |

### Error types per batch

| batch | scans | items | inst mIoU | ok n (loss share) | boundary n (loss share) | fragmented n (loss share) | merged n (loss share) | under-covered n (loss share) | missed n (loss share) |
|---|---|---|---|---|---|---|---|---|---|
| v12 | 20 | 146 | 0.657 | 79 (0.15) | 13 (0.09) | 24 (0.29) | 22 (0.34) | 5 (0.08) | 3 (0.06) |
| v13 | 20 | 127 | 0.600 | 52 (0.12) | 17 (0.12) | 26 (0.34) | 24 (0.33) | 8 (0.10) | 0 (0.00) |

Secondary flags (not exclusive): fragmented 50, merged 96, both 36; items with IoU >= .75 that are still fragmented 0.

### By kind

| kind | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| leaf | 10 | 0.545 | 2 | 1 | 3 | 3 | 1 | 0 | 0.05 |
| pile | 145 | 0.604 | 67 | 13 | 31 | 27 | 6 | 1 | 0.57 |
| procedural | 38 | 0.677 | 19 | 7 | 2 | 7 | 3 | 0 | 0.12 |
| scan | 73 | 0.684 | 41 | 9 | 9 | 9 | 3 | 2 | 0.23 |
| topping | 7 | 0.503 | 2 | 0 | 5 | 0 | 0 | 0 | 0.03 |

### By size (item volume)

| size | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| <15 ml | 97 | 0.492 | 28 | 9 | 30 | 21 | 7 | 2 | 0.49 |
| 15-50 ml | 107 | 0.683 | 59 | 14 | 12 | 18 | 3 | 1 | 0.34 |
| 50-150 ml | 40 | 0.690 | 23 | 4 | 7 | 5 | 1 | 0 | 0.12 |
| >=150 ml | 29 | 0.819 | 21 | 3 | 1 | 2 | 2 | 0 | 0.05 |

### By touching / stacked / rim

| contact | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| isolated | 55 | 0.740 | 38 | 2 | 6 | 2 | 5 | 2 | 0.14 |
| touching | 174 | 0.587 | 72 | 24 | 28 | 41 | 8 | 1 | 0.71 |
| stacked | 44 | 0.666 | 21 | 4 | 16 | 3 | 0 | 0 | 0.15 |

| container rim | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| no rim | 264 | 0.635 | 130 | 30 | 45 | 43 | 13 | 3 | 0.96 |
| rim nearby | 9 | 0.499 | 1 | 0 | 5 | 3 | 0 | 0 | 0.04 |

| same-category neighbour | n items | mIoU | ok | boundary | fragmented | merged | under-covered | missed | share of mIoU loss |
|---|---|---|---|---|---|---|---|---|---|
| no same-category neighbour | 145 | 0.668 | 76 | 18 | 20 | 21 | 8 | 2 | 0.48 |
| same category as a neighbour | 128 | 0.589 | 55 | 12 | 30 | 25 | 5 | 1 | 0.52 |

### SAM presence per error type

| error type | frames with a SAM food mask over >= 50 % of the item | frames where SAM splits the item (>= 2 raw masks each > 20 %) | mean SAM score |
|---|---|---|---|
| ok | 0.99 | 0.06 | 0.74 |
| boundary | 0.96 | 0.19 | 0.69 |
| fragmented | 0.97 | 0.47 | 0.68 |
| merged | 0.96 | 0.20 | 0.69 |
| under-covered | 0.74 | 0.22 | 0.69 |
| missed | 0.02 | 0.00 | 0.33 |

## 2. Fragmented items: where do the fragments come from

58 fragment pairs in 50 items. Per pair, over frames where both fragments (> 20 % of the item each) have >= 20 visible points: SAM `split` = the dominant raw mask under each fragment differs (2D split), `same` = one raw mask covers both, `miss` = one fragment has no food mask.

| sid | item | category | share a/b | co-visible frames | split | same mask | missing | seam pairs |
|---|---|---|---|---|---|---|---|---|
| v12/s000000 | 1 | lettuce | 0.46/0.23 | 21 | 0.81 | 0.10 | 0.10 | 0 |
| v12/s000000 | 7 | Cucumber | 0.36/0.32 | 19 | 1.00 | 0.00 | 0.00 | 91 |
| v12/s000000 | 7 | Cucumber | 0.36/0.30 | 17 | 1.00 | 0.00 | 0.00 | 71 |
| v12/s000000 | 7 | Cucumber | 0.32/0.30 | 16 | 1.00 | 0.00 | 0.00 | 18 |
| v12/s000001 | 2 | Grapes | 0.18/0.40 | 19 | 0.53 | 0.00 | 0.47 | 0 |
| v12/s000003 | 3 | gravy | 0.32/0.35 | 20 | 0.45 | 0.55 | 0.00 | 44 |
| v12/s000003 | 9 | squash | 0.28/0.42 | 13 | 0.69 | 0.31 | 0.00 | 21 |
| v12/s000003 | 10 | squash | 0.36/0.31 | 7 | 0.57 | 0.43 | 0.00 | 0 |
| v12/s000004 | 10 | strawberry | 0.63/0.34 | 20 | 0.45 | 0.55 | 0.00 | 102 |
| v12/s000005 | 2 | Green_beans | 0.20/0.23 | 12 | 0.67 | 0.08 | 0.25 | 9 |
| v12/s000006 | 4 | strawberry | 0.78/0.22 | 19 | 0.63 | 0.37 | 0.00 | 40 |
| v12/s000006 | 6 | strawberry | 0.49/0.43 | 18 | 0.50 | 0.50 | 0.00 | 111 |
| v12/s000007 | 5 | Broccoli | 0.30/0.65 | 12 | 0.17 | 0.83 | 0.00 | 41 |
| v12/s000008 | 8 | Cucumber | 0.26/0.24 | 24 | 0.67 | 0.17 | 0.17 | 11 |
| v12/s000008 | 8 | Cucumber | 0.26/0.30 | 24 | 0.71 | 0.29 | 0.00 | 26 |
| v12/s000008 | 8 | Cucumber | 0.24/0.30 | 24 | 0.67 | 0.17 | 0.17 | 1 |
| v12/s000008 | 17 | Cucumber | 0.40/0.47 | 18 | 0.44 | 0.50 | 0.06 | 42 |
| v12/s000008 | 18 | Cucumber | 0.36/0.48 | 11 | 0.82 | 0.18 | 0.00 | 23 |
| v12/s000008 | 24 | Cucumber | 0.18/0.66 | 22 | 0.36 | 0.45 | 0.18 | 81 |
| v12/s000008 | 28 | cherry tomato | 0.59/0.41 | 12 | 0.00 | 1.00 | 0.00 | 54 |
| v12/s000010 | 7 | olive | 0.38/0.31 | 16 | 0.31 | 0.69 | 0.00 | 59 |
| v12/s000010 | 7 | olive | 0.38/0.29 | 11 | 0.91 | 0.09 | 0.00 | 0 |
| v12/s000010 | 7 | olive | 0.31/0.29 | 10 | 0.80 | 0.20 | 0.00 | 0 |
| v12/s000010 | 8 | olive | 0.35/0.63 | 13 | 0.69 | 0.31 | 0.00 | 48 |
| v12/s000011 | 2 | mixed greens | 0.51/0.36 | 24 | 1.00 | 0.00 | 0.00 | 0 |
| v12/s000014 | 4 | boiled egg | 0.37/0.39 | 9 | 0.44 | 0.44 | 0.11 | 0 |
| v12/s000015 | 8 | Sausages | 0.58/0.36 | 24 | 0.92 | 0.08 | 0.00 | 55 |
| v12/s000016 | 3 | radicchio | 0.52/0.44 | 23 | 0.83 | 0.17 | 0.00 | 35 |
| v12/s000017 | 3 | gravy | 0.41/0.46 | 24 | 0.62 | 0.38 | 0.00 | 174 |
| v12/s000019 | 19 | Cucumber | 0.36/0.23 | 15 | 0.87 | 0.13 | 0.00 | 1 |
| v13/s000000 | 3 | Broccoli | 0.19/0.68 | 18 | 0.56 | 0.44 | 0.00 | 93 |
| v13/s000004 | 3 | Fried mushrooms | 0.38/0.61 | 19 | 0.21 | 0.79 | 0.00 | 147 |
| v13/s000004 | 4 | Fried mushrooms | 0.67/0.33 | 9 | 0.56 | 0.44 | 0.00 | 5 |
| v13/s000004 | 5 | Fried mushrooms | 0.60/0.39 | 19 | 0.05 | 0.95 | 0.00 | 461 |
| v13/s000004 | 7 | Fried mushrooms | 0.40/0.32 | 1 | 1.00 | 0.00 | 0.00 | 5 |
| v13/s000004 | 9 | Fried mushrooms | 0.48/0.38 | 22 | 0.23 | 0.77 | 0.00 | 171 |
| v13/s000004 | 10 | Fried mushrooms | 0.49/0.43 | 24 | 0.00 | 0.92 | 0.08 | 398 |
| v13/s000004 | 14 | Chicken_Nugget | 0.21/0.46 | 24 | 0.04 | 0.96 | 0.00 | 473 |
| v13/s000004 | 17 | Chicken_Nugget | 0.24/0.63 | 24 | 0.21 | 0.79 | 0.00 | 574 |
| v13/s000005 | 3 | brussels sprout | 0.64/0.31 | 15 | 0.60 | 0.40 | 0.00 | 69 |
| v13/s000006 | 1 | Broccoli | 0.77/0.20 | 23 | 0.61 | 0.39 | 0.00 | 19 |
| v13/s000006 | 8 | strawberry | 0.27/0.73 | 6 | 1.00 | 0.00 | 0.00 | 0 |
| v13/s000009 | 4 | Sushi | 0.43/0.56 | 24 | 0.29 | 0.71 | 0.00 | 240 |
| v13/s000010 | 6 | Sushi | 0.52/0.47 | 24 | 0.46 | 0.54 | 0.00 | 682 |
| v13/s000012 | 2 | Grapes | 0.51/0.25 | 5 | 0.00 | 0.00 | 1.00 | 0 |
| v13/s000013 | 5 | Chicken_tender | 0.26/0.35 | 14 | 0.29 | 0.71 | 0.00 | 53 |
| v13/s000015 | 1 | hummus | 0.42/0.19 | 21 | 0.67 | 0.14 | 0.19 | 101 |
| v13/s000015 | 4 | Cucumber | 0.23/0.49 | 22 | 0.64 | 0.36 | 0.00 | 98 |
| v13/s000015 | 6 | Cucumber | 0.33/0.31 | 21 | 0.67 | 0.33 | 0.00 | 45 |
| v13/s000015 | 6 | Cucumber | 0.33/0.33 | 16 | 0.62 | 0.31 | 0.06 | 2 |
| v13/s000015 | 6 | Cucumber | 0.31/0.33 | 18 | 0.78 | 0.17 | 0.06 | 0 |
| v13/s000016 | 2 | Blueberries | 0.56/0.24 | 4 | 1.00 | 0.00 | 0.00 | 8 |
| v13/s000017 | 3 | Tortilla_Chips | 0.53/0.43 | 12 | 0.33 | 0.67 | 0.00 | 294 |
| v13/s000017 | 6 | cherry tomato | 0.32/0.34 | 9 | 1.00 | 0.00 | 0.00 | 0 |
| v13/s000017 | 8 | cherry tomato | 0.32/0.33 | 6 | 0.00 | 0.17 | 0.83 | 16 |
| v13/s000018 | 3 | salsa | 0.53/0.36 | 22 | 0.77 | 0.23 | 0.00 | 48 |
| v13/s000018 | 5 | tomato sauce | 0.40/0.60 | 22 | 0.82 | 0.18 | 0.00 | 63 |
| v13/s000019 | 3 | sour cream | 0.49/0.43 | 20 | 0.45 | 0.55 | 0.00 | 66 |

Pairs split as separate masks in > 50 % of co-visible frames: **36 / 58**; one SAM mask over both in > 50 %: **16 / 58**; neither (split in some views only, i.e. cross-view inconsistency / linking, or a fragment lacks a mask): **6 / 58**. Mean over pairs: split 0.58, same 0.36, missing 0.06.

### Seam properties (40 pairs with >= 10 seam voxel pairs within 4 mm), seam vs control (neighbour pairs inside one fragment)

| quantity | seam | control |
|---|---|---|
| raw-mask id differs between the two sides (mean fraction of frames) | 0.190 | 0.069 |
| seam within 3 px of a raw-mask edge | 0.391 | 0.230 |
| height step |dz| (mm) | 1.214 | 1.063 |
| fraction with |dz| > 3 mm | 0.052 | 0.042 |
| normal angle (deg) | 26.913 | 18.602 |
| fraction with normal angle > 30 deg | 0.331 | 0.183 |
| colour difference (mean |dRGB|, 0-255) | 11.162 | 7.953 |
| fraction of seam pairs with colour difference above the control's 90th percentile | 0.21 | 0.10 |

## 3. Merged items: 2D or 3D?

For each (lift id holding > 20 % of both items g and h): over frames where both items have >= 20 visible points and a food mask, the fraction in which ONE raw SAM mask is dominant under both (2D merge). `id pts` restricts to the points that carry the merged lift id.

| sid | g (category) | h (category) | share of g / h | touching | co-visible | one SAM mask | co-vis (id pts) | one mask (id pts) |
|---|---|---|---|---|---|---|---|---|
| v12/s000001 | 1 (Pork_Chop) | 2 (Grapes) | 0.97/0.22 | yes | 14 | 0.21 | 22 | 0.86 |
| v12/s000001 | 4 (Broccoli) | 5 (Broccoli) | 1.00/0.83 | yes | 22 | 1.00 | 22 | 1.00 |
| v12/s000003 | 3 (gravy) | 5 (cookies) | 0.34/0.95 | yes | 24 | 0.04 | 20 | 0.30 |
| v12/s000003 | 9 (squash) | 12 (squash) | 0.24/0.95 | yes | 15 | 0.27 | 14 | 0.64 |
| v12/s000003 | 10 (squash) | 11 (squash) | 0.35/0.87 | yes | 19 | 0.68 | 11 | 1.00 |
| v12/s000004 | 2 (Kiwi) | 10 (strawberry) | 0.91/0.62 | yes | 24 | 0.25 | 24 | 0.54 |
| v12/s000005 | 1 (Mashed_Potato) | 2 (Green_beans) | 0.92/0.28 | yes | 11 | 0.00 | 14 | 0.64 |
| v12/s000006 | 4 (strawberry) | 6 (strawberry) | 0.23/0.40 | yes | 21 | 0.00 | 19 | 0.74 |
| v12/s000007 | 4 (Broccoli) | 5 (Broccoli) | 0.99/0.23 | yes | 19 | 0.42 | 14 | 0.64 |
| v12/s000008 | 17 (Cucumber) | 24 (Cucumber) | 0.32/0.65 | yes | 23 | 0.52 | 17 | 0.88 |
| v12/s000008 | 19 (Cucumber) | 22 (Cucumber) | 0.87/0.97 | yes | 24 | 1.00 | 24 | 1.00 |
| v12/s000008 | 24 (Cucumber) | 25 (Cucumber) | 0.20/0.97 | yes | 23 | 0.26 | 21 | 0.57 |
| v12/s000008 | 26 (cherry tomato) | 27 (cherry tomato) | 0.92/0.90 | no | 20 | 0.80 | 20 | 0.80 |
| v12/s000008 | 26 (cherry tomato) | 28 (cherry tomato) | 0.92/0.61 | no | 11 | 0.73 | 12 | 0.67 |
| v12/s000008 | 26 (cherry tomato) | 29 (cherry tomato) | 0.92/0.85 | no | 20 | 0.80 | 20 | 0.80 |
| v12/s000008 | 26 (cherry tomato) | 30 (cherry tomato) | 0.92/0.90 | yes | 17 | 0.82 | 17 | 0.82 |
| v12/s000008 | 26 (cherry tomato) | 31 (cherry tomato) | 0.92/1.00 | no | 15 | 0.80 | 15 | 0.80 |
| v12/s000008 | 27 (cherry tomato) | 28 (cherry tomato) | 0.90/0.61 | no | 14 | 0.71 | 15 | 0.67 |
| v12/s000008 | 27 (cherry tomato) | 29 (cherry tomato) | 0.90/0.85 | yes | 24 | 0.79 | 24 | 0.79 |
| v12/s000008 | 27 (cherry tomato) | 30 (cherry tomato) | 0.90/0.90 | no | 21 | 0.81 | 21 | 0.81 |
| v12/s000008 | 27 (cherry tomato) | 31 (cherry tomato) | 0.90/1.00 | yes | 19 | 0.79 | 19 | 0.79 |
| v12/s000008 | 28 (cherry tomato) | 29 (cherry tomato) | 0.61/0.85 | yes | 14 | 0.86 | 15 | 0.80 |
| v12/s000008 | 28 (cherry tomato) | 30 (cherry tomato) | 0.61/0.90 | no | 11 | 0.82 | 12 | 0.75 |
| v12/s000008 | 28 (cherry tomato) | 31 (cherry tomato) | 0.61/1.00 | no | 12 | 0.83 | 13 | 0.77 |
| v12/s000008 | 29 (cherry tomato) | 30 (cherry tomato) | 0.85/0.90 | no | 21 | 0.86 | 21 | 0.86 |
| v12/s000008 | 29 (cherry tomato) | 31 (cherry tomato) | 0.85/1.00 | yes | 19 | 0.84 | 19 | 0.84 |
| v12/s000008 | 30 (cherry tomato) | 31 (cherry tomato) | 0.90/1.00 | yes | 16 | 0.81 | 16 | 0.81 |
| v12/s000010 | 1 (Yeast_bread) | 4 (olive) | 0.68/0.59 | yes | 23 | 0.61 | 15 | 0.73 |
| v12/s000010 | 1 (Yeast_bread) | 5 (olive) | 0.68/0.69 | yes | 24 | 0.58 | 23 | 0.70 |
| v12/s000010 | 1 (Yeast_bread) | 6 (olive) | 0.68/0.88 | yes | 19 | 0.84 | 18 | 0.83 |
| v12/s000010 | 1 (Yeast_bread) | 7 (olive) | 0.68/0.36 | yes | 24 | 0.33 | 17 | 0.53 |
| v12/s000010 | 1 (Yeast_bread) | 8 (olive) | 0.68/0.33 | yes | 24 | 0.38 | 13 | 0.54 |
| v12/s000010 | 4 (olive) | 5 (olive) | 0.59/0.69 | no | 23 | 0.30 | 14 | 0.43 |
| v12/s000010 | 4 (olive) | 6 (olive) | 0.59/0.88 | yes | 19 | 0.42 | 12 | 0.50 |
| v12/s000010 | 4 (olive) | 7 (olive) | 0.59/0.36 | yes | 23 | 0.09 | 14 | 0.21 |
| v12/s000010 | 4 (olive) | 8 (olive) | 0.59/0.33 | yes | 23 | 0.57 | 12 | 0.50 |
| v12/s000010 | 5 (olive) | 6 (olive) | 0.69/0.88 | no | 19 | 0.53 | 17 | 0.59 |
| v12/s000010 | 5 (olive) | 7 (olive) | 0.69/0.36 | yes | 24 | 0.38 | 16 | 0.56 |
| v12/s000010 | 5 (olive) | 8 (olive) | 0.69/0.33 | no | 24 | 0.21 | 12 | 0.08 |
| v12/s000010 | 6 (olive) | 7 (olive) | 0.88/0.36 | yes | 19 | 0.47 | 14 | 0.43 |
| v12/s000010 | 6 (olive) | 8 (olive) | 0.88/0.33 | no | 19 | 0.37 | 13 | 0.46 |
| v12/s000010 | 7 (olive) | 8 (olive) | 0.36/0.33 | no | 24 | 0.17 | 12 | 0.00 |
| v12/s000013 | 1 (lettuce) | 5 (Cucumber) | 0.99/0.50 | yes | 21 | 0.95 | 19 | 1.00 |
| v12/s000013 | 2 (Broccoli) | 3 (Broccoli) | 0.79/0.69 | no | 19 | 0.68 | 18 | 0.67 |
| v12/s000014 | 4 (boiled egg) | 7 (Cauliflower) | 0.31/0.83 | yes | 12 | 0.50 | 16 | 0.88 |
| v12/s000017 | 2 (corn) | 3 (gravy) | 0.84/0.41 | yes | 24 | 0.17 | 24 | 0.71 |
| v12/s000019 | 3 (mixed greens) | 17 (Falafel) | 0.96/1.00 | yes | 15 | 0.60 | 15 | 0.60 |
| v12/s000019 | 9 (Falafel) | 10 (Falafel) | 0.71/0.92 | yes | 24 | 0.88 | 24 | 0.88 |
| v13/s000004 | 1 (oatmeal) | 2 (brown rice) | 0.96/0.83 | no | 21 | 0.57 | 21 | 0.57 |
| v13/s000004 | 1 (oatmeal) | 3 (Fried mushrooms) | 0.96/0.40 | yes | 24 | 0.46 | 19 | 0.68 |
| v13/s000004 | 1 (oatmeal) | 4 (Fried mushrooms) | 0.96/0.67 | no | 16 | 0.56 | 10 | 0.70 |
| v13/s000004 | 1 (oatmeal) | 5 (Fried mushrooms) | 0.96/0.60 | no | 24 | 0.54 | 24 | 0.54 |
| v13/s000004 | 1 (oatmeal) | 7 (Fried mushrooms) | 0.96/0.46 | yes | 15 | 0.60 | 6 | 0.33 |
| v13/s000004 | 1 (oatmeal) | 9 (Fried mushrooms) | 0.96/0.49 | no | 24 | 0.42 | 24 | 0.54 |
| v13/s000004 | 1 (oatmeal) | 10 (Fried mushrooms) | 0.96/0.49 | no | 24 | 0.50 | 24 | 0.50 |
| v13/s000004 | 1 (oatmeal) | 13 (Chicken_Nugget) | 0.96/0.67 | no | 24 | 0.54 | 24 | 0.54 |
| v13/s000004 | 1 (oatmeal) | 14 (Chicken_Nugget) | 0.96/0.21 | no | 24 | 0.46 | 24 | 0.46 |
| v13/s000004 | 1 (oatmeal) | 17 (Chicken_Nugget) | 0.96/0.25 | no | 24 | 0.33 | 24 | 0.50 |
| v13/s000004 | 2 (brown rice) | 3 (Fried mushrooms) | 0.83/0.40 | no | 21 | 0.38 | 16 | 0.56 |
| v13/s000004 | 2 (brown rice) | 4 (Fried mushrooms) | 0.83/0.67 | no | 16 | 0.50 | 10 | 0.60 |
| v13/s000004 | 2 (brown rice) | 5 (Fried mushrooms) | 0.83/0.60 | yes | 21 | 0.62 | 21 | 0.62 |
| v13/s000004 | 2 (brown rice) | 7 (Fried mushrooms) | 0.83/0.46 | no | 12 | 0.67 | 6 | 0.67 |
| v13/s000004 | 2 (brown rice) | 9 (Fried mushrooms) | 0.83/0.49 | no | 21 | 0.48 | 21 | 0.62 |
| v13/s000004 | 2 (brown rice) | 10 (Fried mushrooms) | 0.83/0.49 | yes | 21 | 0.62 | 21 | 0.62 |
| v13/s000004 | 2 (brown rice) | 13 (Chicken_Nugget) | 0.83/0.67 | no | 21 | 0.62 | 21 | 0.62 |
| v13/s000004 | 2 (brown rice) | 14 (Chicken_Nugget) | 0.83/0.21 | no | 21 | 0.57 | 21 | 0.57 |
| v13/s000004 | 2 (brown rice) | 17 (Chicken_Nugget) | 0.83/0.25 | no | 21 | 0.43 | 21 | 0.52 |
| v13/s000004 | 3 (Fried mushrooms) | 4 (Fried mushrooms) | 0.40/0.67 | yes | 16 | 0.69 | 5 | 0.40 |
| v13/s000004 | 3 (Fried mushrooms) | 5 (Fried mushrooms) | 0.40/0.60 | no | 24 | 0.46 | 19 | 0.53 |
| v13/s000004 | 3 (Fried mushrooms) | 7 (Fried mushrooms) | 0.40/0.46 | no | 15 | 0.60 | 6 | 0.50 |
| v13/s000004 | 3 (Fried mushrooms) | 9 (Fried mushrooms) | 0.40/0.49 | no | 24 | 0.29 | 19 | 0.47 |
| v13/s000004 | 3 (Fried mushrooms) | 10 (Fried mushrooms) | 0.40/0.49 | no | 24 | 0.50 | 19 | 0.63 |
| v13/s000004 | 3 (Fried mushrooms) | 13 (Chicken_Nugget) | 0.40/0.67 | no | 24 | 0.50 | 19 | 0.58 |
| v13/s000004 | 3 (Fried mushrooms) | 14 (Chicken_Nugget) | 0.40/0.21 | no | 24 | 0.38 | 19 | 0.42 |
| v13/s000004 | 3 (Fried mushrooms) | 17 (Chicken_Nugget) | 0.40/0.25 | no | 24 | 0.25 | 19 | 0.53 |
| v13/s000004 | 3 (Fried mushrooms) | 4 (Fried mushrooms) | 0.58/0.33 | yes | 16 | 0.69 | 13 | 1.00 |
| v13/s000004 | 4 (Fried mushrooms) | 5 (Fried mushrooms) | 0.67/0.60 | no | 16 | 0.44 | 10 | 0.50 |
| v13/s000004 | 4 (Fried mushrooms) | 7 (Fried mushrooms) | 0.67/0.46 | no | 8 | 0.38 | 0 | 0.00 |
| v13/s000004 | 4 (Fried mushrooms) | 9 (Fried mushrooms) | 0.67/0.49 | no | 16 | 0.62 | 10 | 0.80 |
| v13/s000004 | 4 (Fried mushrooms) | 10 (Fried mushrooms) | 0.67/0.49 | no | 16 | 0.38 | 10 | 0.40 |
| v13/s000004 | 4 (Fried mushrooms) | 13 (Chicken_Nugget) | 0.67/0.67 | no | 16 | 0.62 | 10 | 0.80 |
| v13/s000004 | 4 (Fried mushrooms) | 14 (Chicken_Nugget) | 0.67/0.21 | no | 16 | 0.50 | 10 | 0.60 |
| v13/s000004 | 4 (Fried mushrooms) | 17 (Chicken_Nugget) | 0.67/0.25 | no | 16 | 0.50 | 10 | 0.60 |
| v13/s000004 | 5 (Fried mushrooms) | 7 (Fried mushrooms) | 0.60/0.46 | no | 15 | 0.73 | 6 | 0.67 |
| v13/s000004 | 5 (Fried mushrooms) | 9 (Fried mushrooms) | 0.60/0.49 | yes | 24 | 0.42 | 24 | 0.62 |
| v13/s000004 | 5 (Fried mushrooms) | 10 (Fried mushrooms) | 0.60/0.49 | yes | 24 | 0.62 | 24 | 0.62 |
| v13/s000004 | 5 (Fried mushrooms) | 13 (Chicken_Nugget) | 0.60/0.67 | no | 24 | 0.58 | 24 | 0.58 |
| v13/s000004 | 5 (Fried mushrooms) | 14 (Chicken_Nugget) | 0.60/0.21 | no | 24 | 0.50 | 24 | 0.50 |
| v13/s000004 | 5 (Fried mushrooms) | 17 (Chicken_Nugget) | 0.60/0.25 | no | 24 | 0.42 | 24 | 0.54 |
| v13/s000004 | 7 (Fried mushrooms) | 9 (Fried mushrooms) | 0.46/0.49 | yes | 15 | 0.53 | 6 | 0.50 |
| v13/s000004 | 7 (Fried mushrooms) | 10 (Fried mushrooms) | 0.46/0.49 | no | 15 | 0.80 | 6 | 0.67 |
| v13/s000004 | 7 (Fried mushrooms) | 13 (Chicken_Nugget) | 0.46/0.67 | no | 15 | 0.60 | 6 | 0.33 |
| v13/s000004 | 7 (Fried mushrooms) | 14 (Chicken_Nugget) | 0.46/0.21 | no | 15 | 0.60 | 6 | 0.67 |
| v13/s000004 | 7 (Fried mushrooms) | 17 (Chicken_Nugget) | 0.46/0.25 | no | 15 | 0.40 | 6 | 0.33 |
| v13/s000004 | 7 (Fried mushrooms) | 12 (Fried mushrooms) | 0.30/0.90 | yes | 15 | 0.40 | 1 | 1.00 |
| v13/s000004 | 8 (Fried mushrooms) | 11 (Fried mushrooms) | 0.72/0.88 | yes | 24 | 0.79 | 24 | 0.92 |
| v13/s000004 | 9 (Fried mushrooms) | 10 (Fried mushrooms) | 0.49/0.49 | no | 24 | 0.38 | 24 | 0.54 |
| v13/s000004 | 9 (Fried mushrooms) | 13 (Chicken_Nugget) | 0.49/0.67 | no | 24 | 0.42 | 24 | 0.58 |
| v13/s000004 | 9 (Fried mushrooms) | 14 (Chicken_Nugget) | 0.49/0.21 | no | 24 | 0.50 | 24 | 0.58 |
| v13/s000004 | 9 (Fried mushrooms) | 17 (Chicken_Nugget) | 0.49/0.25 | no | 24 | 0.42 | 24 | 0.58 |
| v13/s000004 | 10 (Fried mushrooms) | 13 (Chicken_Nugget) | 0.49/0.67 | no | 24 | 0.58 | 24 | 0.58 |
| v13/s000004 | 10 (Fried mushrooms) | 14 (Chicken_Nugget) | 0.49/0.21 | yes | 24 | 0.50 | 24 | 0.50 |
| v13/s000004 | 10 (Fried mushrooms) | 17 (Chicken_Nugget) | 0.49/0.25 | no | 24 | 0.38 | 24 | 0.54 |
| v13/s000004 | 13 (Chicken_Nugget) | 14 (Chicken_Nugget) | 0.67/0.21 | yes | 24 | 0.42 | 24 | 0.42 |
| v13/s000004 | 13 (Chicken_Nugget) | 17 (Chicken_Nugget) | 0.67/0.25 | yes | 24 | 0.54 | 24 | 0.75 |
| v13/s000004 | 14 (Chicken_Nugget) | 17 (Chicken_Nugget) | 0.21/0.25 | no | 24 | 0.38 | 24 | 0.42 |
| v13/s000005 | 1 (fried rice) | 2 (radicchio) | 0.85/0.94 | yes | 24 | 0.75 | 24 | 0.75 |
| v13/s000006 | 1 (Broccoli) | 8 (strawberry) | 0.76/0.25 | yes | 15 | 0.33 | 7 | 1.00 |
| v13/s000006 | 1 (Broccoli) | 8 (strawberry) | 0.21/0.75 | yes | 15 | 0.33 | 12 | 0.75 |
| v13/s000006 | 4 (strawberry) | 14 (Almonds) | 0.93/0.32 | no | 7 | 0.86 | 6 | 1.00 |
| v13/s000007 | 1 (mixed greens) | 2 (lettuce) | 0.69/0.56 | yes | 14 | 1.00 | 18 | 1.00 |
| v13/s000009 | 1 (Rice) | 2 (Salmon_Grill) | 0.63/0.99 | yes | 19 | 0.32 | 24 | 0.33 |
| v13/s000009 | 1 (Rice) | 4 (Sushi) | 0.63/0.44 | no | 19 | 0.63 | 24 | 0.79 |
| v13/s000009 | 2 (Salmon_Grill) | 4 (Sushi) | 0.99/0.44 | no | 24 | 0.33 | 24 | 0.38 |
| v13/s000010 | 3 (Waffle) | 4 (sushi) | 1.00/0.83 | yes | 24 | 0.67 | 24 | 0.71 |
| v13/s000010 | 3 (Waffle) | 5 (Sushi) | 1.00/0.86 | yes | 22 | 0.50 | 22 | 0.55 |
| v13/s000010 | 3 (Waffle) | 6 (Sushi) | 1.00/0.52 | yes | 24 | 0.50 | 24 | 0.83 |
| v13/s000010 | 3 (Waffle) | 7 (Sushi) | 1.00/0.76 | yes | 23 | 0.43 | 23 | 0.43 |
| v13/s000010 | 4 (sushi) | 5 (Sushi) | 0.83/0.86 | yes | 22 | 0.77 | 22 | 0.77 |
| v13/s000010 | 4 (sushi) | 6 (Sushi) | 0.83/0.52 | no | 24 | 0.67 | 24 | 0.79 |
| v13/s000010 | 4 (sushi) | 7 (Sushi) | 0.83/0.76 | yes | 23 | 0.65 | 23 | 0.65 |
| v13/s000010 | 5 (Sushi) | 6 (Sushi) | 0.86/0.52 | no | 22 | 0.64 | 22 | 0.73 |
| v13/s000010 | 5 (Sushi) | 7 (Sushi) | 0.86/0.76 | yes | 22 | 0.68 | 22 | 0.73 |
| v13/s000010 | 6 (Sushi) | 7 (Sushi) | 0.52/0.76 | yes | 23 | 0.65 | 23 | 0.61 |
| v13/s000013 | 5 (Chicken_tender) | 6 (Chicken_tender) | 0.26/0.89 | yes | 20 | 0.60 | 18 | 0.72 |
| v13/s000015 | 1 (hummus) | 6 (Cucumber) | 0.20/0.32 | yes | 23 | 0.00 | 17 | 0.82 |
| v13/s000015 | 2 (couscous) | 4 (Cucumber) | 0.92/0.46 | yes | 24 | 0.50 | 24 | 0.58 |
| v13/s000015 | 2 (couscous) | 6 (Cucumber) | 0.92/0.30 | yes | 23 | 0.09 | 23 | 0.57 |
| v13/s000015 | 4 (Cucumber) | 6 (Cucumber) | 0.24/0.34 | yes | 23 | 0.52 | 20 | 1.00 |
| v13/s000015 | 4 (Cucumber) | 6 (Cucumber) | 0.46/0.30 | yes | 23 | 0.52 | 23 | 0.96 |
| v13/s000017 | 2 (boiled egg) | 4 (Tortilla_Chips) | 0.37/0.75 | yes | 18 | 0.61 | 11 | 0.91 |
| v13/s000017 | 3 (Tortilla_Chips) | 5 (Tortilla_Chips) | 0.52/0.95 | yes | 14 | 0.71 | 13 | 0.69 |
| v13/s000017 | 6 (cherry tomato) | 7 (cherry tomato) | 0.36/0.43 | yes | 19 | 0.42 | 10 | 1.00 |
| v13/s000018 | 1 (quinoa) | 3 (salsa) | 0.77/0.51 | yes | 24 | 0.75 | 24 | 0.92 |
| v13/s000018 | 2 (hummus) | 5 (tomato sauce) | 0.88/0.35 | yes | 23 | 0.09 | 22 | 0.82 |
| v13/s000018 | 3 (salsa) | 7 (lemon slice) | 0.38/0.91 | yes | 24 | 0.04 | 22 | 0.32 |
| v13/s000018 | 9 (lemon slice) | 11 (lemon slice) | 0.63/0.93 | yes | 24 | 0.42 | 24 | 0.42 |

Merge pairs where one SAM mask spans both items in > 50 % of co-visible frames (2D), whole items: **70 / 137**; restricted to the points carrying the merged id: **105 / 137** (mean 0.64); mean one-mask fraction 0.53; same category 65 / 137; touching 73 / 137.


## 4. Boundary errors

Per-point errors of the matched lift id (false negatives + false positives, 63652 voxels, 2 mm grid; distance to the item's true boundary):

- within 3 mm: **0.41**, within 6 mm: **0.68**, within 12 mm: 0.84 of all error voxels.

| error type | error voxels | within 3 mm | within 6 mm |
|---|---|---|---|
| ok | 12266 | 0.62 | 0.89 |
| boundary | 6312 | 0.64 | 0.90 |
| fragmented | 12122 | 0.35 | 0.63 |
| merged | 18035 | 0.33 | 0.61 |
| under-covered | 14045 | 0.30 | 0.56 |
| missed | 872 | 0.24 | 0.48 |

If every misassigned voxel within the band were snapped to GT (matching and ids unchanged): instance mIoU 0.631 -> **0.777** (3 mm, +0.146), **0.877** (6 mm, +0.246), 0.926 (12 mm, +0.296). Gain by the item's primary error type (6 mm band, mIoU points):

| error type | gain at 3 mm | gain at 6 mm | gain at 12 mm |
|---|---|---|---|
| ok | 0.032 | 0.045 | 0.048 |
| boundary | 0.021 | 0.033 | 0.037 |
| fragmented | 0.044 | 0.079 | 0.098 |
| merged | 0.035 | 0.064 | 0.081 |
| under-covered | 0.011 | 0.019 | 0.025 |
| missed | 0.003 | 0.005 | 0.006 |
