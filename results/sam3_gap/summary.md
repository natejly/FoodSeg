# SAM 3 gap on synthetic renders: prompt/threshold vs realism

Single-image SAM 3 (image predictor, fp16, Turing patches), 40 synthetic frames (top-down + oblique of 20 hard_v12 scans,
1008 long side, 313 visible items with >= 100 px) and 40 real Nutrition5k overhead frames (79 mapped ingredients).
Reference for real = results/n5k/masks_sam3 (generic "food" at 0.4). GPU time: 161 s inference + 10 s model load.
Code: pipeline/eval/sam3_gap.py (run) and sam3_gap_report.py (report, run with the `synth` env for pandas); raw rows in rows.json;
qualitative synth|real pairs qual_*.jpg (each: frame, mask overlay | real frame, overlay; the five worst synthetic categories then the three best).

## Reading

1. The miss is mostly prompt and path, not realism. Image-level generic "food" recalls 93.5 % of GT food pixels at 0.5 (98 % at 0.3) with 99 % / 96 % precision on the renders; real N5k gets the same call (IoU 0.86 vs the existing masks at 0.5).
2. Oracle names are the weak prompt on both domains: any-mask rate 0.60 (exact) / 0.57 (simple) at 0.5 on synthetic versus 0.52 / 0.63 on real N5k, so names fail on real photos at the same rate; they are not a realism signal.
3. Lowering the named-prompt threshold 0.5 -> 0.15 lifts synthetic hit (IoU >= 0.3) from 0.47 to 0.68; simplified names beat exact ones by ~3 points at 0.15. Still 25 % of items are missed by the best name; generic "food" at 0.15-0.3 finds 86-95 % of items, best of names or "food" 96 % at 0.15.
4. "dish" is useless (item hit 0.06 at 0.3); "meal" merges (recall 0.93 but hit 0.25 at 0.3). Only "food" is usable. "food" at 0.5 returns 5.6 masks for 7.8 visible items (merges), at 0.3 it returns 10.7 (splits plus clutter); item-level hit 0.59 at 0.5, 0.86 at 0.3, 0.94 at 0.15 with precision dropping to 0.54 at 0.15.
5. Because image-level "food"@0.5 already covers 93 % of food pixels, the 55 % unvoted food points in sam3_votes cannot come from recognisability. It points at the video path (tracker propagation, reprompt windows, the 40+ oracle concepts per scan competing under score-based overlap, 960 px label maps) and the lift/matching step; this single-image probe did not exercise those, so that attribution is by elimination, not measured.
6. Realism gap exists but is confined to specific procedural items: name-prompt hit 0.54 for procedural vs 0.77 for MetaFood3D scans at 0.15 (0.27 vs 0.57 detection at 0.5). Worst: procedural_leaves (hit 0.30; lettuce, radicchio, mixed greens), procedural_piece (0.45; rice, mashed potato, hummus, black beans), primitives (squash = orange cubes that read as cheese, cake slice, sushi). Generic "food" still segments these (kind hit at 0.3: procedural 0.93, scan 0.81, asset 0.95).
7. Item size matters only at the small end (hit 0.61 small vs 0.72 medium/large at 0.15; 0.40 vs 0.47-0.53 at 0.5). Top-down is not worse than oblique (0.65 vs 0.72).
8. No category is unfindable: every category is hit by the simple name at 0.15 or by generic prompts at 0.3. Categories whose simple name never works (hit 0): bacon, black beans, brown rice, cake slice, chicken breast, melons, mashed potato, green beans, sushi, squash, radicchio, rice (all found by "food"). Bacon and chicken breast do work on real photos, so those are render-recognisability misses; asparagus, grapes, hummus remain weak even for "food".
9. Recommendation (i) segment.py: make "food" the primary concept at ~0.3 (not oracle names); use names only as a secondary 0.15-0.3 pass with simplified 1-2 word names (SIMPLE table in sam3_gap.py) and never "dish"/"meal"; per-concept thresholds instead of the single 0.5; then compare video vs per-frame image detection on the same scans to find where the 55 % is lost.
10. Recommendation (ii) generator: do not spend effort on scan assets; fix procedural leaves, rice/mashed potato/hummus pieces and the squash/cake/sushi primitives (texture, colour, shape: the squash cubes are cheese-coloured; compare qual_2_squash.jpg). Realism work is secondary to the segment.py path fix.

## 1. Detection by domain x prompt style x threshold

detect = prompt returned any mask with score >= thr; hit = best mask IoU >= 0.3 with the GT item (synthetic) or with the n5k generic-food mask (real; named prompts are not expected to cover the whole dish, read real as a detectability reference); best_iou = mean over rows; recall = mean union recall of the item/reference.

| domain | style | thr | n | detect | hit | best_iou | recall |
|---|---|---|---|---|---|---|---|
| real | exact | 0.150 | 79 | 0.646 | 0.316 | 0.267 | 0.362 |
| real | exact | 0.300 | 79 | 0.595 | 0.228 | 0.221 | 0.307 |
| real | exact | 0.500 | 79 | 0.519 | 0.101 | 0.119 | 0.219 |
| real | simple | 0.150 | 79 | 0.722 | 0.367 | 0.312 | 0.429 |
| real | simple | 0.300 | 79 | 0.671 | 0.266 | 0.259 | 0.358 |
| real | simple | 0.500 | 79 | 0.633 | 0.152 | 0.166 | 0.292 |
| real | generic | 0.150 | 120 | 1.000 | 0.975 | 0.931 | 0.977 |
| real | generic | 0.300 | 120 | 1.000 | 0.908 | 0.860 | 0.909 |
| real | generic | 0.500 | 120 | 0.983 | 0.792 | 0.726 | 0.788 |
| synth | exact | 0.150 | 313 | 0.703 | 0.661 | 0.581 | 0.676 |
| synth | exact | 0.300 | 313 | 0.652 | 0.569 | 0.500 | 0.593 |
| synth | exact | 0.500 | 313 | 0.597 | 0.473 | 0.409 | 0.493 |
| synth | simple | 0.150 | 313 | 0.770 | 0.684 | 0.604 | 0.702 |
| synth | simple | 0.300 | 313 | 0.671 | 0.572 | 0.508 | 0.602 |
| synth | simple | 0.500 | 313 | 0.569 | 0.466 | 0.407 | 0.478 |
| synth | generic | 0.150 | 939 | 1.000 | 0.605 | 0.551 | 0.841 |
| synth | generic | 0.300 | 939 | 1.000 | 0.391 | 0.367 | 0.684 |
| synth | generic | 0.500 | 939 | 0.994 | 0.242 | 0.234 | 0.594 |

## 1b. Generic prompts separately

| domain | prompt | thr | detect | hit | best_iou | recall |
|---|---|---|---|---|---|---|
| real | dish | 0.150 | 1.000 | 0.925 | 0.865 | 0.935 |
| real | dish | 0.300 | 1.000 | 0.750 | 0.680 | 0.739 |
| real | dish | 0.500 | 0.975 | 0.450 | 0.394 | 0.419 |
| real | food | 0.150 | 1.000 | 1.000 | 0.962 | 1.000 |
| real | food | 0.300 | 1.000 | 1.000 | 0.945 | 1.000 |
| real | food | 0.500 | 1.000 | 0.975 | 0.859 | 0.996 |
| real | meal | 0.150 | 1.000 | 1.000 | 0.965 | 0.997 |
| real | meal | 0.300 | 1.000 | 0.975 | 0.956 | 0.987 |
| real | meal | 0.500 | 0.975 | 0.950 | 0.924 | 0.950 |
| synth | dish | 0.150 | 1.000 | 0.249 | 0.246 | 0.572 |
| synth | dish | 0.300 | 1.000 | 0.061 | 0.059 | 0.149 |
| synth | dish | 0.500 | 1.000 | 0.006 | 0.005 | 0.009 |
| synth | food | 0.150 | 1.000 | 0.936 | 0.829 | 0.979 |
| synth | food | 0.300 | 1.000 | 0.859 | 0.768 | 0.970 |
| synth | food | 0.500 | 1.000 | 0.588 | 0.547 | 0.888 |
| synth | meal | 0.150 | 1.000 | 0.629 | 0.579 | 0.972 |
| synth | meal | 0.300 | 1.000 | 0.252 | 0.275 | 0.934 |
| synth | meal | 0.500 | 0.981 | 0.131 | 0.150 | 0.886 |

## 2. Synthetic image-level food recall (union of masks vs all GT food pixels)

| prompt | thr | detect | food_recall | food_precision |
|---|---|---|---|---|
| dish | 0.150 | 1.000 | 0.696 | 0.182 |
| dish | 0.300 | 1.000 | 0.240 | 0.072 |
| dish | 0.500 | 1.000 | 0.011 | 0.009 |
| food | 0.150 | 1.000 | 0.987 | 0.540 |
| food | 0.300 | 1.000 | 0.983 | 0.960 |
| food | 0.500 | 1.000 | 0.935 | 0.992 |
| meal | 0.150 | 1.000 | 0.983 | 0.410 |
| meal | 0.300 | 1.000 | 0.957 | 0.887 |
| meal | 0.500 | 0.950 | 0.880 | 0.943 |

## 3. Synthetic items: best of exact/simple names vs best of food/dish/meal

| thr | n | any_named | hit_named | any_gen | hit_gen | any_either | hit_either |
|---|---|---|---|---|---|---|---|
| 0.150 | 313.000 | 0.808 | 0.748 | 1.000 | 0.946 | 1.000 | 0.962 |
| 0.300 | 313.000 | 0.725 | 0.633 | 1.000 | 0.863 | 1.000 | 0.911 |
| 0.500 | 313.000 | 0.642 | 0.527 | 1.000 | 0.591 | 1.000 | 0.757 |

## 4. Per category, synthetic (simplified name; generic = best of food/dish/meal at 0.3)

| category | n@.15 | detect@.15 | hit@.15 | iou@.15 | detect@.5 | hit@.5 | generic_hit@.3 | real_n | real_detect |
|---|---|---|---|---|---|---|---|---|---|
| bacon | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.667 | 3.000 | 1.000 |
| black beans | 4 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan |
| brown rice | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan |
| cake slice | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan |
| chicken breast | 2 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 3.000 | 1.000 |
| melons | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 3.000 | 0.333 |
| mashed potato | 4 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan |
| green beans | 2 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 3.000 | 0.333 |
| sushi | 12 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan |
| squash | 11 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 3.000 | 1.000 |
| radicchio | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan |
| rice | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 3.000 | 0.667 |
| hummus | 4 | 0.500 | 0.250 | 0.229 | 0.250 | 0.000 | 0.500 | nan | nan |
| mixed greens | 4 | 0.500 | 0.250 | 0.183 | 0.000 | 0.000 | 1.000 | 3.000 | 0.667 |
| asparagus | 7 | 1.000 | 0.286 | 0.312 | 0.571 | 0.143 | 0.429 | 3.000 | 1.000 |
| lettuce | 4 | 1.000 | 0.500 | 0.501 | 0.250 | 0.250 | 0.750 | 3.000 | 1.000 |
| grapes | 4 | 0.500 | 0.500 | 0.408 | 0.500 | 0.250 | 0.250 | 3.000 | 1.000 |
| pasta mixed dishes | 2 | 0.500 | 0.500 | 0.451 | 0.500 | 0.500 | 0.500 | 2.000 | 0.500 |
| salmon grill | 2 | 0.500 | 0.500 | 0.481 | 0.000 | 0.000 | 1.000 | nan | nan |
| sour cream | 2 | 1.000 | 0.500 | 0.486 | 0.000 | 0.000 | 1.000 | nan | nan |
| couscous | 2 | 0.500 | 0.500 | 0.440 | 0.000 | 0.000 | 1.000 | 2.000 | 0.000 |
| gravy | 4 | 1.000 | 0.500 | 0.462 | 0.000 | 0.000 | 1.000 | nan | nan |
| cherry tomato | 26 | 0.769 | 0.692 | 0.632 | 0.231 | 0.231 | 0.923 | 3.000 | 0.667 |
| ice cream | 4 | 0.750 | 0.750 | 0.683 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| bread roll | 4 | 1.000 | 0.750 | 0.735 | 0.000 | 0.000 | 1.000 | nan | nan |
| cookies | 10 | 0.900 | 0.800 | 0.764 | 0.800 | 0.600 | 1.000 | 3.000 | 0.667 |
| boiled egg | 6 | 0.833 | 0.833 | 0.719 | 0.500 | 0.500 | 0.667 | 3.000 | 1.000 |
| cucumber | 45 | 0.956 | 0.889 | 0.681 | 0.956 | 0.533 | 0.622 | nan | nan |
| broccoli | 28 | 0.929 | 0.893 | 0.770 | 0.929 | 0.786 | 0.714 | 3.000 | 0.667 |
| falafel | 27 | 1.000 | 0.963 | 0.860 | 1.000 | 0.963 | 0.963 | nan | nan |
| corn | 4 | 1.000 | 1.000 | 0.953 | 0.250 | 0.250 | 1.000 | 3.000 | 1.000 |
| blueberries | 2 | 1.000 | 1.000 | 0.767 | 0.500 | 0.500 | 1.000 | 2.000 | 0.500 |
| papaya | 2 | 1.000 | 1.000 | 0.972 | 0.000 | 0.000 | 1.000 | nan | nan |
| omelet | 2 | 1.000 | 1.000 | 0.874 | 0.000 | 0.000 | 1.000 | nan | nan |
| olive | 12 | 1.000 | 1.000 | 0.935 | 1.000 | 1.000 | 1.000 | 3.000 | 0.000 |
| melted cheese | 2 | 1.000 | 1.000 | 0.894 | 0.500 | 0.500 | 1.000 | 2.000 | 1.000 |
| kiwi | 4 | 1.000 | 1.000 | 0.924 | 1.000 | 1.000 | 1.000 | nan | nan |
| cauliflower | 6 | 1.000 | 1.000 | 0.905 | 1.000 | 1.000 | 1.000 | 3.000 | 0.667 |
| chicken thighs | 4 | 1.000 | 1.000 | 0.845 | 0.000 | 0.000 | 1.000 | 3.000 | 1.000 |
| pork chop | 2 | 1.000 | 1.000 | 0.965 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 |
| sausages | 4 | 1.000 | 1.000 | 0.958 | 1.000 | 1.000 | 1.000 | 3.000 | 0.667 |
| strawberry | 16 | 1.000 | 1.000 | 0.922 | 1.000 | 1.000 | 0.938 | 3.000 | 1.000 |
| yeast bread | 8 | 1.000 | 1.000 | 0.949 | 0.750 | 0.625 | 1.000 | 2.000 | 1.000 |
| yogurt | 4 | 1.000 | 1.000 | 0.932 | 1.000 | 1.000 | 1.000 | 3.000 | 0.000 |

Categories never hit (IoU >= 0.3) by the simple name at 0.15 nor by generic prompts at 0.3: none


## 5. By item size (pixels at 1008 long side, terciles), simple name

| size | thr | n | detect | hit | iou |
|---|---|---|---|---|---|
| small(<2312px) | 0.150 | 105 | 0.752 | 0.610 | 0.526 |
| small(<2312px) | 0.500 | 105 | 0.552 | 0.400 | 0.345 |
| medium | 0.150 | 104 | 0.769 | 0.721 | 0.641 |
| medium | 0.500 | 104 | 0.625 | 0.529 | 0.468 |
| large(>5894px) | 0.150 | 104 | 0.788 | 0.721 | 0.645 |
| large(>5894px) | 0.500 | 104 | 0.529 | 0.471 | 0.410 |

## 6. By item kind (scan = MetaFood3D, asset = Objaverse/YCB, procedural = fills/pieces/leaves/primitives), simple name

| kind | thr | n | detect | hit | iou |
|---|---|---|---|---|---|
| asset | 0.150 | 20 | 0.650 | 0.600 | 0.564 |
| asset | 0.500 | 20 | 0.600 | 0.500 | 0.474 |
| procedural | 0.150 | 103 | 0.689 | 0.544 | 0.501 |
| procedural | 0.500 | 103 | 0.282 | 0.272 | 0.254 |
| scan | 0.150 | 190 | 0.826 | 0.768 | 0.663 |
| scan | 0.500 | 190 | 0.721 | 0.568 | 0.484 |

By source at 0.15:

| source | n | detect | hit | iou |
|---|---|---|---|---|
| metafood3d | 190 | 0.826 | 0.768 | 0.663 |
| objaverse | 18 | 0.611 | 0.556 | 0.517 |
| primitive | 63 | 0.667 | 0.603 | 0.554 |
| procedural_leaves | 10 | 0.600 | 0.300 | 0.274 |
| procedural_piece | 22 | 0.682 | 0.455 | 0.425 |
| procedural_topping | 8 | 1.000 | 0.625 | 0.576 |
| ycb | 2 | 1.000 | 1.000 | 0.979 |

By view at 0.15:

| view | n | detect | hit |
|---|---|---|---|
| obl | 153 | 0.791 | 0.719 |
| top | 160 | 0.750 | 0.650 |

## 7. Generic 'food' on synthetic items by kind and threshold (hit = IoU>=0.3, hit5 = IoU>=0.5, recall = pixel recall)

| kind | thr | n | hit | hit5 | recall |
|---|---|---|---|---|---|
| asset | 0.150 | 20 | 1.000 | 1.000 | 0.993 |
| asset | 0.300 | 20 | 0.950 | 0.950 | 0.992 |
| asset | 0.500 | 20 | 0.700 | 0.650 | 0.989 |
| procedural | 0.150 | 103 | 0.981 | 0.951 | 0.988 |
| procedural | 0.300 | 103 | 0.932 | 0.913 | 0.981 |
| procedural | 0.500 | 103 | 0.631 | 0.621 | 0.840 |
| scan | 0.150 | 190 | 0.905 | 0.868 | 0.972 |
| scan | 0.300 | 190 | 0.811 | 0.768 | 0.961 |
| scan | 0.500 | 190 | 0.553 | 0.505 | 0.903 |

Masks returned by 'food' per frame vs visible GT items: thr 0.5 mean 5.6 masks / 7.8 items; thr 0.3 mean 10.7 masks / 7.8 items.


## 8. Real N5k generic prompts (reference = n5k food mask from 'food'@0.4 + dish filter)

| prompt | thr | n | detect | mean_masks | iou_vs_n5k_food |
|---|---|---|---|---|---|
| dish | 0.150 | 40 | 1.000 | 9.300 | 0.865 |
| dish | 0.300 | 40 | 1.000 | 2.975 | 0.680 |
| dish | 0.500 | 40 | 0.975 | 1.650 | 0.394 |
| food | 0.150 | 40 | 1.000 | 35.475 | 0.962 |
| food | 0.300 | 40 | 1.000 | 20.925 | 0.945 |
| food | 0.500 | 40 | 1.000 | 12.225 | 0.859 |
| meal | 0.150 | 40 | 1.000 | 10.800 | 0.965 |
| meal | 0.300 | 40 | 1.000 | 3.225 | 0.956 |
| meal | 0.500 | 40 | 0.975 | 1.625 | 0.924 |