# Learned hidden-volume head on synthetic scans

Reproduce (conda env `foodseg`, CPU):

    cd /home/kyle/Desktop/FoodSeg; PY=/home/kyle/miniconda3/envs/foodseg/bin/python
    $PY -m pipeline.eval.volume_head --target envelope     # -> tables_envelope.md, results_envelope.json (default target)
    $PY -m pipeline.eval.volume_head --target solid        # -> tables_solid.md, results_solid.json (same 244 items)
    $PY -m pipeline.eval.volume_head_envelope              # -> mass_two_stage.md, views_aware.md, headline_targets.md (+ .json)
    $PY -m pytest pipeline/tests/test_volume_head.py -q

Target (new, default): `log(V_envelope_gt / V_visible_foodscan)`, V_envelope_gt from `results/rq1/envelope_items.csv` (full GT top surface column-filled to the container; joined by batch/scan/item_id, all 244 valid items covered). Same features as before (no GT-only inputs; `features.npz` unchanged). `--target solid` reproduces the old target on the identical item population (its numbers equal the superseded section below). APE = |V_hat - V_ref| / V_ref per item (rep-averaged for category CV); bias = median signed error; dish = per-scan sum of the evaluated items; CIs are paired scan-cluster bootstraps (1000 draws) of (head - baseline), head = HGB (L1) (best by category-CV for both targets). Tables: `tables_envelope.md`, `tables_solid.md`, `headline_targets.md`, `mass_two_stage.md`, `views_aware.md`.

## Envelope target (current)

### Headline: baseline (no correction) -> head, envelope and solid targets side by side

Item median APE %, diff in pp [95% CI]; dish median APE %.

| target | protocol | n | item median APE | item mean APE | median bias % | dish median APE |
|---|---|---|---|---|---|---|
| envelope | lobo_v12 (5 batches -> v12) | 145 | 51 -> 81 (+29 [-1, +61]) | 50 -> 97 (+47 [+21, +72]) | -51 -> +81 | 38 -> 68 (+29 [+2, +69]) |
| solid | lobo_v12 | 145 | 51 -> 41 (-10 [-21, +6]) | 114 -> 152 (+38 [+10, +80]) | -39 -> +19 | 27 -> 28 (+1 [-20, +23]) |
| envelope | rev_v12 (v12 -> 5 batches) | 99 | 83 -> 58 (-25 [-39, -14]) | 71 -> 56 (-15 [-22, -7]) | -83 -> -44 | 86 -> 59 (-27 [-51, -17]) |
| solid | rev_v12 | 99 | 74 -> 66 (-8 [-20, +11]) | 91 -> 104 (+13 [-22, +51]) | -68 -> -41 | 74 -> 55 (-19 [-25, -8]) |
| envelope | lobo_all (leave-one-batch-out) | 244 | 60 -> 55 (-5 [-26, +14]) | 59 -> 74 (+16 [-4, +33]) | -60 -> +29 | 70 -> 34 (-36 [-50, -11]) |
| solid | lobo_all | 244 | 59 -> 42 (-16 [-26, -5]) | 104 -> 122 (+18 [-2, +39]) | -48 -> +6 | 58 -> 34 (-25 [-41, -7]) |
| envelope | category-disjoint, all batches | 244 | 60 -> 23 (-37 [-43, -29]) | 59 -> 41 (-18 [-26, -5]) | -60 -> +4 | 70 -> 21 (-48 [-58, -34]) |
| solid | category-disjoint, all batches | 244 | 59 -> 34 (-24 [-32, -16]) | 104 -> 117 (+12 [-11, +46]) | -48 -> -1 | 58 -> 29 (-30 [-45, -13]) |
| envelope | category-disjoint, hard_v12 only | 145 | 51 -> 19 (-33 [-40, -23]) | 50 -> 26 (-24 [-30, -17]) | -51 -> +4 | 38 -> 8 (-31 [-36, -25]) |
| solid | category-disjoint, hard_v12 only | 145 | 51 -> 27 (-24 [-32, -11]) | 114 -> 118 (+4 [-15, +28]) | -39 -> -4 | 27 -> 13 (-14 [-21, -1]) |

(Per-model tables for all 8 models, size/kind/stacked/occlusion breakdowns, ablation, learning curve: `tables_envelope.md`.)

### Importance, envelope vs solid (held-out permutation, category CV; increase in mean |log error|; held-out error 0.425 vs 0.645 before)

| group | envelope | solid (before) |
|---|---|---|
| visible volumes | +0.213 | +0.161 |
| footprint | +0.124 | +0.079 |
| views / lift | +0.118 | +0.075 |
| scan context | +0.043 | +0.012 |
| point stats | +0.027 | +0.019 |
| contact / stacking | +0.021 | +0.054 |
| heights | +0.021 | +0.007 |
| kind | +0.000 | +0.053 |

Top single features now: r_voxel_fs, convexity, lift_nvotes, r_hull_fs, n_pts, lv_foodscan (before: r_hull_fs, lift_nvotes, convexity, r_voxel_fs, kind_piece, step_max). The kind and stacking-step features lose their weight: they were carrying the void/solid fraction and stacking effects that the envelope target removes. Learning curve (category CV, HGB L2): 60 baseline -> 34 / 27 / 24 % at 25 / 50 / 100 % of scans (still falling).

### Two-stage mass on hard_v12 (`mass_two_stage.md`; densities = per-kind medians fitted on hard_v10 only)

(a) mass = V x bulk(kind); (b) mass = V / envelope_ratio(kind) x material(kind). Median APE % [95% scan-cluster CI]; 145 items, 20 dishes.

| protocol | path | volume | item APE | item bias | dish APE |
|---|---|---|---|---|---|
| category-disjoint (v12 only) | (a) | baseline V_vis | 65 [60, 70] | -62 | 57 [50, 63] |
| | (a) | head | 36 [31, 41] | -23 | 24 [16, 33] |
| | (b) | baseline V_vis | 66 [60, 72] | -65 | 55 [46, 63] |
| | (b) | head | 41 [36, 45] | -27 | 22 [18, 33] |
| | (a) | oracle V_envelope_gt (floor) | 32 [27, 41] | -19 | 20 [14, 29] |
| batch-disjoint (5 batches -> v12) | (a) | baseline | 65 [60, 70] | -62 | 57 [50, 63] |
| | (a) | head | 61 [51, 76] | +51 | 32 [21, 68] |
| | (b) | baseline | 66 [60, 72] | -65 | 55 [46, 63] |
| | (b) | head | 60 [44, 70] | +43 | 39 [19, 75] |

Paired item-APE difference head - baseline (pp): category-disjoint (a) -28.4 [-33.8, -20.4], (b) -24.7 [-31.6, -17.4]; batch-disjoint (a) -3.9 [-14.2, +14.0] (mean +48.7 [+25.9, +78.8]), (b) -5.9 [-22.2, +7.4]. Density path: (b) - (a) is +1.4 [-0.9, +4.1] pp on the baseline volume and +5.0 [+1.2, +7.5] on the corrected volume (category-disjoint); -0.6 [-12.9, +6.2] batch-disjoint. So bulk density (a) is the better path, by about 1 pp without correction (not significant) and 5 pp with it (just significant); dish APE is the same within CI (24 vs 22). The head takes item mass APE 65 -> 36 %, within 4 pp of the oracle-envelope floor (32 %), i.e. the remaining mass error is the density table, not the volume.

### Views-aware / mixed-density training (`views_aware.md`)

Extra features: scan n_views (frames.jsonl: 8 for v9-v11, 24 for hard, hard_v3, hard_v12; only v12 records it in gt.json provenance), log n_views, points per view, lift votes / n_views, item frames_visible / n_views. NB the existing `n_views` feature (item frames_visible) already equals the scan view count, so "base" is already views-aware; "no views features" drops n_views/lift_*. 67 items in 8-view scans, 177 in 24-view. Median item APE %, head - baseline CI in pp:

| protocol | test density | baseline | no views feats | base feats | views-aware |
|---|---|---|---|---|---|
| scan-disjoint 5-fold, mixed train | 8-view | 92 | 45 [-60,-30] | 43 [-58,-36] | 45 [-58,-37] |
| | 24-view | 51 | 18 [-38,-22] | 18 [-38,-23] | 18 [-38,-22] |
| category-disjoint, mixed train | 8-view | 92 | 45 [-60,-34] | 46 [-57,-38] | 47 [-54,-38] |
| | 24-view | 51 | 20 [-37,-20] | 17 [-39,-22] | 17 [-38,-22] |
| batch-disjoint (leave one batch out) | 8-view | 92 | 53 | 41 | 39 |
| | 24-view | 51 | 64 [-12,+38] | 64 [-10,+38] | 65 [-10,+35] |
| train 8-view only -> 24-view | 24-view | 51 | 244 | 190 | 190 |
| train 24-view only -> 8-view | 8-view | 92 | 77 | 76 | 76 |
| train 24 only -> 24 (scan-disjoint) | 24-view | 51 | 17 | 18 | 18 |
| train 8 only -> 8 (scan-disjoint) | 8-view | 92 | 54 | 55 | 55 |

Per held-out batch (views-aware head): hard 53 -> 10, v3 29 -> 19, v9 93 -> 41, v10 92 -> 28, v11 92 -> 49, v12 51 -> 74. A per-density median ratio (oracle on n_views) gives 43 % on 24-view and 79 % on 8-view.

Reading: mixed 8/24-view training works on held-out scans and held-out categories of both densities (24-view 51 -> 17-18 %, 8-view 92 -> 43-47 %), and adding explicit views features changes nothing (the head already infers the density from point statistics: "no views features" is within 2-3 pp). Training on one density and testing on the other fails badly (190-244 % vs 51 %; 76 % vs 92 %), so both densities must be in the training set, but one batch's worth of each is enough when it is not held out. What still fails is the batch-disjoint case for hard_v12 (74-81 % vs 51 %, bias +60-80 %): the head is trained on three 8-view batches that dominate (V_env/V_vis 13x) and only 32 24-view items from other batches, and it carries their offset. Held-out 8-view batches and the small 24-view batches transfer fine once other batches of that density plus the other density are present. So mixed density fixes the density problem; what is truly needed is enough in-distribution examples of each density (several batches of 24-view data), not density-matched batches per se. Caveat: scan-disjoint splits share generator/scenes with training scans, category-disjoint is the stricter one and gives the same picture.

### Reading (6 lines)

1. Envelope target is easier: in-distribution median APE 51 -> 19 % (v12 only) and 60 -> 23 % (all batches), against 27 / 34 % for solid; the head no longer has to learn void/solid fraction, kind importance drops to 0.
2. Cross-batch it is not solved: v12 as held-out batch is worse than doing nothing (51 -> 81 %, bias +81 %) because 8-view batches (V_env/V_vis ~13x) dominate training; reverse direction improves 83 -> 58 %.
3. Two-stage mass: category-disjoint v12 item APE 65 -> 36 % (floor 32 %), dish 57 -> 24 %; bulk-density path (a) beats the envelope-ratio x material path (b) by 1 pp (baseline) / 5 pp (head), dish equal.
4. Mixed 8+24-view training fixes density: both densities improve on held-out scans and categories; explicit views features add nothing; single-density training does not transfer.
5. The remaining blocker for deployment is the batch/domain shift (number and variety of 24-view training batches), not view density as such.
6. Real-data caveat unchanged: perfect labels, synthetic occlusion statistics, 244 items; treat 19-23 % as an in-distribution floor.

## solid target (superseded)

The sections below are the original run (target log(volume_ml / V_vis), populations `valid` and `all`); numbers on the `valid` population are reproduced by `--target solid`.


Reproduce: `/home/kyle/miniconda3/envs/foodseg/bin/python -m pipeline.eval.volume_head` (CPU, ~minutes). Features: `pipeline/eval/volume_head_features.py` (documented in its docstring); cache `results/volume_head/features.npz`.
Setup: perfect GT labels, foodscan container-relative visible volume as V_vis, targets log(V_true/V_vis) and log V_true; APE = |V_hat - V_true| / V_true per item; bias = median signed (V_hat - V_true) / V_true; dish = per-scan sum over the evaluated items. `valid` drops the manifest's dropped_items (tiny/invisible); `all` keeps everything with >= 20 points. Only features computable from a scan are inputs; occluded_fraction / n_gt_points / true volume are targets or diagnostics only.

### Reading

1. Population `valid` (244 items, 74 scans; 145 of them from hard_v12). Baseline = foodscan visible volume with perfect labels: median APE 59% (bias -48%) pooled, 51% (-39%) on hard_v12.
2. Batches differ hugely: the 8-view batches (v9/v10/v11) hide 5-7x of the volume (median V_true/V_vis 5.2-7.1, baseline 87% APE); the 24-view ones 1.0-1.6x. A model has to learn that offset from the number of views / lift votes, which is why `lift_nvotes` ranks high in category CV (a batch proxy). A per-batch median ratio alone already gets 50% in category CV.
3. Recoverable fraction, within-distribution (category-disjoint CV, tables above): boosted L1 median APE 59 -> 36% (CI of the difference -32..-16 pp), bias ~0, dish-total APE 58 -> 27%, mass APE 63 -> 40%. Inside hard_v12 only (batch offsets removed): 51 -> 29% (ridge 30%), dish 27 -> 14%, mass 51 -> 31%. So roughly 40-45% of the median error is statistically recoverable; the remaining ~30% is not.
4. It does NOT generalise across batches: train on 5 batches, test v12: 51 -> 41% (CI -21..+6 pp, not significant, mean APE worsens +38 pp); train v12 only, test the rest: 74 -> 66% (n.s.). The global median ratio is worse than doing nothing on v12 (94%), because the hidden fraction depends on how many views the scan has.
5. The gain is concentrated in piles (51 -> 26% in category CV), singles (60 -> 39%) and pieces (68 -> 27%), and in items > 10 mL. It fails for toppings and leaves (n = 7 and 14; baseline already over-estimates by +130..+350% from air gaps in the column fill, the model cannot learn that with this few examples: 460-560% APE), for items < 10 mL (82-111% vs 63%), and for lightly occluded items (< 50% hidden), which it over-corrects (+50..+80% bias).
6. Stacking: stacked scans improve in category CV (61 -> 32%) but not across batches (lobo_v12: 50 -> 57%, bias +49%); the stacking features (contact, step heights) are used (top group after volumes/footprint) but one batch's worth of stacks is too little.
7. Importance (held-out permutation, category CV): visible-volume group (esp. hull/voxel to foodscan log ratios) > footprint (convexity) > views/lift > contact/stacking > kind; heights, box, container and scan context add almost nothing. An oracle input of the true occluded_fraction only lowers category-CV APE from 36 to 30%, i.e. knowing the hidden fraction does not remove the rest of the error (shape/void uncertainty of the hidden part).
8. More synthetic scans would help: category-CV median APE for HGB (L2) falls 53 -> 48 -> 40% for 25/50/100% of training scans (baseline 59%) and has not flattened; the lobo_v12 curve is too noisy (sd 13-37 pp) to say.
9. Real-data version: the container-relative visible volumes, point count/density, footprint area and convexity, height percentiles, bbox, contact with other items / container wall and step heights are all available from an iPad scan (cloud + SAM 3 labels + empty-container template). n_views and lift_share/votes need per-frame mask visibility (available from the SAM 3 lift, but the real distribution differs from synthetic). Dropping them costs little in category CV (36 -> 38%) but more in lobo_v12 (41 -> 43%).
10. Caveats: small data (244 items; the model-vs-baseline CI is wide across batches), perfect labels, synthetic occlusion statistics, and a category-CV that still shares scans' containers; treat 29-36% as an optimistic in-distribution floor and 41-66% as the cross-batch reality, i.e. a statistical head is a useful prior for dish totals/mass but not a replacement for completion (SAM 3D cluster-only) on unseen view counts.

## Tables

### Population: valid (n = 244 items, 74 scans)


#### lobo_v12

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 145 | 51 | 114 | -39 | 27 | -26 | 50 | -35 |
| global median ratio | 145 | 94 | 352 | 94 | 135 | 135 | 106 | 106 |
| per-kind median ratio | 145 | 78 | 269 | 64 | 241 | 241 | 78 | 56 |
| ridge | 145 | 83 | 77 | -82 | 84 | -84 | 82 | -82 |
| HGB (L2) | 145 | 45 | 131 | 7 | 27 | -10 | 45 | 9 |
| HGB (L1) | 145 | 41 | 152 | 19 | 28 | 21 | 46 | 29 |
| HGB direct log V | 145 | 53 | 80 | -34 | 44 | -42 | 51 | -31 |
| MLP | 145 | 62 | 63 | -57 | 50 | -50 | 64 | -58 |


#### rev_v12

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 99 | 74 | 91 | -68 | 74 | -73 | 84 | -77 |
| global median ratio | 99 | 71 | 117 | -48 | 72 | -56 | 78 | -63 |
| per-kind median ratio | 99 | 80 | 75 | -58 | 85 | -73 | 75 | -68 |
| ridge | 99 | 63 | 94 | -22 | 47 | -34 | 67 | -22 |
| HGB (L2) | 99 | 71 | 100 | -59 | 71 | -70 | 76 | -63 |
| HGB (L1) | 99 | 66 | 104 | -41 | 55 | -53 | 83 | -55 |
| HGB direct log V | 99 | 73 | 122 | -52 | 70 | -70 | 83 | -52 |
| MLP | 99 | 66 | 101 | -41 | 63 | -52 | 68 | -33 |


#### lobo_all

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 244 | 59 | 104 | -48 | 58 | -55 | 62 | -49 |
| global median ratio | 244 | 87 | 264 | 47 | 89 | -16 | 89 | 53 |
| per-kind median ratio | 244 | 80 | 199 | 25 | 88 | -14 | 77 | 10 |
| ridge | 244 | 78 | 80 | -70 | 50 | -37 | 79 | -72 |
| HGB (L2) | 244 | 46 | 107 | 3 | 31 | -13 | 48 | 3 |
| HGB (L1) | 244 | 42 | 122 | 6 | 34 | -2 | 49 | 13 |
| HGB direct log V | 244 | 43 | 69 | -19 | 23 | -17 | 50 | -22 |
| MLP | 244 | 55 | 59 | -39 | 39 | -23 | 58 | -41 |


#### catcv

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 244 | 59 | 104 | -48 | 58 | -55 | 63 | -45 |
| global median ratio | 244 | 65 | 174 | 2 | 69 | -14 | 68 | 6 |
| per-kind median ratio | 244 | 68 | 119 | 2 | 74 | -9 | 64 | 3 |
| ridge | 244 | 44 | 79 | -1 | 36 | -14 | 50 | 9 |
| HGB (L2) | 244 | 42 | 95 | -4 | 27 | -11 | 46 | 6 |
| HGB (L1) | 244 | 36 | 117 | 0 | 27 | -2 | 40 | 4 |
| HGB direct log V | 244 | 39 | 89 | -2 | 23 | -14 | 45 | 8 |
| MLP | 244 | 43 | 80 | -2 | 26 | -8 | 48 | 7 |


Best learned model by category-CV median APE: **HGB (L1)**.


#### Paired cluster bootstrap (by scan), model - baseline APE, percentage points (95% CI)

| protocol | comparison | d median APE | d mean APE | P(better) % |
|---|---|---|---|---|
| lobo_v12 | HGB (L1) - baseline | -9.5 [-21.2, +5.6] | +38.1 [+9.5, +79.5] | 85 |
| lobo_v12 | HGB (L1) - global | -52.5 [-84.5, -39.9] | -200.4 [-355.1, -111.5] | 100 |
| lobo_v12 | global median ratio - baseline | +42.9 [+24.7, +81.6] | +238.5 [+137.8, +402.5] | 0 |
| rev_v12 | HGB (L1) - baseline | -8.1 [-20.2, +11.1] | +13.1 [-21.8, +51.3] | 81 |
| rev_v12 | HGB (L1) - global | -5.1 [-15.9, +9.7] | -13.5 [-61.5, +28.0] | 76 |
| rev_v12 | global median ratio - baseline | -3.0 [-9.3, +7.3] | +26.6 [+5.6, +52.1] | 81 |
| lobo_all | HGB (L1) - baseline | -16.3 [-26.5, -5.1] | +17.6 [-2.5, +38.7] | 100 |
| lobo_all | HGB (L1) - global | -44.4 [-55.1, -35.0] | -142.2 [-219.9, -81.3] | 100 |
| lobo_all | global median ratio - baseline | +28.0 [+15.6, +40.9] | +159.8 [+94.7, +243.2] | 0 |
| catcv | HGB (L1) - baseline | -24.4 [-32.3, -15.6] | +12.3 [-10.9, +46.3] | 100 |
| catcv | HGB (L1) - global | -32.9 [-41.2, -21.2] | -57.1 [-103.0, -20.9] | 100 |
| catcv | global median ratio - baseline | +8.5 [-3.8, +17.6] | +69.5 [+38.3, +111.5] | 11 |


##### lobo_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 31 | 69 / -23 | 144 / +144 | 82 / +34 |
| 10-50 mL | 75 | 48 / -42 | 86 / +83 | 48 / +35 |
| >50 mL | 39 | 40 / -38 | 96 / +96 | 29 / -2 |


##### lobo_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 74 | 47 / -41 | 85 / +85 | 39 / +32 |
| single | 52 | 46 / -40 | 93 / +90 | 32 / +0 |
| piece | 10 | 49 / -49 | 61 / +61 | 44 / +44 |
| topping | 4 | 265 / +265 | 1048 / +1048 | 821 / +821 |
| leaves | 5 | 1286 / +1286 | 4264 / +4264 | 1188 / +1188 |


##### lobo_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 52 | 50 / -30 | 119 / +119 | 57 / +49 |
| unstacked scans | 93 | 52 / -40 | 92 / +89 | 34 / +4 |


##### lobo_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 30 | 63 / +3 | 224 / +224 | 62 / +62 |
| 50-80% | 87 | 39 / -37 | 100 / +100 | 41 / +31 |
| >=80% | 28 | 67 / -67 | 59 / +5 | 33 / -19 |


##### catcv by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 55 | 63 / -28 | 73 / +39 | 82 / +60 |
| 10-50 mL | 98 | 52 / -43 | 54 / +11 | 27 / +2 |
| >50 mL | 91 | 68 / -66 | 75 / -39 | 30 / -19 |


##### catcv by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 92 | 51 / -46 | 44 / +6 | 26 / +1 |
| single | 72 | 60 / -46 | 70 / +6 | 39 / -2 |
| piece | 59 | 68 / -68 | 79 / -39 | 27 / -11 |
| topping | 7 | 353 / +353 | 814 / +814 | 462 / +462 |
| leaves | 14 | 136 / +129 | 339 / +339 | 562 / +562 |


##### catcv by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 114 | 61 / -51 | 75 / -3 | 32 / -0 |
| unstacked scans | 130 | 56 / -43 | 55 / +9 | 35 / -2 |


##### catcv by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 33 | 63 / +3 | 101 / +101 | 58 / +52 |
| 50-80% | 104 | 44 / -39 | 44 / +18 | 26 / -1 |
| >=80% | 75 | 86 / -86 | 75 / -74 | 49 / -28 |


##### rev_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 24 | 53 / -37 | 56 / +3 | 64 / +52 |
| 10-50 mL | 23 | 58 / -53 | 69 / -24 | 52 / -15 |
| >50 mL | 52 | 87 / -87 | 84 / -78 | 78 / -74 |


##### rev_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 18 | 53 / -53 | 56 / -23 | 78 / +1 |
| single | 20 | 80 / -80 | 73 / -68 | 61 / -56 |
| piece | 49 | 74 / -74 | 72 / -58 | 62 / -56 |
| topping | 3 | 881 / +881 | 1496 / +1496 | 450 / +450 |
| leaves | 9 | 81 / -28 | 69 / +18 | 66 / +35 |


##### rev_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 62 | 76 / -74 | 72 / -57 | 75 / -47 |
| unstacked scans | 37 | 72 / -61 | 69 / -37 | 52 / -15 |


##### rev_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 3 | 61 / -58 | 37 / -32 | 33 / -31 |
| 50-80% | 17 | 67 / -61 | 54 / -37 | 47 / -41 |
| >=80% | 47 | 91 / -91 | 86 / -86 | 81 / -81 |


#### Per-batch structure (valid items)

| batch | n items | median views seen | median V_true/V_vis | baseline med APE % |
|---|---|---|---|---|
| hard | 17 | 24 | 1.17 | 26 |
| hard_v10 | 19 | 8 | 5.66 | 87 |
| hard_v11 | 30 | 8 | 7.14 | 86 |
| hard_v12 | 145 | 24 | 1.63 | 51 |
| hard_v3 | 15 | 24 | 1.04 | 39 |
| hard_v9 | 18 | 8 | 5.19 | 87 |


Diagnostic, category CV with a per-batch median ratio (uses the batch id, i.e. the number of views / difficulty preset): median APE 50%, bias -1%, dish median APE 51%.


#### catcv_v12only: category-disjoint 5-fold inside hard_v12 only (n = 145; removes batch offsets)

| model | n | med APE % | mean APE % | med bias % | dish med APE % | mass med APE % |
|---|---|---|---|---|---|---|
| baseline (V_vis) | 145 | 51 | 114 | -39 | 27 | 51 |
| global median ratio | 145 | 42 | 167 | -1 | 22 | 49 |
| per-kind median ratio | 145 | 46 | 82 | -1 | 24 | 50 |
| ridge | 145 | 30 | 52 | -3 | 14 | 34 |
| HGB (L2) | 145 | 44 | 97 | -10 | 25 | 43 |
| HGB (L1) | 145 | 29 | 118 | -4 | 14 | 31 |
| HGB direct log V | 145 | 44 | 102 | -7 | 31 | 44 |
| MLP | 145 | 36 | 68 | -7 | 16 | 41 |


HGB (L1) - baseline in hard_v12 only: d median APE -24.0 [-32.4, -10.6], d mean APE +3.9 [-14.6, +28.3].


#### Feature-set ablation (HGB (L1))

| feature set | protocol | #feat | med APE % | mean APE % | med bias % | dish med APE % |
|---|---|---|---|---|---|---|
| kind + lv_* only | catcv | 11 | 45 | 119 | -0 | 36 |
| kind + lv_* only | lobo_v12 | 11 | 43 | 122 | 13 | 10 |
| + geometry (point stats, footprint, heights, box) | catcv | 30 | 41 | 132 | -2 | 26 |
| + geometry (point stats, footprint, heights, box) | lobo_v12 | 30 | 41 | 161 | 9 | 21 |
| all features | catcv | 47 | 36 | 117 | 0 | 27 |
| all features | lobo_v12 | 47 | 41 | 152 | 19 | 28 |
| all minus views/lift (needs no SAM masks) | catcv | 44 | 38 | 124 | 1 | 31 |
| all minus views/lift (needs no SAM masks) | lobo_v12 | 44 | 43 | 182 | 17 | 28 |
| all minus scan context + container | catcv | 39 | 36 | 114 | -1 | 27 |
| all minus scan context + container | lobo_v12 | 39 | 40 | 149 | 16 | 21 |
| all + occluded_fraction (GT-only ORACLE, diagnostic) | catcv | 48 | 30 | 82 | -0 | 22 |
| all + occluded_fraction (GT-only ORACLE, diagnostic) | lobo_v12 | 48 | 33 | 123 | 13 | 23 |


#### Held-out permutation importance (HGB (L1); increase in mean |log error| when shuffled)

Baseline held-out mean |log error|: category-CV 0.645, lobo_v12 0.672.


| feature group | category CV | lobo_v12 |
|---|---|---|
| visible volumes | +0.161 | +0.056 |
| footprint | +0.079 | +0.108 |
| views / lift | +0.075 | -0.000 |
| contact / stacking | +0.054 | +0.017 |
| kind | +0.053 | +0.019 |
| point stats | +0.019 | +0.015 |
| scan context | +0.012 | -0.000 |
| heights | +0.007 | +0.004 |
| container | +0.006 | +0.000 |
| box | +0.000 | +0.008 |


Top single features (category CV, fold 0 split seed 300):

| feature | increase |
|---|---|
| r_hull_fs | +0.058 |
| lift_nvotes | +0.057 |
| convexity | +0.047 |
| r_voxel_fs | +0.041 |
| kind_piece | +0.032 |
| step_max | +0.030 |
| kind_leaves | +0.019 |
| step_mean | +0.013 |
| lv_foodscan | +0.013 |
| area_cm2 | +0.012 |
| n_pts | +0.007 |
| peak_ratio | +0.007 |


#### Learning curve: median item APE % vs fraction of training scans


lobo_v12:

| train fraction | ~train items | baseline | global ratio | HGB (L2) | HGB sd |
|---|---|---|---|---|---|
| 0.25 | 24 | 51 | 198 | 63 | 13.3 |
| 0.50 | 49 | 51 | 116 | 65 | 37.1 |
| 1.00 | 99 | 51 | 94 | 45 | 0.0 |


catcv:

| train fraction | ~train items | baseline | global ratio | HGB (L2) | HGB sd |
|---|---|---|---|---|---|
| 0.25 | 48 | 59 | 65 | 53 | 5.2 |
| 0.50 | 97 | 59 | 65 | 48 | 3.5 |
| 1.00 | 195 | 59 | 65 | 40 | 0.0 |


### Population: all (n = 261 items, 76 scans)


#### lobo_v12

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 152 | 53 | 113 | -40 | 28 | -26 | 51 | -38 |
| global median ratio | 152 | 127 | 422 | 127 | 183 | 183 | 134 | 134 |
| per-kind median ratio | 152 | 86 | 299 | 70 | 291 | 291 | 83 | 55 |
| ridge | 152 | 66 | 63 | -65 | 69 | -69 | 67 | -65 |
| HGB (L2) | 152 | 49 | 119 | 9 | 26 | 0 | 52 | 5 |
| HGB (L1) | 152 | 41 | 135 | 12 | 16 | 10 | 45 | 15 |
| HGB direct log V | 152 | 46 | 77 | -24 | 36 | -36 | 48 | -27 |
| MLP | 152 | 58 | 62 | -54 | 50 | -50 | 60 | -54 |


#### rev_v12

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 109 | 79 | 91 | -74 | 80 | -77 | 86 | -81 |
| global median ratio | 109 | 77 | 118 | -56 | 74 | -61 | 81 | -67 |
| per-kind median ratio | 109 | 83 | 76 | -65 | 85 | -75 | 77 | -71 |
| ridge | 109 | 61 | 122 | 4 | 37 | -13 | 66 | 29 |
| HGB (L2) | 109 | 78 | 121 | -58 | 70 | -69 | 84 | -65 |
| HGB (L1) | 109 | 78 | 135 | -53 | 61 | -57 | 87 | -59 |
| HGB direct log V | 109 | 77 | 138 | -53 | 68 | -63 | 84 | -58 |
| MLP | 109 | 72 | 122 | -31 | 62 | -53 | 75 | -32 |


#### lobo_all

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 261 | 61 | 104 | -52 | 60 | -56 | 65 | -52 |
| global median ratio | 261 | 95 | 304 | 62 | 89 | -21 | 95 | 55 |
| per-kind median ratio | 261 | 85 | 216 | 14 | 89 | -16 | 80 | 4 |
| ridge | 261 | 64 | 75 | -44 | 45 | -29 | 66 | -47 |
| HGB (L2) | 261 | 51 | 134 | 3 | 36 | -5 | 53 | 7 |
| HGB (L1) | 261 | 45 | 122 | 3 | 27 | -4 | 48 | 12 |
| HGB direct log V | 261 | 41 | 90 | -18 | 27 | -19 | 44 | -18 |
| MLP | 261 | 50 | 63 | -38 | 36 | -19 | 56 | -41 |


#### catcv

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 261 | 61 | 104 | -52 | 60 | -56 | 66 | -51 |
| global median ratio | 261 | 73 | 181 | 1 | 71 | -9 | 73 | 2 |
| per-kind median ratio | 261 | 73 | 128 | 1 | 75 | -6 | 70 | -0 |
| ridge | 261 | 43 | 83 | -2 | 34 | -14 | 47 | 10 |
| HGB (L2) | 261 | 42 | 121 | -0 | 29 | -10 | 48 | 8 |
| HGB (L1) | 261 | 37 | 121 | -1 | 26 | -5 | 44 | 6 |
| HGB direct log V | 261 | 41 | 98 | -2 | 28 | -15 | 46 | 6 |
| MLP | 261 | 40 | 80 | -1 | 26 | -9 | 45 | 7 |


Best learned model by category-CV median APE: **HGB (L1)**.


#### Paired cluster bootstrap (by scan), model - baseline APE, percentage points (95% CI)

| protocol | comparison | d median APE | d mean APE | P(better) % |
|---|---|---|---|---|
| lobo_v12 | HGB (L1) - baseline | -12.1 [-21.8, -2.0] | +22.7 [-1.2, +51.8] | 99 |
| lobo_v12 | HGB (L1) - global | -86.2 [-129.5, -51.0] | -286.1 [-487.5, -169.5] | 100 |
| lobo_v12 | global median ratio - baseline | +74.1 [+36.0, +121.0] | +308.8 [+184.8, +513.0] | 0 |
| rev_v12 | HGB (L1) - baseline | -1.0 [-15.9, +13.7] | +44.2 [-8.0, +107.8] | 67 |
| rev_v12 | HGB (L1) - global | +1.2 [-13.0, +12.2] | +17.3 [-41.2, +84.5] | 49 |
| rev_v12 | global median ratio - baseline | -2.2 [-8.0, +6.6] | +26.9 [+6.7, +51.8] | 75 |
| lobo_all | HGB (L1) - baseline | -16.5 [-29.9, -5.8] | +17.9 [-4.2, +44.3] | 100 |
| lobo_all | HGB (L1) - global | -50.6 [-64.8, -36.8] | -182.7 [-290.1, -98.9] | 100 |
| lobo_all | global median ratio - baseline | +34.2 [+16.5, +45.1] | +200.6 [+117.5, +301.7] | 0 |
| catcv | HGB (L1) - baseline | -24.2 [-32.0, -13.1] | +17.6 [-7.0, +48.8] | 100 |
| catcv | HGB (L1) - global | -36.4 [-41.7, -24.9] | -59.6 [-114.4, -17.3] | 100 |
| catcv | global median ratio - baseline | +12.2 [+1.8, +20.3] | +77.3 [+43.1, +123.5] | 2 |


##### lobo_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 33 | 77 / -35 | 148 / +148 | 72 / +25 |
| 10-50 mL | 79 | 51 / -44 | 113 / +113 | 39 / +21 |
| >50 mL | 40 | 40 / -39 | 132 / +132 | 34 / -9 |


##### lobo_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 80 | 52 / -45 | 108 / +108 | 38 / +14 |
| single | 53 | 48 / -40 | 127 / +127 | 35 / -7 |
| piece | 10 | 49 / -49 | 94 / +94 | 35 / +31 |
| topping | 4 | 265 / +265 | 1283 / +1283 | 687 / +687 |
| leaves | 5 | 1286 / +1286 | 5157 / +5157 | 1149 / +1149 |


##### lobo_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 54 | 52 / -33 | 156 / +156 | 50 / +39 |
| unstacked scans | 98 | 54 / -41 | 123 / +123 | 37 / -2 |


##### lobo_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 30 | 63 / +3 | 291 / +291 | 83 / +83 |
| 50-80% | 88 | 40 / -37 | 139 / +139 | 33 / +16 |
| >=80% | 34 | 74 / -74 | 81 / -2 | 41 / -35 |


##### catcv by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 59 | 64 / -35 | 78 / +35 | 111 / +111 |
| 10-50 mL | 106 | 54 / -47 | 64 / +12 | 32 / +3 |
| >50 mL | 96 | 71 / -69 | 78 / -36 | 31 / -23 |


##### catcv by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 100 | 52 / -51 | 52 / +3 | 34 / +3 |
| single | 77 | 65 / -53 | 73 / -0 | 39 / -0 |
| piece | 62 | 70 / -70 | 83 / -37 | 30 / -15 |
| topping | 8 | 265 / +265 | 655 / +655 | 648 / +648 |
| leaves | 14 | 136 / +129 | 382 / +382 | 416 / +416 |


##### catcv by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 124 | 68 / -53 | 78 / -2 | 40 / +2 |
| unstacked scans | 137 | 59 / -48 | 65 / +9 | 36 / -0 |


##### catcv by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 33 | 63 / +3 | 109 / +109 | 75 / +75 |
| 50-80% | 106 | 45 / -39 | 49 / +29 | 26 / +1 |
| >=80% | 90 | 87 / -87 | 78 / -74 | 50 / -32 |


##### rev_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 26 | 57 / -37 | 60 / +6 | 89 / +89 |
| 10-50 mL | 27 | 78 / -58 | 75 / -30 | 63 / -40 |
| >50 mL | 56 | 88 / -88 | 86 / -80 | 81 / -80 |


##### rev_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 20 | 53 / -53 | 60 / -22 | 88 / +12 |
| single | 24 | 84 / -84 | 76 / -73 | 68 / -65 |
| piece | 52 | 81 / -81 | 78 / -68 | 73 / -65 |
| topping | 4 | 470 / +470 | 853 / +853 | 588 / +588 |
| leaves | 9 | 81 / -28 | 68 / +21 | 85 / +41 |


##### rev_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 70 | 81 / -79 | 78 / -64 | 82 / -57 |
| unstacked scans | 39 | 75 / -66 | 72 / -44 | 54 / -26 |


##### rev_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 3 | 61 / -58 | 35 / -30 | 40 / -34 |
| 50-80% | 18 | 67 / -61 | 55 / -34 | 54 / -38 |
| >=80% | 56 | 92 / -92 | 87 / -87 | 85 / -85 |
