# target = envelope

## Population: valid (n = 244 items, 74 scans)


### lobo_v12

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 145 | 51 | 50 | -51 | 38 | -38 | 58 | -52 |
| global median ratio | 145 | 187 | 205 | 187 | 264 | 264 | 185 | 185 |
| per-kind median ratio | 145 | 159 | 299 | 159 | 382 | 382 | 154 | 154 |
| ridge | 145 | 78 | 79 | -77 | 77 | -77 | 80 | -77 |
| HGB (L2) | 145 | 67 | 109 | 67 | 61 | 61 | 79 | 76 |
| HGB (L1) | 145 | 81 | 97 | 81 | 68 | 68 | 78 | 72 |
| HGB direct log V | 145 | 49 | 104 | 31 | 33 | 21 | 65 | 48 |
| MLP | 145 | 55 | 54 | -55 | 41 | -41 | 63 | -58 |


### rev_v12

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 99 | 83 | 71 | -83 | 86 | -86 | 87 | -81 |
| global median ratio | 99 | 81 | 70 | -65 | 81 | -71 | 79 | -61 |
| per-kind median ratio | 99 | 80 | 71 | -71 | 81 | -74 | 78 | -58 |
| ridge | 99 | 52 | 56 | -27 | 53 | -45 | 71 | -30 |
| HGB (L2) | 99 | 61 | 58 | -50 | 64 | -56 | 80 | -51 |
| HGB (L1) | 99 | 58 | 56 | -44 | 59 | -59 | 77 | -40 |
| HGB direct log V | 99 | 58 | 59 | -34 | 54 | -54 | 82 | -39 |
| MLP | 99 | 70 | 66 | -43 | 68 | -68 | 82 | -46 |


### lobo_all

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 244 | 60 | 59 | -60 | 70 | -70 | 69 | -64 |
| global median ratio | 244 | 98 | 153 | 87 | 93 | -25 | 97 | 67 |
| per-kind median ratio | 244 | 95 | 214 | 66 | 92 | -21 | 94 | 64 |
| ridge | 244 | 74 | 74 | -65 | 61 | -16 | 74 | -62 |
| HGB (L2) | 244 | 48 | 87 | 37 | 35 | 10 | 62 | 48 |
| HGB (L1) | 244 | 55 | 74 | 29 | 34 | 4 | 61 | 38 |
| HGB direct log V | 244 | 42 | 78 | 15 | 27 | -2 | 58 | 22 |
| MLP | 244 | 50 | 51 | -38 | 25 | -11 | 57 | -38 |


### catcv

| model | n | med APE % | mean APE % | med bias % | dish med APE % | dish bias % | mass med APE % | mass bias % |
|---|---|---|---|---|---|---|---|---|
| baseline (V_vis) | 244 | 60 | 59 | -60 | 70 | -70 | 70 | -59 |
| global median ratio | 244 | 63 | 64 | 1 | 72 | -25 | 66 | 5 |
| per-kind median ratio | 244 | 63 | 80 | 3 | 75 | -7 | 66 | 4 |
| ridge | 244 | 33 | 63 | 3 | 27 | -5 | 49 | 15 |
| HGB (L2) | 244 | 24 | 57 | 3 | 20 | 3 | 44 | 12 |
| HGB (L1) | 244 | 23 | 41 | 2 | 19 | 1 | 42 | 5 |
| HGB direct log V | 244 | 27 | 55 | 4 | 20 | -2 | 45 | 18 |
| MLP | 244 | 24 | 52 | 3 | 17 | 2 | 43 | 9 |


Best learned model by category-CV median APE: **HGB (L1)**.


### Paired cluster bootstrap (by scan), model - baseline APE, percentage points (95% CI)

| protocol | comparison | d median APE | d mean APE | P(better) % |
|---|---|---|---|---|
| lobo_v12 | HGB (L1) - baseline | +29.2 [-1.3, +60.5] | +47.1 [+20.6, +72.2] | 3 |
| lobo_v12 | HGB (L1) - global | -105.9 [-175.6, -67.3] | -107.9 [-152.4, -77.2] | 100 |
| lobo_v12 | global median ratio - baseline | +135.1 [+95.1, +203.6] | +155.0 [+117.1, +204.2] | 0 |
| rev_v12 | HGB (L1) - baseline | -24.9 [-39.4, -14.0] | -14.7 [-21.9, -7.4] | 100 |
| rev_v12 | HGB (L1) - global | -23.0 [-40.1, -9.1] | -14.1 [-22.6, -6.1] | 100 |
| rev_v12 | global median ratio - baseline | -1.9 [-10.5, +8.4] | -0.6 [-10.6, +9.7] | 71 |
| lobo_all | HGB (L1) - baseline | -5.2 [-26.3, +13.7] | +15.8 [-3.9, +33.2] | 69 |
| lobo_all | HGB (L1) - global | -42.5 [-78.0, -30.1] | -78.6 [-100.8, -58.2] | 100 |
| lobo_all | global median ratio - baseline | +37.4 [+20.8, +87.2] | +94.4 [+65.8, +123.7] | 0 |
| catcv | HGB (L1) - baseline | -36.8 [-43.3, -28.6] | -17.6 [-25.7, -5.4] | 100 |
| catcv | HGB (L1) - global | -40.4 [-51.5, -28.2] | -23.6 [-34.4, -10.7] | 100 |
| catcv | global median ratio - baseline | +3.6 [-10.0, +17.1] | +5.9 [-3.5, +17.5] | 36 |


#### lobo_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 22 | 48 / -48 | 208 / +208 | 47 / +34 |
| 10-50 mL | 68 | 55 / -55 | 165 / +165 | 94 / +94 |
| >50 mL | 55 | 41 / -41 | 246 / +246 | 60 / +58 |


#### lobo_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 74 | 54 / -54 | 172 / +172 | 91 / +91 |
| single | 52 | 53 / -53 | 176 / +176 | 59 / +54 |
| piece | 10 | 55 / -55 | 167 / +167 | 63 / +63 |
| topping | 4 | 21 / -11 | 426 / +426 | 216 / +216 |
| leaves | 5 | 21 / -21 | 368 / +368 | 165 / +165 |


#### lobo_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 52 | 51 / -51 | 187 / +187 | 93 / +93 |
| unstacked scans | 93 | 51 / -51 | 187 / +187 | 67 / +65 |


#### lobo_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 30 | 28 / -28 | 326 / +326 | 89 / +89 |
| 50-80% | 87 | 45 / -45 | 224 / +224 | 89 / +89 |
| >=80% | 28 | 81 / -81 | 65 / +14 | 47 / +20 |


#### catcv by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 34 | 59 / -59 | 53 / +4 | 48 / +42 |
| 10-50 mL | 89 | 56 / -56 | 45 / +13 | 20 / +7 |
| >50 mL | 121 | 72 / -72 | 78 / -29 | 21 / -2 |


#### catcv by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 92 | 58 / -58 | 43 / +10 | 21 / +6 |
| single | 72 | 60 / -60 | 68 / +1 | 21 / +3 |
| piece | 59 | 69 / -69 | 85 / -24 | 26 / +3 |
| topping | 7 | 16 / -6 | 136 / +136 | 21 / +21 |
| leaves | 14 | 80 / -80 | 94 / -50 | 44 / +11 |


#### catcv by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 114 | 65 / -65 | 71 / -9 | 22 / -0 |
| unstacked scans | 130 | 58 / -58 | 55 / +5 | 25 / +8 |


#### catcv by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 33 | 31 / -31 | 75 / +75 | 30 / +21 |
| 50-80% | 104 | 52 / -52 | 46 / +21 | 16 / +7 |
| >=80% | 75 | 91 / -91 | 78 / -78 | 42 / -23 |


#### rev_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 12 | 63 / -63 | 34 / -24 | 56 / +29 |
| 10-50 mL | 21 | 59 / -59 | 72 / -15 | 34 / -11 |
| >50 mL | 66 | 92 / -92 | 87 / -84 | 73 / -70 |


#### rev_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 18 | 80 / -80 | 58 / -58 | 49 / -31 |
| single | 20 | 88 / -88 | 75 / -75 | 63 / -51 |
| piece | 49 | 81 / -81 | 88 / -60 | 59 / -50 |
| topping | 3 | 6 / -6 | 93 / +93 | 14 / +10 |
| leaves | 9 | 98 / -98 | 95 / -95 | 86 / -86 |


#### rev_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 62 | 87 / -87 | 82 / -74 | 65 / -52 |
| unstacked scans | 37 | 79 / -79 | 79 / -57 | 48 / -30 |


#### rev_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 3 | 66 / -66 | 29 / -29 | 43 / -43 |
| 50-80% | 17 | 69 / -69 | 49 / -36 | 35 / -17 |
| >=80% | 47 | 95 / -95 | 89 / -89 | 83 / -83 |


### Per-batch structure (valid items)

| batch | n items | median views seen | median V_true/V_vis | baseline med APE % |
|---|---|---|---|---|
| hard | 17 | 24 | 2.15 | 53 |
| hard_v10 | 19 | 8 | 13.07 | 92 |
| hard_v11 | 30 | 8 | 12.66 | 92 |
| hard_v12 | 145 | 24 | 2.06 | 51 |
| hard_v3 | 15 | 24 | 1.40 | 29 |
| hard_v9 | 18 | 8 | 14.39 | 93 |


Diagnostic, category CV with a per-batch median ratio (uses the batch id, i.e. the number of views / difficulty preset): median APE 51%, bias -1%, dish median APE 50%.


### catcv_v12only: category-disjoint 5-fold inside hard_v12 only (n = 145; removes batch offsets)

| model | n | med APE % | mean APE % | med bias % | dish med APE % | mass med APE % |
|---|---|---|---|---|---|---|
| baseline (V_vis) | 145 | 51 | 50 | -51 | 38 | 60 |
| global median ratio | 145 | 41 | 45 | 1 | 28 | 49 |
| per-kind median ratio | 145 | 42 | 46 | 1 | 29 | 51 |
| ridge | 145 | 24 | 34 | 5 | 10 | 38 |
| HGB (L2) | 145 | 19 | 28 | 3 | 9 | 32 |
| HGB (L1) | 145 | 19 | 26 | 3 | 8 | 32 |
| HGB direct log V | 145 | 24 | 48 | 2 | 10 | 35 |
| MLP | 145 | 23 | 39 | 5 | 10 | 36 |


HGB (L1) - baseline in hard_v12 only: d median APE -32.8 [-39.6, -22.9], d mean APE -24.4 [-29.8, -17.3].


### Feature-set ablation (HGB (L1))

| feature set | protocol | #feat | med APE % | mean APE % | med bias % | dish med APE % |
|---|---|---|---|---|---|---|
| kind + lv_* only | catcv | 11 | 34 | 54 | 4 | 30 |
| kind + lv_* only | lobo_v12 | 11 | 53 | 112 | 53 | 29 |
| + geometry (point stats, footprint, heights, box) | catcv | 30 | 26 | 46 | 3 | 22 |
| + geometry (point stats, footprint, heights, box) | lobo_v12 | 30 | 54 | 121 | 53 | 38 |
| all features | catcv | 47 | 23 | 41 | 2 | 19 |
| all features | lobo_v12 | 47 | 81 | 97 | 81 | 68 |
| all minus views/lift (needs no SAM masks) | catcv | 44 | 24 | 46 | 3 | 23 |
| all minus views/lift (needs no SAM masks) | lobo_v12 | 44 | 78 | 132 | 78 | 51 |
| all minus scan context + container | catcv | 39 | 24 | 39 | 3 | 20 |
| all minus scan context + container | lobo_v12 | 39 | 60 | 80 | 59 | 49 |
| all + occluded_fraction (GT-only ORACLE, diagnostic) | catcv | 48 | 21 | 33 | 3 | 16 |
| all + occluded_fraction (GT-only ORACLE, diagnostic) | lobo_v12 | 48 | 51 | 62 | 51 | 49 |


### Held-out permutation importance (HGB (L1); increase in mean |log error| when shuffled)

Baseline held-out mean |log error|: category-CV 0.425, lobo_v12 0.630.


| feature group | category CV | lobo_v12 |
|---|---|---|
| visible volumes | +0.213 | +0.113 |
| footprint | +0.124 | +0.065 |
| views / lift | +0.118 | +0.008 |
| scan context | +0.043 | -0.001 |
| point stats | +0.027 | +0.013 |
| contact / stacking | +0.021 | +0.003 |
| heights | +0.021 | +0.014 |
| box | +0.012 | +0.003 |
| container | +0.003 | +0.011 |
| kind | +0.000 | +0.000 |


Top single features (category CV, fold 0 split seed 300):

| feature | increase |
|---|---|
| r_voxel_fs | +0.114 |
| convexity | +0.100 |
| lift_nvotes | +0.082 |
| r_hull_fs | +0.045 |
| n_pts | +0.025 |
| lv_foodscan | +0.019 |
| fullness | +0.018 |
| area_cm2 | +0.016 |
| step_max | +0.014 |
| scan_lv_total | +0.013 |
| height_to_width | +0.013 |
| peak_ratio | +0.009 |


### Learning curve: median item APE % vs fraction of training scans


lobo_v12:

| train fraction | ~train items | baseline | global ratio | HGB (L2) | HGB sd |
|---|---|---|---|---|---|
| 0.25 | 24 | 51 | 368 | 83 | 65.3 |
| 0.50 | 49 | 51 | 177 | 67 | 32.3 |
| 1.00 | 99 | 51 | 187 | 67 | 0.0 |


catcv:

| train fraction | ~train items | baseline | global ratio | HGB (L2) | HGB sd |
|---|---|---|---|---|---|
| 0.25 | 48 | 60 | 67 | 34 | 3.4 |
| 0.50 | 97 | 60 | 65 | 27 | 2.9 |
| 1.00 | 195 | 60 | 63 | 24 | 0.0 |
