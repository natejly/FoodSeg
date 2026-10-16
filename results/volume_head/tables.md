
## Population: valid (n = 244 items, 74 scans)


### lobo_v12

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


### rev_v12

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


### lobo_all

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


### catcv

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


### Paired cluster bootstrap (by scan), model - baseline APE, percentage points (95% CI)

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


#### lobo_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 31 | 69 / -23 | 144 / +144 | 82 / +34 |
| 10-50 mL | 75 | 48 / -42 | 86 / +83 | 48 / +35 |
| >50 mL | 39 | 40 / -38 | 96 / +96 | 29 / -2 |


#### lobo_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 74 | 47 / -41 | 85 / +85 | 39 / +32 |
| single | 52 | 46 / -40 | 93 / +90 | 32 / +0 |
| piece | 10 | 49 / -49 | 61 / +61 | 44 / +44 |
| topping | 4 | 265 / +265 | 1048 / +1048 | 821 / +821 |
| leaves | 5 | 1286 / +1286 | 4264 / +4264 | 1188 / +1188 |


#### lobo_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 52 | 50 / -30 | 119 / +119 | 57 / +49 |
| unstacked scans | 93 | 52 / -40 | 92 / +89 | 34 / +4 |


#### lobo_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 30 | 63 / +3 | 224 / +224 | 62 / +62 |
| 50-80% | 87 | 39 / -37 | 100 / +100 | 41 / +31 |
| >=80% | 28 | 67 / -67 | 59 / +5 | 33 / -19 |


#### catcv by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 55 | 63 / -28 | 73 / +39 | 82 / +60 |
| 10-50 mL | 98 | 52 / -43 | 54 / +11 | 27 / +2 |
| >50 mL | 91 | 68 / -66 | 75 / -39 | 30 / -19 |


#### catcv by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 92 | 51 / -46 | 44 / +6 | 26 / +1 |
| single | 72 | 60 / -46 | 70 / +6 | 39 / -2 |
| piece | 59 | 68 / -68 | 79 / -39 | 27 / -11 |
| topping | 7 | 353 / +353 | 814 / +814 | 462 / +462 |
| leaves | 14 | 136 / +129 | 339 / +339 | 562 / +562 |


#### catcv by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 114 | 61 / -51 | 75 / -3 | 32 / -0 |
| unstacked scans | 130 | 56 / -43 | 55 / +9 | 35 / -2 |


#### catcv by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 33 | 63 / +3 | 101 / +101 | 58 / +52 |
| 50-80% | 104 | 44 / -39 | 44 / +18 | 26 / -1 |
| >=80% | 75 | 86 / -86 | 75 / -74 | 49 / -28 |


#### rev_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 24 | 53 / -37 | 56 / +3 | 64 / +52 |
| 10-50 mL | 23 | 58 / -53 | 69 / -24 | 52 / -15 |
| >50 mL | 52 | 87 / -87 | 84 / -78 | 78 / -74 |


#### rev_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 18 | 53 / -53 | 56 / -23 | 78 / +1 |
| single | 20 | 80 / -80 | 73 / -68 | 61 / -56 |
| piece | 49 | 74 / -74 | 72 / -58 | 62 / -56 |
| topping | 3 | 881 / +881 | 1496 / +1496 | 450 / +450 |
| leaves | 9 | 81 / -28 | 69 / +18 | 66 / +35 |


#### rev_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 62 | 76 / -74 | 72 / -57 | 75 / -47 |
| unstacked scans | 37 | 72 / -61 | 69 / -37 | 52 / -15 |


#### rev_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 3 | 61 / -58 | 37 / -32 | 33 / -31 |
| 50-80% | 17 | 67 / -61 | 54 / -37 | 47 / -41 |
| >=80% | 47 | 91 / -91 | 86 / -86 | 81 / -81 |


### Per-batch structure (valid items)

| batch | n items | median views seen | median V_true/V_vis | baseline med APE % |
|---|---|---|---|---|
| hard | 17 | 24 | 1.17 | 26 |
| hard_v10 | 19 | 8 | 5.66 | 87 |
| hard_v11 | 30 | 8 | 7.14 | 86 |
| hard_v12 | 145 | 24 | 1.63 | 51 |
| hard_v3 | 15 | 24 | 1.04 | 39 |
| hard_v9 | 18 | 8 | 5.19 | 87 |


Diagnostic, category CV with a per-batch median ratio (uses the batch id, i.e. the number of views / difficulty preset): median APE 50%, bias -1%, dish median APE 51%.


### catcv_v12only: category-disjoint 5-fold inside hard_v12 only (n = 145; removes batch offsets)

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


### Feature-set ablation (HGB (L1))

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


### Held-out permutation importance (HGB (L1); increase in mean |log error| when shuffled)

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


### Learning curve: median item APE % vs fraction of training scans


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


## Population: all (n = 261 items, 76 scans)


### lobo_v12

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


### rev_v12

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


### lobo_all

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


### catcv

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


### Paired cluster bootstrap (by scan), model - baseline APE, percentage points (95% CI)

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


#### lobo_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 33 | 77 / -35 | 148 / +148 | 72 / +25 |
| 10-50 mL | 79 | 51 / -44 | 113 / +113 | 39 / +21 |
| >50 mL | 40 | 40 / -39 | 132 / +132 | 34 / -9 |


#### lobo_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 80 | 52 / -45 | 108 / +108 | 38 / +14 |
| single | 53 | 48 / -40 | 127 / +127 | 35 / -7 |
| piece | 10 | 49 / -49 | 94 / +94 | 35 / +31 |
| topping | 4 | 265 / +265 | 1283 / +1283 | 687 / +687 |
| leaves | 5 | 1286 / +1286 | 5157 / +5157 | 1149 / +1149 |


#### lobo_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 54 | 52 / -33 | 156 / +156 | 50 / +39 |
| unstacked scans | 98 | 54 / -41 | 123 / +123 | 37 / -2 |


#### lobo_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 30 | 63 / +3 | 291 / +291 | 83 / +83 |
| 50-80% | 88 | 40 / -37 | 139 / +139 | 33 / +16 |
| >=80% | 34 | 74 / -74 | 81 / -2 | 41 / -35 |


#### catcv by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 59 | 64 / -35 | 78 / +35 | 111 / +111 |
| 10-50 mL | 106 | 54 / -47 | 64 / +12 | 32 / +3 |
| >50 mL | 96 | 71 / -69 | 78 / -36 | 31 / -23 |


#### catcv by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 100 | 52 / -51 | 52 / +3 | 34 / +3 |
| single | 77 | 65 / -53 | 73 / -0 | 39 / -0 |
| piece | 62 | 70 / -70 | 83 / -37 | 30 / -15 |
| topping | 8 | 265 / +265 | 655 / +655 | 648 / +648 |
| leaves | 14 | 136 / +129 | 382 / +382 | 416 / +416 |


#### catcv by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 124 | 68 / -53 | 78 / -2 | 40 / +2 |
| unstacked scans | 137 | 59 / -48 | 65 / +9 | 36 / -0 |


#### catcv by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 33 | 63 / +3 | 109 / +109 | 75 / +75 |
| 50-80% | 106 | 45 / -39 | 49 / +29 | 26 / +1 |
| >=80% | 90 | 87 / -87 | 78 / -74 | 50 / -32 |


#### rev_v12 by size: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| <10 mL | 26 | 57 / -37 | 60 / +6 | 89 / +89 |
| 10-50 mL | 27 | 78 / -58 | 75 / -30 | 63 / -40 |
| >50 mL | 56 | 88 / -88 | 86 / -80 | 81 / -80 |


#### rev_v12 by kind: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| pile | 20 | 53 / -53 | 60 / -22 | 88 / +12 |
| single | 24 | 84 / -84 | 76 / -73 | 68 / -65 |
| piece | 52 | 81 / -81 | 78 / -68 | 73 / -65 |
| topping | 4 | 470 / +470 | 853 / +853 | 588 / +588 |
| leaves | 9 | 81 / -28 | 68 / +21 | 85 / +41 |


#### rev_v12 by stacked: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| stacked scans | 70 | 81 / -79 | 78 / -64 | 82 / -57 |
| unstacked scans | 39 | 75 / -66 | 72 / -44 | 54 / -26 |


#### rev_v12 by occl: median APE % / median bias %

| group | n | baseline | global ratio | HGB (L1) |
|---|---|---|---|---|
| occluded <50% | 3 | 61 / -58 | 35 / -30 | 40 / -34 |
| 50-80% | 18 | 67 / -61 | 55 / -34 | 54 / -38 |
| >=80% | 56 | 92 / -92 | 87 / -87 | 85 / -85 |
