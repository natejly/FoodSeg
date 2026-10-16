# Views-aware / mixed-density transfer (envelope target)

Items: 244 valid (67 in 8-view scans / 177 in 24-view scans; 42 / 32 scans). Target = envelope. Median item APE %, [paired scan-cluster bootstrap CI of (arm - baseline), pp] / median signed bias %.


#### scan-disjoint 5-fold (mixed train)

| test density | n | baseline | global median ratio | per-density median ratio (reference) | hgb_l1 / no views features | hgb_l1 / base (n_views, lift) | hgb_l1 / views-aware (+ scan n_views, per-view proxies) |
|---|---|---|---|---|---|---|---|
| 8-view | 67 | 92 / -92 | 81 [-18,-7] / -81 | 79 [-23,+3] / +1 | 45 [-60,-30] / -22 | 43 [-58,-36] / -5 | 45 [-58,-37] / -6 |
| 24-view | 177 | 51 / -51 | 55 [-7,+26] / +26 | 43 [-16,+8] / -2 | 18 [-38,-22] / +8 | 18 [-38,-23] / +4 | 18 [-38,-22] / +4 |


#### category-disjoint 5-fold (mixed train)

| test density | n | baseline | global median ratio | per-density median ratio (reference) | hgb_l1 / no views features | hgb_l1 / base (n_views, lift) | hgb_l1 / views-aware (+ scan n_views, per-view proxies) |
|---|---|---|---|---|---|---|---|
| 8-view | 67 | 92 / -92 | 81 [-16,-7] / -81 | 80 [-22,+3] / -4 | 45 [-60,-34] / -17 | 46 [-57,-38] / -16 | 47 [-54,-38] / -18 |
| 24-view | 177 | 51 / -51 | 54 [-8,+25] / +23 | 45 [-16,+8] / -2 | 20 [-37,-20] / +10 | 17 [-39,-22] / +6 | 17 [-38,-22] / +7 |


#### batch-disjoint (leave-one-batch-out)

| test density | n | baseline | global median ratio | per-density median ratio (reference) | hgb_l1 / no views features | hgb_l1 / base (n_views, lift) | hgb_l1 / views-aware (+ scan n_views, per-view proxies) |
|---|---|---|---|---|---|---|---|
| 8-view | 67 | 92 / -92 | 82 [-15,-6] / -82 | 79 [-23,+3] / +1 | 53 [-63,-25] / -30 | 41 [-67,-38] / -21 | 39 [-64,-40] / -23 |
| 24-view | 177 | 51 / -51 | 152 [+65,+145] / +152 | 40 [-20,+5] / -7 | 64 [-12,+38] / +62 | 64 [-10,+38] / +60 | 65 [-10,+35] / +64 |


Per held-out batch (median APE %):


| held-out batch | views | n | baseline | global median ratio | per-density median ratio (reference) | hgb_l1 / no views features | hgb_l1 / base (n_views, lift) | hgb_l1 / views-aware (+ scan n_views, per-view proxies) |
|---|---|---|---|---|---|---|---|---|
| hard | 24 | 17 | 53 | 67 | 67 | 24 | 8 | 10 |
| hard_v10 | 8 | 19 | 92 | 90 | 76 | 53 | 26 | 28 |
| hard_v11 | 8 | 30 | 92 | 82 | 81 | 51 | 49 | 49 |
| hard_v12 | 24 | 145 | 51 | 187 | 35 | 78 | 81 | 74 |
| hard_v3 | 24 | 15 | 29 | 90 | 73 | 25 | 9 | 19 |
| hard_v9 | 8 | 18 | 93 | 83 | 80 | 52 | 41 | 41 |


#### train 8-view only -> test 8-view (scan-disjoint)

| test density | n | baseline | global median ratio | per-density median ratio (reference) | hgb_l1 / no views features | hgb_l1 / base (n_views, lift) | hgb_l1 / views-aware (+ scan n_views, per-view proxies) |
|---|---|---|---|---|---|---|---|
| 8-view | 67 | 92 / -92 | 82 [-20,+3] / +6 | 82 [-20,+3] / +6 | 54 [-50,-20] / -6 | 55 [-50,-18] / +0 | 55 [-50,-18] / +0 |


#### train 24-view only -> test 24-view (scan-disjoint)

| test density | n | baseline | global median ratio | per-density median ratio (reference) | hgb_l1 / no views features | hgb_l1 / base (n_views, lift) | hgb_l1 / views-aware (+ scan n_views, per-view proxies) |
|---|---|---|---|---|---|---|---|
| 24-view | 177 | 51 / -51 | 45 [-16,+8] / +1 | 45 [-16,+8] / +1 | 17 [-39,-24] / +3 | 18 [-38,-23] / +2 | 18 [-38,-23] / +2 |


#### train 8-view only -> test 24-view

| test density | n | baseline | global median ratio | per-density median ratio (reference) | hgb_l1 / no views features | hgb_l1 / base (n_views, lift) | hgb_l1 / views-aware (+ scan n_views, per-view proxies) |
|---|---|---|---|---|---|---|---|
| 24-view | 177 | 51 / -51 | 545 [+409,+625] / +545 | 545 [+409,+625] / +545 | 244 [+115,+227] / +244 | 190 [+102,+197] / +190 | 190 [+102,+197] / +190 |


#### train 24-view only -> test 8-view

| test density | n | baseline | global median ratio | per-density median ratio (reference) | hgb_l1 / no views features | hgb_l1 / base (n_views, lift) | hgb_l1 / views-aware (+ scan n_views, per-view proxies) |
|---|---|---|---|---|---|---|---|
| 8-view | 67 | 92 / -92 | 85 [-11,-4] / -84 | 85 [-11,-4] / -84 | 77 [-24,-8] / -73 | 76 [-26,-8] / -71 | 76 [-26,-8] / -71 |
