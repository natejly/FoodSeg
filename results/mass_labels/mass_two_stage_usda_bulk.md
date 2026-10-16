# Two-stage mass (envelope target, mass source = usda_bulk)

### Densities fitted on hard_v10 only (per kind)

| kind | n | bulk g/mL (mass/envelope) | material g/mL (mass/solid) | envelope/solid |
|---|---|---|---|---|
| _all | 51 | 0.78 | 1.06 | 1.20 |
| leaves | 4 | 0.13 | 1.65 | 12.98 |
| piece | 23 | 0.85 | 0.92 | 1.06 |
| pile | 12 | 0.63 | 1.21 | 1.93 |
| single | 8 | 0.73 | 1.31 | 2.08 |
| topping | 4 | 1.03 | 6.18 | 6.00 |


### Mass on hard_v12, lobo_v12 (batch-disjoint): hgb_l1 head, median APE % [95% scan-cluster CI]

| density path | volume | n items | n dishes | item APE % | item bias % | dish APE % |
|---|---|---|---|---|---|---|
| (a) V x bulk | none | 145 | 20 | 53 [44, 64] | -51 [-64, -40] | 43 [34, 51] |
| (a) V x bulk | hgb_l1 | 145 | 20 | 83 [68, 104] | +83 [+65, +104] | 83 [48, 97] |
| (a) V x bulk | oracle: V_envelope_gt | 145 | 20 | 18 [11, 29] | +3 [-7, +10] | 14 [5, 26] |
| (b) V / ratio x material | none | 145 | 20 | 54 [46, 64] | -54 [-64, -45] | 50 [38, 54] |
| (b) V / ratio x material | hgb_l1 | 145 | 20 | 76 [63, 94] | +76 [+54, +94] | 65 [30, 82] |
| (b) V / ratio x material | oracle: V_envelope_gt | 145 | 20 | 19 [7, 22] | -2 [-16, +3] | 17 [13, 23] |


Paired item-APE differences (pp): corrected - baseline per path, and path b - path a


| comparison | d median APE | d mean APE |
|---|---|---|
| (a) V x bulk | +29.3 [+7.6, +56.6] | +49.6 [+31.6, +68.9] |
| (b) V / ratio x material | +21.6 [+3.4, +43.4] | +42.6 [+23.9, +61.6] |
| path (b) - path (a), baseline volume | +0.9 [-1.7, +5.0] | +1.0 [+0.1, +2.1] |
| path (b) - path (a), corrected volume | -6.9 [-20.9, -0.2] | -6.0 [-9.2, -3.8] |


### Mass on hard_v12, catcv_v12 (category-disjoint, v12 only): hgb_l1 head, median APE % [95% scan-cluster CI]

| density path | volume | n items | n dishes | item APE % | item bias % | dish APE % |
|---|---|---|---|---|---|---|
| (a) V x bulk | none | 145 | 20 | 53 [44, 64] | -51 [-64, -40] | 43 [34, 51] |
| (a) V x bulk | hgb_l1 | 145 | 20 | 25 [21, 29] | -1 [-15, +6] | 13 [7, 25] |
| (a) V x bulk | oracle: V_envelope_gt | 145 | 20 | 18 [11, 29] | +3 [-7, +10] | 14 [5, 26] |
| (b) V / ratio x material | none | 145 | 20 | 54 [46, 64] | -54 [-64, -45] | 50 [38, 54] |
| (b) V / ratio x material | hgb_l1 | 145 | 20 | 24 [21, 31] | -5 [-17, +4] | 16 [10, 24] |
| (b) V / ratio x material | oracle: V_envelope_gt | 145 | 20 | 19 [7, 22] | -2 [-16, +3] | 17 [13, 23] |


Paired item-APE differences (pp): corrected - baseline per path, and path b - path a


| comparison | d median APE | d mean APE |
|---|---|---|
| (a) V x bulk | -28.3 [-40.4, -18.6] | -16.4 [-24.4, -5.5] |
| (b) V / ratio x material | -30.5 [-41.7, -21.0] | -18.8 [-26.0, -8.2] |
| path (b) - path (a), baseline volume | +0.9 [-1.7, +5.0] | +1.0 [+0.1, +2.1] |
| path (b) - path (a), corrected volume | -1.3 [-3.8, +2.6] | -1.4 [-3.6, +0.2] |
