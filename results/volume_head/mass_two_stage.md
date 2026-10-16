# Two-stage mass (envelope target)

### Densities fitted on hard_v10 only (per kind)

| kind | n | bulk g/mL (mass/envelope) | material g/mL (mass/solid) | envelope/solid |
|---|---|---|---|---|
| _all | 51 | 0.69 | 0.90 | 1.20 |
| leaves | 4 | 0.01 | 0.15 | 12.98 |
| piece | 23 | 0.84 | 0.90 | 1.06 |
| pile | 12 | 0.54 | 0.90 | 1.93 |
| single | 8 | 0.42 | 1.02 | 2.08 |
| topping | 4 | 0.17 | 1.02 | 6.00 |


### Mass on hard_v12, lobo_v12 (batch-disjoint): hgb_l1 head, median APE % [95% scan-cluster CI]

| density path | volume | n items | n dishes | item APE % | item bias % | dish APE % |
|---|---|---|---|---|---|---|
| (a) V x bulk | none | 145 | 20 | 65 [60, 70] | -62 [-68, -56] | 57 [50, 63] |
| (a) V x bulk | hgb_l1 | 145 | 20 | 61 [51, 76] | +51 [+18, +70] | 32 [21, 68] |
| (a) V x bulk | oracle: V_envelope_gt | 145 | 20 | 32 [27, 41] | -19 [-25, -7] | 20 [14, 29] |
| (b) V / ratio x material | none | 145 | 20 | 66 [60, 72] | -65 [-70, -56] | 55 [46, 63] |
| (b) V / ratio x material | hgb_l1 | 145 | 20 | 60 [44, 70] | +43 [+9, +65] | 39 [19, 75] |
| (b) V / ratio x material | oracle: V_envelope_gt | 145 | 20 | 36 [31, 41] | -27 [-32, -13] | 19 [14, 22] |


Paired item-APE differences (pp): corrected - baseline per path, and path b - path a


| comparison | d median APE | d mean APE |
|---|---|---|
| (a) V x bulk | -3.9 [-14.2, +14.0] | +48.7 [+25.9, +78.8] |
| (b) V / ratio x material | -5.9 [-22.2, +7.4] | +42.1 [+21.0, +70.8] |
| path (b) - path (a), baseline volume | +1.4 [-0.9, +4.1] | +1.6 [-0.9, +3.8] |
| path (b) - path (a), corrected volume | -0.6 [-12.9, +6.2] | -5.0 [-12.0, +3.7] |


### Mass on hard_v12, catcv_v12 (category-disjoint, v12 only): hgb_l1 head, median APE % [95% scan-cluster CI]

| density path | volume | n items | n dishes | item APE % | item bias % | dish APE % |
|---|---|---|---|---|---|---|
| (a) V x bulk | none | 145 | 20 | 65 [60, 70] | -62 [-68, -56] | 57 [50, 63] |
| (a) V x bulk | hgb_l1 | 145 | 20 | 36 [31, 41] | -23 [-32, -11] | 24 [16, 33] |
| (a) V x bulk | oracle: V_envelope_gt | 145 | 20 | 32 [27, 41] | -19 [-25, -7] | 20 [14, 29] |
| (b) V / ratio x material | none | 145 | 20 | 66 [60, 72] | -65 [-70, -56] | 55 [46, 63] |
| (b) V / ratio x material | hgb_l1 | 145 | 20 | 41 [36, 45] | -27 [-35, -22] | 22 [18, 33] |
| (b) V / ratio x material | oracle: V_envelope_gt | 145 | 20 | 36 [31, 41] | -27 [-32, -13] | 19 [14, 22] |


Paired item-APE differences (pp): corrected - baseline per path, and path b - path a


| comparison | d median APE | d mean APE |
|---|---|---|
| (a) V x bulk | -28.4 [-33.8, -20.4] | -8.9 [-18.5, +6.0] |
| (b) V / ratio x material | -24.7 [-31.6, -17.4] | -6.1 [-16.9, +10.0] |
| path (b) - path (a), baseline volume | +1.4 [-0.9, +4.1] | +1.6 [-0.9, +3.8] |
| path (b) - path (a), corrected volume | +5.0 [+1.2, +7.5] | +4.4 [+1.5, +7.9] |
