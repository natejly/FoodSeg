# Envelope vs solid target, baseline vs head (paired scan-cluster bootstrap)

| target | protocol | n items | item median APE %: base -> head (diff pp [95% CI]) | item mean APE % | median bias % | dish median APE %: base -> head (diff pp [95% CI]) |
|---|---|---|---|---|---|---|
| envelope | lobo_v12 (train 5 batches -> v12) | 145 | 51 -> 81 (+29 [-1, +61]) | 50 -> 97 (+47 [+21, +72]) | -51 -> +81 | 38 -> 68 (+29 [+2, +69]) |
| envelope | rev_v12 (train v12 -> 5 batches) | 99 | 83 -> 58 (-25 [-39, -14]) | 71 -> 56 (-15 [-22, -7]) | -83 -> -44 | 86 -> 59 (-27 [-51, -17]) |
| envelope | lobo_all (leave-one-batch-out) | 244 | 60 -> 55 (-5 [-26, +14]) | 59 -> 74 (+16 [-4, +33]) | -60 -> +29 | 70 -> 34 (-36 [-50, -11]) |
| envelope | catcv all batches | 244 | 60 -> 23 (-37 [-43, -29]) | 59 -> 41 (-18 [-26, -5]) | -60 -> +4 | 70 -> 21 (-48 [-58, -34]) |
| envelope | catcv hard_v12 only | 145 | 51 -> 19 (-33 [-40, -23]) | 50 -> 26 (-24 [-30, -17]) | -51 -> +4 | 38 -> 8 (-31 [-36, -25]) |
| solid | lobo_v12 (train 5 batches -> v12) | 145 | 51 -> 41 (-10 [-21, +6]) | 114 -> 152 (+38 [+10, +80]) | -39 -> +19 | 27 -> 28 (+1 [-20, +23]) |
| solid | rev_v12 (train v12 -> 5 batches) | 99 | 74 -> 66 (-8 [-20, +11]) | 91 -> 104 (+13 [-22, +51]) | -68 -> -41 | 74 -> 55 (-19 [-25, -8]) |
| solid | lobo_all (leave-one-batch-out) | 244 | 59 -> 42 (-16 [-26, -5]) | 104 -> 122 (+18 [-2, +39]) | -48 -> +6 | 58 -> 34 (-25 [-41, -7]) |
| solid | catcv all batches | 244 | 59 -> 34 (-24 [-32, -16]) | 104 -> 117 (+12 [-11, +46]) | -48 -> -1 | 58 -> 29 (-30 [-45, -13]) |
| solid | catcv hard_v12 only | 145 | 51 -> 27 (-24 [-32, -11]) | 114 -> 118 (+4 [-15, +28]) | -39 -> -4 | 27 -> 13 (-14 [-21, -1]) |
