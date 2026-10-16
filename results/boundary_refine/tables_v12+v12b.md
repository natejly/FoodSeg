| setting | inst mIoU | d inst (items, 95% CI) | d scan-mIoU (11 scans, 95% CI) | PQ | boundary IoU 3mm | corrected / lift errors | broken (% of points) | s / scan |
|---|---|---|---|---|---|---|---|---|
| lift (co-vote, reference) | 0.741 | - | - | 0.593 | 0.324 | 0.000 | 0.000 | 0.3 |
| all-scan-tuned {"band_mm": 6.0, "cand_mm": 12.0, "floor": 0.02, "k": 12, "radius_mm": 8.0, "s_c": 5.0, "s_n": 10.0, "s_z": 3.0, "sweeps": 3, "w": 1.0} | 0.754 | +0.013 [+0.005, +0.022] | +0.011 [+0.004, +0.018] | 0.642 | 0.335 | 0.019 | 0.048 | 1.3 |
| leave-one-scan-out | 0.749 | +0.009 [+0.002, +0.017] | +0.007 [+0.002, +0.014] | 0.639 | 0.329 | 0.017 | 0.052 | 1.3 |
| best at band 3 mm | 0.753 | +0.012 [+0.005, +0.021] | +0.010 [+0.004, +0.017] | 0.619 | 0.335 | 0.020 | 0.050 | 1.2 |
| best at band 6 mm | 0.754 | +0.013 [+0.005, +0.022] | +0.011 [+0.004, +0.018] | 0.642 | 0.335 | 0.019 | 0.048 | 1.3 |
| best at band 12 mm | 0.754 | +0.013 [+0.005, +0.021] | +0.011 [+0.004, +0.017] | 0.642 | 0.336 | 0.019 | 0.049 | 1.2 |

Oracle (snap lift-wrong points within band of a lift-label boundary to GT item; our band definition):

| band | oracle inst mIoU | gain | realised by best-at-band | realised by all-scan-tuned | realised by LOO |
|---|---|---|---|---|---|
| 3 mm | 0.810 | +0.069 | 17.2% | 18.8% | 12.8% |
| 6 mm | 0.864 | +0.123 | 10.5% | 10.5% | 7.2% |
| 12 mm | 0.912 | +0.171 | 7.5% | 7.6% | 5.2% |

LOO picks: hard_v12_snap/s000000: 6mm w=1.0 sc=5.0 sn=10.0 sz=3.0; hard_v12_snap/s000001: 6mm w=1.0 sc=5.0 sn=10.0 sz=3.0; hard_v12_snap/s000002: 6mm w=1.0 sc=5.0 sn=10.0 sz=3.0; hard_v12_snap/s000007: 6mm w=1.4 sc=5.0 sn=7.5 sz=3.0; hard_v12_snap/s000009: 6mm w=1.0 sc=5.0 sn=10.0 sz=3.0; hard_v12_snap/s000011: 6mm w=1.0 sc=5.0 sn=10.0 sz=3.0; hard_v12_snap/s000012: 6mm w=1.0 sc=5.0 sn=10.0 sz=3.0; hard_v12_snap/s000013: 6mm w=1.0 sc=5.0 sn=10.0 sz=3.0; hard_v12_snap/s000014: 3mm w=1.4 sc=5.0 sn=10.0 sz=4.5; hard_v12_snap/s000016: 6mm w=1.4 sc=5.0 sn=7.5 sz=3.0; hard_v12_snap/s000017: 6mm w=1.4 sc=5.0 sn=7.5 sz=3.0

Per-scan inst mIoU (lift -> all-scan-tuned -> LOO):

| scan | n vox | lift | tuned | LOO | oracle 6mm |
|---|---|---|---|---|---|
| hard_v12_snap/s000000 | 43460 | 0.689 | 0.688 | 0.688 | 0.784 |
| hard_v12_snap/s000001 | 50267 | 0.552 | 0.555 | 0.555 | 0.716 |
| hard_v12_snap/s000002 | 50714 | 0.804 | 0.827 | 0.827 | 0.911 |
| hard_v12_snap/s000007 | 49597 | 0.866 | 0.881 | 0.880 | 0.979 |
| hard_v12_snap/s000009 | 47905 | 0.968 | 0.968 | 0.968 | 0.998 |
| hard_v12_snap/s000011 | 53788 | 0.751 | 0.754 | 0.754 | 0.921 |
| hard_v12_snap/s000012 | 49542 | 0.916 | 0.910 | 0.910 | 0.985 |
| hard_v12_snap/s000013 | 48779 | 0.355 | 0.365 | 0.365 | 0.447 |
| hard_v12_snap/s000014 | 54201 | 0.750 | 0.779 | 0.746 | 0.864 |
| hard_v12_snap/s000016 | 37164 | 0.768 | 0.780 | 0.779 | 0.906 |
| hard_v12_snap/s000017 | 55657 | 0.702 | 0.732 | 0.731 | 0.978 |