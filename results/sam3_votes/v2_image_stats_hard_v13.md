# SAM 3 real votes vs pseudo votes (v2 image, hard_v13, 20 scans)

20 scans.

| scan | items | wrong real | wrong pseudo | share real | share pseudo | mIoU real | mIoU pseudo | merges r/p | splits r/p | unvoted real |
|---|---|---|---|---|---|---|---|---|---|---|
| s000000 | 4 | 0.153 | 0.388 | 0.898 | 0.874 | 0.700 | 0.510 | 0/1 | 1/1 | 0.007 |
| s000001 | 2 | 0.075 | 0.447 | 0.914 | 0.613 | 0.813 | 0.374 | 0/1 | 0/1 | 0.006 |
| s000002 | 14 | 0.109 | 0.070 | 0.884 | 0.892 | 0.786 | 0.830 | 0/0 | 2/0 | 0.008 |
| s000003 | 2 | 0.115 | 0.087 | 0.862 | 0.891 | 0.796 | 0.821 | 0/0 | 0/0 | 0.012 |
| s000004 | 15 | 0.393 | 0.168 | 0.703 | 0.818 | 0.386 | 0.721 | 2/0 | 5/0 | 0.013 |
| s000005 | 5 | 0.350 | 0.174 | 0.813 | 0.831 | 0.542 | 0.705 | 1/1 | 1/1 | 0.001 |
| s000006 | 7 | 0.162 | 0.139 | 0.896 | 0.935 | 0.791 | 0.750 | 0/0 | 0/0 | 0.010 |
| s000007 | 3 | 0.274 | 0.155 | 0.740 | 0.788 | 0.438 | 0.552 | 1/1 | 0/1 | 0.134 |
| s000008 | 2 | 0.063 | 0.062 | 0.941 | 0.941 | 0.907 | 0.906 | 0/0 | 0/0 | 0.007 |
| s000009 | 6 | 0.111 | 0.058 | 0.883 | 0.929 | 0.775 | 0.907 | 1/0 | 2/0 | 0.011 |
| s000010 | 7 | 0.322 | 0.317 | 0.828 | 0.772 | 0.294 | 0.349 | 1/2 | 2/3 | 0.007 |
| s000011 | 1 | 0.060 | 0.045 | 0.939 | 0.911 | 0.919 | 0.925 | 0/0 | 0/0 | 0.001 |
| s000012 | 5 | 0.043 | 0.045 | 0.967 | 0.943 | 0.935 | 0.923 | 0/0 | 0/0 | 0.001 |
| s000013 | 9 | 0.387 | 0.150 | 0.872 | 0.885 | 0.573 | 0.642 | 1/0 | 1/0 | 0.003 |
| s000014 | 5 | 0.041 | 0.048 | 0.964 | 0.918 | 0.940 | 0.928 | 0/0 | 0/0 | 0.005 |
| s000015 | 6 | 0.533 | 0.340 | 0.795 | 0.838 | 0.259 | 0.309 | 2/3 | 3/3 | 0.002 |
| s000016 | 1 | 0.217 | 0.139 | 0.906 | 0.904 | 0.403 | 0.418 | 0/0 | 0/0 | 0.001 |
| s000017 | 8 | 0.774 | 0.445 | 0.796 | 0.809 | 0.175 | 0.285 | 3/4 | 4/5 | 0.015 |
| s000018 | 15 | 0.157 | 0.164 | 0.893 | 0.822 | 0.729 | 0.762 | 3/1 | 2/1 | 0.005 |
| s000019 | 3 | 0.091 | 0.669 | 0.913 | 0.737 | 0.729 | 0.565 | 0/0 | 1/0 | 0.000 |

| mean | real SAM 3 | pseudo |
|---|---|---|
| food points with lift_label != label | 0.222 | 0.206 |
| food points with no vote | 0.012 | 0.061 |
| mean lift_share (voted food points) | 0.870 | 0.853 |
| per-scan instance mIoU | 0.644 | 0.659 |
| merges (total) | 15 | 14 |
| splits (total) | 24 | 16 |
| SAM instances with no GT match (total) | 204 | - |
| GT items with no SAM match (total) | 10 | - |


Per-item lift recall (points of the GT item that get its own id) by kind / source:

| group | items | mean recall | items with recall < 0.3 |
|---|---|---|---|
| kind=asset | 10 | 0.720 | 2 |
| kind=procedural | 35 | 0.608 | 8 |
| kind=scan | 82 | 0.677 | 15 |
| source=metafood3d | 82 | 0.677 | 15 |
| source=objaverse | 9 | 0.747 | 2 |
| source=primitive | 11 | 0.555 | 3 |
| source=procedural_leaves | 5 | 0.562 | 2 |
| source=procedural_piece | 16 | 0.694 | 2 |
| source=procedural_topping | 3 | 0.427 | 1 |
| source=ycb | 1 | 0.473 | 0 |

Lowest-recall items: fried mushrooms (metafood3d, 0.00), fried mushrooms (metafood3d, 0.00), fried mushrooms (metafood3d, 0.00), chicken nugget (metafood3d, 0.00), almonds (metafood3d, 0.00), lettuce (procedural_leaves, 0.00), tortilla chips (metafood3d, 0.00), tortilla chips (metafood3d, 0.01), fried rice (procedural_piece, 0.01), sushi (objaverse, 0.07), brown rice (procedural_piece, 0.08), cherry tomato (primitive, 0.09)