# SAM 3 real votes vs pseudo votes (v2 image default)

11 scans.

| scan | items | wrong real | wrong pseudo | share real | share pseudo | mIoU real | mIoU pseudo | merges r/p | splits r/p | unvoted real |
|---|---|---|---|---|---|---|---|---|---|---|
| s000000 | 4 | 0.420 | 0.020 | 0.680 | 0.953 | 0.571 | 0.936 | 0/0 | 3/0 | 0.001 |
| s000001 | 5 | 0.316 | 0.065 | 0.625 | 0.908 | 0.394 | 0.733 | 2/1 | 2/2 | 0.001 |
| s000002 | 8 | 0.433 | 0.081 | 0.569 | 0.911 | 0.420 | 0.822 | 4/0 | 5/0 | 0.002 |
| s000007 | 5 | 0.190 | 0.246 | 0.802 | 0.860 | 0.801 | 0.733 | 1/0 | 2/0 | 0.003 |
| s000009 | 3 | 0.227 | 0.099 | 0.906 | 0.762 | 0.753 | 0.762 | 0/1 | 1/1 | 0.001 |
| s000011 | 3 | 0.140 | 0.173 | 0.877 | 0.776 | 0.756 | 0.746 | 0/0 | 1/0 | 0.004 |
| s000012 | 4 | 0.117 | 0.049 | 0.789 | 0.925 | 0.853 | 0.934 | 0/0 | 0/0 | 0.012 |
| s000013 | 2 | 0.059 | 0.047 | 0.958 | 0.949 | 0.755 | 0.828 | 0/0 | 0/0 | 0.007 |
| s000014 | 5 | 0.181 | 0.084 | 0.693 | 0.929 | 0.755 | 0.737 | 1/0 | 1/0 | 0.003 |
| s000016 | 4 | 0.320 | 0.066 | 0.792 | 0.899 | 0.657 | 0.852 | 0/0 | 2/0 | 0.001 |
| s000017 | 4 | 0.301 | 0.198 | 0.704 | 0.832 | 0.562 | 0.674 | 1/1 | 2/1 | 0.009 |

| mean | real SAM 3 | pseudo |
|---|---|---|
| food points with lift_label != label | 0.246 | 0.103 |
| food points with no vote | 0.004 | 0.025 |
| mean lift_share (voted food points) | 0.763 | 0.882 |
| per-scan instance mIoU | 0.662 | 0.796 |
| merges (total) | 9 | 3 |
| splits (total) | 19 | 4 |
| SAM instances with no GT match (total) | 56 | - |
| GT items with no SAM match (total) | 8 | - |


Per-item lift recall (points of the GT item that get its own id) by kind / source:

| group | items | mean recall | items with recall < 0.3 |
|---|---|---|---|
| kind=asset | 4 | 0.681 | 0 |
| kind=procedural | 14 | 0.668 | 0 |
| kind=scan | 34 | 0.669 | 4 |
| source=metafood3d | 34 | 0.669 | 4 |
| source=objaverse | 4 | 0.681 | 0 |
| source=primitive | 5 | 0.813 | 0 |
| source=procedural_leaves | 4 | 0.618 | 0 |
| source=procedural_piece | 4 | 0.624 | 0 |
| source=procedural_topping | 1 | 0.316 | 0 |

Lowest-recall items: broccoli (metafood3d, 0.00), cucumber (metafood3d, 0.00), sushi (metafood3d, 0.05), grapes (metafood3d, 0.14), gravy (procedural_topping, 0.32), ice cream (metafood3d, 0.32), yogurt (procedural_piece, 0.32), sushi (metafood3d, 0.33), cucumber (metafood3d, 0.39), boiled egg (primitive, 0.39), rice (metafood3d, 0.39), lettuce (procedural_leaves, 0.45)