# SAM 3 real votes vs pseudo votes (v2 image, hard_v12, 20 scans)

20 scans.

| scan | items | wrong real | wrong pseudo | share real | share pseudo | mIoU real | mIoU pseudo | merges r/p | splits r/p | unvoted real |
|---|---|---|---|---|---|---|---|---|---|---|
| s000000 | 4 | 0.420 | 0.020 | 0.680 | 0.953 | 0.571 | 0.936 | 0/0 | 3/0 | 0.001 |
| s000001 | 5 | 0.316 | 0.065 | 0.625 | 0.908 | 0.394 | 0.733 | 2/1 | 2/2 | 0.001 |
| s000002 | 8 | 0.433 | 0.081 | 0.569 | 0.911 | 0.420 | 0.822 | 4/0 | 5/0 | 0.002 |
| s000003 | 12 | 0.093 | 0.129 | 0.904 | 0.823 | 0.771 | 0.750 | 0/1 | 0/1 | 0.004 |
| s000004 | 9 | 0.120 | 0.095 | 0.910 | 0.860 | 0.757 | 0.808 | 1/0 | 1/0 | 0.000 |
| s000005 | 6 | 0.098 | 0.078 | 0.945 | 0.911 | 0.804 | 0.824 | 0/0 | 0/0 | 0.008 |
| s000006 | 8 | 0.072 | 0.441 | 0.908 | 0.873 | 0.796 | 0.762 | 1/0 | 2/0 | 0.001 |
| s000007 | 5 | 0.190 | 0.246 | 0.802 | 0.860 | 0.801 | 0.733 | 1/0 | 2/0 | 0.003 |
| s000008 | 13 | 0.501 | 0.274 | 0.838 | 0.779 | 0.330 | 0.585 | 2/2 | 2/1 | 0.014 |
| s000009 | 3 | 0.246 | 0.099 | 0.908 | 0.762 | 0.718 | 0.762 | 0/1 | 1/1 | 0.000 |
| s000010 | 3 | 0.359 | 0.355 | 0.826 | 0.730 | 0.341 | 0.374 | 1/1 | 0/1 | 0.005 |
| s000011 | 3 | 0.140 | 0.173 | 0.877 | 0.776 | 0.756 | 0.746 | 0/0 | 1/0 | 0.004 |
| s000012 | 4 | 0.117 | 0.049 | 0.789 | 0.925 | 0.853 | 0.934 | 0/0 | 0/0 | 0.012 |
| s000013 | 2 | 0.059 | 0.047 | 0.958 | 0.949 | 0.755 | 0.828 | 0/0 | 0/0 | 0.007 |
| s000014 | 5 | 0.181 | 0.084 | 0.693 | 0.929 | 0.755 | 0.737 | 1/0 | 1/0 | 0.003 |
| s000015 | 6 | 0.358 | 0.388 | 0.801 | 0.705 | 0.446 | 0.496 | 0/1 | 1/0 | 0.017 |
| s000016 | 4 | 0.320 | 0.066 | 0.792 | 0.899 | 0.657 | 0.852 | 0/0 | 2/0 | 0.001 |
| s000017 | 4 | 0.301 | 0.198 | 0.704 | 0.832 | 0.562 | 0.674 | 1/1 | 2/1 | 0.009 |
| s000018 | 4 | 0.082 | 0.049 | 0.815 | 0.817 | 0.876 | 0.902 | 0/0 | 0/0 | 0.000 |
| s000019 | 16 | 0.140 | 0.082 | 0.784 | 0.842 | 0.762 | 0.830 | 2/2 | 4/2 | 0.003 |

| mean | real SAM 3 | pseudo |
|---|---|---|
| food points with lift_label != label | 0.227 | 0.151 |
| food points with no vote | 0.005 | 0.043 |
| mean lift_share (voted food points) | 0.806 | 0.852 |
| per-scan instance mIoU | 0.656 | 0.754 |
| merges (total) | 16 | 10 |
| splits (total) | 29 | 9 |
| SAM instances with no GT match (total) | 166 | - |
| GT items with no SAM match (total) | 31 | - |


Per-item lift recall (points of the GT item that get its own id) by kind / source:

| group | items | mean recall | items with recall < 0.3 |
|---|---|---|---|
| kind=asset | 10 | 0.661 | 2 |
| kind=procedural | 49 | 0.646 | 9 |
| kind=scan | 87 | 0.679 | 16 |
| source=metafood3d | 87 | 0.679 | 16 |
| source=objaverse | 9 | 0.624 | 2 |
| source=primitive | 30 | 0.577 | 9 |
| source=procedural_leaves | 5 | 0.691 | 0 |
| source=procedural_piece | 10 | 0.797 | 0 |
| source=procedural_topping | 4 | 0.729 | 0 |
| source=ycb | 1 | 1.000 | 0 |

Lowest-recall items: broccoli (metafood3d, 0.00), bacon (metafood3d, 0.00), cucumber (metafood3d, 0.00), cucumber (metafood3d, 0.00), cherry tomato (primitive, 0.00), cherry tomato (primitive, 0.00), cherry tomato (primitive, 0.00), cucumber (metafood3d, 0.00), asparagus (metafood3d, 0.00), asparagus (metafood3d, 0.00), falafel (metafood3d, 0.00), falafel (metafood3d, 0.00)