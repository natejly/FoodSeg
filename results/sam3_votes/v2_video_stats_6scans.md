# SAM 3 real votes vs pseudo votes (v2_video (6 scans))

6 scans.

| scan | items | wrong real | wrong pseudo | share real | share pseudo | mIoU real | mIoU pseudo | merges r/p | splits r/p | unvoted real |
|---|---|---|---|---|---|---|---|---|---|---|
| s000000 | 4 | 0.447 | 0.020 | 0.942 | 0.953 | 0.335 | 0.936 | 0/0 | 1/0 | 0.352 |
| s000001 | 5 | 0.102 | 0.065 | 0.963 | 0.908 | 0.500 | 0.733 | 1/1 | 0/2 | 0.006 |
| s000002 | 8 | 0.547 | 0.081 | 0.899 | 0.911 | 0.096 | 0.822 | 1/0 | 1/0 | 0.003 |
| s000011 | 3 | 0.514 | 0.173 | 0.943 | 0.776 | 0.324 | 0.746 | 1/0 | 1/0 | 0.001 |
| s000012 | 4 | 0.052 | 0.049 | 0.975 | 0.925 | 0.923 | 0.934 | 0/0 | 0/0 | 0.003 |
| s000013 | 2 | 0.072 | 0.047 | 0.976 | 0.949 | 0.824 | 0.828 | 0/0 | 0/0 | 0.002 |

| mean | real SAM 3 | pseudo |
|---|---|---|
| food points with lift_label != label | 0.289 | 0.073 |
| food points with no vote | 0.061 | 0.010 |
| mean lift_share (voted food points) | 0.950 | 0.904 |
| per-scan instance mIoU | 0.500 | 0.833 |
| merges (total) | 3 | 1 |
| splits (total) | 3 | 2 |
| SAM instances with no GT match (total) | 16 | - |
| GT items with no SAM match (total) | 14 | - |


Per-item lift recall (points of the GT item that get its own id) by kind / source:

| group | items | mean recall | items with recall < 0.3 |
|---|---|---|---|
| kind=asset | 4 | 0.248 | 3 |
| kind=procedural | 5 | 0.302 | 3 |
| kind=scan | 21 | 0.484 | 9 |
| source=metafood3d | 21 | 0.484 | 9 |
| source=objaverse | 4 | 0.248 | 3 |
| source=primitive | 1 | 0.000 | 1 |
| source=procedural_leaves | 3 | 0.503 | 1 |
| source=procedural_piece | 1 | 0.000 | 1 |

Lowest-recall items: lettuce (procedural_leaves, 0.00), cucumber (metafood3d, 0.00), broccoli (metafood3d, 0.00), chicken thighs (metafood3d, 0.00), boiled egg (primitive, 0.00), sushi (metafood3d, 0.00), sushi (objaverse, 0.00), sushi (metafood3d, 0.00), sushi (objaverse, 0.00), sushi (objaverse, 0.00), hummus (procedural_piece, 0.00), broccoli (metafood3d, 0.00)