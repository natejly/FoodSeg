# PTv3 as a semantic cleaner on real SAM 3 votes

40 real-vote scans, 3 folds stratified by batch (v12/v13), voxel-level, item-filtered. Arms: lift = the lift's own reading; full = frozen Sonata PTv3, all inputs; nolift = same without the lift channel. Per-scan means with bootstrap 95% CI over scans; class tables are pooled over all val points (union over folds).

## All scans (40 scans)

| arm | IoU table | IoU container | IoU clutter | IoU food | mIoU (pooled) | mIoU per-scan mean [CI] | container R / P | clutter R / P |
|---|---|---|---|---|---|---|---|---|
| lift | 0.916 | 0.875 | 0.118 | 0.853 | 0.691 | 0.824 [0.780, 0.868] | 0.965 / 0.905 | 0.305 / 0.162 |
| full | 0.917 | 0.872 | 0.218 | 0.830 | 0.709 | 0.831 [0.791, 0.870] | 0.973 / 0.894 | 0.576 / 0.260 |
| nolift | 0.903 | 0.821 | 0.216 | 0.767 | 0.677 | 0.779 [0.741, 0.814] | 0.953 / 0.856 | 0.613 / 0.251 |

## Batch v12 (20 scans)

| arm | IoU table | IoU container | IoU clutter | IoU food | mIoU (pooled) | mIoU per-scan mean [CI] | container R / P | clutter R / P |
|---|---|---|---|---|---|---|---|---|
| lift | 0.903 | 0.928 | 0.100 | 0.924 | 0.713 | 0.826 [0.760, 0.887] | 0.988 / 0.938 | 0.280 / 0.134 |
| full | 0.916 | 0.930 | 0.221 | 0.866 | 0.733 | 0.838 [0.779, 0.892] | 0.989 / 0.939 | 0.557 / 0.268 |
| nolift | 0.913 | 0.888 | 0.251 | 0.813 | 0.716 | 0.803 [0.752, 0.853] | 0.978 / 0.906 | 0.588 / 0.305 |

## Batch v13 (20 scans)

| arm | IoU table | IoU container | IoU clutter | IoU food | mIoU (pooled) | mIoU per-scan mean [CI] | container R / P | clutter R / P |
|---|---|---|---|---|---|---|---|---|
| lift | 0.930 | 0.803 | 0.386 | 0.807 | 0.732 | 0.823 [0.759, 0.884] | 0.929 / 0.856 | 0.459 / 0.708 |
| full | 0.917 | 0.795 | 0.206 | 0.806 | 0.681 | 0.823 [0.768, 0.874] | 0.948 / 0.832 | 0.686 / 0.227 |
| nolift | 0.892 | 0.733 | 0.131 | 0.737 | 0.623 | 0.754 [0.701, 0.805] | 0.916 / 0.786 | 0.760 / 0.137 |

## Food recovered / leaked / lost (per-scan means, CI over scans)

recovered = GT-food points the lift did NOT call food that the net calls food (lift recovers 0 by definition). lost = GT-food points the lift called food that the net turned non-food. leaked = non-food points called food. d leak = net leak minus lift leak, paired per scan.

| arm | batch | scans | food recovered | food lost | leaked (net) | leaked (lift) | d leak vs lift (paired) | d food recall vs lift (paired) |
|---|---|---|---|---|---|---|---|---|
| full | all | 40 | 0.215 [0.157, 0.278] (pooled 0.240) | 0.025 [0.013, 0.038] | 0.0173 [0.0085, 0.0292] | 0.0170 [0.0069, 0.0306] | 0.0004 [-0.0041, 0.0064] | -0.010 [-0.025, 0.004] |
| full | v12 | 20 | 0.155 [0.094, 0.223] (pooled 0.157) | 0.020 [0.007, 0.039] | 0.0078 [0.0015, 0.0178] | 0.0031 [0.0013, 0.0059] | 0.0047 [-0.0003, 0.0141] | -0.012 [-0.032, 0.002] |
| full | v13 | 20 | 0.275 [0.181, 0.381] (pooled 0.281) | 0.029 [0.012, 0.049] | 0.0268 [0.0114, 0.0469] | 0.0308 [0.0119, 0.0557] | -0.0039 [-0.0101, 0.0012] | -0.008 [-0.032, 0.016] |
| nolift | all | 40 | 0.349 [0.267, 0.434] (pooled 0.370) | 0.070 [0.046, 0.098] | 0.0225 [0.0120, 0.0350] | 0.0170 [0.0068, 0.0299] | 0.0055 [-0.0017, 0.0144] | -0.045 [-0.073, -0.020] |
| nolift | v12 | 20 | 0.256 [0.164, 0.363] (pooled 0.288) | 0.066 [0.034, 0.101] | 0.0097 [0.0023, 0.0205] | 0.0031 [0.0013, 0.0059] | 0.0066 [0.0004, 0.0153] | -0.049 [-0.082, -0.019] |
| nolift | v13 | 20 | 0.442 [0.316, 0.569] (pooled 0.410) | 0.075 [0.039, 0.116] | 0.0352 [0.0178, 0.0567] | 0.0308 [0.0116, 0.0543] | 0.0044 [-0.0075, 0.0205] | -0.040 [-0.084, -0.002] |

Lift food-recall (pooled) = 0.939; 10996 GT-food voxels are missed by the lift, 179551 GT-food voxels in total.

## Cost

| arm | inference s/scan (mean, median) | voxels/scan | peak GPU GiB (eval, per fold) |
|---|---|---|---|
| full | 0.21, 0.16 | 36987 | 0.95, 0.95, 0.95 |
| nolift | 0.21, 0.16 | 36987 | 0.95, 0.95, 0.95 |

## Setup

- Task: 4-class semantic (table/background, container, clutter, food) at 2 mm voxel level; items dropped by the manifest filter are excluded. Baseline = lift_label read as a class (253 clutter, 254 container, item id food, 0 background).
- Folds: 40 real SAM 3 vote scans (hard_v12 / hard_v13 `_sam3_v2_image`), 3 folds by scan stratified by batch (14/14/12 val scans). Train per fold = other real scans x3 + split_v3 pseudo-vote train scans minus pseudo copies of the fold's val scans (41-44 pseudo). Splits: `split_f{0,1,2}.json`, `setup.json`.
- Model: Sonata-initialised PTv3, encoder frozen, patch_size 512, 4 epochs x epoch_repeat 4 (about 1900 batches, 3-4 min), fp16 train / fp32 eval, one seed (0). Instance heads trained with weight 0 (`off_w=dir_w=0`, no `decoder=none` exists) and their outputs ignored. The checkpoint is `last.pt` (no val selection). `skip_val` was added to `train.Config`.
- Code: `pipeline/seg3d/semantic.py` (`--setup | --run | --report`). Per-scan predictions in `<arm>_f<k>/preds`, raw counts in `<arm>_f<k>/final.json`.
- Caveats: one seed, 6 min cap never binding; per-class IoU on clutter is dominated by few scans; pooled mIoU is dominated by large scans, per-scan mean is the paired headline. Peak GPU is the same for all folds because eval is capped by the largest scan (fp32).

## Reading

1. On real votes the semantic cleaner adds little overall: per-scan mIoU 0.824 -> 0.831 (+0.007, CIs overlap), food IoU 0.853 -> 0.830 (lower), container IoU 0.875 -> 0.872.
2. It does recover food the lift missed (24% of the 6% of food voxels the lift calls non-food) but also drops 2.5% of the food the lift had right, so net food recall changes by -0.010 [-0.025, 0.004] and non-food leaked into food is unchanged (+0.0004 [-0.004, 0.006]): no downstream win for food masks.
3. The one real gain is clutter on v12 (IoU 0.10 -> 0.22, recall 0.28 -> 0.56), but it over-fires on v13 where the lift's clutter is already good (IoU 0.39 -> 0.21, precision 0.71 -> 0.23), so it is a trade, not a clean improvement; only 2 batches, 40 scans.
4. It needs the lift channel: without it mIoU drops to 0.779 (-0.045 vs full), food recall to -0.045 [-0.073, -0.020] vs lift, food IoU 0.767. The net is a vote corrector, not a standalone segmenter at this training budget (frozen encoder, ~120 train scenes).
5. Cost is small (0.21 s/scan mean, 0.95 GiB peak, 3-4 min to train a fold) but non-zero (spconv/Sonata install, 430 MB weights) for no measurable food gain.
6. Recommendation: do not run by default in the real pipeline. Keep it as an optional clutter/container check; revisit with unfrozen top stages and more real-vote scans (ablation ceiling was 0.84 on pseudo votes) before enabling.
