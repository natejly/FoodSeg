# seg3d refiner: testing and ablation study (2026-10-15)

Everything below is on **11 validation scans / 22 food items**. Differences under roughly 0.05 mIoU are not evidence
(per-scan SE of the instance mIoU is 0.05-0.10, seed-to-seed range for the same config up to 0.07).
Reproduce: `cd pipeline; python -m seg3d.ablate --split --baseline --run; python -m seg3d.ablate --aggregate`
(driver: `pipeline/seg3d/ablate.py`; per-run outputs `results/seg3d/abl_<config>_s<seed>/{train.log,metrics.json,final.json}`;
all numbers in `ablations.json`). Tests: `python -m pytest tests/test_seg3d_*.py -q` -> 13 passed
(8 old + 5 new in `tests/test_seg3d_anchored.py`: lift-anchored decoding on toy blobs, split option, class-disagreement
trigger, corrected/broken metric).

## Setup actually used (deviations from the brief are in bold)

- Pool: every scan with `gt/seg_points.npz`: hard 6, hard_v3 6, hard_v9 20, hard_v10 20 = 52. Fixed split, seed 0, by scan dir,
  stratified by batch, written to `results/seg3d/split.json`, used for every run. Train uses all non-val scans (including
  non-`ok` ones); val is drawn only from `ok` scans of hard_v9 (5 ok) / hard_v10 (8 ok) plus hard/hard_v3 (12, all ok), capped at 60 %
  of a batch's eligible scans so that some `ok` scans stay in train. That cap is why val is small: only 25 scans are eligible at all.
- Regimes. R1 = stored lift as-is (hard_v9/v10) ; hard/hard_v3 have no lift_* keys, so their "stored" lift is GT with the old
  mild synthetic corruption (4 of the 11 val scans). R2 = `augment.HARD_LIFT` applied on top, deterministic (fixed seed per scan):
  merge p 0.6, blob flip p 0.5 with 15-40 mm blobs, drop-instance p 0.15, erode/dilate p 0.7 by 2-4 voxels (2 mm), on stored
  lift too; R2 training applies the same distribution with p=1 to every sample. The lift baseline drops from 0.780 (R1) to 0.648 (R2)
  instance mIoU (the earlier 0.93 was on only 3 scans / 5 items).
- Schedule identical for all PTv3 runs: Sonata init, encoder frozen, **patch_size 512** (another process, a Blender synth job, held
  1.8-4.3 GB of the 6 GB card during the study, so the brief's "other process > 2.5 GB" rule applied), 8 epochs x 164 batches = 1312
  batches (grad-accum 2), `max_minutes=6`. **The card and CPUs were shared, and a few runs hit the 6-minute cap before 1312
  batches (column `iters` is the mean over seeds: nolift 1233, norgb 1248, nodropout 1122 - the last is therefore under-trained
  relative to the rest and confounded)**. Reported checkpoint = `last.pt` (not best.pt: best-on-val over 2 val passes with 11 scans
  would be optimistic).
- Metrics are voxel-level (2 mm). `corr`/`broke` (new): over the food domain (voxels that are GT food, lift food or predicted food),
  fraction of lift-wrong points the model gets right / fraction of lift-right points the model gets wrong. A point is "right" if
  non-food and class matches, or food and in the prediction matched (Hungarian) to its GT item. Pooled over scans, so large items dominate.
  `all` variants are in the json. "inst mIoU" is pooled over the 22 items; the "d vs lift" column is the mean over scans of the per-scan
  difference (a different average, so the two can disagree in sign) +- its SE over 11 scans.
- Decoders: `plain` = DBSCAN on xyz+predicted offset (class from the semantic head); `anchored` = lift-anchored (see
  `cluster.anchored_instances`: start from lift, reassign only where lift_share<0.8, boundary==1, or the head disagrees with the lift
  class; clusters mapped to lift ids by majority overlap, new id if overlap<30 %); `anchored_split` = anchored plus letting >=2 clusters
  that each cover >=25 % of one lift instance split it (beyond the spec; the plain spec cannot undo a merge); `anchored_nob` =
  anchored without the GT-derived `boundary` flag. `anchored_nob` equals `anchored` to 3 decimals in every run, so the boundary flag
  leaks nothing measurable here (it is a GT label-ambiguity band, 0.3 % of points).
- Models without a lift input still use lift when decoded `anchored`, so for those rows (nolift, DGCNN) compare **plain**, not anchored.
- Bug found and fixed during the study: `DGCNNSeg` collapsed at eval (val sem mIoU 0.10, 31 % point accuracy vs 94 % with the same
  weights and batch-statistics BatchNorm; batch = 1 scene, so running stats do not transfer between scenes). `model.py` now keeps BN in
  batch-statistics mode at eval; the two DGCNN configs were re-run with that. The first PTv3 runs are unaffected.

<!-- AUTO:BEGIN -->
### Data and split (auto)

Fixed split (`results/seg3d/split.json`, seed 0, stratified by batch, by scan dir, used for every run): **train 41 scans / 131 items**, **val 11 scans / 22 items** (22 items present in the val clouds).  Per batch (train / val): hard 4/2, hard_v3 4/2, hard_v9 17/3, hard_v10 16/4.  `ok` scans: train 14, val 11 (val is restricted to `ok` scans of hard_v9/v10 + all of hard/hard_v3).


### R1 - stored lift as-is (old files: GT-derived lift with the default mild corruption)

Mean (min-max over seeds) on the val split; `corr` = fraction of lift-wrong points fixed, `broke` = fraction of lift-right points broken (food domain); `SE` = standard error of the per-scan instance mIoU (scan-to-scan noise); peak GiB / s per iter / wall minutes are for the training run.

| config | train | decode | seeds | sem mIoU | inst mIoU | SE | d vs lift (+-SE) | PQ | bIoU | count | corr | broke | iters | GiB | s/it | wall min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lift_label (no training) | - | - | - | 0.826 | 0.780 | 0.059 | - | 0.809 | 0.586 | 0.64 | 0 | 0 | - | - | - | - |
| lift_label, trivially anchored | - | - | - | 0.826 | 0.780 | 0.059 | - | 0.809 | 0.586 | 0.64 | 0 | 0 | - | - | - | - |
| dgcnn_xyz_r2 | r2 | plain | 1 | 0.627 | 0.346 | 0.112 | -0.445 +-0.121 | 0.291 | 0.072 | 0.455 | 0.707 | 0.428 | 1312 | 0.25 | 0.06 | 1.5 |
| dgcnn_xyz_r2 | r2 | anchored | 1 | 0.627 | 0.534 | 0.090 | -0.314 +-0.105 | 0.456 | 0.246 | 0.545 | 0.713 | 0.352 | 1312 | 0.25 | 0.06 | 1.5 |
| dgcnn_xyz_r2 | r2 | anchored_split | 1 | 0.627 | 0.505 | 0.094 | -0.343 +-0.109 | 0.399 | 0.238 | 0.455 | 0.704 | 0.389 | 1312 | 0.25 | 0.06 | 1.5 |
| dgcnn_xyzrgb_r2 | r2 | plain | 2 | 0.592 (0.57-0.62) | 0.465 (0.45-0.48) | 0.102 | -0.291 +-0.102 | 0.310 (0.25-0.37) | 0.142 (0.13-0.16) | 0.455 (0.45-0.45) | 0.824 (0.77-0.87) | 0.466 (0.38-0.56) | 1312 | 0.25 | 0.06 | 1.6 |
| dgcnn_xyzrgb_r2 | r2 | anchored | 2 | 0.592 (0.57-0.62) | 0.599 (0.60-0.60) | 0.098 | -0.176 +-0.093 | 0.440 (0.43-0.45) | 0.273 (0.21-0.33) | 0.591 (0.55-0.64) | 0.839 (0.79-0.89) | 0.387 (0.30-0.47) | 1312 | 0.25 | 0.06 | 1.6 |
| dgcnn_xyzrgb_r2 | r2 | anchored_split | 2 | 0.592 (0.57-0.62) | 0.544 (0.54-0.55) | 0.095 | -0.252 +-0.098 | 0.353 (0.32-0.39) | 0.277 (0.22-0.33) | 0.500 (0.45-0.55) | 0.833 (0.78-0.89) | 0.426 (0.34-0.51) | 1312 | 0.25 | 0.06 | 1.6 |
| full_r1 | r1 | plain | 2 | 0.883 (0.86-0.90) | 0.553 (0.55-0.56) | 0.082 | -0.217 +-0.085 | 0.463 (0.45-0.47) | 0.289 (0.28-0.29) | 0.364 (0.36-0.36) | 0.770 (0.73-0.81) | 0.212 (0.20-0.23) | 1312 | 0.69 | 0.24 | 5.6 |
| full_r1 | r1 | anchored | 2 | 0.883 (0.86-0.90) | 0.768 (0.76-0.78) | 0.063 | -0.026 +-0.058 | 0.790 (0.78-0.80) | 0.439 (0.43-0.45) | 0.818 (0.82-0.82) | 0.777 (0.73-0.82) | 0.066 (0.05-0.08) | 1312 | 0.69 | 0.24 | 5.6 |
| full_r1 | r1 | anchored_split | 2 | 0.883 (0.86-0.90) | 0.674 (0.66-0.69) | 0.068 | -0.137 +-0.080 | 0.613 (0.59-0.64) | 0.448 (0.44-0.46) | 0.500 (0.45-0.55) | 0.764 (0.72-0.81) | 0.153 (0.14-0.17) | 1312 | 0.69 | 0.24 | 5.6 |
| full_r1 | r1 | anchored_nob | 2 | 0.883 (0.86-0.90) | 0.768 (0.76-0.78) | 0.063 | -0.026 +-0.058 | 0.790 (0.78-0.80) | 0.439 (0.43-0.45) | 0.818 (0.82-0.82) | 0.777 (0.73-0.82) | 0.066 (0.05-0.08) | 1312 | 0.69 | 0.24 | 5.6 |
| full_r2 | r2 | plain | 2 | 0.879 (0.88-0.88) | 0.587 (0.58-0.59) | 0.084 | -0.169 +-0.086 | 0.500 (0.47-0.53) | 0.274 (0.24-0.31) | 0.455 (0.36-0.55) | 0.789 (0.78-0.80) | 0.193 (0.19-0.19) | 1312 | 0.69 | 0.21 | 5.2 |
| full_r2 | r2 | anchored | 2 | 0.879 (0.88-0.88) | 0.764 (0.75-0.78) | 0.054 | -0.019 +-0.055 | 0.769 (0.75-0.79) | 0.424 (0.39-0.46) | 0.773 (0.73-0.82) | 0.796 (0.78-0.81) | 0.067 (0.06-0.08) | 1312 | 0.69 | 0.21 | 5.2 |
| full_r2 | r2 | anchored_split | 2 | 0.879 (0.88-0.88) | 0.701 (0.69-0.71) | 0.064 | -0.090 +-0.074 | 0.652 (0.65-0.65) | 0.419 (0.38-0.45) | 0.591 (0.55-0.64) | 0.784 (0.77-0.80) | 0.125 (0.12-0.13) | 1312 | 0.69 | 0.21 | 5.2 |
| full_r2 | r2 | anchored_nob | 2 | 0.879 (0.88-0.88) | 0.764 (0.75-0.78) | 0.054 | -0.019 +-0.055 | 0.769 (0.75-0.79) | 0.424 (0.39-0.46) | 0.773 (0.73-0.82) | 0.796 (0.78-0.81) | 0.067 (0.06-0.08) | 1312 | 0.69 | 0.21 | 5.2 |
| nocorrupt_r2 | r2 | plain | 1 | 0.861 | 0.575 | 0.086 | -0.181 +-0.087 | 0.479 | 0.250 | 0.455 | 0.757 | 0.199 | 1312 | 0.68 | 0.23 | 5.7 |
| nocorrupt_r2 | r2 | anchored | 1 | 0.861 | 0.776 | 0.058 | -0.007 +-0.051 | 0.802 | 0.411 | 0.818 | 0.764 | 0.049 | 1312 | 0.68 | 0.23 | 5.7 |
| nocorrupt_r2 | r2 | anchored_split | 1 | 0.861 | 0.676 | 0.065 | -0.124 +-0.078 | 0.610 | 0.404 | 0.455 | 0.752 | 0.132 | 1312 | 0.68 | 0.23 | 5.7 |
| nodropout_r2 | r2 | plain | 1 | 0.857 | 0.586 | 0.087 | -0.174 +-0.088 | 0.468 | 0.317 | 0.455 | 0.818 | 0.196 | 1122 | 0.68 | 0.31 | 6.4 |
| nodropout_r2 | r2 | anchored | 1 | 0.857 | 0.780 | 0.064 | -0.006 +-0.058 | 0.767 | 0.471 | 0.727 | 0.825 | 0.055 | 1122 | 0.68 | 0.31 | 6.4 |
| nodropout_r2 | r2 | anchored_split | 1 | 0.857 | 0.681 | 0.069 | -0.122 +-0.082 | 0.583 | 0.466 | 0.455 | 0.813 | 0.138 | 1122 | 0.68 | 0.31 | 6.4 |
| nolift_r2 | r2 | plain | 2 | 0.802 (0.78-0.83) | 0.495 (0.48-0.51) | 0.092 | -0.301 +-0.098 | 0.372 (0.34-0.40) | 0.238 (0.23-0.24) | 0.364 (0.36-0.36) | 0.690 (0.68-0.70) | 0.309 (0.31-0.31) | 1233 | 0.69 | 0.25 | 5.5 |
| nolift_r2 | r2 | anchored | 2 | 0.802 (0.78-0.83) | 0.678 (0.65-0.71) | 0.078 | -0.145 +-0.084 | 0.602 (0.60-0.61) | 0.380 (0.38-0.38) | 0.591 (0.55-0.64) | 0.699 (0.69-0.71) | 0.185 (0.17-0.20) | 1233 | 0.69 | 0.25 | 5.5 |
| nolift_r2 | r2 | anchored_split | 2 | 0.802 (0.78-0.83) | 0.614 (0.59-0.64) | 0.077 | -0.210 +-0.089 | 0.486 (0.48-0.50) | 0.395 (0.39-0.40) | 0.409 (0.36-0.45) | 0.689 (0.68-0.70) | 0.235 (0.21-0.26) | 1233 | 0.69 | 0.25 | 5.5 |
| norgb_r2 | r2 | plain | 2 | 0.819 (0.81-0.83) | 0.520 (0.51-0.53) | 0.084 | -0.269 +-0.104 | 0.455 (0.45-0.46) | 0.246 (0.24-0.25) | 0.364 (0.36-0.36) | 0.814 (0.81-0.81) | 0.268 (0.26-0.28) | 1248 | 0.69 | 0.23 | 5.2 |
| norgb_r2 | r2 | anchored | 2 | 0.819 (0.81-0.83) | 0.713 (0.71-0.71) | 0.054 | -0.090 +-0.074 | 0.710 (0.70-0.72) | 0.399 (0.39-0.41) | 0.818 (0.82-0.82) | 0.821 (0.82-0.82) | 0.149 (0.13-0.17) | 1248 | 0.69 | 0.23 | 5.2 |
| norgb_r2 | r2 | anchored_split | 2 | 0.819 (0.81-0.83) | 0.645 (0.64-0.65) | 0.062 | -0.164 +-0.088 | 0.584 (0.57-0.59) | 0.396 (0.39-0.41) | 0.545 (0.55-0.55) | 0.808 (0.81-0.81) | 0.212 (0.19-0.23) | 1248 | 0.69 | 0.23 | 5.2 |
| scratch_r2 | r2 | plain | 1 | 0.730 | 0.422 | 0.076 | -0.353 +-0.086 | 0.138 | 0.147 | 0.091 | 0.757 | 0.465 | 1312 | 0.69 | 0.17 | 4.2 |
| scratch_r2 | r2 | anchored | 1 | 0.730 | 0.626 | 0.064 | -0.180 +-0.070 | 0.315 | 0.318 | 0.364 | 0.767 | 0.332 | 1312 | 0.69 | 0.17 | 4.2 |
| scratch_r2 | r2 | anchored_split | 1 | 0.730 | 0.568 | 0.058 | -0.250 +-0.079 | 0.227 | 0.310 | 0.273 | 0.765 | 0.360 | 1312 | 0.69 | 0.17 | 4.2 |


### R2 - hard lift (merge 0.6, blob 0.5 @15-40 mm, drop 0.15, morph 2-4 voxels; fixed seed per scan)

Mean (min-max over seeds) on the val split; `corr` = fraction of lift-wrong points fixed, `broke` = fraction of lift-right points broken (food domain); `SE` = standard error of the per-scan instance mIoU (scan-to-scan noise); peak GiB / s per iter / wall minutes are for the training run.

| config | train | decode | seeds | sem mIoU | inst mIoU | SE | d vs lift (+-SE) | PQ | bIoU | count | corr | broke | iters | GiB | s/it | wall min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lift_label (no training) | - | - | - | 0.810 | 0.648 | 0.103 | - | 0.687 | 0.503 | 0.64 | 0 | 0 | - | - | - | - |
| lift_label, trivially anchored | - | - | - | 0.810 | 0.648 | 0.103 | - | 0.687 | 0.503 | 0.64 | 0 | 0 | - | - | - | - |
| dgcnn_xyz_r2 | r2 | plain | 1 | 0.629 | 0.384 | 0.113 | -0.194 +-0.156 | 0.329 | 0.037 | 0.364 | 0.700 | 0.430 | 1312 | 0.25 | 0.06 | 1.5 |
| dgcnn_xyz_r2 | r2 | anchored | 1 | 0.629 | 0.479 | 0.103 | -0.132 +-0.142 | 0.389 | 0.158 | 0.455 | 0.677 | 0.446 | 1312 | 0.25 | 0.06 | 1.5 |
| dgcnn_xyz_r2 | r2 | anchored_split | 1 | 0.629 | 0.479 | 0.103 | -0.132 +-0.142 | 0.389 | 0.158 | 0.455 | 0.677 | 0.446 | 1312 | 0.25 | 0.06 | 1.5 |
| dgcnn_xyzrgb_r2 | r2 | plain | 2 | 0.593 (0.57-0.62) | 0.482 (0.48-0.48) | 0.111 | -0.080 +-0.149 | 0.335 (0.33-0.34) | 0.150 (0.14-0.16) | 0.500 (0.45-0.55) | 0.790 (0.77-0.81) | 0.485 (0.40-0.57) | 1312 | 0.25 | 0.06 | 1.6 |
| dgcnn_xyzrgb_r2 | r2 | anchored | 2 | 0.593 (0.57-0.62) | 0.521 (0.51-0.53) | 0.105 | -0.047 +-0.142 | 0.364 (0.35-0.38) | 0.202 (0.18-0.23) | 0.545 (0.55-0.55) | 0.792 (0.76-0.82) | 0.484 (0.40-0.57) | 1312 | 0.25 | 0.06 | 1.6 |
| dgcnn_xyzrgb_r2 | r2 | anchored_split | 2 | 0.593 (0.57-0.62) | 0.514 (0.50-0.53) | 0.107 | -0.061 +-0.148 | 0.342 (0.32-0.36) | 0.205 (0.18-0.23) | 0.455 (0.45-0.45) | 0.807 (0.79-0.82) | 0.508 (0.44-0.57) | 1312 | 0.25 | 0.06 | 1.6 |
| full_r1 | r1 | plain | 2 | 0.880 (0.86-0.90) | 0.530 (0.52-0.54) | 0.072 | -0.062 +-0.103 | 0.428 (0.41-0.45) | 0.270 (0.25-0.29) | 0.318 (0.27-0.36) | 0.731 (0.70-0.76) | 0.198 (0.19-0.21) | 1312 | 0.69 | 0.24 | 5.6 |
| full_r1 | r1 | anchored | 2 | 0.880 (0.86-0.90) | 0.627 (0.60-0.65) | 0.073 | +0.047 +-0.087 | 0.611 (0.57-0.65) | 0.325 (0.30-0.35) | 0.636 (0.64-0.64) | 0.747 (0.71-0.79) | 0.176 (0.16-0.19) | 1312 | 0.69 | 0.24 | 5.6 |
| full_r1 | r1 | anchored_split | 2 | 0.880 (0.86-0.90) | 0.595 (0.57-0.62) | 0.062 | -0.001 +-0.097 | 0.553 (0.52-0.59) | 0.325 (0.30-0.35) | 0.455 (0.45-0.45) | 0.747 (0.71-0.79) | 0.183 (0.17-0.20) | 1312 | 0.69 | 0.24 | 5.6 |
| full_r1 | r1 | anchored_nob | 2 | 0.880 (0.86-0.90) | 0.627 (0.60-0.65) | 0.073 | +0.047 +-0.087 | 0.611 (0.57-0.65) | 0.325 (0.30-0.35) | 0.636 (0.64-0.64) | 0.747 (0.71-0.79) | 0.176 (0.16-0.19) | 1312 | 0.69 | 0.24 | 5.6 |
| full_r2 | r2 | plain | 2 | 0.874 (0.87-0.88) | 0.552 (0.52-0.58) | 0.080 | -0.026 +-0.106 | 0.456 (0.44-0.47) | 0.246 (0.19-0.31) | 0.455 (0.36-0.55) | 0.742 (0.72-0.76) | 0.179 (0.17-0.19) | 1312 | 0.69 | 0.21 | 5.2 |
| full_r2 | r2 | anchored | 2 | 0.874 (0.87-0.88) | 0.626 (0.59-0.66) | 0.069 | +0.048 +-0.091 | 0.590 (0.56-0.62) | 0.304 (0.27-0.34) | 0.591 (0.55-0.64) | 0.750 (0.73-0.77) | 0.170 (0.17-0.17) | 1312 | 0.69 | 0.21 | 5.2 |
| full_r2 | r2 | anchored_split | 2 | 0.874 (0.87-0.88) | 0.617 (0.59-0.64) | 0.068 | +0.031 +-0.096 | 0.575 (0.56-0.59) | 0.304 (0.27-0.34) | 0.591 (0.55-0.64) | 0.750 (0.73-0.77) | 0.171 (0.17-0.17) | 1312 | 0.69 | 0.21 | 5.2 |
| full_r2 | r2 | anchored_nob | 2 | 0.874 (0.87-0.88) | 0.626 (0.59-0.66) | 0.069 | +0.048 +-0.091 | 0.590 (0.56-0.62) | 0.304 (0.27-0.34) | 0.591 (0.55-0.64) | 0.750 (0.73-0.77) | 0.170 (0.17-0.17) | 1312 | 0.69 | 0.21 | 5.2 |
| nocorrupt_r2 | r2 | plain | 1 | 0.855 | 0.563 | 0.082 | -0.007 +-0.131 | 0.454 | 0.258 | 0.364 | 0.732 | 0.194 | 1312 | 0.68 | 0.23 | 5.7 |
| nocorrupt_r2 | r2 | anchored | 1 | 0.855 | 0.657 | 0.073 | +0.103 +-0.112 | 0.657 | 0.310 | 0.636 | 0.761 | 0.159 | 1312 | 0.68 | 0.23 | 5.7 |
| nocorrupt_r2 | r2 | anchored_split | 1 | 0.855 | 0.624 | 0.070 | +0.053 +-0.125 | 0.596 | 0.310 | 0.455 | 0.761 | 0.167 | 1312 | 0.68 | 0.23 | 5.7 |
| nodropout_r2 | r2 | plain | 1 | 0.853 | 0.560 | 0.090 | +0.001 +-0.131 | 0.465 | 0.315 | 0.455 | 0.770 | 0.175 | 1122 | 0.68 | 0.31 | 6.4 |
| nodropout_r2 | r2 | anchored | 1 | 0.853 | 0.640 | 0.078 | +0.069 +-0.120 | 0.618 | 0.354 | 0.636 | 0.790 | 0.173 | 1122 | 0.68 | 0.31 | 6.4 |
| nodropout_r2 | r2 | anchored_split | 1 | 0.853 | 0.620 | 0.075 | +0.043 +-0.125 | 0.571 | 0.354 | 0.455 | 0.790 | 0.181 | 1122 | 0.68 | 0.31 | 6.4 |
| nolift_r2 | r2 | plain | 2 | 0.808 (0.79-0.83) | 0.497 (0.48-0.51) | 0.098 | -0.102 +-0.144 | 0.379 (0.36-0.40) | 0.226 (0.21-0.24) | 0.409 (0.36-0.45) | 0.694 (0.69-0.70) | 0.308 (0.30-0.31) | 1233 | 0.69 | 0.25 | 5.5 |
| nolift_r2 | r2 | anchored | 2 | 0.808 (0.79-0.83) | 0.570 (0.55-0.59) | 0.090 | -0.031 +-0.133 | 0.477 (0.47-0.48) | 0.288 (0.27-0.31) | 0.545 (0.55-0.55) | 0.694 (0.69-0.70) | 0.298 (0.28-0.31) | 1233 | 0.69 | 0.25 | 5.5 |
| nolift_r2 | r2 | anchored_split | 2 | 0.808 (0.79-0.83) | 0.562 (0.55-0.57) | 0.087 | -0.039 +-0.133 | 0.466 (0.46-0.47) | 0.288 (0.27-0.31) | 0.500 (0.45-0.55) | 0.694 (0.69-0.70) | 0.300 (0.29-0.31) | 1233 | 0.69 | 0.25 | 5.5 |
| norgb_r2 | r2 | plain | 2 | 0.818 (0.81-0.83) | 0.514 (0.50-0.53) | 0.085 | -0.082 +-0.149 | 0.436 (0.42-0.46) | 0.242 (0.24-0.25) | 0.409 (0.36-0.45) | 0.763 (0.76-0.77) | 0.268 (0.26-0.27) | 1248 | 0.69 | 0.23 | 5.2 |
| norgb_r2 | r2 | anchored | 2 | 0.818 (0.81-0.83) | 0.606 (0.60-0.61) | 0.072 | +0.017 +-0.128 | 0.571 (0.54-0.60) | 0.323 (0.32-0.32) | 0.636 (0.64-0.64) | 0.772 (0.77-0.77) | 0.255 (0.24-0.27) | 1248 | 0.69 | 0.23 | 5.2 |
| norgb_r2 | r2 | anchored_split | 2 | 0.818 (0.81-0.83) | 0.592 (0.59-0.60) | 0.067 | +0.003 +-0.129 | 0.544 (0.51-0.57) | 0.323 (0.32-0.32) | 0.545 (0.55-0.55) | 0.772 (0.77-0.77) | 0.260 (0.25-0.27) | 1248 | 0.69 | 0.23 | 5.2 |
| scratch_r2 | r2 | plain | 1 | 0.726 | 0.441 | 0.084 | -0.132 +-0.135 | 0.193 | 0.147 | 0.091 | 0.773 | 0.467 | 1312 | 0.69 | 0.17 | 4.2 |
| scratch_r2 | r2 | anchored | 1 | 0.726 | 0.522 | 0.074 | -0.069 +-0.125 | 0.244 | 0.250 | 0.182 | 0.732 | 0.454 | 1312 | 0.69 | 0.17 | 4.2 |
| scratch_r2 | r2 | anchored_split | 1 | 0.726 | 0.509 | 0.071 | -0.088 +-0.128 | 0.230 | 0.252 | 0.091 | 0.732 | 0.465 | 1312 | 0.69 | 0.17 | 4.2 |


Skipped by the driver: nodropout_r2 seed 1 (budget (driver stopped by hand to make room for DGCNN re-runs)); nocorrupt_r2 seed 1 (budget); dgcnn_xyz_r2 seed 1 (budget); scratch_r2 seed 1 (budget); unfreeze_top_r2 seed 0 (GPU not free: other processes (Blender synth job) held 1.8-4 GB)

<!-- AUTO:END -->

## Reading (R2 unless noted; anchored decode = the "refine" metric, plain = replace)

**Baselines.** Lift_label: 0.780 (R1) / 0.648 (R2) instance mIoU, exact-count 0.64, semantic mIoU 0.826 / 0.810. "Trivially
anchored" lift is identical by construction (checks the decoder reproduces the prior).

**Headline.** The refiner does not beat the lift on instances in either regime. Plain decoding is much worse than lift (R1 0.55, R2
0.55-0.59 vs 0.78 / 0.65) and gets the count right on 36-55 % of scans vs 64 % for lift. Anchored decoding recovers the lift level
(R1 0.764-0.768 vs 0.780; R2 0.626 (0.59-0.66) vs 0.648): the paired per-scan difference is -0.02 +-0.06 (R1) and +0.05 +-0.09 (R2), i.e.
zero within noise. What the model clearly does improve is semantic mIoU (0.874-0.883 vs 0.81-0.83 lift, both seeds), and at the point
level it fixes 75-80 % of lift-wrong food-domain points. But it breaks 17 % of lift-right points in R2 (6-8 % in R1 with anchoring;
19 % with plain decoding), so net instance quality does not move. This cannot say the net is "useless": 22 items and one 1312-batch
schedule with a frozen encoder do not bound what a longer or partly unfrozen run does.

**Sonata vs scratch (R2).** Scratch: sem 0.726 vs 0.874, anchored 0.522 vs 0.626, plain 0.441 vs 0.552, breaks 45 % of right points vs 17 %.
Largest effect in the grid and the sign is consistent across metrics, but it is one seed and **confounded**: the scratch encoder trained
at `lr_backbone`=5e-5 (made for fine-tuning) and the same 1312 batches, so this measures "Sonata init vs an under-trained random
encoder", not pretraining as such. Not a clean statement about from-scratch capacity.

**No lift channel.** Sem 0.808 vs 0.874, plain inst 0.497 vs 0.552, breaks 30 % vs 17 % (2 seeds: 0.48-0.51 vs 0.52-0.58). The lift
prior is used and helps semantics; the instance difference (0.05) is at the noise edge.

**No rgb.** Sem 0.818 vs 0.874 (both seeds 0.81-0.83 vs 0.87-0.88, a real, reproducible semantic loss), plain inst 0.514 vs 0.552 and
anchored 0.606 vs 0.626 (noise). Colour helps class separation, not demonstrably instances.

**No channel dropout** (1 seed, 1122 batches, capped): anchored 0.640 vs 0.626, sem 0.853 vs 0.874. Indistinguishable; short run.
**No corruption aug at train** (1 seed): anchored 0.657, plain 0.563, sem 0.855 - not worse than with it (0.626 / 0.552). Caveat: 8
train scans are old files whose lift is GT-synthesised, so without corruption those scans carry a perfect lift. The corruption
augmentation has no demonstrated benefit here; neither does training on R2 vs R1 (full_r1 vs full_r2 in both eval regimes: 0.768 vs 0.764
in R1, 0.627 vs 0.626 in R2).

**DGCNN (after the BN fix; no lift input, so read plain).** xyz: sem 0.629, inst 0.384 ; xyz+rgb (2 seeds): sem 0.593 (0.57-0.62), inst 0.482
(0.48-0.48). PTv3 without lift (comparable inputs plus normals): sem 0.808, inst 0.497. So PTv3-Sonata is far ahead on semantics (+0.2);
on instances the gap to xyz+rgb DGCNN is 0.015, i.e. nothing. Both use 4096 sampled points, so they are not on equal footing with
the 20k-voxel PTv3.

**Decoders.** `anchored_split` is never better than `anchored` (18 config x regime pairs: lower in 16, equal in 2; e.g. full_r2 R1 0.70 vs 0.76, R2 0.62 vs 0.63): the predicted clusters are not reliable enough to split a lift merge, even though R2 merges two instances in 60 % of
scans. The anchored decode's gain over plain (+0.04 to +0.22 instance mIoU, larger in R1) is mostly "do not damage the lift".

**Not run.** `unfreeze=top` (GPU was never < 1 GB free), second seeds for nodropout, nocorrupt, scratch, dgcnn_xyz (stopped at the
time budget to make room for the DGCNN re-runs). Single-seed rows have no spread.

**Cost.** PTv3 frozen, patch 512, 20k voxels: peak 0.69 GiB allocated during training (fp32 eval peaks higher, ~1 GiB), 0.17-0.32 s/batch
(the range is contention, not config), 4-6 min wall incl. val; DGCNN 0.25 GiB, 0.06 s/batch, 1.5 min. "scratch" used `unfreeze=all` and
still peaked at 0.69 GiB at patch 512.

## Biggest surprise

The semantic head gains 0.05-0.07 mIoU over the lift while instance quality does not move: the bottleneck is the instance decoding
(offset + DBSCAN gives the correct item count on 36-55 % of val scans against 64 % for the raw lift), not the point features.
A runner-up: the DGCNN baseline was silently broken (BatchNorm running stats) and its smoke test would not have shown it.

## Three experiments for hard_v11 (noisy pose, more scans)

1. **Statistical power first**: >= 60-100 `ok` val scans (by food asset, not only by scan) and a paired bootstrap of anchored-vs-lift
   per scan; with the current 22 items a +0.03 effect is undetectable. Re-run full vs nolift vs norgb vs scratch (with a proper
   backbone lr, e.g. 5e-4 for random init) x 3 seeds - only then are the ablation rows interpretable.
2. **Train what the anchored decoder uses**: add a per-point "keep lift / change" and a "same-instance-as-neighbour" objective (or a
   learned affinity for splitting merges) rather than offsets+DBSCAN, and evaluate corrected/broken directly; test `unfreeze=top` once
   the GPU is free. Target: corr > 0.8 with broke < 0.05 in R2.
3. **Does synthetic corruption predict real lift errors?** On noisy-pose v11 compare (a) training on the stored noisy lift only,
   (b) + `HARD_LIFT` augmentation, (c) none, evaluated on stored v11 lift, and on any real SAM 3 lifts available. This settles
   whether the corruption suite (merge/blob/drop/morph) is worth keeping, which this study could not (nocorrupt >= full at n=1).

## Round 2 (with hard_v11, split_v2)

**Data.** Pool = hard 6, hard_v3 6, hard_v9 20, hard_v10 20, hard_v11 19 (v11 s000001 has an empty fused cloud and is excluded) = 71 scans. hard_v11
was fused with the worktree's current seg_export (`--votes pseudo`): of 143 items 112 are dropped by the manifest (2 invisible, the rest < 50 fused
points), and only 1 of 20 scans is `ok` (s000017, 2 items), so v11 contributes almost only training data. `results/seg3d/split_v2.json` (seed 0,
`pipeline/seg3d/ablate_v2.py --split`): val = 40 % of the 26 `ok` scans = **10 scans / 15 items in cloud** (hard 1, v3 1, v9 3, v10 5, v11 0),
train 61 scans / 131 items in cloud (16 ok). Greedy category-disjoint pick: 10 val categories, **5 still leak into train**
(blueberries, brown rice, egg whites, lettuce, spinach (raw)); the non-ok scans in train cover 54 categories so full disjointness was infeasible.
So the val set is *smaller* than round 1 (11 scans / 22 items), not larger: the `ok` bottleneck, not scan count, limits it.
Schedule: frozen Sonata, 8 epochs x 244 batches, `max_minutes=6`, grad_accum 2, **patch_size 1024** (GPU had > 3 GB free at launch), runs of 1743-1952
batches. Eval of `last.pt` on both lift regimes. Seeds: 2 each. `unfreeze=top` skipped (others held 2.5 GB, rule < 1.5 GB). Report: `python -m seg3d.report_v2`.
CI = paired per-scan bootstrap (1000 resamples, seeds averaged per scan first) of model-minus-lift instance mIoU; corrected/broken are food-domain point fractions.


#### Eval regime R1

| arm | seeds | decode | sem mIoU | inst mIoU | PQ | count | d vs lift (95% CI) | corrected | broken |
|---|---|---|---|---|---|---|---|---|---|
| lift_label | - | - | 0.705 | 0.968 | 0.907 | 0.80 | - | 0 | 0 |
| full_r1 (r1-trained) | 2 | plain | 0.840 | 0.592 | 0.454 | 0.65 | -0.294 [-0.479, -0.121] | 0.958 | 0.273 |
| full_r1 (r1-trained) | 2 | anchored | 0.840 | 0.776 | 0.718 | 0.90 | -0.156 [-0.272, -0.060] | 0.959 | 0.073 |
| full_r2 (r2-trained) | 2 | plain | 0.840 | 0.541 | 0.422 | 0.60 | -0.334 [-0.544, -0.150] | 0.961 | 0.326 |
| full_r2 (r2-trained) | 2 | anchored | 0.840 | 0.666 | 0.623 | 0.80 | -0.221 [-0.409, -0.065] | 0.964 | 0.117 |
| nolift_r1 (r1-trained) | 2 | plain | 0.752 | 0.427 | 0.321 | 0.45 | -0.500 [-0.698, -0.308] | 0.994 | 0.455 |
| nolift_r1 (r1-trained) | 2 | anchored | 0.752 | 0.641 | 0.592 | 0.90 | -0.317 [-0.518, -0.144] | 0.997 | 0.240 |
| nolift_r2 (r2-trained) | 2 | plain | 0.740 | 0.436 | 0.308 | 0.35 | -0.482 [-0.663, -0.299] | 0.995 | 0.432 |
| nolift_r2 (r2-trained) | 2 | anchored | 0.740 | 0.613 | 0.543 | 0.75 | -0.323 [-0.511, -0.139] | 0.996 | 0.246 |

#### Eval regime R2

| arm | seeds | decode | sem mIoU | inst mIoU | PQ | count | d vs lift (95% CI) | corrected | broken |
|---|---|---|---|---|---|---|---|---|---|
| lift_label | - | - | 0.675 | 0.713 | 0.703 | 0.70 | - | 0 | 0 |
| full_r1 (r1-trained) | 2 | plain | 0.818 | 0.574 | 0.468 | 0.65 | -0.051 [-0.244, +0.175] | 0.908 | 0.350 |
| full_r1 (r1-trained) | 2 | anchored | 0.818 | 0.700 | 0.705 | 0.90 | +0.048 [-0.135, +0.262] | 0.911 | 0.061 |
| full_r2 (r2-trained) | 2 | plain | 0.842 | 0.534 | 0.429 | 0.60 | -0.076 [-0.286, +0.148] | 0.943 | 0.398 |
| full_r2 (r2-trained) | 2 | anchored | 0.842 | 0.647 | 0.641 | 0.85 | +0.028 [-0.222, +0.302] | 0.946 | 0.098 |
| nolift_r1 (r1-trained) | 2 | plain | 0.749 | 0.439 | 0.339 | 0.45 | -0.219 [-0.483, +0.030] | 0.919 | 0.446 |
| nolift_r1 (r1-trained) | 2 | anchored | 0.749 | 0.589 | 0.552 | 0.85 | -0.071 [-0.365, +0.261] | 0.922 | 0.127 |
| nolift_r2 (r2-trained) | 2 | plain | 0.736 | 0.436 | 0.315 | 0.35 | -0.210 [-0.457, +0.052] | 0.926 | 0.411 |
| nolift_r2 (r2-trained) | 2 | anchored | 0.736 | 0.588 | 0.576 | 0.70 | -0.063 [-0.310, +0.249] | 0.928 | 0.141 |

**Reading.**
1. In R2 (hard lift, the headline regime) lift-anchored full_r1/full_r2 are +0.05/+0.03 over lift with CIs [-0.14,+0.26]/[-0.22,+0.30]: still zero within noise, as in round 1 (+0.05 +-0.09). Plain decoding is again worse than lift (point -0.05 to -0.08, CI includes 0).
2. In R1 (stored lift, lift mIoU 0.968 on this val) every arm is significantly *below* lift (anchored -0.16 to -0.22, CIs exclude 0): the stored lift is nearly perfect on these 10 scans, so a refiner can only break it. This differs from round 1 (-0.03 +-0.06, n.s.), mostly because the lift baseline rose 0.78 -> 0.97 with the new val composition.
3. The semantic gain survives (0.84 vs 0.68-0.71 lift) and the net still fixes 91-96 % of lift-wrong points, but breaks 6-12 % (anchored) / 33-40 % (plain) of right ones.
4. No-lift-channel is worse than full in both regimes (sem 0.74-0.75 vs 0.84, broken 0.13-0.25 vs 0.06-0.12), same direction as round 1 but now clearer on semantics; instance differences stay inside the CIs. R1- vs R2-training: indistinguishable.
5. Conclusions unchanged: no demonstrated instance gain over lift. Power did not improve (val 15 items, CIs +-0.2); the experiment needs more `ok` scans (v11 yields 1 of 20), not more raw scans. The patch 1024 vs 512 change is a further difference from round 1.


## Keep-or-change head (decoder=koc)

**What.** `pipeline/seg3d/koc.py`: per point, candidates = own lift label + the K=4 nearest distinct other lift labels within 25 mm (background,
253, 254 and food ids all eligible) + a "none" slot. Score = MLP over (point feature, mean projected feature / lift_share / nvotes of the 16 nearest
points carrying the candidate label within r, role+class one-hots, own share/nvotes, distance, member fraction, detached semantic-head log-prob of
the candidate's class); softmax; learnable KEEP bias (init +2); loss = CE smoothed 5 % toward keep, change points weighted x2. Decode = argmax,
none -> keep; `koc2` = re-build candidates from the updated labels and re-run the head once. Replaces offset+DBSCAN when `--set decoder=koc`
(offset head is still trained and `plain`/`anchored` remain evaluable on the same run).

**Run.** `v2_koc_r2_s0`: R2-trained, split_v2, frozen Sonata, patch 512, 1 seed, 6 min cap, same schedule as round 2 except that **only 621 batches
ran (round-2 arms: 1743-1952)**: the GPU was shared with a Blender job and the candidate build adds CPU per sample. The semantic head is
correspondingly under-trained (sem mIoU of the head: 0.54 vs 0.84 in round 2), and its log-prob is an input of the KOC head. Peak 0.83 GiB train.
Paired bootstrap as in round 2 (`python -m seg3d.report_v2`), model-minus-lift instance mIoU, corrected / broken = food-domain point fractions.

| eval | decode | inst mIoU | PQ | count | d vs lift (95% CI) | corrected | broken | keep-rate | unrecoverable (of lift-wrong) |
|---|---|---|---|---|---|---|---|---|---|
| R2 | lift_label | 0.713 | 0.703 | 0.70 | - | 0 | 0 | 1 | - |
| R2 | anchored (full_r2, 2 seeds) | 0.647 | 0.641 | 0.85 | +0.028 [-0.199, +0.307] | 0.946 | 0.098 | - | - |
| R2 | koc | 0.609 | 0.593 | 0.80 | -0.057 [-0.152, +0.019] | 0.623 | 0.036 | 0.934 | 0.047 (0.411) |
| R2 | koc2 | 0.626 | 0.593 | 0.80 | -0.031 [-0.157, +0.096] | 0.772 | 0.037 | 0.918 | 0.047 (0.411) |
| R1 | lift_label | 0.968 | 0.907 | 0.80 | - | 0 | 0 | 1 | - |
| R1 | anchored (full_r2, 2 seeds) | 0.666 | 0.623 | 0.80 | -0.221 [-0.409, -0.065] | 0.964 | 0.117 | - | - |
| R1 | koc | 0.749 | 0.671 | 0.80 | -0.139 [-0.325, -0.018] | 0.721 | 0.074 | 0.937 | 0.033 (0.337) |
| R1 | koc2 | 0.748 | 0.669 | 0.80 | -0.141 [-0.318, -0.018] | 0.884 | 0.082 | 0.923 | 0.033 (0.337) |

**Reading / caveats.** (1) Breakage drops (R2: 3.6 % vs 9.8 % anchored, 40 % plain) at the price of correction (62 % vs 95 %); the second iteration
recovers part of it (77 %) at no extra breakage. Instance mIoU is still not above lift (R2 -0.03 to -0.06, CIs include 0; R1 below lift, CI excludes 0),
and sits slightly below the round-2 anchored numbers (R2 0.61-0.63 vs 0.65), so no win is demonstrated. (2) The ceiling matters: 41 % (R2) / 34 % (R1)
of lift-wrong points have no correct label among the candidates (merged-away ids, dropped instances, blobs > 25 mm from the right instance), so
corrected is capped well below 1 for a KOC decoder and the numbers are not comparable to the anchored decoder's, which can spawn/assign ids from
clusters. Raising K/r or adding a "new instance" decode would be the next lever. (3) 1 seed, 15 val items, a short under-trained run (see above) on a
shared GPU; keep-bias / smoothing / change weight are untuned. (4) The KOC decode also sets the class from the winning label, so its sem mIoU (0.77-0.81,
0.81-0.86 with koc2) is not the semantic head's. (5) Ids are matched literally (true for synthetic lift; real SAM-lift ids would need a GT-matching step in the target).


## Round 3 (hard_v12, item-level filtering, split_v3)

**What changed.** (1) `data.item_filter=manifest|points|none` (default manifest; `data.item_min_points`=50): items in the manifest's `dropped_items`
(tiny/invisible; for batches whose manifest has no such key, hard and hard_v3, the rule is < 50 fused points) are *ignored*: loss weight 0 on their
points, and their points are removed from every metric (semantic, instance, count, corrected/broken), instead of dropping the whole scan. The points stay
in the network input (and in the lift). A scan is usable if it has >= 1 valid item. Test: `tests/test_seg3d_item_filter.py` (17 seg3d tests pass).
(2) hard_v12 = 20 scans, 24 views, noisy poses, pseudo-vote lift; 146/177 items valid (manifest `ok` said 5/20 scans). (3) `results/seg3d/split_v3.json`
(`python -m seg3d.ablate_v3 --split`, seed 0): val = 35 % of usable scans stratified by batch (per-batch round), greedy for category overlap.
(4) Schedule = round 2 but **patch 512**, frozen Sonata, 8 epochs x 188 batches (1504 iters, ~4 min) for plain-head runs; the koc run is CPU-bound
(candidate build) and the 6-min cap stopped it at **475 iters** (31 % of the plain schedule), so it is under-trained as in round 2. Peak GPU 0.98 GiB
(plain) / 1.61 GiB (koc) allocated. Evaluation: `last.pt`, both regimes, `item_inst_iou` per item now stored in `final.json`.
Run: `python -m seg3d.ablate_v3 --run --xeval`; table: `python -m seg3d.report_v3` (-> `round3.json`).

**Data (split_v3; valid = after item filter).**

| batch | scans | usable (>=1 valid item) | valid items | ignored items | train scans / valid items | val scans / valid items |
|---|---|---|---|---|---|---|
| hard | 6 | 6 | 13 | 4 | 4 / 10 | 2 / 3 |
| hard_v3 | 6 | 6 | 13 | 3 | 4 / 10 | 2 / 3 |
| hard_v9 | 20 | 13 | 18 | 16 | 8 / 13 | 5 / 5 |
| hard_v10 | 20 | 13 | 19 | 8 | 8 / 13 | 5 / 6 |
| hard_v11 | 19 | 16 | 31 | 21 | 10 / 23 | 6 / 8 |
| hard_v12 | 20 | 20 | 146 | 13 | 13 / 101 | 7 / 45 |
| **total** | 91 | 74 | 240 | 52 | 47 / 170 | 27 / 70 |

17 scans (hard_v9 7, hard_v10 7, hard_v11 3) have no valid item and are unused. Val is **27 scans / 70 valid items** (round 2: 10 / 15; round 1: 11 / 22),
but 45 of its 70 items come from 7 hard_v12 scans (the other 5 batches have 1-2 valid items per scan). Categories: train 48, val 32; **7 val categories
still occur in train** (broccoli, cherry tomato, green curry, oatmeal, rice, risotto, yeast_bread); full disjointness is infeasible with this pool.
Lift baseline on this val: R1 inst mIoU 0.773 / sem 0.687 / count 0.74; R2 0.638 / 0.674 / 0.63.

**Results.** Instance mIoU pooled over the 70 items; d vs lift = paired bootstrap (2000 resamples) of model-minus-lift, per scan (seeds averaged per scan) and per
item (cluster bootstrap over scans); corrected / broken = food-domain point fractions; keep = share of points whose lift label is kept (koc only).
Seeds: full_r2 2, nolift_r2 2, koc_r2 1 (second seed skipped for time). Training regime R2 throughout; "plain" = offset+DBSCAN, "anchored" = lift-anchored.

#### Round-3 runs (trained on split_v3), eval regime R1

| arm | seeds | decode | sem mIoU | inst mIoU | PQ | count | d vs lift per scan (95% CI) | d vs lift per item (95% CI) | corrected | broken | keep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lift_label | - | - | 0.687 | 0.773 | 0.704 | 0.74 | - | - | 0 | 0 | 1 |
| full_r2 | 2 | plain | 0.751 | 0.500 | 0.451 | 0.56 | -0.179 [-0.279, -0.085] | -0.272 [-0.409, -0.119] | 0.870 | 0.279 | - |
| full_r2 | 2 | anchored | 0.751 | 0.663 | 0.604 | 0.74 | -0.076 [-0.143, -0.007] | -0.110 [-0.176, -0.024] | 0.870 | 0.172 | - |
| koc_r2 | 1 | plain | 0.723 | 0.516 | 0.389 | 0.56 | -0.190 [-0.299, -0.081] | -0.257 [-0.386, -0.121] | 0.721 | 0.290 | - |
| koc_r2 | 1 | anchored | 0.723 | 0.660 | 0.560 | 0.67 | -0.094 [-0.182, -0.013] | -0.113 [-0.182, -0.032] | 0.722 | 0.188 | - |
| koc_r2 | 1 | koc | 0.735 | 0.768 | 0.706 | 0.74 | -0.003 [-0.011, +0.007] | -0.004 [-0.020, +0.015] | 0.338 | 0.013 | 0.965 |
| koc_r2 | 1 | koc2 | 0.753 | 0.768 | 0.705 | 0.74 | -0.004 [-0.013, +0.005] | -0.005 [-0.022, +0.015] | 0.469 | 0.021 | 0.950 |
| nolift_r2 | 2 | plain | 0.685 | 0.477 | 0.426 | 0.52 | -0.267 [-0.396, -0.151] | -0.295 [-0.425, -0.158] | 0.872 | 0.337 | - |
| nolift_r2 | 2 | anchored | 0.685 | 0.624 | 0.537 | 0.67 | -0.171 [-0.280, -0.077] | -0.149 [-0.223, -0.065] | 0.872 | 0.245 | - |

#### Round-3 runs (trained on split_v3), eval regime R2

| arm | seeds | decode | sem mIoU | inst mIoU | PQ | count | d vs lift per scan (95% CI) | d vs lift per item (95% CI) | corrected | broken | keep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lift_label | - | - | 0.674 | 0.638 | 0.599 | 0.63 | - | - | 0 | 0 | 1 |
| full_r2 | 2 | plain | 0.753 | 0.499 | 0.461 | 0.57 | -0.055 [-0.196, +0.082] | -0.139 [-0.290, +0.031] | 0.839 | 0.254 | - |
| full_r2 | 2 | anchored | 0.753 | 0.575 | 0.564 | 0.74 | +0.003 [-0.103, +0.113] | -0.063 [-0.142, +0.040] | 0.832 | 0.176 | - |
| koc_r2 | 1 | plain | 0.716 | 0.519 | 0.375 | 0.52 | -0.032 [-0.193, +0.117] | -0.119 [-0.282, +0.050] | 0.698 | 0.274 | - |
| koc_r2 | 1 | anchored | 0.716 | 0.582 | 0.505 | 0.70 | +0.017 [-0.107, +0.147] | -0.056 [-0.138, +0.053] | 0.691 | 0.199 | - |
| koc_r2 | 1 | koc | 0.727 | 0.640 | 0.604 | 0.63 | +0.028 [-0.012, +0.085] | +0.002 [-0.028, +0.040] | 0.317 | 0.012 | 0.962 |
| koc_r2 | 1 | koc2 | 0.742 | 0.639 | 0.603 | 0.63 | +0.027 [-0.014, +0.083] | +0.001 [-0.028, +0.039] | 0.447 | 0.025 | 0.947 |
| nolift_r2 | 2 | plain | 0.683 | 0.482 | 0.431 | 0.52 | -0.117 [-0.261, +0.018] | -0.156 [-0.304, -0.001] | 0.841 | 0.316 | - |
| nolift_r2 | 2 | anchored | 0.683 | 0.540 | 0.488 | 0.74 | -0.084 [-0.204, +0.020] | -0.098 [-0.175, -0.003] | 0.834 | 0.261 | - |

#### Round-2 checkpoints evaluated on split_v3 val (no retraining), eval regime R1

| arm | seeds | decode | sem mIoU | inst mIoU | PQ | count | d vs lift per scan (95% CI) | d vs lift per item (95% CI) | corrected | broken | keep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lift_label | - | - | 0.687 | 0.773 | 0.704 | 0.74 | - | - | 0 | 0 | 1 |
| full_r2 | 2 | plain | 0.673 | 0.503 | 0.433 | 0.57 | -0.173 [-0.266, -0.083] | -0.270 [-0.394, -0.120] | 0.733 | 0.285 | - | unseen-by-v2 (10 scans): -0.262 [-0.399, -0.143]
| full_r2 | 2 | anchored | 0.673 | 0.678 | 0.625 | 0.78 | -0.043 [-0.093, +0.013] | -0.095 [-0.166, -0.012] | 0.733 | 0.174 | - | unseen-by-v2 (10 scans): -0.119 [-0.186, -0.058]
| koc_r2 | 1 | plain | 0.557 | 0.432 | 0.349 | 0.44 | -0.317 [-0.414, -0.211] | -0.341 [-0.475, -0.208] | 0.887 | 0.334 | - | unseen-by-v2 (10 scans): -0.299 [-0.429, -0.184]
| koc_r2 | 1 | anchored | 0.557 | 0.591 | 0.531 | 0.78 | -0.177 [-0.262, -0.093] | -0.181 [-0.250, -0.096] | 0.887 | 0.228 | - | unseen-by-v2 (10 scans): -0.197 [-0.270, -0.121]
| koc_r2 | 1 | koc | 0.708 | 0.703 | 0.632 | 0.74 | -0.091 [-0.148, -0.044] | -0.069 [-0.102, -0.041] | 0.372 | 0.085 | 0.945 | unseen-by-v2 (10 scans): -0.059 [-0.089, -0.030]
| koc_r2 | 1 | koc2 | 0.703 | 0.691 | 0.615 | 0.74 | -0.103 [-0.167, -0.054] | -0.082 [-0.122, -0.051] | 0.533 | 0.119 | 0.919 | unseen-by-v2 (10 scans): -0.083 [-0.129, -0.040]
| nolift_r2 | 2 | plain | 0.599 | 0.464 | 0.377 | 0.50 | -0.286 [-0.412, -0.167] | -0.308 [-0.445, -0.162] | 0.872 | 0.360 | - | unseen-by-v2 (10 scans): -0.327 [-0.464, -0.192]
| nolift_r2 | 2 | anchored | 0.599 | 0.623 | 0.513 | 0.70 | -0.164 [-0.264, -0.073] | -0.150 [-0.226, -0.067] | 0.870 | 0.254 | - | unseen-by-v2 (10 scans): -0.193 [-0.301, -0.094]

#### Round-2 checkpoints evaluated on split_v3 val (no retraining), eval regime R2

| arm | seeds | decode | sem mIoU | inst mIoU | PQ | count | d vs lift per scan (95% CI) | d vs lift per item (95% CI) | corrected | broken | keep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lift_label | - | - | 0.674 | 0.638 | 0.599 | 0.63 | - | - | 0 | 0 | 1 |
| full_r2 | 2 | plain | 0.668 | 0.494 | 0.445 | 0.59 | -0.050 [-0.172, +0.074] | -0.144 [-0.287, +0.024] | 0.699 | 0.267 | - | unseen-by-v2 (10 scans): -0.102 [-0.286, +0.078]
| full_r2 | 2 | anchored | 0.668 | 0.584 | 0.566 | 0.74 | +0.022 [-0.053, +0.114] | -0.054 [-0.129, +0.047] | 0.692 | 0.193 | - | unseen-by-v2 (10 scans): -0.053 [-0.147, +0.051]
| koc_r2 | 1 | plain | 0.556 | 0.432 | 0.346 | 0.48 | -0.172 [-0.302, -0.049] | -0.206 [-0.350, -0.050] | 0.868 | 0.339 | - | unseen-by-v2 (10 scans): -0.122 [-0.322, +0.071]
| koc_r2 | 1 | anchored | 0.556 | 0.528 | 0.499 | 0.78 | -0.069 [-0.145, +0.014] | -0.110 [-0.183, -0.020] | 0.860 | 0.240 | - | unseen-by-v2 (10 scans): -0.094 [-0.200, +0.031]
| koc_r2 | 1 | koc | 0.696 | 0.593 | 0.534 | 0.63 | -0.053 [-0.101, -0.013] | -0.045 [-0.070, -0.022] | 0.342 | 0.081 | 0.942 | unseen-by-v2 (10 scans): -0.016 [-0.052, +0.028]
| koc_r2 | 1 | koc2 | 0.693 | 0.586 | 0.528 | 0.63 | -0.060 [-0.109, -0.020] | -0.052 [-0.079, -0.029] | 0.494 | 0.106 | 0.917 | unseen-by-v2 (10 scans): -0.027 [-0.063, +0.021]
| nolift_r2 | 2 | plain | 0.599 | 0.452 | 0.369 | 0.46 | -0.136 [-0.253, -0.029] | -0.186 [-0.331, -0.030] | 0.848 | 0.349 | - | unseen-by-v2 (10 scans): -0.175 [-0.355, -0.002]
| nolift_r2 | 2 | anchored | 0.599 | 0.539 | 0.469 | 0.69 | -0.068 [-0.144, +0.013] | -0.099 [-0.170, -0.012] | 0.830 | 0.255 | - | unseen-by-v2 (10 scans): -0.118 [-0.240, -0.008]

Round-2 checkpoints (`v2_*`, trained on split_v2) were evaluated unchanged. **Caveat:** 17 of the 27 val scans were in their training set, so only the
`unseen-by-v2` column (10 scans: the 7 hard_v12 scans + 3 round-2 val scans) is clean; the pooled rows are optimistic for them.

**Reading (6 lines).**
1. Val is 4.7x larger (70 items vs 15) and the CIs shrank only modestly (per-scan R2 anchored half-width ~0.11 vs ~0.25 in round 2; per item ~0.09): still wide, because the lift baseline is itself 0.64 and per-scan noise is large.
2. Round-1/2 conclusion "refiner does not beat the lift on instances" holds and gets a bit stronger: R2 anchored full_r2 is +0.003 per scan [-0.10, +0.11] (equal) but -0.063 per item [-0.14, +0.04]; in R1 it is significantly below (per scan -0.076 [-0.14, -0.01], per item -0.110 [-0.18, -0.02]); plain decoding is below in both (R2 CI touches 0).
3. The semantic gain persists (0.75 vs 0.67-0.69), it fixes 83-87 % of lift-wrong points but breaks 17-18 % of right ones (anchored), 25-28 % (plain); not-using-lift is worse everywhere (nolift anchored -0.084 per scan R2, sem 0.68 = lift level), as in rounds 1-2.
4. koc decoding is the only arm that does not hurt: breaks 1.2 %, keeps 96 %, corrects 32 % (45 % with koc2), inst mIoU = lift (R1 -0.003 [-0.011, +0.007], R2 +0.028 per scan [-0.012, +0.085], +0.002 per item [-0.03, +0.04]); i.e. a safe near-no-op, with an under-trained head (475 iters, 1 seed), not yet a gain.
5. Denser scans alone do not rescue the round-2 checkpoints: on the clean subset they are still at or below the lift (full_r2 anchored R2 -0.053 [-0.15, +0.05]; R1 -0.119 [-0.19, -0.06]; koc -0.016 [-0.05, +0.03]); in-training scans do not help them (pooled rows: broken 17-24 %). Round-3 retraining on v12 changes the point estimates by < 0.07.
6. Verdict: refiner (anchored) is **equal to the lift in R2 within CIs and below it in R1**; koc is equal to the lift in both. No arm is demonstrably above the lift. Confounds: 1-2 seeds, koc trained 31 % as long, val dominated by hard_v12 (64 % of items), 7 leaked categories, hard/hard_v3/v9-v11 items mostly sparse.


## Round 4 (real SAM 3 votes) - PARTIAL (1 of 3 folds of arm c; arms d, e, b not run)

**Why partial.** The coordinator's render batch took `gpu.lock` at 04:05 for 6-8 h. The fold-1 evaluation waited the agreed 15 min (`flock -w 900`), timed out (rc 99) and the driver stopped by instruction. Done: setup, arm a (CPU), arm c fold 0 trained + evaluated, arm c fold 1 trained (`r4_c_mix_f1_s0/last.pt`, 552-ish iters, **not evaluated**). Not run: c folds 1 (eval) and 2, d (fragmentation aug), e (K=8, r=50 mm), b (pseudo only). Code for all of them is in place and resumable (`python -m seg3d.ablate_v4 --run`, skips finished jobs; `--final r4_c_mix_f1_s0` evaluates the existing checkpoint).

**Data.** Real votes: `data/synth/hard_v12_sam3_v2_image/scans` (11 scans, votes mode sam3, image-mode v2 prompting; lift ids already Hungarian-matched to GT ids where a match exists, other fragments carry fresh ids, so the literal-id keep-or-change target is valid). Linked into `results/seg3d/r4_data/real_v12` with hard_v12's manifest, so the round-3 item filter applies: **49 valid items in 11 scans** (s000002 has 10, others 3-6). Folds (seed 0): f0 = s000002/09/12/13 (4), f1 = s000000/07/11/17 (4), f2 = s000001/14/16 (3). Train per fold = 41 pseudo scans of split_v3 train (pseudo copies of all 11 real scans removed; 6 of them were in split_v3 train, so round-3 checkpoints cannot be reused for arm b) + the other real scans x3 (oversampled). Real scans get no extra synthetic corruption (`aug.corrupt_real=false`); pseudo scans get the r1 corruption (p=0.5). Eval R1 = lift as stored. **This is a tiny set (11 scans / 49 items); the reported fold covers 4 scans / 14 correct-lift items.** Schedule: frozen Sonata, patch 512, koc head K=4 r=25 mm, 6 min cap = 552 iters (peak 0.83 GiB), 1 seed, `last.pt`.

**Fragmentation augmentation (arm d, implemented, not trained).** `augment.fragment_lift`: with `aug.frag_p` per scene, one lift food instance (>= 60 points) is split by a random plane (50 % of 2-piece cases) or k-means (2-3 pieces); largest piece keeps its id, others get unused ids <= 251 with lift_share 0.4-0.9; pseudo scans only. Test `tests/test_seg3d_fragment.py` (19 seg3d tests pass). Arm d = arm c + `aug.frag_p=0.3 aug.merge_p=0.1`.

**Result (fold 0 only, 4 scans, 14 items with lift IoU > 0.5, 9 GT items fragmented in the lift).** CI = paired per-scan bootstrap (2000) of model-minus-lift instance mIoU.

| arm | decode | scans | inst mIoU | PQ | count | d vs lift per scan (95% CI) | sem | corrected | broken | keep | fragmented items made whole | correct items broken | items fixed (IoU <= 0.5 -> > 0.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a lift (real votes), all 11 scans | - | 11 | 0.634 | 0.371 | 0.00 | - | 0.794 | 0 | 0 | 1 | 0 / 24 | 0 / 35 | 0 |
| a lift, same 4 scans | - | 4 | 0.618 | 0.419 | 0.00 | - | 0.907 | 0 | 0 | 1 | 0 / 9 | 0 / 14 | 0 |
| c_mix | plain | 4 | 0.437 | 0.471 | 0.00 | -0.194 [-0.306, -0.064] | 0.877 | 0.629 | 0.291 | - | 3 / 9 | 5 / 14 | 2 |
| c_mix | anchored | 4 | 0.467 | 0.405 | 0.00 | -0.165 [-0.260, -0.068] | 0.877 | 0.523 | 0.252 | - | 2 / 9 | 5 / 14 | 2 |
| c_mix | koc | 4 | 0.619 | 0.419 | 0.00 | +0.002 [-0.000, +0.003] | 0.904 | 0.021 | 0.002 | 0.998 | 0 / 9 | 0 / 14 | 0 |
| c_mix | koc2 | 4 | 0.620 | 0.420 | 0.00 | +0.003 [-0.000, +0.007] | 0.904 | 0.021 | 0.003 | 0.998 | 0 / 9 | 0 / 14 | 0 |

(corrected/broken = food-domain point fractions; exact-count is 0 for every arm including the lift because SAM 3 leaves extra fragments / unmatched instances in every scan.)

**Reading (6 lines).**
1. On real votes the refiner does NOT beat the lift: anchored/plain decoding is significantly worse (-0.17 / -0.19, CIs exclude 0), the same pattern as R1 in rounds 1-3, because it breaks 25-29 % of right points and 5 of 14 correct items.
2. No semantic gain either: sem mIoU 0.877 (plain/anchored) vs 0.907 for the lift on these 4 scans (koc 0.904).
3. Plain/anchored do fuse some fragments (3 and 2 of 9 fragmented items become whole, 2 items fixed) but break 5 correct items: net negative.
4. The 552-iteration koc head (K=4, r=25 mm) degenerates to a no-op on real votes (keep 99.8 %, corrected 2 %); the pseudo-trained keep bias dominates and fragment merging needs a neighbour's id, which the nearly-always-keep head never picks. Equal to the lift (+0.002 [-0.000, +0.003]), merges 0 fragments.
5. Which arm/augmentation matters is **not answered**: d (fragmentation aug, merge 0.1), e (K=8, r=50 mm) and b (pseudo-only) were not run; single fold, single seed, 4 scans, so the CI above is only a per-scan bootstrap of 4 numbers.
6. Verdict: no evidence the refiner beats real-vote lift; the open question is whether fragmentation-aware training (d) and the wide window (e) make koc actually merge fragments.

**To finish when the GPU is free (about 8 min per job, 9 jobs left, ~75 min):**
`cd pipeline; python -m seg3d.ablate_v4 --final r4_c_mix_f1_s0` then `python -m seg3d.ablate_v4 --run --budget-min 80` (order c f2, d f0-2, e f0-2, b) and `python -m seg3d.ablate_v4 --report` (table + `round4.json`). More real votes: `eval.sam3_synth_votes run --variant v2_image` takes ~50 s/scan; the 9 remaining hard_v12 scans and v13 add up to ~20 more scans / ~80 items: add them to `ablate_v4.REAL_SRC` handling (link into `r4_data`, regenerate folds with `--setup`) and use 5 folds; at 30+ scans per-scan CIs become informative, and a second seed for c/d becomes meaningful.


## Fragment merge (CPU baseline)

Code `pipeline/seg3d/merge.py`, tests `pipeline/tests/test_seg3d_merge.py` (4 pass), raw numbers `results/seg3d/merge/results.json` (+ `real_merged_lifts.npz`, `run.log`, `summary.md`).
No GPU, no backbone: fragment graph over the lift instances (nodes = food ids 1..251; edges = pairs within 4 mm), 18 edge features (contact ratio = seam points of the smaller side / its
boundary points, seam point count, Lab dE and RGB-histogram chi2 across the seam (points within 6 mm of the seam), normal angle, median |dz| across the seam, size ratio and log sizes, mean
lift_share / lift_nvotes of each side, hull convexity (V_a+V_b)/V_union, fraction of the seam touching the container), agglomerative merge (best edge first, features recomputed for the merged node).
Same-SAM-name flag skipped (instance names are not stored per lift id). Safety rules (all arms except `no safety`): convexity >= 0.6; seam container-fraction <= 0.5 (the
"no merge across the container" rule: a seam that mostly lies on container points is food-wall-food); no merge of a side with lift_share < 0.35 into a side with >= 400 points.
Learned arms train on edges from oracle agglomeration (every edge state met while merging GT-same pairs), labelled "same GT item" (majority GT item of each node, purity >= .5): 4435 pseudo edges
(1867 positive; 47 split_v3 train scans x 8 seeds of corrupt_lift + 1-3 `fragment_lift`) plus 216 real edges (86 positive, weight 5); probability > 0.5 merges.

**Real SAM 3 votes: 11 hard_v12 scans, 52 valid items, leave-one-scan-out** (thresholds / classifiers refit without the held-out scan; grid {"t1": [0.1, 0.2, 0.3, 0.45, 0.6, 0.8], "t2": [3, 5, 8, 12, 18, 1000000000.0], "t3": [30, 45, 60, 90, 181], "t4": [0.0015, 0.003, 0.006, 0.012, 1000000000.0]}; objective = pooled item IoU; d = paired per-scan bootstrap, 2000 resamples; "made whole" = IoU>0.5 and >=85 % of the item in one id, among items the lift splits into >=2 ids holding >=15 % each).

| arm | inst mIoU | PQ | exact count | d vs lift per scan (95% CI) | fragments made whole | correct items broken | items fixed | ids merged |
|---|---|---|---|---|---|---|---|---|
| lift | 0.632 | 0.371 | 0.00 | - | 0 / 24 | 0 / 35 | 0 | 0 |
| heuristic, untuned defaults (t1=.15,t2=15,t3=40,t4=6mm) | 0.489 | 0.460 | 0.18 | -0.100 [-0.203, +0.001] | 6 / 24 | 11 / 35 | 3 | 49 |
| heuristic, thresholds tuned leave-one-scan-out | 0.608 | 0.465 | 0.45 | -0.006 [-0.084, +0.061] | 5 / 24 | 4 / 35 | 2 | 30 |
| heuristic tuned, no safety rules | 0.600 | 0.474 | 0.36 | -0.014 [-0.103, +0.061] | 5 / 24 | 4 / 35 | 2 | 33 |
| learned HGB (pseudo + other real scans, LOSO) | 0.630 | 0.487 | 0.09 | +0.010 [-0.036, +0.052] | 5 / 24 | 3 / 35 | 3 | 28 |
| learned HGB, pseudo edges only | 0.640 | 0.391 | 0.00 | +0.007 [+0.000, +0.021] | 1 / 24 | 0 / 35 | 0 | 5 |
| learned logistic regression (pseudo + real LOSO) | 0.620 | 0.481 | 0.18 | +0.000 [-0.061, +0.065] | 6 / 24 | 5 / 35 | 3 | 31 |
| oracle merge (GT-same edges; ceiling of merging only) | 0.713 | 0.626 | 0.55 | +0.087 [+0.046, +0.135] | 9 / 24 | 0 / 35 | 4 | 41 |

Thresholds picked per fold (LOSO): {"t1": 0.2, "t2": 8, "t3": 45, "t4": 0.0015}; {"t1": 0.3, "t2": 8, "t3": 60, "t4": 0.0015}; {"t1": 0.45, "t2": 8, "t3": 60, "t4": 0.003}; {"t1": 0.6, "t2": 12, "t3": 60, "t4": 0.003}; {"t1": 0.6, "t2": 8, "t3": 60, "t4": 0.0015}; fit on all 11 scans: {"t1": 0.3, "t2": 8, "t3": 60, "t4": 0.0015} (t1 contact ratio >, t2 Lab dE <, t3 normal angle deg <, t4 height step m <). Selection is unstable across folds (5 distinct picks). A narrower first grid
(t1<=.45, t4>=3 mm) gave LOSO inst 0.653 / d +0.029 [-0.017, +0.081]; the wider grid above is what is reported (grid widened after seeing a boundary pick, so neither number is a clean estimate).

Edge classifier on the 117 labelled real edges of the unmodified lift (57 same-item), LOSO, decision including safety rules (heuristic row = tuned LOSO thresholds):

| arm | predicted merge | true merges | precision | recall |
|---|---|---|---|---|
| heur | 29 | 23 | 0.79 | 0.40 |
| hgb | 36 | 27 | 0.75 | 0.47 |
| lr | 45 | 32 | 0.71 | 0.56 |
| hgb_pseudo | 6 | 6 | 1.00 | 0.11 |

HGB permutation importance (average precision, real edges): chi2 0.060, share_small 0.021, dE 0.014, hstep 0.010, ang 0.009, contact_ratio 0.009.
Logistic-regression standardized coefficients (largest |coef|): chi2 -1.17, dE -0.94, ang -0.52, log_n_small -0.46, hstep -0.45, size_ratio +0.43.

**Second check: split_v3 val pseudo-votes (27 scans x 3 seeds, R1 corruption + 1-3 fragment_lift; seeds averaged per scan for the CI).** Heuristic thresholds {"t1": 0.45, "t2": 12, "t3": 60, "t4": 0.003}; thresholds and real-edge training exclude the real scans that are copies of val scans.

| arm | inst mIoU | PQ | exact count | d vs lift per scan (95% CI) | fragments made whole | correct items broken | items fixed | ids merged |
|---|---|---|---|---|---|---|---|---|
| lift | 0.499 | 0.268 | 0.04 | - | 0 / 84 | 0 / 94 | 0 | 0 |
| heuristic, thresholds tuned leave-one-scan-out | 0.454 | 0.358 | 0.20 | +0.051 [-0.013, +0.120] | 21 / 84 | 21 / 94 | 23 | 135 |
| heuristic tuned, no safety rules | 0.461 | 0.401 | 0.25 | +0.081 [+0.006, +0.155] | 27 / 84 | 22 / 94 | 30 | 160 |
| learned HGB (pseudo + other real scans, LOSO) | 0.558 | 0.450 | 0.17 | +0.101 [+0.048, +0.161] | 26 / 84 | 1 / 94 | 28 | 120 |
| learned HGB, pseudo edges only | 0.562 | 0.433 | 0.22 | +0.109 [+0.056, +0.168] | 27 / 84 | 2 / 94 | 26 | 112 |
| learned logistic regression (pseudo + real LOSO) | 0.506 | 0.419 | 0.20 | +0.087 [+0.021, +0.155] | 27 / 84 | 12 / 94 | 26 | 139 |
| oracle merge (GT-same edges; ceiling of merging only) | 0.590 | 0.524 | 0.44 | +0.168 [+0.103, +0.238] | 40 / 84 | 5 / 94 | 39 | 152 |

**Reading (6 lines).**
1. On real votes an explicit merge does not beat the lift beyond noise: tuned heuristic -0.006 [-0.084, +0.061], learned HGB +0.010 [-0.036, +0.052]; untuned thresholds are harmful (-0.100 [-0.203, +0.001]), so thresholds matter and are unstable across 11 scans.
2. Only 5-6 of 24 fragmented items become whole (oracle merging of whole lift ids reaches 9), so even a perfect merge lifts inst mIoU only from 0.632 to 0.713; exact count goes 0.00 -> 0.45 (heuristic) / 0.55 (oracle).
3. Cost: tuned heuristic breaks 4 of 35 correct items, HGB 3, LR 5; every method fixes only 2-3 items. Safety rules matter little at tuned thresholds (33 vs 30 merges) but the untuned heuristic breaks 11.
4. Learned vs heuristic on real edges: LOSO precision/recall HGB 0.75/0.47, heuristic 0.79/0.40, LR 0.71/0.56; no learned arm is better in instance mIoU (CIs overlap); the pseudo-only classifier merges almost nothing on real edges (recall 0.11), so the fragmentation augmentation does not transfer. Colour (chi2, dE) dominates, share_small, hstep and normal angle follow.
5. On pseudo val the learned classifier does beat the heuristic (+0.101 [+0.048, +0.161] vs +0.051 [-0.013, +0.120]) with 1 broken items vs 21, but plane / k-means fragments are easier than SAM 3's.
6. Verdict: the CPU merge is an interpretable baseline that roughly equals the lift on real votes (no significant gain; 11 scans / 52 items); a learned refiner must beat about 0 +-0.06 per scan with < 4 broken items to be worth it. Limits: tiny real set, grid-selection noise, same-name flag unused, no per-item CI.
