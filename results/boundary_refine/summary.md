# 40-scan validation (hard_v12 + hard_v13, real SAM 3 votes, co-vote lift), 2026-10-16

**Decision: boundary CRF stays default OFF (`run_meal --boundary-refine` opt-in).** Rule (default-on only if pooled paired per-scan CI excludes 0 AND no batch has a negative point estimate AND broken < 0.2 % of points): fixed 11-scan params pooled d scan-mIoU +0.001 [-0.002, +0.005] (CI includes 0), v13 -0.002 (negative), broken 0.168 % (passes). Fails 2 of 3 conditions. The 11-scan gain (+0.011 [+0.004, +0.018]) does not replicate: v12 (20 scans, 11 of them the tuning scans plus 9 new) +0.004 [-0.002, +0.011], v13 -0.002 [-0.005, +0.002]. Leave-one-batch-out retune is no better (pooled -0.002 [-0.005, +0.002]; tuning on v13 and testing on v12 picks band 3 mm w=1.0 sc=5 sn=30 sz=6, the reverse picks band 6 w=1.4 sc=7 sn=7.5 sz=3, so the optimum is not stable). PQ rises 0.487 -> 0.510 (fixed) but boundary IoU is flat (0.332 -> 0.334) and mIoU is flat: the CRF mostly cleans small id pieces.

Data: the co-vote-linked lift was regenerated for all 40 scans (the live seg_points `lift_label` is the older repair-linked lift, pooled 0.601 in results/lift_sweep/scans40): `python -m foodscan.boundary_refine replay40` (segment_scan from the raw SAM 3 maps + labels.lift_for_meal, cos+qual; writes results/lift_replay40 and soft votes `cache40/`), pooled inst mIoU 0.631 on 273 items (= the cos+qual row of lift_sweep). Grid = the 483 unique configs of grid_v12 + grid_v12b (`grid40 --tag r40`, 22 CPU procs, ~45 min), `report40 --tag r40` -> `tables_r40.md`, `report_r40.json`. Selection objective = pooled item IoU on the training batch. Deltas are paired per-scan bootstrap (2000 resamples); "d items" is per GT item.

| arm | set | n scans | lift inst mIoU | CRF inst mIoU | d scan-mIoU (95% CI) | d items (95% CI) | lift PQ -> CRF PQ | boundary IoU lift -> CRF | scans +/-  | corrected / errors | broken (% pts) | broken pts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed 11-scan params | v12 | 20 | 0.657 | 0.662 | +0.004 [-0.002, +0.011] | +0.005 [-0.000, +0.010] | 0.524 -> 0.560 | 0.337 -> 0.344 | 12/8 | 0.012 | 0.093 | 920 |
| fixed 11-scan params | v13 | 20 | 0.600 | 0.600 | -0.002 [-0.005, +0.002] | -0.000 [-0.004, +0.003] | 0.447 -> 0.458 | 0.328 -> 0.322 | 10/10 | 0.023 | 0.267 | 1994 |
| fixed 11-scan params | pooled | 40 | 0.631 | 0.633 | +0.001 [-0.002, +0.005] | +0.002 [-0.001, +0.005] | 0.487 -> 0.510 | 0.332 -> 0.334 | 22/18 | 0.017 | 0.168 | 2914 |
| leave-one-batch-out retune | v12 | 20 | 0.657 | 0.656 | +0.000 [-0.005, +0.006] | -0.001 [-0.012, +0.010] | 0.524 -> 0.555 | 0.337 -> 0.333 | 9/11 | 0.013 | 0.097 | 955 |
| leave-one-batch-out retune | v13 | 20 | 0.600 | 0.599 | -0.003 [-0.008, +0.001] | -0.001 [-0.006, +0.002] | 0.447 -> 0.456 | 0.328 -> 0.317 | 10/10 | 0.023 | 0.274 | 2051 |
| leave-one-batch-out retune | pooled | 40 | 0.631 | 0.630 | -0.002 [-0.005, +0.002] | -0.001 [-0.007, +0.005] | 0.487 -> 0.507 | 0.332 -> 0.326 | 19/21 | 0.018 | 0.174 | 3006 |
| all-40 oracle-tuned (optimistic) | v12 | 20 | 0.657 | 0.663 | +0.004 [-0.001, +0.010] | +0.006 [+0.001, +0.012] | 0.524 -> 0.546 | 0.337 -> 0.341 | 12/8 | 0.012 | 0.078 | 764 |
| all-40 oracle-tuned (optimistic) | v13 | 20 | 0.600 | 0.601 | +0.000 [-0.002, +0.003] | +0.001 [-0.003, +0.003] | 0.447 -> 0.459 | 0.328 -> 0.321 | 12/8 | 0.023 | 0.232 | 1734 |
| all-40 oracle-tuned (optimistic) | pooled | 40 | 0.631 | 0.634 | +0.002 [-0.001, +0.005] | +0.003 [+0.000, +0.007] | 0.487 -> 0.505 | 0.332 -> 0.332 | 24/16 | 0.017 | 0.144 | 2498 |

LOBO picks: train on other batch -> test v12: band 3 w=1.0 sc=5.0 sn=30.0 sz=6.0; train on other batch -> test v13: band 6 w=1.4 sc=7.0 sn=7.5 sz=3.0; all-40 best: band 3 w=1.0 sc=7.0 sn=10.0 sz=4.5

Corrected = lift-wrong points made right / all lift-wrong points; broken = lift-right points made wrong, as % of evaluated points. Per-scan table in `tables_r40.md`.

---

# Point-level boundary refinement of the lifted labels (CPU, alpha-expansion CRF)

Code `pipeline/foodscan/boundary_refine.py` (tests `pipeline/tests/test_boundary_refine.py`), flag `run_meal --boundary-refine` (default off). Method: kNN graph (k=12, 8 mm cap) on the 2 mm voxel scene; unary = -log of normalised co-vote shares over the ids found within 12 mm (floor 0.02, one merged non-food class for 0/252/253/254); Potts weight `w*exp(-dLab^2/2sc^2)*exp(-ang^2/2sn^2)*exp(-dz^2/2sz^2)`; PyMaxflow alpha-expansion (3 sweeps), only points within `band` of a food-label boundary may change. Same scene/metrics as `eval/lift_sweep.evaluate` (reference 0.741 reproduced). Grid: band {3,6,12} x w x s_c x s_n x s_z, 486 configs (grid_v12, grid_v12b); selection = pooled item IoU; LOO = leave-one-scan-out via `lift_sweep.loso_pick`. Chosen: **band 6 mm, w=1.0, s_c=5 (Lab), s_n=10 deg, s_z=3 mm**; 3 of 4 params are interior optima, s_n=10 was near the grid edge but 5/7.5 gave no gain (grid_v12b). Runtime 1.2 s per scan on one CPU core (0.2 s graph + ~1 s solve, ~50k voxels); full-cloud entry `refine_cloud` ~5 s for 117k points.

Data note: the live `data/synth/hard_v12_sam3_v2_image` seg_points were regenerated by another job mid-study (different frame); everything here uses a frozen copy of the pre-change files in `snap/` (from /tmp/k12bak) and the vote cache in `cache/` built before that, so numbers match results/lift_replay and lift_errors.

## 11 hard_v12 scans, co-vote lift (all-scan-tuned is optimistic; LOO is the honest row)

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

## Second check: split_v3 val scans with pseudo votes (params fixed from the hard_v12 tuning, no selection here)

No soft votes exist for these, so the unary is built from `lift_label` + `lift_share` (own label gets the share, the rest is spread over labels within 12 mm). Most of these lifts are near-GT synthetic pseudo labels (23 scans incl. 7 hard_v12 ones; 13 of them have lift mIoU >= 0.95), so this is a weak test. The `w=0.0001` row is a control (unary only, no smoothing).

**all: 23 scans, 64 items**

| setting | inst mIoU | d inst (items, 95% CI) | PQ | boundary IoU | corrected / lift errors | broken % pts | s / scan |
|---|---|---|---|---|---|---|---|
| lift (pseudo votes) | 0.766 | - | 0.680 | 0.346 | - | - | - |
| band 3.0 w=1.0 sc=5.0 sn=10.0 sz=3.0 | 0.775 | +0.009 [+0.003, +0.017] | 0.711 | 0.345 | 0.012 | 0.321 | 0.5 |
| band 6.0 w=1.0 sc=5.0 sn=10.0 sz=3.0 | 0.773 | +0.007 [-0.003, +0.019] | 0.704 | 0.347 | 0.014 | 0.546 | 0.4 |
| band 12.0 w=1.0 sc=5.0 sn=10.0 sz=3.0 | 0.772 | +0.006 [-0.007, +0.018] | 0.705 | 0.352 | 0.013 | 0.760 | 0.5 |
| band None w=1.0 sc=5.0 sn=10.0 sz=3.0 | 0.772 | +0.006 [-0.007, +0.018] | 0.705 | 0.352 | 0.013 | 0.760 | 0.7 |
| band 6.0 w=0.0001 sc=5.0 sn=10.0 sz=3.0 | 0.766 | +0.000 [-0.002, +0.003] | 0.666 | 0.345 | 0.002 | 0.071 | 0.4 |

**lift mIoU < 0.95: 10 scans, 50 items**

| setting | inst mIoU | d inst (items, 95% CI) | PQ | boundary IoU | corrected / lift errors | broken % pts | s / scan |
|---|---|---|---|---|---|---|---|
| lift (pseudo votes) | 0.702 | - | 0.603 | 0.353 | - | - | - |
| band 3.0 w=1.0 sc=5.0 sn=10.0 sz=3.0 | 0.714 | +0.012 [+0.004, +0.021] | 0.641 | 0.352 | 0.013 | 0.366 | 1.1 |
| band 6.0 w=1.0 sc=5.0 sn=10.0 sz=3.0 | 0.711 | +0.008 [-0.005, +0.022] | 0.631 | 0.354 | 0.014 | 0.617 | 0.9 |
| band 12.0 w=1.0 sc=5.0 sn=10.0 sz=3.0 | 0.709 | +0.006 [-0.011, +0.022] | 0.631 | 0.354 | 0.014 | 0.864 | 1.0 |
| band None w=1.0 sc=5.0 sn=10.0 sz=3.0 | 0.709 | +0.006 [-0.011, +0.022] | 0.631 | 0.354 | 0.014 | 0.864 | 1.6 |
| band 6.0 w=0.0001 sc=5.0 sn=10.0 sz=3.0 | 0.702 | -0.000 [-0.004, +0.003] | 0.584 | 0.352 | 0.001 | 0.082 | 0.7 |

## Reading

1. A classic colour/normal/height-aware Potts CRF realises only a small part of the oracle: +0.013 mIoU (0.741 -> 0.754 all-scan-tuned, +0.009 LOO) against an oracle bound of +0.069 (3 mm) / +0.123 (6 mm) with our lift-boundary band definition (+0.098 / +0.163 with the GT-boundary band of lift_errors): 10-19 % of the 3/6 mm bound, 7-13 % under LOO.
2. The paired CI excludes zero on the hard_v12 scans (items +0.005..+0.022; LOO +0.002..+0.017) but the effect is small and most of it is on s000002/7/13/14/16/17 (+0.010 to +0.030); s000009/11 and s000000/1 are unchanged, s000012 loses 0.006.
3. It is a fragment cleaner more than a boundary snapper: only 865 of 541k voxels (0.16 %) change, 1.9 % of lift-error points are corrected, and the gain comes with fewer ids (65 -> 61), fp 25 -> 19, fn 12 -> 10, PQ +0.049 (0.593 -> 0.642); boundary IoU only 0.324 -> 0.335.
4. Why the bound is not reached: the 3 mm error voxels are not noisy edges but systematic offsets (SAM mask edges bleed across the true edge, consistently across views), so the votes and the smoothness term agree on the wrong side; colour edges at food/food contacts are weak (similar foods, shadows), and the unary-only control shows no gain by itself (-0.003).
5. What it breaks: 0.05 % of points (hard_v12) / 0.3-0.8 % (val set, pseudo votes with near-perfect labels) go right -> wrong; on the val set the gain shrinks to +0.006..+0.009 and its CI includes 0 for band >= 6 mm (band 3 mm: +0.009 [+0.003, +0.017]); band 'none' is no better than 6 mm and 2x slower.
6. Parameters are stable (LOO picks the all-scan setting in 7/11 folds, the rest are neighbours with w=1.4 or s_n=7.5; only s000014 flips to 3 mm and loses), larger bands add nothing, so the usable lever is only the first few mm.
7. It is cheap (1.2 s per scan, no GPU) and never worse than -0.006 per scan on hard_v12, but at +0.01 it is not the "largest remaining lever" the oracle suggested.
8. Recommendation: keep `--boundary-refine` off by default for now (gain 0.009 LOO is within what other pending changes will change; val CI includes 0); enable it as an optional cleanup, and a learned boundary model (e.g. the PTv3 refiner trained on boundary targets) or a 2D mask-edge correction is needed for the remaining bound.
