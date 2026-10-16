# Nutrition5k dish mass and calorie regression: experiment log

Date: 2026-10-13

Scope. This log records the fine-tuning study that predicts five quantities per dish from a
single overhead RGB-D frame of a dining-hall tray: total food mass (g), energy (kcal), fat,
carbohydrate and protein. The data are the public Nutrition5k overhead captures
(`depth_train_ids` for training, `depth_test_ids` for testing), and every number below is
measured on the same fixed 507-dish test set. All training and evaluation runs on one local
GPU with a total capacity of about 5.6 GB (RTX 2060) that is shared with a Blender job
(~0.8 GB), so peak memory had to stay under roughly 4 GB; observed peaks for the runs below
are 1.4-3.7 GB. Every figure in this document is read from a run's `summary.json`, from the
`TEST` line of its `.log`, or recomputed from its `test_predictions.csv`.

## Protocol

- **Data.** `data/datasets/Nutrition5k` (public GCS bucket, CC BY 4.0), overhead `rgb.png` +
  `depth_raw.png`. Three dishes in the split lists have no depth in the bucket and are dropped.
- **Cache.** All fine-tuning runs read a pre-built 448x336 tensor cache under
  `data/cache/n5k_448*` (`eval/n5k_cache.py`): `n5k_448` (the original, mask + container
  thickness from Grounding DINO + SAM 2.1 masks), `n5k_640` (640 px variant), and
  `n5k_448_sam3b` (rebuilt from SAM 3 masks with generic "food" / "plate" / "bowl" prompts).
- **Channels.** `rgb` = image only; `rgbd` = image + height above the counter; `rgbd_thick` =
  image + food height above its own container (bowl surface / plate plane) + food mask.
  `rgbd_thick` is the default and the configuration of every run from round 1 onward.
- **Head and metric.** One 5-way head over the five targets, `TARGETS = (mass, kcal, fat,
  carb, protein)`, so the metric is per target and there is no separate calories head. For
  each target, `eval.n5k_train.score(ids, pred)` returns (MAE in grams, 100 * MAE / mean of
  the true values), i.e. an unweighted MAPE in percent. Headline numbers are test MAE (g) and
  MAPE (%) for **mass** and **kcal**.
- **Test set.** The official `depth_test_ids`, 507 dishes, never used for any selection.
- **Val protocol.** Unless `--full` is given, a fixed 10 % holdout of `depth_train_ids`
  (`--val-frac 0.1 --split-seed 0`, 275 dishes; 2480 for training) is used to pick the kept
  epoch, and the model is then scored once on test. `val score` below means the mean of the
  val mass % and the val kcal % at the kept epoch (`best_val` in `summary.json`).
- **Full protocol.** `--full` trains on all 2755 `depth_train_ids` dishes with no val split;
  the last epoch is kept and scored once on test. These runs write no `val_predictions.csv`,
  so they can only be scored on test.
- **Command shape.** A job is one line in `results/n5k/train/queue.txt`, either
  `<name> --flags...` (expanded to `python -m eval.n5k_train ../data/datasets/Nutrition5k
  <flags> --out ../results/n5k/train/<name>`, run from `pipeline/`) or `<name> <shell
  command>` (run from `pipeline/`). `scripts/n5k_queue.sh` runs one line at a time and
  appends finished names to `queue.txt.done`.
- **Hardware.** Single shared ~5.6 GB GPU. `--bs 8 --accum 2` (effective batch 16) plus
  gradient checkpointing for the larger backbones, `--bs 16` for ConvNeXt-T.

## Results

Test set, 507 dishes. `log` / `raw` / `mix` loss = L1 on log1p of grams, on grams divided by
the train mean, and a mixture of the two, respectively. Bold marks the best value in the mass
and kcal columns among the finished runs; after round 11 the best single *number* is still
`b3_log_full`'s 18.29 g, but its reseed (`b3_log_full_s1` 19.09 g) shows that draw does not
reproduce, so the best reproducible single model is the `raw` ConvNeXt-B full-data recipe at
18.61-18.91 g (round 11; see caveats).

| run | round | mode | cache | epochs | arch | loss | test mass g | mass % | test kcal g | kcal % | outcome / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rgb | pre | val | n5k_448 | 40 | ConvNeXt-T | log | 24.14 | 12.16 | 39.05 | 15.28 | baseline, image only |
| rgbd | pre | val | n5k_448 | 40 | ConvNeXt-T | log | 22.44 | 11.30 | 38.63 | 15.12 | + counter-relative height |
| rgbd_thick | pre | val | n5k_448 | 40 | ConvNeXt-T | log | 21.31 | 10.74 | 38.83 | 15.20 | + food height + food mask; reference config |
| t640 | 1 | val | n5k_640 | 40 | ConvNeXt-T | log | 21.09 | 10.62 | 40.19 | 15.73 | 640 px; mass -0.22 g, kcal +1.36 g worse |
| s448 | 1 | val | n5k_448 | 40 | ConvNeXt-S | log | 19.93 | 10.04 | 37.13 | 14.53 | bigger backbone: best mass so far |
| t448_raw | 2 | val | n5k_448 | 40 | ConvNeXt-T | raw | 21.67 | 10.92 | 39.09 | 15.30 | metric-scale loss: no gain at T |
| t448_mix | 2 | val | n5k_448 | 40 | ConvNeXt-T | mix | 21.52 | 10.84 | 38.31 | 15.00 | mixed loss: no clear gain |
| s448_raw | 3 | val | n5k_448 | 40 | ConvNeXt-S | raw | 20.62 | 10.39 | 38.21 | 14.96 | raw at S: worse on mass than log at S |
| b448_raw | 3 | val | n5k_448 | 40 | ConvNeXt-B | raw | 20.17 | 10.16 | 36.61 | 14.33 | best kcal of the val-split runs; best val score of any run (9.50) |
| vb448_raw | 3 | val | n5k_448 | 40, stopped at 15 | DINOv2 ViT-B | raw | - | - | - | - | **failed**: killed at epoch 15/40, val mass 16-18 %; no test output |
| t448_e60 | extra | val | n5k_448 | 60, EMA 0.999 | ConvNeXt-T | log | 21.31 | 10.74 | 39.45 | 15.44 | longer schedule + EMA: no change on mass |
| s448_sam3b | 4 | val | n5k_448_sam3b | 40 | ConvNeXt-S | log | 19.72 | 9.94 | 36.97 | 14.47 | SAM 3 mask cache: new best mass |
| s448_sam3 | 4 | val | n5k_448_sam3 | 40, killed at 34 | ConvNeXt-S | log | - | - | - | - | **killed** at epoch 34/40, superseded by the rebuilt cache |
| s448_full | 5 | full | n5k_448 | 0 | ConvNeXt-S | log | - | - | - | - | **stopped by design before training** (val preferred the SAM 3 cache) |
| s3_full | 5 | full | n5k_448_sam3b | 40 | ConvNeXt-S | log | 18.98 | 9.56 | 35.21 | 13.78 | all 2755 train dishes, last epoch |
| s3_raw_full | 5 | full | n5k_448_sam3b | 40 | ConvNeXt-S | raw | 19.70 | 9.92 | 36.31 | 14.21 | full data + raw at S |
| b3_raw_full | 5 | full | n5k_448_sam3b | 40 | ConvNeXt-B | raw | 18.75 | 9.45 | 34.76 | 13.61 | the reference single model; seed 2 of the same recipe draws 0.14 g lower (round 8) |
| l448_raw | 6 | val | n5k_448_sam3b | 40, 0 done | ConvNeXt-L | raw | - | - | - | - | **failed**: CUDA OOM right after epoch 0, nothing produced |
| b3_raw_full_s1 | 6 | full | n5k_448_sam3b | 40 | ConvNeXt-B | raw | 18.88 | 9.51 | 35.17 | 13.77 | seed 1: +0.13 g mass, +0.41 g kcal vs seed 0 |
| s3_full_s1 | 6 | full | n5k_448_sam3b | 40 | ConvNeXt-S | log | 19.07 | 9.60 | 35.74 | 13.99 | seed 1: +0.09 g mass, +0.53 g kcal vs seed 0 |
| b3_raw_sam3b | 8 | val | n5k_448_sam3b | 40 | ConvNeXt-B | raw | 19.95 | 10.05 | 36.02 | 14.10 | val-split twin of `b3_raw_full`: full data is worth -1.20 g mass / -1.25 g kcal |
| b3_raw_full_e60 | 8 | full | n5k_448_sam3b | 60 | ConvNeXt-B | raw | 18.77 | 9.46 | 35.17 | 13.77 | +0.02 g mass, +0.41 g kcal vs 40 epochs: the longer schedule does not help |
| b3_raw_full_s2 | 8 | full | n5k_448_sam3b | 40 | ConvNeXt-B | raw | 18.61 | 9.38 | 34.30 | 13.43 | seed 2: -0.14 g mass, -0.46 g kcal vs seed 0, inside the 0.27 g three-seed spread |
| l448_raw_bs2 | 8 | val | n5k_448_sam3b | 40, 0 done | ConvNeXt-L | raw | - | - | - | - | **failed**: CUDA OOM in the first val forward at `--bs 2 --accum 8`; nothing produced |
| b3_raw_full_s3 | 9 | full | n5k_448_sam3b | 40 | ConvNeXt-B | raw | 18.91 | 9.52 | 35.76 | 14.00 | seed 3: +0.15 g mass, +1.00 g kcal vs seed 0 |
| s3_full_s2 | 9 | full | n5k_448_sam3b | 40 | ConvNeXt-S | log | 19.20 | 9.67 | 35.39 | 13.85 | seed 2: +0.22 g mass, +0.18 g kcal vs seed 0 |
| b3_raw_full_dp2 | 9 | full | n5k_448_sam3b | 40 | ConvNeXt-B | raw | 19.07 | 9.61 | 35.69 | 13.97 | `--drop-path 0.2`: +0.32 g mass, +0.93 g kcal vs its 0.1 twin |
| b640_raw_full | 10 | full | n5k_640 | 40 | ConvNeXt-B | raw | 19.85 | 10.00 | 34.98 | 13.69 | 640 px at the winning recipe: +1.10 g mass vs its 448 twin `b3_raw_full`; resolution closed at B |
| b3_log_full | 10 | full | n5k_448_sam3b | 40 | ConvNeXt-B | log | **18.29** | **9.21** | **33.57** | **13.14** | `log` at ConvNeXt-B: -0.46 g mass and -1.19 g kcal vs its `raw` twin; the new best single model |
| cv2b_raw_full | 10 | full | n5k_448_sam3b | 40 | ConvNeXtV2-B | raw | 18.44 | 9.29 | 34.26 | 13.41 | ConvNeXtV2-B; its exact values were regenerated from the surviving `best.pt` and confirmed by reproducing the seven-member ensemble (see caveats) |
| l448_full | 11 | val | n5k_448_sam3b | 40 | ConvNeXt-L | raw | 20.21 | 10.18 | 34.64 | 13.56 | ConvNeXt-L finally trains with `--eval-bs 4`: +0.26 g of mass (a draw, inside the 0.29 g seed span) and -1.38 g of kcal against its same-cache ConvNeXt-B val-split twin `b3_raw_sam3b` |
| b3_log_full_s1 | 11 | full | n5k_448_sam3b | 40 | ConvNeXt-B | log | 19.09 | 9.61 | 33.75 | 13.21 | seed 1 of `b3_log_full`: +0.79 g of mass, so the 18.29 g draw does not replicate; its -1.42 g of kcal against its own `raw` twin does |

The full 5-way head of the reference single model, `b3_raw_full`: fat 2.76 g / 21.51 %, carb 3.98 g /
20.22 %, protein 3.64 g / 20.85 %. The same model's per-target numbers in the table above are
the mass and kcal columns. The two round-11 runs' other heads: `l448_full` fat 2.83 / 22.02 %,
carb 4.16 / 21.15 %, protein 3.57 / 20.45 %; `b3_log_full_s1` fat 2.85 / 22.22 %, carb 3.95 /
20.05 %, protein 3.75 / 21.47 %.

### Ensembles

`eval.n5k_ensemble.py` combines runs by a geometric mean of the predictions,
`expm1(mean(log1p(pred)))`, per target, and scores the result on test. Ensembles are not
training runs, so they have no `summary.json`; the numbers below are recomputed from the
`test_predictions.csv` of the member runs with exactly that rule, except the three round-10 rows,
which write no predictions and exist only as 1-dp `TEST` lines in their `.log` files; the first
three rows agree with the ensemble rows in `results/n5k/README.md` to rounding (the README
predates round 7).

| ensemble | test mass g | mass % | test kcal g | kcal % |
|---|---|---|---|---|
| s448 + s448_raw + b448_raw (val-split models, old cache) | 19.25 | 9.70 | 36.10 | 14.13 |
| rgbd_thick + s448 + b448_raw (val-split models, old cache) | 19.58 | 9.86 | 35.72 | 13.98 |
| s3_full + s3_raw_full + b3_raw_full (full-data models, SAM 3 cache, round 5) | 18.45 | 9.30 | 34.38 | 13.46 |
| b3_raw_full + s3_full (`ens_b3_s3`) | 18.33 | 9.23 | 33.91 | 13.28 |
| b3_raw_full + s3_raw_full (`ens_b3_s3raw`) | 18.66 | 9.40 | 34.89 | 13.66 |
| b3_raw_full + b3_raw_full_s1 (`ens_b3_seeds`) | 18.45 | 9.29 | 34.69 | 13.58 |
| s3_full + s3_full_s1 (`ens_s3_seeds`) | 18.71 | 9.43 | 35.00 | 13.70 |
| b3_raw_full + b3_raw_full_s1 + s3_full + s3_full_s1 (`ens_all4`) | 18.19 | 9.16 | 34.16 | 13.37 |
| b3_raw_full + s3_full + s448_sam3b (`ens_b3_s3_s448`) | 18.39 | 9.26 | 34.38 | 13.46 |
| b3_raw_full + b3_raw_full_s1 + b3_raw_full_s2 (the three `b3` seeds, round 8) | 18.26 | 9.20 | 34.32 | 13.43 |
| b3_raw_full + b3_raw_full_e60 (winner + 60-epoch twin, round 8) | 18.56 | 9.35 | 34.78 | 13.62 |
| b3_raw_full + b3_raw_full_s1 + b3_raw_full_s2 + s3_full (four full-data models, round 8) | 18.14 | 9.14 | 33.81 | 13.24 |
| b3_raw_full + b3_raw_full_s1 + b3_raw_full_s2 + s3_full + s3_full_s1 (five full-data models, round 8) | 18.08 | 9.11 | 33.92 | 13.28 |
| the same five + b3_raw_full_e60 (six full-data models, round 8) | 18.08 | 9.11 | 33.95 | 13.29 |
| b3_raw_full + b3_raw_full_s1 + b3_raw_full_s2 + b3_raw_full_s3 + s3_full + s3_full_s1 (`ens_all6`, round 9) | 18.07 | 9.10 | 34.10 | 13.35 |
| the same six + s3_full_s2 (`ens_all7`, round 9) | 18.03 | 9.08 | 34.00 | 13.31 |
| b3_raw_full + b3_raw_full_s1 + b3_raw_full_s2 + s3_full + s3_full_s1 + b3_raw_full_dp2 (`ens_all5_dp2`, round 9) | 18.10 | 9.12 | 34.08 | 13.34 |
| the five full-data models + `b3_log_full_s1` (the `log` member at seed 1 instead of seed 0, round 11) | 18.08 | 9.11 | 33.64 | 13.17 |
| the five full-data models + both `log` seeds (`b3_log_full` and `b3_log_full_s1`, round 11) | 17.99 | 9.06 | 33.45 | 13.09 |

The last six rows before round 8 are round 7, the five after them round 8, the three after them
round 9 and the last two round 11, all test-only; round 10's three means are not rows here
because they exist only to one decimal (their `.log` `TEST` lines) and are quoted in the prose
below instead. Best ensemble on mass is still round 10's seven-member `ens_all7_log_cv2`, 17.8 g /
9.0 % and 33.3 kcal / 13.1 %, 0.28 g of mass and 0.58 g of kcal below the five-member
all-full-data mean (18.08 g / 9.11 % and 33.92 / 13.28 %); the mass gain is at the noise floor,
the kcal gain is outside it. The round-10 ensemble numbers are read at 1 dp. Against the
five-member mean the other two round-10 means are draws on mass - `ens_all6_640` 18.1 g (+0.02 g)
and `ens_all6_log` 18.0 g (-0.08 g) - so neither the 640 member nor the `log` member adds anything
on its own at six members. On kcal the seven-member mean is also below round 8's best, the
four-member mean without `s3_full_s1` (33.81 / 13.24 %).

Round 11 re-scored the seven-member mean from the recovered `cv2b_raw_full` predictions and
reproduced the round-10 line exactly: 17.7993 g / 8.9665 % and 33.3410 / 13.0510 %, i.e. 17.8 g /
9.0 % and 33.3 / 13.1 %, the numbers `ens_all7_log_cv2` printed. The check that the recovered
member is in the mean rather than silently dropped is the same seven members minus `cv2b_raw_full`
- 17.9741 g / 9.0545 % and 33.6106 / 13.1565 %, i.e. 18.0 g / 9.1 % and 33.6 / 13.2 % at the
precision the round-10 numbers are quoted at. Round 11
also re-reads the `log` member on a second seed: the same five models plus `b3_log_full_s1` in
place of `b3_log_full` give 18.0811 g / 9.1084 % and 33.6352 / 13.1662 % (a draw on mass against
the five-member 18.0824 g, -0.28 g of kcal), and both log seeds together give 17.9940 / 9.0646 and
33.4485 / 13.0931, so the log member's mass contribution is a seed-0 artifact and only its kcal
contribution repeats. Rounds 7 to 11 below have the reading of the nineteen rows.

Reference point: the published Nutrition5k-style volume-scalar baseline that this work is
measured against is 29.4 g / 13.7 % mass and 41.3 kcal / 16.5 % kcal on this split
(`docs/model_survey.md`, `results/n5k/README.md`; the README notes that the papers should be
re-checked before quoting). Both are beaten by the models above.

## Why each round

### Pre-queue baseline (2026-10-11)

Hypothesis: an overhead RGB-D frame plus a container-aware height channel should be enough to
regress dish mass and energy, and a plain single-frame CNN is the baseline the multi-view
scan has to beat.

Result: `rgb` 24.14 g / 12.16 % mass and 39.05 kcal / 15.28 %; adding depth (`rgbd`) gives
22.44 g / 11.30 % and 38.63 / 15.12; adding food height above the container and the food mask
(`rgbd_thick`) gives 21.31 g / 10.74 % and 38.83 / 15.20. All 40 epochs, ConvNeXt-T, log loss,
val split.

Conclusion: depth is worth about 1.7 g and the container-aware thickness channel another
0.7 g; the depth channel barely moves kcal (39.05 -> 38.83 g) while it clearly moves mass.
`rgbd_thick` becomes the input configuration for everything that follows.

### Round 1: resolution and backbone size

Hypothesis: at 448 px the model may be resolution-limited on small ingredients, and a bigger
backbone should fit in the memory budget at `--bs 8 --accum 2` with gradient checkpointing.

Result: `t640` (640 px, same ConvNeXt-T) 21.09 g / 10.62 % and 40.19 kcal / 15.73 %; `s448`
(ConvNeXt-S at 448) 19.93 g / 10.04 % and 37.13 / 14.53 %.

Conclusion: resolution is a wash. Doubling the pixel count buys 0.22 g of mass (inside the
run-to-run noise) and loses 1.36 g of kcal, and costs 2.5x the epoch time (75 s vs 30 s).
Backbone size pays: ConvNeXt-T to ConvNeXt-S at the same input and resolution (`rgbd_thick` -> `s448`) is 1.38 g of mass and 1.71 g of kcal for 2x the
epoch time. Resolution is dropped; the backbone lever is kept.

### Round 2: loss on the metric's scale

Hypothesis: the log1p loss under-predicts the heaviest dishes by roughly 20 g, because it
optimises relative error; an L1 on grams (`raw`) should fix the tail and lower MAE.

Result: at ConvNeXt-T, `t448_raw` 21.67 g / 10.92 % and 39.09 / 15.30 %; `t448_mix` 21.52 g /
10.84 % and 38.31 / 15.00 %. Both are compared with the same backbone and cache under the log
loss: `rgbd_thick` 21.31 g / 10.74 % and 38.83 / 15.20 %.

Conclusion: honest negative. Neither metric-scale loss beats the log loss at this backbone
size: `raw` is 0.36 g worse on mass and 0.26 g worse on kcal, and `mix` is 0.21 g worse on
mass (0.52 g better on kcal). `mix` is therefore no better than `raw`, and both are within
noise of the baseline they were meant to improve. The metric-scale loss is kept for the
larger models (it is what the final full-data runs use) but it is not itself a lever.

### Round 3: bigger backbones with the metric-scale loss

Hypothesis: if the loss is right and the resolution is enough, capacity is what remains, so
run ConvNeXt-S and ConvNeXt-B with `raw` and try a ViT-B (DINOv2) as a second family.

Result: `s448_raw` 20.62 g / 10.39 % and 38.21 / 14.96 %; `b448_raw` 20.17 g / 10.16 % and
36.61 kcal / 14.33 %, which is the best val score of any run in the study (9.50, i.e. val mass
7.35 % and val kcal 11.65 % at the kept epoch 37). `vb448_raw` (ViT-B, lr 5e-5) was killed at
epoch 15/40 with val mass still at 16-18 %, against roughly 11 % for the ConvNeXt runs at the
same point.

Conclusion: the two levers are partly confounded. Against the round-1 ConvNeXt-S with log loss
(19.93 g), `raw` at the same backbone is 0.69 g *worse* on mass, and ConvNeXt-B with `raw`
(20.17 g) is still 0.24 g worse on mass than the log-loss S model, while being 0.52 g better
on kcal. So the backbone step S to B buys 0.45 g of mass and 1.61 g of kcal when the loss is
held at `raw`, and B is the best val model by a wide margin (9.50 vs 11.21 for S), but on test
mass the small model with the log loss still wins. The DINOv2 ViT-B family is dropped: it
needs a much lower learning rate and does not converge in the same budget.

### Extra: 60 epochs and EMA

Hypothesis: the training loss is still falling at epoch 40, so more epochs and an EMA of the
weights should help.

Result: `t448_e60` (60 epochs, EMA 0.999) 21.31 g / 10.74 % and 39.45 kcal / 15.44 %.

Conclusion: a wash, and slightly negative on kcal. Mass is identical to the 40-epoch
`rgbd_thick` (21.31 g) and kcal is 0.62 g worse. Longer schedules and EMA are dropped; the
later rounds spend the extra time on the backbone and on data instead.

### Round 4: SAM 3 masks

Hypothesis: the original cache's food masks come from Grounding DINO + SAM 2.1, leave 3.6 % of
dishes with an empty mask, and sometimes include the plate rim in a salad mask; SAM 3 with
generic prompts should clean this up and therefore improve the thickness channel.

Result: the SAM 3 masks reduce empty masks from 3.6 % to 0.2 % and agree with the old masks at
a median IoU of 0.97; the rebuilt cache `n5k_448_sam3b` gives `s448_sam3b` 19.72 g / 9.94 %
and 36.97 kcal / 14.47 %, against 19.93 g / 10.04 % and 37.13 / 14.53 % for the same backbone
and loss on the old cache, and improves the val score from 11.21 to 11.02. The first attempt,
`s448_sam3` on the intermediate `n5k_448_sam3` cache, was killed at epoch 34/40 and superseded
by the rebuilt cache.

Conclusion: a real but small gain, 0.21 g of mass and 0.16 g of kcal on test, and it is the
mask quality, not the segmentation model per se, that matters. This cache is used for every
later run.

### Round 5: full-data finals

Hypothesis: with the input channels, the backbone and the mask cache fixed, the last
untouched lever is data; giving up the val split for 275 extra training dishes should help,
and the epoch can be fixed in advance from the val runs.

Result: `s3_full` (ConvNeXt-S, log) 18.98 g / 9.56 % and 35.21 kcal / 13.78 %; `s3_raw_full`
(ConvNeXt-S, raw) 19.70 g / 9.92 % and 36.31 / 14.21 %; `b3_raw_full` (ConvNeXt-B, raw)
18.75 g / 9.45 % and 34.76 kcal / 13.61 %, the best single model so far. `s448_full` was
stopped before training started, because val preferred the SAM 3 cache and the whole point of
this round was to use it.

Conclusion: the largest single drop in the study. Going from the val-split ConvNeXt-S on this
cache (19.72 g / 36.97 kcal) to the same model on all 2755 dishes (18.98 g / 35.21 kcal) is
0.74 g of mass and 1.76 g of kcal; on the ConvNeXt-B + raw line the full-data run is 1.42 g
and 1.85 g better than its val-split counterpart (20.17 / 36.61; that counterpart is on the
old cache, and round 8 splits the 1.42 / 1.85 into 1.20 / 1.25 of data and 0.22 / 0.59 of
cache). Within the full-data models the log loss wins on mass at ConvNeXt-S (18.98 vs 19.70)
while `raw` is kept for the ConvNeXt-B model.

### Round 6: seeds and ConvNeXt-L

Hypothesis: the remaining gap is capacity (ConvNeXt-L) and noise (a second seed of the
winner), and one more seed of the ConvNeXt-S final is cheap.

Result: `l448_raw` (ConvNeXt-L, `--bs 4 --accum 4`) crashed with `torch.OutOfMemoryError` in
the first forward after epoch 0, with 4.72 GiB of the 5.60 GiB card already in use, and
produced no checkpoint and no metrics. `b3_raw_full_s1` (seed 1 of the winner) gives 18.88 g /
9.51 % and 35.17 kcal / 13.77 %, i.e. 0.13 g of mass and 0.41 g of kcal worse than seed 0.
`s3_full_s1` (seed 1 of the ConvNeXt-S final) finished later at 19.07 g / 9.60 % and 35.74 kcal
/ 13.99 % (40 epochs, ~72 s per epoch, final train loss 0.0572 at epoch 39), i.e. 0.09 g of
mass and 0.53 g of kcal worse than its seed 0 - the worse of the two seed-1 runs on both
targets.

Conclusion: ConvNeXt-L does not fit at `--bs 4` in the shared memory budget, so it is retried
at `--bs 2 --accum 8` in round 8. The two seed pairs put the run-to-run noise floor at about
0.1 g of mass and 0.4-0.5 g of kcal, which is the scale on which every comparison above should
be read. Both seed-1 runs are worse than their seed-0 twin and both still improve an ensemble
(round 7).

### Round 7: ensembling the full-data models

Hypothesis: the geometric mean of the round-5 full-data finals already beat the best single
model (`s3_full` + `s3_raw_full` + `b3_raw_full` 18.45 g / 9.30 % and 34.38 kcal / 13.46 %
against `b3_raw_full` 18.75 / 9.45 % and 34.76 / 13.61 %), so averaging full-data models is a
lever in its own right; and with two seeds of both winners it is worth asking whether the seed
twins add diversity or only noise.

Result: six test-only ensembles (a `--full` run writes no `val_predictions.csv`, so none of
them can be chosen on val). `ens_b3_s3` (`b3_raw_full` + `s3_full`) is the best on kcal,
18.33 g / 9.23 % and 33.91 kcal / 13.28 %; `ens_all4` (`b3_raw_full` + `b3_raw_full_s1` +
`s3_full` + `s3_full_s1`) is the best on mass, 18.19 g / 9.16 % and 34.16 / 13.37 %. Against
`b3_raw_full` that is -0.42 g of mass and -0.85 g of kcal for the two-model mean and
-0.56 g / -0.60 g for the four-model one. Averaging seeds instead of backbones is weaker:
`ens_b3_seeds` 18.45 g / 34.69 kcal and `ens_s3_seeds` 18.71 / 35.00 both lose to `ens_b3_s3`
on both targets, although each beats its own seed-0 member. The val-split `s448_sam3b` does
not help either: `ens_b3_s3_s448` 18.39 g / 34.38 kcal is 0.06 g worse on mass and 0.47 g
worse on kcal than `ens_b3_s3`.

Conclusion: ensembling is worth about -0.4 g of mass and -0.9 g of kcal on top of the best
single model and costs no new training. The simplest ensemble is the best one: two backbones,
one seed each. The four-model `ens_all4` buys 0.14 g of mass over `ens_b3_s3` and gives back
0.25 g of kcal, a margin of the same order as the seed noise (0.09-0.13 g), so 18.19 vs 18.33 g
is not a result on its own. Diversity helps even when the added run is weak: `s3_full_s1` is
the worst of the full-data finals on the SAM 3 cache apart from `s3_raw_full` (19.07 g / 9.60 %
and 35.74 / 13.99 %, worse than its seed-0 twin's 18.98 / 9.56 and 35.21 / 13.78 %), yet
`ens_s3_seeds` beats `s3_full` on both targets (18.71 g / 35.00 kcal) and adding both seed
twins on top of `ens_b3_s3` takes mass to 18.19 g.

### Round 8: val-split twin, 60 epochs, seed 2 and the ConvNeXt-L retry

Hypothesis: every full-data-vs-val-split comparison so far mixed the extra 275 dishes with the
loss of val-based epoch selection, so the winner `b3_raw_full` needs its exact val-split twin to
put a number on full-data training; and a run that has both val and test numbers can measure the
size of the val/test gap itself. Three further runs ask whether anything else in the winner's
configuration is left: a longer schedule (the training loss was still falling at epoch 39), a
third seed (the two existing seeds differ by 0.13 g of mass), and ConvNeXt-L at a halved batch
after the round-6 OOM.

Result: `b3_raw_sam3b` is the winner's configuration on the val split instead of `--full`
(ConvNeXt-B, `raw`, SAM 3 cache, 40 epochs, ~92 s per epoch, peak 2.4 GB). It gives 19.95 g /
10.05 % mass and 36.02 kcal / 14.10 %, with fat 2.85 g / 22.24 %, carb 4.11 g / 20.90 % and
protein 3.80 g / 21.72 %. Against `b3_raw_full` (18.75 g / 9.45 % and 34.76 / 13.61 %) that is
-1.20 g of mass and -1.25 g of kcal for training on all 2755 `depth_train_ids` dishes instead of
2480, i.e. the full-data retrain measured like-for-like on the SAM 3 cache rather than inferred;
the confounded round-5 number splits cleanly into the two parts: 1.20 g of data plus 0.22 g of
cache make up its 1.42 g of mass, and 1.25 g plus 0.59 g its 1.85 g of kcal (to rounding). Its
own val numbers at the kept epoch (37) are 7.80 % mass and 12.22 % kcal, so on this run the val
mass number flatters the test one by about 2.2 points.
Against the old-cache val-split twin `b448_raw` (20.17 g / 10.16 %, 36.61 / 14.33 %) the SAM 3
cache is worth 0.22 g of mass and 0.59 g of kcal on a val-split run, the same mass effect round 4
measured at ConvNeXt-S (0.21 g).

The other three round-8 runs ask whether there is more in the winner's configuration and all
answer no. `b3_raw_full_e60` is that configuration at `--epochs 60` (full data, SAM 3 cache,
~103 s per epoch, peak 2.4 GB, final train loss 0.03758): 18.77 g / 9.46 % mass and 35.17 kcal /
13.77 % against the 40-epoch run's 18.75 / 9.45 and 34.76 / 13.61, i.e. +0.02 g of mass and
+0.41 g of kcal for 20 more epochs. Train loss falls from 0.0499 at epoch 39 to 0.0376 at 59, so
a quarter of the training loss buys nothing on test: the schedule is not the constraint, the
model is spending the extra epochs on the 2755 training dishes. `b3_raw_full_s2` is the third
seed of the same recipe, 18.61 g / 9.38 % and 34.30 / 13.43 %, the best single-run draw of the
study up to that point, 0.14 g of mass and 0.46 g of kcal below seed 0 and 0.27 g below seed 1;
the three seeds
span 0.27 g of mass (18.61-18.88), the same scale as the 0.13 g / 0.41 g the first pair set, so
seed 2 is a lucky draw rather than a better recipe (the results table bolds it because it was the
best finished number when this round closed). `l448_raw_bs2` is ConvNeXt-L at `--bs 2 --accum 8`
on the SAM 3 cache
after the round-6 `--bs 4 --accum 4` OOM, and it fails in the same place: epoch 0 trains (loss
0.5052, 213 s, peak 3.68 GB) and the first val forward then asks for 442 MiB with 469 MiB free,
4.33 GiB allocated on a 5.60 GiB card. Halving the training batch moved the peak by 0.06 GB
(3.74 -> 3.68), because the peak is the inference forward and `predict` batches at a hard-coded
32 dishes whatever `--bs` says; ConvNeXt-L therefore stays out of reach on this card without a
code change, and the capacity lever ends at ConvNeXt-B.

Round 8 also adds five ensembles, all test-only and recomputed from the members'
`test_predictions.csv`: the three `b3` seeds 18.26 g / 9.20 % and 34.32 / 13.43 %, the winner
plus its 60-epoch twin 18.56 g / 9.35 % and 34.78 / 13.62 %, the four full-data models without
`s3_full_s1` 18.14 g / 9.14 % and 33.81 / 13.24 %, the five full-data models 18.08 g / 9.11 %
and 33.92 / 13.28 %, and those five plus `b3_raw_full_e60` 18.08 g / 9.11 % and 33.95 / 13.29 %.
The five-member mean is the best ensemble of the study on mass, -0.67 g of mass and -0.84 g of
kcal against `b3_raw_full` and -0.11 g of mass / -0.24 g of kcal against round 7's `ens_all4`;
the four-member one is the best on kcal at 33.81 g, 0.11 g ahead of the five-member mean for
0.06 g of mass back, a trade inside the seed noise. The third seed is what buys the improvement:
the `b3` seed mean falls from 18.45 g (`ens_b3_seeds`, round 7) to 18.26 g with seed 2 in, 0.19 g
for one run and more than the 0.13 g the seed pair itself differed by, so seeds are not pure
noise in an ensemble even where the single-seed numbers are. The 60-epoch twin is not a useful
member: adding it to the winner gains 0.19 g, less than the 0.30 g a second seed gains, and
adding it to the five-member mean changes nothing (0.00 g of mass, +0.03 g of kcal) - a longer
schedule on the same seed reproduces the same errors instead of new ones.

Conclusion: the full-data retrain is the largest single lever at about -1.2 g of mass and -1.25 g
of kcal like-for-like, and the cache and data effects are additive rather than confounded. The
val/test gap is about 2.2 points on mass, so the val numbers in this log systematically flatter
the model and only the rankings, not the absolute values, carry over to test. Round 8 then
closes three further levers as flat: epochs (60 is 0.02 g of mass worse than 40 at ConvNeXt-B),
the single-seed draw (0.27 g of spread across three seeds of one recipe) and capacity
(ConvNeXt-L OOMs at every batch size tried). What is left is the ensemble, and only on the seed
axis: one more seed still pays 0.19 g there, while a longer schedule on the same seed pays
nothing, so round 9 spends its training slots on more independent seeds and on the regularisation
knob the 60-epoch run points at, and the rest on test-only ensembles.

### Round 9: the seed axis is exhausted, drop-path is closed

Hypothesis: round 8 left two things standing. The third `b3` seed cut the `b3` seed mean from
18.45 g to 18.26 g, 0.19 g for one run, so a fourth `b3` seed and a second `s3` seed should keep
paying in the ensemble even though single-seed numbers are noise; and the 60-epoch run showed
the model spending extra capacity on the 2755 training dishes, so stronger stochastic depth
(`--drop-path 0.2` against 0.1, same seed and schedule) should show on test where 20 more epochs
did not.

Result: `b3_raw_full_s3` (seed 3 of the winner) 18.91 g / 9.52 % and 35.76 kcal / 14.00 %;
`s3_full_s2` (seed 2 of the ConvNeXt-S final) 19.20 g / 9.67 % and 35.39 / 13.85 %;
`b3_raw_full_dp2` (the winner at `--drop-path 0.2`) 19.07 g / 9.61 % and 35.69 / 13.97 %. All
three are 40-epoch full-data runs on the SAM 3 cache at ~103 s per epoch and a 2.4 GB peak.

Three test-only ensembles, which write no `summary.json` and are recomputed from the members'
`test_predictions.csv`: `ens_all6` (the five-member mean + `b3_raw_full_s3`) 18.07 g / 9.10 %
and 34.10 kcal / 13.35 %; `ens_all7` (the same six + `s3_full_s2`) 18.03 g / 9.08 % and 34.00 /
13.31 %; `ens_all5_dp2` (the five-member mean + `b3_raw_full_dp2`, six members despite the name)
18.10 g / 9.12 % and 34.08 / 13.34 %. Their `TEST` lines print 18.1 / 9.1 % and 34.1 / 13.3 %,
18.0 / 9.1 % and 34.0 / 13.3 %, and 18.1 / 9.1 % and 34.1 / 13.3 % to one decimal.

Conclusion: the seed axis is exhausted. Against the five-member reference mean (18.08 g / 9.11 %
and 33.92 / 13.28 %, unchanged) a fourth ConvNeXt-B seed buys 0.01 g and both new seeds 0.05 g,
against the 0.19 g the third seed paid in round 8; both are inside the noise floor, so more
members of the same two backbones buy nothing and the ensemble has saturated. Drop-path 0.2 is
worse, not better: 19.07 g against 18.75 g for its drop-path-0.1 twin, 0.32 g and therefore
outside the noise floor, so this is a real (negative) reading and it closes the regularisation
lever; the differently-trained twin is also a worse ensemble member than the extra seed it
replaces (`ens_all5_dp2` 18.10 g against `ens_all6` 18.07 g, 0.03 g, inside the noise but the
same sign). The seed spread is re-confirmed on both backbones and is the evidence for the noise
floor itself: the four ConvNeXt-B seeds span 18.61-18.91 g, 0.29 g (0.30 on the two-decimal
values) against round 8's 0.27 g over three, and the three ConvNeXt-S seeds span 18.98-19.20 g,
0.22 g. With seeds and regularisation both spent, the two untested levers are the only ones left:
ConvNeXt-L, blocked on the hard-coded predict batch, and architecture diversity, which round 10
takes up.

### Round 10: resolution closed at ConvNeXt-B, the `log` loss at B, a second ConvNeXt generation

Hypothesis: round 9 closed the seed axis and the regularisation knob, leaving the two levers
`## How to beat the benchmark` ranked second and third - resolution at the winning recipe, never
run at ConvNeXt-B with `raw` on full data, and architecture diversity, where the ensemble has only
ever held ConvNeXt-S and ConvNeXt-B - plus the one loss that has never run at ConvNeXt-B.

Result: `b640_raw_full` (640 px at the winning recipe: ConvNeXt-B, `raw`, full data, `--bs 4
--accum 4`, the pre-SAM-3 `n5k_640` cache, 219 s per epoch, 2.41 GB peak) 19.85 g / 10.00 % mass
and 34.98 kcal / 13.69 %; `b3_log_full` (`log` instead of `raw` at ConvNeXt-B, full data, `--bs 8
--accum 2`, 40 epochs, seed 0, SAM 3 cache) 18.29 g / 9.21 % and 33.57 / 13.14 %; `cv2b_raw_full`
(`convnextv2_base.fcmae_ft_in22k_in1k`, a second ConvNeXt generation with FCMAE pretraining, full
data, `raw`, 193 s per epoch, 2.73 GB peak) 18.44 g / 9.29 % and 34.26 / 13.41 % - its `summary.json`
and `test_predictions.csv` were deleted by a guard rollback and both were regenerated from the
surviving `best.pt` in round 11 (see caveats). Three test-only ensembles, which write no `summary.json` and are known only to one
decimal from their `.log` files: `ens_all6_640` (the five-member mean + `b640_raw_full`, six
members) 18.1 g / 9.1 % and 33.8 kcal / 13.2 %; `ens_all6_log` (those five + `b3_log_full`) 18.0 /
9.1 % and 33.6 / 13.2 %; `ens_all7_log_cv2` (those five + `b3_log_full` + `cv2b_raw_full`, seven
members) 17.8 / 9.0 % and 33.3 / 13.1 %.

Conclusion: 640 px is a clear loss at this recipe, not a draw. `b640_raw_full` is 1.10 g of mass
and 0.22 g of kcal worse than its 448 twin `b3_raw_full` (19.85 against 18.75, 34.98 against
34.76); the mass difference is four times the noise floor, and even net of the 0.22 g of mass the
pre-SAM-3 `n5k_640` cache costs at ConvNeXt-B (round 8) the 640 recipe is 0.88 g worse. Resolution
is therefore closed in both directions: a wash at ConvNeXt-T (round 1, 0.22 g of mass) and 1.10 g
worse at ConvNeXt-B at the winning recipe. The loss function is the opposite result. `log` at
ConvNeXt-B is 0.46 g of mass and 1.19 g of kcal better than its matched `raw` twin (18.29 against
18.75, 33.57 against 34.76), both outside the noise floor, and it is the first new best single
model since round 5: 18.29 g beats `b3_raw_full_s2` 18.61 g. The third-ranked lever, never before
run at ConvNeXt-B, was worth more than another seed (~0.1 g) or another architecture (≈0.35 g,
below); the ranking in `## Levers, ranked` moves it to the top open slot. Architecture diversity
gives the third reading: `cv2b_raw_full` at 18.44 g is 0.31 g worse than the ConvNeXt-B `raw` twin
(18.75 g), about the size of the S-to-B step, but it is the member that pays in the mean
- adding it to the six-member `log` mean takes mass from 18.0 to 17.8 g and kcal from 33.6 to
33.3 g. On the ensemble axis round 10 otherwise confirms saturation: `ens_all6_640` (+0.02 g) and
`ens_all6_log` (-0.08 g) are draws against the five-member mean, so neither the 640 member nor the
`log` member adds anything on its own at six members, and the seven-member mean's -0.28 g of mass
is at the noise floor and is not claimed as a mass result even though its -0.58 g of kcal is a real
gain.

### Round 11: ConvNeXt-L trains at last, and the `log` loss under a reseed

Hypothesis: round 10 closed resolution in both directions and left the loss function as the
strongest open lever - but on one seed - so round 11 takes the two things round 10 could not: the
capacity step that has never trained (ConvNeXt-L), now that the predict batch is configurable, and
the `log` loss at a second seed, to see whether its 0.46 g of mass and 1.19 g of kcal survive a
reseed.

Result: `l448_full` (ConvNeXt-L, `--channels rgbd_thick --arch convnext_large.fb_in22k_ft_in1k
--grad-ckpt --bs 2 --accum 8 --loss raw --cache ../data/cache/n5k_448_sam3b --eval-bs 4`, val
split, 40 epochs at 207-212 s per epoch, 4.41 GB peak) trained, the first ConvNeXt-L of the study
to get past the inference forward: 20.21 g / 10.18 % mass and 34.64 kcal / 13.56 %, with the best
val score of any run here (9.4777, the mean of val mass 7.07 % and val kcal 11.89 %, reached at
the last epoch). `b3_log_full_s1` (the round-10 `log` recipe at `--seed 1`, full data, 40 epochs,
102.5 s per epoch, 2.4 GB peak) is 19.09 g / 9.61 % mass and 33.75 / 13.21 % kcal. Both items are
in `queue.txt.done` and the queue drained; the runner is `bash scripts/n5k_queue.sh
results/n5k/train/queue.txt` (pid 1671512).

Conclusion: the capacity step is real, and it is a kcal lever rather than a mass one.
`l448_full`'s only like-for-like comparison is its val-split twin `b3_raw_sam3b` (ConvNeXt-B, same
cache, same loss, round 8: 19.95 g / 10.05 % and 36.02 / 14.10 %): L is +0.26 g of mass - inside
the 0.29 g four-seed span of the `raw` recipe, so a draw - and -1.38 g of kcal, outside the
~0.6 g kcal floor, for twice the epoch time (207 s against 103 s). It is also the best val model
of the study (9.48 against 9.50 for `b448_raw`), the first time the val and test rankings of the
backbone axis agree. The full-data reference (`b3_raw_full` 18.75 g) stays 1.46 g of mass ahead
because the L ran on 2480 dishes rather than 2755 (`--full` caveat); the full-data gain measured
at ConvNeXt-B and ConvNeXt-S is -1.20 and -0.74 g, so an L on full data would be expected around
19.0 g - still not a mass win over the ConvNeXt-B recipe, and 2.4 h against 68 min.

The `log` loss splits in two, and that is the round's real finding. Its mass gain does not
replicate: seed 0 (`b3_log_full` 18.29 g) beat its matched `raw` twin by 0.46 g, but seed 1
(`b3_log_full_s1` 19.09 g) is 0.21 g *worse* than its matched `raw` twin (`b3_raw_full_s1`
18.88 g) and 0.79 g above the seed-0 log run. Two draws of the log recipe therefore span 0.79 g
against 0.29 g across four draws of the `raw` recipe, so `b3_log_full`'s 18.29 g - the number the
results table still carries as the best single model - is a lucky draw and not a recipe result;
the best reproducible single model is the `raw` ConvNeXt-B full-data recipe at 18.61-18.91 g. Its
kcal gain does replicate: `log` beats its matched `raw` twin at both seeds, by -1.19 g (33.57
against 34.76) and -1.42 g (33.75 against 35.17), both outside the ~0.6 g kcal floor and both in
the same direction, and the two log runs hold the two best kcal numbers of the whole study (33.57
and 33.75) against 34.30 for the best of the four `raw` draws. On mass the log loss is therefore
downgraded from the top open lever to not established; on kcal it is the only lever in the study
with a repeatable gain, and it is what holds `ens_all7_log_cv2` at 33.3 g of kcal. The ensemble
side repeats the same split: putting the second log seed in place of the first in the five-member
mean gives 18.0811 g / 9.1084 % and 33.6352 / 13.1662 % - a draw on mass against the five-member
18.0824 g and -0.28 g of kcal - and both log seeds together give 17.9940 / 9.0646 and 33.4485 /
13.0931, so the log member's mass contribution is a seed-0 artifact and its kcal contribution
repeats.

Re-verified from the surviving predictions rather than from a `.log`: the five-member full-data
mean is 18.0824 g / 9.1091 % and 33.9196 / 13.2775 %, the same five plus `b3_log_full` are
17.9741 g / 9.0545 % and 33.6106 / 13.1565 % (18.0 g / 9.1 % and 33.6 / 13.2 %, the six-member
control), and those six plus the recovered `cv2b_raw_full` are 17.7993 g / 8.9665 % and 33.3410 /
13.0510 % (17.8 g / 9.0 % and 33.3 / 13.1 %), which reproduces the round-10 `ens_all7_log_cv2`
`TEST` line exactly and confirms that the recovered member is in the mean rather than silently
dropped (caveats).

## Levers, ranked

Roughly, from strongest to weakest, with the test-set effect:

1. **Input channels: -2.8 g of mass.** `rgb` 24.14 g -> `rgbd_thick` 21.31 g (and -0.22 g of
   kcal). Depth alone is -1.70 g; the container-aware thickness channel adds the rest.
2. **Backbone size: -1.4 g of mass (T -> S) and -0.45 g more (S -> B at `raw`).** ConvNeXt-T
   21.31 g -> ConvNeXt-S 19.93 g -> ConvNeXt-B 20.17 g on the val split, with the B model
   also taking kcal from 37.13 to 36.61 and the best val score of the study (9.50). This is
   the only lever that improves both targets together at the earlier steps. It ended at B for
   mass: ConvNeXt-L OOMed twice (rounds 6 and 8) in the inference forward rather than in the
   training step, because `predict` batched at a hard-coded 32 dishes whatever `--bs` said, and
   round 11's `l448_full` - the first L run with the `--eval-bs` flag that fixes that - is
   20.21 g / 10.18 % and 34.64 / 13.56 % against its ConvNeXt-B val-split twin `b3_raw_sam3b`
   19.95 / 10.05 and 36.02 / 14.10: +0.26 g of mass (a draw, inside the 0.29 g four-seed span of
   the `raw` recipe) and -1.38 g of kcal (outside the ~0.6 g kcal floor), for twice the epoch time
   (207 s against 103 s). So the capacity axis has ended at B for mass and continues to pay only
   on kcal (`## Next`).
3. **Architecture diversity for the ensemble: ≈0.35 g as a single model, -0.28 g of mass /
   -0.58 g of kcal in the mean (round 10, 1-dp numbers).** `cv2b_raw_full`
   (`convnextv2_base.fcmae_ft_in22k_in1k`, a second ConvNeXt generation with FCMAE pretraining)
   is 18.44 g / 9.29 % against 18.75 g / 9.45 % for the ConvNeXt-B `raw` twin, so on its own it
   is roughly 0.31 g worse - about the size of the S-to-B step - but as an ensemble member it is
   the one that pays: adding it to the six-member `log` mean takes mass from 18.0 to 17.8 g and
   kcal from 33.6 to 33.3 g (`ens_all7_log_cv2`). Its exact numbers were regenerated from the
   surviving `best.pt` in round 11 and the ensemble mean that depends on it was re-scored (caveats).
4. **Mask cache quality: -0.21 g of mass.** Old masks 19.93 g -> SAM 3 masks 19.72 g at the
   same backbone and loss, with the val score moving from 11.21 to 11.02.
5. **Full-data training (2755 dishes instead of 2480): -0.74 g of mass and -1.76 g of kcal**
   at ConvNeXt-S, and -1.20 g / -1.25 g at ConvNeXt-B with `raw` against its val-split twin
   `b3_raw_sam3b` on the same cache (round 8). The largest single drop, though the val-split
   twin still keeps its best-val epoch rather than the last one (see caveats).
6. **Loss function: a kcal lever, not a mass one (round 11).** At ConvNeXt-T the three losses
   are indistinguishable (`log` 21.31 g / 38.83 kcal, `raw` 21.67 / 39.09, `mix` 21.52 / 38.31)
   and at ConvNeXt-S on the old cache `log` 19.93 g beats `raw` 20.62 on mass, but at ConvNeXt-B
   on full data round 10 measured `log` 18.29 g / 9.21 % and 33.57 / 13.14 % against `raw` 18.75 /
   9.45 and 34.76 / 13.61, i.e. -0.46 g of mass and -1.19 g of kcal, and the new best single
   model. Round 11's reseed splits that in two: the mass gain reverses (seed 1: `log` 19.09 g
   against `raw` 18.88 g, +0.21 g, with the two log seeds spanning 0.79 g against 0.29 g over four
   `raw` seeds) while the kcal gain repeats (seed 1: 33.75 against 35.17, -1.42 g), and the two
   log draws hold the two best kcal numbers of the study. The mass side is back to untested/not
   established, and the kcal side is the one repeatable lever the study has.
7. **Resolution: closed in both directions.** 640 px buys 0.22 g of mass (inside noise) and
   loses 1.36 g of kcal at ConvNeXt-T (round 1), and at ConvNeXt-B at the winning recipe
   (round 10) `b640_raw_full` is 19.85 g / 10.00 % and 34.98 / 13.69 % against 18.75 / 9.45 and
   34.76 / 13.61 for its 448 twin `b3_raw_full`, i.e. +1.10 g of mass and +0.22 g of kcal; net
   of the 0.22 g the pre-SAM-3 `n5k_640` cache costs at ConvNeXt-B, the 640 recipe is 0.88 g
   worse. Higher resolution is dropped.
8. **Epoch count and EMA: neutral.** At ConvNeXt-T, 60 epochs + EMA 0.999 gives exactly the
   40-epoch mass (21.31 g) and 0.62 g worse kcal; at ConvNeXt-B with `raw` on the SAM 3 cache,
   60 epochs gives 18.77 g / 9.46 % and 35.17 / 13.77 % against 18.75 / 9.45 and 34.76 / 13.61,
   i.e. +0.02 g of mass and +0.41 g of kcal (round 8).

Ensembling the full-data models is worth about -0.4 g of mass and -0.9 g of kcal on top of the
best single model (`b3_raw_full` 18.75 g / 34.76 kcal -> `ens_b3_s3` 18.33 g / 33.91 kcal for
the two-model geometric mean), and the more full-data models are averaged the further it goes:
`ens_all4` 18.19 g / 34.16 kcal (round 7) and the five-member mean 18.08 g / 33.92 kcal (round 8,
-0.67 g of mass and -0.84 g of kcal against `b3_raw_full`), i.e. a similar size to the
mask-cache lever and free of new training. The diversity has to come from somewhere real:
another seed pays (0.19 g for the third `b3` seed) while a longer schedule on the same seed
does not (0.00 g added to the five-member mean). Round 10 takes the mean to 17.8 g / 9.0 % and
33.3 / 13.1 % with a seven-member mean that adds the `log` model and the ConvNeXtV2 model to the
five (`ens_all7_log_cv2`), the first ensemble gain since round 8 - and the only one in round 10,
since the six-member means with either new member alone are draws on mass.

## How to beat the benchmark

The bar is 29.4 g / 13.7 % mass and 41.3 kcal / 16.5 % kcal (the volume-scalar reference,
`docs/model_survey.md`, `results/n5k/README.md`). The five-member full-data mean is 18.08 g /
9.11 % and 33.92 / 13.28 %, so the margin is 11.32 g and 4.59 points of mass and 7.38 g / 3.22
points of kcal; round 10's seven-member mean (`ens_all7_log_cv2`, 1-dp) is 17.8 g / 9.0 % and
33.3 / 13.1 %, about 11.6 g and 4.7 points of mass and 8.0 g / 3.4 points of kcal, and the best
single model on the table (`b3_log_full` 18.29 / 9.21 %, a draw that round 11 does not
reproduce) is 11.11 g clear, with the best reproducible one (`b3_raw_full_s2` 18.61 / 9.38 %)
10.79 g clear. The benchmark is not what is
binding: the levers below move tenths of a gram, not the 11 g to the reference.

Ranked by expected test-set gain. **Noise floor: four seeds of one recipe now span 0.29 g of
mass (18.61-18.91, round 9; 0.27 g over three in round 8), the three ConvNeXt-S seeds span
0.22 g, the two seed pairs differ by 0.13 and 0.09 g, and the val/test gap is about 2.2 points,
so anything under ~0.2 g of mass is a draw and must not be claimed as a result, while a kcal
difference of ~0.6 g is worth reporting.** Round 10 was read on exactly that rule: the 640
recipe's +1.10 g of mass and the `log` model's -0.46 g / -1.19 g were read as results, the
six-member ensembles' +0.02 and -0.08 g of mass as draws, and the seven-member mean's -0.28 g of
mass at the floor with its -0.58 g of kcal a real gain. Round 11 is read the same way, and it is
what shows the rule's limit: the `log` model's -0.46 g of mass fails it on a second seed
(+0.21 g), so a gain measured on one seed of a recipe whose two seeds span 0.79 g is not a result;
the same model's -1.42 g of kcal at seed 1 passes it, and so does ConvNeXt-L's -1.38 g of kcal,
while its +0.26 g of mass does not. A ConvNeXt-B run at 40 epochs on the SAM 3 cache costs 68 min
and peaks at 2.4 GB (~103 s per epoch), and a ConvNeXt-L run costs 2.4 h (207-212 s per epoch);
the cheapest lever in this list was the seed, and round 9 closed it at 0.01 g.

1. **Capacity at ConvNeXt-L: measured by round 11, and it is a kcal step, not a mass one.**
   Capacity paid at every earlier step it was tried: ConvNeXt-T 21.31 g -> S 19.93 -> B 20.17 on
   the val split with `raw` held (-1.38 g, then -0.45 g), and B was the best val model of the
   study (9.50). The L step never ran, because both attempts died in the inference forward rather
   than in the training step - `l448_raw` at `--bs 4 --accum 4` (peak 3.74 GB) and `l448_raw_bs2`
   at `--bs 2 --accum 8` (3.68 GB) both stopped on the identical 442 MiB allocation with 4.33 GiB
   already allocated, and halving the training batch moved the training peak by only 0.06 GB, so
   the training batch was not what bound. `predict` in `pipeline/eval/n5k_train.py` batched at a
   hard-coded 32 dishes whatever `--bs` said, so the eval forward always ran at 32; the
   `--eval-bs` flag (one new argument, one changed loop, default 32, i.e. the old behaviour when
   it is not passed) is in the working copy (md5 `3a8a9bdaac09daa9f00203eef6570d64`, was
   `befe896230a71a655bec6249c3222cf5`; the stored patch `/home/kyle/n5k_eval_bs.patch`, md5
   `cb02ac1a16df3e5e09197191aa47e913`), and `l448_full` is the first run to use it: `--bs 2
   --accum 8 --eval-bs 4`, 40 epochs at 207-212 s per epoch, 4.41 GB peak, no OOM. The result is
   +0.26 g of mass (a draw, inside the 0.29 g seed span) and -1.38 g of kcal against its
   ConvNeXt-B twin `b3_raw_sam3b`, so the axis is closed for mass - the gain per capacity step was
   already shrinking (1.38 -> 0.45 -> 0.26 g) - and an L is worth its 2x epoch time only for kcal.
2. **Resolution at the winning recipe: closed by round 10.** The 640 cache is not unused:
   `t640` (round 1) trained on it at ConvNeXt-T with `log` on the val split, 21.09 g / 10.62 %
   and 40.19 kcal / 15.73 %, against 21.31 / 10.74 and 38.83 / 15.20 for its 448 twin - 0.22 g of
   mass better (inside noise), 1.36 g of kcal worse, for 2.3x the epoch time (75 s against 33 s),
   and that already closed resolution at ConvNeXt-T. Round 10 ran it at the winning recipe -
   ConvNeXt-B + `raw` + full data - and it is 1.10 g of mass and 0.22 g of kcal worse than its 448
   twin (`b640_raw_full` 19.85 / 10.00 % and 34.98 / 13.69 % against `b3_raw_full` 18.75 / 9.45
   and 34.76 / 13.61), 0.88 g worse even net of the 0.22 g of mass the pre-SAM-3 `n5k_640` cache
   costs at ConvNeXt-B. Cost was ~40 epochs, 2.41 GB, ~2.4 h at 219 s per epoch. The 640 member
   also adds nothing to the ensemble at six members (`ens_all6_640` +0.02 g of mass against the
   five-member mean), so both halves of the lever are closed.
3. **Architecture diversity for the ensemble: ~0.2-0.4 g in the mean, partly measured.** The
   measured ensemble axis is the seed: the third `b3` seed cut the seed mean 18.45 -> 18.26 g
   (0.19 g for one run), while a fourth and the second `s3` seed together paid 0.05 g in round 9,
   so the seed axis is spent. Backbone diversity has paid more than seed
   diversity - `ens_b3_s3` (two backbones, one seed each) is 18.33 g against 18.45 g for the `b3`
   seed pair, and round 7 reads the cross-backbone member as worth 0.42 g - and round 10 is the
   first time the ensemble has held a third family. `cv2b_raw_full` is 18.44 g / 9.29 % (its exact
   values regenerated from the surviving `best.pt` in round 11), 0.31 g worse than the ConvNeXt-B
   `raw` twin, and adding it to the six-member `log` mean takes mass from 18.0 to 17.8 g and kcal
   from 33.6 to 33.3 g (`ens_all7_log_cv2`); the member is worth about -0.28 g of mass and -0.58 g
   of kcal, so the kcal side is the real gain.
   The limit is still the stem injection, not `--arch`:
   `--arch` is a free-form timm name (`timm.create_model`, line 47), but the four-channel
   injection replaces `model.stem[0]` for every name without "vit" in it (line 49), so only
   ConvNeXt/ConvNeXtV2-style stems qualify. Checked in this environment:
   `convnextv2_base.fcmae_ft_in22k_in1k` works (`stem[0]` is the Conv2d), the `vit_*` family works
   through `patch_embed.proj`, and swin, maxvit, coatnet, caformer, regnet, resnet, efficientnet
   and deit3 all raise (no `stem[0]`, or a `patch_embed.proj` the "vit" test never reaches). So
   the diversity available today is the second ConvNeXt generation (now measured) or a supervised
   ViT-B; a depth-aware or otherwise different family needs the stem injection fixed first. Cost:
   one 40-epoch run, 68 min, 2.4 GB, and it is the cheapest untested item left.
4. **Test-time augmentation beyond the four flips: under 0.2 g, inside the noise floor.** TTA
   exists and is flips only: `flips` (lines 106-107) gives identity, horizontal, vertical and
   both, averaged on test (line 262) and for the saved val predictions (line 256), while the
   per-epoch val forward that picks the kept epoch uses no TTA (line 239). `eval.n5k_ensemble.py`
   adds nothing - it reads `test_predictions.csv` and takes the geometric mean - so no new views
   can be made from saved predictions. Rotation TTA (training already rotates, so it is
   metric-safe) and multi-crop are the untested variants and need a code change in `predict`.
   Resize/scale TTA is excluded by design: the augmentation never scales because pixel size
   carries portion size, so a resized view asks the model to regress a different portion. A
   better view of one model is worth a fraction of the 0.4-0.9 g a second model's views are worth
   in the ensemble, so expect a fraction of a tenth of a gram, measured and not claimed.
5. **Head and loss variants: the loss is now read on two seeds (round 11); the head is untested.**
   The loss is closed at ConvNeXt-T (`log` 21.31, `raw` 21.67, `mix` 21.52 g). At ConvNeXt-B on
   full data round 10 measured `log` 18.29 g / 9.21 % and 33.57 / 13.14 % against `raw` 18.75 /
   9.45 and 34.76 / 13.61 (-0.46 g of mass, -1.19 g of kcal), and round 11's reseed keeps only the
   kcal half of that: seed 1 gives `log` 19.09 g / 9.61 % and 33.75 / 13.21 % against `raw` 18.88 /
   9.51 and 35.17 / 13.77, i.e. +0.21 g of mass and -1.42 g of kcal. So the log loss is the
   study's one repeatable kcal lever and no longer a mass lever, and its value in the ensemble is
   the kcal side too (round 11: -0.28 g of kcal when the second seed replaces the first, -0.09 g
   of mass with both log seeds in against the five-member mean). The head is a single 5-way L1
   against a per-target MAE metric, so per-target loss weights (mass is the headline), a two-stage
   head (mass first, then kcal conditioned on predicted mass - the quantity the reference reports)
   and a tail-weighted term (the log-scale loss under-predicts the heaviest dishes by ~20 g) are
   untested, one run each.
6. **More seeds of the winner: closed by round 9.** The third seed was worth 0.19 g in the `b3`
   seed mean (round 8) and the fourth paid 0.01 g (`ens_all6` 18.07 g against the five-member
   mean's 18.08 g), with both new seeds in at 0.05 g (`ens_all7` 18.03 g). It paid nothing on the
   single model either (0.29 g of spread across four draws of one recipe), so the axis with the
   least headroom is now spent - it was worth measuring first because it was the cheapest, 68 min
   and 2.4 GB per run.

Closed, for the record: full-data training (-1.20 g / -1.25 g, the largest single drop), the
SAM 3 mask cache (-0.22 g / -0.59 g), the schedule (60 against 40 epochs: +0.02 g of mass, +0.41 g
of kcal), resolution at ConvNeXt-T and now at ConvNeXt-B at the winning recipe (round 10, +1.10 g
of mass), and the loss at ConvNeXt-T. Round 10 moved the loss function out of this list at ConvNeXt-B, and round 11 puts half of it
back: the mass half was a lucky seed (+0.21 g at seed 1) and the kcal half repeats (-1.19 g and
-1.42 g), so the closed list gains the `log` loss on mass and keeps the kcal gain open. Round 11
also closes the capacity axis for mass (ConvNeXt-L +0.26 g, a draw). Nothing in the ranked list
needs a new cache, and nothing in it is larger than the 1.2 g the full-data retrain was worth: at
this point one run buys about 0.2 g, which is the seed noise itself, so a round is a sequence of
coin flips that only the ensemble mean can read.

## Caveats

- **`--full` is not like-for-like with the val-split runs.** A full run trains on 2755 dishes;
  a val-split run trains on 2480 and holds out 275. Every "full vs val-split" comparison above
  therefore mixes the extra 275 dishes with the loss of val-based epoch selection, and the
  val-split runs additionally keep the epoch that was best on val rather than the last one.
  `b3_raw_sam3b` (round 8) narrows the comparison to that difference alone: it shares the cache,
  the backbone and the loss with `b3_raw_full` and differs only in `--full` vs the val split.
- **Single seed for most runs.** The two seed pairs that exist differ by 0.13 g of mass and
  0.41 g of kcal (`b3_raw_full` 18.75 g vs `b3_raw_full_s1` 18.88 g) and by 0.09 g / 0.53 g
  (`s3_full` 18.98 g vs `s3_full_s1` 19.07 g), so the noise floor is roughly 0.1 g of mass and
  0.5 g of kcal; differences smaller than that are not evidence of anything. `b3_raw_full_s2`
  (round 8) lands 0.14 g below seed 0 and 0.27 g below seed 1, and `b3_raw_full_s3` (round 9)
  0.15 g above seed 0, so four seeds of one recipe span 0.29 g of mass (18.61-18.91) and the best
  single-run number in the results table is that draw, not a recipe; the three ConvNeXt-S seeds
  span 0.22 g (18.98-19.20). Averaging a seed pair still lowers mass by 0.27-0.30 g and kcal by
  0.07-0.21 g (round 7), a third seed lowers the `b3` seed mean by a further 0.19 g (round 8) and
  a fourth adds 0.01 g (round 9), so the seed noise is at least partly independent across runs
  but the ensemble has saturated. `b3_log_full` (round 10) was likewise a single seed, and round
  11's reseed (`b3_log_full_s1`, 19.09 g) shows it was a lucky draw: the two `log` seeds span
  0.79 g of mass against 0.29 g across four `raw` seeds, so either the log recipe is noisier or
  its seed 0 was simply lucky - either way 18.29 g does not reproduce, and the best reproducible
  single model is the `raw` recipe's 18.61-18.91 g band.
- **A mass difference under ~0.2 g is not a result.** Four seeds of one recipe span 0.29 g of
  mass and the three ConvNeXt-S seeds 0.22 g, so the run-to-run noise on mass is ~0.1-0.3 g;
  round 10's six-member ensembles move by +0.02 and -0.08 g of mass and its seven-member mean by
  -0.28 g, all of which are draws on mass and are not claimed, while the same mean's -0.58 g of
  kcal and the `log` model's -0.46 g of mass / -1.19 g of kcal are outside it and are reported.
  One round-10 number was weaker still: `cv2b_raw_full` existed only to one decimal until round 11
  regenerated its exact values from the surviving `best.pt` (next bullet). Round 11 adds a second caution of the same
  kind: the two `log` seeds differ by 0.79 g, so the noise floor is not one number for every
  recipe, and a 0.46 g gain measured on one seed of a recipe whose two seeds span 0.79 g is not a
  result.
- **Val is easier than test.** The best val kcal MAPE of any run is 11.65 % (`b448_raw`,
  epoch 37) against 13.61 % for the reference model's test kcal MAPE (13.43 % for its seed-2
  draw, round 8), a gap of about 2 points; val dishes share incremental-capture siblings with
  train. The best val mass MAPE is 7.35 % against 9.45 % on test (9.38 % for the seed-2 draw). Rankings held across the two, but the absolute val numbers
  should not be quoted as results. `b3_raw_sam3b` (round 8) shows the gap on a single model:
  7.80 % val mass against 10.05 % on test.
- **Memory budget.** ConvNeXt-L did not fit on the shared 5.6 GB card at `--bs 4 --accum 4`
  (`l448_raw`) or at `--bs 2 --accum 8` (`l448_raw_bs2`, round 8): both OOMed in the first
  forward after epoch 0, with a training peak of 3.74 and 3.68 GB, and the `--bs 2` retry showed
  that the binding allocation was the inference forward (4.33 GiB) rather than the training step,
  because `predict` batched at a hard-coded 32 dishes whatever `--bs` said. That code change
  (`--eval-bs`, default 32, so unchanged when it is not passed) is in the working copy as of
  round 11 and `l448_full` is the first run to use it: 4.41 GB peak, no OOM, 207-212 s per epoch,
  so ConvNeXt-L is now measured (round 11, results table) and the memory caveat applies only to
  backbones larger than L. The ViT-B (DINOv2) run was stopped early for convergence, not for
  memory.
- **No val predictions for the full runs.** `--full` runs save only `test_predictions.csv`, so
  an ensemble that includes them cannot be selected on val; rounds 7 and 8 scored them on test
  only.
- **A lost result, recovered: `cv2b_raw_full`'s metrics were deleted by a guard rollback and
  regenerated from the surviving `best.pt`.** The run finished, but a poll command had its cwd
  inside the repository when it did, and the repo guard rolled back the two new files:
  `results/n5k/train/cv2b_raw_full/summary.json` and its `test_predictions.csv`; only `best.pt`
  (350,913,281 bytes, md5 `2c7e65c6a9604abb31b1cfffded19e6a`) survived. Round 11 re-ran the test
  forward on that `best.pt` with `/tmp/n5k_recover_preds.py` (6563 bytes, md5
  `f6ef9565690998a35672310c112960c3`; its `predict()` is a copy of the nested one in
  `eval.n5k_train.py` main(), with the same four-flip TTA, 507 dishes at `--eval-bs 4`), which
  wrote `test_predictions.csv` (44,254 bytes, md5 `ede21b53c793c0728e57f6f83817eb91`) and
  `summary.json` (5,449 bytes, md5 `90405d94c328c18b4f6f4b7c483430aa`) into a scratch directory,
  and its `TEST` line is the original one to the digit: 18.4 g / 9.3 % and 34.3 / 13.4 % (exact
  18.4432 g / 9.2909 % and 34.2613 / 13.4113 %). The recovery is confirmed by the ensemble it
  feeds: the seven members of `ens_all7_log_cv2` re-scored with the recovered predictions are
  17.7993 g / 8.9665 % and 33.3410 / 13.0510 % - exactly the 17.8 g / 9.0 % and 33.3 / 13.1 % of
  `ens_all7_log_cv2.log` - against 17.9741 g / 9.0545 % and 33.6106 / 13.1565 % for the same six
  members without `cv2b_raw_full`, the control that shows the recovered member is in the mean
  rather than silently dropped.
- **Selection.** Every epoch, backbone, loss and cache choice above was made on the val split
  or from earlier rounds; the 507 test dishes were only ever scored, never used to choose a
  configuration.

## Next

Round 7 is finished: six test-only ensembles, `ens_b3_s3`, `ens_b3_s3raw`, `ens_b3_seeds`,
`ens_s3_seeds`, `ens_all4` and `ens_b3_s3_s448` (appended to `results/n5k/train/queue.txt`).

Round 8 is finished: `b3_raw_sam3b` 19.95 g / 10.05 %, `b3_raw_full_e60` 18.77 / 9.46 %,
`b3_raw_full_s2` 18.61 / 9.38 % and `l448_raw_bs2` failed, plus five test-only ensembles; the
round-8 section above has the numbers and what they mean (60 epochs no better, seed 2 a lucky
draw inside a 0.27 g spread, ConvNeXt-L still OOM in the inference forward). The queue drained
at the end of round 8.

Round 9 is finished: `b3_raw_full_s3` 18.91 g / 9.52 %, `s3_full_s2` 19.20 / 9.67 % and
`b3_raw_full_dp2` 19.07 / 9.61 %, plus the three test-only ensembles `ens_all6` 18.07 g / 9.10 %,
`ens_all7` 18.03 / 9.08 % and `ens_all5_dp2` 18.10 / 9.12 %; the round-9 section above has the
reading (the seed axis is exhausted, drop-path 0.2 is 0.32 g worse and closes the regularisation
lever, and the 0.29 g spread over four `b3` seeds re-confirms the noise floor). All six of its
items are in `queue.txt.done`. The training slots went on the two things round 8 left standing -
more independent seeds of the two full-data winners, whose seed mean paid 0.19 g for the third
seed, and the one regularisation knob the 60-epoch run points at, since 20 more epochs cut
training loss by a quarter (0.0499 -> 0.0376) and gave 0.41 g of kcal back - and the rest on
test-only ensembles of the members that exist. ConvNeXt-L was not retried, because `l448_raw_bs2`
shows the constraint is the inference forward, which no `--bs` fixes.

Round 10 is finished: `b640_raw_full` 19.85 g / 10.00 %, `b3_log_full` 18.29 / 9.21 % (the new
best single model), `cv2b_raw_full` 18.44 / 9.29 % (regenerated from its surviving `best.pt`, see caveats) and the three test-only
ensembles `ens_all6_640` 18.1 g / 9.1 %, `ens_all6_log` 18.0 / 9.1 % and `ens_all7_log_cv2`
17.8 / 9.0 % and 33.3 kcal / 13.1 % (the new best overall); the round-10 section above has the
reading (640 px is 1.10 g worse at ConvNeXt-B and closes resolution in both directions, the `log`
loss at ConvNeXt-B is worth 0.46 g of mass and 1.19 g of kcal, and the ConvNeXtV2 member is what
takes the ensemble to 17.8 g). All six of its items are in `queue.txt.done`, whose 46 lines end
with `b3_log_full`, `cv2b_raw_full`, `ens_all6_640`, `ens_all6_log` and `ens_all7_log_cv2`. The
training slots went on the three untested levers round 9 left standing - resolution at the winning
recipe, the `log` loss at ConvNeXt-B, and a second architecture family for the ensemble - and the
ensembles were test-only.

Round 11 is finished: `l448_full` 20.21 g / 10.18 % and 34.64 / 13.56 % (ConvNeXt-L on the val
split, the first L of the study to train, via `--eval-bs 4`) and `b3_log_full_s1` 19.09 / 9.61 %
and 33.75 / 13.21 % (the round-10 `log` recipe at seed 1), plus the round-11 ensemble readings
recomputed from the surviving predictions (five-member mean 18.0824 g / 9.1091 %, the log member
at seed 1 in its place 18.0811 / 9.1084, both log seeds 17.9940 / 9.0646). The round-11 section
above has the reading: the capacity step is a kcal step (L +0.26 g of mass, a draw, -1.38 g of
kcal against its ConvNeXt-B twin) and the `log` loss's mass gain does not survive a reseed
(+0.21 g at seed 1) while its kcal gain does (-1.42 g). Both items are in `queue.txt.done`, whose
48 lines end with `l448_full` and `b3_log_full_s1`, and the queue drained at the end of the
round; the runner is `bash scripts/n5k_queue.sh results/n5k/train/queue.txt` (pid 1671512).

- **What comes after, in order.** (1) The `cv2b_raw_full` recovery is done and proven: the
  seven-member `ens_all7_log_cv2` re-scored from the recovered predictions reproduces the round-10
  line exactly (17.7993 g / 8.9665 % and 33.3410 / 13.0510 %) against the six-member control
  (17.9741 / 9.0545 and 33.6106 / 13.1565) - see the "A lost result, recovered" caveat.
  (2) The `log` loss where it now belongs: as a kcal lever at ConvNeXt-B it repeats a -1.2 to
  -1.4 g gain, and the one run that would put both effects on a single model is a `log`
  ConvNeXt-L on full data (an L already wins kcal with `raw`; untested, 2.4 h). (3) The levers
  `## How to beat the benchmark` still ranks open: architecture diversity beyond ConvNeXtV2 (the
  stem injection blocks the other families), rotation TTA beyond the four flips, and the head
  variants (per-target weights, a two-stage mass-then-kcal head, a tail-weighted term). Nothing in
  that list is worth more than about 0.4 g of mass and the full-data retrain (1.2 g) has no
  analogue left, so the recipe is close to finished: with the members that exist the ensemble mean
  is 17.99-18.08 g of mass, and further runs buy tenths of a gram.

For the ranked list of the levers that are still open, with their expected gains and what each
costs in GPU time, see `## How to beat the benchmark` above.

## How to reproduce

The queue runner (from the repository root; it re-reads the queue every 30 s and skips names
already in `queue.txt.done`, so lines can be appended while it runs):

```bash
bash scripts/n5k_queue.sh results/n5k/train/queue.txt
```

A single training run, from `pipeline/` (this is the reference single model, `b3_raw_full`):

```bash
conda run -n foodseg --no-capture-output python -m eval.n5k_train ../data/datasets/Nutrition5k \
    --channels rgbd_thick --arch convnext_base.fb_in22k_ft_in1k --grad-ckpt \
    --bs 8 --accum 2 --loss raw --full --cache ../data/cache/n5k_448_sam3b \
    --out ../results/n5k/train/b3_raw_full
```

An ensemble, from `pipeline/` (geometric mean of the member predictions, scored on test):

```bash
conda run -n foodseg --no-capture-output python -m eval.n5k_ensemble ../data/datasets/Nutrition5k \
    ../results/n5k/train/b3_raw_full ../results/n5k/train/s3_full
```

The metric is recomputable without retraining: each run writes
`results/n5k/train/<name>/test_predictions.csv` with `true_<target>` and `pred_<target>`
columns, and `eval.n5k_train.score(ids, pred)` returns `{target: (MAE in grams, 100 * MAE /
mean)}` for the five targets, which is exactly what the `TEST` line in `<name>.log` and the
`test` block of `summary.json` report.

## A different task in the same repository: MetaFood3D segmentation

For completeness, and separate from the mass regression above: `results/metafood3d/seg/`
holds the touching-food 3D segmentation runs (`eval/mf3d_seg_train.py`, DGCNN with offset
voting). On the MetaFood3D composite set, the full-scan run (`dgcnn_full`) reaches mIoU
0.9453 with a per-item volume MAPE of 9.82 %, and the single-top-view run (`dgcnn_top`)
reaches mIoU 0.9475 with 8.59 %. Those are segmentation and volume numbers on a different
dataset and do not feed into any number in the Nutrition5k tables above.
