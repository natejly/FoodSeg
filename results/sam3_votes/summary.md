# SAM 3 real votes vs pseudo votes (oracle names)

11 scans.

| scan | items | wrong real | wrong pseudo | share real | share pseudo | mIoU real | mIoU pseudo | merges r/p | splits r/p | unvoted real |
|---|---|---|---|---|---|---|---|---|---|---|
| s000000 | 4 | 0.945 | 0.020 | 0.778 | 0.953 | 0.092 | 0.936 | 0/0 | 1/0 | 0.851 |
| s000001 | 5 | 0.526 | 0.065 | 0.939 | 0.908 | 0.333 | 0.733 | 1/1 | 0/2 | 0.463 |
| s000002 | 8 | 1.000 | 0.081 | 0.873 | 0.911 | 0.000 | 0.822 | 0/0 | 0/0 | 0.980 |
| s000007 | 5 | 0.522 | 0.246 | 0.883 | 0.860 | 0.461 | 0.733 | 1/0 | 0/0 | 0.498 |
| s000009 | 3 | 0.688 | 0.099 | 0.914 | 0.762 | 0.342 | 0.762 | 0/1 | 0/1 | 0.684 |
| s000011 | 3 | 0.237 | 0.173 | 0.858 | 0.776 | 0.594 | 0.746 | 0/0 | 0/0 | 0.165 |
| s000012 | 4 | 0.355 | 0.049 | 0.954 | 0.925 | 0.703 | 0.934 | 0/0 | 0/0 | 0.321 |
| s000013 | 2 | 0.959 | 0.047 | 0.882 | 0.949 | 0.355 | 0.828 | 0/0 | 0/0 | 0.934 |
| s000014 | 5 | 0.161 | 0.084 | 0.981 | 0.929 | 0.659 | 0.737 | 1/0 | 0/0 | 0.012 |
| s000016 | 4 | 0.542 | 0.066 | 0.913 | 0.899 | 0.395 | 0.852 | 0/0 | 0/0 | 0.493 |
| s000017 | 4 | 0.765 | 0.198 | 0.879 | 0.832 | 0.203 | 0.674 | 1/1 | 0/1 | 0.632 |

| mean | real SAM 3 | pseudo |
|---|---|---|
| food points with lift_label != label | 0.609 | 0.103 |
| food points with no vote | 0.548 | 0.025 |
| mean lift_share (voted food points) | 0.896 | 0.882 |
| per-scan instance mIoU | 0.376 | 0.796 |
| merges (total) | 4 | 3 |
| splits (total) | 1 | 4 |
| SAM instances with no GT match (total) | 17 | - |
| GT items with no SAM match (total) | 32 | - |

## Oracle names vs generic 'food' (same 4 scans: s000000 s000001 s000002 s000011)

# SAM 3 real votes vs pseudo votes (oracle names, 4 scans)

4 scans.

| scan | items | wrong real | wrong pseudo | share real | share pseudo | mIoU real | mIoU pseudo | merges r/p | splits r/p | unvoted real |
|---|---|---|---|---|---|---|---|---|---|---|
| s000000 | 4 | 0.945 | 0.020 | 0.778 | 0.953 | 0.092 | 0.936 | 0/0 | 1/0 | 0.851 |
| s000001 | 5 | 0.526 | 0.065 | 0.939 | 0.908 | 0.333 | 0.733 | 1/1 | 0/2 | 0.463 |
| s000002 | 8 | 1.000 | 0.081 | 0.873 | 0.911 | 0.000 | 0.822 | 0/0 | 0/0 | 0.980 |
| s000011 | 3 | 0.237 | 0.173 | 0.858 | 0.776 | 0.594 | 0.746 | 0/0 | 0/0 | 0.165 |

| mean | real SAM 3 | pseudo |
|---|---|---|
| food points with lift_label != label | 0.677 | 0.085 |
| food points with no vote | 0.615 | 0.012 |
| mean lift_share (voted food points) | 0.862 | 0.887 |
| per-scan instance mIoU | 0.255 | 0.809 |
| merges (total) | 1 | 1 |
| splits (total) | 1 | 2 |
| SAM instances with no GT match (total) | 14 | - |
| GT items with no SAM match (total) | 17 | - |

# SAM 3 real votes vs pseudo votes (generic 'food', 4 scans)

4 scans.

| scan | items | wrong real | wrong pseudo | share real | share pseudo | mIoU real | mIoU pseudo | merges r/p | splits r/p | unvoted real |
|---|---|---|---|---|---|---|---|---|---|---|
| s000000 | 4 | 0.501 | 0.020 | 0.988 | 0.953 | 0.243 | 0.936 | 0/0 | 0/0 | 0.497 |
| s000001 | 5 | 0.110 | 0.065 | 0.978 | 0.908 | 0.468 | 0.733 | 1/1 | 0/2 | 0.029 |
| s000002 | 8 | 0.550 | 0.081 | 0.964 | 0.911 | 0.094 | 0.822 | 1/0 | 1/0 | 0.003 |
| s000011 | 3 | 0.514 | 0.173 | 0.943 | 0.776 | 0.324 | 0.746 | 1/0 | 1/0 | 0.001 |

| mean | real SAM 3 | pseudo |
|---|---|---|
| food points with lift_label != label | 0.419 | 0.085 |
| food points with no vote | 0.133 | 0.012 |
| mean lift_share (voted food points) | 0.968 | 0.887 |
| per-scan instance mIoU | 0.282 | 0.809 |
| merges (total) | 3 | 1 |
| splits (total) | 2 | 2 |
| SAM instances with no GT match (total) | 2 | - |
| GT items with no SAM match (total) | 16 | - |

## Notes
- Prompts: names = item name + category (deduped, lowercase) + aux plate/bowl/cup/fork/knife/spoon/napkin/phone; generic = "food" + the same aux. SAM 3 video predictor, fp16 Turing patches, frames at full 1920x1440, label maps stored at 960x720, thresh 0.5, reprompt_every 30 (so a single window for 24 frames).
- Time: 235-282 s/scan (3-4 food concepts + 8 aux, ~20 s per concept propagation; aux prompts are ~160 s of that); generic 201 s. OOM (4.4 GB peak) on s000003 (14 items) and s000018 (skipped); s000004-6, 8, 10, 15, 19 not run (budget).
- SAM ids -> GT ids: Hungarian on pooled 2-D IoU against gt/seg_ids (min 0.1), unmatched SAM instances get fresh ids K+1.. . Remapped maps in <scan>/masks_sam3_gt*, copied as masks_sam3 into the _sam3 data dirs.
- "wrong" counts food points (label 1..251) with lift_label != label, including points with no vote (lift_label 0). merges/splits: an id holding >= 20 % of two GT items / a GT item with >= 20 % in two ids (items with >= 200 points).
- Real SAM 3 recall on the synthetic renders is the dominant failure: on s000000 only the "cucumber" concept produced masks; lettuce, tuna salad, yeast bread (named by the oracle) returned nothing at thresh 0.5.


---

# v2 prompting (2026-10-16): generic "food" + simplified names, video vs per-frame image

Code: `pipeline/foodscan/segment.py` (`--mode video|image`, `refine_frame`, `link_instances`), runner `pipeline/eval/sam3_synth_votes.py`
(variants `v2_video` -> `masks_sam3_v2`, `v2_image` -> `masks_sam3_v2i`; commands `run`, `relink` (CPU, re-tunes the 3D linking from the saved `raw/` maps), `match`, `copy`, `export`, `stats`).
Data: `data/synth/hard_v12_sam3_v2_image/scans` (11 scans), `data/synth/hard_v12_sam3_v2/scans` (video, 6 scans: s000000/01/02/11/12/13); earlier copies untouched.
Raw tables: `v2_image_stats.md` (11 scans), `v2_video_stats_6scans.md`, probe rows `v2_probe.jsonl`.

## What changed
* Primary concept "food" at 0.3; the oracle item/category names are simplified to 1-2 words (SIMPLE table from sam3_gap, else last two words), deduplicated, capped at 12 per scan, run as a secondary pass at 0.2 (0.3 for vague one-word names: meat, fruit, vegetable, sauce, salad, grain, dip, fish); "dish"/"meal"/"food" are never used as names.
* `refine_frame` per frame: a name mask n carves n&f out of a "food" instance f (own instance, named) when IoU(n, f) < 0.7, n is mostly inside f (>= 50 %), n&f > 5 % of f and n does not cover all of f; the remainder stays "food" (a remainder <= 5 % goes to the carved piece). Otherwise instances are named by the name whose masks cover >= 50 % of them; food-only instances are kept as "food"; a name mask >= 0.5 that covers no instance becomes a named instance.
* Image mode: SAM 3 image model on every frame (1008 px), per-frame-unique ids (uint16), then `link_instances`: lift to the fused cloud, `labels.repair_ids` (view-consensus merge; min_support 1 because per-frame ids give each id pair one node pair; ids seen in < 2 frames dropped). Video mode: tracker + window re-association as before.
* OOM: concepts are processed in chunks of 6 with a GPU cache flush between; on a CUDA OOM the predictor is rebuilt and the current pass / frame retried once (`rebuild=`), the runner also rebuilds and skips the scan if it still fails. No OOM occurred in these runs (image peak 2.6 GB, video peak 4.0 GB).
* Aux threshold default lowered 0.5 -> 0.3 after the probe below; the v2 runs here still used 0.5.

## Lift statistics (oracle names simplified; same stats code as above)

| metric | old real (video, oracle names, 11) | new real VIDEO (6 scans) | new real IMAGE (11) | pseudo (11) |
|---|---|---|---|---|
| food points with no vote | 0.548 | 0.061 | 0.004 | 0.025 |
| lift_label != label | 0.609 | 0.289 | 0.246 | 0.103 |
| mean lift_share (voted) | 0.896 | 0.950 | 0.763 | 0.882 |
| per-scan instance mIoU | 0.376 | 0.500 | 0.662 | 0.796 |
| merges / splits (total) | 4 / 1 | 3 / 3 | 9 / 19 | 3 / 4 |
| SAM instances with no GT match | 17 | 16 | 56 | - |
| GT items with no SAM match | 32 | 14 | 8 | - |

Same 6 scans only (s000000/01/02/11/12/13): old real 0.670 wrong / 0.619 unvoted / mIoU 0.346; new video 0.289 / 0.061 / 0.500; new image 0.248 / 0.004 / 0.625; pseudo 0.073 / 0.010 / 0.833.
Image mode on the 6 scans: share 0.750, merges/splits 6/11, unmatched SAM 38, unmatched GT 6; video: share 0.950, 3/3, 16 unmatched SAM, 14 unmatched GT.

Reading: the 55 % unvoted loss is gone (image 0.4 %, video 6 %). Image mode is the better vote source (mIoU 0.66 vs 0.50 on the same scans, fewer missing items), at the price of more fragments: 19 splits and lift_share 0.76 (per-frame instances are linked by 3D overlap, so an item that SAM cuts differently in different frames stays as 2-3 ids). Video keeps sharper ids (share 0.95) but loses whole items: on s000002 "food" at 0.3 in the tracker returned 4 blobs for 8 items (mIoU 0.10) and on s000011 no name mask fired (0 of 3 instances named). Cost: image ~50 s/scan, video ~270 s/scan (13 concepts x ~20 s).
`relink` with overlap 0.3 instead of 0.5 changed nothing, so the splits are segmentation fragments, not linker thresholds.

## What is still missed (image mode, per-item lift recall = points that get their own id)
kind=scan 0.669 (34 items), asset 0.681 (4), procedural 0.668 (14); source primitive 0.81, procedural_leaves 0.62, procedural_piece 0.62, topping 0.32 (1 item).
Items with recall < 0.5: broccoli 0.00 / 0.49 (of 12 broccoli items), cucumber 0.00 / 0.39 (of 3), sushi 0.05 / 0.33 / 0.48 (of 6), grapes 0.14, ice cream 0.32, rice 0.39 (scan); lettuce 0.45, boiled egg 0.39, yogurt 0.32, gravy 0.32 (procedural). The video path additionally loses the sushi items (all 5 at 0.00), chicken thighs, hummus, lettuce. Cucumber/broccoli failures are the many-small-pieces case (merged into one blob or cut into slices that are linked to a neighbour); sushi/gravy/yogurt/rice are the low-recognisability procedural and primitive items found in sam3_gap.

## Aux container / clutter prompts on synthetic frames (264 frames of the image run, raw detections >= 0.15 vs GT seg_ids)
Detections per frame at 0.15 / 0.3 / 0.5 and the fraction of the mask that lies on GT container (254) / clutter (253) / food:
plate 1.56 / 1.10 / 1.03 (86 % on container at score >= 0.5, 64 % below), bowl 0.41 / 0.26 / 0.22 (88 % / 80 % on container), cup 0.15 / 0.12 / 0.09 (100 % lands on GT clutter, not container, i.e. the generator labels cups as clutter), knife 0.49 / 0.33 / 0.27 (95 % clutter), spoon 0.25 / 0.14 / 0.14 (95 %), fork 0.13 / 0.07 / 0.07 (89 %), napkin 0.38 / 0.16 / 0.09 (91 % at >= 0.5 but 61 % below), phone 0.12 / 0.09 / 0.06 (100 %).
Contamination of food is ~0: 0.000 at >= 0.5, 0.001 at 0.3-0.5, 0.012 for containers at 0.15-0.3. Score bands 0.3-0.5 add 37 container and 44 clutter detections (vs 355 / 166 above 0.5) with >= 57 % / 89 % own-class overlap, so 0.3 is safe and recovers about 16 % more aux masks; 0.15-0.3 is too noisy for containers (64 % own-class). The default is now 0.3; napkin is the only prompt with mostly low scores (mean 0.36).
Generic "food" at 0.3 returns 7.2 masks per frame (12.5 at 0.15, 4.2 at 0.5); 98 % of the mask area at >= 0.5 and 90 % at 0.15-0.5 lies on GT food. Name prompts that fire are precise (>= 0.95 on food) but rare: bread 0.45 dets/frame at 0.3, broccoli 1.5, cucumber 0.7, grapes 0.5, sushi 0.00, lettuce 0.00, rice 0.02, chicken 0.31, egg 0.08.

## Cost
GPU (under flock, excluding lock waits): image 57 s smoke test + 586 s for 11 scans, video 1649 s for 6 scans (budget stop), about 2300 s in total (38 min) of the 45 min allowance.
