# Labelling real scans for segmentation (`tools/label_items.py`)

Goal: a few clicks per item so a real iPad scan can be scored for *segmentation* (instance mIoU, PQ, boundary IoU, item count), not just mass.
Budget: **5-10 min per scan** (about 1 min per item, plus 2 min to inspect and fix).

## Run (from `pipeline/`, env `foodseg`)
```
python -m tools.label_items label SCAN --dish "white rice, grilled chicken, broccoli" --annotator you --export-frames
python -m tools.label_items score SCAN [--pred refined.npz]
```
`SCAN` is the `run_meal.py` output dir (needs `lift.npz` with `xyz`; colours come from `fused*_points.ply`; without `lift.npz`
the cloud is pre-clustered with DBSCAN). `--grow geodesic` grows by kNN connectivity (colour dE<=20, normal angle<=50 deg, radius cap
`--radius 0.12` m) instead of taking whole lift instances. Results: `SCAN/labels_manual.npz` (+ `.json`), `SCAN/masks_manual/*.png`
(2D label PNGs per frame; use `--scale 1 --fill-px 4` for full-res).

## Loop (open3d `VisualizerWithEditing`; one window per round)
1. Window opens, cloud coloured by lift instance (pastel), your items saturated, container dark blue.
2. **Shift + left-click** a seed on each item (one point is enough for lift growing; shift+right-click un-picks; drag rotates, wheel zooms). Press **Q** (or close the window).
3. At the `cmd>` prompt type:
   - `rice` (any text) = new item from all picked seeds; a number picks that name from the printed candidate list (from `--dish` / `masks/instances.json`)
   - `split rice sauce` = one item per name, picks consumed in order (`split rice*2 sauce` = first two picks to rice). Use when one lift instance holds two foods: its points go to the nearest seed by geodesic distance.
   - merge: pick seeds in **two lift instances** and name once; they become one item.
   - `con` / `clu` / `bg` = mark the picked instance(s) container / clutter / background
   - `u` undo, `l` list, `g geodesic|lift` switch growing, `w` save, `q` save and quit, empty line = reopen viewer
4. Window reopens with the new colours. Repeat until every food is saturated, then `q`.

Ids are 1..N in labelling order, 0 background, 253 clutter, 254 container.

## Headless
`python -m tools.label_items label SCAN --seeds seeds.json --out-dir DIR` with
`{"grow":"lift","items":[{"name":"rice","seeds":[10,99]}],"marks":[{"cls":"container","seeds":[7]}]}` (point indices).

## Strata to capture (aim for a few scans of each)
touching items (rice against curry, stacked food); rice / mash / porridge (no visible edge, height-only cues); sauces and soups
(thin, spread, mixed into others); small items (nuts, herbs, <5 mm pieces; label as one item); glossy / wet foods (depth holes, specular colour).
Record the stratum in `--annotator` notes or the scan folder name.

## When a lift instance merges two foods
Pick one seed in each food and use `split a b`. If the boundary is poor (the halves are interleaved), switch with `g geodesic` and
re-seed with several seeds per food on each side of the boundary (`split a*3 b*3`). When one food is split over two lift ids, pick
seeds in both and name once (merge). Leftover wrong pieces: `bg` or `clu` on a seed in them.

## Limits
Needs a display; picking is by point (not box/lasso); the viewer cannot show names, so use `l` at the prompt. Camera is kept between rounds.
