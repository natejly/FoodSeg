# seg3d - PTv3 (Sonata init) refiner for lifted food-scan masks

Fine-tunes Point Transformer V3, initialised from Sonata self-supervised weights, on synthetic fused iPad-style
scans. Per point it predicts semantic class {table/other, container, clutter, food} and a centre-offset vector;
food instances are recovered by DBSCAN on `xyz + offset`. Input includes a "lifted-vote prior" (SAM-style
instance ids lifted to 3D) which the net must refine. The bar to beat is "use lift_label as the prediction"
(reported by every eval as `lift_baseline`).

## Install (what actually worked here, env `foodseg`, torch 2.11.0+cu128, RTX 2060 sm_75)

    export PIP_CONFIG_FILE=/dev/null
    pip install --isolated spconv-cu126 addict einops pytest     # spconv-cu126 2.3.8, cumm-cu126 0.7.11 (runs on the cu128 torch)
    # Sonata: `pip install ./sonata` fails (setup.py needs pkg_resources), so the package is VENDORED
    # unchanged in seg3d/_vendor/sonata (Apache-2.0, LICENSE inside). Weights come from HuggingFace
    # facebook/sonata (sonata.pth, ~430 MB, cached in ~/.cache/sonata/ckpt) on first run.
    # torch_scatter does not build for torch 2.11 -> seg3d/compat.py injects a pure-torch `segment_csr`
    # (torch.segment_reduce). flash-attn is not used (enable_flash=False; sm_75 has none).

No Pointcept needed. Always `import seg3d.compat; compat.setup()` (model.py does) before `import sonata`.

## Run (from `/home/kyle/Desktop/FoodSeg/pipeline`)

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -m seg3d.train --smoke                          # 20 iters + val + lift baseline, results/seg3d/smoke
    python -m seg3d.train --run ptv3_a --epochs 12         # real run (default: 3 frozen epochs, then top stages)
    python -m seg3d.train --run x --set model=dgcnn pretrained=false aug.crop_p=0.3 data.roots='["data/synth/hard_v10"]'
    python -m seg3d.eval --ckpt ../results/seg3d/ptv3_a/best.pt --dump ../results/seg3d/ptv3_a/preds
    python -m seg3d.eval --baseline                        # lift-label baseline only
    python -m pytest tests/test_seg3d_*.py -q

`--config file.yaml` takes any field of `train.Config` (nested `data:` / `aug:` dicts); `--set a.b=v` overrides.
Outputs: `results/seg3d/<run>/{config.json,metrics.json (per epoch),baseline_lift.json,best.pt,last.pt}`.
`best.pt` is selected on **val** instance mIoU only.

## Data contract

`<root>/scans/<id>/gt/seg_points.npz`: xyz, rgb, normal, label (+ optional lift_label int32, lift_share f32,
lift_nvotes i16, boundary u8) and `<id>/gt.json` (items[].id/category/volume_ml/mass_g, table_plane_arkit).
Labels: food ids 1..251 (scene-local), 252 table, 253 clutter, 254 container, 0 table/artefact.
On load scenes are rotated so the gt.json table normal (default ARKit +y) is +z, z=0 at the table plane, xy
centred on the container. Old files (no lift_*) get lift synthesised from `label` with the corruption suite on
every sample (train and val; val uses a fixed seed). New files use the real lift (extra corruption p=0.5 in
train only). `boundary==1` points get loss weight 0.

Features (30): centred coord x10 (3), rgb*2-1 (3), normal (3) [= Sonata's input layout, stem widened 9->30 with
zero init for the new channels], lift one-hot over 16 randomly permuted slots + overflow (17), lift is-container,
is-clutter (2), lift_share (1), lift_nvotes/20 (1). Coordinates are metres x `coord_scale`=10 because Sonata
expects ~2 cm voxels; a 2 mm voxel then looks like 2 cm. Offsets are in the scaled units; eval divides back.
Voxel: 2 mm eval, train jitters 1.5-3 mm. Split: by scan dir, fixed seed, `val_frac` 0.2 (min 1);
`data.split_by=category` holds out whole food categories.
Item filter (`data.item_filter=manifest|points|none`, default manifest): tiny/invisible items (manifest `dropped_items`, else < `item_min_points`=50
fused points) get loss weight 0 and their points leave all metrics; a scan stays usable if >= 1 item is valid (round 3, `ablate_v3.py`, `split_v3.json`).

## Model notes / memory (6 GB shared card)

Sonata checkpoint is encoder-only; we add our own light decoder (per-stage Linear -> unpool -> sum), a raw-feature
skip (so lift reaches the heads even with a frozen encoder), two sparse 3x3x3 convs, then semantic + offset
(+ optional mass) heads. Phases: `none` (encoder frozen, run under no_grad: ~1 GiB peak at 20k voxels,
~0.35 s/iter), `top` (stages 3-4 trainable, ~2.1 GiB peak + Adam states), `all` (not recommended here).
Training steps that OOM are skipped; 5 in a row fall back to the frozen phase. Eval runs fp32 because spconv's
fp16 inference tuner fails on sm_75 (train-mode fp16 autocast is fine). Train caps `data.max_points`=20000
voxels (keeps object voxels, thins table), `patch_size` 1024 (use 512 if tight).

## Metrics (metrics.py)

Semantic mIoU, instance mIoU (Hungarian, unmatched=0), PQ (IoU>0.5), boundary IoU (3 mm band around GT label
changes), exact-count rate, volume APE (convex hull of instance points + flat base at the item's 1st-percentile
height, vs gt.json volume_ml; `vol_ape_gtpts` = the same estimator on the GT points = its floor). All at voxel
level. Hull volume is a weak estimator on surface-only scans (~60% error even on GT points).

## Stubbed / open

- Mass head: implemented (`mass_head=true mass_w>0`) but untested; synthetic gt.json mass_g is null so it is a no-op.
- DGCNN baseline is a re-implementation (eval/mf3d_seg_train.py's model is xyz-only, offsets-only), subsampled to
  4096 pts/scene with nearest-neighbour copy-back; only smoke-tested.
- Metrics are on voxels, not propagated back to full-resolution points.
- Only 12 scans exist (hard, hard_v3): val is 3 scans / ~5 items - numbers are anecdotal until hard_v10 lands.
- lift one-hot slot permutation is per sample; clutter/container lift appear as flags, not slots.
