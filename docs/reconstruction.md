# 3D reconstruction: what works for our scans

## Result on scan 20261002_210342 (Cutwater 4-pack box, 162 frames / 10.8 s, dim light)

| Method | Output | Box size (cm) | Notes |
|---|---|---|---|
| LiDAR TSDF (`foodscan.fuse`, 2 mm, all frames, smoothed depth, conf≥1) | `results/recon/lidar/` | 12.1 × 16.3 × 17.2 | Soft, blurry edges; TSDF fattening |
| **Apple Object Capture, `full`** (RGB + LiDAR depth for scale) | `results/recon/full/` | **13.0 × 14.2 × 14.6** (after ARKit alignment) | Sharp, legible texture; 160/162 frames registered |
| Object Capture `raw` | `results/recon/raw/` | 13.2 × 14.3 × 14.5 | Same ~30k tris as full: detail is input-limited |

Object Capture → ARKit world alignment (`scripts/objcap/align.py`, Umeyama on camera centres):
scale 0.966 (Object Capture is ~3 % large), camera residual 3 mm median / 7.7 mm max.
The aligned mesh's bottom sits at y = −0.243 m vs the TSDF desk plane at −0.242 m.
The underside is open (never seen); close it against the fitted table plane for volume.

**Use `results/recon/full/oc_full_arkit.obj`** (metric, ARKit world, +Y up, same frame as
`frames.jsonl` poses, so masks can be lifted directly).

```bash
swiftc -O scripts/objcap/main.swift -o scripts/objcap/objcap
swiftc -O scripts/objcap/render.swift -o scripts/objcap/render
scripts/objcap/objcap SCAN results/recon/full/oc_full.usdz full mask      # ~3 min on M2 Pro
scripts/objcap/render results/recon/full/oc_full.usdz results/recon/full/view results/recon/full/oc_full.obj
pipeline/.venv/bin/python scripts/objcap/align.py SCAN results/recon/full/oc_full.obj \
    results/recon/full/oc_full.poses.json results/recon/full/oc_full_arkit.obj
```

## Why the LiDAR TSDF was bad (diagnosis, 2026-10-13)

Measured on `results/recon/lidar/lidar_crop.ply` against the aligned Object Capture mesh:

- **Sensor is fine where it looks head-on:** desk plane 1.3 mm RMS; box top face 1.5 mm median.
- **Silhouettes smear:** ~3.6k points float 1–4 cm outside the box near its top edge. ARKit
  sceneDepth is densified from a sparse LiDAR dot grid and ramps across depth jumps. This is
  what inflated the TSDF box to 16–17 cm against 14.6 cm.
- **Sides are grazing:** vertical faces are 4–6 mm off, because the scan covered only 203° of
  azimuth in 10.8 s, so half the sides were never seen head-on.
- **Motion blur:** median camera speed 17.6 cm/s (p90 31) in dim light, so RGB frames were blurred.

Fixes: `fuse` now drops the background side of silhouettes (`--edge-rel`, on) and blurred
frames (`--min-sharpness`, on); a grazing-angle filter (`--max-incidence`) exists but is off.
On the synthetic box with LiDAR-like edge smear (`tests/make_synthetic_scan.py --edge-smear 3`)
stray points > 5 mm go from 472 to 0 and the footprint from 10.3×7.8 to 6.5×4.5 cm (true 6×4).
The app's live guide and `foodscan.scan_report` target the technique half.

Food sim (`tests/sim_scan_quality.py`, 5 rendered scenes / 8 items from synth v1, container
support, item MAPE):

| Condition | no filters | edge | edge + 75° incidence | defaults |
|---|---|---|---|---|
| mild smear (synth.lidar as rendered), full orbit | 22.6 % | 22.7 % | 22.8 % | 22.8 % |
| wide smear (7×7 within 3 px of jumps, high conf), full orbit | 64.2 % | 27.3 % | 25.8 % | 27.2 % |
| wide smear, 200° arc only | 50.5 % | 41.9 % | 43.5 % | 42.9 % |

- The edge filter does nearly all the work. It must keep the foreground rim: an earlier
  version that cleared both sides eroded small items (9 ml cheese cube −21 → −48 %).
- Incidence filtering erodes small items (cheese cube −21 → −43 % under mild smear), so it is off.
- The sharpness filter is untested here: renders have no motion blur.
- No filter rescues a half orbit, and small items vanish: coverage has to come from capture.
- Caveats: 24 frames per sim scan (real ≈ 160), 8 items, and the wide-smear model is a guess
  until it is fitted to real calibration scans.

## Literature (summary of a web survey, 2026-10)

- **Best pipeline (GPU):** blur/duplicate frame filtering → COLMAP/GLOMAP poses Sim(3)-aligned to
  ARKit (don't seed BA with ARKit poses) → PromptDA LiDAR-prompted depth → PGSR or 2DGS with depth
  supervision → masked mesh closed against the plate plane. VolE (same structure) reports 2.2 %
  MAPE ([arXiv 2505.10205](https://arxiv.org/html/2505.10205v1)); MetaFood'24 winner VolETA 10.97 %
  ([report](https://arxiv.org/html/2407.09285v1)). DTU Chamfer: PGSR 0.47 mm, GOF 0.74, 2DGS 0.80.
- **Object Capture** with custom samples uses depth only for scale and gravity only for
  orientation; no API to inject ARKit poses ([forum](https://developer.apple.com/forums/thread/840946)).
  ~1.5–1.9 mm dimensional error in a PLoS ONE 2024 study ([PMC11637407](https://pmc.ncbi.nlm.nih.gov/articles/PMC11637407/)).
- **Raw LiDAR TSDF** is ~±1 cm on small objects ([PMC8593014](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593014/)): not volume-grade.
- **Feed-forward (VGGT, MASt3R, MapAnything):** good for poses/sanity checks, not volume-grade alone.

## Capture protocol (biggest lever)

- 45–90 s per meal (700–1300 frames), keep the sharpest 150–250.
- Three 360° orbits at ~30°, 50°, 75° elevation + a top-down pass, ≥70 % overlap.
- 35–50 cm away, plate ~60 % of frame; bright diffuse light (short exposure → no blur); move slowly.
- Textured placemat with an ArUco/checkerboard under the plate for an independent scale check.
- Avoid glossy plates and glare.
