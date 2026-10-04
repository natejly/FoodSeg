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
