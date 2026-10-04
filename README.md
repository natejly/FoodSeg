# 3D Food Segmentation — thesis code

Two parts:

- `capture-app/` — iPad/iPhone (LiDAR) app that records RGB + depth + confidence + camera poses.
- `pipeline/` — Python: load a scan, TSDF-fuse it into a metric mesh, compare against single RGB-D frames.

## Capture app

### One-time setup

1. Install **Xcode** from the Mac App Store (~12 GB). Open it once so it installs its components.
2. On the iPad: Settings > Privacy & Security > **Developer Mode** on (iPadOS 16+), then restart.
3. Generate and open the project:
   ```bash
   cd capture-app
   xcodegen generate          # already done once; rerun after editing project.yml
   open FoodScan.xcodeproj
   ```
4. In Xcode: select the **FoodScan** target > *Signing & Capabilities* > pick your Team
   (add your Apple ID under Xcode > Settings > Accounts if it isn't listed). A free
   Apple ID works; the installed app expires after 7 days and you re-run from Xcode.
5. Plug in the iPad, pick it as the run destination, press ⌘R. First time: on the iPad,
   Settings > General > VPN & Device Management > trust your developer certificate.

### Using it

- Point at the plate, wait for tracking to read `normal`, type a meal name, press record.
- Orbit the plate slowly for 20–40 s at ~25–35 cm, covering all sides and a top-down pass.
  LiDAR minimum range is ~20 cm; closer than that depth drops out.
- Stop. The scan is saved to *Files > On My iPad > FoodScan > scans/<timestamp>_<name>/*.
- Get it to the Mac: Finder (iPad plugged in > Files tab > FoodScan, drag out) or the
  share button in the app's scan list (zips + AirDrop).

At 15 fps a 30 s scan is ~450 frames, roughly 250 MB (mostly JPEG).

### Scan format

```
meta.json               device, format notes
frames.jsonl            one JSON per frame: index, timestamp, trackingState,
                        intrinsics (3x3, full-res RGB), transform (4x4 camera-to-world,
                        ARKit convention), image/depth sizes, exposure
rgb/000000.jpg          1920x1440 color (sensor/landscape orientation)
depth/000000.f32        256x192 float32 LE meters, row-major (ARKit sceneDepth)
smooth_depth/000000.f32 same, ARKit smoothedSceneDepth
conf/000000.u8          256x192 uint8: 0 low, 1 medium, 2 high
```

ARKit cameras are +Y up / −Z forward; `pipeline/foodscan/loader.py` converts to OpenCV
(+Y down / +Z forward) for Open3D. Depth is pre-registered to the color image, so
depth intrinsics = RGB intrinsics × (256/1920, 192/1440).

## Pipeline

```bash
cd pipeline
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # already done
source .venv/bin/activate

# fuse a scan into a mesh
python -m foodscan.fuse /path/to/scan --voxel 0.003 --stride 2

# view the mesh, with frame 40's single RGB-D cloud overlaid in red
python -m foodscan.view /path/to/scan --frame 40

# synthetic test scan (6x4x3 cm box on a table, 72 cm^3) for testing without the iPad
python tests/make_synthetic_scan.py /tmp/synth && python -m foodscan.fuse /tmp/synth --stride 1
```

Key API: `Scan(dir).good_frames()`, `Frame.depth() / confidence() / rgb()`,
`Frame.K_depth`, `Frame.T_cw` / `T_wc`, `fuse(scan)`, `single_frame_cloud(frame)`.

On Bouchet use `--device CUDA:0` after installing the CUDA Open3D wheel.

### Modules

| Module | Does | Validated |
|---|---|---|
| `foodscan/loader.py` | Scan/Frame access, ARKit→OpenCV poses, confidence masking | synthetic |
| `foodscan/fuse.py` | TSDF fusion (VoxelBlockGrid, default 1.5 mm), single-frame cloud | synthetic box: table at z≈0, top at 30.3 mm |
| `foodscan/volume.py` | RANSAC plane, DBSCAN object split, mesh-prism / height-map / Poisson / hull volumes | synthetic box 72 cm³: mesh-prism +2.5 % (1.5 mm), +1.3 % (1 mm) |
| `foodscan/labels.py` | Lift per-frame masks into the cloud by visibility-checked voting; vote margin as confidence | synthetic masks, 8 frames: 99.9 % point accuracy; with 20 % corrupted pixels 98.3 %, and the 21 % low-margin points are exactly the wrong ones |
| `foodscan/container.py` | Empty-container template (scan each plate/bowl once) → occlusion-aware ICP under the food → height field → food volume relative to the container surface | synthetic bowl, 12 mm of food below the rim (40.7 cm³): template method **−1.4 %**, naive above-plane **0** (−100 %) |
| `foodscan/view.py` | Open3D viewer | — |
| `foodscan/nutrition.py` | FNDDS lookup via FoodData Central (cached, ranked), density table, volume→mass→kcal/macros | live on 2 queries with DEMO_KEY; needs a real key (`FDC_API_KEY`) |
| `foodscan/calibrate.py` | Calibration report: every estimator + single-frame baseline vs true volumes, MAPE/bias | synthetic |
| `foodscan/segment.py` | SAM 3 / 3.1 text-prompted video masks → `masks/*.png` + `instances.json` | **untested** (needs GPU + gated weights); written to the verified request API |
| `foodscan/depth_upsample.py` | PromptDA LiDAR→RGB-res metric depth, KNN hole fill → `depth_hr/` | **untested** (needs GPU) |
| `foodscan/refine_poses.py` | pycolmap: SIFT + sequential match, triangulate with poses fixed, BA poses only, re-align → `poses_refined.json` | **untested** (synthetic scan has no texture); written for 3.12, cluster has 4.2 |
| `foodscan/naming.py` | VLM item naming from 2 views (Gemini, or any OpenAI-compatible endpoint e.g. vLLM Qwen3-VL); strict JSON | **untested** (needs API key) |
| `foodscan/run_meal.py` | End to end: fuse → plane or container template → masks/clusters → per-item volume → nutrition → JSON | synthetic box 76.0 (+5.5 %, stride 2), synthetic bowl 40.1 (−1.5 %) |

```bash
# a full meal, plate on a table, SAM 3 masks already in SCAN/masks
python -m foodscan.run_meal data/scans/m001_orbit --out results/m001.json
# food in a pre-scanned bowl
python -m foodscan.run_meal data/scans/m002_orbit --container data/containers/bowl_A.npz
# build a container template once
python -c "from foodscan.container import ContainerTemplate as C; C.from_scan('data/scans/cal_bowlA_empty_01', name='bowl_A').save('data/containers/bowl_A.npz')"
```

`Scan(dir, poses="refined", depth="depth_hr")` makes fusion use the refined poses / HR depth (geometry config B).

Tests: `tests/test_volume_synthetic.py`, `tests/test_labels_synthetic.py`, `tests/test_container_synthetic.py`
(`make_synthetic_scan.py --scene box|bowl --fill 0.012 --noise 0.0005` builds the inputs).

Two conventions that bit during testing, both handled in code: TSDF fattens convex walls by ~1–2 voxels
(use ≤1.5 mm voxels for volume; mesh-prism, not grid height maps), and ARKit/COLMAP put pixel centres
at +0.5 while Open3D uses integer indices (`K_depth` is shifted by half a pixel; `K_rgb` is not).

### Cluster

```bash
scripts/bouchet/sync.sh                     # push code + data
ssh bouchet
cd /nfs/roberts/project/cpsc4900/cpsc4900_njl36
sbatch scripts/bouchet/download_datasets.sbatch                       # datasets + weights
N=$(ls -d data/scans/*/ | wc -l); sbatch --array=0-$((N-1))%8 scripts/bouchet/fuse_scans.sbatch
sbatch --gpus=l40s:1 scripts/bouchet/gpu_job.sbatch python -m foodscan.segment data/scans/m001_orbit --names "white rice,grilled chicken"
```
Never run conda/pip/python work on the login node; it gets killed. Use `salloc`/`sbatch`.

## Next pieces (not written yet)

- `eval/` — per-item MAPE tables across meals, Bland–Altman, RQ1 comparison table.
- `naming.py` — VLM (Gemini / Qwen3-VL) item naming from 2 crops → names for `segment.py`.
- Bowl / curved-plate support surface (see docs when the research lands).

## Docs

- `docs/model_survey.md` — SOTA per stage, test matrix, SFT plans, compute budget
- `docs/segmentation_robustness.md` — corruption/prompt/video robustness chapter plan
- `docs/synthetic_data.md` — BlenderProc + MetaFood3D rendering recipe, LiDAR noise model
- `docs/collection_protocol.md` — how to collect calibration and meal scans
# FoodSeg
