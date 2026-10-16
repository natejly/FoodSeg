# MetaFood3D: volume from views, density floor, touching-food 3D segmentation

Run 2026-10-11/12. Data: MetaFood3D (CC BY-NC) 3D meshes (streamed from the tarball,
never extracted), Point_cloud/4096 and the nutrition sheet (weight, kcal, macros,
volume), in `data/metafood3d` / `~/Downloads`. 637 objects, 108 categories.
The RGBD_videos "depth" frames are colourised 8-bit JPEGs with no intrinsics, so they
are not usable as metric depth; everything below uses the meshes / point clouds.

## 1. Volume vs number of views (`pipeline/eval/mf3d_views.py`, `views/`)

Each real mesh is set on a table and ray-cast from virtual depth cameras 35 cm away
(320x240, 60 deg FoV, 1 mm depth noise, perfect food mask). Plane closing = top-surface
height map down to the table (what one depth frame allows); space carving = 2 mm voxels
removed wherever any camera sees past them. Mass / kcal = volume x per-category density
and kcal/g, leave-one-out, category names known. Objects >= 20 mL (548; `summary.json`):

| views | estimator | volume median APE | median bias | mass MAPE | kcal MAPE |
|---|---|---|---|---|---|
| (sheet volume) | ground truth | 0 | 0 | **24.0** | 29.0 |
| 1 overhead | plane closing | 40.0 | +40.0 | 31.6 | 36.0 |
| 1 at 45 deg | plane closing | 25.1 | +24.8 | 32.4 | 36.6 |
| 16 + overhead | plane closing | 61.3 | +61.3 | 31.1 | 35.5 |
| 2 + overhead | space carving | 27.6 | +27.5 | 27.0 | 30.4 |
| 4 + overhead | space carving | 19.3 | +19.1 | 25.5 | 29.0 |
| 8 + overhead | space carving | 13.6 | +12.7 | 25.2 | 28.8 |
| 16 + overhead | space carving | **9.0** | +8.1 | **25.1** | **28.7** |

- Plane closing counts every column under the visible top as solid: +40 % even for one
  ideal overhead frame (a sphere would read 1.5x), and more views make it worse.
- Carving uses the extra views: volume error falls monotonically with views, 40 -> 9 %.
- With 16 views mass error is within 1.1 pt of the perfect-volume floor: what remains is
  density variation inside a food category, not geometry.
- `summary_all_sizes.json` includes tiny items (single almonds, < 20 mL), where 2 mm voxels
  and 1 mm noise dominate. `views_planeonly_v1/` is the first run (plane closing only).

## 2. Point-cloud estimators on the sampled clouds (`pipeline/eval/metafood3d.py`, `summary.json`)

4096-point clouds are too sparse for closed-surface volume (hull +70 % bias, Poisson
+22 %); kept for the record, superseded by section 1.

## 3. Touching foods: 3D instance segmentation (`pipeline/eval/mf3d_composite.py`, `mf3d_seg_train.py`, `seg/`)

Scenes of 2-5 real scanned objects packed on a plane (gaps -5..+20 mm, i.e. many
touching), objects split 80/20 so test objects are unseen; 4000 train / 500 test scenes;
4096 points per scene; full scan vs top-view-only points. Geometry only (no colour).

| view | method | mIoU | recall @ IoU 0.5 | per-item volume MAPE | exact count |
|---|---|---|---|---|---|
| full | DBSCAN (eps 8 mm, best on train) | 0.63 | 65 % | 99.7 % | 12 % |
| full | DGCNN offset voting + DBSCAN | **0.95** | **97 %** | **9.8 %** | 53 % |
| top | DBSCAN (eps 8 mm) | 0.68 | 73 % | 58.6 % | 3 % |
| top | DGCNN offset voting + DBSCAN | **0.95** | **98 %** | **8.6 %** | 48 % |

Per-item volume here = convex hull of the predicted vs ground-truth instance points, so it
isolates segmentation error. DBSCAN either merges touching items or fragments them; the
learned segmenter separates them from either view. Low exact-count = small stray clusters
(min_samples 8); a size filter would fix most of it.

## 4. Single-image benchmark (MFP3D protocol) (`pipeline/eval/mf3d_train.py`, `img/`)

MFP3D's released h5 files (github.com/jingema99/MFP3D) fix the split: 401 train / 95 test
objects, one 224x224 still photo each, targets volume / energy / mass / macros. Test = those
95 photos exactly; MAPE = mean over objects of |pred - gt| / gt. 3 train objects with negative
volume / missing weight dropped. ConvNeXt fine-tuned on RGB (log-L1, flips + rotation,
4-flip TTA); 40 train objects held out for epoch selection.

| run | input | val | volume MAPE / MAE | energy MAPE / MAE |
|---|---|---|---|---|
| MFP3D (published) | RGB + 3D point cloud | | 41.4 % / 62.6 mL | 68.1 % / 78.0 kcal |
| s_rgb | ConvNeXt-S, h5 photos + video frames | 52.8 | 39.6 % / 62.6 | 63.0 % / 98.4 |
| s_rgb_h5only | ConvNeXt-S, h5 photos only | 47.7 | 30.5 % / 61.5 | 52.6 % / 91.1 |
| s_aruco | + ArUco-board scale features | 50.0 | 31.2 % / 58.0 | 50.2 % / 85.4 |
| b_h5only | ConvNeXt-B | 43.5 | 31.4 % / 55.1 | 60.3 % / 98.1 |
| b_aruco | ConvNeXt-B + ArUco | 50.0 | 32.4 % / 59.3 | 56.6 % / 82.3 |
| **mean of the four h5 runs** | | | **28.6 % / 55.1 mL** | **50.3 % / 81.5 kcal** |

Beats MFP3D on volume (both metrics) and energy MAPE; energy MAE still 3.5 kcal worse.
PortionNet reports 17.4 % / 15.4 % but its protocol is unclear (12,752 images, i.e. ~20 per
object, while "following MFP3D's 80/20 split"); a by-image split would put test objects in
training. Video frames (a different capture source than the h5 stills) hurt; ArUco scale gave
no reliable val gain (40 val objects are noisy).
