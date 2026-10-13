# Accuracy studies: iPhone/iPad LiDAR for small objects and mobile-depth food volume

Summary table of quantitative results (each row is sourced in the sections below):

| Study | Object / size | Device, app, mode | Distance / capture | Metric and value |
|---|---|---|---|---|
| Luetzenburg et al. 2021, Sci Rep ([PMC8593014](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593014/)) | 14 rigid boxes, 14x6x2 cm to 50x30x52 cm | iPhone 12 Pro / iPad Pro 2020 LiDAR, 3d Scanner App | Point density 7,225 pts/m² at 25 cm vs 150 pts/m² at 250 cm | Absolute accuracy ±1 cm, precision ±1 cm for side > 10 cm. Precision degrades below 10 cm. Detection limit ~5 cm |
| Łabędź et al. 2022, Sensors ([PMC9657006](https://pmc.ncbi.nlm.nih.gov/articles/PMC9657006/)) | Plaster figurine 99.2x40.8x90 mm | iPad Pro 12.9 LiDAR, Scaniverse vs DSLM + Metashape | Not given | Cloud-to-reference median 1.68 mm (Q3 2.95 mm), 53% of points beyond σ=1.5 mm, 1,254 points. Photogrammetry median 0.16 mm (Q3 0.27 mm) |
| Łabędź et al. 2022 (same) | Bust 678x458x754 mm | same | — | iPad median 3.47 mm (Q3 6.13 mm). Photogrammetry median 1.50 mm (Q3 2.52 mm) |
| Teppati Losè et al. 2022, Remote Sens. ([PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)) | Heritage monuments/statues (metre-scale) | iPad Pro / iPhone 12 Pro LiDAR, SiteScape, EveryPoint, 3D Scanner App | ~1 m typical; 3D Scanner App min distance setting 0.3 m | C2C vs TLS: SiteScape 81.9% of points in 0–2 cm. Conclusion: "precision of few centimetres" for small/medium objects |
| Vogt et al. 2021, Technologies (via [Díaz-Vilariño et al. 2022](https://isprs-archives.copernicus.org/articles/XLIII-B4-2022/303/2022/isprs-archives-XLIII-B4-2022-303-2022.pdf)) | Lego bricks (~cm) | iPad Pro LiDAR + TrueDepth vs industrial scanner | 300 mm, 65° | LiDAR "impractical" for Lego-size objects. Average deviation 0.16 mm reported (sensor attribution unclear, see Gaps) |
| Herzig et al. 2020, JMIR mHealth ([PMC7142738](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7142738/)) | 48 meals, 128 items | iPhone X TrueDepth (structured light), not LiDAR | Single-capture app; viewing angle had no effect | Weight MAE 35.1 g (relative 14.0%). Energy 41 kcal (12.7%) |
| Fujita & Yanai 2025, LiDARCalorieCam ([slides](http://mm.cs.uec.ac.jp/pub/conf25/251111fujita_3_ppt.pdf), [chapter](https://link.springer.com/chapter/10.1007/978-981-95-4398-4_10)) | 5 Japanese foods, 53–354 cm³ | iPhone LiDAR depth map + RGB, DeepLabV3 mask, convex hull | Single view, no reference object | Volume-table MAPE: fried chicken 64.5% (53 cm³), skewers 39.5% (84 cm³), croquette 31.0% (163 cm³), toast 13.9% (354 cm³), hot dog 8.8% (347 cm³). Calorie MAPE 5.6–11.5% |
| VolE (Haroon et al. 2025, [arXiv 2505.10205](https://arxiv.org/html/2505.10205); Sci Rep 2026 [version](https://www.nature.com/articles/s41598-026-38756-5)) | 21 Foodkit items + MTF 13 scenes | ARKit/ARCore poses + COLMAP + masked MVS, no depth sensor | 360° video, 715–1,209 frames per item | Foodkit MAE 1.22%. MTF MAPE 3.08% in Table III (abstract says 2.22%). VolETA 7.84%, ININ 14.47%, FoodRiddle 10.26% on same subset |
| MetaFood CVPR 2024 challenge ([arXiv 2407.09285](https://arxiv.org/html/2407.09285v1)) | 20 foods; easy 200 / medium 30 / hard 1 image | RGB video + 1.2 cm checkerboard; GT from 3D scanner | — | Winner VolETA MAPE 10.973% (18 scenes). FoodRiddle mean Chamfer 0.0032 m |

## Small-object accuracy: errors for iPhone/iPad LiDAR on objects < 30 cm, and which apps and modes won

### Takeaway
For iPhone/iPad LiDAR alone, about ±1 cm absolute accuracy is the realistic floor for objects with sides above 10 cm. Below about 10 cm, precision drops quickly, and objects around 5 cm are at the detection limit. On a 10 cm figurine, LiDAR (Scaniverse) gave a median error of 1.7 mm with half the points beyond 1.5 mm. Photogrammetry on the same figurine was about 10x better (0.16 mm median). For 5–30 cm plated food, LiDAR-only volume error is therefore dominated by about 1 cm of geometric error, which is a large fraction of a 2–5 cm-tall portion. Photogrammetry or hybrid capture wins for small objects.

### Cited Findings
- iPhone 12 Pro / iPad Pro LiDAR (3d Scanner App) creates "accurate high-resolution models of small objects with a side length > 10 cm with an absolute accuracy of ± 1 cm". The precision error was also 1 cm. The test objects were 14 boxes from 14x6x2 cm to 50x30x52 cm, with 5–7 repeat scans of 5 boxes. — [Luetzenburg et al. 2021, Sci Rep / PMC8593014](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593014/)
- In the same study, precision decreases for surfaces under 10 cm side length, and the detection limit is about 5 cm. Accuracy and precision improve with object size. — [PMC8593014](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593014/)
- In the same study, LiDAR point density on a flat white 20x20 cm square fell from 7,225 points/m² at 25 cm to 150 points/m² at 250 cm. — [PMC8593014](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593014/)
- Łabędź et al. 2022 compared iPad Pro 12.9 LiDAR (Scaniverse) with DSLM photogrammetry (Agisoft Metashape) on a 99.2x40.8x90 mm plaster figurine.
  - iPad: 1,254 points, distance to reference Q1/median/Q3 = 0.75/1.68/2.95 mm, 53.03% of points beyond σ=1.5 mm.
  - Photogrammetry: 3.6M points, 0.08/0.16/0.27 mm, 0.13% beyond σ.
  - The authors call the iPad "unsuitable for imaging fine architectural details and ornaments". Up to 50% of points were misaligned, attributed to sparse sampling and smoothed meshes. — [Łabędź et al. 2022, Sensors 22(21):8504 / PMC9657006](https://pmc.ncbi.nlm.nih.gov/articles/PMC9657006/)
- In the same study, on a 678x458x754 mm bust, iPad error was median 3.47 mm (Q3 6.13 mm) against photogrammetry's 1.50 mm (Q3 2.52 mm). On a calibration-sphere distance of 269.4 mm, the iPad measured 270.81 mm (σ 4.06 mm), a 0.52% deviation. — [PMC9657006](https://pmc.ncbi.nlm.nih.gov/articles/PMC9657006/)
- Vogt et al. 2021 scanned Lego bricks with the iPad Pro LiDAR and TrueDepth at 300 mm and 65°. The iPad's scanning technology was "impractical for scanning small objects such as Lego bricks". The study also looked at the effect of brick colour and concluded the accuracy is insufficient for most industrial uses but may suit lower-requirement applications. — [Díaz-Vilariño et al. 2022 ISPRS summary](https://isprs-archives.copernicus.org/articles/XLIII-B4-2022/303/2022/isprs-archives-XLIII-B4-2022-303-2022.pdf); [Teppati Losè et al. 2022 summary](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf); [TUHH record](https://tore.tuhh.de/handle/11420/9247)
- Teppati Losè et al. 2022 (iPad Pro / iPhone 12 Pro) concluded that the Apple LiDAR with suitable apps can document "small and medium-size objects (e.g., statues, museum objects…) with a precision of few centimetres". The authors also say the iPad Pro and iPhone 12 Pro appear to use the same LiDAR sensor. — [Teppati Losè et al. 2022, Remote Sens. 14:4157](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)
- Teppati Losè et al. 2022, app comparison:
  - SiteScape gave the densest clouds (up to 3.5M pts/m² in 20 s) and the best C2C vs TLS (81.9% of points in 0–2 cm, 15.1% in 2–4 cm). It showed systematic drift along the scan direction.
  - EveryPoint was worst: 33.4% of points beyond 4 cm.
  - 3D Scanner App had normally distributed residuals without drift. Its roughness was a constant 0.2 mm, so the authors suspect built-in filtering that reduces noise but may cut detail.
  - Point density ranges from 200 (3D Scanner App) to 2,000,000 (SiteScape) pts/m² at 1 m.
  - "The crucial role of the software component" is a central conclusion. — [Teppati Losè et al. 2022](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)
- 3D Scanner App's LiDAR settings include a resolution from 5 mm to 20 mm in 1 mm steps and a confidence setting (low/medium/high). Its maximum-distance setting can be reduced from 5 m down to 0.3 m. — [Teppati Losè et al. 2022](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)
- Other close-range results:
  - The iPad Pro (in Spreafico et al. 2021) had RMSE below 2 cm on an outdoor stair, against below 3 mm for a TLS.
  - Murtiyoso et al. 2021 found SSL noise "does not yet allow for high-precision applications" and recommended static acquisitions and lower point-density resolution in difficult scenes. — [Díaz-Vilariño et al. 2022 review section](https://isprs-archives.copernicus.org/articles/XLIII-B4-2022/303/2022/isprs-archives-XLIII-B4-2022-303-2022.pdf)
- A 2026 comparison of iPhone 12 Pro and 15 Pro reported overall RMSE of about 2.05–2.06 cm against known shapes and a C2C RMSE of 0.90 cm between the phones. The 15 Pro was better on small objects at close range and worse at longer distance. These figures are from a search snippet only; the full text returned 403. — [Remote Sensing Letters 2026, doi 10.1080/2150704X.2026.2720055](https://www.tandfonline.com/doi/full/10.1080/2150704X.2026.2720055)
- Apple Object Capture (Photocatch v1.6) on 11 lithic handaxes had mean caliper-vs-model errors of 0.47 mm (width) and 0.23 mm (length) when scaled by total object length. Scaling by a 2.8 mm stamp gave 1.48 mm and 1.90 mm. Image counts ranged from 64 to 1,189 per model, and 91% of models processed in under 15 min. The study did not test LiDAR or report volume error. — [Hurst et al. 2024, PLoS ONE 19(12):e0314560 / PMC11637407](https://pmc.ncbi.nlm.nih.gov/articles/PMC11637407/)
- Medical (TrueDepth / face) results:
  - iPhone 13 Pro TrueDepth (EM3D app) vs CBCT in 30 patients: mean surface discrepancy 0.387 ± 0.361 mm.
  - iPhone 11 Pro + Bellus3D vs 3dMD: RMS 0.86 ± 0.31 mm, 97% of landmarks within 2 mm.
  - Comparison against Vectra M5: TrueDepth and smartphone photogrammetry had landmark-distance limits of agreement within 2 mm, but volumetric limits above 2 cc. Photogrammetry was favoured for volume. — [PMC11723895 review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11723895/); [open.fau.de](https://open.fau.de/items/f1972ae0-59d8-42d2-a274-44415bed9817/full); [MDPI search listing](https://www.mdpi.com/search?q=TrueDepth+scanner)
- Wound study (smartphone LiDAR app, 28 wounds): Pearson r = 0.995 vs ImageJ for area. The mean LiDAR area was 73.59 cm² vs 112.99 cm² by ruler. Area only, no volume. — [J Clin Med 2023, 12(18):6042](https://www.mdpi.com/2077-0383/12/18/6042)

### Inferences
- Rear LiDAR's raw resolution (sparse ~24x24-ish dToF array upsampled by ARKit to a 256x192 depth map) plus mesh smoothing explains the ~1 cm floor. Error relative to object size gets much worse below 10 cm, which is the size range of individual food items on a plate.
- Across studies, the ranking for small objects is consistent: dense photogrammetry or Object Capture (sub-mm to ~0.5 mm) beats front TrueDepth (sub-mm on faces at ~20–40 cm), which beats rear LiDAR (mm to cm). A food pipeline should use LiDAR for metric scale and coarse geometry, and RGB multi-view for shape.
- App choice changes results as much as hardware does. Apps differ in filtering (3D Scanner App smooths) and pose drift (SiteScape and EveryPoint drift). Building our own capture means controlling the confidence filter, voxel size, and pose refinement ourselves.

### Gaps
- I could not open Vogt et al. 2021 (MDPI 403). The 0.16 mm "average deviation" quoted by Díaz-Vilariño et al. is ambiguous: it is probably TrueDepth or the industrial scanner, since the same sentence says LiDAR gave no usable result.
- I did not verify "Kim 2023" or a study directly comparing Polycam, Scaniverse and Record3D on objects under 30 cm.
- Luetzenburg's per-box accuracy values are only in Supplementary Table S2, which I did not retrieve.

## Food volume with mobile depth or ARKit: MAPE, capture protocol, reconstruction method

### Takeaway
I found no peer-reviewed study that validates iPhone/iPad LiDAR food volume on a standard benchmark.
- The only LiDAR-specific food work (LiDARCalorieCam 2025) uses single-view LiDAR depth with a convex hull. Its volume MAPE runs from about 9–14% for large items (~350 cm³) to 31–65% for small items (53–163 cm³).
- The best reported numbers (2–11% MAPE) come from multi-view RGB reconstruction (VolE, VolETA, MetaFood) with metric scale from ARKit poses or a checkerboard, not from LiDAR depth.
- Structured-light TrueDepth (iPhone X) achieved 14% weight error over 128 items.

### Cited Findings
- Herzig et al. 2020 used an iPhone X with structured-light depth (TrueDepth) on 48 meals and 128 items, with weighed ground truth.
  - Weight MAE was 35.1 g (SD 42.8), relative 14.0% (12.2%). Carbohydrate 5.5 g (14.8%), fat 1.3 g (12.3%), protein 2.4 g (13.0%), energy 41 kcal (12.7%).
  - Viewing angle did not affect accuracy. Cooked meals did slightly worse than breakfasts and snacks.
  - Segmentation needed manual adjustment for 7 of 128 items. Processing averaged 22.9 s. — [Herzig et al. 2020, JMIR mHealth uHealth 8(3):e15294 / PMC7142738](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7142738/)
- LiDARCalorieCam (Fujita & Yanai, UEC, 2025) uses iPhone LiDAR depth and RGB, DeepLabV3 food masks, and a point cloud converted to XYZ with convex-hull volume. It needs no reference object and no fixed angle, and covers 10 categories.
  - Per-item table, "MAPE [%]" with average volume: fried chicken 64.5% (53.2 cm³), croquette 31.0% (163.1 cm³), grilled chicken skewers 39.5% (83.5 cm³), toast 13.9% (354.4 cm³), hot dog 8.8% (347.4 cm³).
  - Calorie MAPE: 7.87% (fried chicken), 6.10% (croquette), 11.50% (hot dog), 5.59% (skewers), 8.64% (toast). DepthCalorieCam had 52.84% and 20.11% on the first two, and the slides claim an 84% MAE improvement.
  - The authors cite low point-cloud density, missing points in regions where foods overlap, and convex-hull overestimation as error sources. — [LiDARCalorieCam poster, UEC](http://mm.cs.uec.ac.jp/pub/conf25/251111fujita_3_ppt.pdf); [Springer chapter](https://link.springer.com/chapter/10.1007/978-981-95-4398-4_10)
- VolE (Haroon, AlMughrabi, Zoumpekas, Marques, Radeva) is "reference- and depth-free".
  - Pipeline: ARKit/ARCore images and camera poses, then COLMAP pose refinement, FoodMem video segmentation, multi-view mask-consistent point filtering, Delaunay/graph-cut/marching-cubes mesh, then divergence-theorem volume.
  - Foodkit: 21 items, 360° video with 715–1,209 frames each, water-displacement ground truth (±5 mL, 5 trials). Mean absolute error 1.22%.
  - MTF (13 easy/medium scenes): Table III MAPE is 3.08% (abstract says 2.22%, an internal inconsistency), vs VolETA 7.84%, FoodRiddle 10.26%, ININ 14.47%. On Chamfer, VolE (0.0044) was not the best; FoodRiddle (0.0028) and ININ (0.0032) were better.
  - The paper does not specify capture distance. It lists transparent and reflective foods as future work. — [arXiv 2505.10205 (HTML)](https://arxiv.org/html/2505.10205); [Sci Rep 2026](https://www.nature.com/articles/s41598-026-38756-5)
- MetaFood CVPR 2024 challenge (arXiv 2407.09285 is the challenge report, not the VolETA paper itself).
  - Setup: 20 foods, each with a checkerboard/pattern mat (1.2 cm squares) for scale and ground truth from a 3D scanner. Easy levels had ~200 video frames (8 items), medium 30 images (7 items), hard 1 top-view image (5 items). 16 teams submitted. Steak and chicken nugget were excluded for data quality.
  - Winner VolETA: about 34.8% of frames kept as keyframes (deduplication plus FFT blur detection), SAM + XMem++ masks, PixSfM (SuperPoint/SuperGlue) poses, NeuS2 mesh, isolated-piece removal (5% diameter), checkerboard scaling. MAPE 10.973% over 18 scenes.
  - FoodRiddle (best mesh): COLMAP + SuperPoint/SuperGlue, 2D Gaussian Splatting with TSDF fusion, outlier removal, bottom vertices projected onto the support plane, Poisson reconstruction. Mean Chamfer 0.0032 m.
  - ININ-VIAUN: COLMAP + NeRF2Mesh/DiffusioNeRF, MeshFix, Laplacian smoothing. — [MetaFood 2024 report](https://arxiv.org/html/2407.09285v1); [UB AIBA winner note](https://www.ub.edu/aiba/?p=733)
- A 2026 review notes that mobile LiDAR spatial resolution is limited, and that earlier depth sensors fail on reflective or transparent food surfaces (soup, metal cutlery) and have low accuracy for small objects. — [Food Portion Estimation: From Pixels to Calories, arXiv 2602.05078](https://arxiv.org/html/2602.05078v1)
- Self-reported portion-size error averages above 20%, the baseline any automated method must beat. — [Two-view 3D reconstruction for food volume, arXiv 1701.03330](https://arxiv.org/pdf/1701.03330)

### Inferences
- LiDARCalorieCam's results show a clear inverse size trend: about 9–14% for ~350 cm³ items and 31–65% for 50–160 cm³ items. This matches the ~1 cm LiDAR geometric floor from the small-object studies. Since a 1 cm error on a 3–4 cm-tall item is about 25–35% of its height, single-view LiDAR is unlikely to beat ~20–30% MAPE on small items without multi-view fusion.
- The best food results (VolE 1–3%, VolETA ~8–11%) all use dense multi-view RGB with a metric scale anchor. A LiDAR-equipped iPad's main contribution is probably metric scale and a support-plane estimate, which replace the checkerboard, rather than the food shape itself.
- Volume method matters: convex hull overestimates concave and overlapping food. Watertight meshing closed against the plate or table plane (as FoodRiddle does with bottom-vertex projection) is the safer choice.

### Gaps
- I did not retrieve Nutrition5k (overhead RealSense depth) volume or mass numbers or NutritionVerse results in this pass, so they are not reported here.
- I found no MetaFood 2025 report numbers from a primary source. A 2026 paper (arXiv 2602.13041) mentions a 0.21 MAPE for a 2025 challenge, which I did not verify.
- I found no study reporting food-volume MAPE for TSDF fusion of the iPad LiDAR depth stream.

## Effects of distance, lighting, surface type and number of views

### Takeaway
- Distance is the dominant LiDAR factor: point density drops roughly 50x from 25 cm to 250 cm.
- Lighting and material had little measured effect on LiDAR on matte building materials, but glossy, transparent and reflective surfaces are repeatedly cited as failure modes, mostly for photogrammetry and generic depth sensors.
- More views help photogrammetry. Re-scanning the same area does not necessarily help LiDAR apps and can add drift.

### Cited Findings
- Point density falls from 7,225 pts/m² at 25 cm to 150 pts/m² at 250 cm. Accuracy and precision improve with object size. — [Luetzenburg et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593014/)
- SiteScape recommends 1–4 m. Point spacing differences grew with distance and lower resolution, reaching up to 2.3 cm at 4 m on the lowest setting. 3D Scanner App allows a minimum of 0.3 m. — [Teppati Losè et al. 2022](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)
- Direct sun vs shade made little difference on pink plaster, and "no specific impact of the surface material" was found across the tested materials (brick and river stone had slightly higher roughness). The authors note that other materials and colours cannot be excluded. — [Teppati Losè et al. 2022](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)
- Re-scanning the same portion raised point counts for SiteScape and EveryPoint but not for 3D Scanner App, and "did not necessarily improve the quality". The authors recommend fast coverage without abrupt movements. Drift accumulated along the scan path for SiteScape and EveryPoint. — [Teppati Losè et al. 2022](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)
- Ambient light limits iPad SSL because tracking relies on RGB. LiDAR shares conventional LiDAR's limitation on reflective surfaces. Recommendations from Murtiyoso et al. 2021 include static acquisition and lower density in difficult scenes. — [Díaz-Vilariño et al. 2022](https://isprs-archives.copernicus.org/articles/XLIII-B4-2022/303/2022/isprs-archives-XLIII-B4-2022-303-2022.pdf)
- Vogt et al. specifically studied the effect of brick colour on iPad LiDAR and TrueDepth scans. — [Teppati Losè et al. 2022 summary of Vogt](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)
- Uneven lighting and reflections degraded photogrammetry on the small plaster sculpture. — [Łabędź et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9657006/)
- Flash or ambient effects increased M3C2 differences at greater sensor-to-target distances, and IMU or registration error dominated over LiDAR ranging error at larger scale. — [Luetzenburg et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593014/)
- Number of views and failure modes in food benchmarks:
  - MetaFood: easy (~200 frames) vs medium (30 images) vs hard (1 image) defines difficulty.
  - VolETA kept only about 35% of frames after deduplication and blur filtering.
  - Depth sensors fail on reflective or transparent food (soup, cutlery).
  - The iPhone X app was unaffected by viewing angle. — [MetaFood 2024](https://arxiv.org/html/2407.09285v1); [arXiv 2602.05078](https://arxiv.org/html/2602.05078v1); [Herzig 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7142738/)

### Inferences
- For tabletop food, the iPad should be held at the closest distance the LiDAR supports (around 25–50 cm) to maximise point density. Pose quality depends on RGB tracking, so even lighting and a textured surround help.
- Blur and duplicate rejection on keyframes (as VolETA does) is a validated preprocessing step that matches this branch's blur-rejection work.

### Gaps
- I found no controlled study of iPhone/iPad LiDAR on glossy sauces, dark foods, or translucent items such as jelly, drinks or rice grains specifically.

## LiDAR-only TSDF vs photogrammetry vs hybrid for small objects

### Takeaway
Direct comparisons consistently favour photogrammetry for objects under 30 cm, by about 3–10x lower median error. LiDAR gives metric scale, speed and coverage of recesses. Mesh fusion from ARKit LiDAR (as in the scanning apps) smooths away detail. In the food benchmark, the top-geometry hybrid (FoodRiddle: 2DGS + TSDF fusion + Poisson) won on Chamfer. I found no paper that isolates "LiDAR TSDF" against "LiDAR-scaled photogrammetry" on the same small objects.

### Cited Findings
- Figurine (~10 cm): iPad Scaniverse median 1.68 mm vs photogrammetry 0.16 mm. Bust (~70 cm): 3.47 mm vs 1.50 mm. Facade: similar (2.9 vs 2.5 cm). The iPad advantages are immediate scaled results and better reach into recesses indoors. — [Łabędź et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9657006/)
- A 2023 AGILE study found photogrammetry most accurate ("sub-centimetre details"). iPhone LiDAR was less accurate and less dense but faster. — [AGILE GIScience Series 4:49, 2023](https://agile-giss.copernicus.org/articles/4/49/2023/agile-giss-4-49-2023.pdf)
- In Luetzenburg's cliff-scale comparison, EveryPoint (ARKit LiDAR Mesh mode) and SfM-MVS could not reconstruct vegetation, whereas 3d Scanner App LiDAR could. — [Luetzenburg et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593014/)
- FoodRiddle's hybrid (2D Gaussian Splatting rendered to depth, TSDF fusion, support-plane projection, Poisson) won the MetaFood 2024 best-mesh award. VolE's masked MVS reached the lowest volume MAPE on MTF. — [MetaFood 2024](https://arxiv.org/html/2407.09285v1); [VolE](https://arxiv.org/html/2505.10205)
- Teppati Losè et al. recommend future apps embed SLAM that exploits the visible cameras to correct LiDAR pose drift, i.e., a hybrid. — [Teppati Losè et al. 2022](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)

### Inferences
- The most promising design for 5–30 cm food is hybrid:
  1. ARKit poses refined by SfM (COLMAP or PixSfM).
  2. Dense RGB reconstruction (MVS, NeuS2 or 2DGS).
  3. Metric scale and plate or table plane from LiDAR depth instead of a checkerboard.
  4. A watertight mesh closed against the support plane.
- LiDAR-only TSDF is likely bounded near the ±1 cm floor.

### Gaps
- No primary source in this pass reports TSDF-from-ARKit-depth volume error on small objects. Record3D and Polycam (LiDAR vs Photo mode) head-to-head results for objects under 30 cm were not found.
