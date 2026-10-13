# Use LiDAR for scale, RGB for shape

Apple's mobile LiDAR cannot, by itself, deliver "extremely clean" geometry of 5-30 cm food. It measures only **576 true dToF dots per frame on 2020-2022 devices and about 112 on iPhone 15 Pro and later**. ARKit's 256x192 `sceneDepth` is an ML interpolation of those dots guided by the RGB image. Every published small-object study therefore lands at roughly **±1 cm absolute accuracy for objects over 10 cm, with a detection limit near 5 cm**, while photogrammetry on the same objects is 3-10x more accurate. Your own measurements fit this physics exactly. Head-on planes come out at 1.3-1.5 mm RMS because the ML interpolates flat surfaces well. Silhouette skirts of 1-4 cm and side walls 4-6 mm off appear because edges and grazing surfaces sit between dots and are smoothed or hallucinated. The cleanest results, from Object Capture, Polycam Photo mode and the MetaFood/VolE food-volume winners at 1-11% MAPE, all come from **dense multi-view RGB with LiDAR or ARKit supplying metric scale, a support plane and a prior**, not from LiDAR fusion alone. For this pipeline, the highest-value changes are, in order:

1. Fix the capture: a full 360° orbit at three heights, slower, in brighter diffuse light, with locked focus, white balance and short exposure.
2. Measure and correct a possible near-bias in ARKit depth, which would inflate every item.
3. Replace hard edge and grazing deletion with weighted TSDF integration.
4. Move the shape estimate to PromptDA-upsampled depth and, on the cluster, a depth-supervised Gaussian-splatting or MVS reconstruction masked by SAM.

No filter rescues a half orbit, and that is the current dominant error.

## Every pixel of sceneDepth is a network output anchored to sparse dots

### The sensor samples about one point per 1-4 cm at plate range

The rear LiDAR is a direct time-of-flight system: a Sony stacked SPAD receiver paired with a VCSEL emitter. On the 2020 iPad Pro, a Lumentum VCSEL and a diffractive optical element (DOE) project a **24x24 = 576-dot grid** ([Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf); [EE Times Japan](https://eetimes.itmedia.co.jp/ee/articles/2006/09/news059_3.html)).

The iPhone 15 Pro module was redesigned around **112 individually addressed VCSEL cavities producing an 8x14 dot pattern** with a refractive lens instead of a DOE. Its receiver is a 116x92 SPAD array at 10.1 µm pitch, and the module has a 3.0 mm emitter-receiver baseline ([Yole, ISSW 2024](https://imagesensors.org/Past%20Workshops/2024%20ISSW/Papers/R06.1.pdf)). One developer counted roughly 111 dots on a 15 Pro against 576 on a 12 Pro, and Apple declined to comment ([Apple forum 813071](https://developer.apple.com/forums/thread/813071)). A teardown reported in Japan found the same pattern on the 16 Pro ([iphone-mania](https://iphone-mania.jp/?p=588361)). No teardown documents the M4/M5 iPad Pro module, so **which generation your iPad uses is unknown**. You can check it in a minute by filming the dot pattern with an IR-sensitive camera.

The geometry follows from these numbers, though this is my inference. The depth field of view is about 60°x48°. On the 576-dot generation, dots are about 2.5° apart, which is roughly **1.3 cm between real measurements at 30 cm**. A 10 cm food item therefore gets about 40-60 true samples on old hardware and perhaps 6-15 on new. Zea & Hanebeck found that objects narrower than about 15 depth pixels (~6% of image width) are not detected at all, consistent with the ~11 px dot spacing ([Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)). Rice-grain relief, a 2 cm cookie edge or a plate rim lies mostly between dots.

### ARKit fuses LiDAR and RGB with ML, and you cannot turn it off

Apple describes sceneDepth plainly: the RGB image and LiDAR readings "are fused together using advanced machine learning algorithms to create a dense depth map", at 60 Hz, as z-depth from the camera plane ([WWDC20 Explore ARKit 4](https://developer.apple.com/videos/play/wwdc2020/10611/)). There is no API for raw dots, and as of iOS 15 the ML stage could not be disabled ([Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)). Nothing surfaced in this research suggests that has changed.

`smoothedSceneDepth` additionally averages "over time to lessen its frame-to-frame delta" ([Apple docs](https://developer.apple.com/tutorials/data/documentation/arkit/arframe/smoothedscenedepth.md)). Apple does not document the window or whether it is motion-compensated.

The closest thing to raw data is AVFoundation's `builtInLiDARDepthCamera` (iOS 15.4+). With `isFilteringEnabled = false` it "excludes low confidence points" instead of filling them, at up to **320x240 streaming or 768x576 with photos** ([WWDC22 session 110429](https://developer.apple.com/videos/play/wwdc2022/110429/)). It does not give you ARKit's VIO poses.

Zea & Hanebeck's characterisation of the iPad Pro 2021 is the only rigorous one, and it explains your artifacts:

- **Invalid returns are never encoded as zero.** Their confidence is set to 0 and the depth is "extrapolated based on surrounding depth values and semantic cues from the color image".
- **Smoothed discontinuities usually carry confidence 1 or 0.** For example, the back of a box merges into the floor.
- **Noise is spatially correlated but nearly temporally independent.** Whole surfaces "shift and deform" coherently each frame (spatial correlation 0.25 vs 0.09 for a Kinect 2).
- **Depth is quantised to 1/1024 m between 1 and 2 m.**
- **Bias was always negative at 1-5 m: objects read 1-2% too near**, growing roughly linearly with distance.

All from [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf). Covering the LiDAR makes ARKit hallucinate depth from RGB alone; their test turned a monitor 20 cm away into a cube at 2 m. Shiny surfaces can trigger the same behaviour locally.

Confidence (0/1/2) is defined by Apple only qualitatively. Apple says reflective or highly absorbing surfaces lower it, and that it is "useful in filtering out lower-accuracy depth values" ([Apple ARDepthData](https://developer.apple.com/tutorials/data/documentation/arkit/ardepthdata.md)).

### What this means for your numbers

The following is inference from the sources above. Extrapolating Zea's noise polynomial to 0.3 m gives a random σ of only about 0.4 mm, below the quantisation step. **Per-frame noise is therefore not your problem.** The problems are:

1. Edge interpolation, which produces your 1-4 cm skirts.
2. Smoothing at grazing incidence, which produces the 4-6 mm side-wall error.
3. Pose error.
4. A possible systematic near-bias.

The bias matters most for volume. A reading 1-2% too near, seen from all around an object, pushes each surface patch toward the camera that saw it, which inflates the object outward. At 0.3 m that is **3-6 mm per side: on a 5 cm item, roughly 12-24% diameter and 40-90% volume error.** The bias was only ever measured at 1-5 m on one device. Verifying it on-device at 20-50 cm is the single most important unknown in this report.

Physics sets the floor that processing cannot remove:

- Too few true samples on small items.
- Weak returns on dark food (nori, char, dark sauces).
- Specular glazes and wet surfaces, which send the pulse away or create multipath that reads too deep.
- Translucent or scattering food (rice, jelly, fat), which, by generic dToF physics, also reads slightly deep.
- Unreliable readings below about 10 cm ([Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)).

None of these material effects has been measured on Apple LiDAR. An inference worth testing: multipath and scattering bias depth *away* from the camera, while the global ML bias is *toward* it, so the sign of the residual on a known object hints at which effect dominates.

## Published accuracy puts LiDAR at about 1 cm, an order behind photogrammetry

The small-object literature is consistent, and none of it is encouraging for LiDAR-only fusion.

**Luetzenburg et al. (2021)** scanned 14 boxes with an iPhone 12 Pro / iPad Pro using 3d Scanner App. They found "absolute accuracy of ±1 cm" for sides over 10 cm, degrading below 10 cm, with a ~5 cm detection limit. Point density on a flat target fell from 7,225 pts/m² at 25 cm to 150 pts/m² at 250 cm ([PMC8593014](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593014/)).

**Łabędź et al. (2022)** scanned a 10 cm plaster figurine with an iPad Pro (Scaniverse). LiDAR gave a **median cloud-to-reference error of 1.68 mm, with 53% of points beyond 1.5 mm**. DSLR photogrammetry on the same figurine gave a **0.16 mm median**, ten times better ([PMC9657006](https://pmc.ncbi.nlm.nih.gov/articles/PMC9657006/)).

**Vogt et al. (2021)**, as summarised by Díaz-Vilariño et al., found iPad LiDAR "impractical" for LEGO-sized objects at 300 mm ([ISPRS 2022](https://isprs-archives.copernicus.org/articles/XLIII-B4-2022/303/2022/isprs-archives-XLIII-B4-2022-303-2022.pdf)).

**Teppati Losè et al. (2022)** compared apps and stressed "the crucial role of the software component". SiteScape was densest but drifted along the scan path. 3d Scanner App had no drift but a suspiciously constant 0.2 mm roughness, suggesting heavy built-in smoothing. Re-scanning an area "did not necessarily improve the quality" ([Remote Sens. 14:4157](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-04157/article_deploy/remotesensing-14-04157.pdf)).

**Motion matters as much as range.** A built-environment study found static scans within <1 mm of a plane but handheld dynamic scans at about 1 cm typical error ([Teo & Yang 2023](https://doaj.org/article/9cbab8e9cd04423aae61f174ca458b1e)).

**MobileBrick** is the closest benchmark to your setup: iPhone 13 Pro / iPad Pro, 20-50 cm, small LEGO models. The authors report that the LiDAR "fails when it is very close" to the object. Open3D TSDF fusion of ARKit depth reached **Chamfer 13.78 mm and F1 of only 28.8% at 2.5 mm**. With ground-truth depth and perturbed poses the same fusion reached ~85% at 1 mm, showing that **depth quality, not fusion, is the bottleneck** ([MobileBrick, CVPR 2023](https://ar5iv.labs.arxiv.org/html/2303.01932)). The same work measured ARKit pose error at **4.45 mm translation RMSE and 0.58° rotation**, reduced to 2.06 mm by bundle adjustment.

Food-volume results follow the same pattern:

| Method | Depth source | Volume error |
|---|---|---|
| LiDARCalorieCam (Fujita & Yanai 2025): single-view iPhone LiDAR, DeepLabV3 masks, convex hull ([slides](http://mm.cs.uec.ac.jp/pub/conf25/251111fujita_3_ppt.pdf)) | LiDAR, one view | MAPE 8.8-13.9% on ~350 cm³ items; 31-65% on 53-163 cm³ items |
| Herzig et al. 2020, iPhone X TrueDepth, 128 items ([PMC7142738](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7142738/)) | Structured light | 14.0% weight error |
| VolETA, MetaFood 2024 winner: PixSfM + SAM/XMem++ + NeuS2, checkerboard scale ([report](https://arxiv.org/html/2407.09285v1)) | None (RGB) | 10.97% MAPE |
| VolE 2025: ARKit poses → COLMAP, FoodMem masks, mask-consistent MVS ([arXiv 2505.10205](https://arxiv.org/html/2505.10205)) | None (RGB) | 1.22% MAE (Foodkit); 3.08% MAPE (MTF, abstract says 2.22%) |

The inverse size trend in LiDARCalorieCam is the ~1 cm floor showing through, since 1 cm is a third of a 3 cm-tall item. Your sim's 27% MAPE after edge filtering sits in the same band. The best food systems are all **multi-view RGB with masks**, using ARKit or a checkerboard only for scale. No peer-reviewed study reports volume error for TSDF fusion of the iPad LiDAR stream on food, so your scans would be novel data. Apple publishes no accuracy figures for Object Capture. The only independent study found measured 0.23-0.47 mm caliper error on handaxes ([Hurst et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11637407/)), consistent with your observation that it is sharper than LiDAR TSDF.

## Clean capture is mostly light, speed and coverage

### Video formats and high-resolution stills

The default format bins a 3840x2880 sensor region 2x2 down to 1920x1440 at up to 60 fps. The 4K format (`recommendedVideoFormatFor4KResolution`) is non-binned 3840x2160 at 30 fps, 16:9, and is the only family that supports HDR ([WWDC22 Discover ARKit 6](https://developer.apple.com/videos/play/wwdc2022/10126/)).

The more useful API is **`ARSession.captureHighResolutionFrame`**. It triggers "an out-of-band capture of a high-resolution image" (12 MP on iPhone 13) and returns a full ARFrame whose pose and frame semantics are populated, without disturbing the video stream. Pick `recommendedVideoFormatForHighResolutionFrameCapturing` / `isRecommendedForHighResolutionFrameCapturing` for the best stills. Only one capture can be in flight at a time ([Apple docs](https://developer.apple.com/documentation/arkit/arsession/capturehighresolutionframe(completion:))). iOS 26 adds `captureHighResolutionFrame(using: AVCapturePhotoSettings?)` ([Apple docs](https://developer.apple.com/documentation/arkit/arsession/capturehighresolutionframe(using:completion:))). Holding ARFrames too long drops tracking to `.limited` ([WWDC22](https://developer.apple.com/videos/play/wwdc2022/10126/)).

**Untested on device:** Apple does not document whether `sceneDepth` is populated in the 4K or hi-res-recommended formats, or how the hi-res frame's depth aligns with its still. Check `supportsFrameSemantics` per format and log non-nil depth on returned hi-res frames. Always use the returned frame's own `camera.intrinsics` and `imageResolution`.

### Locking exposure, focus and white balance

Since iOS 16, `ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera` returns the AVCaptureDevice. It is nil on devices without an ultra-wide camera. After `lockForConfiguration()` you can call `setExposureModeCustom(duration:iso:)`, `setWhiteBalanceModeLocked(with:)` and `setFocusModeLocked(lensPosition:)` ([Apple docs](https://developer.apple.com/documentation/arkit/arconfiguration/configurablecapturedeviceforprimarycamera)).

An Apple engineer recommended exactly this for motion blur in dim ARKit sessions, warning that it "may affect tracking performance, particularly on devices with a single rear camera" ([forum 833148](https://developer.apple.com/forums/thread/833148)). The same route locks exposure and white balance for multi-frame consistency ([forum 832172](https://developer.apple.com/forums/thread/832172)). Object Capture guidance is: "don't make changes to any camera settings while shooting" ([Apple](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture)). Community reports attribute per-frame intrinsics changes to autofocus lens motion ([forum 840946](https://developer.apple.com/forums/thread/840946)). That is unconfirmed by Apple, but it argues for a focus lock during the orbit. EXIF (exposure, ISO, white balance) is available on every ARFrame ([WWDC22](https://developer.apple.com/videos/play/wwdc2022/10126/)). Avoid HDR for reconstruction, since tone mapping varies frame to frame (inference).

### Lighting, distance, speed and passes

**Lighting.** Use bright, diffuse light. Apple warns that "hard light, such as from an on-camera flash, direct sunlight, or a bare light bulb can cause problems" and recommends bouncing or diffusing it ([Apple](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture)). Sunlight overlapping the ToF band adds noise and dropouts on comparable sensors ([RealSense L515](https://realsenseai.com/?p=12939)). Dim light forces long shutters, which blur both the RGB and, because sceneDepth is RGB-guided, the depth (inference). That is likely part of your 10.8 s dim-light scan's edge smear.

**Distance.** At around 25-40 cm the plate fills 60-80% of the frame. Density is highest near 25 cm per Luetzenburg, but MobileBrick and Zea warn about very close range. Plot confidence and error against distance on the target device.

**Speed.** My estimate (inference, not sourced): at 25 cm with 1920 px across ~65° HFOV, keeping blur under 1 px at a 1/250 s shutter needs about **≤5 cm/s translation**. That is one-third to one-quarter of your measured 18 cm/s and implies roughly 20 s per ring.

**Passes.** Apple recommends three passes, and for objects that cannot be flipped, "three different heights". Keep ≥70% overlap between neighbouring shots, since below 50% reconstruction "may fail" ([WWDC23 Meet Object Capture](https://developer.apple.com/videos/play/wwdc2023/10191/); [Apple](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture)). For plated food this means rings at roughly 30°, 50° and 70° elevation plus a near-top-down pass, each closed to 360° and overlapping the previous ring: about 60-100 keyframes over 45-90 s. A matte, textured placemat helps ARKit tracking on low-texture tables and simplifies segmentation (inference).

**Storage.** Keep raw `sceneDepth`, not only smoothed, plus confidence, per-frame intrinsics, tracking state and EXIF. This is the Stray Scanner layout: 16-bit mm depth PNGs, 0/1/2 confidence PNGs, `odometry.csv` with per-frame fx, fy, cx, cy ([Stray format](https://github.com/strayrobots/scanner/blob/main/docs/format.md)). Matching it lets nerfstudio-style tooling ingest your scans.

## Processing should weight evidence rather than delete it

### Edges, confidence and denoising

Industry ToF practice is to invalidate mixed foreground/background pixels at edges and let other views fill them. Azure Kinect zeroes them in its depth engine because surfaces invalidated from one viewpoint "may be visible from another" ([Microsoft](https://learn.microsoft.com/en-us/azure/kinect-dk/depth-camera)). On ARKit, fringe points form "different curved lines each frame", so they are temporally inconsistent and multi-view fusion can out-vote them ([Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)).

Your jump-edge test (2% of depth ≈ 6 mm at 0.3 m) sits about 15 quantisation steps above noise, so it is safe. Because ARKit smooths edges into multi-pixel ramps, though, a 3x3 window misses the ramp interior (inference). Keeping `confidence == high` is right: medium and low mark exactly the smoothed edges.

Per-frame bilateral or joint-bilateral smoothing adds little. The output is already ML-smoothed and the noise is spatially correlated, so **TSDF averaging is the denoiser** ([Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf); [Berkeley dense SLAM notes](https://www.cs.berkeley.edu/%7Epabbeel/cs287-fa13/slides/Dense%20SLAM%20Lecture.pdf)). Use raw `sceneDepth` for fusion, since smoothed depth mixes frames with different poses (inference; `fuse.py` already defaults to raw).

### TSDF parameters and weighting

A 1.5 mm voxel matches the 1.35 mm depth-pixel footprint at 0.3 m. Truncation should exceed the combined misalignment: ~4.5 mm ARKit pose RMSE plus any bias. The current 12 mm (Open3D's default 8-voxel multiplier, which `fuse.py` does not override) is reasonable with ARKit poses. Drop it to 6-8 mm only after pose refinement (inference built on [MobileBrick](https://ar5iv.labs.arxiv.org/html/2303.01932) pose figures).

The principled replacement for the default-off grazing filter is a per-pixel weight rather than rejection: **w = cos θ_incidence / σz²(z)**, using Zea's σz² = 1e-6·(0.07z³ − 0.32z² + 0.64z − 0.02) ([Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)). Cap the total weight so one stationary pause does not dominate. Open3D's VoxelBlockGrid supports this through its "Customized Integration" path ([Open3D](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html)). The exact weight values are my suggestion, not published defaults.

AGS-Mesh offers an alternative cited rule: drop LiDAR pixels whose depth-derived normal disagrees with a monocular normal prior (Omnidata) by more than **10°** ([AGS-Mesh](https://arxiv.org/html/2411.19271v2)). This targets exactly the smoothed rims and ramps.

### Pose refinement

ARKit poses (4.45 mm, 0.58°) are three times your voxel size, and refinement to about 2 mm is demonstrated ([MobileBrick](https://ar5iv.labs.arxiv.org/html/2303.01932)). PoRF reports 0.6-1.8 mm ([PoRF](https://arxiv.org/pdf/2310.07449)). There are two practical routes:

- **COLMAP with ARKit priors.** Run `point_triangulator` on ARKit poses, then bundle adjustment with intrinsics fixed, then `model_aligner` back to ARKit camera centres to restore metric scale. Alternatively, use `pose_prior_mapper` with `--prior_position_std_*` around 5-10 mm rather than the 1 m default ([COLMAP FAQ](https://colmap.github.io/faq.html)). Your `refine_poses.py` already does the fixed-intrinsics, re-align variant.
- **Colored ICP of frames to the TSDF model.** Use Open3D's multi-scale schedule scaled down about 10x for tabletop work, e.g. radii [6, 3, 1.5] mm ([Open3D colored ICP](https://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html); the scaling is my inference).

There is also a free by-product, inferred but cheap: once COLMAP has triangulated points in the metric ARKit frame, the median ratio of their depths to ARKit sceneDepth at the same pixels is **a per-scan estimate of the depth bias**.

### Upsampling, splatting and Object Capture

**Prompt Depth Anything (PromptDA)** is built for exactly your input: ARKit 192x256 depth prompting Depth Anything to give metric depth at RGB resolution. On ARKitScenes it cuts L1 from **0.0251 m (raw ARKit) to 0.0132 m**. On ScanNet++ iPhone data it lifts **TSDF F-score from 0.679 to 0.762**. Monocular metric models are far worse (Depth Pro 0.0437 L1; Metric3D v2 TSDF F 0.65) ([PromptDA](https://arxiv.org/html/2412.14015)). It is Apache-2.0, with Large (340M), Small (25M) and a Small-Transparent variant fine-tuned with simulated iPhone LiDAR. The authors flag inherited LiDAR flicker and poor long-range depth ([GitHub](https://github.com/DepthAnything/PromptDA)). None of these results is on close-range food; that is untested.

**Gaussian splatting with sensor depth** is where phone captures gain most:

- On ScanNet++, DN-Splatter's ablation moves F-score from **0.134 with no depth to 0.583 with iPhone sensor depth**, against ~0.14 for ZoeDepth or Metric3D priors.
- On MuSHRoom, DN-Splatter reaches Chamfer-L1 0.0216, while vanilla 2DGS gets 0.0687 and SuGaR 0.0620 ([DN-Splatter](https://arxiv.org/html/2403.17822)).
- AGS-Mesh's 2DGS variant reaches 0.0276-0.0285 on ScanNet++ in about 40 minutes, versus GOF at 0.1187 ([AGS-Mesh](https://arxiv.org/html/2411.19271v2)).
- PGSR leads on DTU (Chamfer ~0.47-0.53; [PGSR](https://github.com/zju3dv/PGSR)), but no evidence shows purely photometric surface methods holding up on phone captures. The DN-Splatter and AGS-Mesh tables show 2DGS and GOF collapsing there.

All of these benchmarks are room-scale, where 2 cm Chamfer is a different proposition from a 5 cm food item.

**Apple Object Capture** is a hybrid of auto-selected sharp photos and LiDAR points, which helps low-texture objects. On iOS it reconstructs only at "reduced" detail; higher levels need a Mac. It outputs `.poses` but cannot take your ARKit poses as input ([WWDC23](https://developer.apple.com/videos/play/wwdc2023/10191/)). Its avoid-list (reflective, transparent, thin, deformable) overlaps food.

**MetaFood's best-mesh team (FoodRiddle)** gives a citable closing recipe: COLMAP poses, then 2DGS, then TSDF, then project bottom vertices onto the support plane, then Poisson ([MetaFood report](https://arxiv.org/html/2407.09285v1)).

## Commercial apps get clean scans by leaving LiDAR for photos

The pattern across apps is clear. LiDAR modes give instant, metric, coarse meshes. Small objects get routed to image-based or hybrid modes.

**3d Scanner App** exposes the clearest settings: confidence LOW/MEDIUM/HIGH ("HIGH only keeps the best quality data"), max depth 0.3-5 m, voxel resolution 5-20 mm (10-15 mm recommended), and OBJECT masking. Its Photo mode calls Apple's Object Capture ([3d Scanner App help](https://3dscannerapp.com/help/)).

**Polycam** says LiDAR gives true scale "provided the object is not too small" and steers small objects to Photo mode with object masking ([Vuforia/Polycam](https://developer.vuforia.com/library/using-3d-scans/model-targets-polycam)). Its developer-mode export carries globally optimised poses that nerfstudio trusts over raw ARKit ([nerfstudio](https://docs.nerf.studio/quickstart/custom_dataset.html)).

**Scaniverse** moved to on-device Gaussian splats with an "Enhance" pass and has shipped release-note improvements for thin objects and small detail via depth mapping ([Scaniverse](https://scaniverse.com/news/scaniverse-introduces-support-for-3d-gaussian-splatting); [release notes](https://scaniverse.com/about/releasenotes)).

**KIRI Engine** offers Photo, 3DGS, LiDAR and a Featureless (neural surface) mode, recommends three orbit layers and auto object masking ([KIRI](https://kiriengine.app/faq/how-to-capture-with-featureless-object-scan)).

**Record3D, Stray Scanner and NeRFCapture** are recorders. They save RGB, ARKit poses, per-frame intrinsics, 256x192 depth and confidence for offline processing ([Record3D](https://record3d.app/); [Stray](https://github.com/strayrobots/scanner)).

**Heges**' developer claims TrueDepth is about 10x more accurate than LiDAR for small objects ([App Store](https://apps.apple.com/app/id1382310112)). That claim is unverified, but it is consistent with TrueDepth studies reaching sub-mm accuracy on faces ([PMC11723895](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11723895/)).

**Apple's ObjectCaptureSession** sets the reference for live guidance: a per-direction coverage dial, auto-capture of sharp frames that "will stop and remind you to slow down", too-dark, too-close and too-far prompts, an off-frame arrow, and a point-cloud review between passes ([WWDC23](https://developer.apple.com/videos/play/wwdc2023/10191/)). RealityScan adds focus peaking and a live quality point cloud ([RealityScan 1.8](https://forums.unrealengine.com/t/realityscan-mobile-1-8-release-notes/2676487)).

**None of these apps publishes how it cleans LiDAR internally.** Settings and outputs are documented; filtering algorithms are not. No independent benchmark compares them on objects under 30 cm.

## Ranked recommendations for this pipeline

Rank reflects expected impact on item volume per unit effort, judged against your measured failure modes. "Basis" separates cited facts (V) from inferences (I).

| # | Change | Where | Concrete settings | Basis | Test on-device |
|---|---|---|---|---|---|
| 1 | **Enforce full coverage before Stop**: three closed 360° rings plus a top-down pass, ≥70% overlap | `ScanRecorder.swift` live guide, `scan_report.py` grade | Rings at ~30°/50°/70° elevation plus near-top-down; block Stop until each azimuth bin (e.g. 36 × 10°) per ring has a sharp frame; warn under 50% overlap | V (Apple 3 heights, 70%/50% overlap); I (bin sizes) | Coverage vs MAPE on known-volume objects |
| 2 | **Gate speed and add light** | Live guide; `docs/collection_protocol.md` | Target ≤5 cm/s (vs 18 cm/s measured); refuse frames when est. blur > 1 px; require bright diffuse light (bounce or diffuser, no sun or flash); matte textured placemat | V (Apple diffuse light, slow motion); I (5 cm/s math) | Blur px vs speed at 25 cm |
| 3 | **Lock focus, WB and exposure after framing** | `ScanRecorder.swift` session start | `configurableCaptureDeviceForPrimaryCamera` → `lockForConfiguration()` → `setFocusModeLocked(lensPosition: device.lensPosition)`, `setWhiteBalanceModeLocked(with: device.deviceWhiteBalanceGains)`, `setExposureModeCustom(duration: ≤1/250 s, iso: auto-compensated)`; keep `isAutoFocusEnabled` only until lock; log EXIF per frame | V (APIs, Apple engineer advice); I (1/250 s) | Tracking stability with locks; whether `camera.intrinsics` stays constant |
| 4 | **Measure and correct the ARKit depth near-bias** | New step in `calibrate.py`; applied in `loader.py` before `fuse.py` | Scan a gauge object (box/cylinder of known size) and a flat plane at 15/20/25/30/40 cm; fit depth' = a·depth + b per device; also compute the per-scan median ratio of COLMAP-triangulated depth to sceneDepth in `refine_poses.py` | V (1-2% near-bias at 1-5 m); I (applies at 20-50 cm) | Yes: the highest-value unknown |
| 5 | **Replace hard grazing/edge deletion with weighted integration** | `fuse.py` (custom VoxelBlockGrid integrate) | w = conf_high · cos θ / σz²(z) with Zea's σz²; keep the foreground-preserving jump-edge test but widen to 5x5 or dilate the edge mask 1-2 px on the *background* side only; cap weight ≈ 50-100 frames | V (σz² model, Kinect invalidation practice); I (weights) | Sim (`sim_scan_quality.py`) and real box sides (target < 2 mm) |
| 6 | **Keyframe by pose change, not every 4th frame** | `loader.py` `sharp_frames` | Sharpest frame per bin of >1-2 cm translation or >3-5° rotation; keep the Laplacian < 0.5x median rejection; drop frames with `trackingState != .normal` | V (VolETA keeps ~35% after blur/dup filtering); I (bin sizes) | No |
| 7 | **Always refine poses, then tighten truncation** | `refine_poses.py` → `fuse.py` | Keep fixed-intrinsics BA + `model_aligner` re-alignment; then trunc 6-8 mm (4-5 voxels via `trunc_voxel_multiplier`) instead of 12 mm; optional colored ICP to the model with radii [6, 3, 1.5] mm | V (ARKit 4.45 → 2.06 mm); I (trunc and radii) | Box-side error before/after |
| 8 | **PromptDA upsampling as default depth for fusion** | `depth_upsample.py` (cluster for Large; Small locally) | Prompt with high-confidence-only LiDAR; run at 1440x1920 on A40/L40S (Large) or 768x1024 locally (Small); fuse at 1.5 mm; compare edges and skirt width vs raw | V (ScanNet++ TSDF F 0.68 → 0.76); I (food) | Glossy sauces, dark food, rims |
| 9 | **Add high-res stills for texture and masks** | `ScanRecorder.swift` | `recommendedVideoFormatForHighResolutionFrameCapturing`; call `captureHighResolutionFrame` every ~10-15° of orbit; store HEIC/PNG; keep the 60 fps stream for tracking and depth | V (API); I (cadence) | Whether hi-res frames carry sceneDepth; tracking with the non-default format |
| 10 | **Cluster reconstruction: SAM-masked DN-Splatter / AGS-Mesh (2DGS) with LiDAR depth and normal priors** | New cluster stage after `segment.py` | ARKit/refined poses; depth loss from PromptDA or raw LiDAR; AGS 10° depth-normal consistency filter; TSDF or Poisson mesh; close the bottom on the plate surface (FoodRiddle) | V (DN-Splatter, AGS-Mesh, FoodRiddle); I (transfer to food) | Volume MAPE vs TSDF on same scans |
| 11 | **Keep Object Capture as a benchmark, not the pipeline** | Offline comparison | Run on the same captures (Mac, `full` detail); compare silhouettes and volume | V (no pose input, reduced detail on iOS) | Food failure cases |
| 12 | **Probe AVFoundation unfiltered LiDAR and TrueDepth for a static top-down shot** | Experimental capture mode | `builtInLiDARDepthCamera`, `isFilteringEnabled = false`, 768x576 photo depth; separately test front TrueDepth at 20-30 cm | V (holes instead of smears, 768x576); I (usefulness) | Cannot run alongside ARKit (unverified) |

Beyond these ranks, keep fusing raw `sceneDepth` and treat `smoothedSceneDepth` as a display aid. Keep `min-conf 2`. Lower `max_depth` from 0.8 m to about 0.6 m to match the capture band (3d Scanner App goes to 0.3 m). Volume closure should use the plate surface around the food rather than the table, because no published method handles concave supports and fitting the rim overestimates by the dish depth (inference).

## Conclusion

The research reframes the problem. Your LiDAR is not noisy; it is *sparse and opinionated*. Flat surfaces come out at a millimetre because the network interpolates planes well. Edges and walls fail because that is where the network invents geometry between dots 1-4 cm apart. Better filtering can only delete invented geometry, never recover the real thing, which is why the edge filter helped the sim (64% to 27%) but cannot rescue missing views. The path to "extremely clean" therefore runs through RGB: sharp, well-lit, fully covering multi-view images supply the shape, while LiDAR supplies scale, the support plane, a depth prior for splatting or upsampling, and a check on hallucinated surfaces.

Two cheap experiments could change the ranking. First, the near-range bias calibration: if ARKit really reads 1-2% near at 25 cm, every fused item is inflated by several millimetres per side, a bigger effect than any filter tuned so far. Second, the per-format test of high-res frame capture with sceneDepth, which decides whether one capture app can feed both a LiDAR TSDF and a 12 MP photogrammetry or splatting reconstruction from a single orbit. The remaining unknowns (dot count on your iPad, PromptDA on food, Object Capture accuracy on plates) all need your own scans of known-volume objects. No published study covers this exact regime.
