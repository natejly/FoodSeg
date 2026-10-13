# Classical (non-neural-rendering) processing of noisy mobile ToF/LiDAR RGB-D for small-object geometry

Context: ARKit 256x192 sceneDepth + confidence, 1920x1440 RGB, ARKit poses, Open3D VoxelBlockGrid TSDF (1.5 mm voxel, trunc 8 voxels = 12 mm, confidence = high, max depth 0.8 m), flying-pixel filter (3x3 range of median-filtered depth > 2% of depth, keep foreground), blur rejection, optional COLMAP refinement. Targets are 5-30 cm objects seen from 20-50 cm.

Research budget note: ~17 tool calls. Several primary sources could not be fetched (the RealSense post-processing whitepaper host did not resolve; the Apple smoothedSceneDepth page returned no body; InfiniTAM/KinectFusion default parameters were not found in fetched text). Those items are listed under Gaps rather than stated from memory.

## Q0. What the iPad/iPhone LiDAR depth actually is (this drives every other choice)

### Takeaway
ARKit sceneDepth is not raw ToF. It is a sparse LiDAR dot pattern (~24 dots/row) densified by an ML interpolation that uses RGB. Its noise is spatially correlated but temporally uncorrelated, it has a negative (too-near) bias of 1-2% of depth (measured at 1-5 m), and it smooths away depth edges. So multi-frame averaging (TSDF) works well, single-frame spatial smoothing adds little, and edges need aggressive masking.

### Cited Findings
- ARKit depth is 256x192 px, float32 metres, already registered to the colour image, ~60° H x 48° V FoV. The confidence map has the same resolution with values 0 (low), 1 (medium) and 2 (high) — [Zea & Hanebeck 2022, "Modeling Spatial Uncertainty for the iPad Pro Depth Sensor", J. Adv. Inf. Fusion 17(2)](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- Invalid measurements are **not** encoded as depth 0. Their confidence is set to 0 and the depth is "extrapolated based on surrounding depth values and semantic cues from the color image". The gaps are invisible in the depth image alone, so filtering must use the confidence map — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- Large numbers of "fringe points"/flying pixels appear at object borders because of the low LiDAR resolution. Planar surfaces are not always flat and 90° corners are usually not preserved — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- The ML interpolation "aggregat[es] measurements into coherent surfaces and 'smooth[s]' out sudden changes in depth". Example: the back of a 2 cm-high box merges into the floor at 1 m. These smoothed regions are usually marked confidence 1 or 0, "allowing them to be filtered out". As of iOS 15 it cannot be turned off to get raw data — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- Narrow objects narrower than about 6% of the image (~15 depth px) fail to be detected. This is probably tied to LiDAR dot spacing (256 px/row ÷ 24 dots/row ≈ 11 px). Detection also fails where depth changes quickly at image edges — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- Covering the LiDAR makes ARKit hallucinate depth from RGB alone (a monitor at 20 cm was reconstructed as a cube at 2 m). Shiny or reflective objects can trigger the same behaviour locally — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- Depth quantisation is 1/1024 m (~0.977 mm) below 2 m and 1/512 m between 2 and 4 m — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- **Bias:** measured on a chessboard at 1-5 m with high-confidence pixels only, the bias "was always negative, that is, the iPad tells us that the object is closer than it really is". It grew roughly linearly and stayed within 1-2% of true depth — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- **Noise variance model:** σz² = 1e-6·(0.07z³ − 0.32z² + 0.64z − 0.02) with z in metres. Noise is unimodal and roughly Gaussian, with a support of only 2-3 quantisation levels at short range — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- **Correlation:** iPad noise is strongly spatially correlated (mean spatial correlation coefficient 0.25 vs 0.09 for Kinect 2). Correlation is ~0.2 within 20 px on the same surface. It is not temporally correlated: "entire surfaces appear to move and deform 'coherently' each frame" — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- Small objects relative to the FoV appear "smoothened", with hard corners flattened and surfaces fused together. The remedy is keeping the object "much larger than 11 px". A 15 cm-radius sphere was estimated at 15.5 cm near 1 m but 17.5 cm when only 32 px wide (it flattens into the floor) — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- MobileBrick (iPhone 13 Pro / iPad Pro, ARKit) captured at 20-50 cm. The authors note the LiDAR "fails when it is very close [to] the object" and gives "low-quality depth maps compared to those of commodity-level sensors". Depth is 192x256, RGB 1440x1920 at ~10 fps — [Li et al., MobileBrick, CVPR 2023](https://ar5iv.labs.arxiv.org/html/2303.01932) ([CVF PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_MobileBrick_Building_LEGO_for_3D_Reconstruction_on_Mobile_Devices_CVPR_2023_paper.pdf))
- An iPad Pro flat-surface study found ~90% of points within 1 mm of the best-fit plane when static at <1.5 m. This dropped to <20% within 1 mm (~80% within 1 cm) when the device was moving — [Vilniustech GAC article, search snippet](https://journals.vilniustech.lt/index.php/GAC/article/download/20797/12792) (not fetched in full)
- Apple LiDAR (iPhone 12 Pro) modelled small objects with sides >10 cm at about ±1 cm absolute accuracy — [Luetzenburg et al., Scientific Reports 2021 (DOAJ record)](https://doaj.org/article/05f6da2bb22e4343bec5db6fcf0ecf88) (via search snippet)

### Inferences
- Evaluating the Zea polynomial at our working range (it was fitted on 1-5 m, so this is extrapolation) gives σz ≈ 0.31 mm at 0.2 m, 0.38 mm at 0.3 m and 0.48 mm at 0.5 m. That is below the ~1 mm quantisation step. **Per-frame random noise is therefore not the main error source at 20-50 cm.** The main sources are (a) interpolation smoothing and edge errors, (b) depth bias, (c) pose error (~4.5 mm RMSE, see Q4) and (d) the ~11 px dot spacing. At 0.3 m with a 60° HFoV, one depth pixel covers ≈1.35 mm, but one LiDAR dot spacing covers ≈15 mm (≈10 mm at 0.2 m, ≈25 mm at 0.5 m). Detail finer than ~1-2 cm is largely interpolated from RGB.
- **The bias is a likely fattening source.** A depth that reads 1-2% too near puts each observed surface patch toward the camera that saw it. When views surround the object, that shift is outward, so the object is inflated. At 0.3 m, 1-2% is 3-6 mm per side. On a 5 cm object that is a 12-24% diameter error, or roughly 40-90% volume error. This bias was measured only at 1-5 m on one device and must be verified at 20-50 cm. Consider calibrating a per-device depth scale (and offset) against a known-size object, or against COLMAP-triangulated points in metric ARKit frames (see Q4).
- Because the noise is spatially correlated but temporally independent, temporal/multi-view averaging (TSDF) is the right denoiser. Per-frame spatial smoothing (bilateral, joint bilateral) mostly re-smooths an already smooth ML output and further rounds edges.
- Keep 'confidence == high' only. Medium and low mark exactly the interpolated and smoothed edge regions.
- Do not trust any depth pixel on a specular, glossy or wet surface (sauces, glazes) even at high confidence. RGB-hallucinated depth is spatially smooth but temporally inconsistent, so multi-view TSDF disagreement could flag it.

### Gaps
- No peer-reviewed bias/noise characterisation of iPad/iPhone LiDAR at 0.2-0.5 m was found. Zea measured 1-5 m. The bias sign and magnitude at close range is the highest-value unknown and should be measured in-house (e.g., scan a gauge block or a known-volume container).
- Apple's documentation on what `smoothedSceneDepth` averages (window, latency) could not be retrieved; the page returned no body text.

## Q1. Flying pixel / mixed pixel removal

### Takeaway
The industry practice is invalidation at edges: Azure Kinect invalidates mixed foreground/background pixels in its depth engine. On top of that, apply a neighbour-difference (jump-edge) test on the depth map before back-projection, then a light statistical/radius outlier pass on the fused or per-frame cloud. For ARKit, the confidence map already marks most smoothed edges. Erode the high-confidence mask by 1-2 px around depth discontinuities as well, because fringe points also occur at high confidence.

### Cited Findings
- Azure Kinect (AMCW ToF) invalidates pixels (depth = 0) for: outside illumination mask, saturated IR, low IR, "filter outlier", and multi-path interference. "Another common case of multipath is pixels that contain the mixed signal from foreground and background (such as around object edges). During fast motion, you may see more invalidated pixels around the edges" because of the raw-capture exposure interval — [Microsoft, Azure Kinect DK depth camera](https://learn.microsoft.com/en-us/azure/kinect-dk/depth-camera)
- Surfaces invalidated from one viewpoint "may be visible from another". This is the argument for discarding aggressively per frame and letting multi-view fusion fill in — [Microsoft, Azure Kinect DK depth camera](https://learn.microsoft.com/en-us/azure/kinect-dk/depth-camera)
- Flying pixels arise at depth edges where foreground and background light land on the same pixel. Mask-ToF (CVPR 2021) attacks this in hardware with microlens masks plus a learned decoder — [Chugunov et al., Mask-ToF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chugunov_Mask-ToF_Learning_Microlens_Masks_for_Flying_Pixel_Correction_in_Time-of-Flight_CVPR_2021_paper.pdf)
- Vendor practice is a "Depth Discontinuity Filter" that removes a pixel whose depth differs strongly from its neighbours, with median filtering as a common alternative (vendor article, not peer-reviewed) — [e-con Systems via edge-ai-vision.com, 2024](https://www.edge-ai-vision.com/2024/07/what-is-flying-pixel-and-how-can-it-be-mitigated-in-3d-imaging-for-time-of-flight-cameras/)
- On ARKit, fringe points are prominent and form "different curved lines each frame", meaning they are temporally inconsistent — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- Open3D point-cloud outlier filters: `remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)` ("the lower this number the more aggressive") and `remove_radius_outlier(nb_points=16, radius=0.05)` are the tutorial values. The radius value is for room-scale data and must be scaled down for small objects — [Open3D outlier removal tutorial](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html)

### Inferences
- The current filter (3x3 range of the median-filtered depth > 2% of depth, keep the foreground side) is a standard jump-edge test. At 0.3 m, 2% is 6 mm, which is ~15 quantisation steps and ~16σ of the random noise. It will not fire on real surface noise, so it is safe. It could be tightened to ~1% (3 mm) or to a fixed ~3-4 mm at close range. Because ARKit smooths edges into ramps over several pixels, a 3x3 window may miss the interior of a ramp. Options: use a 5x5 window, iterate/dilate the edge mask by 1-2 px, or add a normal-angle test.
- Normal-angle (grazing) test: compute normals from the depth map. Reject pixels whose angle between the viewing ray and the normal exceeds ~70-75°. Ramps from smoothed edges show up as near-grazing "surfaces". The repo's commit message says the incidence filter was turned off, presumably because it removed real rim geometry. An alternative is to down-weight rather than reject (Q3).
- "Keep the foreground side" is correct for food on a plate: the background is the plate/table, which is better observed from other views.
- Statistical outlier removal on the fused cloud with scaled parameters (e.g., nb_neighbors 20-30, std_ratio 1.5-2.0; radius ~3-5 voxels = 5-8 mm with nb_points ~8-16) removes residual floaters. Keep only the largest connected component near the look-at point (already done in run_meal per git log). These values are my scaling of the Open3D example values, not published defaults.

### Gaps
- The Intel RealSense "Depth Post-Processing" whitepaper (dev.intelrealsense.com) could not be fetched (DNS failure), so its exact spatial/temporal filter defaults (alpha, delta, persistence) are not given here. From memory they are roughly spatial alpha 0.5, delta 20 (disparity units), temporal alpha 0.4, delta 20, but this is **unverified**.
- No quantitative comparison of jump-edge thresholds for ARKit depth was found.

## Q2. Depth denoising: bilateral / joint bilateral / temporal / median

### Takeaway
For ARKit depth, per-frame spatial denoising has low value and real cost. The output is already smoothed by Apple's ML interpolation, and noise is spatially correlated, so spatial filters cannot average it out. Temporal/multi-view averaging inside the TSDF is the denoiser. A median filter is useful only as a robust pre-step for edge detection, not to replace the depth that gets fused.

### Cited Findings
- iPad depth noise is spatially correlated (~0.25 mean, ~0.2 within 20 px) and temporally uncorrelated — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- ARKit offers `smoothedSceneDepth`, described as an averaged distance measurement for steadier visuals, alongside raw-per-frame `sceneDepth`; Unity exposes these as raw vs. temporally smoothed environment depth — [Apple ARFrame docs](https://developer.apple.com/documentation/arkit/arframe/scenedepth); [Unity AR Foundation ARKit occlusion](https://docs.unity3d.com/Packages/com.unity.xr.arkit@6.0/manual/arkit-occlusion.html)
- The KinectFusion-style TSDF update is a weighted running average, and simply setting per-frame weight = 1 "provides good results". Volumetric fusion is the standard temporal denoiser — [Berkeley CS287 Dense SLAM lecture](https://www.cs.berkeley.edu/%7Epabbeel/cs287-fa13/slides/Dense%20SLAM%20Lecture.pdf); [DynamicFusion slides](https://www.slideshare.net/KenSakurada/dynamicfusion-reconstruction-and-tracking-of-nonrigid-scenes-in-realtime)

### Inferences
- Use raw `sceneDepth`, not `smoothedSceneDepth`, for fusion. The smoothed map averages across frames whose poses differ. It adds temporal lag (depth from time t−k paired with pose t) that blurs geometry under motion. Mark this as an inference: Apple does not document the algorithm.
- A joint-bilateral (RGB-guided) upsample of the 256x192 depth to RGB resolution would make depth edges align with colour edges. That is useful for segmentation masks, but it adds no geometric information. ARKit's interpolator already uses RGB, and the true sampling is ~11 px dot spacing.
- Practical recipe: median 3x3 only to compute the edge mask (as now). Fuse unfiltered high-confidence depth, minus the edge-masked pixels.

### Gaps
- No study was found comparing bilateral or temporal pre-filtering against plain TSDF on ARKit depth for small objects.

## Q3. TSDF best practice: voxel size, truncation, weighting, carving, frame selection, avoiding fattening

### Takeaway
1.5 mm voxels roughly match the depth-pixel footprint at 0.3 m (1.35 mm). Going finer gains nothing, given ~1 mm quantisation and ~15 mm dot spacing. Truncation should cover the expected misalignment (pose error ~4.5 mm RMSE plus depth bias of a few mm), not just random noise. The current 12 mm (8 voxels) is a sensible upper end; 6-9 mm reduces thin-structure bleed. Weight observations by a sensor noise model (∝ 1/σz²) and by incidence angle instead of hard rejection. Treat systematic bias as the main fattening risk, which TSDF averaging cannot remove.

### Cited Findings
- The truncation limit should approximate sensor noise; signed distances are clamped to ±μ to reject outliers — [Berkeley CS287 Dense SLAM lecture](https://www.cs.berkeley.edu/%7Epabbeel/cs287-fa13/slides/Dense%20SLAM%20Lecture.pdf); [ASH, Dong et al. 2021](https://arxiv.org/pdf/2110.00511)
- KinectFusion-style fusion caps the running weight at a maximum (W_max). OpenCV kinfu exposes `voxelSize`, `tsdf_trunc_dist` and `tsdf_max_weight` ("max number of frames per voxel") — [OpenCV kinfu Params](https://docs.opencv.org/5.x/de/d81/structcv_1_1kinfu_1_1Params.html)
- Open3D's VoxelBlockGrid integration does a weighted average by projecting voxels into images. The tutorial example uses voxel_size = 3.0/512 ≈ 5.9 mm and block_resolution = 8. For custom weighting it points to a "Customized Integration" page — [Open3D t-reconstruction integration](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html)
- Nguyen, Izadi & Lovell (2012) modelled Kinect noise as separate axial and lateral components, each a function of distance and incidence angle, over 0.4-3.0 m, for "improved 3D reconstruction and tracking" — [cited via review/search; see Koppel thesis](https://cescg.org/wp-content/uploads/2018/04/Koppel-Extracting-Sensor-Noise-Models-–-considering-X-Y-and-Z-Noise.pdf)
- For the iPad, σz² = 1e-6·(0.07z³ − 0.32z² + 0.64z − 0.02) (z in m) gives a ready-made weight 1/σz² — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- MobileBrick reports TSDF-Fusion (Open3D) from depth at Chamfer 13.78 mm, F1 28.77% at 2.5 mm and 53.39% at 5 mm. With ground-truth depth and perturbed poses, TSDF reached ~85% accuracy at 1 mm, showing that depth quality, not fusion, limits accuracy — [MobileBrick, CVPR 2023](https://ar5iv.labs.arxiv.org/html/2303.01932)
- ARKitScenes rejected frames whose LR (ARKit) depth had RMSE > 7 cm against downscaled laser depth, or > 20% of pixels off by > 5 cm. These are room-scale data-cleaning thresholds — [Baruch et al., ARKitScenes, NeurIPS 2021 D&B](https://ar5iv.labs.arxiv.org/html/2111.08897)

### Inferences
- **Voxel size:** the information limit is set by the LiDAR dot spacing (≈10-25 mm at 0.2-0.5 m), not by voxels. 1.5 mm is fine near 0.3 m. At 0.5 m one depth pixel covers ≈2.3 mm, so 1.5 mm voxels just add cost and holes; 2 mm would be adequate. Voxels much smaller than the pixel footprint leave unobserved voxels between rays, which Open3D's projective integration handles.
- **Truncation:** the zero crossing is only correct when trunc ≥ ~3x the combined misalignment. With ARKit translation RMSE ≈4.5 mm (Q4), 12 mm is about right. If poses are refined to ~2 mm, try 6-8 mm (4-5 voxels) to cut bleed on thin rims and leaves. Too small a trunc with imperfect poses produces holes and double surfaces.
- **Weighting:** w = conf_high · cos(θ_incidence) / σz²(z), clamped. This down-weights grazing ramps instead of deleting them, which avoids the rim loss that motivated disabling the hard incidence filter. Cap total weight (e.g., the equivalent of 50-100 frames) so late good frames can still move the surface. These numbers are my suggestions, not published defaults.
- **Free-space carving:** projective TSDF already updates the free-space band out to +trunc. For floaters beyond trunc, run a separate visibility check. Mark voxels in front of observed surfaces across many views as empty, and delete mesh components that lie in space that most views saw through.
- **Avoiding fattening:** (1) correct depth bias (scale factor); (2) avoid mixed-pixel edges (Q1); (3) good poses (Q4); (4) do not lower the mesh-extraction weight threshold so far that single-frame voxels appear; (5) Open3D marching cubes extracts at the zero level, so do not dilate masks.
- **Frame selection:** the current blur rejection plus a keyframe stride by pose change (e.g., new frame if > 1-2 cm translation or > 3-5° rotation) gives balanced viewpoint coverage and keeps one stationary pause from dominating the weights. This is common practice, not a cited rule.
- **Surfels:** ElasticFusion-style surfel fusion handles thin structures and lets per-surfel radius scale with depth and incidence. For a single-object offline pipeline with watertight-volume output, TSDF plus marching cubes is simpler. Surfel output still needs Poisson meshing for volume.

### Gaps
- InfiniTAM, BundleFusion and KinectFusion default μ, voxel size and max-weight values could not be confirmed from fetched sources (μ = 0.02 m and maxW = 100 are commonly quoted but **unverified** here).
- Open3D VoxelBlockGrid `trunc_voxel_multiplier` default (believed 8.0) and the API for per-pixel weights were not confirmed from the fetched page.

## Q4. Pose refinement: ARKit drift and refinement options; keeping metric scale

### Takeaway
ARKit poses at 20-50 cm object scale are good to about 4-5 mm translation RMSE and ~0.5° rotation (MobileBrick). That is already larger than the 1.5 mm voxel and the sub-mm depth noise, so refinement pays off. Options: (a) colored ICP (Park 2017) of each frame or fragment against the TSDF model, which is cheap and keeps metric scale; (b) feature-based BA (COLMAP `point_triangulator` with ARKit poses, then `bundle_adjuster`, or `pose_prior_mapper` with position priors), followed by a similarity alignment back to ARKit centres to restore metric scale.

### Cited Findings
- ARKit vs refined poses on MobileBrick calibration sequences: translation RMSE **4.454 mm** (ARKit) vs 2.060 mm (refined); rotation 0.581° vs 0.522° — [MobileBrick, CVPR 2023](https://ar5iv.labs.arxiv.org/html/2303.01932)
- MobileBrick refined ARKit poses by BA to "alleviate camera drift caused by ARKit". Features were matched across neighbouring frames, 3D points came from the aligned model, and matches > 2 cm apart in 3D were discarded — [MobileBrick, CVPR 2023](https://ar5iv.labs.arxiv.org/html/2303.01932)
- PoRF reports ARKit initial poses on MobileBrick with ~0.3-0.6° rotation and ~0.9-4.2 mm translation error per scene. Refinement reaches ~0.14-0.28° and 0.6-1.8 mm (values from search summary of the table) — [Bian et al., PoRF 2023](https://arxiv.org/pdf/2310.07449)
- ARKitScenes ground-truth poses: 2D-3D matches against laser-scan renderings, RANSAC+PnP, then joint optimisation minimising "a dense photometric error metric" — [ARKitScenes](https://ar5iv.labs.arxiv.org/html/2111.08897)
- Colored ICP (Park, Zhou, Koltun, ICCV 2017) optimises λE_G + (1−λ)E_C. The colour term stops sliding along tangent planes, which plain point-to-plane ICP cannot prevent. Open3D's multi-scale schedule is voxel radii [0.04, 0.02, 0.01] m with max iterations [50, 30, 14], `relative_fitness=1e-6`, `relative_rmse=1e-6`, and normals with radius 2×voxel and max_nn = 30 — [Open3D colored point cloud registration](https://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html)
- Open3D's reconstruction system (after Choi, Zhou & Koltun 2015, "Robust Reconstruction of Indoor Scenes") runs four stages: make fragments (RGBD odometry + multiway registration + integration), register fragments (global registration + ICP + multiway), refine registration, integrate scene. Example config: `depth_diff_max` 0.07, `preference_loop_closure_odometry` 0.1, `preference_loop_closure_registration` 5.0, `icp_method` "color", `global_registration` "ransac", `voxel_size` 0.05 (room scale) — [Open3D reconstruction system](https://www.open3d.org/docs/release/tutorial/reconstruction_system/system_overview.html)
- COLMAP with known poses: build `cameras.txt` / `images.txt` (empty `points3D.txt`), run `feature_extractor`, a matcher, then `point_triangulator` (`--input_path` = hand-built model). The pose-prior mapper is `pose_prior_mapper`, the incremental mapper with prior position constraints. `--overwrite_priors_covariance` with `--prior_position_std_x/y/z` (default 1.0 m) sets prior uncertainty. `model_aligner` estimates a 3D similarity from camera-centre references (≥ 3 images) with RANSAC threshold `--alignment_max_error`. Intrinsics are refined by default except the principal point; this is controlled by `refine_*` BA options — [COLMAP FAQ](https://colmap.github.io/faq.html)

### Inferences
- **Keeping metric scale:** with `point_triangulator` (poses fixed), scale is inherited from ARKit. If you then run free `bundle_adjuster` or `pose_prior_mapper`, run `model_aligner` (similarity, ARKit camera centres as reference, `--alignment_max_error` ≈ 0.01-0.02 m) afterwards to restore metric scale. Alternatively use `pose_prior_mapper` with `--prior_position_std_*` ≈ 0.005-0.01 m (matching ARKit's ~4.5 mm RMSE) so priors anchor scale. The 1 m default is far too loose for this use.
- **Intrinsics:** fix intrinsics to the per-frame ARKit values (refine_focal_length = 0, refine_extra_params = 0) unless many views are available. ARKit intrinsics are factory-calibrated per frame, and refining focal length lets scale and focal length trade off.
- **Depth-to-SfM scale check:** after COLMAP triangulation in the ARKit metric frame, compare the triangulated sparse points with the ARKit depth at their projections. The median ratio estimates the depth bias/scale from Q0 per scan. This is a cheap self-calibration.
- **Colored ICP for small objects:** scale the Open3D radii down ~10x for a 5-30 cm object, e.g. [0.006, 0.003, 0.0015] m, aligning each frame (or ~10-frame fragment) to the current TSDF model, then re-fuse. Use the pose graph / multiway step only if loop closure across a 360° orbit is needed. These radii are my scaling, not published values.
- **Learned matchers (hloc SuperPoint+LightGlue, MASt3R-SfM, VGGT):** not researched in depth within this budget. For low-texture food (rice, puree), SIFT may give too few matches and learned matchers help. All of these give up-to-scale or approximate-scale results, so they need the `model_aligner` step or ARKit-anchored BA to restore metric scale.

### Gaps
- No primary source fetched on GLOMAP with priors, pycolmap's PosePriorBundleAdjuster API, hloc, MASt3R-SfM or VGGT accuracy on small objects. Search turned up only secondary mentions; for example, a 2026 arXiv paper claims pycolmap exposes Cauchy-loss BA in 4.2.0.dev0 ([arXiv 2608.21008](https://arxiv.org/pdf/2608.21008)), which is unverified.
- BundleFusion's parameters (sparse-to-dense, 4 mm voxels?) were not confirmed.

## Q5. Depth/RGB time sync, rolling shutter, intrinsics scaling

### Takeaway
ARKit depth is registered to the colour image and shares its 4:3 aspect and FoV. Depth intrinsics are therefore the RGB intrinsics scaled by 256/1920 = 192/1440 = 0.1333. Timing details are undocumented in the sources found.

### Cited Findings
- The depth image "is already registered to the color image and has the same aspect ratio of 4 to 3", with ~60°x48° FoV for both — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)
- ARKit outputs per-frame intrinsics and extrinsics. MobileBrick gives no details on sensor time sync — [MobileBrick](https://ar5iv.labs.arxiv.org/html/2303.01932)
- ARKitScenes does not discuss depth/RGB sync — [ARKitScenes](https://ar5iv.labs.arxiv.org/html/2111.08897)

### Inferences
- Intrinsics pitfall: when scaling K from 1920x1440 to 256x192, scale fx, fy, cx and cy by 0.13333. Strictly, for pixel-centre conventions cx' = (cx + 0.5)·s − 0.5. The difference is ~0.43 px at the 256-px scale, ≈0.6 mm at 0.3 m, which is not negligible against 1.5 mm voxels. ARKit intrinsics are per frame (focus/OIS can change them), so use each frame's K, not the first frame's.
- Because ARKit depth comes from ML fusion of the LiDAR with the RGB frame, depth and RGB are paired in one ARFrame. Motion blur and rolling shutter in RGB will also corrupt the depth interpolation. Blur-based frame rejection therefore also improves depth quality. Slow orbits (< ~10-15 cm/s at 0.3 m) limit rolling-shutter skew. This is an inference, not a cited figure.

### Gaps
- No primary documentation on ARKit depth/RGB timestamp offsets or rolling-shutter readout time for the iPad wide camera was found.

## Q6. Meshing and closing for volume

### Takeaway
Marching cubes on the TSDF gives an open surface (the bottom is unseen). For volume, close it against the support plane: fit the table/plate plane with RANSAC, cut the mesh at it, and cap it. Alternatively run screened Poisson on the fused points plus synthetic points on the plane footprint, then trim low-density extrapolation.

### Cited Findings
- Open3D Poisson needs oriented normals (`estimate_normals` plus `orient_normals_consistent_tangent_plane(100)`) and returns per-vertex densities. The example trims vertices below the 0.01 density quantile, at depth = 9. Poisson "even extrapolates into some areas" (low-density regions) — [Open3D surface reconstruction](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)
- Ball-pivoting radii example [0.005, 0.01, 0.02, 0.04] m. Alpha-shape example alpha = 0.03 — [Open3D surface reconstruction](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)

### Inferences
- For a 30 cm bounding box, Poisson depth 9 gives ~0.6 mm cells and depth 8 ~1.2 mm. Depth 8-9 matches the 1.5 mm TSDF. Use a stronger density trim (e.g., 5-10% quantile) on the unseen underside, but only after the support-plane points are added. Otherwise Poisson balloons the bottom and inflates volume.
- Best-practice volume path: (1) RANSAC plane on table/plate points (threshold ~2-3 mm); (2) keep object points above the plane; (3) add a dense grid of plane points inside the object's projected footprint (alpha shape or convex hull of the projection) with normals pointing down; (4) screened Poisson; (5) clip at the plane; (6) check `is_watertight()` and compute the signed volume. Alternatively compute volume directly by integrating the height field above the plane on a 1-2 mm grid, which avoids meshing artifacts for convex-from-above food.
- For food on a plate, the plate's concave surface, not the table, is the true support. Fitting a plane to the plate rim overestimates volume by the plate's dish depth. Fit the support from plate points immediately around the food, or use a plate-only scan or prior.

### Gaps
- No primary source was fetched for screened Poisson parameters (Kazhdan & Hoppe 2013; point weight / `screening` default 4 in PoissonRecon) or on volume accuracy of plane-closed meshes for food. The well-known screening default is stated here from memory and is **unverified**.
