# Camera pose estimation for short handheld RGB-D scans of plated meals: research notes

Date: 2026-10-17. Web research done with search snippets plus a few page fetches. Many papers could not be
opened in full (PDF text did not decode; Springer and Nature pages require a cookie redirect), so every
number below is tagged:

- **[V]** = seen in a fetched or searched source (link given).
- **[S]** = secondary or snippet-level source, not checked against the paper.
- **[U]** = unverified. This is my own knowledge or inference, and you should check it before citing.

Nothing here was run on our data. Section 5 is a plan, not a result.

---

## 0. Bottom line

1. **Use the table.** The scene is dominated by one plane, and that is what makes depth-ICP slide. Anchor the poses with something
   that is *not* the plane: a fiducial board or texture on the table, or photometric and feature terms.
2. **Cheapest high-payoff test:** put an ArUco/ChArUco/AprilTag board (or a printed
   high-texture mat) under the plate. Solve for the camera pose from the board via PnP on the 1920x1440 RGB.
   Expect roughly 1-3 mm and under 0.5 deg when the board is large and well seen **[U]**. That is better than anything
   markerless on a 10 s handheld arc. Cost: about 1 day. Cost to the user: one printed sheet.
3. **Best markerless candidate:** RGB-D photometric+geometric odometry or feature-based BA with
   *depth-derived 3D-3D constraints* (our current "features" path), followed by pose-graph or windowed BA over all
   frames. Do not use COLMAP BA alone, since it has no metric depth and is weak on a textureless table.
4. **Feed-forward models** (VGGT, MASt3R-SLAM, MapAnything) are the main 2025-26 option for a markerless fallback.
   They are the easiest to integrate, but their accuracy on mm-level tabletop geometry is **not** established in any
   source I found. Treat them as an initialiser or a robustness check, not the final pose source.
5. **Pose accuracy needed.** The relevant error is *inter-frame inconsistency* of the fused surface, not global drift.
   Rough estimate (section 4.3): for ~5% volume error, consistency of about 1-2 mm and under 0.3-0.5 deg.

---

## 1. ARKit tracking failure modes

### 1.1 Documented behaviour

- Apple's own "Understanding World Tracking" says VIO fuses motion sensing with image feature tracking, and that quality
  drops with featureless views ("a blank wall"), darkness, and excessive motion ("too far, too fast, or shaking too
  vigorously ... blurred image or too much distance for tracking features between video frames")
  **[V]** [Apple: Understanding World Tracking](https://developer.apple.com/documentation/arkit/understanding-world-tracking).
  It also says ARKit does better if the device is moving, "even if ... only subtly."
- The tracking-state reasons are `.excessiveMotion`, `.insufficientFeatures`, `.initializing`, `.relocalizing`. The
  flag is a coarse quality signal. Apple's own docs give no numeric thresholds for lighting, texture, speed, or
  distance **[V: none found]**. I found no concrete Apple numbers, so the capture protocol must be set empirically.
- Apple engineers on the forum advise discarding frames when the state degrades. A forum answer says to discard a frame or two
  if the number is low, and investigate if it stays low **[S]** [Apple forums: tracking-state thread](https://developer.apple.com/forums/thread/699163).
- Apple has no published accuracy figure for LiDAR depth; engineers recommend testing yourself and filtering by
  `ARConfidenceLevel` **[V]** [Apple forums: Depth accuracy of the LiDAR](https://developer.apple.com/forums/thread/650321).

### 1.2 Evidence of runaway while `normal`

- Direct evidence is anecdotal. A forum thread reports reproducible, directional world-tracking drift on LiDAR devices
  (iPhone 14 Pro, 16 Pro) on iOS 26.4. Non-LiDAR devices were stable, and no tracking-state data or metric magnitudes
  were given. Apple acknowledged a known issue, and a user later reported that Apple
  indicated a fix in iOS 26.6.1, secondhand **[V, anecdotal]**
  [Apple forums: ARKit World Tracking Drift Regression on LiDAR Devices, iOS 26.4+](https://developer.apple.com/forums/thread/827240).
  **Check which iOS version produced our runaway scans.** If it was 26.4-26.6, part of the problem may be an OS
  regression, and the same scan would behave differently after updating.
  Other forum reports of transform drift: [thread 788370](https://developer.apple.com/forums/thread/788370) **[S, not opened]**.
- No peer-reviewed study that I found documents "normal-state" runaway. Assertion that this is common is **[U]**;
  our own scans are the best evidence.

### 1.3 Is ARKit pose + LiDAR accurate enough for small objects?

- Benchmark of consumer VIO against OptiTrack: ARKit was the most stable of ARKit/ARCore/T265/ZED 2, with drift of about
  0.02 m per second **[S]** [Kim et al., Sensors 2022 (PMC9785098)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9785098).
  At that rate a 10 s arc could accumulate ~20 cm. This is a *typical* drift figure, not the bound for small-object work, and a
  typical rate says nothing about the runaway tail.
- iPhone 12 Pro Max LiDAR on stockpile models: MAPE 0.56-1.88% on regular shapes and 4.45% on an irregular sand pile
  **[V]** [ISPRS Archives 2025](https://isprs-archives.copernicus.org/articles/XLVIII-M-10-2025/91/2026/). These are large
  objects (stockpiles), and the page does not say how poses were estimated, so it is only weakly transferable.
- iOS LiDAR scans of larger objects show 0.9-9.3 cm bias in dimensions **[S]**
  [SAE 2021-01-0891](https://saemobilus.sae.org/content/2021-01-0891/).
- A 2021 Technologies paper on iPad Pro LiDAR scans of Lego bricks found them worse than an industrial scanner but possibly
  adequate for some uses **[S]** (see the list result for "Lego iPad Pro LiDAR TrueDepth", not opened).
- Verdict: LiDAR depth at 256x192 is cm-level noisy on single frames, so mm-level volume accuracy comes from *fusing many
  frames with consistent poses*. That makes pose consistency the main bottleneck.

---

## 2. Methods for accurate poses on short RGB-D sequences

Our context: 150 frames or so, 256x192 depth, 1920x1440 RGB, a plane-dominated scene, no CUDA locally (Apple Silicon); a Bouchet GPU cluster is
available for offline runs.

| Method | Uses | Expected accuracy (published) | Flat-table robustness | Mac runtime / integration |
|---|---|---|---|---|
| **Colored ICP** (Park, Zhou, Koltun, ICCV 2017) | geometry + colour, multi-scale | Better than geometry-only and prior colour ICP in the paper; no planar-scene numbers found | Paper states plain ICP "tends to become unstable over smooth surfaces" and colour resolves the ambiguity **[V]**. Needs colour variation on the plane, so a plain table gives little help | Already in Open3D (CPU). Seconds per pair. Already being tested |
| **Open3D reconstruction system** (fragments -> pose-graph -> loop closure -> TSDF) | RGB-D odometry + ICP, pose graph | No published small-scene numbers found | Odometry is photometric+geometric; fragments add local BA-like robustness **[S]** | CPU, Python scripts; the tutorial is for PrimeSense-like data, needs a config. 1-2 days |
| **RGB-D odometry / DVO** (Kerl, Sturm, Cremers, ICRA/IROS 2013) | all pixels, photometric+depth error, keyframes, loop closure | Reported as more accurate than earlier methods on TUM RGB-D; exact numbers not retrieved | Photometric term constrains in-plane motion when the plane has texture. Open3D has `compute_rgbd_odometry` (hybrid term) **[U on exact API]** | Real-time on one CPU core per authors **[V]**. Open3D implements it, so this is the cheapest step |
| **BundleFusion** (Dai et al. 2017) | sparse features + dense photometric/geometric, global re-optimisation | ATE ~1-2 cm on TUM sequences (1.6 cm fr1/desk, 1.1 fr2/xyz, 2.2 fr3/office) **[S]** [RTAB-Map paper](https://arxiv.org/pdf/2403.06341) | Good in principle (feature + dense) | Needs CUDA and Windows; not usable on a Mac. Use only as a reference design |
| **ORB-SLAM3 RGB-D** | ORB features + depth, BA, loop closing | Very accurate (~1 cm) on TUM RGB-D when it tracks **[S]** [Robust SLAM study](https://arxiv.org/pdf/2109.13160); fails to initialise or track on short or low-texture runs in some tables **[S]** [DROID-SLAM paper](https://arxiv.org/pdf/2108.10869) | Weak on a textureless table; the food and plate supply some features | CPU, compiles on macOS with effort (Pangolin, Eigen, OpenCV). 10 s sequence may be too short for init. 2-3 days **[U]** |
| **DROID-SLAM** | learned dense flow + DBA, RGB-D mode | 3.8 cm avg ATE over TUM fr1 (monocular) **[V]**; strong on hard cases | Dense learned flow handles low texture better than ORB **[U]** | Needs CUDA (custom kernels) -> cluster only, or the community MPS ports **[U]**. Overkill for ~150 frames |
| **hloc + COLMAP / GLOMAP** | SuperPoint+LightGlue, SfM | GLOMAP "on par or superior" to COLMAP, "orders of magnitude" faster **[V]** [GLOMAP, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5646_ECCV_2024_paper.php) | Learned matches help on weak texture, but a bare table still gives few points. We measured plain COLMAP BA = 49 cm on the runaway scan | CPU for COLMAP/GLOMAP; SuperPoint/LightGlue run on CPU or MPS **[U]**. No metric scale, so align with ARKit or depth afterwards |
| **DUSt3R / MASt3R** | pairwise point-maps | Pairwise only; global alignment is slow | n/a | Heavy; superseded for our use by MASt3R-SfM or VGGT |
| **MASt3R-SfM** | MASt3R matches + retrieval + low-memory global alignment | Outperforms existing methods in small and medium settings **[V abstract]** [arXiv 2409.19152](https://arxiv.org/abs/2409.19152) | Designed for weak overlap and little motion **[V abstract]** | GPU; cluster. About 1 day to wire up **[U]** |
| **MASt3R-SLAM** | MASt3R priors, real-time dense SLAM | 3.0 cm (calibrated) / 6.0 cm (uncalibrated) avg ATE on TUM RGB-D **[S]** [arXiv 2412.12392](https://arxiv.org/abs/2412.12392) | Works on low-texture, no planar-specific claims verified | Needs a CUDA GPU **[U]** |
| **VGGT** (CVPR 2025) | feed-forward cameras, depth, points | "under one second" for reconstruction **[V]** [arXiv 2503.11651](https://arxiv.org/abs/2503.11651); best ATE among the feed-forward group on Sintel (~0.17) and Bonn (~0.013) in E3D-Bench **[S]** [E3D-Bench](https://arxiv.org/pdf/2506.01933) | Unknown for a planar tabletop at mm level **[U]** | Weights are large; a community MPS port exists but is unverified **[S]** (VGGT-MPS). Cluster is safer |
| **Spann3R / CUT3R** | streaming feed-forward | CUT3R about second behind VGGT; Spann3R generally lowest in E3D-Bench **[S]** | Unknown | GPU |
| **VGGT-SLAM 1/2, VGGT-SLAM++** | VGGT submaps + SL(4) alignment | ~5.3 / 4.1 cm ATE on TUM (2.0 figure from a secondary site) **[S]** [VGGT-SLAM](https://arxiv.org/pdf/2505.12549), [2.0](https://arxiv.org/html/2601.19887v1) | Unknown | GPU |
| **MapAnything** (Meta, 3DV 2026) | feed-forward, accepts optional intrinsics, poses, **depth** as input, outputs metric scene + cameras | Reported to match or beat specialists on 12+ tasks **[S]** [arXiv 2509.13414](https://arxiv.org/abs/2509.13414) | Taking our LiDAR depth and intrinsics as input is the right interface; not tested on planar scenes **[U]** | GPU; cluster. Most interesting 2025 option for us |

Notes on the table:

- **Accuracy scale mismatch.** Published ATE numbers are cm-level on room-scale TUM sequences. Our need is mm-level
  consistency on a 30 cm scene, so none of these numbers transfers directly. Treat them as relative rankings only.
- **How each handles the flat table.** Depth-only point-to-plane ICP is degenerate in the 2 in-plane translations and the rotation
  about the plane normal. Fixes in the literature fall into three groups:
  1. **Photometric terms** (DVO, colored ICP, BundleFusion's dense term): only help if the table has texture. A printed mat
     or a board adds it.
  2. **Sparse features with depth** (ORB-SLAM3, BundleFusion's sparse stage, our `features` path): the plate pattern, food
     and table texture constrain in-plane motion.
  3. **Plane constraints**: these *remove* the plane-normal DOFs (height, tilt) rather than the in-plane ones, so they do not
     fix the slide. Plane-only systems are degenerate with fewer than three non-parallel normals
     **[V]** [Ataer-Cansizoglu et al., ICCV-W 2013](https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/papers/Ataer-Cansizoglu_Tracking_an_RGB-D_2013_ICCV_paper.pdf);
     adding IMU or point/line features is the standard remedy
     **[V]** [Dense Planar-Inertial SLAM](https://frc.ri.cmu.edu/~kaess/pub/Hsiao18icra.html). We already have an IMU-fused
     pose (ARKit), so using ARKit rotation as a prior and constraining only translation is the analogous move.
  4. **Our object is not planar.** The plate, food and any items on the table are 3D. Masking out the table and registering on
     the plate region alone may give ICP real constraints. **[U: untested idea, but cheap.]**

---

## 3. Fiducial approach

### 3.1 What I could verify

- Sub-millimetre tracking at up to ~1 m for ArUco/AprilTag in a KIT comparison **[S, abstract only]**
  [KIT publication 1000092221](https://publikationen.bibliothek.kit.edu/1000092221).
- Rotation error below 0.1 deg average for AprilTag in a TopoTag robot sequence **[S]** [TopoTag](https://arxiv.org/pdf/1908.01450).
- Under partial occlusion, single-marker accuracy collapses: at 10% occlusion ArUco reached 18.8 deg and 91 mm error in
  one 2026 preprint **[S]** [BlobBoards](https://arxiv.org/pdf/2608.28830). **Occlusion by the plate and the user's hand is the real
  risk**, which argues for a *board* (many markers), not a single tag.
- ChArUco corners are more accurate than ArUco corners alone (roughly 0.3 px vs 0.6 px on an OpenCV forum report **[S]**)
  [OpenCV forum](https://forum.opencv.org/t/problem-with-charuco-calibration/18391). Low reprojection error
  does not guarantee low metric error (a forum user saw 0.05 px reprojection yet mm-level offsets) **[S]**
  [RoboDK forum](https://robodk.com/forum/Thread-OpenCv-CharUco-Calibration). A warped board gives bad poses
  **[S]** [OpenCV forum](https://forum.opencv.org/t/why-does-this-charuco-pose-estimation-fail-for-this-particular-image-and-how-can-i-avoid-it/13682); mount on rigid board.
- Use of marker boards under the object for camera poses has precedent in object datasets: a Kinect Azure dataset places the object on a
  7x5 ArUco board and gets poses from OpenCV **[S]**
  [Mendeley dataset](https://data.mendeley.com/datasets/jd8w5r3ncw/2). Food-dosimetry work has long used checkerboards or
  reference cards for scale **[S]** (Purdue thesis and Michigan Tech dissertation hits; not opened).
- I found **no** head-to-head turntable vs handheld accuracy numbers. A turntable with a stationary camera removes handheld motion
  blur and VIO risk but needs the markers on the rotating platform. Whether this is used for food volume at scale is **[U]**.

### 3.2 Practical implications for us

- **Board under the plate** keeps markers visible except where the plate covers them. Put them on a ring or a frame around the plate. A
  full-page ChArUco with at least 20 markers should survive the hand and plate occlusions. Compute pose with
  `cv2.aruco` ChArUco interpolation + `solvePnP` and use the 1920x1440 RGB intrinsics (ARKit supplies them per frame).
- **Metric scale** then comes from the printed board size. Verify print scale with a ruler (printers distort by ~0.5-1% **[U]**).
- **Fusion**: use marker-derived pose where enough corners are visible (e.g. at least 8 ChArUco corners), and fall back to
  ARKit or ICP between. Pose-graph with robust loss.
- **Handheld vs turntable**: keep handheld (matches the real deployment). A turntable would be a valid validation setup for the
  thesis (ground-truth-like poses), not a product path.
- Cost: user must carry a printed board. It is a legitimate limitation to state, and a good *ground-truth source* for
  evaluating the markerless methods even if the final product has no board. This is the main use I would push: **record
  scans with a board to build a pose ground truth, then score every markerless method against it.**

---

## 4. Food-specific related work

### 4.1 What I found

| Work | Sensor | Poses | Reported error | Source |
|---|---|---|---|---|
| Nutrition5k (Thames et al., CVPR 2021) | overhead RGB-D (Intel RealSense) + side-angle video | fixed rig, no pose problem | RGB-D direct calorie MAE 18.8%, macronutrient 20.9% **[V]**; volume error not retrieved | [arXiv 2103.03375](https://arxiv.org/pdf/2103.03375) |
| DPF-Nutrition (follow-up on Nutrition5k) | monocular + predicted depth | n/a | PMAE: calorie 14.7%, mass 10.6%, protein 20.2%, fat 22.6%, carb 20.7% **[S]** | listed in the search result for Nutrition5k |
| Depth-sensing iPhone X preclinical study | structured-light depth | single/few views | Relative absolute weight error 14.0%, energy and macros 12-15% **[S]** | [Univ. Augsburg record](https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/docId/124680) |
| VolE (Sci. Rep. 2026) | ARCore/ARKit images + camera poses, free motion, mesh reconstruction | **ARKit/ARCore poses** (our exact situation) | Error figures not retrieved (page needs cookie redirect) **[U]** | [Nature](https://www.nature.com/articles/s41598-026-38756-5) |
| "Mobile Food Calorie Estimation Using Smartphone LiDAR Sensor" (Springer 2025, Yanai lab context) | iPhone LiDAR | not retrieved | not retrieved; the ResearchGate entry mentions <6% energy error under ideal segmentation, but that abstract may describe a two-camera sensor, so unverified **[U]** | [Springer](https://link.springer.com/chapter/10.1007/978-981-95-4398-4_10), [ResearchGate](https://www.researchgate.net/publication/397436194_Mobile_Food_Calorie_Estimation_Using_Smartphone_LiDAR_Sensor) |
| Review "Food Portion Estimation: From Pixels to Calories" (2026) | review | n/a | Notes mobile LiDAR resolution is often insufficient for dietary assessment, and granular foods (rice, nuts) overestimate volume because gaps are counted as solid **[S via search summary]** | [arXiv 2602.05078](https://arxiv.org/pdf/2602.05078) |
| Monocular 3D-scaling (2024) | RGB only | n/a | 17.67% mean calorie error **[S]** | [arXiv 2404.12257](https://arxiv.org/pdf/2404.12257) |
| Older multi-view / marker methods | phone + checkerboard / reference card | marker-based | one journal paper reports volume error under 10% (setup unclear) **[S]** | Purdue / Michigan Tech hits from search; not opened |

**Gap to flag in the thesis:** I could not find a peer-reviewed study with iPhone/iPad LiDAR, ARKit poses, and real meal
volume error with error bars. Our work would sit in that gap, and the closest published anchor is the 12-15% error from a
depth-sensing phone and the 10-18% calorie errors from RGB-D learning on Nutrition5k. Do a proper Scholar pass before claiming novelty
(Google Scholar and the full texts of VolE and the Springer chapter are the first things to read).

### 4.2 Benchmarks for "good enough"

- Published food systems achieve roughly 10-15% volume or energy error, so **5-10% volume error would be competitive**,
  and under 5% would be the best I know of, **[U]** on the "best" claim.
- Separate pose error from other error sources: segmentation, density (kcal/g), occluded base, and LiDAR bias on specular or wet food
  often exceed pose error. Report a pose-error budget in the thesis.

### 4.3 Pose accuracy needed for 5-10% volume error (my own estimate, not sourced)

Treat the fused surface as a shell. If frames are mutually misaligned by roughly sigma, the fused top surface becomes
thicker or offset by about sigma. Example: a 20 cm plate with 600 mL of food, 300 cm^2 top area:

- sigma = 3 mm gives ~90 mL, so about 15% error;
- sigma = 1.5 mm gives about 7%;
- sigma = 1 mm gives about 5%.

Rotation: 0.5 deg at 0.4 m range is ~3.5 mm at the far edge of the scene, so you need under about 0.3 deg inter-frame
consistency. **Global drift matters less** than frame-to-frame consistency: a smooth 9 cm loop-closure error over a 1 m arc may
mostly shear the volume rather than inflate it, but a 5-10 cm in-plane jitter (what we have now) will blur it by well over
5 mm. So the current 5-10 cm jitter is not good enough, and the target is at least 10x better. This calculation is a
first-order estimate **[U]** and should be tested with synthetic jitter on a known-volume reconstruction (the existing
`results/synth` set is a natural place).

---

## 5. Recommendation: ranked plan

Effort is for one person who already has the pipeline (depth_odometry.py, refine_poses.py).

| Rank | Step | Effort | Target pose accuracy | Why / risk |
|---|---|---|---|---|
| 0 | **Measure first.** Build a pose evaluation: ChArUco board scans give reference poses (step 1), plus a synthetic-jitter-to-volume test | 1 day (after step 1) | gives the numbers below | Without ground truth we are comparing ICP variants by eye. Also check iOS version of runaway scans against the 26.4 regression |
| 1 | **ChArUco board capture + PnP poses** on a few scans, used as ground truth, and as the production pose source if acceptable | 1-2 days | 1-3 mm, under 0.5 deg **[U]** | Highest accuracy for the least code. Risk: occlusion, board flatness, user burden |
| 2 | **Fix the flat-plane degeneracy cheaply**: keep ARKit rotation as prior, restrict ICP to translation, mask table and register on plate + food; add RGB-D photometric term (Open3D `compute_rgbd_odometry` hybrid, or our colored ICP) | 1-2 days | 5-10 mm | Directly attacks the slide; no new dependencies |
| 3 | **Feature odometry with depth (3D-3D) + windowed BA / pose graph**, using SuperPoint+LightGlue instead of ORB/SIFT; ARKit as initial guess; robust loss; reject any ARKit frame whose relative motion disagrees with depth | 3-4 days | 3-5 mm **[U]** | Textured tabletop or printed mat helps a lot. COLMAP-alone failed, so keep depth in the loop |
| 4 | **Open3D reconstruction pipeline** (fragments + pose graph) to enforce loop-closure consistency over the short arc | 2 days | 3-5 mm | Mostly a global consistency polish on top of 2-3 |
| 5 | **Feed-forward check on cluster**: MapAnything (takes depth + intrinsics) and VGGT / MASt3R-SfM, scored against board ground truth | 2-3 days | unknown; if under 5 mm adopt as a robust fallback | Highest uncertainty, but a good 2025-26 thesis comparison. Needs GPU |
| 6 | ORB-SLAM3 / DROID-SLAM / BundleFusion | skip unless 1-5 fail | n/a | Build pain on Mac (ORB-SLAM3), CUDA only (DROID, BundleFusion); the 10 s sequence is short for SLAM initialisation |

Suggested thesis framing: ARKit poses (baseline) -> depth ICP (runaway fixed, in-plane jitter) -> photometric/feature
refinements -> marker ground truth, with volume error vs pose error as the headline plot.

Runaway detection should stay in place regardless: compare ARKit relative motion with depth ICP and with IMU-integrated
displacement, and flag the scan if the two disagree beyond a threshold (the existing 2 cm / 3 deg gate).

---

## References

- Apple, Understanding World Tracking: https://developer.apple.com/documentation/arkit/understanding-world-tracking
- Apple forums, tracking state: https://developer.apple.com/forums/thread/699163
- Apple forums, ARKit drift regression on LiDAR devices (iOS 26.4+): https://developer.apple.com/forums/thread/827240
- Apple forums, transform drift: https://developer.apple.com/forums/thread/788370
- Apple forums, LiDAR depth accuracy: https://developer.apple.com/forums/thread/650321
- Kim et al., VIO benchmark (Sensors 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC9785098
- ISPRS Archives 2025, iPhone LiDAR stockpile volume: https://isprs-archives.copernicus.org/articles/XLVIII-M-10-2025/91/2026/
- SAE 2021-01-0891, iOS LiDAR bias: https://saemobilus.sae.org/content/2021-01-0891/
- Park, Zhou, Koltun, Colored Point Cloud Registration Revisited (ICCV 2017): https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf
- Open3D colored ICP tutorial: https://www.open3d.org/html/tutorial/pipelines/colored_pointcloud_registration.html
- Open3D reconstruction system: https://www.open3d.org/docs/latest/tutorial/reconstruction_system/make_fragments.html and https://arxiv.org/pdf/1801.09847
- Kerl, Sturm, Cremers, robust RGB-D odometry (ICRA 2013) and DVO code: https://cvg.cit.tum.de/data/software/dvo
- DROID-SLAM: https://arxiv.org/pdf/2108.10869
- Robust SLAM Systems study: https://arxiv.org/pdf/2109.13160
- RTAB-Map paper (BundleFusion numbers): https://arxiv.org/pdf/2403.06341
- GLOMAP: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5646_ECCV_2024_paper.php , code https://github.com/colmap/glomap
- hloc: https://github.com/cvg/Hierarchical-Localization (not fetched), LightGlue: https://github.com/cvg/LightGlue (not fetched)
- MASt3R-SfM: https://arxiv.org/abs/2409.19152
- MASt3R-SLAM: https://arxiv.org/abs/2412.12392
- VGGT: https://arxiv.org/abs/2503.11651
- VGGT-SLAM: https://arxiv.org/pdf/2505.12549 ; 2.0: https://arxiv.org/html/2601.19887v1 ; ++: https://arxiv.org/pdf/2604.06830
- MapAnything: https://arxiv.org/abs/2509.13414
- E3D-Bench: https://arxiv.org/pdf/2506.01933
- Light3R-SfM: https://arxiv.org/pdf/2501.14914
- Plane degeneracy: https://openaccess.thecvf.com/content_iccv_workshops_2013/W02/papers/Ataer-Cansizoglu_Tracking_an_RGB-D_2013_ICCV_paper.pdf ; Dense Planar-Inertial SLAM: https://frc.ri.cmu.edu/~kaess/pub/Hsiao18icra.html
- ArUco/AprilTag accuracy: https://publikationen.bibliothek.kit.edu/1000092221 ; TopoTag https://arxiv.org/pdf/1908.01450 ; BlobBoards https://arxiv.org/pdf/2608.28830 ; OpenCV ArUco docs https://docs.opencv.org/5.0/main_modules/objdetect_aruco.html
- ChArUco forum reports: https://forum.opencv.org/t/problem-with-charuco-calibration/18391 , https://forum.opencv.org/t/why-does-this-charuco-pose-estimation-fail-for-this-particular-image-and-how-can-i-avoid-it/13682 , https://robodk.com/forum/Thread-OpenCv-CharUco-Calibration
- Kinect Azure ArUco-board dataset: https://data.mendeley.com/datasets/jd8w5r3ncw/2
- Nutrition5k: https://arxiv.org/pdf/2103.03375
- Depth-sensing iPhone X food study: https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/docId/124680
- VolE: https://www.nature.com/articles/s41598-026-38756-5
- Mobile Food Calorie Estimation Using Smartphone LiDAR: https://link.springer.com/chapter/10.1007/978-981-95-4398-4_10
- Food Portion Estimation: From Pixels to Calories: https://arxiv.org/pdf/2602.05078
- Food Portion Estimation via 3D Object Scaling: https://arxiv.org/pdf/2404.12257
- Dietary assessment review: https://arxiv.org/pdf/2108.02947

## Not verified / open items

- Apple's numeric capture guidance (lighting lux, speed, distance): none found.
- Nutrition5k volume-specific error, VolE errors, and the Springer LiDAR calorie chapter's numbers: pages not readable.
- Runtime on Apple Silicon for every learned method, and whether VGGT/MapAnything fit in unified memory: not measured, no source.
- Whether any method has been benchmarked on tabletop plane-dominated RGB-D at mm scale: none found.
- Expected ChArUco board accuracy of 1-3 mm: my estimate, to be measured in step 0/1.
- Hloc, LightGlue, and Pi3 papers and repositories were not opened.
