# Learned and photogrammetric reconstruction from mobile RGB + low-res LiDAR (small objects, 5-30 cm)

Pipeline context: ARKit poses, 1920x1440 RGB, 256x192 LiDAR depth + confidence, local 6 GB RTX 2060, A40/L40S cluster. Notes compiled 2026-10-13. Most benchmark numbers are room-scale (ARKitScenes, ScanNet++, MuSHRoom) or DTU. I found no benchmark built specifically for 5-30 cm objects scanned with phone LiDAR. The closest are the food datasets (MetaFood/MTF, Foodkit).

## Depth completion / upsampling with LiDAR prompts (PromptDA and alternatives, temporal consistency)

### Takeaway
Prompt Depth Anything is the clear fit for our data. It takes ARKit 256x192 LiDAR depth plus RGB and returns metric depth at full RGB resolution. On ARKitScenes it roughly halves the L1 error of raw ARKit depth, and on ScanNet++ iPhone data it improves TSDF reconstruction F-score from 0.68 to 0.76. It is Apache-2.0 and has a 25M-parameter small model that should fit the RTX 2060. Its documented failure modes are temporal flicker inherited from the LiDAR and noise at long range. Monocular metric models (Depth Pro, Metric3D v2, Depth Anything V2) are much worse than raw LiDAR on these benchmarks.

### Cited Findings
- PromptDA (CVPR 2025) uses low-cost LiDAR as a "prompt" to Depth Anything, with multi-scale fusion of the LiDAR inside the DPT decoder. It is trained on synthetic data with simulated LiDAR plus real data with pseudo-GT depth, and outputs up to 4K resolution — [arXiv 2412.14015](https://arxiv.org/abs/2412.14015)
- ARKitScenes depth (L1 / RMSE, metres):
  - PromptDA at 768x1024: 0.0132 / 0.0315. At 1440x1920: 0.0138 / 0.0316. Synthetic-only zero-shot: 0.0161 / 0.0376.
  - Raw ARKit depth: 0.0251 / 0.0424. Depth Prompting (D.P.): 0.0251 / 0.0422. BPNet: 0.1494 / 0.2106.
  - Monocular models: Depth Pro (fine-tuned, no LiDAR) 0.0437 / 0.0672. Depth Anything V2 with RANSAC alignment 0.0716 / 0.1686. Metric3D v2 with RANSAC 0.0626 / 0.2104.
  - Caveat: the extracted table was partly garbled, so some pairings may be off — [arXiv HTML 2412.14015](https://arxiv.org/html/2412.14015)
- ScanNet++ iPhone depth and TSDF reconstruction:

  | Method | L1 | AbsRel | Acc | Comp | F-score |
  |---|---|---|---|---|---|
  | PromptDA | 0.0250 | 0.0175 | 0.0699 | 0.0616 | 0.7619 |
  | ARKit depth | 0.0351 | 0.0241 | 0.0811 | 0.0743 | 0.6785 |
  | Metric3D v2 | 0.0585 | — | — | — | 0.6515 |
  | Depth Pro | 0.0638 | — | — | — | 0.6187 |

  — [arXiv HTML 2412.14015](https://arxiv.org/html/2412.14015)
- Ablations on ARKitScenes L1 / ScanNet++ F-score:
  - Full model: 0.0132 / 0.7619.
  - No prompting: 0.0605 / 0.5696.
  - No foundation-model initialisation: 0.0194 / 0.7077.
  - ControlNet-style prompting: 0.0239 / 0.6899.
  - Prompt fusion adds only about 5.7% FLOPs — [arXiv HTML 2412.14015](https://arxiv.org/html/2412.14015)
- Runtime on an A100:
  - ViT-L: 20.4 FPS at 768x1024, 2.0 FPS at 2160x3840.
  - ViT-S: 80 FPS at 768x1024, 10.3 FPS at 2160x3840 — [arXiv HTML 2412.14015](https://arxiv.org/html/2412.14015)
- Limitations stated by the authors:
  - iPhone LiDAR is noisy at long range, so long-range depth is poor.
  - LiDAR temporal flicker propagates into flickering predicted depth. They propose temporal prompt tuning as future work — [arXiv HTML 2412.14015](https://arxiv.org/html/2412.14015)
- Code, from the repo:
  - Apache-2.0 licence, about 1.2k stars.
  - Models: Large (340M), Small (25.1M) and Small-Transparent (25.1M, fine-tuned on HAMMER with simulated iPhone LiDAR). Only Large is benchmarked in the paper.
  - Input example is ARKit LiDAR at 192x256 in metres. Output is at the RGB HxW.
  - Includes Stray Scanner inference and video scripts that write "_smooth" outputs. How that smoothing works is undocumented.
  - "PromptDA++" is announced, with code to follow — [GitHub DepthAnything/PromptDA](https://github.com/DepthAnything/PromptDA)
- Video Depth Anything (CVPR 2025) replaces the Depth Anything V2 head with a spatio-temporal head and adds a temporal gradient-consistency loss. It targets minutes-long videos, and its smallest model runs at about 30 FPS — [CVPR 2025 poster](https://cvpr.thecvf.com/virtual/2025/poster/33862), [HF papers 2501.12375](https://huggingface.co/papers/2501.12375)
- An informal CMU student study found the metric Video Depth Anything beat single-frame metric depth on geometry but was too costly for edge use — [CMU MSCV project](https://mscvprojects.ri.cmu.edu/2026teamf12/?p=27)
- In DN-Splatter's ablation (below), supervising Gaussian splatting with iPhone sensor depth beats ZoeDepth and Metric3D monocular depth by a wide margin: F-score 0.583 vs about 0.14 — [arXiv HTML 2403.17822](https://arxiv.org/html/2403.17822)

### Inferences
- For our pipeline, PromptDA-Small (or Large on the cluster) should be the default upsampler from 256x192 to 1440x1920. Its ScanNet++ result shows that it improves TSDF fusion, not just per-frame metrics.
- Per-frame flicker matters less once frames are fused with ARKit poses (TSDF or GS), but confidence-masking the prompt (dropping low-confidence LiDAR pixels before prompting) is worth testing.
- The large error gap makes monocular metric models (Depth Pro, UniDepth, Metric3D v2, MoGe-2, DA V2/V3 metric) poor substitutes when LiDAR exists. They are mainly useful as normal or relative-depth priors.
- Small objects at 20-50 cm sit in the LiDAR's good range. Thin edges and food rims are where we expect PromptDA's RGB-guided upsampling to help most. This is not verified on food.

### Gaps
- I did not retrieve benchmark numbers for UniDepth, MoGe-2, Depth Anything 3 metric, or DepthSplat on ARKit data.
- PromptDA's VRAM use on a 6 GB card is not reported. Small (25M) should fit, but this is unverified.
- No PromptDA evaluation exists on close-range tabletop objects or food. Specular, transparent and dark sauces are untested, apart from the transparent-object variant.

## Gaussian splatting / surface reconstruction with depth priors (2DGS, PGSR, GOF, RaDe-GS, DN-Splatter, AGS-Mesh, MonoSDF, Neuralangelo)

### Takeaway
On iPhone-captured data (MuSHRoom, ScanNet++), supervising with sensor depth is the dominant factor. DN-Splatter and AGS-Mesh, which add depth and normal priors to 3DGS/2DGS, cut Chamfer about 3x versus vanilla 2DGS, SuGaR or GOF, and match or beat MonoSDF in far less training time. Pure photometric surface methods (PGSR, 2DGS, GOF) lead on DTU but degrade badly on phone captures.

### Cited Findings
- MuSHRoom mesh results (average of 6 scenes; Acc / Comp / Chamfer-L1 / NC / F-score):
  - DN-Splatter, with sensor depth: .0239 / .0194 / .0216 / .8822 / .9243.
  - MonoSDF, with depth: .0310 / .0190 / .0250 / .8846 / .9211.
  - Poisson: .0399 / .0222 / .0306 / .8688 / .8823.
  - Depth-Nerfacto: C-L1 .0502.
  - SuGaR: C-L1 .0620. 2DGS: C-L1 .0687. Splatfacto: C-L1 .0652.
  - Source: [DN-Splatter arXiv HTML](https://arxiv.org/html/2403.17822)
- ScanNet++ (2 scenes, RTX 4090), Chamfer-L1 and training time:
  - DN-Splatter: .0667, 36.9 min.
  - MonoSDF: .0438, 47.5 min.
  - 2DGS: .1035, 33.5 min.
  - SuGaR: .0975, 70 min.
  - Source: [DN-Splatter arXiv HTML](https://arxiv.org/html/2403.17822)
- DN-Splatter depth-supervision ablation (ScanNet++ scene b20a261fdf; Chamfer-L1 / F-score):

  | Supervision | Chamfer-L1 | F-score |
  |---|---|---|
  | None | .2359 | .1343 |
  | ZoeDepth | .1918 | .1455 |
  | Metric3D | .1938 | .1439 |
  | MVSNet | .2748 | .1903 |
  | Pearson loss with Metric3D | .1474 | .2236 |
  | iPhone sensor depth | .1021 | .5833 |

  — [DN-Splatter arXiv HTML](https://arxiv.org/html/2403.17822)
- On the MuSHRoom VR room, adding a depth loss to Splatfacto moves C-L1 from 7.0 to 2.6 and F1 from 56.3 to 86.6. Adding normals as well gives 2.4 / 87.7, and training time rises from 12.9 to 17.5 min — [DN-Splatter arXiv HTML](https://arxiv.org/html/2403.17822)
- DN-Splatter uses an adaptive depth loss weighted by RGB image gradients, plus normal priors and smoothness. It was published at WACV 2025 — [arXiv 2403.17822](https://arxiv.org/abs/2403.17822)
- AGS-Mesh (3DV 2025):
  - Uses the iPhone's 256x192 sensor depth and Omnidata normals.
  - Filters depth by depth-normal consistency: k-NN normals from depth are compared with Omnidata, and pixels off by more than 10° are dropped after step 7k. Normals are filtered adaptively the same way.
  - 30k iterations.
  - Source: [AGS-Mesh arXiv HTML](https://arxiv.org/html/2411.19271v2)
- AGS-Mesh MuSHRoom Chamfer-L1 / F-score:
  - Splatfacto + AGS (TSDF): .0209 / .9433.
  - 2DGS + AGS (IsoOctree): .0229 / .9146.
  - DN-Splatter (TSDF): .0215 / .9381.
  - GOF: .1277 / .4515.
  - Source: [AGS-Mesh arXiv HTML](https://arxiv.org/html/2411.19271v2)
- AGS-Mesh ScanNet++ Chamfer-L1 / F-score:
  - 2DGS + AGS: .0276-.0285 / about .903, 40.4 min.
  - DN-Splatter: .066.
  - GOF: .1187, 142 min.
  - Source: [AGS-Mesh arXiv HTML](https://arxiv.org/html/2411.19271v2)
- In AGS-Mesh's ablation, ZoeDepth monocular depth gives F1 .7224 versus .9051 for raw sensor depth on vr_room — [AGS-Mesh arXiv HTML](https://arxiv.org/html/2411.19271v2)
- The AGS-Mesh code lives in the nerfstudio-based dn-splatter repo, which supports MuSHRoom, ScanNet++ and Replica — [search summary: dn-splatter repo / Oulu repo](https://oulurepo.oulu.fi/handle/10024/63064)
- PGSR on DTU, mean Chamfer: 0.53 in the paper, 0.47 with Code V1.0, 0.52 without ICP. Training takes about 0.5-0.6 h per DTU scene. TnT F1 is 0.50-0.51. The repo has 1.1k stars — [GitHub zju3dv/PGSR](https://github.com/zju3dv/PGSR)
- 2DGS and GOF DTU Chamfer reported by the 2DGH paper: 0.823 and 0.804 under their setup. Papers use different DTU protocols, so numbers should not be mixed across tables — [2DGH arXiv 2408.16982](https://arxiv.org/pdf/2408.16982)
- VolE reports DTU mean Chamfer under one shared protocol:

  | Method | DTU Chamfer | Time per scene (RTX 3090) |
  |---|---|---|
  | VolE (MVS) | 0.68 | 3.5 min |
  | NeuS2 | 0.70 | 5 min |
  | NeuS | 0.77 | 8 h |
  | COLMAP | 1.36 | 1 h |

  — [VolE arXiv HTML](https://arxiv.org/html/2505.10205v1)

### Inferences
- For our small objects, the best-evidenced recipe is DN-Splatter or AGS-Mesh-style GS: ARKit poses, depth loss from LiDAR (or PromptDA-upsampled depth), normal prior, depth-normal-consistency filtering, then TSDF or Poisson meshing. Both are nerfstudio-based. Room-scale training took about 37-40 min on a 4090; a masked tabletop object should be cheaper. That is plausible on the 2060, but a better fit for the A40/L40S.
- Depth supervision helps far more on phone captures than on DTU. DN-Splatter's F-score goes from 0.13 with no depth to 0.58 with sensor depth on one scene.
- I found no direct comparison of PromptDA-upsampled depth versus raw 256x192 depth as GS supervision. This is a natural experiment for us.

### Gaps
- I did not retrieve RaDe-GS or Neuralangelo numbers, licence or stars. MonoSDF numbers came only as DN-Splatter baselines.
- No paper benchmarks these methods on 5-30 cm objects with phone LiDAR. MuSHRoom and ScanNet++ are rooms, where 2 cm Chamfer is roughly 10% of a food item's size.
- Licences and last-update dates for the DN-Splatter and PGSR repos were not confirmed.

## Feed-forward 3D (VGGT, MASt3R/MASt3R-SfM, DUSt3R, MapAnything, π3)

### Takeaway
Feed-forward models are fast, but on DTU they are still around 0.4-1.0 mm Chamfer behind optimisation methods, with large sensitivity to pose input. MapAnything is the only one found that natively accepts intrinsics, poses and depth and outputs metric geometry, making it the natural candidate for injecting ARKit poses and LiDAR.

### Cited Findings
- MapAnything takes images plus optional intrinsics, poses, depth or partial reconstructions. It regresses metric geometry and cameras in one pass using a factored representation: depth, ray maps, poses and a metric scale factor. The paper is CC BY 4.0 — [arXiv 2509.13414](https://arxiv.org/abs/2509.13414)
- A 2026 DTU uncertainty study reports these mean Chamfer distances, but warns that scene subsets and hardware differ between entries:
  - VGGT: about 0.97 mm without pose input, about 0.59 mm with pose input.
  - MASt3R: 0.37 mm with known camera orientation.
  - DUSt3R: 1.74 mm.
  - Source: [arXiv 2606.16479](https://arxiv.org/pdf/2606.16479)
- Depth Anything 3 claims average geometry gains of 25.1% over VGGT and 21.5% over π3 — [arXiv 2511.10647](https://arxiv.org/pdf/2511.10647)
- Pi3X is a metric variant of π3 that predicts metric depth without scale alignment and supports injecting priors — [search summary citing arXiv 2605.27367/related](https://arxiv.org/pdf/2605.27367)

### Inferences
- Since we already have ARKit poses and metric LiDAR, feed-forward models add value mainly as dense initialisers or fallbacks, for example for poorly textured food. Their main strength, recovering unknown poses, is redundant for us.
- The VGGT pose-input gain (0.97 to 0.59 mm) suggests that conditioning helps substantially. MapAnything with ARKit poses and LiDAR is the one to test.

### Gaps
- I found no study of VGGT, π3 or MapAnything on 5-30 cm objects with metric depth injection.
- I did not retrieve VRAM figures. Feed-forward ViT-L models on many 518-px frames likely exceed 6 GB, which would mean cluster-only use, but this is not verified.

## Apple Object Capture (ObjectCaptureSession / PhotogrammetrySession)

### Takeaway
Object Capture on iOS 17+ records RGB plus LiDAR points (fused into a point cloud that helps low-texture objects). On-device reconstruction is limited to the "reduced" detail level. Medium, full, raw and custom levels need a Mac. Apple publishes no quantitative accuracy figures, and the API estimates its own poses: I found no way to inject ARKit poses.

### Cited Findings
- WWDC23 session 10191:
  - On supported devices, the session collects LiDAR points from multiple views and fuses them into a point cloud, improving coverage on low-texture objects.
  - iOS supports only the reduced detail level, producing USDZ with diffuse, AO and normal maps. On-device reconstruction takes "a few minutes."
  - macOS supports reduced, medium, full, raw and a new custom level (control of decimation and textures).
  - `isOverCaptureEnabled` saves extra images for Mac reconstruction.
  - `.poses` can be requested as an output, alongside `.modelFile`.
  - Source: [WWDC23 10191](https://developer.apple.com/videos/play/wwdc2023/10191/)
- Apple's stated limitations:
  - Avoid reflective, transparent or very thin objects.
  - Don't flip deformable objects.
  - Flipping textureless objects is hard.
  - Use diffuse lighting.
  - Three scan passes are recommended.
  - Device support starts at iPhone 12 Pro / iPad Pro 2021.
  - No accuracy claims are made.
  - Source: [WWDC23 10191](https://developer.apple.com/videos/play/wwdc2023/10191/)
- The PhotogrammetrySample doc page did not render for scraping — [Apple docs](https://developer.apple.com/documentation/realitykit/photogrammetrysample). From prior knowledge only, unverified, the documented fields are `image`, `depthDataMap`, `objectMask`, `gravity`, `metadata` and `id`, with no camera pose or transform field.

### Inferences
- Poses are an output (`.poses`), not an input, so Object Capture cannot consume our ARKit trajectory. It also runs its own SfM, which can fail on textureless food.
- Food is often deformable, specular (sauces) or thin (leaves). These match Apple's own "avoid" list, so Object Capture is a reasonable baseline but likely not the best path.
- A custom-sample route (PhotogrammetrySession fed PhotogrammetrySamples with our LiDAR depth and SAM masks as objectMask) runs only on macOS, which is outside our Linux/CUDA stack.

### Gaps
- No published metric-accuracy numbers for Object Capture were found, and no independent benchmark on small objects was retrieved.
- The exact PhotogrammetrySample property list could not be verified from Apple's page. WWDC21 session 10076 was not fetched.

## Food-specific pipelines (VolE, VolETA, MetaFood challenge)

### Takeaway
Top food-volume methods all mask the food (SAM, XMem++ or FoodMem), reconstruct with NeuS2, 2DGS or MVS, then clean and close the mesh. The best reported MAPEs are about 3% (VolE on MTF) and about 11% (VolETA's MetaFood 2024 winning score). Depth sensors were barely used: scale came from checkerboards or ARKit poses.

### Cited Findings
- MetaFood CVPR 2024 challenge:
  - 20 MetaFood3D food items in three difficulty tiers: Easy (~200 frames), Medium (30 images), Hard (1 image).
  - A depth image was provided per frame, and a checkerboard or pattern mat gave scale.
  - Phase I scored volume MAPE. Phase II scored Chamfer.
  - 16 teams took part.
  - Source: [MetaFood report arXiv 2407.09285](https://arxiv.org/html/2407.09285v1)
- VolETA (1st place, University of Barcelona):
  - Keyframe selection removes blurry and duplicate frames, keeping about 34.8% of frames.
  - PixSfM for poses.
  - SAM click masks propagated with XMem++.
  - NeuS2 on RGBA inputs.
  - Removes isolated pieces below 5% of the diameter.
  - Manual checkerboard scale (12 mm squares), fine-tuned with depth and masks.
  - Single-view case: One-2-3-45.
  - MAPE 10.973%. The table was garbled, so it is unclear whether this is the mean or the sum.
  - Source: [MetaFood report](https://arxiv.org/html/2407.09285v1), [UB announcement](https://www.ub.edu/aiba/?p=733)
- ININ-VIAUN (2nd place):
  - COLMAP with DiffusioNeRF / NeRF2Mesh.
  - Automatic checkerboard-corner scale.
  - MeshFix for holes, plus Laplacian smoothing.
  - Single-view case: ZeroNVS.
  - Source: [MetaFood report](https://arxiv.org/html/2407.09285v1)
- FoodRiddle (Best mesh):
  - COLMAP with SuperPoint and SuperGlue.
  - 2DGS, then TSDF to a dense point cloud.
  - Outliers removed.
  - Support-surface contour found, and bottom vertices projected onto the support plane.
  - Poisson reconstruction for a watertight mesh.
  - Average Chamfer 0.0032 m.
  - Source: [MetaFood report](https://arxiv.org/html/2407.09285v1)
- VolE (arXiv 2505.10205, same Barcelona group):
  - ARKit/ARCore poses refined with COLMAP.
  - FoodMem video masks (SeTR plus XMem++). Points are kept only if they fall inside the mask in all views.
  - OpenMVS-style Delaunay tetrahedralisation, graph-cut, then marching cubes. Uses a "close-holes 50" setting.
  - MAPE on MTF: VolE 3.08% (the abstract says 2.22%, an inconsistency), VolETA 7.84%, FoodRiddle 10.26%, ININ 14.47%.
  - MAPE on Foodkit: 1.22%.
  - About 5-9 min per capture on an RTX 3090.
  - Code URL hidden for review.
  - Source: [VolE arXiv HTML](https://arxiv.org/html/2505.10205v1), [abstract](https://arxiv.org/abs/2505.10205v1)
- VolE describes itself as reference- and depth-sensor-free, but the paper says MTF scenes used a reference board for scale — [VolE arXiv HTML](https://arxiv.org/html/2505.10205v1)

### Inferences
- ARKit pose scale plus mask-consistent MVS already reaches low single-digit MAPE in VolE. LiDAR depth supervision, via DN-Splatter-style GS or PromptDA-fused TSDF, should mainly help textureless, glossy or concave foods where MVS fails. This is plausible but not demonstrated.
- FoodRiddle's bottom closure (support plane, project bottom vertices, Poisson) is a concrete, citable recipe we can reuse.

### Gaps
- No food paper reports the gain from adding iPhone LiDAR depth to reconstruction.
- VolE's code availability is unconfirmed, and I did not fetch a 2025 MetaFood challenge report, if one exists.

## Practical: masking, plate/table handling, closing the bottom

### Takeaway
All strong food pipelines segment the food in 2D first and propagate the masks through video (SAM/XMem++ or FoodMem). They then drop 3D points outside the masks and close the underside against a fitted support plane before computing a watertight mesh's volume.

### Cited Findings
- Mask-then-reconstruct patterns:
  - VolETA: SAM click masks propagated with XMem++, then RGBA input to NeuS2 — [MetaFood report](https://arxiv.org/html/2407.09285v1)
  - VolE: FoodMem masks, keeping points inside the mask in all views — [VolE](https://arxiv.org/html/2505.10205v1)
- Bottom closure:
  - FoodRiddle detects the support-surface contour, projects bottom vertices onto it, then runs Poisson for a watertight mesh — [MetaFood report](https://arxiv.org/html/2407.09285v1)
  - ININ-VIAUN uses MeshFix — [MetaFood report](https://arxiv.org/html/2407.09285v1)
- Isolated-component removal (VolETA, 5% of diameter) and Laplacian smoothing (ININ) are used as cleanup — [MetaFood report](https://arxiv.org/html/2407.09285v1)
- AGS-Mesh's depth-normal-consistency filter (10° threshold) is a principled way to reject noisy LiDAR pixels, for example at grazing angles and rims, before GS supervision — [AGS-Mesh](https://arxiv.org/html/2411.19271v2)

### Inferences
- Suggested pipeline for us:
  1. SAM 2/3 masks propagated across frames.
  2. PromptDA-upsampled, confidence-filtered depth.
  3. Either TSDF fusion with ARKit poses (fast, 2060-feasible), or DN-Splatter/AGS-style 2DGS with masked depth and normal losses (cluster).
  4. Fit the table plane with RANSAC on unmasked LiDAR, then project or close the bottom to that plane (FoodRiddle).
  5. Poisson or watertight meshing, then volume.
- Plate handling likely needs a separate "plate" mask and plate-surface fitting, because the support plane under food on a plate is the plate surface, not the table. No cited paper covers this explicitly.

### Gaps
- I found no published method that explicitly handles plates or bowls (concave supports) when closing the bottom of food meshes.
- I did not retrieve SAM 2 or SAM 3 masking-accuracy numbers on food video.
