# Apple mobile LiDAR: hardware and the ARKit depth pipeline (for 5-30 cm food at 20-50 cm range)

Scope: what the rear LiDAR (iPad Pro 2020 onward, iPhone 12 Pro onward) physically measures, how ARKit and AVFoundation turn it into dense depth, and which scan artifacts come from physics/hardware and which from Apple's software. Current as of Oct 2026. Accuracy studies are only summarised briefly here because another researcher covers them.

## Sensor hardware: dToF SPAD receiver, VCSEL emitter, dot pattern, wavelength, range, generations

### Takeaway
The LiDAR is a direct time-of-flight (dToF) system: a Sony stacked SPAD NIR sensor with a Lumentum VCSEL (later all-Sony). The 2020-2022 generation projects only 576 dots, as 3x3 DOE copies of an 8x8 VCSEL block forming a 24x24 grid. "9x64" and "24x24" are the same 576-dot pattern. From iPhone 15 Pro onward (and reportedly 16 Pro), the module was redesigned: 112 individually addressed VCSEL cavities, an 8x14 (about 111-112) dot pattern with no DOE, and a smaller 116x92 SPAD array. So newer phones measure about 5x fewer true depth points per frame. Everything else in the 256x192 map is inferred.

### Cited Findings
- **2020 iPad Pro module (System Plus/Yole teardown).** The emitter is a Lumentum VCSEL with a DOE and a Texas Instruments driver that generates the pulse and controls VCSEL power and beam shape. The receiver is a Sony near-infrared CMOS SPAD sensor that measures time of flight directly. Further details:
  - 10 µm pixels, about 30 kpixel resolution.
  - The first consumer CIS with in-pixel hybrid bonding (DBI) between the SPAD wafer and the logic wafer.
  - Metal-filled deep trench isolation between pixels.
  - Sources: [EE Times / edge-ai-vision (Cambou, Yole; Ayari, System Plus)](https://www.edge-ai-vision.com/?p=25828), [Yole report page](https://www.yolegroup.com/product/report/apple-ipad-pro-lidar-module), [EE Asia](https://www.eetasia.com/look-inside-ipad-pro-11s-lidar-scanner/).
- **2020 dot pattern.** The VCSEL plus DOE projects a "9x64" dot matrix of 576 dots, and the SPAD sensor measures distance to those dots — [EE Times Japan (System Plus figures)](https://eetimes.itmedia.co.jp/ee/articles/2006/09/news059_3.html).
  - Zea and Hanebeck (KIT) describe the light being split "into 3x3 blocks with 8x8 dots each … each depth frame is interpolated from only 24x24 = 9x64 = 576 measurements", citing 4da.tech's teardown analysis — [Zea & Hanebeck, J. Adv. Information Fusion 17(2), Dec 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf).
  - iFixit's IR teardown of the 2020 iPad Pro showed a regular dot grid, "considerably fewer" dots than TrueDepth — [Notebookcheck on iFixit](https://www.notebookcheck.net/Apple-iPad-Pro-2020-iFixit-teardown-demonstrates-sparser-LIDAR-scanning-compared-to-Apple-s-FaceID-sensor.459426.0.html), [AppleInsider](https://appleinsider.com/articles/20/03/28/second-2020-ipad-teardown-shows-how-lidar-differs-from-face-id).
- **Sensor part numbers.** TechInsights names the iPhone 15 Pro/Pro Max LiDAR sensor as Sony IMX591, a stacked BSI dToF SPAD with 10.1 µm pitch that "outperforms the IMX590 used in the previous generation". It has basically the same pixel size as its predecessor but a smaller pixel array and lower resolution — [TechInsights](https://www.techinsights.com/ja/node/54416), [TechInsights process flow](https://techinsights.com/ko/node/54450).
  - This implies the 2020-2022 part is IMX590. That is inferred from TechInsights' wording; I did not find a separate primary confirmation.
  - Sony's IMX611 (SPAD, 28% photon detection efficiency, announced 2023) was only speculated by Notebookcheck to be the iPhone 15 Pro sensor and is not confirmed — [Notebookcheck](https://www.notebookcheck.com/Sony-enthuellt-IMX611-als-moeglichen-ToF-Sensor-fuer-den-LiDAR-Scanner-des-Apple-iPhone-15-Pro.699203.0.html).
- **iPhone 15 Pro module (Yole, ISSW 2024 paper by P. L. Bonanno)** — [ISSW 2024 R06.1](https://imagesensors.org/Past%20Workshops/2024%20ISSW/Papers/R06.1.pdf), [Yole iPhone 15 Pro LiDAR report](https://www.yolegroup.com/product/report/iphone-15-pro-lidar-module/):
  - All active dies come from Sony.
  - "The 8x14 dot pattern is created by the VCSEL's 112 independently addressed cavities". The VCSEL is bottom-emitting and flip-chip gold-bumped to the driver ASIC, and the transmit optic is a refractive lens rather than a DOE.
  - SPAD array is 116x92 = 10,672 pixels at 10.1 µm pitch, with 0.73 mm² total detecting area. The process is 90 nm SPAD on 28 nm logic with Cu-Cu hybrid bonding at 5 µm pitch. The 28 nm layer "also performs histogramming" (TDC histograms, the standard dToF approach).
  - Emitter-receiver baseline is 3.0 mm. Module size is 10.3x7.6x3.0 mm, at an estimated cost of $6.60.
- **Dot counts compared.** A user counted about 111 dots on the iPhone 15 Pro and about 576 on the 12 Pro with an IR night-vision camera. Apple DTS declined to confirm hardware details, calling them proprietary. This is consistent with Yole's 8x14 = 112 — [Apple Developer Forums thread 813071](https://developer.apple.com/forums/thread/813071).
- **iPhone 16 Pro.** A Chinese teardown channel, reported by iPhone Mania, found the same LiDAR pattern as the 15 Pro — [iphone-mania.jp](https://iphone-mania.jp/?p=588361). This is a secondary report.
- **Wavelength and range.** TrendForce describes the 2020 scanner as a "940nm 3W VCSEL direct ToF" with "sense distance up to 5 meters" — [TrendForce 2020](https://www.trendforce.com/news/2020/03/19/apple-reveals-new-129-inch-ipad-pro-with-3d-sensing-solution-adopting-lidar-scanner/). This is a single source. I found no Apple statement of the wavelength.
  - Zea & Hanebeck found that anything beyond 5 m gets confidence 0 (inferred depth), and that values below 10 cm are marked low or medium confidence, giving "reconstructed" depths that "may have little relation to the actual physical distance" — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf).
- **Operating principle and rate (Apple).** "The LiDAR shoots light onto the surroundings and then collects the light reflected off the surfaces … depth is estimated by measuring the time it took for the light to go … and reflect back … this entire process runs millions of times every second" — [WWDC20 "Explore ARKit 4"](https://developer.apple.com/videos/play/wwdc2020/10611/).
- **Devices.** Apple lists iPad Pro 11-inch and 13-inch (M4 and M5) among LiDAR models — [Apple Support](https://support.apple.com/en-gb/guide/ipad/aside/ipad8a95fdc0/ipados). Devices for the AVFoundation LiDAR camera: "iPhone 12 Pro or later, iPad Pro 11-inch (3rd gen) or later, iPad Pro 12.9-inch (5th gen) or later" — [Apple sample: Capturing depth using the LiDAR camera](https://developer.apple.com/tutorials/data/documentation/avfoundation/capturing-depth-using-the-lidar-camera.md).

### Inferences
- **Geometry at food range.** The depth map's field of view is about 60°x48° ([Zea & Hanebeck](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf)). With a 24x24 grid, dots are about 2.5° apart. With 8x14, they are roughly 4-7° apart, if the pattern spans the same FOV (unverified).
  - At 30 cm, a 2.5° pitch is about 1.3 cm between true measurements on the 2020-2022 generation. A 4-7° pitch is about 2-3.7 cm on iPhone 15 Pro and later.
  - A 10 cm food item at 30 cm therefore gets roughly 40-60 real dToF samples on old hardware and only about 6-15 on new hardware. Everything between them is RGB-guided interpolation.
  - This dominates small-object fidelity. Food relief such as rice grains or a 2 cm cookie lies mostly between dots.
- **Parallax.** The 3 mm emitter-receiver baseline means parallax at 20 cm is negligible. Near-range problems are more plausibly from SPAD saturation/pile-up, optical crosstalk inside the module, and focus or calibration of the dot pattern than from triangulation geometry. This is speculative: there is no primary source on the near-range mechanism.
- **Generations.** Device generation matters. An iPad Pro 2020-2022 (576 dots) may give denser true measurements than an iPhone 15/16 Pro (about 112 dots), unless the M4/M5 iPad Pro adopted the newer module. That is unknown (see Gaps).

### Gaps
- No teardown found documenting the LiDAR in the iPad Pro M4 (2024) or M5, or in the iPhone 17 Pro: dot count, sensor part, whether it matches the 15 Pro module. Measuring this yourself with an IR-sensitive camera is easy.
- No primary Apple source for the wavelength. 940 nm is TrendForce only.
- The native LiDAR frame or pulse rate is not published. Apple says only "millions of times every second" for pulses and 60 Hz for ARKit output.
- No published minimum range spec. Only Zea & Hanebeck's observation of <10 cm, and only for the iPad Pro 2021.
- Whether the 15 Pro's dots are scanned or time-multiplexed across frames (independently addressed cavities would allow it) is unknown. Apple declined to answer the question in [thread 813071](https://developer.apple.com/forums/thread/813071).

## How ARKit produces 256x192 sceneDepth; smoothedSceneDepth; raw access via AVFoundation

### Takeaway
No public API exposes the raw dToF points. ARKit's `sceneDepth` (256x192 float32, 60 Hz) is a dense map produced by an ML model that fuses sparse LiDAR with the wide RGB image. `smoothedSceneDepth` additionally averages over time, and Apple does not say over how many frames. AVFoundation's `builtInLiDARDepthCamera` (iOS 15.4+) runs a similar ML fusion. It gives up to 320x240 streaming depth or 768x576 photo depth, and with `isFilteringEnabled = false` it drops low-confidence points, which is the closest to "raw" available.

### Cited Findings
- **ARKit fusion and rate (WWDC20).** "The colored RGB image from the wide-angle camera and the depth ratings from the LiDAR scanner are fused together using advanced machine learning algorithms to create a dense depth map … This operation runs at 60 times per second with the depth map available on every AR frame." Depth is the distance "from plane of the camera to a point in the world" (z-depth, not radial) — [WWDC20 Explore ARKit 4](https://developer.apple.com/videos/play/wwdc2020/10611/).
- **Format.** The map is 256x192, 32-bit float metres, registered to the 1920x1440 color image at 4:3. The confidence map is 256x192 UInt8 with values 0/1/2. Measured on iPad Pro 2021, ARKit 5, iOS 15.2 — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf).
- **Quantisation.** Depth comes out in steps of 1/1024 m (about 0.98 mm) between 1 and 2 m, and 1/512 m between 2 and 4 m, i.e. 1024 values per power-of-two interval — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf). Extrapolating the pattern gives about 0.5 mm steps at 0.5-1 m and about 0.24 mm at 0.25-0.5 m. That extrapolation is my inference.
- **smoothedSceneDepth (Apple docs).** It is "an average of distance measurements … that creates smoother visuals … similar to sceneDepth except that the framework smoothes the depth data over time to lessen its frame-to-frame delta." It must be requested via its frame semantic and is supported on select devices and configurations — [Apple: smoothedSceneDepth](https://developer.apple.com/tutorials/data/documentation/arkit/arframe/smoothedscenedepth.md).
- **Raw output and the ML system cannot be accessed.** "As of iOS 15, there is no way to turn it off and obtain the raw data." The ML system "acts by aggregating measurements into coherent surfaces and 'smoothing' out sudden changes in depth" — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf). A 2025-26 forum commenter likewise knew of no raw-point API, and Apple DTS did not contradict it — [forum 813071](https://developer.apple.com/forums/thread/813071).
- **Behaviour when one input is missing (Zea & Hanebeck 2022)** — [source](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf):
  - Cover the RGB camera and depth becomes "extremely blurred, lacking sharp corners".
  - Cover the LiDAR and confidence is low everywhere, while depth is hallucinated from RGB. In their test, a monitor 20 cm away showing a rendered cube produced a "cube at a distance of 2 m".
  - Shiny or reflective objects such as a laminated picture can trigger the same inference-only behaviour locally.
- **Temporal and spatial correlation of noise (Zea & Hanebeck 2022, iPad at about 2 m)** — [source](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf):
  - Noise is "strongly spatially correlated": the iPad gives "a smooth, almost flat surface that shifts and deforms each frame", unlike the Kinect 2's independent per-pixel noise. Mean spatial correlation is 0.25 for the iPad versus 0.09 for the Kinect.
  - Lag-1 temporal autocorrelation is higher than the Kinect's (0.034 vs −0.019) and takes about 10 frames to decay. The authors still treat it as roughly time-independent.
- **AVFoundation LiDAR Depth Camera (WWDC22 "Discover advances in iOS camera capture")** — [WWDC22 session 110429](https://developer.apple.com/videos/play/wwdc2022/110429/):
  - It "produces high quality depth data by fusing sparse output from the LiDAR Scanner with the color image from the rear-facing wide-angle camera … processed using a machine learning model that outputs a dense depth map."
  - Video formats run from 640x480 to 4032x3024. "While streaming, it can output depth maps up to 320 by 240 … for photo capture … 768 by 576", with slightly different sizes for 16:9 vs 4:3. Apple says this is "more than twice as big as ARKit's depth resolution of 256 by 192."
  - "Depth data is filtered by default. Filtering reduces noise and fills in missing values, or holes … computer vision apps should prefer non-filtered depth … When filtering is disabled, the LiDAR Depth Camera excludes low confidence points." Filtering is controlled by `isFilteringEnabled` on `AVCaptureDepthDataOutput`.
  - It produces absolute (metric) depth and is available from iOS 15.4.
- **Sample code.** The LiDAR camera combines a LiDAR and a YUV camera, uses the `kCVPixelFormatType_DepthFloat16` depth format, and syncs depth and video with `AVCaptureDataOutputSynchronizer` — [Apple sample](https://developer.apple.com/tutorials/data/documentation/avfoundation/capturing-depth-using-the-lidar-camera.md).
- **Developer experience.** One developer reports that on iPhone 12/13 Pro the AVFoundation filtered depth "jumped" between values, and the unfiltered depth "had a lot of holes and sometimes … go all black". This is anecdotal and went unanswered by Apple — [forum 724152](https://developer.apple.com/forums/thread/724152).
- **Apple patent on ML densification.** US11238604B1 "Densifying sparse depth maps" (Apple; Baig, Ulbricht) predicts dense depth from sparse depth plus an image, using sub-networks that predict semantics and surface normals — [Google Patents](https://patents.google.com/patent/US11238604). Its stated sparse source is SLAM-style depth. It is not confirmed to be the LiDAR pipeline. Zea & Hanebeck cite a different patent [33] for the iPad ML system, which I did not retrieve.
- **People occlusion.** Requesting `personSegmentationWithDepth` gives `sceneDepth` at no extra power cost — [WWDC20](https://developer.apple.com/videos/play/wwdc2020/10611/).

### Inferences
- **Every sceneDepth pixel is model output.** Even "high confidence" pixels are network outputs anchored near real dots, not raw time-of-flight samples. Edge shapes, filled holes and flat-surface "melting" are software artifacts. Missing returns are a hardware matter, but ARKit hides them by filling.
- **Unfiltered AVFoundation depth for small food.** It is the best available proxy for real measurements: 320x240 at video rate, with low-confidence points removed rather than hallucinated. Expect holes rather than smears. It cannot run at the same time as an ARKit session (both need the wide camera), so you lose ARKit's VIO poses. That is my inference; I did not verify Apple documentation of this mutual exclusion.
- **Use sceneDepth, not smoothedSceneDepth, when moving.** For multi-view fusion with known poses, prefer `sceneDepth`. Temporal averaging in `smoothedSceneDepth` under camera motion will lag or smear geometry. Apple does not document whether smoothing is pose-compensated, so this is an inference.

### Gaps
- AVFoundation LiDAR camera depth frame rate: the WWDC22 transcript does not state it. Developers would need to query `videoMinFrameDurationRange` on device. The commonly quoted "30 fps" is unverified.
- The smoothing window or filter of `smoothedSceneDepth`, whether it is motion-compensated, and its latency are all undocumented by Apple. I found no measurement of it.
- I found no published measurement of sceneDepth latency relative to the RGB frame or the IMU.
- I found no confirmation of whether newer iOS versions (17-26) changed the ARKit densification model or allow raw access. The WWDC21-25 sessions surfaced nothing new in this search.

## ARConfidenceLevel semantics and behaviour

### Takeaway
Confidence has three levels: 0 = low, 1 = medium, 2 = high. Apple defines them only qualitatively. Empirically, low marks pixels with no valid return (out of range, reflective or absorbing surfaces, depth discontinuities, image borders) whose depth was extrapolated. Medium marks smoothed-over regions such as edges merged into the background. Filtering to high-only is the main lever for removing flying pixels.

### Cited Findings
- **Apple's definitions.** Low: "Depth-value accuracy in which the framework is less confident". Medium: "moderately confident". High: "fairly confident" — [Apple: ARConfidenceLevel](https://developer.apple.com/tutorials/data/documentation/arkit/arconfidencelevel.md).
- **Apple on surfaces and light.** The confidence map indicates how accurate each depth value is and is "useful in filtering out lower-accuracy depth values". Natural light affects `depthMap` such that ARKit is "less confident about the accuracy of the LiDAR Scanner's depth measurements for surfaces that are highly reflective, or that have high light absorption" — [Apple: ARDepthData.confidenceMap](https://developer.apple.com/documentation/arkit/ardepthdata/confidencemap.md), [Apple: ARDepthData](https://developer.apple.com/tutorials/data/documentation/arkit/ardepthdata.md). WWDC20 says the same thing: "Challenging surfaces, such as those which are highly reflective or those with high absorption, can lower the accuracy" — [WWDC20](https://developer.apple.com/videos/play/wwdc2020/10611/).
- **Apple engineer on accuracy.** The accuracy of individual depth values "highly depends on this confidence level" — [Apple forum thread 650321](https://developer.apple.com/forums/thread/650321).
- **Measured behaviour (Zea & Hanebeck 2022)** — [source](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf):
  - Invalid measurements are not encoded as depth 0. Instead confidence is set to 0 and the depth is "extrapolated based on surrounding depth values and semantic cues from the color image".
  - Low confidence appears "around the edges of the table, or at the image borders", but "these gaps are not evident when looking at the depth image on its own".
  - Smoothed-over discontinuities, such as a box back merging into the floor, "can usually be identified by their confidence level of 1 (medium) or 0 (low)".
  - Causes of invalid measurements include "a reflective or bright material, a drastic change in depth, or the depth being outside of the operative range".
  - Narrow objects under about 6% of image width (about 15 px of 256) are not detected, which they relate to the dot spacing of about 11 px.
  - Confidence is low or medium below 10 cm and 0 beyond 5 m.
- **Old forum report.** In 2021 a developer reported confidence above Low not working on an iOS 14.5 beta. This is historical — [Apple forum 676368 area / search result](https://developer.apple.com/forums/thread/676368).

### Inferences
- **At a plate rim or food edge**, expect a band of about 1 dot spacing (roughly 10 px in sceneDepth) of medium or low confidence. Dropping those pixels removes most flying pixels, but it also erodes small food items, since a 10 cm item at 30 cm spans only about 60-70 px of 256 in width.
- **Confidence is a software estimate**, presumably from return-signal strength and model uncertainty. A wrong-but-consistent depth, such as a multipath or specular bias on a glazed plate, can still be marked high. No source quantifies this.

### Gaps
- Apple publishes no numerical error band per confidence level.
- I found no study of confidence specifically on food, dark sauces, or translucent items.

## Known failure modes: physics/hardware vs Apple software

### Takeaway
Several artifacts come from the physics and hardware:
- sparse sampling of 576 or about 112 dots
- weak returns on dark or absorbing surfaces
- specular or mirror-like surfaces that send light away or produce multipath
- subsurface scattering in translucent food, which delays and spreads the return and biases depth deeper
- sunlight raising SPAD background
- unreliability under about 10 cm

These are software artifacts:
- smeared or "melted" edges and fringe/flying pixels
- planes not flat and corners not 90°
- coherent frame-to-frame surface "breathing"
- hallucinated depth where LiDAR returns are missing
- temporal lag in `smoothedSceneDepth`

The software artifacts come from ML densification and are amplified by the sparse dot grid.

### Cited Findings
- **Flying pixels and edges.** There is "a large amount of 'fringe points' (also known as 'flying pixels') at the borders … as a result of the relatively low LIDAR resolution". Planar surfaces "are not necessarily shown flat, and 90° corners are not usually preserved". The ability to discern depth discontinuities falls with distance: a box back merges into the floor at 1 m, and a chessboard 20 cm from a wall "melts" into the wall at 4 m — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf).
- **Curvature loss.** A 15 cm-radius sphere was estimated at 15.5 cm near range and about 17.5 cm farther away. Fringe-pixel proportion rises with distance, and "curved objects … smoothened out" — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf). These were measured at 1-2.3 m, not food range.
- **Reflective and absorbing surfaces.** Apple says highly reflective or high-absorption surfaces lower accuracy and confidence — [WWDC20](https://developer.apple.com/videos/play/wwdc2020/10611/), [Apple confidenceMap doc](https://developer.apple.com/documentation/arkit/ardepthdata/confidencemap.md). A shiny laminated picture can trigger depth hallucinated from RGB — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf).
- **Minimum range.** Depths below 10 cm get low or medium confidence with "reconstructed" values — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf). No source confirmed a 20-25 cm bias threshold; see Gaps.
- **Sunlight.** Active phone depth sensors "do not operate well outdoors because the signal light is much weaker than ambient sunlight". 940 nm plus ambient rejection is the standard mitigation. This is generic, from a Samsung structured-light paper, not Apple-specific — [Park et al., ICCVW 2017](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w34/usage_bh711.parksamsung.com_y.c.kehsamsung.com_ofldhsamsung.com_ICCV_2017_paper.pdf).
- **Small objects.** Vogt et al. 2021 (Technologies 9(2)), comparing iPad Pro LiDAR and TrueDepth with an industrial scanner at 300 mm, reportedly concluded that the iPad LiDAR is impractical for small objects such as LEGO bricks. That conclusion is as cited by a 2022 ISPRS paper, which also confusingly quotes a 0.16 mm deviation — [ISPRS Archives XLIII-B4-2022-303](https://isprs-archives.copernicus.org/articles/XLIII-B4-2022/303/2022/isprs-archives-XLIII-B4-2022-303-2022.pdf). I did not verify the original.
  - Luetzenburg et al. 2021 (Sci. Rep.) found iPhone 12 Pro models of objects with side length >10 cm to be accurate to ±1 cm — [DOAJ record](https://doaj.org/article/05f6da2bb22e4343bec5db6fcf0ecf88).
- **Depth bias.** On the iPad Pro 2021 at 1-5 m, bias was always negative (measured nearer than truth) at 1-2% of distance. Precision variance was σz² = 1e-6·(0.07z³ − 0.32z² + 0.64z − 0.02) m², with z in metres — [Zea & Hanebeck 2022](https://isas.iar.kit.edu/pdf/JAIF22_Zea.pdf). Extrapolating to 0.3 m gives σ of about 0.4 mm. That is my extrapolation outside their measured range and is not validated.
- **Handheld versus static.** Static scans put points within <1 mm of a best-fit plane, but dynamic handheld scans gave about 1 cm typical error. The authors recommend 1-1.5 m range for building work — Teo & Yang 2023, Developments in the Built Environment, [DOAJ](https://doaj.org/article/9cbab8e9cd04423aae61f174ca458b1e).

### Inferences
- **Physics/hardware, which more processing cannot fix:**
  - Too few true samples on small items, especially on iPhone 15 Pro and later.
  - Dropouts or weak returns on black or dark food such as nori, burnt crust or dark sauces.
  - Glazed plates and wet or oily food: specular reflection sends the pulse away, so there is no return, or causes a two-bounce multipath that appears too deep.
  - Translucent or scattering food (rice, jelly, milk, fat, fruit flesh): subsurface scattering lengthens time of flight, making the surface appear slightly deeper.
  - Sunlight raising SPAD background noise.
  - Unreliable readings under about 10-15 cm.
- **Apple software, which can be mitigated by confidence filtering, unfiltered AVFoundation capture, multi-view fusion, or RGB-based masking:**
  - Fringe and flying pixels interpolated between foreground and background, such as a plate rim against the table or food against the plate.
  - Rounded corners and flattened relief.
  - Coherent per-frame surface wobble.
  - Hallucinated depth in no-return regions, which is still filled, with low confidence.
  - Lag from temporal smoothing when using `smoothedSceneDepth`.
- **Multipath versus subsurface scattering.** These are generic dToF physics and lack Apple-specific measurements. They push depth farther, whereas Zea's global bias is nearer. The two sign conventions could help diagnose which artifact is present.

### Gaps
- I found no Apple-specific measurement of multipath on glossy ceramic, subsurface-scattering bias in food, or sunlight washout thresholds. These are flagged as generic dToF physics, not verified on Apple LiDAR.
- No source quantifies temporal lag when the device moves, for either `sceneDepth` or `smoothedSceneDepth`.
- No source measures a minimum-range bias at 20-25 cm. Zea's 10 cm is the only figure. A quick test is to scan a flat plane at 15/20/25/30 cm on the target device and compare with a ruler.
- I could not retrieve the full text of Vogt et al. 2021, the Aalto 2021 thesis on iPad Pro 2020 depth (the Aalto server returned 403), or Teppati Losè 2022. These may contain near-range figures.
