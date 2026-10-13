# How mobile 3D scanning apps get clean scans on LiDAR iPhones/iPads (as of Oct 2026)

Research scope: Polycam, Scaniverse, Record3D, 3d Scanner App, KIRI Engine, Luma AI, RealityScan, Metascan, SiteScape, Stray Scanner, NeRFCapture, Heges, Apple Object Capture. Coverage is uneven. Several help centres (Polycam learn.poly.cam, MDPI, Southampton eprints) returned HTTP 403, and Scaniverse's support page now redirects to Niantic Spatial. Luma AI, Metascan and SiteScape were not reached within the tool budget (see Gaps).

## 1. Reconstruction method, recorded data and exposed settings per app

### Takeaway
Two families: (a) LiDAR depth fusion (3d Scanner App, Polycam LiDAR, Scaniverse mesh, KIRI LiDAR, Heges) gives instant, metric-scale but coarse meshes, with settings such as confidence threshold, max depth and voxel resolution. (b) Image-based methods (Object Capture photogrammetry, Polycam Photo, KIRI Photo/NSR/3DGS, Scaniverse splats, RealityScan) give much finer detail. Apple's Object Capture is a hybrid that adds LiDAR points to photogrammetry. The clearest exposed settings belong to 3d Scanner App: confidence LOW/MED/HIGH, max depth 0.3–5.0 m, resolution 5–20 mm, and OBJECT masking.

### Cited Findings
**3d Scanner App (Laan Labs)**
- Has two LiDAR modes. Low Resolution is "simplest… best for large areas such as a house." High Resolution is for "smaller objects like a couch or chair". — [3d Scanner App Help](https://3dscannerapp.com/help/)
- High Res settings:
  - **Confidence** LOW/MEDIUM/HIGH: "Choosing HIGH only keeps the best quality data, but reduces the amount of data available."
  - **Max Depth / Range** discards sensor data beyond a set distance, which "reduces scan size and improves accuracy."
  - **Resolution** 5–20 mm, where smaller means finer detail but limits scan size. "10–15 mm is recommended."
  - **Masking** NONE/OBJECT/PERSON. OBJECT "tries to isolate the main object in view… keep the object fully in view, and use on a simple background without other clutter."
  - Source: [3d Scanner App Help](https://3dscannerapp.com/help/)
- A post-scan "Refine" tool offers smoothing and changes to mesh resolution. Small objects get a texture automatically. — [3d Scanner App Help](https://3dscannerapp.com/help/)
- Numeric ranges (citing the app's user guide): max depth 0.3–5.0 m in 0.1 m steps, resolution 5–20 mm in 1 mm steps. — [MDPI Remote Sensing 14(17):4157, "Apple LiDAR Sensor for 3D Surveying…" (search snippet; page 403'd)](https://www.mdpi.com/2072-4292/14/17/4157)
- The App Store listing adds a RoomPlan mode, a TrueDepth mode "for smaller, detailed objects", and a Photo Mode that uses Apple's Object Capture photogrammetry API. — [App Store listing (via search snippet)](https://apps.apple.com/us/app/-/id1419913995)
- J.S. Held lists the High Res settings as Confidence, Range, Masking, Resolution. — [J.S. Held comparison](https://www.jsheld.com/insights/articles/a-comparison-of-mobile-phone-lidar-capture-and-established-ground-based-3d-scanning-methodologies)

**Underlying ARKit data (what all LiDAR apps build on)**
- ARKit exposes `sceneDepth` / `smoothedSceneDepth` on `ARFrame`, each with "a depth and a confidence buffer": depth in metres plus per-pixel confidence. — [Apple Developer Forums thread 663691 (Apple engineer reply)](https://developer.apple.com/forums/thread/663691)
- `confidenceMap` "is useful in filtering out lower-accuracy depth values". — [Apple ARDepthData docs (search snippet)](https://developer.apple.com/documentation/arkit/ardepthdata.md)

**Polycam**
- LiDAR mode and Photo mode. Photo mode is recommended for small objects. With LiDAR "you get a true-to-scale model provided the object is not too small". In Photo Mode you can flip the object, but you must "enable the Use Object Masking before uploading". Photo-mode models need their scale corrected in a mesh editor. — [Vuforia: Model Targets with Polycam](https://developer.vuforia.com/library/using-3d-scans/model-targets-polycam)
- Polycam's own large-object article (snippet) says LiDAR captures overall shape quickly "but may not capture finer details as well as Photo Mode". — [Polycam Learn (403 on fetch; search snippet)](https://learn.poly.cam/hc/en-us/articles/30549135578388-How-to-Capture-a-Large-Objects-in-Polycam)
- Raw-data export, after enabling "Developer mode" (applies to future captures only), produces a zip of images and globally optimized poses. Nerfstudio notes Polycam poses are "globally optimized… more robust to drift than ARKit or SLAM", so no COLMAP step is needed. Nerfstudio advises: use LiDAR or Room mode, good lighting, many angles, move slowly, or use manual shutter to reduce blur. — [Nerfstudio custom data docs](https://docs.nerf.studio/quickstart/custom_dataset.html)
- Polycam's raw-data article (snippet) includes confidence images. Gray marks medium confidence, and low-confidence areas "should be excluded from precision measurements". — [Polycam Learn: How to Extract Raw Data (snippet; 403 on fetch)](https://learn.poly.cam/hc/en-us/articles/38276871185044-How-to-Extract-Raw-Data-and-What-Is-Included)
- Third-party review workflow: do a quick LiDAR scan first to see problem areas, then capture in Photo mode. Photo mode gives "a much better result" for 3D printing. — [3DWithUs Polycam review](https://3dwithus.com/polycam-3d-scanning-app-review-lidar-vs-photo-mode). One tutorial claims a 20-photo minimum, but this was not verified against official docs — [aiindigo tutorial](https://aiindigo.com/tutorials/getting-started-with-polycam-professional-3d-scanning-on-mobile)

**Scaniverse (Niantic; support now redirects to Niantic Spatial)**
- Splat mode: New Scan > Splat > Process. Initial result in about 1 minute. "Enhance" trains about 1 more minute for finer detail. All processing is on-device. Marketed as suitable "from small objects to large buildings" (marketing claim). — [Scaniverse splat announcement](https://scaniverse.com/news/scaniverse-introduces-support-for-3d-gaussian-splatting), [CGPress](https://cgpress.org/archives/scaniverse-introduces-support-for-3d-gaussian-splatting.html)
- Release notes mention a "Detail Mode" (3.0.1 crash fix when frames fail to align), better reconstruction of thin objects and detail for splats (3.2.0), and better small details from improved depth mapping (2.1.0). — [Scaniverse release notes](https://scaniverse.com/about/releasenotes)

**Record3D**
- Records RGBD "3D videos" (point clouds) using LiDAR or the TrueDepth/FaceID camera. Exports glTF/PLY. Streams live over USB (open-source C++/Python libs) or Wi-Fi (paid). — [record3d.app](https://record3d.app/)
- Nerfstudio import uses the "EXR + JPG sequence" export. Poses come from LiDAR/ARKit, so no COLMAP. An optional PLY point-cloud sequence (`--ply`) can initialise splats, with `--voxel-size` defaulting to 0.8. — [Nerfstudio custom data docs](https://docs.nerf.studio/quickstart/custom_dataset.html)

**Stray Scanner (open source)**
- Records `rgb.mp4` (HEVC) plus per-frame depth as 16-bit PNGs, **256×192**, in mm.
- Records per-frame **confidence** PNGs, 256×192, with values {0,1,2}.
- Writes `odometry.csv` with timestamp, frame, position, quaternion and per-frame fx, fy, cx, cy, plus `imu.csv` (accel and gyro) and `camera_matrix.csv`.
- Writes optional per-frame lens-distortion lookup tables.
- Sources: [strayrobots/scanner docs/format.md](https://raw.githubusercontent.com/strayrobots/scanner/main/docs/format.md), [GitHub repo](https://github.com/strayrobots/scanner)
- Companion tools: StrayVisualizer and Autolabel (volumetric labelling). — [GitHub repo](https://github.com/strayrobots/scanner)

**NeRFCapture (open source)**
- Collects ARKit-posed RGB images, plus depth on LiDAR devices.
- Offline mode saves an instant-ngp-format zip. Online mode streams over CycloneDDS to `nerfcapture2nerf.py` in instant-ngp, and the author recommends Online because getting files off iOS is hard.
- A "Reset" button re-origins the coordinate system.
- Source: [jc211/NeRFCapture](https://github.com/jc211/NeRFCapture)

**KIRI Engine**
- Modes: Photo Scan (photogrammetry), Featureless Object Scan (Neural Surface Reconstruction for smooth, shiny or single-colour objects), 3DGS, and LiDAR. — [KIRI featureless capture FAQ](https://kiriengine.app/faq/how-to-capture-with-featureless-object-scan), [80.lv](https://80.lv/articles/kiri-engine-introduces-featureless-object-mode/?amp=1), [KIRI LiDAR page](https://www.kiriengine.app/features/lidar-scan)
- "Auto Object Masking" separates the object from the background by contour detection. It is needed when the object moved or a turntable was used, and KIRI recommends enabling it before upload. — [KIRI FAQ: auto object masking](https://www.kiriengine.app/faq/what-is-auto-object-masking)
- Developer mode can export camera poses for nerfstudio (`ns-train nerfacto --data …`). — [Nerfstudio custom data docs](https://docs.nerf.studio/quickstart/custom_dataset.html)

**Heges**
- Exposes a precision setting, where higher precision means shorter range.
- The developer says TrueDepth (FaceID) is "much more accurate than LiDAR, by roughly one order of magnitude", recommends TrueDepth at the 0.5 mm setting (claims "up to 2 mm accuracy"), and says LiDAR is "not suitable for small, detailed objects".
- iPhone 13 and later TrueDepth gives "slightly lower-accuracy results".
- TrueDepth cannot capture glass.
- These are developer claims, not independent measurements. — [Heges App Store listing](https://apps.apple.com/app/id1382310112)

**Spectacular AI (nerfstudio-supported recorder)**
- VISLAM fuses IMU, camera and LiDAR. Nerfstudio suggests `--key_frame_distance=0.05` for small scans and 0.15 for rooms. — [Nerfstudio custom data docs](https://docs.nerf.studio/quickstart/custom_dataset.html)

**Apple Object Capture (iOS 17+, RealityKit)**
- Hybrid method: photogrammetry from auto-selected sharp, well-exposed images, plus LiDAR points from various view angles. "We now support the reconstruction of low-texture objects by leveraging the LiDAR Scanner."
- On-device iOS reconstruction supports only the "reduced" detail level and outputs diffuse, AO and normal maps.
- `isOverCaptureEnabled` stores extra images for higher-detail Mac reconstruction, which also uses the saved LiDAR data.
- `.poses` can be requested.
- Requires iPhone 12 Pro / iPad Pro 2021 or later.
- Source: [WWDC23 "Meet Object Capture for iOS"](https://developer.apple.com/videos/play/wwdc2023/10191)

**RealityScan Mobile (Epic)**
- Cloud photogrammetry with auto-capture on by default and a 300-image limit per project. — [RealityScan step-by-step guide](https://dev.epicgames.com/documentation/realityscan-mobile/realityscan-step-by-step-guide)

### Inferences
- The settings that matter for a food app map directly onto 3d Scanner App's High Res controls:
  - ARKit confidence threshold. Keep medium or high near object edges.
  - Near-range depth cutoff, e.g. 0.3–0.6 m for a plate.
  - Voxel or TSDF size. Use 5 mm or finer for food, below the 10–15 mm default meant for furniture.
  - Object masking.
- Because ARKit depth is only 256×192 (per Stray's format), LiDAR-only meshes of plate-sized food will be coarse. Most apps send small objects to image-based or hybrid pipelines for that reason. A hybrid like Object Capture (photo detail, LiDAR scale and support) is the best-fitting model for food volume.
- Our recorder should save what Stray/Record3D save: RGB, depth, confidence, per-frame intrinsics and poses, and IMU. That way offline re-fusion with different confidence and voxel settings is possible, and nerfstudio importers (record3d, polycam) can be reused.

### Gaps
- Luma AI, Metascan and SiteScape: modes and settings not retrieved (budget). Luma's iOS app is known to do cloud NeRF/splat capture with guided rings, but this is not verified here.
- Polycam help-centre pages returned 403. Polycam claims come from search snippets and third parties.
- Record3D's in-app depth or confidence filtering settings were not documented on the pages reached.
- Scaniverse: no documented mesh-resolution or range control found.

## 2. Official tips for scanning small objects

### Takeaway
The guidance is consistent across vendors:
- Use diffuse, even lighting.
- Move slowly.
- Keep the whole object in frame against a plain, uncluttered background.
- Do three passes at different heights. If the object is rigid and textured, flip it and scan three orientations.
- Avoid reflective, transparent and thin structures.
- Use turntables only with textured objects and object masking.

For food this means: don't flip the food (it deforms), so do three height passes instead. Glossy sauces and transparent items are known failure cases.

### Cited Findings
- Apple:
  - "For the best model reconstruction, we recommend completing three scan passes."
  - Rigid objects: "flipping is a good idea". Deformable objects: "it is better not to move them".
  - When flipping, scan "in three orientations". If the object can't be flipped, capture "from three different heights".
  - Avoid symmetric or repetitive textures for flipping, because textureless objects can't be stitched across flips.
  - Use diffuse lights to minimise reflections.
  - "Avoid objects that are reflective, transparent, or contain very thin structures."
  - Source: [WWDC23 10191](https://developer.apple.com/videos/play/wwdc2023/10191)
- 3d Scanner App:
  - "Move slowly and pan the device around slowly". Quick or jerky motion reduces quality.
  - Several passes may be needed to learn the technique.
  - Start with smaller scans.
  - For masking, keep the object fully in view on a simple background.
  - Source: [3d Scanner App Help](https://3dscannerapp.com/help/)
- KIRI:
  - Close-up shots are fine for sharper detail, but keep the whole object in frame. Slightly wider framing gives more consistent results.
  - Photo Scan works best on textured objects. Avoid featureless, reflective or transparent surfaces in Photo Scan.
  - Featureless mode: orbit from at least three perspectives (top, middle and bottom layers) in a steady video.
  - Use turntables only for Photo Scan and 3DGS, with textured objects. The app has a turntable timer.
  - Sources: [KIRI framing FAQ](https://www.kiriengine.app/faq/photo-scan-framing-requirements), [KIRI turntable tutorial](https://www.kiriengine.app/tutorial/how-to-scan-with-a-turntable), [KIRI featureless FAQ](https://kiriengine.app/faq/how-to-capture-with-featureless-object-scan), [KIRI bottom-of-object tutorial](https://www.kiriengine.app/tutorial/how-to-3d-scan-the-bottom-of-an-object)
- Polycam via Nerfstudio: reduce motion blur, view from many angles, good lighting, move slowly, or use manual shutter. — [Nerfstudio docs](https://docs.nerf.studio/quickstart/custom_dataset.html)
- RealityScan: move slowly around the object. Auto-capture is on, with a 300-image cap. — [RealityScan guide](https://dev.epicgames.com/documentation/realityscan-mobile/realityscan-step-by-step-guide)
- Heges: use TrueDepth (front camera) at 0.5 mm precision for small objects. Glass can't be captured. — [Heges App Store](https://apps.apple.com/app/id1382310112)

### Inferences
- For food on a plate, the transferable rules are:
  - Three height rings (low, mid, top-down). No flipping.
  - Diffuse light and a plain, matte, non-patterned placemat.
  - Slow orbit.
  - Keep the plate fully in frame.
- Turntables are attractive for food, but they require masking because the background is static relative to the moving object.
- Matte spray is used for non-food objects but is not food-safe. No source on it was reached.
- Heges' TrueDepth claim suggests the front-facing camera could beat rear LiDAR for close-range food depth (TrueDepth works at short range). It is worth an internal test, but this is a developer claim only.

### Gaps
- No vendor gave a specific numeric object distance or photo count for small objects in the sources reached. Apple's iOS doc on photo capture (developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture) did not render.
- Matte spray guidance and the Polycam small-object article were not retrieved.

## 3. Live UX that improves capture quality

### Takeaway
Apple's ObjectCaptureSession is the reference design:
- A bounding-box "detecting" step.
- A capture dial that fills per viewing direction.
- A live point cloud.
- Auto-capture of sharp, well-exposed frames.
- Auto-capture pauses with prompts for "move slower", too dark, too close or too far, and object out of frame (arrow).
- A point-cloud preview after each pass.
- Explicit prompts for a new pass or a flip.

RealityScan adds a live quality point cloud with captured-camera positions and focus peaking.

### Cited Findings
- Object Capture:
  - The capture dial shows "where we have collected enough images… and where you still need to capture more". A pass completes when the dial is full.
  - The view shows a point cloud during capture. `ObjectCapturePointCloudView` pauses the session to let the user inspect coverage from all angles.
  - Auto-selects "image shots with good sharpness, clarity, and exposure".
  - "If you move too fast, automatic capture will stop and remind you to slow down."
  - Prompts when too dark, or when too far or too close.
  - When the object leaves the field of view, capture pauses and an arrow appears.
  - Session states: initializing, ready, detecting (box selection), capturing, finishing, completed, failed.
  - `beginNewScanPass()` and `beginNewScanPassAfterFlip()` drive multi-pass capture.
  - Sources: [WWDC23 10191](https://developer.apple.com/videos/play/wwdc2023/10191), [WWDC Notes](https://wwdcnotes.com/notes/wwdc23/10191)
- RealityScan Mobile:
  - 1.8 added "AR Guidance", a live quality point cloud over the subject plus positions of captured photos, and a focus-peaking overlay marking sharp regions. — [RealityScan Mobile 1.8 release notes](https://forums.unrealengine.com/t/realityscan-mobile-1-8-release-notes/2676487), [Digital Production](https://digitalproduction.com/2025/11/21/realityscan-mobile-1-8-with-tools-for-on-the-go-scanning/)
  - The quality-render point cloud in camera view highlights areas needing more coverage. — [RealityScan guide](https://dev.epicgames.com/documentation/realityscan-mobile/realityscan-step-by-step-guide)
  - Epic's desktop docs say green means sufficient coverage and red or orange needs more photos. That is the review mode, not confirmed for the mobile AR overlay. — [Epic RealityScan docs](https://docs-ak.unrealengine.com/realityscan)
- 3d Scanner App and Polycam LiDAR show the mesh building live. This is from general knowledge and was not confirmed in the docs reached.

### Inferences
Our capture guide should replicate these:
- An azimuth-by-elevation coverage dial with three rings.
- A speed gate (pause frame acceptance above an angular or linear velocity threshold) with a "slow down" toast.
- A blur and exposure gate on keyframes.
- Distance band prompts, e.g. 0.25–0.5 m for a plate.
- An off-frame arrow.
- A post-pass point-cloud review.

Focus peaking and per-region quality colouring are cheap extra wins.

### Gaps
- Exact numeric thresholds Apple uses (speed, distance) are not published.
- Polycam's and KIRI's live coverage indicators were not documented in the sources reached.

## 4. Independent comparisons of small-object accuracy

### Takeaway
Few independent measured comparisons were reachable. Available evidence (a heritage-surveying MDPI study, a 2025 Southampton study on a handheld controller, a J.S. Held forensic comparison, and hands-on reviews) agrees that iPhone LiDAR is metric but coarse. Photo, splat and hybrid modes give much better fine detail on small objects. Most "accuracy" figures in app listings are marketing.

### Cited Findings
- The MDPI 2022 study evaluated iPad/iPhone LiDAR apps, including 3d Scanner App, for cultural-heritage survey. Specific error numbers could not be retrieved (403). — [MDPI Remote Sens. 14(17):4157](https://www.mdpi.com/2072-4292/14/17/4157)
- A University of Southampton paper (VR1027 revision) compared Scaniverse mesh vs splat scans of a small handheld controller against a reference model for dimensional accuracy. Numbers not retrieved (403). — [Southampton eprints 506892](https://eprints.soton.ac.uk/506892/1/VR1027_revision.pdf)
- J.S. Held compared phone LiDAR capture with ground-based scanners. — [J.S. Held](https://www.jsheld.com/insights/articles/a-comparison-of-mobile-phone-lidar-capture-and-established-ground-based-3d-scanning-methodologies)
- Hands-on review: Polycam Photo mode is "much better" than LiDAR for small printable objects. This is qualitative. — [3DWithUs](https://3dwithus.com/polycam-3d-scanning-app-review-lidar-vs-photo-mode)
- Heges claims TrueDepth is about 10x more accurate than LiDAR, with up to 2 mm accuracy. This is unverified marketing. — [Heges App Store](https://apps.apple.com/app/id1382310112)
- Polycam's own docs (snippet) concede LiDAR misses fine detail versus Photo mode. — [Polycam Learn (snippet)](https://learn.poly.cam/hc/en-us/articles/30549135578388-How-to-Capture-a-Large-Objects-in-Polycam)

### Inferences
- For food volume, where errors of a few mm across a 10–20 cm item matter, LiDAR-only TSDF is likely the accuracy floor. Image-based refinement, or at least multi-pass, confidence-filtered fusion at about 5 mm voxels, is the lever.

### Gaps
- No quantitative numbers were extracted. Fetch the MDPI and Southampton PDFs from an unblocked network to get RMS and mean errors.
- YouTube comparison transcripts were not reviewed.

## 5. Open-source capture apps and how nerfstudio and others process them

### Takeaway
Open recorders (Stray Scanner, Record3D, NeRFCapture, Spectacular AI) all save RGB plus ARKit poses and intrinsics, and on LiDAR devices also depth (256×192) and confidence. Nerfstudio ingests Record3D, Polycam, KIRI and Spectacular AI directly and skips COLMAP, because device or VISLAM poses are used.

### Cited Findings
- Nerfstudio commands, all of which bypass COLMAP:
  - `ns-process-data polycam` (raw zip from developer mode)
  - `ns-process-data record3d` (EXR+JPG, optional `--ply` with `--voxel-size` default 0.8)
  - KIRI developer-mode export, passed straight to `ns-train`
  - Spectacular AI `sai-cli process --key_frame_distance=0.05` for small scans
  - Source: [Nerfstudio custom data docs](https://docs.nerf.studio/quickstart/custom_dataset.html)
- Stray Scanner format: depth 256×192 uint16 mm, confidence 0/1/2, odometry with per-frame intrinsics, IMU, and distortion LUTs. — [Stray format.md](https://raw.githubusercontent.com/strayrobots/scanner/main/docs/format.md)
- NeRFCapture streams posed RGB and depth over CycloneDDS to instant-ngp, or saves an instant-ngp zip. — [NeRFCapture](https://github.com/jc211/NeRFCapture)
- Record3D streams RGBD live over USB with open-source C++/Python libs. — [record3d.app](https://record3d.app/)

### Inferences
- Adopting the Stray Scanner on-disk layout, or a nerfstudio `transforms.json` with depth and confidence paths, would let us reuse existing tooling: Open3D TSDF, nerfstudio depth-nerfacto, and splatfacto with LiDAR point initialisation.
- Keeping confidence maps lets offline fusion apply "confidence ≥ 1 or 2" and a near max-depth cutoff, mirroring 3d Scanner App's High Res mode.

### Gaps
- Nerfstudio's handling of Record3D and Polycam depth or confidence (whether it is used for depth supervision) is not stated on the docs page.
