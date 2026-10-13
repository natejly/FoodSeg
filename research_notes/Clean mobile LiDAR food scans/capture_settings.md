# Capture-side settings and technique for clean ARKit LiDAR RGB-D scans of small objects (plated food, 5-30 cm at 20-50 cm)

Context: the current app uses ARWorldTrackingConfiguration with the default 1920x1440 @ 60 fps format, sceneDepth + smoothedSceneDepth, autofocus on, gravity alignment, and saves every 4th frame (15 fps) as JPEG q0.92.

## ARKit video formats: 4K, high-res frame capture, sceneDepth compatibility

### Takeaway
There are three relevant capture paths. (1) The default binned 1920x1440 @ up to 60 fps. (2) The 4K format, 3840x2160 @ 30 fps, non-binned, 16:9, which allows HDR. (3) Out-of-band `captureHighResolutionFrame` stills at full sensor resolution (12 MP on iPhone 13), which do not interrupt the video stream and come with a full ARFrame (pose and anchors). iOS 26 adds `captureHighResolutionFrame(using: AVCapturePhotoSettings?)`. Apple's docs don't say whether sceneDepth works with each format, or how hi-res stills register to the 256x192 depth. That has to be checked on the device.

### Cited Findings
- The default path bins a 3840x2880 sensor area 2x2 down to 1920x1440, which allows "up to 60 frames per second". Binning also reduces sensor noise in low light. — [WWDC22 Discover ARKit 6](https://developer.apple.com/videos/play/wwdc2022/10126/)
- 4K mode: "an image area of 3840x2160 pixels is used and you can capture video at 30 frames per second". It is enabled with `ARConfiguration.recommendedVideoFormatFor4KResolution`, which returns nil if 4K is unsupported. It is available on "iPhone 11 and up and on any iPad Pro with an M1 chip". The aspect ratio is 16:9, so iPad full-screen display is cropped at the sides. Apple still suggests full HD at 60 fps for apps that need a high refresh rate. — [WWDC22 Discover ARKit 6](https://developer.apple.com/videos/play/wwdc2022/10126/)
- `ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing` picks the format best suited to hi-res stills. `ARSession.captureHighResolutionFrame(completion:)` and its async form "triggers an out-of-band capture of a high-resolution image… an ARFrame containing the high-resolution image and all other frame properties". Apple's example is the full 12 MP of the iPhone 13 wide camera. — [WWDC22 Discover ARKit 6](https://developer.apple.com/videos/play/wwdc2022/10126/)
- Docs on high-resolution capture:
  - "ARKit populates the frame's properties other than pixel data, including pose information, anchors, and frame semantics."
  - Delivery "doesn't affect the other frames that the session receives".
  - "For the highest resolution captured image, choose a non-binned videoFormat". Use `VideoFormat.isRecommendedForHighResolutionFrameCapturing`.
  - A call fails if a previous capture hasn't finished (`highResolutionFrameCaptureInProgress`) or the pipeline fails (`highResolutionFrameCaptureFailed`).
  - — [captureHighResolutionFrame(completion:) docs](https://developer.apple.com/documentation/arkit/arsession/capturehighresolutionframe(completion:))
- iOS 26.0+ adds `captureHighResolutionFrame(using photoSettings: AVCapturePhotoSettings?)`, plus an async variant. The docs say: "Some video formats do not support a significantly higher still image resolution than the streaming camera resolution… obtain a `defaultPhotoSettings` object from the video format and modify it." — [captureHighResolutionFrame(using:) docs](https://developer.apple.com/documentation/arkit/arsession/capturehighresolutionframe(using:completion:))
- Image retention: "Do not hold on to an ARFrame for too long… Ultimately, the ARCamera's tracking state might fall back to limited". This matters especially for full-resolution stills. — [WWDC22 Discover ARKit 6](https://developer.apple.com/videos/play/wwdc2022/10126/)
- The `sceneDepth` frame semantic is supported only on "select devices and configurations". Check with `supportsFrameSemantics(_:)`. — [ARFrame.sceneDepth docs](https://developer.apple.com/documentation/arkit/arframe/scenedepth)
- ARKit depth is 256x192. AVFoundation's LiDAR Depth Camera goes up to 320x240 while streaming and 768x576 with photo capture ("more than twice as big as ARKit's"). Both are captured in the wide camera's field of view. — [WWDC22 Discover advancements in iOS camera capture](https://developer.apple.com/videos/play/wwdc2022/110429/)
- The sceneDepth depth map is "smaller in resolution compared to the captured image… which still presents the same aspect ratio". Depth is 60 Hz, "available on every AR frame". RGB and LiDAR "are fused together using advanced machine learning algorithms to create a dense depth map". — [WWDC20 Explore ARKit 4](https://developer.apple.com/videos/play/wwdc2020/10611/)

### Inferences
- For food scans, 15 fps at 1920x1440 is very redundant, because neighbouring frames are a few mm apart. A better design keeps the 60 fps (or 4K30) stream for tracking and live depth, and calls `captureHighResolutionFrame` every ~10-15° of orbit. Each call yields 12 MP RGB plus pose plus (probably) the scene depth in the returned ARFrame. Then fuse depth from the dense stream and texture/segment from the stills.
- The 4K format is 16:9 against depth's 4:3 at 1920x1440. If sceneDepth is supported in 4K mode, the depth map would follow the image aspect ratio (Apple says depth "presents the same aspect ratio"). That narrows the vertical FOV, which is a cost when keeping a plate in frame from a 20-30 cm distance on a landscape iPad.
- Intrinsics: in both the 4K and hi-res still cases, use the `camera.intrinsics` and `imageResolution` of the returned ARFrame, not the stream's. Don't reuse one camera_matrix across formats.
- AVFoundation's builtInLiDARDepthCamera has higher-res depth (768x576 with photos) but no ARKit tracking. It is only worth it for a turntable/static-pose pipeline that estimates poses offline.

### Gaps
- No Apple doc found that confirms whether `.sceneDepth` / `.smoothedSceneDepth` are populated in 4K or hi-res-recommended formats on every device, or whether the hi-res frame's `sceneDepth` is temporally aligned with its still. This needs an on-device test: check `supportsFrameSemantics` per format and look for non-nil sceneDepth on hi-res frames.
- No documentation found on RGB/depth misregistration (temporal or spatial) for any format.
- I couldn't get a per-device `supportedVideoFormats` list (resolutions and fps for the iPad Pro M4, iPhone 16/17 Pro). Log `ARWorldTrackingConfiguration.supportedVideoFormats` on the target device.

## Camera control under ARKit (iOS 16+): exposure, focus, white balance, HDR

### Takeaway
`ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera` returns the AVCaptureDevice. After `lockForConfiguration()` you can use standard AVFoundation calls such as `setExposureModeCustom(duration:iso:)`, `setWhiteBalanceModeLocked(with:)` and `setFocusModeLocked(lensPosition:)`. Apple warns that changing settings "may affect tracking performance, particularly on devices with a single rear camera". The property is nil on devices without an ultra-wide camera.

### Cited Findings
- Declaration: `class var configurableCaptureDeviceForPrimaryCamera: AVCaptureDevice?` (iOS 16+).
- The docs say: "Alter the device's settings with caution, as extreme changes can affect ARKit's features that rely on the capturedImage". Also: "This property is nil on devices that aren't equiped with an ultra-wide camera."
- — [configurableCaptureDeviceForPrimaryCamera docs](https://developer.apple.com/documentation/arkit/arconfiguration/configurablecapturedeviceforprimarycamera)
- Apple engineer, June 2026, on motion blur in dim conditions: "switch to manual exposure in challenging capturing situations using setExposureModeCustom(duration:iso:completionhandler:). However, you should modify capture settings with caution, as they may affect tracking performance, particularly on devices with a single rear camera." — [Apple Dev Forums: Motion Blur during ARKit Sessions](https://developer.apple.com/forums/thread/833148)
- Apple engineer: to lock exposure and white balance consistently across frames, get `configurableCaptureDeviceForPrimaryCamera`, "lock that device for configuration and then set exposure or white balance controls." — [Apple Dev Forums: ARKit multi-frame capture](https://developer.apple.com/forums/thread/832172)
- The image "is also actively used by ARKit to analyze the scene. So any changes like strong overexposure might have a negative effect on the output quality of ARKit." You can also trigger focus events through the device. — [WWDC22 Discover ARKit 6](https://developer.apple.com/videos/play/wwdc2022/10126/)
- AVFoundation requires `lockForConfiguration()` and checking that a mode is supported before setting it. `setExposureModeCustom` takes explicit duration and ISO, with "current" sentinel values. — [AVFoundation capture device exposure](https://developer.apple.com/documentation/avfoundation/capture-device-exposure)
- Unity's ARKit wrapper also documents advanced camera configuration as requiring iOS 16+ and an ultra-wide camera. — [Unity ARKit camera docs](https://docs.unity3d.com/Packages/com.unity.xr.arkit@6.3/manual/arkit-camera.html)
- `ARWorldTrackingConfiguration.isAutoFocusEnabled` is "fixed focus or autofocus behavior; …ARKit enables autofocus by default". — [isAutoFocusEnabled docs](https://developer.apple.com/documentation/arkit/arworldtrackingconfiguration/isautofocusenabled)
- HDR: check `videoFormat.isVideoHDRSupported` and set `config.videoHDRAllowed = true`. "Currently, only non-binned video formats support HDR" (so 4K, not 1920x1440). It has a performance impact. — [WWDC22 Discover ARKit 6](https://developer.apple.com/videos/play/wwdc2022/10126/)
- EXIF tags (white balance, exposure, …) are "now available with every ARFrame". — [WWDC22 Discover ARKit 6](https://developer.apple.com/videos/play/wwdc2022/10126/)
- Community reports say per-frame `ARCamera.intrinsics` change during a session, and attribute this to autofocus lens movement. This is not confirmed by Apple. — [TechnetExperts](https://www.technetexperts.com/arkit-camera-intrinsics-sharpness-fixed/), [Apple Dev Forums PhotogrammetrySession thread](https://developer.apple.com/forums/thread/840946)
- Object Capture guidance: "If possible, don't make changes to any camera settings while shooting, including the focal length (zoom), aperture, shutter speed, or ISO." — [Capturing photographs for RealityKit Object Capture](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture)

### Inferences
- Recommended sequence:
  1. Run the session with AF/AE/AWB on.
  2. Wait until the plate is framed and tracking is `.normal`.
  3. Lock white balance at the current gains with `setWhiteBalanceModeLocked(with: device.deviceWhiteBalanceGains)`. Color consistency matters for food segmentation and texture.
  4. Lock focus with `setFocusModeLocked(lensPosition: device.lensPosition)` at the mid-orbit distance.
  5. Use custom exposure with a capped duration, for example ≤ 1/250-1/500 s, and let ISO compensate.
  6. Leave autofocus locked for the whole orbit. Constant focus means constant intrinsics, which makes per-frame intrinsics consistent for multi-view fusion.
- Even with focus locked, still save per-frame `camera.intrinsics`. iPhone/iPad lenses have OIS, which could alter the principal point. This is unverified.
- Locking exposure in a static indoor food scene shouldn't hurt tracking much, because the lighting is constant. The forum warning is about extreme or dim settings and single-rear-camera devices.
- 1920x1440 cannot use HDR. Avoid HDR for reconstruction anyway: tone mapping varies from frame to frame, which works against photometric consistency.
- Store `ARFrame.exifData` (property name to be checked in Xcode; the WWDC talk says only that EXIF is available on every ARFrame) for exposure duration, ISO and WB per frame. This allows blur prediction (exposure × angular velocity) and post-hoc exposure normalization.

### Gaps
- No primary source quantifies how much locked exposure or focus degrades ARKit tracking, or says whether locking focus changes `camera.intrinsics`. Measure this on the target iPad.
- No source found on whether the lens position used by ARKit's depth fusion is affected by a focus lock.
- The exact `ARFrame` EXIF property name wasn't verified in the docs during this session.

## Motion blur: exposure limits, camera speed, frame selection

### Takeaway
Apple gives only qualitative guidance: "move around the object slowly and smoothly". Object Capture stops capturing automatically when the device moves too fast. The supported numeric control is `setExposureModeCustom`, which caps shutter time. At 60 fps the frame interval allows up to ~1/60 s, and at 30 fps (4K) up to ~1/30 s, so dim indoor scenes can blur.

### Cited Findings
- "move around the object slowly and smoothly while holding the camera steadily to avoid blurry images. If you move too fast, automatic capture will stop and remind you to slow down." — [WWDC23 Meet Object Capture for iOS](https://developer.apple.com/videos/play/wwdc2023/10191/)
- In dim outdoor conditions, ARKit's auto-exposure selects "longer shutter speeds" that cause blur. Apple's answer is manual exposure via `setExposureModeCustom`, with the tracking caveat. — [Apple Dev Forums 833148](https://developer.apple.com/forums/thread/833148)
- For hard cases: "put your camera on a tripod… Use a remote trigger". In low light, raise ISO or add light. — [Capturing photographs for Object Capture](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture)

### Inferences
- The exposure ceiling equals the frame interval (1/60 s at 60 fps), which is an AVFoundation constraint. At 25 cm, a 1 px blur on a 1920 px wide image (~65° HFOV) is roughly 0.15 mm lateral at the target per ms. Keeping blur under ~1 px at 1/250 s therefore needs translation speeds of about ≤ 5 cm/s. A 30 cm diameter orbit (~95 cm circumference) at ~5 cm/s takes about 20 s per ring. This is my own estimate, not sourced.
- Frame selection: per frame, compute angular velocity from consecutive ARFrame poses (or the gyro) × exposure duration from EXIF, and add a sharpness metric (Laplacian variance) on the plate ROI. Keep the sharpest frame per ~5° pose bin rather than a fixed every-4th-frame schedule.
- Increasing light, ideally diffuse, is the cleanest fix. It lets auto-exposure pick short shutter times without needing manual exposure.

### Gaps
- Apple publishes no recommended max shutter speed or camera speed for ARKit scanning. No source found on `activeMaxExposureDuration` behavior under ARKit.

## Lighting: diffuse vs direct, IR/sunlight effects

### Takeaway
Use bright, diffuse indoor light: no hard shadows, no specular highlights, no direct sun. Sunlight's IR overlaps the ToF wavelength and adds noise and dropouts to LiDAR depth (shown for comparable ToF/LiDAR sensors). Glossy sauces and reflective plates reduce LiDAR confidence.

### Cited Findings
- "Use diffused lighting if possible. Hard light, such as from an on-camera flash, direct sunlight, or a bare light bulb can cause problems". Bounce light off a wall or ceiling, or diffuse it through thin white fabric. Outdoors, "keep the sun out of your images… or shoot on an overcast day." — [Capturing photographs for Object Capture](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture)
- "We recommend using diffuse lights to minimize reflection or highlights on the object surface." Also: "If it's too dark, you'll receive a reminder". Avoid "reflective, transparent, or contain very thin structures" objects. "For mostly textureless objects… put them against a textured background". — [WWDC23 Meet Object Capture for iOS](https://developer.apple.com/videos/play/wwdc2023/10191/)
- "Challenging surfaces, such as those which are highly reflective or those with high absorption, can lower the accuracy of the depth." — [WWDC20 Explore ARKit 4](https://developer.apple.com/videos/play/wwdc2020/10611/)
- On a comparable ToF sensor (Intel RealSense L515, 860 nm), the same wavelength in sunlight creates noise, causing low-confidence or missing depth. Near objects can oversaturate the receiver. This is not Apple-specific. — [RealSense L515 range article](https://realsenseai.com/?p=12939)
- A 2022 ISPRS paper says strong sunlight degrades depth camera accuracy outdoors. — [ISPRS Archives 2022](https://isprs-archives.copernicus.org/articles/XLVIII-2-W1-2022/101/2022/)

### Inferences
- Indoors under diffuse overhead light, plus a reflector or bounce card to fill shadows, is ideal. Avoid windows with direct sun on the table.
- A matte, textured placemat helps ARKit tracking (plates and tables are often low-texture) and the segmentation background.
- Glossy foods (sauces, soups) and dark absorbing foods will have low-confidence depth. Expect holes there and rely on multi-view fusion or confidence filtering.

### Gaps
- No Apple-published data on the iPad/iPhone LiDAR wavelength, sunlight immunity, or lux thresholds.

## Distance, geometry, orbit patterns and coverage

### Takeaway
Object Capture recommends: fill the frame, orbit at three heights (or three object orientations if the object can be flipped), keep ≥70% overlap between neighbouring shots (failure risk below 50%), and keep overlap between passes. Plated food can't be flipped, so do three elevation rings. Apple gives no minimum LiDAR range. ARKit depth on small objects is around ±1 cm absolute (secondary citation). That is large relative to 5-30 cm food, so multi-view fusion and RGB matter.

### Cited Findings
- "For the best model reconstruction, we recommend completing three scan passes." Also: "If an object cannot be flipped, we suggest capturing it from three different heights". "Part of the object in a scan pass should be captured in previous passes." Flipping is bad for deformable objects. The distance should make the object "fit in the camera frame properly" (the UI warns when too far or too close). — [WWDC23 Meet Object Capture for iOS](https://developer.apple.com/videos/play/wwdc2023/10191/)
- ObjectCaptureSession APIs:
  - States: `ObjectCaptureSession.start(imagesDirectory:configuration:)`, `startDetecting()`, `startCapturing()`.
  - Passes: `userCompletedScanPass`, `beginNewScanPass()`, `beginNewScanPassAfterFlip()`.
  - Configuration: `Configuration.isOverCaptureEnabled` and `checkpointDirectory`.
  - It "collects point cloud data with LiDAR". Supported devices are iPhone 12 Pro and iPad Pro 2021 or later.
  - — [WWDC23 Meet Object Capture for iOS](https://developer.apple.com/videos/play/wwdc2023/10191/)
- "Position sequential images so they have a 70% overlap or more. Anything less than 50% overlap… may fail". Fill "as much of the camera's frame as possible". Take photos "from different angles at different heights". Depth data in images lets RealityKit recover real-world scale. — [Capturing photographs for Object Capture](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture)
- iPhone 12 Pro LiDAR models of small objects (>10 cm sides) reached ~±1 cm absolute accuracy (Luetzenburg et al. 2021). This is a secondary citation. — [Yanchuk, Geodetski vestnik](https://geodetski-vestnik.com/arhiv/70/1/47_Yanchuk.pdf)
- A developer reports ARKit depth values are "basically correct below 1.5m". This is anecdotal. Apple publishes no tolerance figures. — [Apple Dev Forums 773571](https://developer.apple.com/forums/thread/773571), [Apple Dev Forums 812600](https://developer.apple.com/forums/thread/812600)

### Inferences
- For a 5-30 cm plate, use about 25-40 cm, roughly the distance where the plate fills ~60-80% of the frame. Closer helps RGB detail. The dense depth comes from ML fusion of sparse LiDAR points, so very close ranges risk the near-range artefacts seen in ToF sensors. Validate on-device by plotting depth confidence against distance.
- Suggested pattern:
  - Three rings at ~30°, ~50° and ~70° elevation (high elevations see into bowls), plus a near-top-down pass.
  - At 70% overlap, roughly 24-36 viewpoints per ring.
  - About 60-100 keyframes total, over 45-90 s.
  - Make each ring overlap the previous one, and close each loop so ARKit can correct drift.

### Gaps
- No Apple or primary source gives a LiDAR minimum range or a recommended depth-capture distance for ARKit.
- The original Luetzenburg et al. paper wasn't fetched.

## Tracking quality: relocalization, world map, scene reconstruction, drift

### Takeaway
I found no primary source that quantifies ARKit drift over a short orbit. Scene reconstruction (`sceneReconstruction = .mesh`) aggregates depth across frames into a coarse mesh, whose triangles vary in size with surface detail. It is useful as a live coverage/QA preview, not as the final food geometry. Holding ARFrames too long degrades tracking.

### Cited Findings
- Scene geometry: "depth data across multiple frames are aggregated and processed to construct a 3D mesh". "Triangles vary in size to show the optimum detail for each surface." — [WWDC20 Explore ARKit 4](https://developer.apple.com/videos/play/wwdc2020/10611/)
- Retaining ARFrames causes frame drops, and "the ARCamera's tracking state might fall back to limited". — [WWDC22 Discover ARKit 6](https://developer.apple.com/videos/play/wwdc2022/10126/)
- Object Capture pauses when the object leaves the frame. — [WWDC23 Meet Object Capture for iOS](https://developer.apple.com/videos/play/wwdc2023/10191/)

### Inferences
- Record `camera.trackingState` and `trackingStateReason` per frame, and drop frames that aren't `.normal`.
- Copy pixel buffers out quickly, or let a writer queue convert them, so ARFrames are released.
- `.mesh` reconstruction costs compute and is too coarse for 5 cm items. Enable it only if you show a coverage overlay.
- ARWorldMap/relocalization is only needed for multi-session captures. It isn't needed for a single continuous orbit.

### Gaps
- No published numbers on ARKit pose drift (mm/°) for short orbits. Measure with a known-size target such as a checkerboard or ArUco placemat.

## Storage format

### Takeaway
Store depth losslessly as 16-bit PNG in millimetres, or float32, at the native 256x192, along with the confidence map (0/1/2) and per-frame pose and intrinsics. That is what StrayScanner does. Store raw `sceneDepth` (not only smoothed), because smoothing blends frames over time. The RealityKit guidance prefers the highest resolution and RAW where possible. JPEG q0.92 is acceptable, but HEIC is smaller at similar quality.

### Cited Findings
- StrayScanner format:
  - `rgb.mp4` (HEVC).
  - `depth/`: 16-bit grayscale PNG, 256x192, millimetres.
  - `confidence/`: PNG with values 0/1/2.
  - `odometry.csv`: per-frame timestamp, pose and intrinsics fx, fy, cx, cy ("use the odometry.csv columns rather than camera_matrix.csv").
  - `imu.csv` and optional distortion `.bin`.
  - — [StrayScanner docs/format.md](https://github.com/strayrobots/scanner/blob/main/docs/format.md)
- `smoothedSceneDepth` is "similar to sceneDepth except that the framework smoothes the depth data over time to lessen its frame-to-frame delta." — [smoothedSceneDepth docs](https://developer.apple.com/documentation/arkit/arframe/smoothedscenedepth)
- Each depth pixel has an `ARConfidenceLevel` of low, medium or high "to filter depth based on the requirements of your application". Depth is in metres, measured from the camera plane. — [WWDC20 Explore ARKit 4](https://developer.apple.com/videos/play/wwdc2020/10611/)
- For computer vision, "prefer non-filtered depth data"; unfiltered LiDAR depth "excludes low confidence points" (AVFoundation `isFilteringEnabled = false`). — [WWDC22 Discover advancements in iOS camera capture](https://developer.apple.com/videos/play/wwdc2022/110429/)
- "Shoot at the highest resolution your camera supports and use RAW format if possible." — [Capturing photographs for Object Capture](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture)

### Inferences
- Save both raw `sceneDepth` and its confidence. Smoothed depth can smear edges during motion, because it averages over time. That is relevant to the current edge/rim filtering.
- Save per frame: timestamp, `camera.transform`, `camera.intrinsics`, `imageResolution`, `trackingState`, EXIF (exposure duration, ISO, WB), and the gravity vector.
- `rawFeaturePoints` are cheap and useful for drift QA.
- If storage allows, HEIC at ~0.9 quality or PNG for the key hi-res stills avoids JPEG blocking artifacts at food boundaries.

### Gaps
- No source compared JPEG and HEIC artifacts for reconstruction. No Record3D, NeRFCapture or Polycam capture code was reviewed in this session.
