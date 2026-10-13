import ARKit
import Combine
import CoreImage
import Foundation
import SceneKit
import UIKit

/// Drives the ARSession and, while recording, writes every Nth frame to disk.
///
/// On-disk layout of one scan (all indices zero-padded to 6 digits):
///
///     Documents/scans/<name>/
///       meta.json              device, resolutions, conventions
///       frames.jsonl           one JSON object per saved frame (see FrameRecord)
///       rgb/000000.jpg         1920x1440 color, sensor (landscape) orientation
///       depth/000000.f32       256x192 float32 little-endian, meters, row-major
///       smooth_depth/000000.f32  same, ARKit temporally smoothed depth
///       conf/000000.u8         256x192 uint8: 0 low, 1 medium, 2 high
///
/// Camera convention is ARKit's: +X right, +Y up, -Z forward (OpenGL style),
/// transform is camera-to-world. Intrinsics are for the full-resolution color
/// image; scale by (depthW/rgbW, depthH/rgbH) to use with the depth map.
final class ScanRecorder: NSObject, ObservableObject, ARSessionDelegate {

    // MARK: Published state for the UI
    @Published var isRecording = false
    @Published var savedFrames = 0
    @Published var droppedFrames = 0
    @Published var trackingState = "initializing"
    @Published var depthSupported = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    @Published var currentScanName: String?
    /// Point cloud of the frames saved so far in the current scan.
    @Published var previewGeometry: SCNGeometry?
    @Published var previewPoints = 0

    // Live capture-quality guidance (see CaptureGuide).
    @Published var guide = CaptureGuide.Status()
    private var guideState = CaptureGuide()
    private var lastGuidePublish: TimeInterval = 0

    /// Save every `stride`-th ARFrame. ARKit runs at 60 fps, so 4 -> 15 fps.
    var stride = 4

    let session = ARSession()

    private var frameCounter = 0
    private var writer: FrameWriter?
    private let writeQueue = DispatchQueue(label: "foodscan.write", qos: .userInitiated)
    private var pendingWrites = 0
    private let maxPendingWrites = 6
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var cloud = PointCloud()
    private let previewQueue = DispatchQueue(label: "foodscan.preview", qos: .utility)
    private var previewBusy = false
    private var previewFrames = 0

    override init() {
        super.init()
        session.delegate = self
    }

    // MARK: Session lifecycle

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        var semantics: ARConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            semantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            semantics.insert(.smoothedSceneDepth)
        }
        config.frameSemantics = semantics
        // Keep the standard 1920x1440 stream. High-res capture has reported
        // RGB/depth misregistration on some iPad Pro models.
        config.isAutoFocusEnabled = true
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func pause() {
        if isRecording { stopRecording() }
        session.pause()
    }

    // MARK: Recording

    func startRecording(name: String) {
        guard !isRecording else { return }
        do {
            let dir = try ScanStore.newScanDirectory(named: name)
            writer = try FrameWriter(directory: dir)
            frameCounter = 0
            savedFrames = 0
            droppedFrames = 0
            previewQueue.async { self.cloud = PointCloud(); self.previewFrames = 0 }
            previewGeometry = nil
            previewPoints = 0
            guideState = CaptureGuide()
            guide = CaptureGuide.Status()
            currentScanName = dir.lastPathComponent
            isRecording = true
        } catch {
            print("Failed to start recording: \(error)")
        }
    }

    func stopRecording() {
        guard isRecording else { return }
        isRecording = false
        let w = writer
        writer = nil
        writeQueue.async {
            w?.finish(savedFrames: self.savedFrames)
        }
        previewQueue.async {
            let g = self.cloud.geometry()
            DispatchQueue.main.async { self.previewGeometry = g }
        }
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let state: String
        switch frame.camera.trackingState {
        case .normal: state = "normal"
        case .notAvailable: state = "not available"
        case .limited(let reason):
            switch reason {
            case .initializing: state = "limited: initializing"
            case .excessiveMotion: state = "limited: excessive motion"
            case .insufficientFeatures: state = "limited: insufficient features"
            case .relocalizing: state = "limited: relocalizing"
            @unknown default: state = "limited"
            }
        }
        if state != trackingState { trackingState = state }

        // Publish at ~5 Hz: every publish re-renders ContentView, including the cloud pane.
        let status = guideState.update(frame: frame, recording: isRecording)
        if status != guide && (frame.timestamp - lastGuidePublish > 0.2 || status.advice != guide.advice) {
            guide = status
            lastGuidePublish = frame.timestamp
        }

        guard isRecording, let writer = writer else { return }
        frameCounter += 1
        guard frameCounter % stride == 0 else { return }

        guard let depth = frame.sceneDepth else { return }

        // Back-pressure: if the writer is behind, drop rather than stall ARKit.
        guard pendingWrites < maxPendingWrites else {
            droppedFrames += 1
            return
        }

        // Copy everything we need off the ARFrame now; the frame's buffers are
        // recycled by ARKit after this callback returns.
        let index = savedFrames
        let record = FrameRecord(
            index: index,
            timestamp: frame.timestamp,
            trackingState: state,
            intrinsics: frame.camera.intrinsics,
            transform: frame.camera.transform,
            imageWidth: Int(frame.camera.imageResolution.width),
            imageHeight: Int(frame.camera.imageResolution.height),
            depthWidth: CVPixelBufferGetWidth(depth.depthMap),
            depthHeight: CVPixelBufferGetHeight(depth.depthMap),
            exposureDuration: frame.camera.exposureDuration,
            exposureOffset: frame.camera.exposureOffset
        )
        let depthData = PixelBufferCopy.floatPlane(depth.depthMap)
        let confData = depth.confidenceMap.map { PixelBufferCopy.bytePlane($0) }
        let smoothData = frame.smoothedSceneDepth.map { PixelBufferCopy.floatPlane($0.depthMap) }
        let ciImage = CIImage(cvPixelBuffer: frame.capturedImage)

        // Live preview: skip frames while the previous one is still being fused.
        if !previewBusy {
            previewBusy = true
            previewQueue.async { [ciContext] in
                let rgba = PointCloud.rgba(ciImage, width: record.depthWidth, height: record.depthHeight,
                                           context: ciContext)
                self.cloud.add(record: record, depth: depthData, confidence: confData, rgba: rgba)
                self.previewFrames += 1
                // Rebuilding the SceneKit geometry is the expensive part; ~3 Hz is plenty.
                let geometry = self.previewFrames % 5 == 1 ? self.cloud.geometry() : nil
                let points = self.cloud.count
                DispatchQueue.main.async {
                    self.previewBusy = false
                    self.previewPoints = points
                    if let g = geometry { self.previewGeometry = g }
                }
            }
        }

        savedFrames += 1
        pendingWrites += 1
        writeQueue.async { [ciContext] in
            defer { DispatchQueue.main.async { self.pendingWrites -= 1 } }
            guard let jpeg = ciContext.jpegRepresentation(
                of: ciImage,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.92]
            ) else { return }
            do {
                try writer.write(record: record, jpeg: jpeg, depth: depthData,
                                 smoothDepth: smoothData, confidence: confData)
            } catch {
                print("write failed for frame \(index): \(error)")
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        trackingState = "error: \(error.localizedDescription)"
    }

    func sessionWasInterrupted(_ session: ARSession) {
        trackingState = "interrupted"
        if isRecording { stopRecording() }
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        start()
    }
}

/// Copies a CVPixelBuffer plane into a tightly packed Data (handles row padding).
enum PixelBufferCopy {
    static func floatPlane(_ buffer: CVPixelBuffer) -> Data {
        copyPlane(buffer, bytesPerPixel: 4)
    }

    static func bytePlane(_ buffer: CVPixelBuffer) -> Data {
        copyPlane(buffer, bytesPerPixel: 1)
    }

    private static func copyPlane(_ buffer: CVPixelBuffer, bytesPerPixel: Int) -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = width * bytesPerPixel
        var data = Data(count: rowBytes * height)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return data }
        data.withUnsafeMutableBytes { dst in
            for row in 0..<height {
                let src = base.advanced(by: row * bytesPerRow)
                dst.baseAddress!.advanced(by: row * rowBytes)
                    .copyMemory(from: src, byteCount: rowBytes)
            }
        }
        return data
    }
}

/// Turns ARKit poses into live scanning advice. Fed every ARFrame on the main thread.
///
/// Scan 20261002_210342 covered only 203 deg of azimuth in 10.8 s while moving at a median
/// 18 cm/s in dim light; its LiDAR sides were 4-6 mm off and RGB was blurred. This makes
/// those three things (coverage, duration, blur) visible while scanning.
struct CaptureGuide {
    struct Status: Equatable {
        /// Estimated motion blur in RGB pixels: speed x exposure x focal length / distance.
        var blurPixels: Float = 0
        var exposureMs: Float = 0
        /// Fraction of 10-degree azimuth bins seen around the target.
        var orbitCoverage: Float = 0
        /// Seen from >= 70 deg elevation (the top-down pass).
        var topDown = false
        var seconds: Double = 0

        static let maxBlurPixels: Float = 4
        /// ARKit's 60 fps stream caps exposure at ~16.7 ms, so dim light shows up as
        /// exposure pinned near that cap. 12 ms is a first guess; calibrate it against
        /// exposureDuration in frames.jsonl of good and bad scans.
        static let maxExposureMs: Float = 12
        var tooFast: Bool { blurPixels > Self.maxBlurPixels }
        var tooDark: Bool { exposureMs > Self.maxExposureMs }

        /// One instruction at a time, most important first.
        var advice: String? {
            if tooDark { return "Too dark: add light (exposure \(Int(exposureMs.rounded())) ms)" }
            if tooFast { return "Slow down" }
            if seconds > 0 && orbitCoverage < 0.9 { return "Keep circling the plate" }
            if seconds > 0 && !topDown { return "Now a pass from directly above" }
            if seconds > 0 && seconds < 30 { return "Good. Keep going a bit longer" }
            return nil
        }
    }

    private static let bins = 36
    private var pivot: simd_float3?
    private var seen = [Bool](repeating: false, count: CaptureGuide.bins)
    private var lastPos: simd_float3?
    private var lastTime: TimeInterval = 0
    private var startTime: TimeInterval?
    private var speed: Float = 0     // m/s, smoothed
    private var status = Status()

    mutating func update(frame: ARFrame, recording: Bool) -> Status {
        let cam = frame.camera
        let t = cam.transform
        let pos = SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
        status.exposureMs = Float(cam.exposureDuration * 1000)

        if let last = lastPos, frame.timestamp > lastTime {
            let v = simd_distance(pos, last) / Float(frame.timestamp - lastTime)
            speed = 0.8 * speed + 0.2 * v
        }
        lastPos = pos
        lastTime = frame.timestamp

        // Target distance: centre pixel of the LiDAR depth, else the protocol's 30 cm.
        var dist: Float = 0.3
        if let d = frame.sceneDepth?.depthMap, let c = Self.centreDepth(d), c > 0.05 { dist = c }
        let fx = cam.intrinsics[0][0]
        status.blurPixels = speed * Float(cam.exposureDuration) * fx / dist

        guard recording else {
            pivot = nil; startTime = nil
            seen = [Bool](repeating: false, count: Self.bins)
            status.orbitCoverage = 0; status.topDown = false; status.seconds = 0
            return status
        }
        if startTime == nil { startTime = frame.timestamp }
        status.seconds = frame.timestamp - (startTime ?? frame.timestamp)
        // Orbit centre: where the first recorded frame's optical axis hits the scene.
        if pivot == nil {
            let forward = -SIMD3<Float>(t.columns.2.x, t.columns.2.y, t.columns.2.z)
            pivot = pos + forward * dist
        }
        let r = pos - pivot!
        let len = simd_length(r)
        guard len > 0.05 else { return status }
        // World is gravity-aligned with +Y up, so azimuth is in the XZ plane.
        let elevation = asin(max(-1, min(1, r.y / len))) * 180 / .pi
        if elevation >= 70 { status.topDown = true }
        if elevation < 75 {   // azimuth is meaningless when looking straight down
            let az = atan2(r.z, r.x) + .pi
            let b = min(Self.bins - 1, Int(az / (2 * .pi) * Float(Self.bins)))
            seen[b] = true
        }
        status.orbitCoverage = Float(seen.filter { $0 }.count) / Float(Self.bins)
        return status
    }

    private static func centreDepth(_ buf: CVPixelBuffer) -> Float? {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return nil }
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
        let row = base.advanced(by: (h / 2) * CVPixelBufferGetBytesPerRow(buf))
        let v = row.assumingMemoryBound(to: Float32.self)[w / 2]
        return v.isFinite ? v : nil
    }
}
