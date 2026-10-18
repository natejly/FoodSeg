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
    /// Record pressed, tracking reset, waiting for the readiness gate to pass.
    @Published var isArming = false
    /// Why we are not ready yet (nil = ready). Also shown as a hint while idle.
    @Published var gateMessage: String? = "Hold still…"
    @Published var linearSpeed = 0.0    // m/s
    @Published var angularSpeed = 0.0   // deg/s
    @Published var lightWarning = false
    @Published var failureMessage: String?

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

    // Readiness gate. Apple publishes no numbers; these are practical values to tune on device.
    private let minFeaturePoints = 100
    private let minAmbient = 400.0                 // ARKit ambientIntensity: ~1000 = neutral indoor light
    private let depthRange = 0.25...0.8            // m, median of centre depth
    private let normalHold = 1.0                   // s of continuous .normal
    private var normalSince: TimeInterval?
    private var armedName = ""

    // Motion / runaway. Good scans: 6-18 cm/s. Cap 0.6 m/s sustained ~0.3 s (runaway was >1.5 m/s).
    let runawaySpeed = 0.6
    private let runawayHold = 0.3
    private var lastPose: (t: TimeInterval, m: simd_float4x4)?
    private var overSince: TimeInterval?
    private var overAtFrame = 0

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
        isArming = false
        if isRecording { stopRecording() }
        session.pause()
    }

    // MARK: Recording

    /// Record button: reset tracking, then wait for the readiness gate; saving starts automatically.
    func startRecording(name: String) {
        guard !isRecording, !isArming else { return }
        armedName = name
        failureMessage = nil
        normalSince = nil
        gateMessage = "Hold still…"
        isArming = true
        // Fresh VIO state per scan: a long-running session can diverge (runaway poses).
        start()
    }

    func cancelArming() { isArming = false }

    private func beginRecording(name: String) {
        isArming = false
        do {
            let dir = try ScanStore.newScanDirectory(named: name)
            writer = try FrameWriter(directory: dir)
            frameCounter = 0
            savedFrames = 0
            droppedFrames = 0
            previewQueue.async { self.cloud = PointCloud(); self.previewFrames = 0 }
            previewGeometry = nil
            previewPoints = 0
            currentScanName = dir.lastPathComponent
            lastPose = nil
            overSince = nil
            isRecording = true
        } catch {
            print("Failed to start recording: \(error)")
        }
    }

    func stopRecording(failure: TrackingFailure? = nil) {
        guard isRecording else { return }
        isRecording = false
        let w = writer
        writer = nil
        let n = savedFrames
        writeQueue.async {
            w?.finish(savedFrames: n, failure: failure)
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

        let features = frame.rawFeaturePoints?.points.count ?? 0
        let ambient = frame.lightEstimate?.ambientIntensity
        lightWarning = (ambient ?? .infinity) < minAmbient

        if isArming { updateGate(frame, normal: state == "normal", features: features) }
        guard isRecording, let writer = writer else { return }

        // Speed from poses ~0.1 s apart (60 fps differences are too noisy).
        if let (lt, lm) = lastPose {
            let dt = frame.timestamp - lt
            if dt >= 0.1 {
                let m = frame.camera.transform
                let dp = m.columns.3 - lm.columns.3
                linearSpeed = Double(simd_length(SIMD3(dp.x, dp.y, dp.z))) / dt
                let dq = simd_quatf(lm).inverse * simd_quatf(m)
                angularSpeed = Double(2 * acos(min(1, abs(dq.real)))) * 180 / .pi / dt
                lastPose = (frame.timestamp, m)
                if linearSpeed > runawaySpeed {
                    if overSince == nil { overSince = frame.timestamp; overAtFrame = savedFrames }
                    if frame.timestamp - overSince! >= runawayHold {
                        failureMessage = "Tracking lost — scan stopped, please rescan"
                        stopRecording(failure: TrackingFailure(
                            frame: overAtFrame,
                            reason: String(format: "linear speed %.2f m/s > %.1f m/s for %.1f s while tracking normal",
                                           linearSpeed, runawaySpeed, runawayHold)))
                        return
                    }
                } else { overSince = nil }
            }
        } else { lastPose = (frame.timestamp, frame.camera.transform) }

        // Poses from limited tracking (initializing/excessive motion) are unreliable.
        guard state == "normal" else { return }
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
            exposureOffset: frame.camera.exposureOffset,
            featurePoints: features,
            ambientIntensity: ambient.map(Double.init),
            linearSpeed: linearSpeed,
            angularSpeed: angularSpeed
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

    private func updateGate(_ frame: ARFrame, normal: Bool, features: Int) {
        var msg: String?
        if !normal { normalSince = nil } else if normalSince == nil { normalSince = frame.timestamp }
        if !normal || frame.timestamp - (normalSince ?? frame.timestamp) < normalHold {
            msg = "Hold still…"
        } else if features < minFeaturePoints {
            msg = "Low texture: add a patterned placemat"
        } else if lightWarning {
            msg = "Too dark"
        } else if let d = Self.centreDepth(frame), !depthRange.contains(d) {
            msg = d < depthRange.lowerBound ? "Move farther" : "Move closer"
        }
        if msg != gateMessage { gateMessage = msg }
        if msg == nil { beginRecording(name: armedName) }
    }

    /// Median depth (m) of a 33x33 window at the image centre; nil if no valid samples.
    private static func centreDepth(_ frame: ARFrame) -> Double? {
        guard let map = frame.sceneDepth?.depthMap else { return nil }
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        let stride = CVPixelBufferGetBytesPerRow(map)
        var v: [Float] = []
        for y in (h / 2 - 16)..<(h / 2 + 17) {
            let row = base.advanced(by: y * stride).assumingMemoryBound(to: Float.self)
            for x in (w / 2 - 16)..<(w / 2 + 17) where row[x].isFinite && row[x] > 0 { v.append(row[x]) }
        }
        guard !v.isEmpty else { return nil }
        return Double(v.sorted()[v.count / 2])
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        trackingState = "error: \(error.localizedDescription)"
    }

    func sessionWasInterrupted(_ session: ARSession) {
        trackingState = "interrupted"
        isArming = false
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
