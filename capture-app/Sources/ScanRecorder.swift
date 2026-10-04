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
