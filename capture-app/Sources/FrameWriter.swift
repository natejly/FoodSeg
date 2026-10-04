import Foundation
import simd
import UIKit

/// One line of frames.jsonl.
struct FrameRecord: Codable {
    let index: Int
    let timestamp: TimeInterval
    let trackingState: String
    /// Row-major 3x3 for the full-resolution color image.
    let intrinsics: [[Float]]
    /// Row-major 4x4 camera-to-world, ARKit convention (+Y up, -Z forward).
    let transform: [[Float]]
    let imageWidth: Int
    let imageHeight: Int
    let depthWidth: Int
    let depthHeight: Int
    let exposureDuration: TimeInterval
    let exposureOffset: Float

    init(index: Int, timestamp: TimeInterval, trackingState: String,
         intrinsics: simd_float3x3, transform: simd_float4x4,
         imageWidth: Int, imageHeight: Int, depthWidth: Int, depthHeight: Int,
         exposureDuration: TimeInterval, exposureOffset: Float) {
        self.index = index
        self.timestamp = timestamp
        self.trackingState = trackingState
        // simd matrices are column-major; emit rows so the JSON reads naturally.
        self.intrinsics = (0..<3).map { r in (0..<3).map { c in intrinsics[c][r] } }
        self.transform = (0..<4).map { r in (0..<4).map { c in transform[c][r] } }
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.depthWidth = depthWidth
        self.depthHeight = depthHeight
        self.exposureDuration = exposureDuration
        self.exposureOffset = exposureOffset
    }
}

struct ScanMeta: Encodable {
    let formatVersion = 1
    let app = "FoodScan"
    let device: String
    let systemVersion: String
    let startedAt: String
    var finishedAt: String?
    var savedFrames: Int?
    let rgbFormat = "jpeg, sensor (landscape) orientation"
    let depthFormat = "float32 little-endian meters, row-major, ARKit sceneDepth"
    let smoothDepthFormat = "float32 little-endian meters, ARKit smoothedSceneDepth"
    let confidenceFormat = "uint8: 0 low, 1 medium, 2 high"
    let cameraConvention = "ARKit: +X right, +Y up, -Z forward; transform is camera-to-world; intrinsics for full-res RGB"
}

/// Writes frames of one scan to its directory. Call only from one serial queue.
final class FrameWriter {
    let directory: URL
    private let jsonlHandle: FileHandle
    private var meta: ScanMeta
    private let encoder = JSONEncoder()

    init(directory: URL) throws {
        self.directory = directory
        let fm = FileManager.default
        for sub in ["rgb", "depth", "smooth_depth", "conf"] {
            try fm.createDirectory(at: directory.appendingPathComponent(sub),
                                   withIntermediateDirectories: true)
        }
        let jsonl = directory.appendingPathComponent("frames.jsonl")
        fm.createFile(atPath: jsonl.path, contents: nil)
        jsonlHandle = try FileHandle(forWritingTo: jsonl)

        meta = ScanMeta(device: UIDevice.current.modelIdentifier,
                        systemVersion: UIDevice.current.systemVersion,
                        startedAt: ISO8601DateFormatter().string(from: Date()))
        try writeMeta()
    }

    func write(record: FrameRecord, jpeg: Data, depth: Data,
               smoothDepth: Data?, confidence: Data?) throws {
        let name = String(format: "%06d", record.index)
        try jpeg.write(to: directory.appendingPathComponent("rgb/\(name).jpg"))
        try depth.write(to: directory.appendingPathComponent("depth/\(name).f32"))
        if let s = smoothDepth {
            try s.write(to: directory.appendingPathComponent("smooth_depth/\(name).f32"))
        }
        if let c = confidence {
            try c.write(to: directory.appendingPathComponent("conf/\(name).u8"))
        }
        var line = try encoder.encode(record)
        line.append(0x0A)
        jsonlHandle.write(line)
    }

    func finish(savedFrames: Int) {
        meta.finishedAt = ISO8601DateFormatter().string(from: Date())
        meta.savedFrames = savedFrames
        try? writeMeta()
        try? jsonlHandle.close()
    }

    private func writeMeta() throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(meta).write(to: directory.appendingPathComponent("meta.json"))
    }
}

extension UIDevice {
    /// e.g. "iPad14,3" — more useful than the marketing name for the dataset.
    var modelIdentifier: String {
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
    }
}
