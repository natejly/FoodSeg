import CoreImage
import Foundation
import SceneKit
import SwiftUI
import simd

/// Voxel-downsampled colored point cloud built by back-projecting depth frames.
/// Used for the live preview and for viewing/exporting saved scans.
/// Not thread-safe: touch it from one queue only.
final class PointCloud {
    /// Tuning knobs: depth pixel step, voxel edge (m), max depth (m).
    static let pixelStep = 2
    static let voxelSize: Float = 0.004
    static let maxDepth: Float = 1.5

    private(set) var positions: [Float] = []   // packed xyz
    private(set) var colors: [Float] = []      // packed rgb, 0...1
    private var voxels = Set<Int64>()

    var count: Int { voxels.count }

    /// - Parameters:
    ///   - depth: float32 meters, depthWidth x depthHeight, row-major.
    ///   - confidence: uint8 per depth pixel; only high (2) is kept. nil keeps all.
    ///   - rgba: RGBA8 color resampled to depth resolution.
    func add(record r: FrameRecord, depth: Data, confidence: Data?, rgba: [UInt8]) {
        let w = r.depthWidth, h = r.depthHeight
        let sx = Float(w) / Float(r.imageWidth), sy = Float(h) / Float(r.imageHeight)
        let fx = r.intrinsics[0][0] * sx, cx = r.intrinsics[0][2] * sx
        let fy = r.intrinsics[1][1] * sy, cy = r.intrinsics[1][2] * sy
        let t = r.transform
        let m = simd_float4x4(rows: (0..<4).map { SIMD4(t[$0][0], t[$0][1], t[$0][2], t[$0][3]) })

        depth.withUnsafeBytes { (d: UnsafeRawBufferPointer) in
            let d = d.bindMemory(to: Float.self)
            for v in Swift.stride(from: 0, to: h, by: Self.pixelStep) {
                for u in Swift.stride(from: 0, to: w, by: Self.pixelStep) {
                    let i = v * w + u
                    if let c = confidence, c[i] < 2 { continue }
                    let z = d[i]
                    guard z > 0, z < Self.maxDepth else { continue }
                    // ARKit camera: +X right, +Y up, -Z forward; image v grows downward.
                    let p = m * SIMD4((Float(u) - cx) / fx * z, -(Float(v) - cy) / fy * z, -z, 1)
                    let k = SIMD3<Int64>(Int64((p.x / Self.voxelSize).rounded(.down)),
                                         Int64((p.y / Self.voxelSize).rounded(.down)),
                                         Int64((p.z / Self.voxelSize).rounded(.down)))
                    // 21 bits per axis: ±4 km at 4 mm voxels.
                    let key = ((k.x & 0x1FFFFF) << 42) | ((k.y & 0x1FFFFF) << 21) | (k.z & 0x1FFFFF)
                    guard voxels.insert(key).inserted else { continue }
                    positions += [p.x, p.y, p.z]
                    colors += [Float(rgba[i * 4]) / 255, Float(rgba[i * 4 + 1]) / 255, Float(rgba[i * 4 + 2]) / 255]
                }
            }
        }
    }

    func geometry() -> SCNGeometry? {
        guard count > 0 else { return nil }
        let pos = SCNGeometrySource(data: Data(bytes: positions, count: positions.count * 4),
                                    semantic: .vertex, vectorCount: count, usesFloatComponents: true,
                                    componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let col = SCNGeometrySource(data: Data(bytes: colors, count: colors.count * 4),
                                    semantic: .color, vectorCount: count, usesFloatComponents: true,
                                    componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let element = SCNGeometryElement(indices: Array(0..<Int32(count)), primitiveType: .point)
        element.pointSize = 4
        element.minimumPointScreenSpaceRadius = 1
        element.maximumPointScreenSpaceRadius = 6
        let g = SCNGeometry(sources: [pos, col], elements: [element])
        g.firstMaterial?.lightingModel = .constant
        return g
    }

    /// Binary little-endian PLY, opens in MeshLab / CloudCompare / Blender.
    func plyData() -> Data {
        var out = Data("""
        ply
        format binary_little_endian 1.0
        element vertex \(count)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """.utf8)
        for i in 0..<count {
            withUnsafeBytes(of: positions[i * 3]) { out.append(contentsOf: $0) }
            withUnsafeBytes(of: positions[i * 3 + 1]) { out.append(contentsOf: $0) }
            withUnsafeBytes(of: positions[i * 3 + 2]) { out.append(contentsOf: $0) }
            out.append(contentsOf: (0..<3).map { UInt8(colors[i * 3 + $0] * 255) })
        }
        return out
    }

    /// RGBA8 resample of an image to depth resolution so pixel i lines up with depth pixel i.
    static func rgba(_ image: CIImage, width: Int, height: Int, context: CIContext) -> [UInt8] {
        let scaled = image.transformed(by: CGAffineTransform(
            scaleX: CGFloat(width) / image.extent.width, y: CGFloat(height) / image.extent.height))
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        context.render(scaled, toBitmap: &buf, rowBytes: width * 4,
                       bounds: CGRect(x: 0, y: 0, width: width, height: height),
                       format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        return buf
    }

    /// Rebuilds the cloud of a saved scan from disk.
    // ponytail: uses every 2nd frame (voxels dedupe the overlap); drop frameStep to 1 if sparse.
    static func load(scanDirectory dir: URL, frameStep: Int = 2) -> PointCloud {
        let cloud = PointCloud()
        let context = CIContext()
        guard let text = try? String(contentsOf: dir.appendingPathComponent("frames.jsonl"), encoding: .utf8) else {
            return cloud
        }
        let records = text.split(separator: "\n").compactMap {
            try? JSONDecoder().decode(FrameRecord.self, from: Data($0.utf8))
        }
        for r in Swift.stride(from: 0, to: records.count, by: frameStep).map({ records[$0] }) {
            let name = String(format: "%06d", r.index)
            guard let depth = try? Data(contentsOf: dir.appendingPathComponent("depth/\(name).f32")),
                  let image = CIImage(contentsOf: dir.appendingPathComponent("rgb/\(name).jpg")) else { continue }
            let conf = try? Data(contentsOf: dir.appendingPathComponent("conf/\(name).u8"))
            cloud.add(record: r, depth: depth, confidence: conf,
                      rgba: rgba(image, width: r.depthWidth, height: r.depthHeight, context: context))
        }
        return cloud
    }
}

/// Orbitable SceneKit view of a point cloud. Auto-frames the cloud as it grows
/// until the user touches it; `recenter` toggles auto-framing back on.
struct CloudView: UIViewRepresentable {
    let geometry: SCNGeometry?
    var recenter = 0

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = SCNScene()
        view.backgroundColor = .black
        view.allowsCameraControl = true
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.worldUp = SCNVector3(0, 1, 0)
        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.camera?.zNear = 0.005
        view.scene?.rootNode.addChildNode(camera)
        view.pointOfView = camera
        view.scene?.rootNode.addChildNode(context.coordinator.node)
        let touch = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.touched))
        touch.delegate = context.coordinator
        touch.cancelsTouchesInView = false
        view.addGestureRecognizer(touch)
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        let c = context.coordinator
        if recenter != c.lastRecenter { c.lastRecenter = recenter; c.autoFrame = true }
        c.node.geometry = geometry
        guard c.autoFrame, geometry != nil, let pov = view.pointOfView else { return }
        // Look down at the cloud from the front, ~45° above, at a distance that fits it.
        let (lo, hi) = c.node.boundingBox
        let center = SIMD3<Float>((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
        let radius = max(simd_length(SIMD3<Float>(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)) / 2, 0.05)
        pov.simdPosition = center + simd_normalize(SIMD3<Float>(0, 1, 1)) * radius * 2.5
        pov.simdLook(at: center)
        view.defaultCameraController.target = SCNVector3(center)
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        let node = SCNNode()
        var autoFrame = true
        var lastRecenter = 0

        @objc func touched() { autoFrame = false }

        func gestureRecognizer(_ g: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool { true }
    }
}
