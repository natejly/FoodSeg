// Apple Object Capture reconstruction of a FoodScan scan on macOS.
// Feeds each RGB frame with its LiDAR depth (gives metric scale) to RealityKit.
//
//   swiftc -O scripts/objcap/main.swift -o scripts/objcap/objcap
//   scripts/objcap/objcap SCAN_DIR OUT.usdz [preview|reduced|medium|full|raw] [mask|nomask] [step]
//
// Also writes OUT.poses.json (Object Capture camera poses per frame index) so
// the model can be aligned to ARKit world coordinates afterwards.
import CoreImage
import Foundation
import RealityKit

let args = CommandLine.arguments
guard args.count >= 3 else {
    print("usage: objcap SCAN_DIR OUT.usdz [detail] [mask|nomask] [step]"); exit(2)
}
let scanDir = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])
let detail: PhotogrammetrySession.Request.Detail = [
    "preview": .preview, "reduced": .reduced, "medium": .medium, "full": .full, "raw": .raw,
][args.count > 3 ? args[3] : "full"] ?? .full
let masking = (args.count > 4 ? args[4] : "mask") == "mask"
let step = args.count > 5 ? Int(args[5]) ?? 1 : 1

struct Record: Decodable { let index: Int; let depthWidth: Int; let depthHeight: Int }
let records = try String(contentsOf: scanDir.appendingPathComponent("frames.jsonl"), encoding: .utf8)
    .split(separator: "\n").map { try JSONDecoder().decode(Record.self, from: Data($0.utf8)) }
    .enumerated().filter { $0.offset % step == 0 }.map(\.element)

let ci = CIContext()

func pixelBuffer(_ w: Int, _ h: Int, _ format: OSType) -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(nil, w, h, format,
                        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
    return pb!
}

func sample(_ r: Record) -> PhotogrammetrySample? {
    let name = String(format: "%06d", r.index)
    guard let image = CIImage(contentsOf: scanDir.appendingPathComponent("rgb/\(name).jpg")),
          let depth = try? Data(contentsOf: scanDir.appendingPathComponent("depth/\(name).f32"))
    else { return nil }
    let rgb = pixelBuffer(Int(image.extent.width), Int(image.extent.height), kCVPixelFormatType_32BGRA)
    ci.render(image, to: rgb)

    let d = pixelBuffer(r.depthWidth, r.depthHeight, kCVPixelFormatType_DepthFloat32)
    CVPixelBufferLockBaseAddress(d, [])
    let base = CVPixelBufferGetBaseAddress(d)!, rowBytes = CVPixelBufferGetBytesPerRow(d)
    depth.withUnsafeBytes { src in
        for row in 0..<r.depthHeight {
            base.advanced(by: row * rowBytes).copyMemory(
                from: src.baseAddress!.advanced(by: row * r.depthWidth * 4), byteCount: r.depthWidth * 4)
        }
    }
    CVPixelBufferUnlockBaseAddress(d, [])

    var s = PhotogrammetrySample(id: r.index, image: rgb)
    s.depthDataMap = d
    return s
}

var config = PhotogrammetrySession.Configuration()
config.sampleOrdering = .sequential
config.featureSensitivity = .high      // dim, low-contrast captures
config.isObjectMaskingEnabled = masking

let session = try PhotogrammetrySession(input: AnySequence(records.lazy.compactMap(sample)),
                                        configuration: config)
print("\(records.count) samples, detail \(args.count > 3 ? args[3] : "full"), masking \(masking)")

let posesURL = outURL.deletingPathExtension().appendingPathExtension("poses.json")
Task {
    for try await output in session.outputs {
        switch output {
        case .requestProgress(_, let f):
            print(String(format: "progress %.0f%%", f * 100))
        case .requestComplete(_, .poses(let poses)):
            // id -> row-major 4x4 camera-to-world in Object Capture's frame.
            var out: [String: [[Float]]] = [:]
            for (id, p) in poses.posesBySample {
                let m = p.transform.matrix
                out[String(id)] = (0..<4).map { r in (0..<4).map { c in m[c][r] } }
            }
            try JSONEncoder().encode(out).write(to: posesURL)
            print("poses: \(out.count)/\(records.count) frames registered -> \(posesURL.path)")
        case .requestComplete(_, let result):
            print("complete: \(result)")
        case .requestError(_, let e):
            print("error: \(e)")
        case .invalidSample(let id, let reason):
            print("invalid sample \(id): \(reason)")
        case .skippedSample(let id):
            print("skipped sample \(id)")
        case .processingComplete:
            print("done"); exit(0)
        default:
            break
        }
    }
}
try session.process(requests: [.poses, .modelFile(url: outURL, detail: detail)])
RunLoop.main.run()
