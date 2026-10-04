// Headless turntable snapshots of a model (usdz/obj/ply) + its bounding box in meters.
//   swiftc -O scripts/objcap/render.swift -o scripts/objcap/render
//   scripts/objcap/render MODEL OUT_PREFIX [EXPORT.obj]
import Foundation
import Metal
import ModelIO
import SceneKit
import SceneKit.ModelIO

let args = CommandLine.arguments
let url = URL(fileURLWithPath: args[1])
let prefix = args[2]

let scene = url.pathExtension == "usdz" ? try SCNScene(url: url) : SCNScene(mdlAsset: MDLAsset(url: url))
let (lo, hi) = scene.rootNode.boundingBox
let size = SIMD3<Float>(Float(hi.x - lo.x), Float(hi.y - lo.y), Float(hi.z - lo.z))
let center = SIMD3<Float>(Float(lo.x + hi.x), Float(lo.y + hi.y), Float(lo.z + hi.z)) / 2
print(String(format: "bbox size (m): %.3f x %.3f x %.3f", size.x, size.y, size.z))

let camera = SCNNode()
camera.camera = SCNCamera()
camera.camera?.zNear = 0.001
scene.rootNode.addChildNode(camera)
let renderer = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
renderer.scene = scene
renderer.pointOfView = camera
renderer.autoenablesDefaultLighting = true
scene.background.contents = NSColor(white: 0.15, alpha: 1)

let radius = simd_length(size) * 1.3
for (i, (az, el)) in [(0.0, 35.0), (90.0, 35.0), (180.0, 35.0), (270.0, 35.0), (0.0, 85.0)].enumerated() {
    let a = Float(az * .pi / 180), e = Float(el * .pi / 180)
    camera.simdPosition = center + radius * SIMD3(cos(e) * sin(a), sin(e), cos(e) * cos(a))
    camera.simdLook(at: center)
    let img = renderer.snapshot(atTime: 0, with: CGSize(width: 800, height: 600), antialiasingMode: .multisampling4X)
    let rep = NSBitmapImageRep(data: img.tiffRepresentation!)!
    try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: "\(prefix)_\(i).png"))
}
print("wrote \(prefix)_0..4.png")
if args.count > 3 {
    let asset = MDLAsset(scnScene: scene)
    try asset.export(to: URL(fileURLWithPath: args[3]))
    print("exported \(args[3])")
}
