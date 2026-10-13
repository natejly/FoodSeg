import ARKit
import SceneKit
import SwiftUI

struct ContentView: View {
    @StateObject private var recorder = ScanRecorder()
    @State private var scanName = ""
    @State private var showScans = false

    @State private var recenter = 0

    var body: some View {
        HStack(spacing: 2) {
            cameraPane
            scanPane
        }
        .background(.black)
        .onAppear { recorder.start() }
        .onDisappear { recorder.pause() }
        .sheet(isPresented: $showScans) { ScanListView() }
        .alert("LiDAR depth not available", isPresented: .constant(!recorder.depthSupported)) {
            Button("OK") {}
        } message: {
            Text("This device does not report ARKit scene depth. Use a LiDAR iPad Pro or iPhone Pro.")
        }
    }

    private var cameraPane: some View {
        ZStack {
            ARPreview(session: recorder.session)
            VStack {
                statusBar
                if let advice = recorder.guide.advice {
                    Text(advice)
                        .font(.headline)
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(adviceColor.opacity(0.85), in: Capsule())
                        .foregroundStyle(.white)
                }
                Spacer()
                if recorder.isRecording { guideBar }
                controls
            }
            .padding()
        }
        .clipped()
    }

    /// Live point cloud of what has been saved so far in this scan.
    private var scanPane: some View {
        ZStack(alignment: .top) {
            CloudView(geometry: recorder.previewGeometry, recenter: recenter)
            if recorder.previewGeometry == nil {
                Text(recorder.isRecording ? "Building scan…" : "Press record — the scan builds up here")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxHeight: .infinity)
            }
            HStack {
                Text("\(recorder.previewPoints.formatted()) points")
                    .font(.caption.monospaced())
                Spacer()
                Button { recenter += 1 } label: { Image(systemName: "scope") }
            }
            .padding(8)
            .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
            .foregroundStyle(.white)
            .padding()
        }
        .clipped()
    }

    private var statusBar: some View {
        HStack {
            Label(recorder.trackingState, systemImage: trackingIcon)
                .font(.caption.monospaced())
            Spacer()
            if recorder.isRecording {
                Text("● \(recorder.savedFrames) frames")
                    .foregroundStyle(.red)
                    .font(.caption.monospaced())
                if recorder.droppedFrames > 0 {
                    Text("dropped \(recorder.droppedFrames)")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.orange)
                }
            }
        }
        .padding(8)
        .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
        .foregroundStyle(.white)
    }

    private var adviceColor: Color {
        recorder.guide.tooDark || recorder.guide.tooFast ? .orange : .blue
    }

    /// Live coverage / blur readout while recording.
    private var guideBar: some View {
        let g = recorder.guide
        return HStack(spacing: 16) {
            Text(String(format: "%.0f s", g.seconds))
            Label("\(Int((g.orbitCoverage * 100).rounded()))%", systemImage: "arrow.triangle.2.circlepath")
                .foregroundStyle(g.orbitCoverage >= 0.9 ? Color.green : Color.white)
            Label("top", systemImage: g.topDown ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(g.topDown ? Color.green : Color.white)
            Text(String(format: "blur %.1f px", g.blurPixels))
                .foregroundStyle(g.tooFast ? Color.orange : Color.white)
            Text(String(format: "%.0f ms", g.exposureMs))
                .foregroundStyle(g.tooDark ? Color.orange : Color.white)
        }
        .font(.caption.monospaced())
        .padding(8)
        .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
        .foregroundStyle(.white)
    }

    private var trackingIcon: String {
        recorder.trackingState == "normal" ? "checkmark.circle" : "exclamationmark.triangle"
    }

    private var controls: some View {
        VStack(spacing: 12) {
            if !recorder.isRecording {
                TextField("meal name, e.g. chicken_rice_01", text: $scanName)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
            } else if let name = recorder.currentScanName {
                Text(name).font(.caption.monospaced()).foregroundStyle(.white)
            }

            HStack(spacing: 24) {
                Button {
                    showScans = true
                } label: {
                    Image(systemName: "folder")
                        .font(.title2)
                        .frame(width: 56, height: 56)
                        .background(.black.opacity(0.5), in: Circle())
                        .foregroundStyle(.white)
                }
                .disabled(recorder.isRecording)

                Button {
                    if recorder.isRecording {
                        recorder.stopRecording()
                    } else {
                        recorder.startRecording(name: scanName)
                    }
                } label: {
                    ZStack {
                        Circle().stroke(.white, lineWidth: 4).frame(width: 80, height: 80)
                        if recorder.isRecording {
                            RoundedRectangle(cornerRadius: 6).fill(.red).frame(width: 34, height: 34)
                        } else {
                            Circle().fill(.red).frame(width: 64, height: 64)
                        }
                    }
                }
                .disabled(recorder.trackingState != "normal" && !recorder.isRecording)

                // Spacer of the same width to keep the record button centered.
                Color.clear.frame(width: 56, height: 56)
            }
        }
    }
}

/// Live camera preview backed by the shared ARSession.
struct ARPreview: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = session
        view.automaticallyUpdatesLighting = false
        view.debugOptions = [.showFeaturePoints]
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

struct ScanListView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var scans: [ScanSummary] = []

    var body: some View {
        NavigationStack {
            List {
                if scans.isEmpty {
                    Text("No scans yet. Scans are also visible in the Files app under On My iPad > FoodScan > scans.")
                        .foregroundStyle(.secondary)
                }
                ForEach(scans) { scan in
                    NavigationLink {
                        ScanDetailView(scan: scan)
                    } label: {
                        VStack(alignment: .leading) {
                            Text(scan.name).font(.body.monospaced())
                            Text("\(scan.frameCount) frames · \(String(format: "%.0f", scan.sizeMB)) MB")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete { idx in
                    idx.map { scans[$0] }.forEach(ScanStore.delete)
                    scans = ScanStore.listScans()
                }
            }
            .navigationTitle("Scans")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear { scans = ScanStore.listScans() }
        }
    }
}

/// Orbitable point cloud of a saved scan, with raw (.zip) and point cloud (.ply) export.
struct ScanDetailView: View {
    let scan: ScanSummary
    @State private var cloud: PointCloud?
    @State private var geometry: SCNGeometry?
    @State private var shareURL: URL?
    @State private var exporting = false
    @State private var recenter = 0

    var body: some View {
        ZStack(alignment: .bottom) {
            CloudView(geometry: geometry, recenter: recenter)
            if let cloud {
                Text("\(cloud.count.formatted()) points · \(scan.frameCount) frames")
                    .font(.caption.monospaced())
                    .padding(8)
                    .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                    .foregroundStyle(.white)
                    .padding()
            } else {
                ProgressView("Building point cloud…")
                    .tint(.white).foregroundStyle(.white)
                    .frame(maxHeight: .infinity)
            }
        }
        .background(.black)
        .navigationTitle(scan.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button { recenter += 1 } label: { Image(systemName: "scope") }
                Menu {
                    Button("Raw scan (.zip)") { exportZip() }
                    Button("Point cloud (.ply)") { exportPLY() }.disabled(cloud == nil)
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .disabled(exporting)
            }
        }
        .sheet(item: $shareURL) { url in
            ShareSheet(items: [url])
        }
        .onAppear {
            guard cloud == nil else { return }
            DispatchQueue.global(qos: .userInitiated).async {
                let c = PointCloud.load(scanDirectory: scan.url)
                let g = c.geometry()
                DispatchQueue.main.async { cloud = c; geometry = g }
            }
        }
    }

    private func exportZip() {
        exporting = true
        ScanStore.zip(scan) { url in
            DispatchQueue.main.async {
                exporting = false
                shareURL = url
            }
        }
    }

    private func exportPLY() {
        guard let cloud else { return }
        exporting = true
        DispatchQueue.global(qos: .userInitiated).async {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(scan.name).ply")
            let ok = (try? cloud.plyData().write(to: url)) != nil
            DispatchQueue.main.async {
                exporting = false
                shareURL = ok ? url : nil
            }
        }
    }
}

extension URL: Identifiable {
    public var id: String { absoluteString }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
