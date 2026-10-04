import Foundation

struct ScanSummary: Identifiable {
    let url: URL
    var id: String { url.lastPathComponent }
    var name: String { url.lastPathComponent }
    var frameCount: Int
    var sizeMB: Double
}

/// Scans live in Documents/scans so they show up in the Files app
/// (UIFileSharingEnabled) and in Finder when the iPad is plugged in.
enum ScanStore {
    static var root: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("scans", isDirectory: true)
    }

    static func newScanDirectory(named name: String) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        let stamp = Self.stampFormatter.string(from: Date())
        let safe = name.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: "[^A-Za-z0-9_-]+", with: "_", options: .regularExpression)
        let dir = root.appendingPathComponent("\(stamp)_\(safe.isEmpty ? "scan" : safe)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func listScans() -> [ScanSummary] {
        let fm = FileManager.default
        guard let dirs = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else {
            return []
        }
        return dirs
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .map { dir in
                let rgb = dir.appendingPathComponent("rgb")
                let frames = (try? fm.contentsOfDirectory(atPath: rgb.path).count) ?? 0
                return ScanSummary(url: dir, frameCount: frames, sizeMB: directorySizeMB(dir))
            }
            .sorted { $0.name > $1.name }
    }

    static func delete(_ scan: ScanSummary) {
        try? FileManager.default.removeItem(at: scan.url)
    }

    /// Zips a scan folder into a temp file for the share sheet (AirDrop, Files).
    static func zip(_ scan: ScanSummary, completion: @escaping (URL?) -> Void) {
        let coordinator = NSFileCoordinator()
        var error: NSError?
        coordinator.coordinate(readingItemAt: scan.url, options: .forUploading, error: &error) { zipURL in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("\(scan.name).zip")
            try? FileManager.default.removeItem(at: dest)
            do {
                try FileManager.default.copyItem(at: zipURL, to: dest)
                completion(dest)
            } catch {
                completion(nil)
            }
        }
        if error != nil { completion(nil) }
    }

    private static func directorySizeMB(_ url: URL) -> Double {
        let fm = FileManager.default
        guard let e = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total = 0
        for case let f as URL in e {
            total += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return Double(total) / 1_048_576
    }

    private static let stampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd_HHmmss"
        return f
    }()
}
