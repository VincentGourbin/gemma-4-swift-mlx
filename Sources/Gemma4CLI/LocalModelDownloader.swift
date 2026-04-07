// Backward-compatible alias — le vrai downloader est dans Gemma4Swift.Gemma4ModelDownloader

import Foundation
import Gemma4Swift

enum LocalModelDownloader {
    static func download(
        modelId: String,
        to destination: URL,
        token: String? = nil,
        progress: @Sendable @escaping (Double) -> Void = { _ in }
    ) async throws {
        let _ = try await Gemma4ModelDownloader.download(
            modelId: modelId,
            token: token
        ) { p in
            progress(p.fraction)
        }
    }
}
