// RuntimeBeacon.swift — beacon de runtime IA (schema v1), opt-in : pendant une operation
// lourde (generation, entrainement, chargement), un manifeste JSON dans
// ~/Library/Application Support/ai-runtime-beacons/ dit aux moniteurs (SiliconScope) ce
// que fait le processus. Contrat : SiliconScope, docs/ai-runtime-beacons.md ; ce fichier
// est son producteur de reference, identite fixee pour Gemma 4.

import Foundation
import os

public enum RuntimeBeacon {
    // MARK: Identity — set these for your runtime
    static let runtimeID = "gemma-4-swift-mlx"         // stable machine id
    static let runtimeDisplayName = "Gemma 4"         // shown in monitors
    static let environmentVariable = "GEMMA4_RUNTIME_BEACON"

    public static let schemaVersion = 1

    private static let enabledState = OSAllocatedUnfairLock(initialState: false)

    /// Global opt-in. Off by default: nothing is written unless the host sets this or
    /// exports `<environmentVariable>=1`.
    public static var isEnabled: Bool {
        get { enabledState.withLock { $0 } }
        set { enabledState.withLock { $0 = newValue } }
    }

    /// Read live (getenv, not ProcessInfo's launch snapshot) so tests can unset it.
    private static var environmentEnabled: Bool {
        getenv(environmentVariable).map { String(cString: $0) == "1" } ?? false
    }

    /// Test hook: keeps tests out of the real shared directory.
    private static let directoryOverrideState = OSAllocatedUnfairLock<URL?>(initialState: nil)
    static var directoryOverride: URL? {
        get { directoryOverrideState.withLock { $0 } }
        set { directoryOverrideState.withLock { $0 = newValue } }
    }

    public static var directory: URL {
        directoryOverride
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("ai-runtime-beacons", isDirectory: true)
    }

    /// Starts a session for one heavy operation, or returns nil when the beacon is off.
    /// Never throws: a filesystem failure just leaves the session without a file.
    public static func begin(task: String, model: String? = nil) -> Session? {
        guard isEnabled || environmentEnabled else { return nil }
        let dir = directory
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        removeStaleManifests(in: dir)
        return Session(directory: dir, task: task, model: model)
    }

    /// Deletes manifests left by dead processes (kill -9). Never touches our own pid.
    private static func removeStaleManifests(in dir: URL) {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil) else { return }
        let me = ProcessInfo.processInfo.processIdentifier
        for url in entries where url.pathExtension == "json" {
            guard let token = url.lastPathComponent.split(separator: "-").first,
                  let pid = pid_t(token), pid != me else { continue }
            if kill(pid, 0) == -1 && errno == ESRCH {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }

    /// One live manifest: written by begin, refreshed by update, deleted by end.
    /// deinit also ends it, but call sites should still `defer { beacon?.end() }`.
    public final class Session: Sendable {
        private struct Manifest: Encodable {
            let version: Int
            let pid: Int32
            let runtime: String
            let displayName: String
            let task: String
            let model: String?
            var phase: String?
            var step: Int?
            var totalSteps: Int?
            let startedAt: Date
            var updatedAt: Date
        }

        private struct State {
            var manifest: Manifest
            var ended = false
            var lastWrite = Date.distantPast
        }

        private let fileURL: URL
        private let state: OSAllocatedUnfairLock<State>

        private static let encoder: JSONEncoder = {
            let e = JSONEncoder()
            e.dateEncodingStrategy = .iso8601    // no fractional seconds, as the contract requires
            e.outputFormatting = [.sortedKeys]
            return e
        }()

        fileprivate init(directory: URL, task: String, model: String?) {
            let pid = ProcessInfo.processInfo.processIdentifier
            let id = UUID().uuidString.prefix(8)            // no "-" in the first 8 chars
            fileURL = directory.appendingPathComponent("\(pid)-\(id).json")
            let now = Date()
            let manifest = Manifest(
                version: RuntimeBeacon.schemaVersion, pid: pid,
                runtime: RuntimeBeacon.runtimeID, displayName: RuntimeBeacon.runtimeDisplayName,
                task: task, model: model, phase: nil, step: nil, totalSteps: nil,
                startedAt: now, updatedAt: now)
            state = OSAllocatedUnfairLock(initialState: State(manifest: manifest, lastWrite: now))
            Self.write(manifest, to: fileURL)
        }

        deinit { end() }

        /// Reports the current phase and step. A phase change is written at once; step
        /// changes within the same phase are written at most once per second.
        public func update(phase: String, step: Int? = nil, totalSteps: Int? = nil) {
            state.withLock { s in
                guard !s.ended else { return }
                let phaseChanged = s.manifest.phase != phase
                s.manifest.phase = phase
                s.manifest.step = step
                s.manifest.totalSteps = totalSteps
                let now = Date()
                guard phaseChanged || now.timeIntervalSince(s.lastWrite) >= 1 else { return }
                s.manifest.updatedAt = now
                s.lastWrite = now
                Self.write(s.manifest, to: fileURL)
            }
        }

        /// Deletes the manifest. Idempotent.
        public func end() {
            state.withLock { s in
                guard !s.ended else { return }
                s.ended = true
                try? FileManager.default.removeItem(at: fileURL)
            }
        }

        /// Atomic write, kept inside the lock: an update racing with end() can never
        /// recreate a deleted file, and two updates can never land out of order.
        private static func write(_ manifest: Manifest, to url: URL) {
            guard let data = try? encoder.encode(manifest) else { return }
            try? data.write(to: url, options: .atomic)
        }
    }
}
