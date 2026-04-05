// ChromeTraceExporter.swift - Export vers Chrome Trace JSON
//
// Visualisation via Perfetto UI (https://ui.perfetto.dev/) ou chrome://tracing
// Format spec: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU

import Foundation

/// Exporte les donnees de ProfilingSession au format Chrome Trace JSON
public struct ChromeTraceExporter {

    /// Export d'une session unique
    public static func export(session: ProfilingSession) -> Data {
        var traceEvents: [[String: Any]] = []
        let pid = 1

        // Metadata : nom du processus
        traceEvents.append(metadataEvent(
            name: "process_name",
            pid: pid,
            tid: 0,
            args: ["name": "Gemma 4 (\(session.modelVariant))"]
        ))

        // Noms des threads pour chaque lane
        let threadNames: [(Int, String)] = [
            (1, "Model Loading"),
            (2, "Tokenization"),
            (3, "Inference"),
            (4, "KV Cache"),
            (5, "Multimodal Encoders"),
            (6, "Memory"),
            (7, "eval() Syncs"),
        ]
        for (tid, name) in threadNames {
            traceEvents.append(metadataEvent(name: "thread_name", pid: pid, tid: tid, args: ["name": name]))
        }

        // Conversion des events
        let events = session.getEvents()
        for event in events {
            var traceEvent: [String: Any] = [
                "name": event.name,
                "cat": event.category.rawValue,
                "ph": event.phase.rawValue,
                "ts": Int(event.timestampUs),
                "pid": pid,
                "tid": event.threadId,
            ]

            if let dur = event.durationUs, event.phase == .complete {
                traceEvent["dur"] = Int(dur)
            }

            var args: [String: Any] = [:]
            if let active = event.mlxActiveBytes {
                args["mlx_active_mb"] = String(format: "%.1f", Double(active) / 1_048_576)
            }
            if let cache = event.mlxCacheBytes {
                args["mlx_cache_mb"] = String(format: "%.1f", Double(cache) / 1_048_576)
            }
            if let peak = event.mlxPeakBytes {
                args["mlx_peak_mb"] = String(format: "%.1f", Double(peak) / 1_048_576)
            }
            if let footprint = event.processFootprintBytes {
                args["process_mb"] = String(format: "%.1f", Double(footprint) / 1_048_576)
            }
            if let step = event.stepIndex { args["step"] = step }
            if let total = event.totalSteps { args["total_steps"] = total }

            if !args.isEmpty { traceEvent["args"] = args }
            if event.phase == .instant { traceEvent["s"] = "g" }

            traceEvents.append(traceEvent)
        }

        // Memory timeline en counter events
        let timeline = session.getMemoryTimeline()
        for entry in timeline {
            traceEvents.append([
                "name": "Memory" as Any,
                "cat": "memory" as Any,
                "ph": "C" as Any,
                "ts": Int(entry.timestampUs) as Any,
                "pid": pid as Any,
                "tid": 6 as Any,
                "args": [
                    "MLX Active (MB)": round(entry.mlxActiveMB * 10) / 10,
                    "MLX Cache (MB)": round(entry.mlxCacheMB * 10) / 10,
                    "MLX Peak (MB)": round(entry.mlxPeakMB * 10) / 10,
                    "Process (MB)": round(entry.processFootprintMB * 10) / 10,
                ] as [String: Any],
            ])
        }

        // Session metadata
        traceEvents.append([
            "name": "Session Info" as Any,
            "cat": "metadata" as Any,
            "ph": "i" as Any,
            "ts": 0 as Any,
            "pid": pid as Any,
            "tid": 0 as Any,
            "s": "g" as Any,
            "args": [
                "device": session.deviceArchitecture,
                "ram_gb": session.systemRAMGB,
                "model": session.modelVariant,
                "quantization": session.quantization,
                "prompt_tokens": session.promptTokenCount,
                "generated_tokens": session.generatedTokenCount,
                "max_tokens": session.maxTokens,
                "kv_bits": session.kvBits as Any,
                "session_id": session.sessionId,
            ] as [String: Any],
        ])

        let trace: [String: Any] = ["traceEvents": traceEvents]

        do {
            return try JSONSerialization.data(withJSONObject: trace, options: [.prettyPrinted, .sortedKeys])
        } catch {
            return "{ \"traceEvents\": [] }".data(using: .utf8)!
        }
    }

    /// Export de sessions multiples pour comparaison (process IDs separes)
    public static func exportComparison(sessions: [(label: String, session: ProfilingSession)]) -> Data {
        var traceEvents: [[String: Any]] = []

        for (index, entry) in sessions.enumerated() {
            let pid = index + 1
            let session = entry.session

            traceEvents.append(metadataEvent(name: "process_name", pid: pid, tid: 0, args: ["name": entry.label]))

            for (tid, name) in [(1, "Model Loading"), (2, "Tokenization"), (3, "Inference"), (4, "KV Cache"), (5, "Multimodal"), (6, "Memory")] {
                traceEvents.append(metadataEvent(name: "thread_name", pid: pid, tid: tid, args: ["name": name]))
            }

            for event in session.getEvents() {
                var traceEvent: [String: Any] = [
                    "name": event.name, "cat": event.category.rawValue,
                    "ph": event.phase.rawValue, "ts": Int(event.timestampUs),
                    "pid": pid, "tid": event.threadId,
                ]
                if let dur = event.durationUs, event.phase == .complete { traceEvent["dur"] = Int(dur) }
                if event.phase == .instant { traceEvent["s"] = "g" }
                traceEvents.append(traceEvent)
            }

            for memEntry in session.getMemoryTimeline() {
                traceEvents.append([
                    "name": "Memory" as Any, "cat": "memory" as Any, "ph": "C" as Any,
                    "ts": Int(memEntry.timestampUs) as Any, "pid": pid as Any, "tid": 6 as Any,
                    "args": [
                        "MLX Active (MB)": round(memEntry.mlxActiveMB * 10) / 10,
                        "Process (MB)": round(memEntry.processFootprintMB * 10) / 10,
                    ] as [String: Any],
                ])
            }
        }

        let trace: [String: Any] = ["traceEvents": traceEvents]
        do {
            return try JSONSerialization.data(withJSONObject: trace, options: [.prettyPrinted, .sortedKeys])
        } catch {
            return "{ \"traceEvents\": [] }".data(using: .utf8)!
        }
    }

    // MARK: - Helpers

    private static func metadataEvent(name: String, pid: Int, tid: Int, args: [String: Any]) -> [String: Any] {
        ["name": name, "ph": "M", "pid": pid, "tid": tid, "args": args]
    }
}
