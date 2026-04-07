// BenchmarkRunner.swift - Analyse statistique des benchmarks

import Foundation

/// Resultat d'un benchmark avec analyse statistique
public struct BenchmarkResult: Sendable {
    public struct PhaseStats: Sendable {
        public let name: String
        public let meanMs: Double
        public let stdMs: Double
        public let minMs: Double
        public let maxMs: Double
        public let count: Int
    }

    public let phaseStats: [PhaseStats]
    public let stepStats: PhaseStats?
    public let totalStats: PhaseStats
    public let peakMLXActiveMB: Double
    public let peakProcessMB: Double
    public let warmupRuns: Int
    public let measuredRuns: Int
    public let modelVariant: String
    public let quantization: String
    public let promptTokenCount: Int
    public let maxTokens: Int
    public let generatedTokenCount: Int
    public let kvBits: Float?

    public func generateReport() -> String {
        var report = """

        \u{256D}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256E}
        \u{2502}             GEMMA 4 BENCHMARK REPORT                      \u{2502}
        \u{251C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}

        """

        report += "  Model: \(modelVariant)  Quant: \(quantization)\n"
        let kvDesc = kvBits != nil ? "TurboQuant \(kvBits!)-bit" : "Standard"
        report += "  KV Cache: \(kvDesc)  Context: \(promptTokenCount) tokens\n"
        report += "  Generated: \(generatedTokenCount)/\(maxTokens)  Warm-up: \(warmupRuns)  Runs: \(measuredRuns)\n\n"

        report += "  PHASE TIMINGS (mean \u{00B1} std)\n"
        report += "  \(String(repeating: "\u{2500}", count: 58))\n"

        for phase in phaseStats {
            let name = phase.name.padding(toLength: 25, withPad: " ", startingAt: 0)
            report += "  \(name) \(formatMs(phase.meanMs)) \u{00B1} \(formatMs(phase.stdMs))"
            report += "  [\(formatMs(phase.minMs)) - \(formatMs(phase.maxMs))]\n"
        }

        report += "  \(String(repeating: "\u{2500}", count: 58))\n"
        let totalName = "TOTAL".padding(toLength: 25, withPad: " ", startingAt: 0)
        report += "  \(totalName) \(formatMs(totalStats.meanMs)) \u{00B1} \(formatMs(totalStats.stdMs))"
        report += "  [\(formatMs(totalStats.minMs)) - \(formatMs(totalStats.maxMs))]\n"

        if let step = stepStats {
            report += "\n  TOKEN GENERATION\n"
            report += "  \(String(repeating: "\u{2500}", count: 58))\n"
            report += "  Average per token: \(formatMs(step.meanMs)) \u{00B1} \(formatMs(step.stdMs))\n"
            report += "  Range: [\(formatMs(step.minMs)) - \(formatMs(step.maxMs))]\n"
            report += "  Throughput: \(String(format: "%.1f", 1000.0 / step.meanMs)) tok/s\n"
        }

        report += "\n  MEMORY\n"
        report += "  \(String(repeating: "\u{2500}", count: 58))\n"
        report += "  Peak MLX Active: \(String(format: "%.1f", peakMLXActiveMB)) MB\n"
        report += "  Peak Process: \(String(format: "%.1f", peakProcessMB)) MB\n"

        report += "\n\u{2570}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256F}\n"

        return report
    }

    private func formatMs(_ ms: Double) -> String {
        if ms < 1000 {
            return String(format: "%.1fms", ms)
        } else if ms < 60000 {
            return String(format: "%.2fs", ms / 1000)
        } else {
            let min = Int(ms / 60000)
            let sec = (ms - Double(min) * 60000) / 1000
            return String(format: "%dm %.1fs", min, sec)
        }
    }
}

/// Agrege des sessions en statistiques de benchmark
public struct BenchmarkAggregator {

    public static func aggregate(
        sessions: [ProfilingSession],
        warmupCount: Int
    ) -> BenchmarkResult {
        guard let first = sessions.first else {
            return BenchmarkResult(
                phaseStats: [], stepStats: nil,
                totalStats: .init(name: "TOTAL", meanMs: 0, stdMs: 0, minMs: 0, maxMs: 0, count: 0),
                peakMLXActiveMB: 0, peakProcessMB: 0,
                warmupRuns: warmupCount, measuredRuns: 0,
                modelVariant: "", quantization: "", promptTokenCount: 0, maxTokens: 0,
                generatedTokenCount: 0, kvBits: nil
            )
        }

        var phaseDurations: [String: [Double]] = [:]
        var allStepDurations: [Double] = []
        var totalDurations: [Double] = []
        var peakMLXActive: Double = 0
        var peakProcess: Double = 0

        for session in sessions {
            let events = session.getEvents()
            var beginTimestamps: [String: UInt64] = [:]
            var sessionTotal: Double = 0

            for event in events {
                switch event.phase {
                case .begin:
                    beginTimestamps[event.name] = event.timestampUs
                case .end:
                    if let beginTs = beginTimestamps[event.name] {
                        let durationMs = Double(event.timestampUs - beginTs) / 1000.0
                        phaseDurations[event.name, default: []].append(durationMs)
                        sessionTotal += durationMs
                        beginTimestamps.removeValue(forKey: event.name)
                    }
                case .complete:
                    if event.category == .generationStep, let dur = event.durationUs {
                        allStepDurations.append(Double(dur) / 1000.0)
                    }
                default:
                    break
                }
            }

            totalDurations.append(sessionTotal)

            for entry in session.getMemoryTimeline() {
                peakMLXActive = max(peakMLXActive, entry.mlxActiveMB)
                peakProcess = max(peakProcess, entry.processFootprintMB)
            }
        }

        let phaseStats = phaseDurations.map { (name, durations) in
            computeStats(name: name, values: durations)
        }.sorted { a, b in
            let order = ["1.", "2.", "3.", "4.", "5.", "6."]
            let aIdx = order.firstIndex(where: { a.name.hasPrefix($0) }) ?? order.count
            let bIdx = order.firstIndex(where: { b.name.hasPrefix($0) }) ?? order.count
            return aIdx < bIdx
        }

        let stepStats = allStepDurations.isEmpty ? nil :
            computeStats(name: "Token Generation", values: allStepDurations)

        let totalStats = computeStats(name: "TOTAL", values: totalDurations)

        return BenchmarkResult(
            phaseStats: phaseStats,
            stepStats: stepStats,
            totalStats: totalStats,
            peakMLXActiveMB: peakMLXActive,
            peakProcessMB: peakProcess,
            warmupRuns: warmupCount,
            measuredRuns: sessions.count,
            modelVariant: first.modelVariant,
            quantization: first.quantization,
            promptTokenCount: first.promptTokenCount,
            maxTokens: first.maxTokens,
            generatedTokenCount: first.generatedTokenCount,
            kvBits: first.kvBits
        )
    }

    private static func computeStats(name: String, values: [Double]) -> BenchmarkResult.PhaseStats {
        guard !values.isEmpty else {
            return .init(name: name, meanMs: 0, stdMs: 0, minMs: 0, maxMs: 0, count: 0)
        }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(values.count)
        let std = sqrt(variance)
        return .init(name: name, meanMs: mean, stdMs: std, minMs: values.min() ?? 0, maxMs: values.max() ?? 0, count: values.count)
    }
}
