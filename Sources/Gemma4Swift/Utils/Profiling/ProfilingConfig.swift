// ProfilingConfig.swift - Configuration des sessions de profiling

import Foundation

/// Configuration du profiling
public struct ProfilingConfig: Sendable {
    /// Capturer la memoire aux transitions de phase
    public var trackMemory: Bool

    /// Capturer la memoire a chaque token genere
    public var trackPerStepMemory: Bool

    /// Mode benchmark : nombre de runs mesures (nil = run unique)
    public var benchmarkRuns: Int?

    /// Nombre de runs de warmup avant la mesure
    public var warmupRuns: Int

    /// Repertoire de sortie pour les fichiers trace
    public var outputDirectory: URL?

    /// Exporter en Chrome Trace JSON
    public var exportChromeTrace: Bool

    /// Afficher le rapport en console
    public var printSummary: Bool

    public init(
        trackMemory: Bool = true,
        trackPerStepMemory: Bool = false,
        benchmarkRuns: Int? = nil,
        warmupRuns: Int = 1,
        outputDirectory: URL? = nil,
        exportChromeTrace: Bool = true,
        printSummary: Bool = true
    ) {
        self.trackMemory = trackMemory
        self.trackPerStepMemory = trackPerStepMemory
        self.benchmarkRuns = benchmarkRuns
        self.warmupRuns = warmupRuns
        self.outputDirectory = outputDirectory
        self.exportChromeTrace = exportChromeTrace
        self.printSummary = printSummary
    }

    /// Config par defaut pour un run profile unique
    public static let singleRun = ProfilingConfig()

    /// Config pour le benchmarking
    public static func benchmark(runs: Int = 3, warmup: Int = 1) -> ProfilingConfig {
        ProfilingConfig(
            trackMemory: true,
            trackPerStepMemory: false,
            benchmarkRuns: runs,
            warmupRuns: warmup,
            exportChromeTrace: false,
            printSummary: true
        )
    }

    /// Config detaillee avec memoire par token
    public static let detailed = ProfilingConfig(
        trackMemory: true,
        trackPerStepMemory: true,
        exportChromeTrace: true,
        printSummary: true
    )
}
