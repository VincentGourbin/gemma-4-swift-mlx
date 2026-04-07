import Testing
import Foundation
@testable import Gemma4Swift

@Suite("Profiling")
struct ProfilingTests {

    // MARK: - ProfilingSession

    @Test("Session creation et metadata")
    func testSessionCreation() {
        let session = ProfilingSession(config: .singleRun)
        #expect(!session.sessionId.isEmpty)
        #expect(session.systemRAMGB > 0)
        #expect(!session.deviceArchitecture.isEmpty)
    }

    @Test("Begin/end phase produit des events corrects")
    func testPhaseEvents() {
        let session = ProfilingSession(config: .singleRun)
        session.beginPhase("Test Phase", category: .modelLoad)
        session.endPhase("Test Phase", category: .modelLoad)

        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].phase == .begin)
        #expect(events[0].name == "Test Phase")
        #expect(events[0].category == .modelLoad)
        #expect(events[1].phase == .end)
        #expect(events[1].timestampUs >= events[0].timestampUs)
    }

    @Test("Memory tracking capture des snapshots")
    func testMemoryTracking() {
        let config = ProfilingConfig(trackMemory: true)
        let session = ProfilingSession(config: config)
        session.beginPhase("Load", category: .modelLoad)
        session.endPhase("Load", category: .modelLoad)

        let timeline = session.getMemoryTimeline()
        #expect(timeline.count == 2)
        #expect(timeline[0].context == "begin:Load")
        #expect(timeline[1].context == "end:Load")
        #expect(timeline[0].processFootprintMB > 0)
    }

    @Test("Memory tracking desactive ne produit pas de snapshots")
    func testMemoryTrackingDisabled() {
        let config = ProfilingConfig(trackMemory: false)
        let session = ProfilingSession(config: config)
        session.beginPhase("Load", category: .modelLoad)
        session.endPhase("Load", category: .modelLoad)

        let timeline = session.getMemoryTimeline()
        #expect(timeline.isEmpty)
    }

    @Test("recordGenerationStep avec per-step memory")
    func testGenerationStep() {
        let config = ProfilingConfig(trackPerStepMemory: true)
        let session = ProfilingSession(config: config)
        session.recordGenerationStep(index: 1, total: 10, durationUs: 5000)
        session.recordGenerationStep(index: 2, total: 10, durationUs: 4800)

        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].phase == .complete)
        #expect(events[0].category == .generationStep)
        #expect(events[0].stepIndex == 1)
        #expect(events[0].totalSteps == 10)
        #expect(events[0].durationUs == 5000)

        let timeline = session.getMemoryTimeline()
        #expect(timeline.count == 2)
        #expect(timeline[0].context == "token:1/10")
    }

    @Test("recordComplete enregistre un event X avec duree")
    func testRecordComplete() {
        let session = ProfilingSession()
        session.recordComplete("Op", category: .custom, durationUs: 1234)

        let events = session.getEvents()
        #expect(events.count == 1)
        #expect(events[0].phase == .complete)
        #expect(events[0].durationUs == 1234)
    }

    @Test("recordMemorySnapshot ajoute a la timeline")
    func testRecordMemorySnapshot() {
        let session = ProfilingSession()
        session.recordMemorySnapshot(context: "checkpoint")

        let timeline = session.getMemoryTimeline()
        #expect(timeline.count == 1)
        #expect(timeline[0].context == "checkpoint")
    }

    @Test("elapsedSeconds est positif")
    func testElapsedSeconds() {
        let session = ProfilingSession()
        #expect(session.elapsedSeconds >= 0)
    }

    // MARK: - Category Inference

    @Test("Inference de categorie depuis le nom de phase")
    func testCategoryInference() {
        #expect(ProfilingSession.inferCategory("1. Model Loading") == .modelLoad)
        #expect(ProfilingSession.inferCategory("2. Tokenization") == .tokenization)
        #expect(ProfilingSession.inferCategory("4. Prefill") == .prefill)
        #expect(ProfilingSession.inferCategory("5. Token Generation") == .generation)
        #expect(ProfilingSession.inferCategory("KV Cache Allocation") == .kvCache)
        #expect(ProfilingSession.inferCategory("Vision Encoder") == .visionEncode)
        #expect(ProfilingSession.inferCategory("Audio processing") == .audioEncode)
        #expect(ProfilingSession.inferCategory("Something else") == .custom)
    }

    // MARK: - ProfilingConfig

    @Test("Config presets")
    func testConfigPresets() {
        let single = ProfilingConfig.singleRun
        #expect(single.trackMemory == true)
        #expect(single.trackPerStepMemory == false)
        #expect(single.exportChromeTrace == true)

        let bench = ProfilingConfig.benchmark(runs: 5, warmup: 2)
        #expect(bench.benchmarkRuns == 5)
        #expect(bench.warmupRuns == 2)
        #expect(bench.exportChromeTrace == false)

        let detailed = ProfilingConfig.detailed
        #expect(detailed.trackPerStepMemory == true)
    }

    // MARK: - ProfilingEvent

    @Test("Thread ID par categorie")
    func testThreadIds() {
        #expect(ProfilingCategory.modelLoad.threadId == 1)
        #expect(ProfilingCategory.prefill.threadId == 3)
        #expect(ProfilingCategory.generation.threadId == 3)
        #expect(ProfilingCategory.generationStep.threadId == 3)
        #expect(ProfilingCategory.memoryOp.threadId == 6)
    }

    @Test("Thread names par categorie")
    func testThreadNames() {
        #expect(ProfilingCategory.modelLoad.threadName == "Model Loading")
        #expect(ProfilingCategory.prefill.threadName == "Inference")
        #expect(ProfilingCategory.visionEncode.threadName == "Multimodal Encoders")
    }

    // MARK: - ChromeTraceExporter

    @Test("Export Chrome Trace JSON valide")
    func testChromeTraceExport() throws {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "test-model"
        session.beginPhase("Load", category: .modelLoad)
        session.endPhase("Load", category: .modelLoad)

        let data = ChromeTraceExporter.export(session: session)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let traceEvents = json["traceEvents"] as! [[String: Any]]

        // Metadata (process_name + thread names) + 2 events (B/E) + memory counter events + session info
        #expect(traceEvents.count >= 4)

        // Verifier la presence d'un event begin
        let beginEvents = traceEvents.filter { ($0["ph"] as? String) == "B" }
        #expect(!beginEvents.isEmpty)
        #expect(beginEvents[0]["name"] as? String == "Load")
    }

    @Test("Export comparison multi-sessions")
    func testComparisonExport() throws {
        let s1 = ProfilingSession()
        s1.beginPhase("Phase1", category: .prefill)
        s1.endPhase("Phase1", category: .prefill)

        let s2 = ProfilingSession()
        s2.beginPhase("Phase1", category: .prefill)
        s2.endPhase("Phase1", category: .prefill)

        let data = ChromeTraceExporter.exportComparison(sessions: [
            (label: "Run 1", session: s1),
            (label: "Run 2", session: s2),
        ])
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let traceEvents = json["traceEvents"] as! [[String: Any]]

        // Verifier que les deux runs ont des PIDs differents
        let pids = Set(traceEvents.compactMap { $0["pid"] as? Int })
        #expect(pids.contains(1))
        #expect(pids.contains(2))
    }

    // MARK: - Report Generation

    @Test("generateReport contient les phases")
    func testReportGeneration() {
        let session = ProfilingSession()
        session.modelVariant = "test"
        session.beginPhase("1. Load", category: .modelLoad)
        session.endPhase("1. Load", category: .modelLoad)
        session.recordGenerationStep(index: 1, total: 5, durationUs: 10000)

        let report = session.generateReport()
        #expect(report.contains("GEMMA 4 PROFILING REPORT"))
        #expect(report.contains("1. Load"))
        #expect(report.contains("TOKEN GENERATION STATISTICS"))
        #expect(report.contains("Throughput"))
    }

    // MARK: - BenchmarkAggregator

    @Test("Aggregation de sessions en statistiques")
    func testBenchmarkAggregation() {
        var sessions: [ProfilingSession] = []
        for _ in 0 ..< 3 {
            let s = ProfilingSession()
            s.modelVariant = "test"
            s.beginPhase("1. Load", category: .modelLoad)
            s.endPhase("1. Load", category: .modelLoad)
            sessions.append(s)
        }

        let result = BenchmarkAggregator.aggregate(sessions: sessions, warmupCount: 0)
        #expect(result.measuredRuns == 3)
        #expect(result.phaseStats.count == 1)
        #expect(result.phaseStats[0].name == "1. Load")
        #expect(result.phaseStats[0].count == 3)
    }

    @Test("Benchmark report contient les stats")
    func testBenchmarkReport() {
        let result = BenchmarkResult(
            phaseStats: [.init(name: "Load", meanMs: 100, stdMs: 5, minMs: 95, maxMs: 108, count: 3)],
            stepStats: nil,
            totalStats: .init(name: "TOTAL", meanMs: 100, stdMs: 5, minMs: 95, maxMs: 108, count: 3),
            peakMLXActiveMB: 8000,
            peakProcessMB: 9000,
            warmupRuns: 1,
            measuredRuns: 3,
            modelVariant: "test",
            quantization: "bf16",
            promptTokenCount: 30,
            maxTokens: 100,
            generatedTokenCount: 95,
            kvBits: nil
        )
        let report = result.generateReport()
        #expect(report.contains("BENCHMARK REPORT"))
        #expect(report.contains("Load"))
        #expect(report.contains("8000.0"))
    }

    // MARK: - Concurrence

    @Test("Acces concurrent thread-safe")
    func testConcurrentAccess() async {
        let session = ProfilingSession()

        await withTaskGroup(of: Void.self) { group in
            for i in 0 ..< 50 {
                group.addTask {
                    session.beginPhase("Phase-\(i)", category: .custom)
                    session.endPhase("Phase-\(i)", category: .custom)
                }
            }
        }

        let events = session.getEvents()
        #expect(events.count == 100) // 50 begin + 50 end
    }
}
