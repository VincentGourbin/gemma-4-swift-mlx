// ProfilingEvent.swift - Types d'evenements pour le profiling du pipeline LLM

import Foundation

/// Categorie d'un evenement de profiling (correspond aux lanes dans Chrome Trace)
public enum ProfilingCategory: String, Codable, Sendable {
    // Phases LLM
    case modelLoad = "model_load"
    case tokenization = "tokenization"
    case prefill = "prefill"
    case generation = "generation"
    case generationStep = "generation_step"
    case kvCache = "kv_cache"

    // Phases multimodales
    case visionEncode = "vision_encode"
    case audioEncode = "audio_encode"

    // Infra
    case evalSync = "eval_sync"
    case memoryOp = "memory_op"
    case custom = "custom"

    /// Thread ID pour le regroupement en lanes Chrome Trace
    public var threadId: Int {
        switch self {
        case .modelLoad: return 1
        case .tokenization: return 2
        case .prefill, .generation, .generationStep: return 3
        case .kvCache: return 4
        case .visionEncode, .audioEncode: return 5
        case .memoryOp: return 6
        case .evalSync: return 7
        case .custom: return 8
        }
    }

    /// Nom lisible pour Chrome Trace
    public var threadName: String {
        switch self {
        case .modelLoad: return "Model Loading"
        case .tokenization: return "Tokenization"
        case .prefill, .generation, .generationStep: return "Inference"
        case .kvCache: return "KV Cache"
        case .visionEncode, .audioEncode: return "Multimodal Encoders"
        case .memoryOp: return "Memory"
        case .evalSync: return "eval() Syncs"
        case .custom: return "Other"
        }
    }
}

/// Phase Chrome Trace Event Format
public enum ProfilingPhase: String, Codable, Sendable {
    case begin = "B"
    case end = "E"
    case complete = "X"
    case instant = "i"
    case counter = "C"
    case metadata = "M"
}

/// Un evenement de profiling avec timing et snapshot memoire optionnel
public struct ProfilingEvent: Sendable, Codable {
    public let name: String
    public let category: ProfilingCategory
    public let phase: ProfilingPhase
    public let timestampUs: UInt64
    public let durationUs: UInt64?
    public let threadId: Int

    // Snapshot memoire (optionnel)
    public let mlxActiveBytes: Int?
    public let mlxCacheBytes: Int?
    public let mlxPeakBytes: Int?
    public let processFootprintBytes: Int64?

    // Metadata de step de generation
    public let stepIndex: Int?
    public let totalSteps: Int?

    public init(
        name: String,
        category: ProfilingCategory,
        phase: ProfilingPhase,
        timestampUs: UInt64,
        durationUs: UInt64? = nil,
        threadId: Int? = nil,
        mlxActiveBytes: Int? = nil,
        mlxCacheBytes: Int? = nil,
        mlxPeakBytes: Int? = nil,
        processFootprintBytes: Int64? = nil,
        stepIndex: Int? = nil,
        totalSteps: Int? = nil
    ) {
        self.name = name
        self.category = category
        self.phase = phase
        self.timestampUs = timestampUs
        self.durationUs = durationUs
        self.threadId = threadId ?? category.threadId
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
        self.processFootprintBytes = processFootprintBytes
        self.stepIndex = stepIndex
        self.totalSteps = totalSteps
    }
}

/// Entree de timeline memoire pour les counter events
public struct MemoryTimelineEntry: Sendable, Codable {
    public let timestampUs: UInt64
    public let context: String
    public let mlxActiveMB: Double
    public let mlxCacheMB: Double
    public let mlxPeakMB: Double
    public let processFootprintMB: Double

    public init(timestampUs: UInt64, context: String, mlxActiveMB: Double, mlxCacheMB: Double, mlxPeakMB: Double, processFootprintMB: Double) {
        self.timestampUs = timestampUs
        self.context = context
        self.mlxActiveMB = mlxActiveMB
        self.mlxCacheMB = mlxCacheMB
        self.mlxPeakMB = mlxPeakMB
        self.processFootprintMB = processFootprintMB
    }
}
