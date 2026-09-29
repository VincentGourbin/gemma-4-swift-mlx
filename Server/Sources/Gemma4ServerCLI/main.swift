// gemma4-server : charge un modele Gemma 4 avec un profil de reference et l'expose en
// API OpenAI (/v1/chat/completions, /v1/models, /healthz, /metrics).

import ArgumentParser
import Foundation
import Gemma4Server
import Gemma4Swift

@main
struct Gemma4ServerCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gemma4-server",
        abstract: "Serveur OpenAI-compatible pour Gemma 4 (loopback par defaut)"
    )

    @Option(name: .long, help: "Dossier du modele (config.json, safetensors, tokenizer)")
    var modelPath: String

    @Option(name: .long, help: "Profil de reference (ex. e4b/4bit-fast ; voir `gemma4-cli references`)")
    var reference: String?

    @Option(name: .long, help: "Adresse d'ecoute (defaut 127.0.0.1 ; autre chose exige --api-key)")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Port")
    var port: Int = 8080

    @Option(name: .long, help: "Cle d'API (Bearer) ; aussi lue dans GEMMA4_SERVER_API_KEY")
    var apiKey: String?

    @Option(name: .long, help: "max_tokens maximal accepte")
    var maxTokensCap: Int = 8192

    @Option(name: .long, help: "Requetes en attente au-dela de celle en cours (ensuite 429)")
    var maxQueue: Int = 16

    @Option(name: .long, help: "Budget des caches de conversation reutilises, en Go (0 = pas de reutilisation)")
    var conversationCacheGb: Double = 2

    @Option(name: .long, help: "Conversations gardees au plus (LRU)")
    var conversationCacheCount: Int = 8

    @Flag(name: .long, help: "Ne pas charger la tour audio (E2B/E4B : -0,6 Go)")
    var noAudio = false

    func run() async throws {
        let key = apiKey ?? ProcessInfo.processInfo.environment["GEMMA4_SERVER_API_KEY"]
        let url = URL(fileURLWithPath: modelPath)
        var config = Gemma4ServerConfiguration(
            host: host, port: port, apiKey: key, modelID: url.lastPathComponent,
            maxQueueDepth: maxQueue, maxTokensCap: maxTokensCap)
        try config.validate()

        let profile = try reference.map { id -> Gemma4ReferenceProfile in
            guard let profile = Gemma4ReferenceProfile.named(id) else {
                throw ValidationError("profil inconnu : \(id)")
            }
            return profile
        }
        FileHandle.standardError.write(Data("chargement de \(url.lastPathComponent)…\n".utf8))
        let engine = try await Gemma4ChatEngine.load(from: url, profile: profile, audio: !noAudio)
        if conversationCacheGb <= 0 {
            await engine.setReusesConversation(false)
        } else {
            await engine.configureConversationCache(
                capacity: conversationCacheCount, budgetBytes: Int(conversationCacheGb * 1_073_741_824))
        }
        config.modelID = profile.map { "\(url.lastPathComponent) (\($0.qualifiedID))" } ?? url.lastPathComponent
        FileHandle.standardError.write(Data("ecoute sur http://\(host):\(port)\n".utf8))
        try await Gemma4Server(backend: engine, configuration: config).run()
    }
}
