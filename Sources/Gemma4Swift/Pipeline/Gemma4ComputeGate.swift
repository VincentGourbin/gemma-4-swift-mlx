// Garde de processus entre inference et entrainement

import Foundation

/// Empeche un entrainement et une inference du paquet de tourner en meme temps.
///
/// Pourquoi : jusqu'a mlx-swift 0.31, un gradient et un forward sur deux threads
/// figeaient le process (deadlock ABBA entre `CompiledFunction.call` et `vjp`/`jvp`,
/// corrige en mlx-swift 0.32, #461). La garde reste : les deux se disputeraient la
/// memoire du GPU et l'etat global de MLX (generateur aleatoire, limites `Memory`).
///
/// Regles :
/// - les inferences peuvent tourner entre elles en parallele (pas de gradient) ;
/// - un entrainement est exclusif : il refuse de demarrer si une inference ou un
///   autre entrainement tourne, et toute inference qui demarre pendant un
///   entrainement echoue avec `trainingInProgress`.
///
/// La garde est non bloquante (elle echoue plutot que d'attendre) : les boucles
/// d'entrainement sont synchrones, et attendre des heures la fin d'un entrainement
/// est pire qu'une erreur explicite. Elle ne couvre que les appels du paquet : un
/// consommateur qui appelle MLX directement (forward sur `ModelContainer`, gradient
/// maison) doit serialiser lui-meme.
public final class Gemma4ComputeGate: @unchecked Sendable {

    public static let shared = Gemma4ComputeGate()

    public enum GateError: Error, LocalizedError, Equatable {
        case trainingInProgress
        case inferenceInProgress(count: Int)
        case trainingAlreadyRunning

        public var errorDescription: String? {
            switch self {
            case .trainingInProgress:
                return "Un entrainement est en cours : inference refusee (entrainement et inference sont serialises)."
            case .inferenceInProgress(let count):
                return "\(count) inference(s) en cours : entrainement refuse (entrainement et inference sont serialises)."
            case .trainingAlreadyRunning:
                return "Un autre entrainement est deja en cours."
            }
        }
    }

    private let lock = NSLock()
    private var inferences = 0
    private var training = false

    init() {}

    /// Nombre d'inferences en cours (pour une UI qui veut griser « entrainer »).
    public var activeInferenceCount: Int { lock.withLock { inferences } }

    /// Vrai pendant un entrainement (pour une UI qui veut griser « generer »).
    public var isTrainingActive: Bool { lock.withLock { training } }

    public func beginInference() throws {
        try lock.withLock {
            guard !training else { throw GateError.trainingInProgress }
            inferences += 1
        }
    }

    public func endInference() {
        lock.withLock { inferences = max(0, inferences - 1) }
    }

    public func beginTraining() throws {
        try lock.withLock {
            guard !training else { throw GateError.trainingAlreadyRunning }
            guard inferences == 0 else { throw GateError.inferenceInProgress(count: inferences) }
            training = true
        }
    }

    public func endTraining() {
        lock.withLock { training = false }
    }

    /// Execute un entrainement sous la garde.
    public func withTraining<T>(_ body: () throws -> T) throws -> T {
        try beginTraining()
        defer { endTraining() }
        return try body()
    }

}
