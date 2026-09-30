import Testing
import Foundation
@testable import Gemma4Swift

/// Regles de Gemma4ComputeGate (K-9, S-04). Instances neuves : ne touche pas a
/// `shared`, utilise par les tests d'entrainement de la suite.
@Suite("Garde inference / entrainement")
struct ComputeGateTests {

    @Test("inferences concurrentes autorisees")
    func testConcurrentInferences() throws {
        let gate = Gemma4ComputeGate()
        try gate.beginInference()
        try gate.beginInference()
        #expect(gate.activeInferenceCount == 2)
        gate.endInference()
        gate.endInference()
        #expect(gate.activeInferenceCount == 0)
    }

    @Test("entrainement refuse pendant une inference")
    func testTrainingRefusedDuringInference() throws {
        let gate = Gemma4ComputeGate()
        try gate.beginInference()
        #expect(throws: Gemma4ComputeGate.GateError.inferenceInProgress(count: 1)) {
            try gate.beginTraining()
        }
        gate.endInference()
        try gate.beginTraining()
        gate.endTraining()
    }

    @Test("inference refusee pendant un entrainement, acceptee apres")
    func testInferenceRefusedDuringTraining() throws {
        let gate = Gemma4ComputeGate()
        try gate.withTraining {
            #expect(gate.isTrainingActive)
            #expect(throws: Gemma4ComputeGate.GateError.trainingInProgress) {
                try gate.beginInference()
            }
        }
        #expect(!gate.isTrainingActive)
        try gate.beginInference()
        gate.endInference()
    }

    @Test("un seul entrainement a la fois")
    func testSingleTraining() throws {
        let gate = Gemma4ComputeGate()
        try gate.beginTraining()
        #expect(throws: Gemma4ComputeGate.GateError.trainingAlreadyRunning) {
            try gate.beginTraining()
        }
        gate.endTraining()
    }

    @Test("withTraining libere la garde meme si le corps echoue")
    func testReleaseOnError() throws {
        struct Boom: Error {}
        let gate = Gemma4ComputeGate()
        #expect(throws: Boom.self) { try gate.withTraining { throw Boom() } }
        #expect(!gate.isTrainingActive)
    }
}
