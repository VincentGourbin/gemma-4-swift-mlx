import Testing
import Foundation
import MLXLMCommon
import MLXLLM
@testable import Gemma4Swift

/// Garde-fou sur le chemin de chargement : `Gemma4Registration.loadContainer`
/// passe par `LLMModelFactory.shared`, dont on peuple le `typeRegistry`, et
/// jamais par la fonction libre `loadModelContainer` / `ModelFactoryRegistry`
/// (ordre fige VLM-avant-LLM : l'amont publie ses propres entrees "gemma4" /
/// "gemma4_unified" dans `MLXVLM.VLMTypeRegistry`, qu'on ne peut pas surcharger
/// puisque le paquet ne lie pas MLXVLM).
@Suite("Gemma4Registration")
struct Gemma4RegistrationTests {

    /// Config minimale : 2 couches, petit vocab — on instancie des modeles
    /// reels, il faut qu'ils restent legers.
    static let tinyConfigJSON = """
    {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "head_dim": 16,
            "global_head_dim": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 128,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "max_position_embeddings": 128,
            "attention_bias": false,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "tie_word_embeddings": true,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "layer_types": ["sliding_attention", "full_attention"]
        },
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 280,
        "tie_word_embeddings": true
    }
    """

    @Test("register() peuple les quatre types dans LLMTypeRegistry")
    func testRegisterPopulatesLLMTypeRegistry() async {
        await Gemma4Registration.register(multimodal: true)

        for type in ["gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"] {
            let present = await LLMTypeRegistry.shared.contains(type)
            #expect(present, "type manquant : \(type)")
        }
    }

    @Test("LLMModelFactory.shared consulte bien le registre qu'on patche")
    func testFactoryUsesPatchedRegistry() async {
        await Gemma4Registration.register(multimodal: true)

        // loadContainer appelle LLMModelFactory.shared.loadContainer : si la
        // fabrique consultait un autre registre, register() serait sans effet
        // et on retomberait sur le bug de course avec MLXVLM.
        let present = await LLMModelFactory.shared.typeRegistry.contains("gemma4")
        #expect(present)
    }

    @Test("multimodal: du dernier register() est celui qui sert")
    func testRegisterIsLastWriteWins() async throws {
        let config = Data(Self.tinyConfigJSON.utf8)

        await Gemma4Registration.register(multimodal: false)
        let textOnly = try await LLMTypeRegistry.shared.createModel(
            configuration: config, modelType: "gemma4")
        #expect(textOnly is Gemma4LLMModel)

        await Gemma4Registration.register(multimodal: true)
        let multimodal = try await LLMTypeRegistry.shared.createModel(
            configuration: config, modelType: "gemma4")
        #expect(multimodal is Gemma4MultimodalLLMModel)
    }
}
