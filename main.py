import numpy as np

from config import (
    SEED,
    D_MODEL,
    D_K,
    VOCAB_SIZE,
    ENCODER_SEQ_LEN,
    DECODER_SEQ_LEN,
    BATCH_SIZE,
    VOCAB,
    ID_TO_TOKEN,
)
from math_utils import softmax
from masking import create_causal_mask
from cross_attention import (
    initialize_cross_attention_weights,
    cross_attention,
)
from inference import autoregressive_inference
from visualization import (
    plot_decoder_lab_pipeline,
    plot_causal_mask_flow,
    plot_cross_attention_flow,
    plot_autoregressive_inference_flow,
)


def main():
    print("=== LABORATÓRIO 3: DECODER ===")
    print(f"Seed: {SEED}")
    print(f"D_MODEL: {D_MODEL}")
    print(f"D_K: {D_K}")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"ENCODER_SEQ_LEN: {ENCODER_SEQ_LEN}")
    print(f"DECODER_SEQ_LEN: {DECODER_SEQ_LEN}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")

    print("\nVocabulário:")
    for token, idx in VOCAB.items():
        print(f"{token}: {idx}")

    print("\nTeste inicial concluído.")

    print("\n=== TAREFA 1: MÁSCARA CAUSAL ===")

    seq_len = 4

    mask = create_causal_mask(seq_len)
    print("\nMáscara causal:")
    print(mask)

    scores = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.5, 2.5, 3.5, 4.5],
        [0.1, 0.2, 0.3, 0.4],
        [2.0, 1.0, 0.5, 0.1]
    ])

    print("\nScores originais:")
    print(scores)

    masked_scores = scores + mask
    print("\nScores após aplicar máscara causal:")
    print(masked_scores)

    attention_probs = softmax(masked_scores, axis=-1)
    print("\nProbabilidades após softmax:")
    print(attention_probs)

    print("\nSoma das linhas:")
    print(np.sum(attention_probs, axis=-1))

    print("\n=== TAREFA 2: CROSS-ATTENTION ===")

    encoder_output = np.random.randn(BATCH_SIZE, ENCODER_SEQ_LEN, D_MODEL)
    decoder_state = np.random.randn(BATCH_SIZE, DECODER_SEQ_LEN, D_MODEL)

    print("\nShape do encoder_output:", encoder_output.shape)
    print("Shape do decoder_state:", decoder_state.shape)

    w_q, w_k, w_v = initialize_cross_attention_weights(D_MODEL, D_K)

    cross_output, cross_debug = cross_attention(
        encoder_output=encoder_output,
        decoder_state=decoder_state,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
    )

    print("\nShape de Q:", cross_debug["q"].shape)
    print("Shape de K:", cross_debug["k"].shape)
    print("Shape de V:", cross_debug["v"].shape)
    print("Shape dos scores:", cross_debug["scores"].shape)
    print("Shape dos scaled_scores:", cross_debug["scaled_scores"].shape)
    print("Shape dos attention_weights:", cross_debug["attention_weights"].shape)
    print("Shape da saída final:", cross_output.shape)

    print("\nSoma das linhas dos attention_weights:")
    print(np.sum(cross_debug["attention_weights"], axis=-1))

    print("\n=== TAREFA 3: INFERÊNCIA AUTO-REGRESSIVA ===")

    bos_token_id = VOCAB["<BOS>"]
    eos_token_id = VOCAB["<EOS>"]

    generated_sequence, all_probs = autoregressive_inference(
        encoder_output=encoder_output,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        vocab_size=VOCAB_SIZE,
        max_len=10
    )

    print("\nSequência gerada (IDs):")
    print(generated_sequence)

    generated_tokens = [ID_TO_TOKEN[token_id] for token_id in generated_sequence]
    print("\nSequência gerada (tokens):")
    print(generated_tokens)

    print("\nDistribuições de probabilidade por passo:")
    for step, probs in enumerate(all_probs):
        print(f"\nPasso {step}:")
        print(probs)
        print("Soma das probabilidades:", np.sum(probs))

    print("\n=== VISUALIZAÇÕES DO DECODER ===")

    plot_decoder_lab_pipeline(
        output_dir="outputs",
        filename="decoder_lab_pipeline.png",
        show=True
    )

    plot_causal_mask_flow(
        output_dir="outputs",
        filename="causal_mask_flow.png",
        show=True
    )

    plot_cross_attention_flow(
        output_dir="outputs",
        filename="cross_attention_flow.png",
        show=True
    )

    plot_autoregressive_inference_flow(
        output_dir="outputs",
        filename="autoregressive_inference_flow.png",
        show=True
    )

    print("Diagramas salvos em outputs/")        


if __name__ == "__main__":
    main()
