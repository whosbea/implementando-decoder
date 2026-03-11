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


def main():
    print("\n=== TAREFA 1: MÁSCARA CAUSAL ===")

    seq_len = 4

    mask = create_causal_mask(seq_len)
    print("\nMáscara causal:")
    print(mask)

    # Scores fictícios para teste
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
    
if __name__ == "__main__":
    main()