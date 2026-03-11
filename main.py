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

if __name__ == "__main__":
    main()