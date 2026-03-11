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


if __name__ == "__main__":
    main()