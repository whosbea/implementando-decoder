import numpy as np


def generate_next_token(
    current_sequence: list[int],
    encoder_output: np.ndarray,
    vocab_size: int,
    step: int
) -> np.ndarray:
    """
    Gera uma distribuição de probabilidade fictícia para o próximo token.

    Parâmetros:
    - current_sequence: sequência atual de tokens já gerados
    - encoder_output: saída do encoder (não usada semanticamente aqui,
      mas mantida para respeitar a assinatura do laboratório)
    - vocab_size: tamanho do vocabulário
    - step: passo atual da geração

    Retorna:
    - probs: vetor de probabilidades com shape (vocab_size,)
    """
    logits = np.zeros(vocab_size)

    # Simulação controlada de geração:
    # passo 0 -> "eu"
    # passo 1 -> "gosto"
    # passo 2 -> "de"
    # passo 3 -> "pinguins"
    # passo 4 -> <EOS>
    if step == 0:
        logits[3] = 10.0   # eu
    elif step == 1:
        logits[4] = 10.0   # gosto
    elif step == 2:
        logits[5] = 10.0   # de
    elif step == 3:
        logits[6] = 10.0   # pinguins
    else:
        logits[2] = 10.0   # <EOS>

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    return probs


def autoregressive_inference(
    encoder_output: np.ndarray,
    bos_token_id: int,
    eos_token_id: int,
    vocab_size: int,
    max_len: int = 10
) -> tuple[list[int], list[np.ndarray]]:
    """
    Executa o loop auto-regressivo.

    Retorna:
    - generated_sequence: sequência gerada
    - all_probs: lista das distribuições de probabilidade de cada passo
    """
    generated_sequence = [bos_token_id]
    all_probs = []

    for step in range(max_len):
        probs = generate_next_token(
            current_sequence=generated_sequence,
            encoder_output=encoder_output,
            vocab_size=vocab_size,
            step=step
        )

        all_probs.append(probs)

        next_token = int(np.argmax(probs))
        generated_sequence.append(next_token)

        if next_token == eos_token_id:
            break

    return generated_sequence, all_probs