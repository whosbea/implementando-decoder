import numpy as np
from math_utils import softmax


def initialize_cross_attention_weights(d_model: int, d_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inicializa os pesos da cross-attention.

    Shapes:
    - W_Q: (d_model, d_k)
    - W_K: (d_model, d_k)
    - W_V: (d_model, d_k)
    """
    w_q = np.random.randn(d_model, d_k)
    w_k = np.random.randn(d_model, d_k)
    w_v = np.random.randn(d_model, d_k)

    return w_q, w_k, w_v


def cross_attention(
    encoder_output: np.ndarray,
    decoder_state: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Executa a cross-attention entre decoder e encoder.

    Entrada:
    - encoder_output: (batch_size, encoder_seq_len, d_model)
    - decoder_state: (batch_size, decoder_seq_len, d_model)

    Saída:
    - output: (batch_size, decoder_seq_len, d_k)
    - debug_info: tensores intermediários
    """
    # Query vem do decoder
    q = decoder_state @ w_q

    # Key e Value vêm do encoder
    k = encoder_output @ w_k
    v = encoder_output @ w_v

    # Transpor K nos dois últimos eixos
    k_transposed = np.transpose(k, (0, 2, 1))

    # Scores
    scores = q @ k_transposed

    # Escalonamento
    d_k = q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Softmax
    attention_weights = softmax(scaled_scores, axis=-1)

    # Saída final
    output = attention_weights @ v

    debug_info = {
        "q": q,
        "k": k,
        "v": v,
        "scores": scores,
        "scaled_scores": scaled_scores,
        "attention_weights": attention_weights,
        "output": output,
    }

    return output, debug_info
