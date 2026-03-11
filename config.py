import numpy as np

# Reprodutibilidade
SEED = 42
np.random.seed(SEED)

# Hiperparâmetros
D_MODEL = 512
D_K = 64
VOCAB_SIZE = 8

# Shapes usados no laboratório
ENCODER_SEQ_LEN = 10
DECODER_SEQ_LEN = 4
BATCH_SIZE = 1

# Tokens especiais do vocabulário fictício
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

# Vocabulário fictício para a tarefa 3
VOCAB = {
    PAD_TOKEN: 0,
    BOS_TOKEN: 1,
    EOS_TOKEN: 2,
    "eu": 3,
    "gosto": 4,
    "de": 5,
    "pinguins": 6,
    "muito": 7,
}

ID_TO_TOKEN = {idx: token for token, idx in VOCAB.items()}