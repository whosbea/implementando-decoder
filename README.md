# README.md

# Implementação didática de componentes do Transformer Decoder

## Descrição

Este projeto implementa, de forma didática, os principais blocos lógicos associados ao **Decoder** de um Transformer, com foco em três mecanismos centrais:

- **máscara causal (look-ahead mask)**
- **cross-attention entre decoder e encoder**
- **inferência auto-regressiva**

A implementação foi construída com fins acadêmicos para compreender o comportamento do decoder passo a passo, sem treinamento de modelo e sem uso de frameworks de deep learning.


## Objetivo do laboratório

O objetivo deste laboratório é reproduzir, de forma simplificada, o funcionamento de etapas importantes do **Transformer Decoder**, mostrando como o modelo:

1. impede que tokens vejam o futuro durante a geração;
2. consulta a saída do encoder por meio de **cross-attention**;
3. gera uma sequência token por token até encontrar o token de fim.

## Visão geral da arquitetura do decoder

No Transformer, o decoder recebe uma sequência parcial já gerada e a processa em etapas.

De forma simplificada, o fluxo estudado neste laboratório é:

→Entrada do Decoder  
→ Máscara Causal  
→ Cross-Attention  
→ Inferência Auto-Regressiva  
→ Próximo Token  
→ Parada em `<EOS>`

### Papel de cada bloco

- **Máscara causal**  
  Garante que cada posição da sequência só possa acessar os tokens anteriores e o token atual, nunca os futuros.

- **Cross-attention**  
  Permite que o decoder consulte a saída do encoder.  
  Nesse mecanismo:
  - a **Query (Q)** vem do decoder;
  - as **Keys (K)** e **Values (V)** vêm do encoder.

- **Inferência auto-regressiva**  
  O decoder gera um token por vez, adiciona esse token ao contexto e repete o processo até gerar o token especial de fim `<EOS>`.


## Estrutura do projeto

```
implementando-decoder/
├── main.py
├── config.py
├── math_utils.py
├── masking.py
├── cross_attention.py
├── inference.py
├── visualization.py
├── requirements.txt
├── outputs/
└── README.md
```


## Descrição dos arquivos

- **main.py**  
  Arquivo principal do projeto. Executa as três tarefas do laboratório, imprime os resultados e pode chamar as visualizações.

- **config.py**  
  Centraliza os hiperparâmetros e constantes do projeto, como `D_MODEL`, `D_K`, tamanho do vocabulário, tokens especiais e comprimentos das sequências.

- **math_utils.py**  
  Contém funções matemáticas auxiliares, como a implementação de `softmax`.

- **masking.py**  
  Implementa a criação da **máscara causal**, utilizada para impedir acesso a posições futuras.

- **cross_attention.py**  
  Implementa a **cross-attention**, onde o decoder gera `Q` e o encoder fornece `K` e `V`.

- **inference.py**  
  Implementa a lógica de **inferência auto-regressiva**, simulando a geração de tokens até `<EOS>`.

- **visualization.py**  
  Gera diagramas do fluxo do decoder e das etapas do laboratório.

- **requirements.txt**  
  Lista as dependências necessárias para executar o projeto.

- **outputs/**  
  Pasta onde são salvos os diagramas e demais arquivos gerados durante a execução.

- **README.md**  
  Documentação do projeto.


## Hiperparâmetros e configurações utilizadas

As configurações principais do laboratório são:

- `D_MODEL = 512`
- `D_K = 64`
- `VOCAB_SIZE = 8`
- `ENCODER_SEQ_LEN = 10`
- `DECODER_SEQ_LEN = 4`
- `BATCH_SIZE = 1`

Também foi utilizado um vocabulário fictício com os tokens:

- `<PAD>`
- `<BOS>`
- `<EOS>`
- `eu`
- `gosto`
- `de`
- `pinguins`
- `muito`


## Etapa 1 — Máscara causal

A primeira etapa do laboratório implementa a **máscara causal**.

### Ideia

Durante a geração, o decoder não pode olhar para tokens futuros.  
Se a sequência tem tamanho 4, a máscara causal tem a seguinte forma:

```

[
 [0, -∞, -∞, -∞],
 [0,  0, -∞, -∞],
 [0,  0,  0, -∞],
 [0,  0,  0,  0]
]

```

### Interpretação

- `0` indica posições permitidas;
- `-∞` indica posições proibidas.

Essa máscara é somada aos scores da atenção antes do `softmax`.  
Como `exp(-∞) = 0`, as probabilidades dos tokens futuros se tornam zero.

### Fórmula usada

$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$

Onde:

- `QK^T / sqrt(d_k)` representa os scores da atenção;
- `M` representa a máscara causal.

### O que foi validado

Nesta etapa, o projeto mostra que:

- a máscara foi construída corretamente;
- após aplicar o `softmax`, os tokens futuros recebem probabilidade `0.0`;
- cada linha da distribuição continua somando `1`.


## Etapa 2 — Cross-Attention

A segunda etapa implementa a **cross-attention** entre decoder e encoder.

### Ideia

Diferente da self-attention, aqui os vetores vêm de fontes diferentes:

- **Q** é gerado a partir do `decoder_state`
- **K** e **V** são gerados a partir do `encoder_output`

Assim, o decoder consegue consultar a saída produzida pelo encoder.

### Shapes utilizados

- `encoder_output`: `(1, 10, 512)`
- `decoder_state`: `(1, 4, 512)`

Após as projeções lineares:

- `Q`: `(1, 4, 64)`
- `K`: `(1, 10, 64)`
- `V`: `(1, 10, 64)`

Cálculo dos scores:

- `scores = QK^T`
- `scores`: `(1, 4, 10)`

Saída final:

- `output`: `(1, 4, 64)`

### Interpretação

Cada uma das 4 posições do decoder passa a “olhar” para as 10 posições do encoder.

Ou seja:

- o decoder consulta a representação produzida pelo encoder;
- isso permite que ele use o contexto da entrada codificada para gerar a saída.

### O que foi validado

Nesta etapa, o projeto confirma que:

- as dimensões de `Q`, `K` e `V` estão corretas;
- os scores possuem shape compatível;
- os pesos de atenção somam `1` em cada linha;
- a saída final da cross-attention possui shape consistente.


## Etapa 3 — Inferência auto-regressiva

A terceira etapa simula o processo de geração de tokens.

### Ideia

A inferência auto-regressiva funciona assim:

1. a sequência começa com `<BOS>`;
2. o decoder gera uma distribuição de probabilidade para o próximo token;
3. o token mais provável é escolhido com `argmax`;
4. esse token é adicionado à sequência;
5. o processo continua até gerar `<EOS>`.

### Exemplo de fluxo

→`<BOS>`  
→ `eu`  
→ `gosto`  
→ `de`  
→ `pinguins`  
→ `<EOS>`


### O que foi implementado

Como o projeto é didático e não envolve treinamento, a geração foi simulada com uma distribuição controlada de probabilidades, apenas para demonstrar a lógica do processo.

### O que foi validado

Nesta etapa, o projeto mostra que:

- a sequência começa corretamente em `<BOS>`;
- a cada passo, um novo token é gerado;
- o token gerado é incorporado ao contexto;
- o processo é interrompido ao gerar `<EOS>`.


## Fluxo completo do laboratório

O fluxo completo implementado neste projeto pode ser resumido da seguinte forma:

1. Inicialização do ambiente e dos hiperparâmetros  
2. Teste da máscara causal  
3. Teste da cross-attention  
4. Simulação da inferência auto-regressiva  
5. Geração de diagramas explicativos


## Fórmulas principais utilizadas

### Atenção mascarada


$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$


### Softmax



$\mathrm{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$


### Escolha do próximo token

$next_token = \arg\max(\mathrm{probs})$


## Como executar

### Instalar dependências

```

python -m pip install -r requirements.txt

```

### Executar o projeto

```

python main.py

```

Ou, dependendo do ambiente:

```

python3 main.py

```


## Saídas esperadas

Ao executar o projeto, espera-se obter:

- impressão da máscara causal;
- scores antes e depois da máscara;
- probabilidades após `softmax`;
- shapes da cross-attention;
- sequência gerada na inferência auto-regressiva;
- diagramas salvos na pasta `outputs/`.


## Diagramas gerados

Os diagramas produzidos pelo projeto ajudam a visualizar:

- o fluxo geral do decoder;
- o funcionamento da máscara causal;
- a arquitetura da cross-attention;
- o ciclo da inferência auto-regressiva.

Esses arquivos são salvos na pasta `outputs/`.


## Limitações do projeto

Este projeto tem caráter didático e, por isso:

- não realiza treinamento;
- não utiliza embeddings aprendidos;
- não integra um encoder real ao decoder;
- usa `encoder_output` simulado;
- usa geração controlada de tokens na inferência.

Ou seja, o objetivo não é produzir geração linguística realista, mas demonstrar a lógica estrutural e matemática do decoder.


## Conclusão

A implementação permitiu compreender, de forma prática, os mecanismos centrais do Transformer Decoder.

O laboratório evidenciou três ideias fundamentais:

- o decoder não pode olhar o futuro durante a geração;
- o decoder consulta a saída do encoder por meio da cross-attention;
- a geração de saída ocorre de forma auto-regressiva, token por token, até encontrar `<EOS>`.

Com isso, o projeto fornece uma base clara para entender o funcionamento interno do decoder em arquiteturas Transformer.


## Referência principal

VASWANI, Ashish et al. **Attention Is All You Need**. 2017.

