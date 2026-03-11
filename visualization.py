import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def ensure_output_dir(output_dir: str = "outputs") -> None:
    """
    Garante que a pasta de saída exista.
    """
    os.makedirs(output_dir, exist_ok=True)


def draw_box(ax, x, y, w, h, text, fontsize=11):
    """
    Desenha uma caixa arredondada com texto centralizado.
    """
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        fill=False,
        linewidth=1.8
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True
    )


def draw_arrow(ax, x1, y1, x2, y2):
    """
    Desenha uma seta entre dois pontos.
    """
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.8)
    )


def plot_decoder_lab_pipeline(
    output_dir: str = "outputs",
    filename: str = "decoder_lab_pipeline.png",
    show: bool = True
) -> None:
    """
    Plota o fluxo geral do Laboratório 3.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 5)
    ax.axis("off")

    y = 2.0
    w = 2.2
    h = 1.0

    boxes = [
        (0.6, y, "Entrada do\nDecoder"),
        (3.3, y, "Tarefa 1\nMáscara Causal"),
        (6.0, y, "Tarefa 2\nCross-Attention"),
        (8.7, y, "Tarefa 3\nInferência\nAuto-Regressiva"),
        (11.4, y, "Gerar\nPróximo Token"),
        (14.3, y, "Parar ao gerar\n<EOS>"),
    ]

    for x, y_box, text in boxes:
        draw_box(ax, x, y_box, w, h, text, fontsize=11)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + w
        y1 = boxes[i][1] + h / 2
        x2 = boxes[i + 1][0]
        y2 = boxes[i + 1][1] + h / 2
        draw_arrow(ax, x1, y1, x2, y2)

    ax.text(
        9,
        4.3,
        "Fluxo Geral do Laboratório 3 - Decoder",
        ha="center",
        va="center",
        fontsize=16
    )

    ax.text(
        9,
        1.0,
        "Objetivo: mascarar o futuro, consultar a saída do encoder e gerar tokens até <EOS>",
        ha="center",
        va="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
    
def plot_causal_mask_flow(
    output_dir: str = "outputs",
    filename: str = "causal_mask_flow.png",
    show: bool = True
) -> None:
    """
    Plota o fluxo da aplicação da máscara causal.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(18, 5.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    y = 2.2
    w = 2.8
    h = 1.1

    boxes = [
        (0.4, y, "Scores da Atenção\n(QKᵀ / √dₖ)"),
        (3.8, y, "Criar Máscara\nCausal"),
        (7.2, y, "Somar Máscara\nnos Scores"),
        (10.6, y, "Aplicar\nSoftmax"),
        (14.0, y, "Probabilidades\nsem olhar o futuro"),
    ]

    for x, y_box, text in boxes:
        draw_box(ax, x, y_box, w, h, text, fontsize=11)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + w
        y1 = boxes[i][1] + h / 2
        x2 = boxes[i + 1][0]
        y2 = boxes[i + 1][1] + h / 2
        draw_arrow(ax, x1, y1, x2, y2)

    ax.text(
        9,
        4.8,
        "Tarefa 1 - Fluxo da Máscara Causal",
        ha="center",
        va="center",
        fontsize=16
    )

    ax.text(
        9,
        1.0,
        "A máscara coloca -∞ nas posições futuras; após o softmax, essas probabilidades viram 0",
        ha="center",
        va="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_cross_attention_flow(
    output_dir: str = "outputs",
    filename: str = "cross_attention_flow.png",
    show: bool = True
) -> None:
    """
    Plota o fluxo da cross-attention.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6)
    ax.axis("off")

    draw_box(ax, 0.8, 3.4, 3.0, 1.0, "Decoder State\nshape = (1, 4, 512)", fontsize=11)
    draw_box(ax, 0.8, 1.2, 3.0, 1.0, "Encoder Output\nshape = (1, 10, 512)", fontsize=11)

    draw_box(ax, 4.8, 3.4, 2.5, 1.0, "Query (Q)\n(1, 4, 64)", fontsize=11)
    draw_box(ax, 4.8, 1.2, 2.5, 1.0, "Key (K)\n(1, 10, 64)", fontsize=11)
    draw_box(ax, 8.0, 1.2, 2.5, 1.0, "Value (V)\n(1, 10, 64)", fontsize=11)

    draw_box(ax, 8.0, 3.4, 2.8, 1.0, "Scores\nQKᵀ\n(1, 4, 10)", fontsize=11)
    draw_box(ax, 11.5, 3.4, 2.8, 1.0, "Softmax\nAttention Weights\n(1, 4, 10)", fontsize=11)
    draw_box(ax, 14.8, 3.4, 2.5, 1.0, "Output\n(1, 4, 64)", fontsize=11)

    draw_arrow(ax, 3.8, 3.9, 4.8, 3.9)
    draw_arrow(ax, 3.8, 1.7, 4.8, 1.7)
    draw_arrow(ax, 7.3, 1.7, 8.0, 1.7)
    draw_arrow(ax, 7.3, 3.9, 8.0, 3.9)
    draw_arrow(ax, 10.8, 3.9, 11.5, 3.9)
    draw_arrow(ax, 14.3, 3.9, 14.8, 3.9)
    draw_arrow(ax, 9.25, 2.2, 9.25, 3.4)

    ax.text(
        9,
        5.3,
        "Tarefa 2 - Fluxo da Cross-Attention",
        ha="center",
        va="center",
        fontsize=16
    )

    ax.text(
        9,
        0.4,
        "O decoder gera Q; o encoder fornece K e V",
        ha="center",
        va="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_autoregressive_inference_flow(
    output_dir: str = "outputs",
    filename: str = "autoregressive_inference_flow.png",
    show: bool = True
) -> None:
    """
    Plota o fluxo da inferência auto-regressiva.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(18, 5.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    y = 2.2
    w = 2.7
    h = 1.1

    boxes = [
        (0.4, y, "Começar com\n<BOS>"),
        (3.6, y, "Gerar distribuição\npara próximo token"),
        (6.8, y, "Escolher token\ncom argmax"),
        (10.0, y, "Adicionar token\nà sequência"),
        (13.2, y, "Verificar se é\n<EOS>"),
    ]

    for x, y_box, text in boxes:
        draw_box(ax, x, y_box, w, h, text, fontsize=11)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + w
        y1 = boxes[i][1] + h / 2
        x2 = boxes[i + 1][0]
        y2 = boxes[i + 1][1] + h / 2
        draw_arrow(ax, x1, y1, x2, y2)

    # loop de volta
    draw_arrow(ax, 14.55, 2.2, 14.55, 1.0)
    draw_arrow(ax, 14.55, 1.0, 4.95, 1.0)
    draw_arrow(ax, 4.95, 1.0, 4.95, 2.2)

    ax.text(
        15.8,
        1.35,
        "se não for <EOS>,\ncontinua",
        ha="center",
        va="center",
        fontsize=10
    )

    ax.text(
        15.8,
        3.9,
        "se for <EOS>,\npara",
        ha="center",
        va="center",
        fontsize=10
    )

    ax.text(
        9,
        4.8,
        "Tarefa 3 - Fluxo da Inferência Auto-Regressiva",
        ha="center",
        va="center",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()