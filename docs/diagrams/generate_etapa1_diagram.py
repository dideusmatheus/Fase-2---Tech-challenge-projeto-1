"""
Gera a imagem estática (PNG) do diagrama de arquitetura da Etapa 1
(módulo src/genetic_algorithm/). Usado no README.md em vez de um bloco
```mermaid```, porque nem todo visualizador de Markdown renderiza Mermaid
(ex: preview padrão do VS Code) — uma imagem PNG funciona em qualquer lugar.

Rodar com: python docs/diagrams/generate_etapa1_diagram.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import ArrowStyle
import matplotlib.patches as mpatches

# Paleta consistente com o gráfico de convergência (reports/ga_optimization)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
NODE_FILL = "#ffffff"
GROUP_ALPHA = 0.10


def draw_node(ax, xy, text, w=2.0, h=0.8, edgecolor=INK_PRIMARY, pill=False, fontsize=8.3):
    x, y = xy
    box_style = "round,pad=0.02,rounding_size=0.4" if pill else "round,pad=0.02,rounding_size=0.08"
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=box_style,
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=NODE_FILL,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
             color=INK_PRIMARY, zorder=4, linespacing=1.4, clip_on=False)


def draw_group(ax, x0, y0, x1, y1, label, color):
    rect = mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color,
        alpha=GROUP_ALPHA,
        zorder=1,
    )
    ax.add_patch(rect)
    ax.text(x0 + 0.15, y1 - 0.18, label, ha="left", va="top",
             fontsize=9.5, color=color, fontweight="bold", zorder=2, clip_on=False)


def draw_arrow(ax, start, end, color=INK_MUTED):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops={
            "arrowstyle": ArrowStyle("-|>", head_length=0.5, head_width=0.25),
            "color": color, "linewidth": 1.3, "shrinkA": 8, "shrinkB": 8,
        },
        zorder=2,
    )


fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# ── Posições dos nós (x, y) ─────────────────────────────────────
n_search = (0, 4.5)
n_individual = (0, 3.55)
n_operators = (0, 2.6)
n_fitness = (0, 1.65)
n_dataloader = (0, 0.5)

n_engine = (3.6, 2.5)

n_experiments = (7.2, 3.6)
n_optimize = (7.2, 1.2)

n_pipeline = (10.6, 2.5)

# ── Caixas de agrupamento (desenhadas primeiro, ficam atrás) ────
draw_group(ax, -1.15, -0.15, 1.15, 5.65, "Blocos de construção", BLUE)
draw_group(ax, 2.4, 1.65, 4.8, 3.85, "Motor do Algoritmo Genético", ORANGE)
draw_group(ax, 6.0, 0.35, 8.4, 4.85, "Usos do motor", AQUA)

# ── Nós ──────────────────────────────────────────────────────────
draw_node(ax, n_search, "hyperparameter_space.py\nespaço de busca (genes) +\nfábrica de modelos sklearn", fontsize=7.2)
draw_node(ax, n_individual, "individual.py\ncria e decodifica\nindivíduos (genes)")
draw_node(ax, n_operators, "operators.py\nseleção, cruzamento,\nmutação")
draw_node(ax, n_fitness, "fitness.py\ntreina e avalia:\nrecall, F1, accuracy")
draw_node(ax, n_dataloader, "data_loader.py\ncarrega splits já\nprocessados no Módulo 1")

draw_node(ax, n_engine, "ga_engine.py\nloop de gerações\n(elitismo + evolução)", w=2.1, h=1.0)

draw_node(ax, n_experiments, "experiments.py\n3 configs do Algoritmo\nGenético x 1 modelo")
draw_node(ax, n_optimize, "optimize_models.py\n1 config do Algoritmo\nGenético x 8 modelos")

draw_node(ax, n_pipeline, "ga_pipeline.py\nroda tudo com\n1 comando", w=2.0, h=0.9, pill=True)

# ── Setas (dependências) ─────────────────────────────────────────
draw_arrow(ax, n_search, n_individual)
draw_arrow(ax, n_individual, n_engine)
draw_arrow(ax, n_operators, n_engine)
draw_arrow(ax, n_fitness, n_engine)
draw_arrow(ax, n_dataloader, n_experiments)
draw_arrow(ax, n_dataloader, n_optimize)
draw_arrow(ax, n_engine, n_experiments)
draw_arrow(ax, n_engine, n_optimize)
draw_arrow(ax, n_experiments, n_pipeline)
draw_arrow(ax, n_optimize, n_pipeline)

ax.set_xlim(-2.2, 12.4)
ax.set_ylim(-1.0, 6.0)
ax.axis("off")

fig.suptitle("Arquitetura da Etapa 1 — src/genetic_algorithm/", fontsize=13, color=INK_PRIMARY, y=0.99)

output_path = "docs/diagrams/etapa1_arquitetura.png"
fig.savefig(output_path, facecolor=SURFACE)
plt.close(fig)

print(f"Diagrama salvo em {output_path}")
