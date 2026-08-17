import os
import pandas as pd
import matplotlib.pyplot as plt

from src.genetic_algorithm.ga_engine import run_genetic_algorithm
from src.genetic_algorithm.data_loader import load_processed_splits

REPORTS_DIR = "reports/ga_optimization"

# ── 3 CONFIGURAÇÕES DE EXPERIMENTO ─────────────────────────────
#
# Cada experimento varia tamanho de população, número de gerações e taxa
# de mutação, para observar o impacto de cada parâmetro na convergência
# do algoritmo genético (exigência do desafio: "ao menos 3 experimentos
# com diferentes configurações").

EXPERIMENTS = [
    {
        "name": "Exp1_populacao_pequena",
        "population_size": 10,
        "generations": 10,
        "mutation_rate": 0.05,
        "crossover_rate": 0.8,
        "elitism_size": 1,
    },
    {
        "name": "Exp2_populacao_media",
        "population_size": 20,
        "generations": 12,
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
        "elitism_size": 2,
    },
    {
        "name": "Exp3_populacao_grande_mutacao_alta",
        "population_size": 30,
        "generations": 15,
        "mutation_rate": 0.4,
        "crossover_rate": 0.8,
        "elitism_size": 3,
    },
]


def run_experiments(model_name="decision_tree"):
    """
    Roda os 3 experimentos definidos acima no MESMO modelo (por padrão,
    Decision Tree). Esse modelo foi escolhido como referência por ter
    espaço real de melhoria via tuning neste dataset — modelos que já
    saturam perto do teto de desempenho (ex: Random Forest, SVM) convergem
    na 1ª geração independente da configuração do GA, o que não ilustra
    bem o efeito de população/mutação na curva de convergência.

    Salva:
      - CSV com o resumo comparativo dos 3 experimentos
      - PNG com a curva de convergência (fitness x geração) de cada um
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    X_train, X_val, _, y_train, y_val, _ = load_processed_splits()

    results = []

    for config in EXPERIMENTS:
        print("\n" + "=" * 55)
        print(f"🧪 {config['name']}  (modelo: {model_name})")
        print(f"   população={config['population_size']} | gerações={config['generations']} "
              f"| mutação={config['mutation_rate']}")
        print("=" * 55)

        result = run_genetic_algorithm(
            model_name, X_train, y_train, X_val, y_val,
            population_size=config["population_size"],
            generations=config["generations"],
            mutation_rate=config["mutation_rate"],
            crossover_rate=config["crossover_rate"],
            elitism_size=config["elitism_size"],
        )

        results.append({
            "experimento": config["name"],
            "populacao": config["population_size"],
            "geracoes": config["generations"],
            "taxa_mutacao": config["mutation_rate"],
            "melhor_fitness": result["best_fitness"],
            "recall": result["best_metrics"]["recall"],
            "f1": result["best_metrics"]["f1"],
            "accuracy": result["best_metrics"]["accuracy"],
            "melhores_hiperparametros": result["best_individual"],
            "history": result["history"],
        })

    # ── RESUMO COMPARATIVO ─────────────────────────────────────
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("history", "melhores_hiperparametros")}
        for r in results
    ])

    print("\n" + "=" * 55)
    print("📊 COMPARATIVO DOS 3 EXPERIMENTOS")
    print("=" * 55)
    print(summary_df.to_string(index=False))

    summary_path = f"{REPORTS_DIR}/experiments_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Resumo salvo em {summary_path}")

    _plot_convergence(results, model_name)

    return results


def _plot_convergence(results, model_name):
    """
    Gera o gráfico de convergência: evolução do melhor fitness ao longo
    das gerações, uma linha por experimento.

    Paleta categórica fixa (azul / laranja / água) e estilo minimalista:
    linhas finas, grid discreto, sem elementos decorativos.
    """
    SURFACE = "#fcfcfb"
    INK_PRIMARY = "#0b0b0b"
    INK_MUTED = "#898781"
    GRID = "#e1e0d9"
    SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1, 2, 3 do tema categórico

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, r in enumerate(results):
        generations = range(1, len(r["history"]) + 1)
        ax.plot(
            generations, r["history"],
            label=r["experimento"],
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
            linewidth=2,
        )

    ax.set_title(f"Convergência do Algoritmo Genético — {model_name}", color=INK_PRIMARY, fontsize=12)
    ax.set_xlabel("Geração", color=INK_MUTED)
    ax.set_ylabel("Melhor Fitness", color=INK_MUTED)

    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)

    ax.legend(frameon=False, labelcolor=INK_PRIMARY)

    plot_path = f"{REPORTS_DIR}/experiments_convergence.png"
    fig.tight_layout()
    fig.savefig(plot_path, facecolor=SURFACE)
    plt.close(fig)

    print(f"📈 Gráfico de convergência salvo em {plot_path}")


if __name__ == "__main__":
    run_experiments()
