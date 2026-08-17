from src.genetic_algorithm.experiments import run_experiments
from src.genetic_algorithm.optimize_models import optimize_all_models


def run_ga_pipeline():
    """
    Pipeline completo da Etapa 1 (Otimização via Algoritmos Genéticos).

    Fluxo:
      1. Roda 3 experimentos com diferentes configurações do ALGORITMO GENÉTICO
         (população, gerações, taxa de mutação) em um modelo de referência,
         para comparar convergência.
      2. Aplica o ALGORITMO GENÉTICO nos 8 modelos do Módulo 1 e compara cada um
         com sua versão original (hiperparâmetros padrão).

    Resultados salvos em reports/ga_optimization/ e
    models/ga_optimized/.
    """

    # ── ETAPA 1a: EXPERIMENTOS COM DIFERENTES CONFIGURAÇÕES ──
    print("\n" + "#" * 55)
    print("# ETAPA 1a — EXPERIMENTOS COM CONFIGURAÇÕES DO ALGORITMO GENÉTICO")
    print("#" * 55)
    experiment_results = run_experiments()

    # ── ETAPA 1b: OTIMIZAÇÃO DE TODOS OS MODELOS ─────────────
    print("\n" + "#" * 55)
    print("# ETAPA 1b — OTIMIZAÇÃO DOS 8 MODELOS (ALGORITMO GENÉTICO x ORIGINAL)")
    print("#" * 55)
    comparison_df = optimize_all_models()

    # Retorna os dois resultados para uso externo (notebooks, relatório)
    return experiment_results, comparison_df


if __name__ == "__main__":
    run_ga_pipeline()
