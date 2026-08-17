import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score

from src.genetic_algorithm.ga_engine import run_genetic_algorithm
from src.genetic_algorithm.individual import decode_individual
from src.genetic_algorithm.data_loader import load_processed_splits
from src.genetic_algorithm.hyperparameter_space import MODEL_CLASSES

MODELS_DIR = "models/machine_learning"
GA_MODELS_DIR = "models/ga_optimized"
REPORTS_DIR = "reports/ga_optimization"

# Configuração padrão do GA usada para otimizar TODOS os modelos.
# (Os experimentos em experiments.py exploram o efeito de mudar estes valores;
# aqui usamos uma configuração intermediária e equilibrada.)
DEFAULT_GA_CONFIG = {
    "population_size": 20,
    "generations": 12,
    "mutation_rate": 0.2,
    "crossover_rate": 0.8,
    "elitism_size": 2,
}


def _evaluate_baseline(model, X_val, y_val):
    """Avalia o modelo original (Módulo 1) no mesmo conjunto de validação
    usado pelo GA, para que a comparação seja justa (mesmos dados)."""
    y_pred = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_val, y_pred, pos_label=1, zero_division=0),
    }


def optimize_all_models(ga_config=None):
    """
    Para cada um dos 8 modelos do Módulo 1:
      1. Carrega o modelo original (.pkl) e mede sua performance na validação
      2. Roda o algoritmo genético para buscar hiperparâmetros melhores
      3. Compara: modelo original x modelo otimizado
      4. Salva o modelo otimizado e um resumo comparativo final

    Retorna um DataFrame com o comparativo de todos os modelos.
    """
    ga_config = ga_config or DEFAULT_GA_CONFIG

    os.makedirs(GA_MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    X_train, X_val, _, y_train, y_val, _ = load_processed_splits()

    comparison_rows = []
    best_hyperparams = {}

    for model_name in MODEL_CLASSES.keys():

        print("\n" + "=" * 55)
        print(f"🧬 OTIMIZANDO: {model_name.upper()}")
        print("=" * 55)

        # ── BASELINE: modelo original do Módulo 1 ───────────────
        baseline_path = f"{MODELS_DIR}/{model_name}.pkl"
        baseline_model = joblib.load(baseline_path)
        baseline_metrics = _evaluate_baseline(baseline_model, X_val, y_val)

        print(f"  Original   → Recall: {baseline_metrics['recall']:.4f} | "
              f"F1: {baseline_metrics['f1']:.4f} | Accuracy: {baseline_metrics['accuracy']:.4f}")

        # ── OTIMIZAÇÃO: algoritmo genético ──────────────────────
        ga_result = run_genetic_algorithm(
            model_name, X_train, y_train, X_val, y_val,
            population_size=ga_config["population_size"],
            generations=ga_config["generations"],
            mutation_rate=ga_config["mutation_rate"],
            crossover_rate=ga_config["crossover_rate"],
            elitism_size=ga_config["elitism_size"],
            verbose=False,
        )
        optimized_metrics = ga_result["best_metrics"]

        print(f"  Otimizado  → Recall: {optimized_metrics['recall']:.4f} | "
              f"F1: {optimized_metrics['f1']:.4f} | Accuracy: {optimized_metrics['accuracy']:.4f}")
        print(f"  Hiperparâmetros encontrados: {ga_result['best_individual']}")

        # ── SALVA O MODELO OTIMIZADO ─────────────────────────────
        optimized_model = decode_individual(model_name, ga_result["best_individual"])
        optimized_model.fit(X_train, y_train)
        joblib.dump(optimized_model, f"{GA_MODELS_DIR}/{model_name}.pkl")

        best_hyperparams[model_name] = ga_result["best_individual"]

        comparison_rows.append({
            "modelo": model_name,
            "recall_original": baseline_metrics["recall"],
            "recall_otimizado": optimized_metrics["recall"],
            "f1_original": baseline_metrics["f1"],
            "f1_otimizado": optimized_metrics["f1"],
            "accuracy_original": baseline_metrics["accuracy"],
            "accuracy_otimizado": optimized_metrics["accuracy"],
            "ganho_recall": optimized_metrics["recall"] - baseline_metrics["recall"],
        })

    # ── COMPARATIVO FINAL ───────────────────────────────────────
    comparison_df = pd.DataFrame(comparison_rows).sort_values("ganho_recall", ascending=False)

    print("\n" + "=" * 55)
    print("📊 COMPARATIVO FINAL — ORIGINAL x OTIMIZADO (GA)")
    print("=" * 55)
    print(comparison_df.to_string(index=False))

    comparison_path = f"{REPORTS_DIR}/comparison_baseline_vs_ga.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n💾 Comparativo salvo em {comparison_path}")

    hyperparams_path = f"{REPORTS_DIR}/best_hyperparameters.json"
    with open(hyperparams_path, "w", encoding="utf-8") as f:
        json.dump(best_hyperparams, f, indent=2, default=str, ensure_ascii=False)
    print(f"💾 Melhores hiperparâmetros salvos em {hyperparams_path}")

    return comparison_df


if __name__ == "__main__":
    optimize_all_models()
