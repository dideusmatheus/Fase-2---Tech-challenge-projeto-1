import os
import json

from src.llm_interpretation.diagnosis_explainer import explain_sample_patients
from src.llm_interpretation.metrics_narrator import narrate_comparison
from src.llm_interpretation.evaluate_quality import evaluate_batch

# Pasta onde salvamos os resultados da Etapa 3 (mesma convenção da
# Etapa 1, que salva em reports/ga_optimization/).
REPORTS_DIR = "reports/llm_interpretation"


def run_llm_pipeline():
    """
    Pipeline completo da Etapa 3 (Integração com LLMs).

    Fluxo:
      1. Gera explicações em linguagem natural para 2 pacientes de cada uma
         das 4 categorias possíveis (Câncer Perdido, Acerto Maligno, Alarme
         Falso, Acerto Benigno), com as categorias "Real Maligno" primeiro.
      2. Gera um resumo executivo interpretando as métricas comparativas
         da Etapa 1 (Algoritmo Genético).
      3. Avalia a qualidade de todos os textos gerados com uma checklist
         de regras simples.
      4. Salva tudo em arquivos Markdown + JSON dentro de
         reports/llm_interpretation/.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── ETAPA 3a: EXPLICAÇÕES DE DIAGNÓSTICO ──────────────────────
    print("\n" + "=" * 55)
    print("🩺 GERANDO EXPLICAÇÕES DE DIAGNÓSTICO")
    print("=" * 55)

    diagnosis_results_by_category = explain_sample_patients(n_per_category=2)

    for category_name, patients in diagnosis_results_by_category:
        print(f"\n--- {category_name} ---")
        for result in patients:
            print(f"\n  Paciente {result['patient_id']} | Predição: {result['prediction']} "
                  f"({result['probability']:.1%}) | Real: {result['real_label']}")
            print(f"  [Técnica]  {result['explanation']}")
            print(f"  [Paciente] {result['explanation_leiga']}")

    # ── ETAPA 3b: NARRATIVA DE MÉTRICAS ───────────────────────────
    print("\n" + "=" * 55)
    print("📊 GERANDO RESUMO EXECUTIVO DAS MÉTRICAS")
    print("=" * 55)

    metrics_result = narrate_comparison()
    print(f"\n{metrics_result['narrative']}")

    # ── ETAPA 3c: AVALIAÇÃO DE QUALIDADE ──────────────────────────
    print("\n" + "=" * 55)
    print("✅ AVALIANDO QUALIDADE DAS INTERPRETAÇÕES")
    print("=" * 55)

    # "Achata" a lista agrupada por categoria numa lista só de pacientes,
    # só pra facilitar juntar todos os textos gerados (explicação técnica +
    # leiga de cada paciente + narrativa de métricas) e avaliar tudo de
    # uma vez com a mesma checklist.
    all_diagnosis_results = [
        result for _, patients in diagnosis_results_by_category for result in patients
    ]
    all_texts = [r["explanation"] for r in all_diagnosis_results]
    all_texts += [r["explanation_leiga"] for r in all_diagnosis_results]
    all_texts.append(metrics_result["narrative"])

    quality_report = evaluate_batch(all_texts)

    print(f"\n  Score médio de qualidade: {quality_report['average_score']:.1%}")
    for i, evaluation in enumerate(quality_report["individual_evaluations"], 1):
        print(f"  Texto {i}: {evaluation['passed']}/{evaluation['total']} checagens "
              f"({evaluation['score']:.0%})")

    # ── SALVAMENTO DOS RESULTADOS ──────────────────────────────────

    _save_diagnosis_report(diagnosis_results_by_category)
    _save_metrics_report(metrics_result)
    _save_quality_report(quality_report)

    print("\n" + "=" * 55)
    print(f"💾 Resultados salvos em {REPORTS_DIR}/")
    print("=" * 55)

    return {
        "diagnosis_results_by_category": diagnosis_results_by_category,
        "metrics_result": metrics_result,
        "quality_report": quality_report,
    }


def _save_diagnosis_report(diagnosis_results_by_category):
    """Salva as explicações de diagnóstico em um arquivo Markdown legível,
    organizadas pelas mesmas 4 categorias mostradas no console (## por
    categoria, ### por paciente dentro dela)."""
    lines = ["# Explicações de Diagnóstico — Etapa 3\n"]

    for category_name, patients in diagnosis_results_by_category:
        lines.append(f"## {category_name}\n")
        for result in patients:
            lines.append(f"### Paciente {result['patient_id']}\n")
            lines.append(f"- **Predição do modelo:** {result['prediction']} "
                          f"({result['probability']:.1%} de confiança)")
            lines.append(f"- **Diagnóstico real:** {result['real_label']}")
            lines.append("- **Medidas mais notáveis:**")
            for feature_name, value in result["top_features"]:
                lines.append(f"  - {feature_name}: {value:+.2f} desvios-padrão")
            lines.append(f"\n**Explicação técnica (para o médico):**\n\n{result['explanation']}\n")
            lines.append(f"**Explicação para o paciente (linguagem simples):**\n\n{result['explanation_leiga']}\n")

    # "\n".join(lines) transforma a lista de linhas em um único texto,
    # e write() salva esse texto no arquivo.
    with open(f"{REPORTS_DIR}/diagnosis_explanations.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_metrics_report(metrics_result):
    """Salva o resumo executivo das métricas em Markdown."""
    content = (
        "# Interpretação das Métricas — Etapa 3\n\n"
        "## Dados usados como base\n\n"
        f"{metrics_result['comparison_table_text']}\n\n"
        "## Resumo executivo gerado\n\n"
        f"{metrics_result['narrative']}\n"
    )
    with open(f"{REPORTS_DIR}/metrics_insights.md", "w", encoding="utf-8") as f:
        f.write(content)


def _save_quality_report(quality_report):
    """Salva a avaliação de qualidade em JSON (formato fácil de reprocessar)."""
    with open(f"{REPORTS_DIR}/quality_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_llm_pipeline()
