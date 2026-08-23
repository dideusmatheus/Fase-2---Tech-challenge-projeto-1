import pandas as pd

from src.llm_interpretation.client import ask_claude
from src.llm_interpretation.prompts import SYSTEM_PROMPT_METRICS, build_metrics_prompt

# Arquivo gerado na Etapa 1 (optimize_models.py) com a comparação entre
# os modelos originais e os otimizados pelo Algoritmo Genético.
COMPARISON_CSV = "reports/ga_optimization/comparison_baseline_vs_ga.csv"


def _format_comparison_table(df):
    """
    Transforma o DataFrame de comparação em um texto simples (não uma
    tabela markdown), porque LLMs geralmente interpretam melhor texto
    corrido explicando cada linha do que uma tabela densa de números.

    Ex. de uma linha gerada:
    "decision_tree: recall subiu de 0.8824 para 0.9412 (+5.9 pontos
    percentuais); accuracy subiu de 0.9121 para 0.9451"
    """
    lines = []
    for _, row in df.iterrows():
        ganho_pp = (row["recall_otimizado"] - row["recall_original"]) * 100
        lines.append(
            f"- {row['modelo']}: recall foi de {row['recall_original']:.4f} para "
            f"{row['recall_otimizado']:.4f} ({ganho_pp:+.1f} pontos percentuais); "
            f"accuracy foi de {row['accuracy_original']:.4f} para "
            f"{row['accuracy_otimizado']:.4f}; "
            f"F1 foi de {row['f1_original']:.4f} para {row['f1_otimizado']:.4f}."
        )
    # "\n".join(lines) junta todas as linhas da lista em um único texto,
    # cada uma separada por uma quebra de linha.
    return "\n".join(lines)


def narrate_comparison():
    """
    Lê o CSV de comparação da Etapa 1, monta um resumo em texto, e pede
    pro Claude transformar isso em um resumo executivo com recomendações
    práticas para o hospital.
    """
    df = pd.read_csv(COMPARISON_CSV)

    comparison_text = _format_comparison_table(df)
    prompt = build_metrics_prompt(comparison_text)

    # max_tokens maior aqui porque pedimos um resumo + lista de
    # recomendações (texto mais longo que a explicação de 1 paciente).
    narrative = ask_claude(SYSTEM_PROMPT_METRICS, prompt, max_tokens=700)

    return {
        "comparison_table_text": comparison_text,
        "narrative": narrative,
    }
