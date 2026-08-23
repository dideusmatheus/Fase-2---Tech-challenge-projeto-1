# ── AVALIAÇÃO DA QUALIDADE DAS INTERPRETAÇÕES ───────────────────
#
# O PDF do desafio pede pra "avaliar a qualidade das interpretações
# geradas". Em vez de gastar mais chamadas de API pedindo pro próprio
# Claude se autoavaliar (o que custaria dinheiro extra a cada avaliação),
# usamos uma checklist de regras simples e determinísticas — mais barato,
# reprodutível, e fácil de entender o motivo de cada nota.
#
# Cada checagem é uma função que recebe o texto gerado e devolve
# True (passou) ou False (falhou). O score final é a % de checagens que
# passaram.

# Palavras que indicam que o texto menciona o diagnóstico de forma
# reconhecível (maligno/benigno, em qualquer variação de maiúsculas).
DIAGNOSIS_KEYWORDS = ["maligno", "benigno"]

# Frases que indicam que o texto reforça que a decisão final é do
# médico — importante clinicamente, e pedido explicitamente no
# system prompt (SYSTEM_PROMPT_DIAGNOSIS).
DISCLAIMER_KEYWORDS = ["médico", "profissional", "clínic"]

# Palavras/frases que indicariam excesso de certeza — o texto NÃO deveria
# conter isso, já que o modelo é uma ferramenta de apoio, não uma
# verdade absoluta. "não é câncer" e "boa notícia" foram adicionados depois
# de um caso real: a versão leiga tratou a predição (errada, um falso
# negativo) como fato confirmado, o que é perigoso nesse contexto.
OVERCONFIDENT_KEYWORDS = [
    "certeza absoluta", "com certeza total", "garantido",
    "não é câncer", "não tem câncer", "boa notícia", "ótima notícia",
]

MIN_LENGTH = 40    # texto menor que isso provavelmente está vazio/quebrado
MAX_LENGTH = 2000   # texto maior que isso provavelmente fugiu do escopo pedido


def _contains_any(text, keywords):
    """Verifica se o texto contém pelo menos uma das palavras da lista,
    ignorando maiúsculas/minúsculas."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def evaluate_explanation(text):
    """
    Roda a checklist de qualidade em UM texto gerado (explicação de
    diagnóstico ou narrativa de métricas) e retorna um dicionário com
    o resultado de cada checagem + a nota final.
    """
    checks = {
        "menciona_o_diagnostico": _contains_any(text, DIAGNOSIS_KEYWORDS),
        "reforca_decisao_medica": _contains_any(text, DISCLAIMER_KEYWORDS),
        "nao_e_excesso_de_certeza": not _contains_any(text, OVERCONFIDENT_KEYWORDS),
        "tamanho_adequado": MIN_LENGTH <= len(text) <= MAX_LENGTH,
    }

    # sum(checks.values()) soma quantos checks deram True (True conta
    # como 1, False como 0, em Python).
    passed = sum(checks.values())
    total = len(checks)

    return {
        "checks": checks,
        "score": passed / total,  # ex: 0.75 = passou em 3 de 4 checagens
        "passed": passed,
        "total": total,
    }


def evaluate_batch(texts):
    """
    Roda evaluate_explanation em uma lista de textos e devolve a
    avaliação individual de cada um + a média geral do lote.
    """
    evaluations = [evaluate_explanation(text) for text in texts]

    # Calcula a média dos scores individuais.
    average_score = sum(e["score"] for e in evaluations) / len(evaluations)

    return {
        "individual_evaluations": evaluations,
        "average_score": average_score,
    }
