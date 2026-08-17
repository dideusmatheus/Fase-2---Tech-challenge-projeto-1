from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.genetic_algorithm.individual import decode_individual


def evaluate_fitness(model_name, individual, X_train, y_train, X_val, y_val,
                      weights=(0.6, 0.25, 0.15)):
    """
    Função fitness do algoritmo genético.

    - Constrói o modelo a partir do indivíduo (genes -> hiperparâmetros)
    - Treina com o conjunto de TREINO
    - Avalia no conjunto de VALIDAÇÃO (nunca usa o teste aqui)
    - Combina três métricas em um único score de fitness:
        60% Recall Maligno   -> prioridade clínica (evitar falso negativo)
        25% F1 Maligno       -> equilíbrio entre precision e recall
        15% Accuracy         -> desempenho geral

    Se a combinação de hiperparâmetros for inválida (ex: algum conflito
    de parâmetros do sklearn), o indivíduo recebe fitness 0 e é descartado
    naturalmente pela seleção, sem quebrar o algoritmo.

    Retorna: (fitness, metrics_dict)
    """
    w_recall, w_f1, w_acc = weights

    try:
        # decode_individual: transforma os genes do indivíduo (dict de hiperparâmetros)
        # num modelo sklearn de verdade (ex: RandomForestClassifier(n_estimators=180, ...)).
        # Usa a função pronta de individual.py pra não duplicar essa lógica.
        model = decode_individual(model_name, individual)

        # Treina o modelo candidato só com dados de TREINO (X_train/y_train) — os
        # mesmos dados usados pelos modelos originais do Módulo 1, garantindo que
        # a comparação entre indivíduos seja justa (mesma base de aprendizado).
        model.fit(X_train, y_train)

        # Gera as predições no conjunto de VALIDAÇÃO (X_val) — nunca no teste.
        # É com essas predições que medimos se os genes desse indivíduo são
        # bons ou ruins, sem nunca "espiar" o conjunto de teste.
        y_pred = model.predict(X_val)
    except Exception:
        return 0.0, {"accuracy": 0.0, "recall": 0.0, "f1": 0.0}

    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)

    # Resolve o problema de comparar indivíduos com 3 métricas diferentes:
    # transforma recall, F1 e accuracy em 1 único número (média ponderada),
    # priorizando Recall por ser a métrica mais crítica clinicamente.
    fitness = (w_recall * recall) + (w_f1 * f1) + (w_acc * accuracy)

    metrics = {"accuracy": accuracy, "recall": recall, "f1": f1}

    # fitness: 0.9334
    # metrics: {'accuracy': 0.95, 'recall': 0.9134, 'f1': 0.9333}
    return fitness, metrics
