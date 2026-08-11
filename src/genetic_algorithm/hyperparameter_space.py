from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ── ESPAÇO DE BUSCA (CODIFICAÇÃO DOS GENES) ───────────────────
#
# Para cada modelo, definimos quais hiperparâmetros o algoritmo genético
# pode ajustar e os limites de cada um. Cada hiperparâmetro é um "gene".
#
# Formato de cada gene:
#   ("int",    min, max)              -> número inteiro entre min e max
#   ("float",  min, max, log_scale)   -> número decimal entre min e max
#                                         (log_scale=True sorteia em escala
#                                          logarítmica, melhor para C, alpha, etc.)
#   ("choice", [opções])              -> escolhe um valor de uma lista fixa
#
# Um "indivíduo" da população é um dicionário {nome_do_gene: valor}.


GENE_SPACES = {
    "logistic_regression": {
        "C": ("float", 0.001, 100.0, True),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 2, 30),
        "min_samples_split": ("int", 2, 10),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("choice", ["sqrt", "log2", None]),
    },
    "decision_tree": {
        "max_depth": ("int", 2, 30),
        "min_samples_split": ("int", 2, 10),
        "min_samples_leaf": ("int", 1, 10),
        "criterion": ("choice", ["gini", "entropy"]),
    },
    "knn": {
        "n_neighbors": ("int", 1, 30),
        "weights": ("choice", ["uniform", "distance"]),
        "p": ("choice", [1, 2]),
    },
    "svm": {
        "C": ("float", 0.001, 100.0, True),
        "kernel": ("choice", ["rbf", "linear", "poly"]),
        "gamma": ("choice", ["scale", "auto"]),
    },
    "gradient_boosting": {
        "n_estimators": ("int", 50, 300),
        "learning_rate": ("float", 0.001, 0.3, True),
        "max_depth": ("int", 2, 10),
        "subsample": ("float", 0.5, 1.0, False),
    },
    "extra_trees": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 2, 30),
        "min_samples_split": ("int", 2, 10),
        "min_samples_leaf": ("int", 1, 10),
    },
    "mlp": {
        "hidden_layer_sizes": ("choice", [(32,), (64,), (64, 32), (128, 64)]),
        "alpha": ("float", 0.00001, 0.1, True),
        "learning_rate_init": ("float", 0.0001, 0.1, True),
    },
}

# ── PARÂMETROS FIXOS POR MODELO ───────────────────────────────
#
# Hiperparâmetros que NÃO fazem parte do algoritmo genético (ex: random_state
# para reprodutibilidade, ou parâmetros técnicos exigidos pelo sklearn).
# São sempre aplicados junto com os genes sorteados/evoluídos.

FIXED_PARAMS = {
    "logistic_regression": {"max_iter": 10000, "random_state": 42},
    "random_forest": {"random_state": 42},
    "decision_tree": {"random_state": 42},
    "knn": {},
    "svm": {"probability": True, "random_state": 42},
    "gradient_boosting": {"random_state": 42},
    "extra_trees": {"random_state": 42},
    "mlp": {"max_iter": 1000, "random_state": 42},
}

# ── FÁBRICA DE MODELOS ─────────────────────────────────────────

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "svm": SVC,
    "gradient_boosting": GradientBoostingClassifier,
    "extra_trees": ExtraTreesClassifier,
    "mlp": MLPClassifier,
}


def build_model(model_name, hyperparams):
    """
    Constrói uma instância do modelo sklearn a partir do nome e de um
    dicionário de hiperparâmetros (genes já decodificados).
    Combina os genes com os parâmetros fixos do modelo.
    """
    model_class = MODEL_CLASSES[model_name]
    params = {**FIXED_PARAMS[model_name], **hyperparams}

    # cria um modelo sklearn com os hiperparâmetros sorteados, ex:
    # GradientBoostingClassifier(learning_rate=0.019663849003272903, max_depth=4, n_estimators=245, random_state=42, subsample=0.8373682101415366)
    return model_class(**params)
