import joblib
import pandas as pd

from src.llm_interpretation.client import ask_claude
from src.llm_interpretation.prompts import (
    SYSTEM_PROMPT_DIAGNOSIS,
    build_diagnosis_prompt,
    SYSTEM_PROMPT_DIAGNOSIS_LEIGA,
    build_diagnosis_prompt_leiga,
)

# Caminhos dos arquivos gerados pelo Módulo 1 (pipeline de ML).
# Reaproveitamos os mesmos dados de teste já processados, em vez de
# recarregar e reprocessar o CSV bruto.
PROCESSED_DIR = "data/machine_learning/processed"
MODEL_PATH = "models/machine_learning/svm.pkl"  # melhor modelo do Módulo 1

# Quantas medidas "fora do padrão" mostrar na explicação de cada paciente.
TOP_N_FEATURES = 5

# Mapeia a saída numérica do modelo (0 ou 1) para o rótulo em português.
# Vem da codificação feita em preprocess.py: M (maligno) -> 1, B (benigno) -> 0.
LABEL_MAP = {0: "Benigno", 1: "Maligno"}


def _load_test_data():
    """
    Carrega os dados de teste já processados (escalonados) + os IDs
    originais dos pacientes, pra conseguir identificar cada paciente na
    explicação gerada.
    """
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    id_test = pd.read_csv(f"{PROCESSED_DIR}/id_test.csv").squeeze()
    return X_test, y_test, id_test


def _top_notable_features(patient_row, n=TOP_N_FEATURES):
    """
    Identifica quais medidas do exame estão mais "fora do padrão" para
    esse paciente específico.

    Como as features já foram padronizadas pelo StandardScaler no
    Módulo 1 (média 0, desvio padrão 1), o valor de cada coluna JÁ É
    a distância em desvios-padrão da média do dataset. Então basta
    ordenar pelo valor absoluto e pegar as N maiores.

    Isso é uma aproximação simples (não é feature importance real do
    modelo, tipo SHAP) — mas serve bem pra apontar ao médico quais
    medidas mais chamam atenção nesse paciente, sem exigir um cálculo
    específico por tipo de modelo.
    """
    # .abs() converte todos os valores pra positivo, só pra poder ordenar
    # por "quão longe da média", independente da direção (acima/abaixo).
    # .sort_values(ascending=False) coloca os maiores desvios primeiro.
    # .head(n) pega só os N primeiros depois de ordenado.
    ordered = patient_row.abs().sort_values(ascending=False).head(n)

    # Para cada feature selecionada, pega de volta o valor ORIGINAL
    # (com sinal), não o valor absoluto — o médico precisa saber se é
    # acima ou abaixo da média, não só "diferente".
    return [(feature_name, patient_row[feature_name]) for feature_name in ordered.index]


def explain_patient(row_index, model=None, X_test=None, y_test=None, id_test=None):
    """
    Gera a explicação em linguagem natural para UM paciente do conjunto
    de teste, identificado pela posição (row_index) dentro de X_test.

    Os parâmetros model/X_test/y_test/id_test são opcionais — se não
    forem passados, a função carrega tudo sozinha. Isso permite reusar
    os mesmos dados carregados quando explicamos vários pacientes em
    sequência (evita reler os CSVs e o .pkl toda vez).
    """
    if model is None:
        model = joblib.load(MODEL_PATH)
    if X_test is None or y_test is None or id_test is None:
        X_test, y_test, id_test = _load_test_data()

    # .iloc[[row_index]] com colchete duplo mantém o resultado como
    # DataFrame de 1 linha (em vez de Series) — é o formato que o
    # model.predict() espera receber.
    patient_features = X_test.iloc[[row_index]]

    # predict() devolve a classe prevista (0 ou 1).
    prediction = model.predict(patient_features)[0]

    # predict_proba() devolve a probabilidade de cada classe;
    # [0][prediction] pega a probabilidade da classe que foi escolhida.
    probability = model.predict_proba(patient_features)[0][prediction]

    prediction_label = LABEL_MAP[prediction]

    # patient_features.iloc[0] converte a linha de volta pra Series,
    # formato que _top_notable_features espera (indexado por nome da
    # feature, ex: "radius_mean").
    top_features = _top_notable_features(patient_features.iloc[0])

    patient_id = id_test.iloc[row_index]
    real_label = LABEL_MAP[y_test.iloc[row_index]]

    prompt = build_diagnosis_prompt(
        patient_summary=f"ID {patient_id}",
        prediction_label=prediction_label,
        probability=probability,
        model_name="svm",
        top_features=top_features,
    )

    explanation_text = ask_claude(SYSTEM_PROMPT_DIAGNOSIS, prompt, max_tokens=400)

    # Segunda chamada, mesma predição/features, mas explicada para o
    # PACIENTE (sem jargão técnico) em vez de para o médico. Fica ao lado
    # da explicação técnica — nenhuma substitui a outra.
    prompt_leiga = build_diagnosis_prompt_leiga(
        patient_summary=f"ID {patient_id}",
        prediction_label=prediction_label,
        probability=probability,
        top_features=top_features,
    )
    explanation_leiga_text = ask_claude(
        SYSTEM_PROMPT_DIAGNOSIS_LEIGA, prompt_leiga, max_tokens=300
    )

    # Retorna um dicionário com tudo que foi usado/gerado — útil tanto
    # pra salvar no relatório final quanto pra avaliar a qualidade
    # depois (evaluate_quality.py).
    return {
        "patient_id": patient_id,
        "prediction": prediction_label,
        "real_label": real_label,       # diagnóstico verdadeiro (gabarito)
        "probability": probability,
        "top_features": top_features,
        "explanation": explanation_text,
        "explanation_leiga": explanation_leiga_text,
    }


# As 4 categorias possíveis de uma predição binária (matriz de confusão),
# usando os MESMOS nomes já usados na "Tabela de Acertos e Erros por
# Diagnóstico" do Módulo 1 (src/machine_learning/validation.py), pra manter
# o vocabulário consistente em todo o projeto.
#
# Cada item é (nome_da_categoria, label_real, label_previsto). As duas
# categorias "Real Maligno" vêm PRIMEIRO — é aí que está a importância
# clínica do projeto (o modelo existe pra encontrar casos malignos), então
# a ordem prioriza mostrar esses casos antes dos de "Real Benigno".
CATEGORIES = [
    ("Câncer Perdido (Falso Negativo — Real Maligno, Previsto Benigno)", 1, 0),
    ("Acerto Maligno (Verdadeiro Positivo — Real Maligno, Previsto Maligno)", 1, 1),
    ("Alarme Falso (Falso Positivo — Real Benigno, Previsto Maligno)", 0, 1),
    ("Acerto Benigno (Verdadeiro Negativo — Real Benigno, Previsto Benigno)", 0, 0),
]


def explain_sample_patients(n_per_category=2):
    """
    Gera explicações para uma amostra de pacientes do conjunto de teste,
    organizada pelas 4 categorias possíveis de uma predição binária (mesmos
    nomes da Tabela de Acertos e Erros de validation.py). Pega até
    n_per_category pacientes de cada categoria.

    Retorna uma lista de tuplas (nome_da_categoria, lista_de_resultados),
    na mesma ordem de CATEGORIES — "Real Maligno" primeiro.
    """
    model = joblib.load(MODEL_PATH)
    X_test, y_test, id_test = _load_test_data()

    predictions = model.predict(X_test)

    results_by_category = []
    for category_name, real_label, predicted_label in CATEGORIES:
        # Índices dos pacientes de teste que se encaixam nessa categoria
        # específica (ex: real=1/maligno e previsto=0/benigno = falso
        # negativo). O slice [:n_per_category] já limita ao total pedido —
        # se a categoria tiver menos pacientes que isso (ex: só existem 2
        # falsos negativos no dataset todo), pega só os que existem, sem
        # dar erro.
        matching_indices = [
            i for i in range(len(y_test))
            if y_test.iloc[i] == real_label and predictions[i] == predicted_label
        ][:n_per_category]

        patients_in_category = [
            explain_patient(row_index, model, X_test, y_test, id_test)
            for row_index in matching_indices
        ]
        results_by_category.append((category_name, patients_in_category))

    return results_by_category
