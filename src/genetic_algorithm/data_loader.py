import pandas as pd

PROCESSED_DIR = "data/machine_learning/processed"


def load_processed_splits():
    """
    Carrega os splits já processados pelo Módulo 1 (Módulo de Machine
    Learning). Reaproveita os mesmos treino/validação/teste usados para
    treinar os modelos originais, garantindo que a comparação entre
    modelo original x modelo otimizado pelo GA seja justa (mesmos dados).
    """
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_val = pd.read_csv(f"{PROCESSED_DIR}/X_val.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")

    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_val = pd.read_csv(f"{PROCESSED_DIR}/y_val.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()

    return X_train, X_val, X_test, y_train, y_val, y_test
