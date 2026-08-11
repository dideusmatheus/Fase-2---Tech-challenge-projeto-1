import random
import math
from src.genetic_algorithm.hyperparameter_space import GENE_SPACES, build_model


def random_gene_value(gene_spec):
    """
    Sorteia um valor válido para UM gene, de acordo com a especificação
    definida em GENE_SPACES (hyperparameter_space.py).
    """
    gene_type = gene_spec[0]

    if gene_type == "int":
        _, low, high = gene_spec
        # retorna um inteiro aleatório entre low e high (inclusive), ex: 23, 2, 7, 278
        return random.randint(low, high)

    if gene_type == "float":
        _, low, high, log_scale = gene_spec
        if log_scale:
            # Sorteia em escala logarítmica: dá a mesma chance para
            # 0.001-0.01, 0.01-0.1, 0.1-1, etc. Importante para
            # hiperparâmetros como C e alpha, que variam em ordens de grandeza.
            log_low, log_high = math.log10(low), math.log10(high)

            # retorna um float aleatório entre low e high, sorteado em escala logarítmica
            return 10 ** random.uniform(log_low, log_high)
        
        # retorna um float aleatório entre low e high, sorteado em escala linear
        return random.uniform(low, high)

    if gene_type == "choice":
        _, options = gene_spec
        # retorna um valor aleatório de uma lista fixa de opções, ex: "sqrt", "None", "log2"
        return random.choice(options)


    raise ValueError(f"Tipo de gene desconhecido: {gene_type}")


def create_individual(model_name):
    """
    Cria um indivíduo aleatório: um dicionário {hiperparâmetro: valor},
    sorteando cada gene dentro do espaço de busca do modelo escolhido.
    """
    # Espaço de busca do modelo escolhido: dict {nome_do_gene: especificação}
    space = GENE_SPACES[model_name]

    # Indivíduo começa vazio e vai ganhando um valor por gene
    individual = {}

    # Percorre cada hiperparâmetro definido no espaço de busca...
    for gene_name, gene_spec in space.items():
        # ...e sorteia um valor válido para ele, guardando no indivíduo
        individual[gene_name] = random_gene_value(gene_spec)

    # retorna o indivíduo completo, pronto para ser avaliado, 
    # ex: {'max_depth': 22, 'min_samples_split': 3, 'min_samples_leaf': 3, 'criterion': 'entropy'}
    return individual


def decode_individual(model_name, individual):
    """
    Converte um indivíduo (dicionário de genes) em um modelo sklearn
    pronto para ser treinado.
    """
    return build_model(model_name, individual)
