import random
from src.genetic_algorithm.hyperparameter_space import GENE_SPACES
from src.genetic_algorithm.individual import random_gene_value


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Seleção por torneio: sorteia `tournament_size` indivíduos aleatórios
    da população e retorna o de maior fitness entre eles.

    Por que torneio: é simples, rápido e mantém pressão seletiva sem
    deixar a busca convergir cedo demais para um único ótimo local
    (diferente da seleção puramente proporcional ao fitness).

    Escolhe os PAIS (sorteia alguns indivíduos e pega o melhor)
    """
    
    # contestants: lista de pares (indivíduo, fitness) sorteados da população,
    # ex: [({'max_depth': 22, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'entropy'}, 0.9334),
    #      ({'max_depth': 9, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'gini'}, 0.9120),
    #      ({'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'}, 0.9010)]
    contestants = random.sample(list(zip(population, fitnesses)), tournament_size)

    # winner: o par (indivíduo, fitness) com o maior fitness entre os contestants,
    # ex: ({'max_depth': 22, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'entropy'}, 0.9334)
    winner = max(contestants, key=lambda pair: pair[1])

    # retorna apenas o indivíduo vencedor do torneio (dict de genes),
    # ex: {'max_depth': 22, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'entropy'}
    return winner[0]


def crossover(parent1, parent2, crossover_rate, model_name):
    """
    Cruzamento uniforme: para cada gene, o filho herda o valor do parent1
    ou do parent2 com 50% de chance cada.

    Uniforme foi escolhido (em vez de ponto único) porque os genes aqui
    não têm uma ordem espacial que faça sentido preservar em blocos —
    cada hiperparâmetro é independente dos demais.

    Com probabilidade (1 - crossover_rate), o cruzamento não ocorre e os
    filhos são cópias diretas dos pais.

    Combina genes de 2 pais pra gerar filhos
    """
    if random.random() > crossover_rate:
        # não houve cruzamento: os filhos saem como cópias exatas dos pais,
        # ex: ({'max_depth': 22, 'min_samples_split': 7, 'min_samples_leaf': 1, 'criterion': 'gini'},
        #      {'max_depth': 9, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'entropy'})
        return dict(parent1), dict(parent2)

    genes = GENE_SPACES[model_name].keys()
    child1, child2 = {}, {}

    for gene_name in genes:
        if random.random() < 0.5:
            child1[gene_name] = parent1[gene_name]
            child2[gene_name] = parent2[gene_name]
        else:
            child1[gene_name] = parent2[gene_name]
            child2[gene_name] = parent1[gene_name]

    # retorna os dois filhos com genes misturados dos pais,
    # ex: ({'max_depth': 22, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'gini'},
    #      {'max_depth': 9, 'min_samples_split': 7, 'min_samples_leaf': 1, 'criterion': 'entropy'})
    return child1, child2


def mutate(individual, model_name, mutation_rate):
    """
    Mutação por reset aleatório: cada gene do indivíduo tem
    `mutation_rate` de chance de ser substituído por um novo valor
    sorteado aleatoriamente dentro do espaço de busca.

    Reset aleatório (em vez de perturbação/gaussian) foi escolhido porque
    o espaço de genes é heterogêneo (int, float, categórico) — um reset
    funciona igual para todos os tipos, mantendo o código simples.

    Altera genes aleatoriamente pra manter diversidade
    """
    space = GENE_SPACES[model_name]
    mutated = dict(individual)

    for gene_name, spec in space.items():
        if random.random() < mutation_rate:
            mutated[gene_name] = random_gene_value(spec)

    # retorna o indivíduo com alguns genes possivelmente alterados,
    # ex: {'max_depth': 30, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'entropy'}
    return mutated
