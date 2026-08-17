from src.genetic_algorithm.individual import create_individual
from src.genetic_algorithm.fitness import evaluate_fitness
from src.genetic_algorithm.operators import tournament_selection, crossover, mutate


def run_genetic_algorithm(model_name, X_train, y_train, X_val, y_val,
                           population_size=20, generations=15,
                           mutation_rate=0.2, crossover_rate=0.8,
                           elitism_size=2, verbose=True):
    """
    Executa o algoritmo genético completo para otimizar os hiperparâmetros
    de UM modelo.

    Fluxo por geração:
      1. Avalia o fitness de todos os indivíduos da população atual
      2. Preserva os `elitism_size` melhores sem alteração (elitismo)
         -> garante que a melhor solução encontrada nunca seja perdida
      3. Gera o restante da nova população via seleção + cruzamento + mutação
      4. Repete até o número de gerações definido

    Retorna um dicionário com:
      - best_individual: melhores hiperparâmetros encontrados
      - best_fitness: fitness desse indivíduo
      - best_metrics: accuracy/recall/f1 desse indivíduo na validação
      - history: lista com o melhor fitness de cada geração (para plotar
        a curva de convergência)
    """
    population = [create_individual(model_name) for _ in range(population_size)]
    history = []
    best_individual, best_fitness, best_metrics = None, -1.0, None

    for gen in range(1, generations + 1):

        # ── AVALIAÇÃO ──────────────────────────────────────────
        # Calcula fitness de cada indivíduo da geração atual
        evaluated = []
        for individual in population:
            fitness, metrics = evaluate_fitness(model_name, individual, X_train, y_train, X_val, y_val)
            evaluated.append((individual, fitness, metrics))

        fitnesses = [item[1] for item in evaluated]

        # Ordena do melhor para o pior fitness
        evaluated.sort(key=lambda item: item[1], reverse=True)
        gen_best_individual, gen_best_fitness, gen_best_metrics = evaluated[0]

        # Atualiza o melhor de todas as gerações (não só desta)
        if gen_best_fitness > best_fitness:
            best_individual = gen_best_individual
            best_fitness = gen_best_fitness
            best_metrics = gen_best_metrics

        history.append(best_fitness)

        if verbose:
            print(f"  Geração {gen:>2}/{generations} | Melhor fitness: {gen_best_fitness:.4f} "
                  f"(Recall: {gen_best_metrics['recall']:.4f} | F1: {gen_best_metrics['f1']:.4f})")

        # ── NOVA POPULAÇÃO ─────────────────────────────────────

        # Elitismo: os N melhores da geração atual passam direto
        new_population = [ind for ind, _, _ in evaluated[:elitism_size]]

        # Preenche o restante via seleção + cruzamento + mutação
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2, crossover_rate, model_name)

            child1 = mutate(child1, model_name, mutation_rate)
            child2 = mutate(child2, model_name, mutation_rate)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return {
        "model_name": model_name,
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "best_metrics": best_metrics,
        "history": history,
    }
