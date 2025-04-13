import random
from harmonograph import Harmonograph

class Population:
    def __init__(self, population_size):
        self.population_size = population_size
        self.initialize()

    def initialize(self):
        self.individuals = [Harmonograph() for _ in range(self.population_size)]
        self.generations = 0

    def evolve(self):
        new_generation = [None] * self.population_size
        self.sort_individuals_by_fitness()

        preferred = self.get_preferred_indivs_shuffled()
        elite_size_adjusted = min(self.elite_size, len(preferred))

        # Copy elite
        for i in range(elite_size_adjusted):
            new_generation[i] = self.individuals[i].get_copy()

        # Breed rest
        i = elite_size_adjusted
        while i < self.population_size:
            if random.random() < self.crossover_rate:
                parent1 = self.tournament_selection_v2()
                parent2 = self.tournament_selection_v2()
                new_indivs = parent1.uniform_crossover(parent2)
            else:
                new_indivs = [self.tournament_selection().get_copy(), self.tournament_selection().get_copy()]

            new_generation[i] = new_indivs[0]
            if i + 1 < self.population_size:
                new_generation[i + 1] = new_indivs[1]
            i += 2

        # Mutate
        for i in range(elite_size_adjusted, self.population_size):
            new_generation[i].mutate()

        self.individuals = new_generation

        # Reset fitness
        for indiv in self.individuals:
            indiv.set_fitness(0)

        self.generations += 1

    def tournament_selection_v2(self):
        preferred = self.get_preferred_indivs_shuffled()
        if len(preferred) > 1:
            selection_pool = random.sample(preferred, len(preferred))
        elif len(preferred) == 1:
            return preferred[0]
        else:
            selection_pool = self.individuals

        tournament = [random.choice(selection_pool) for _ in range(self.tournament_size)]
        return max(tournament, key=lambda indiv: indiv.get_fitness())

    def tournament_selection(self):
        tournament = [random.choice(self.individuals) for _ in range(self.tournament_size)]
        return max(tournament, key=lambda indiv: indiv.get_fitness())

    def get_preferred_indivs_shuffled(self):
        return [indiv for indiv in self.individuals if indiv.get_fitness() > 0]

    def sort_individuals_by_fitness(self):
        self.individuals.sort(key=lambda indiv: indiv.get_fitness(), reverse=True)

    def get_indiv(self, index):
        return self.individuals[index]

    def get_size(self):
        return len(self.individuals)

    def get_generations(self):
        return self.generations