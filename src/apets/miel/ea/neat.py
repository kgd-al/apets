import functools
import pprint
from copy import copy
from random import Random

from abrain.neat.config import Config
from abrain.neat.evolver import Evolver as _NEAT, EvaluationResult
from ..api import Evolver, Genotype, Phenotype, Individual


def wrapped(fn, *args, **kwargs):
    return EvaluationResult(fitness=fn(*args, **kwargs))


def _should_wrap(fn):
    return fn.__annotations__["return"] is not EvaluationResult


class NEATEvolver(Evolver[Genotype, Phenotype, _NEAT]):
    def __init__(self, genome, evaluator,
                 seed=0, initial_distance=1):
        super().__init__()

        self.genome = genome
        config = Config(seed=seed)
        config.population_size = 100
        config.initial_distance_threshold = initial_distance
        config.symlink_last = True

        if _should_wrap(evaluator):
            self._original_evaluator = evaluator
            evaluator = functools.partial(wrapped, self._original_evaluator)

        self.evolver = _NEAT(
            config=config,
            evaluator=evaluator,
            process_initializer=None,
            genotype_interface=_NEAT.Interface(genome, data=dict(rng=Random(config.seed))))

    @property
    def generation(self): return self.evolver.generation

    def run(self, generations):
        self.evolver.run(generations)

    @staticmethod
    def _individual(neat_ind, species_index=None):
        stats = copy(neat_ind.stats or {})
        if species_index is not None:
            stats["species"] = species_index
        return Individual(
                genotype=neat_ind.genome, phenotype=None,
                fitness=[neat_ind.fitness],
                stats=stats
            )

    def select(self):
        # Random (stupid) sampling
        # candidates = [self._individual(individual) for individual in
        #               self.rng.choices(list(self.evolver.population), k=3)]

        # Top species champions
        n_candidates = 4
        species = self.evolver.species
        candidates = [self._individual(s.representative, i) for i, s in
                      enumerate(species[:min(n_candidates, len(species))])]

        return candidates, dict(population=self.evolver.population,
                                odir=self.evolver.config.data_root)

    def feedback(self, individuals):
        def _debug_species():
            pprint.pprint({
                s.id: (s.age, s.last_improved)
                for s in self.evolver.species
            })

        _debug_species()
        winner = sorted(individuals, key=lambda i: i.preference)[-1]
        looser = sorted(individuals, key=lambda i: i.preference)[0]

        species = self.evolver.species
        winner, looser = [species[int(i.stats["species"])] for i in [winner, looser]]
        print(f"{winner.id=}, {looser.id=}")

        # Tweak age-related parameters to give an advantage
        winner.last_improved = winner.age
        looser.age *= 2
        _debug_species()
