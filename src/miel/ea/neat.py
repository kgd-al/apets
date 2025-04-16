import functools
from random import Random

from abrain.neat.config import Config
from abrain.neat.evolver import Evolver as _NEAT, EvaluationResult
from ..api import Evolver, Genotype, Phenotype, Individual


def wrapped(fn, *args, **kwargs):
    return EvaluationResult(fitness=fn(*args, **kwargs))


def _should_wrap(fn):
    return fn.__annotations__["return"] is not EvaluationResult


class NEATEvolver(Evolver[Genotype, Phenotype, _NEAT]):
    def __init__(self, genome, evaluator):
        super().__init__()

        self.genome = genome
        self.rng = Random(0)
        config = Config()
        config.population_size = 100
        config.initial_distance_threshold = 1

        if _should_wrap(evaluator):
            self._original_evaluator = evaluator
            evaluator = functools.partial(wrapped, self._original_evaluator)

        self.evolver = _NEAT(
            config=config,
            evaluator=evaluator,
            process_initializer=None,
            genotype_interface=_NEAT.Interface(genome, data=dict(rng=self.rng)))

    @property
    def generation(self): return self.evolver.generation

    def run(self, generations):
        self.evolver.run(generations)

    def select(self):
        return [
            Individual(
                genotype=individual.genome, phenotype=None,
                fitness=[individual.fitness],
                stats=individual.stats
            )
            for individual in
            self.rng.choices(list(self.evolver.population), k=3)
        ]

    def feedback(self, individuals):
        print("using feedback (this is a lie)")

