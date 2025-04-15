import functools
from random import Random

from abrain.neat.config import Config
from abrain.neat.evolver import Evolver, EvaluationResult


def wrapped(fn, *args, **kwargs):
    return EvaluationResult(fitness=fn(*args, **kwargs))


def _should_wrap(fn):
    return fn.__annotations__["return"] is not EvaluationResult


class NEATEvolver:
    def __init__(self, genome, evaluator):
        self.genome = genome
        rng = Random(0)
        config = Config()
        config.population_size = 100

        if _should_wrap(evaluator):
            self._original_evaluator = evaluator
            evaluator = functools.partial(wrapped, self._original_evaluator)

        self.evolver = Evolver(
            config=config,
            evaluator=evaluator,
            process_initializer=None,
            genotype_interface=Evolver.Interface(genome, data=dict(rng=rng)))

    @property
    def population(self): return self.evolver.population

    @property
    def generation(self): return self.evolver.generation

    def run(self, generations):
        self.evolver.run(generations)

    def feedback(self, candidates, preferences):
        print(preferences)

