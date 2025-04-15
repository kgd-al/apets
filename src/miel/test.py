from random import Random
from typing import Type, Callable

from abrain.neat.evolver import EvaluationResult
from miel.ea.neat import NEATEvolver
from miel.rgbxy.genome import RGBXYGenome
from miel.rgbxy.selector import RGBXYGenomeSelector


def default_run():
    RGBXYGenome.a = .25
    RGBXYGenome.s = .1
    selector = RGBXYGenomeSelector()
    run(
        genome=RGBXYGenome,
        evaluator=RGBXYGenome.evaluate,
        evolver=NEATEvolver,
        pre_selector=selector.pre_select,
        selector=selector.select,
    )


def run(genome: Type,
        evaluator: Callable[[RGBXYGenome], EvaluationResult],
        evolver: Type,
        pre_selector: Callable[[list], list],
        selector: Callable[[list], list[float]],):

    _evolver = evolver(genome=genome, evaluator=evaluator)

    gen = 0

    while gen < 100:
        _evolver.run(10)

        candidates = pre_selector(_evolver.population)
        preferences = selector(candidates)
        _evolver.feedback(candidates, preferences)
        gen = _evolver.generation


if __name__ == "__main__":
    default_run()
