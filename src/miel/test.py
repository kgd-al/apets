from typing import Type

from miel.ea.neat import NEATEvolver
from miel.rgbxy.evaluator import RGBXYGenomeEvaluator
from miel.rgbxy.genome import RGBXYGenome
from miel.api import Evolver, Evaluator


def default_run():
    RGBXYGenome.a = .25
    RGBXYGenome.s = .1
    run(
        genome=RGBXYGenome,
        evolver=NEATEvolver,
        evolver_kwargs=dict(evaluator=RGBXYGenome.evaluate),
        evaluator=RGBXYGenomeEvaluator,
    )


def run(genome: Type,
        evolver: Type[Evolver],
        evolver_kwargs: dict,
        evaluator: Type[Evaluator]):

    _evolver = evolver(genome=genome, **evolver_kwargs)
    _evaluator = evaluator()

    gen = 0
    steps = 10

    while gen < 100:
        _evolver.run(steps)

        candidates = _evolver.select()
        _evaluator.evaluate(candidates)
        _evolver.feedback(candidates)
        gen += steps


if __name__ == "__main__":
    default_run()
