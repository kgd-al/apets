import argparse
from types import SimpleNamespace
from typing import Type, Optional

import thefuzz.process

from apets.miel.ea.map_elite import MapEliteEvolver
from apets.miel.rgbxy.genome import RGBXYGenome
from apets.miel.api import Evolver, Evaluator
from apets.miel.ea.neat import NEATEvolver
from apets.miel.rgbxy.evaluator import RGBXYGenomeEvaluator, RGBXYAutoEvaluator


def __to_dict(_list): return {a: a.__name__ for a in _list}


EVOLVERS = __to_dict([NEATEvolver, MapEliteEvolver])
EVALUATORS = __to_dict([RGBXYGenomeEvaluator, RGBXYAutoEvaluator])

FITNESS_RANGES = {
    RGBXYGenome: (RGBXYGenome.fitness_range(),),
}
FEATURES_RANGES = {
    RGBXYGenome: RGBXYGenome.features_range(),
}
LABELS = {
    RGBXYGenome: ["Match", "x", "y"]
}

EVOLVER_KWARGS = {
    NEATEvolver: lambda g: dict(seed=1, evaluator=g.evaluate,
                                initial_distance=max(g.a, 0.1)),
    MapEliteEvolver: lambda g: dict(
        grid_shape=(50, 50),
        fitness_domain=FITNESS_RANGES[g],
        features_domain=FEATURES_RANGES[g],
        labels=LABELS[g],
        evaluator=lambda _g: dict(fitness=_g.evaluate(_g), features=_g.features(_g)),
        options=SimpleNamespace(
            id=1, seed=1, base_folder="tmp",
            verbosity=1,
            threads=None,
            overwrite=True,
            initial_mutations=10,
            tournament=4
        )
    )
}
EVALUATOR_KWARGS = {
    RGBXYGenomeEvaluator: lambda _: dict(),
    RGBXYAutoEvaluator: lambda target: dict(target=RGBXYGenome.from_string(target)),
}


def __fuzzy_match(query, collection, label):
    match, confidence, item = thefuzz.process.extractOne(query, collection)
    if confidence < 100:
        print(f"Specific {label} '{query}' was not found. Using {match} instead.")
    return item


# def default_run():
#     RGBXYGenome.a = .25
#     RGBXYGenome.s = .1
#     run(
#         genome=RGBXYGenome,
#         evolver=NEATEvolver,
#         evolver_kwargs=dict(evaluator=RGBXYGenome.evaluate, seed=1),
#         # evaluator=RGBXYGenomeEvaluator,
#         evaluator=RGBXYAutoEvaluator,
#         evaluator_kwargs=dict(target=RGBXYGenome(1, 0, 0, 1, 0)),
#     )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolver", type=str, default=list(EVOLVERS.values())[0])
    parser.add_argument("--evaluator", type=str, default=list(EVALUATORS.values())[0])
    parser.add_argument("--evaluator-arg", type=str, default="")
    args = parser.parse_args()

    evolver = __fuzzy_match(args.evolver, EVOLVERS, "evolver")
    evaluator = __fuzzy_match(args.evaluator, EVALUATORS, "evaluator")

    genotype = evaluator.GENOTYPE

    evolver_kwargs = EVOLVER_KWARGS[evolver](genotype)
    evaluator_kwargs = EVALUATOR_KWARGS[evaluator](args.evaluator_arg)

    run(
        genome=genotype,
        evolver=evolver, evolver_kwargs=evolver_kwargs,
        evaluator=evaluator, evaluator_kwargs=evaluator_kwargs,
    )


def run(genome: Type,
        evolver: Type[Evolver],
        evaluator: Type[Evaluator],
        evolver_kwargs: Optional[dict] = None,
        evaluator_kwargs: Optional[dict] = None):

    evolver_kwargs = evolver_kwargs or {}
    _evolver = evolver(genome=genome, **evolver_kwargs)

    evaluator_kwargs = evaluator_kwargs or {}
    _evaluator = evaluator(**evaluator_kwargs)

    gen = 0
    steps = 10

    while gen < 100:
        _evolver.run(steps)

        candidates, eval_kwargs = _evolver.select()
        _evaluator.evaluate(candidates, eval_kwargs)
        _evolver.feedback(candidates)
        gen += steps


if __name__ == "__main__":
    main()
