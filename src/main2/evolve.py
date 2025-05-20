"""Main script for the example."""
import os
os.environ.pop('DISPLAY', None)

import functools
import argparse
import math
import time
from datetime import timedelta

import humanize

from abrain.core.genome import dot_found
from abrain.neat.evolver import Evolver, EvaluationResult
from config import Config
from evaluator import Evaluator, Options, performance_compare
from genotype import Genotype


def main(config: Config) -> None:
    start_time = time.perf_counter()

    if config.resume is None:
        data = Genotype.Data(config, config.seed)
        evolver = Evolver(config,
                          evaluator=Evaluator.evaluate,
                          process_initializer=functools.partial(
                              Evaluator.initialize,
                              config=config, options=None),
                          genotype_interface=Genotype.neat_interface(data))
        Evaluator.initialize(config=config, options=None)

    else:
        evolver = Evolver.restore(config.resume, evaluator=Evaluator.evaluate)
        config = evolver.config  # Use the one loaded from file
        data = evolver.individual.genome_data
        Evaluator.initialize(config=config, options=None)

    with evolver:
        generation_digits = math.ceil(math.log10(config.generations))
        i = 0

        def save():
            _path = config.data_root.joinpath(f"champion-{i:0{generation_digits}}.json")
            best_robot.genome.to_file(_path,
                                      dict(fitness=best_robot.fitness))
            return _path

        best_robot = evolver.champion
        best_robot_path = save()

        for i in range(evolver.generation, config.generations):
            evolver.step()

            if evolver.champion.fitness > best_robot.fitness:
                best_robot = evolver.champion
                best_robot_path = save()

    evolver.generate_plots(ext="png", options=dict())

    champion = config.data_root.joinpath("champion.json")
    champion.symlink_to(best_robot_path.name)

    if dot_found:
        best_robot.genome.to_dot(champion, "png", data)

    config.num_simulators = 1
    Evaluator.initialize(
        config=config, options=Options(
            rerun=True,
            movie=True,
            headless=True
        ), verbose=False)
    reeval_results = Evaluator.evaluate(best_robot.genome)

    Evaluator.rename_movie(champion)

    if performance_compare(reeval_results,
                           EvaluationResult(best_robot.fitness, best_robot.stats),
                           verbosity=0):
        config.logger.error(f"Re-evaluation gave different results")
    else:
        config.logger.info(f"Optimal fitness: {reeval_results.fitness}")

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start_time))
    config.logger.info(f"Completed evolution in {duration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main evolution script")
    Config.populate_argparser(parser)
    parsed_config = parser.parse_args(namespace=Config())

    main(parsed_config)
