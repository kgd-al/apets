"""Main script for the example."""
import argparse
import json
import logging
import math
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import humanize
import numpy as np
import numpy.typing as npt
from rich.logging import RichHandler
from rich.progress import Progress

from config import Config
from evaluator import Evaluator, Options
from genotype import Genotype
from individual import Individual
from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng


class ParentSelector(Selector):
    rng: np.random.Generator
    offspring_size: int

    def __init__(self, offspring_size: int, data: Genotype.Data,
                 rng: np.random.Generator) -> None:
        self.offspring_size = offspring_size
        self.data = data
        self.rng = rng

    def select(
        self, population: list[Individual], **kwargs: Any
    ) -> tuple[npt.NDArray[np.int_], dict[str, list[Individual]]]:
        return np.array(
            [
                selection.multiple_unique(
                    selection_size=2,
                    population=[individual.genotype for individual in population],
                    fitnesses=[individual.fitness for individual in population],
                    selection_function=lambda _, fitnesses: selection.tournament(
                        rng=self.rng, fitnesses=fitnesses, k=1
                    ),
                )
                for _ in range(self.offspring_size)
            ],
        ), {"parent_population": population}


class SurvivorSelector(Selector):
    """Selector class for survivor selection."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def select(
        self, population: list[Individual], **kwargs: Any
    ) -> tuple[list[Individual], dict[str, Any]]:
        offspring = kwargs.get("children")
        offspring_fitness = kwargs.get("child_task_performance")
        if offspring is None or offspring_fitness is None:
            raise ValueError(
                "No offspring was passed with positional argument 'children'"
                " and / or 'child_task_performance'."
            )

        original_survivors, offspring_survivors = population_management.steady_state(
            old_genotypes=[i.genotype for i in population],
            old_fitnesses=[i.fitness for i in population],
            new_genotypes=offspring,
            new_fitnesses=offspring_fitness,
            selection_function=lambda n, genotypes, fitnesses: selection.multiple_unique(
                selection_size=n,
                population=genotypes,
                fitnesses=fitnesses,
                selection_function=lambda _, _fitnesses: selection.tournament(
                    rng=self.rng, fitnesses=_fitnesses, k=2
                ),
            ),
        )

        return [
            Individual(
                population[i].genotype,
                population[i].fitness,
            )
            for i in original_survivors
        ] + [
            Individual(
                offspring[i],
                offspring_fitness[i],
            )
            for i in offspring_survivors
        ], {}


class CrossoverReproducer(Reproducer):
    def __init__(self, data: Genotype.Data) -> None:
        self.data = data

    def reproduce(
        self, population: npt.NDArray[np.int_], **kwargs: Any
    ) -> list[Genotype]:
        parent_population: list[Individual] | None = kwargs.get("parent_population")
        if parent_population is None:
            raise ValueError("No parent population given.")

        offspring_genotypes = [
            Genotype.crossover(
                parent_population[parent1_i].genotype,
                parent_population[parent2_i].genotype,
                self.data,
            ).mutated(self.data)
            for parent1_i, parent2_i in population
        ]
        return offspring_genotypes


def find_best_robot(
    current_best: Individual | None, population: list[Individual]
) -> Individual:
    return max(
        population if current_best is None else [current_best] + population,
        key=lambda x: x.fitness,
    )


def get_next_tmp_data_root():
    root = Path("tmp")
    root.mkdir(exist_ok=True)

    next_id = max([int(str(path)
                       .split("/")[1]
                       .split("-")[0][3:]) for path in root.glob("run*/")] + [-1]) + 1
    return root.joinpath(f"run{next_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")


def setup_logging(folder: Path):
    log = folder.joinpath("log")
    log.unlink(missing_ok=True)

    file_handler = logging.FileHandler(log)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s"))

    rich_handler = RichHandler()
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(
        level=logging.WARNING,
        handlers=[file_handler, rich_handler],
    )

    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    w = 40
    logger.info("="*w)
    logger.info(f"{'New log starts here.':^{w}}")
    logger.info("="*w)

    return logger


def main(config: Config) -> None:
    start_time = time.perf_counter()

    if config.data_root is None:
        config.data_root = get_next_tmp_data_root()
    config.data_root.mkdir(exist_ok=config.overwrite, parents=True)

    config.logger = logger = setup_logging(config.data_root)

    data = Genotype.Data(config, config.seed)
    rng = make_rng(config.seed)

    evaluator = Evaluator(config=config, options=None)
    parent_selector = ParentSelector(offspring_size=config.population_size,
                                     data=data, rng=rng)
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(data=data)

    modular_robot_evolution = ModularRobotEvolution(
        parent_selection=parent_selector,
        survivor_selection=survivor_selector,
        evaluator=evaluator,
        reproducer=crossover_reproducer,
    )

    # Create an initial population as we cant start from nothing.
    logger.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(data) for _ in range(config.population_size)
    ]

    # Evaluate the initial population.
    logger.info("Evaluating initial population.")
    initial_fitnesses = evaluator.evaluate(initial_genotypes)

    # Create a population of individuals, combining genotype with fitness.
    population = [
        Individual(genotype, fitness)
        for genotype, fitness in zip(initial_genotypes, initial_fitnesses,
                                     strict=True)
    ]

    # Save the best robot
    best_robot = find_best_robot(None, population)

    generation_digits = math.ceil(math.log10(config.generations))
    current_champion = None

    # Start the actual optimization process.
    logger.info("Start optimization process.")
    progress = Progress()
    progress.start()

    task = progress.add_task("Evolving...", total=config.generations)

    for gen in range(config.generations):

        """
        In contrast to the previous example we do not explicitly stat the order
         of operations here, but let the ModularRobotEvolution object do the scheduling.
        This does not give a performance boost, but is more readable and less prone
         to errors due to mixing up the order.
        
        Not that you are not restricted to the classical ModularRobotEvolution object,
         since you can adjust the step function as you want.
        """
        logger.debug(f"> Start of generation {gen}")
        population = modular_robot_evolution.step(
            population
        )  # Step the evolution forward.

        # Find the new best robot
        best_robot = find_best_robot(best_robot, population)

        # print(f"Best robot until now: [{best_robot.genotype.body.id()}] {best_robot.fitness}")
        # print(f"Genotype pickle: {pickle.dumps(best_robot)!r}")

        current_champion = config.data_root.joinpath(f"champion-{gen:0{generation_digits}}.pkl")
        with open(current_champion, "wb") as f:
            # print(f"Wrote best robot to {file}")
            pickle.dump(best_robot, f)

        progress.update(task, advance=1, refresh=True,
                        description=f"[blue][{gen+1}/{config.generations}]"
                                    f" best fitness: {best_robot.fitness}")

        fitnesses = [ind.fitness for ind in population]
        f_min, f_max = np.quantile(fitnesses, q=[0, 1])
        f_mean, f_dev = np.average(fitnesses), np.std(fitnesses)
        logger.info(f"[Gen {gen:{generation_digits}d}]"
                    f" {f_min:5.3g} <= {f_mean:8.3g}/{f_dev:<8.3g} <= {f_max:8.3g}")
        logger.debug(f"<   End of generation {gen}")

    progress.refresh()
    progress.stop()

    champion = config.data_root.joinpath("champion.pkl")
    champion.symlink_to(current_champion)











    config.num_simulators = 1
    evaluator = Evaluator(config=config,
                          options=Options(
                              rerun=False,
                              movie=False,
                              file=champion,
                              headless=True
                          ))
    fitness = evaluator.evaluate([best_robot.genotype])
    logger.info(f"> fitness: {fitness}")
    config.num_simulators = 2
    evaluator = Evaluator(config=config,
                          options=Options(
                              rerun=False,
                              movie=False,
                              file=champion,
                              headless=True
                          ))
    fitness = evaluator.evaluate([best_robot.genotype])
    logger.info(f"> fitness: {fitness}")

    logger.info("Rerunning best robot")
    evaluator = Evaluator(config=config,
                          options=Options(
                              rerun=True,
                              movie=True,
                              file=champion,
                              headless=False
                          ))
    fitness = evaluator.evaluate([best_robot.genotype])
    logger.info(f"> fitness: {fitness}")
    if fitness != best_robot.fitness:
        raise RuntimeError(f"Re-evaluation gave different fitness:"
                           f" {best_robot.fitness} != {fitness}")

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start_time))
    logger.info(f"Completed evolution in {duration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main evolution script")
    Config.populate_argparser(parser)
    parser.add_argument("--overwrite", action="store_true")
    config = parser.parse_args(namespace=Config())

    main(config)
