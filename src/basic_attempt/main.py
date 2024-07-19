"""Main script for the example."""

import logging
import pickle
from pathlib import Path
from typing import Any

import config
import numpy as np
import numpy.typing as npt
from evaluator import Evaluator
from genotype import Genotype
from individual import Individual

from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector
from revolve2.experimentation.logging import setup_logging
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


def main() -> None:
    """Run the program."""
    # Set up logging.
    log = Path("log.txt")
    log.unlink(missing_ok=True)
    setup_logging(level=logging.WARNING, file_name=str(log))

    seed = 2
    data = Genotype.Data(seed)
    rng = make_rng(seed)

    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)
    parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE,
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
    print("Generating initial population.")
    initial_genotypes = [
        Genotype.random(data) for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    print("Evaluating initial population.")
    initial_fitnesses = evaluator.evaluate(initial_genotypes)

    # Create a population of individuals, combining genotype with fitness.
    population = [
        Individual(genotype, fitness)
        for genotype, fitness in zip(initial_genotypes, initial_fitnesses,
                                     strict=True)
    ]

    # Save the best robot
    best_robot = find_best_robot(None, population)

    # Set the current generation to 0.
    generation_index = 0

    # Start the actual optimization process.
    print("Start optimization process.")
    while generation_index < config.NUM_GENERATIONS:
        print(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")

        """
        In contrast to the previous example we do not explicitly stat the order
         of operations here, but let the ModularRobotEvolution object do the scheduling.
        This does not give a performance boost, but is more readable and less prone
         to errors due to mixing up the order.
        
        Not that you are not restricted to the classical ModularRobotEvolution object,
         since you can adjust the step function as you want.
        """
        population = modular_robot_evolution.step(
            population
        )  # Step the evolution forward.

        # Find the new best robot
        best_robot = find_best_robot(best_robot, population)

        print(f"Best robot until now: {best_robot.fitness}")
        # print(f"Genotype pickle: {pickle.dumps(best_robot)!r}")

        file = "best.pkl"
        with open(file, "wb") as f:
            # print(f"Wrote best robot to {file}")
            pickle.dump(best_robot, f)

        # Increase the generation index counter.
        generation_index += 1

    print("Rerunning best robot")
    logging.warning("\n\nRerunning best robot\n")
    evaluator = Evaluator(
        headless=True,
        num_simulators=1,
    )
    fitness = evaluator.evaluate([best_robot.genotype])[0]
    print(f"> fitness: {fitness}")
    if fitness != best_robot.fitness:
        raise RuntimeError(f"Re-evaluation gave different fitness:"
                           f" {best_robot.fitness} != {fitness}")


if __name__ == "__main__":
    main()
