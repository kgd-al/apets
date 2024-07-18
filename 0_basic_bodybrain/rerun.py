"""Rerun a robot with given body and parameters."""

import logging
import pickle
import sys

from evaluator import Evaluator
from individual import Individual

from revolve2.experimentation.logging import setup_logging


def main() -> None:
    """Perform the rerun."""
    setup_logging()

    try:
        file = sys.argv[1]
    except IndexError:
        file = "best.pkl"

    with open(file, "rb") as f:
        individual: Individual = pickle.load(f)

    logging.info(f"Fitness from pickle: {individual.fitness}")

    evaluator = Evaluator(
        headless=False,
        num_simulators=1,
    )
    fitness = evaluator.evaluate([individual.genotype])[0]
    logging.info(f"Rerun fitness: {fitness}")


if __name__ == "__main__":
    main()
