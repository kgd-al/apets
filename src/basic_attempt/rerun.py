"""Rerun a robot with given body and parameters."""
import argparse
import logging
import pickle
from pathlib import Path

from basic_attempt.config import Config
from basic_attempt.genotype import Genotype
from evaluator import Evaluator, Options
from individual import Individual
from revolve2.experimentation.logging import setup_logging


def main() -> None:
    """Perform the rerun."""
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--movie", action="store_true")

    config = Config()
    config.rerun = True
    parser.parse_args(namespace=config)
    print(config)

    file = config.file or Path("tmp/last/champion.json")
    config.data_root = file.parent

    genome, data = Genotype.from_file(file)
    fitness = data["fitness"]

    logging.info(f"Fitness from file: {fitness}")

    Evaluator.initialize(
        config=config, options=Options(
            rerun=True,
            movie=True,
            file=None,
            headless=True
        ), verbose=False)

    try:
        fitness = Evaluator.evaluate(genome)
        logging.info(f"Rerun fitness: {fitness}")

    except Exception as e:
        print("Stuff did not work:", e)
        # raise e


if __name__ == "__main__":
    main()
