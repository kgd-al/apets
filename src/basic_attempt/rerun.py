"""Rerun a robot with given body and parameters."""
import argparse
import logging
import pickle
import pprint
from pathlib import Path

from abrain.neat.evolver import EvaluationResult, Evolver
from basic_attempt.config import ExperimentType
from config import Config
from genotype import Genotype
from evaluator import Evaluator, Options, performance_compare
from individual import Individual
from revolve2.experimentation.logging import setup_logging


def get_config(file: Path):
    print(file)
    root = file.parent.resolve()
    print(root)
    while not (config := root.joinpath('evolution.json')).exists():
        print(root, '->', root.parent)
        root = root.parent
        if root == Path('/') or root == Path('.'):
            raise ValueError(f"Could not find evolution.json in any parent"
                             f" directory to '{file}'")
    config = Evolver.load_config(config)
    config.data_root = config.data_root.resolve()
    assert config is not None
    return config


def main() -> None:
    """Perform the rerun."""
    setup_logging()

    parser = argparse.ArgumentParser()
    Options.populate_argparser(parser)
    parser.add_argument("file", type=Path,
                        help="Path to the genome to re-evaluate.")
    parser.add_argument("--experiment", type=ExperimentType,
                        help="The experiment type to re-run (can be different"
                             " from the one used during evolution).")

    options = parser.parse_args(namespace=Options())

    options.rerun = True
    print(options)

    if options.file == Path("last"):
        options.file = Path("tmp/last/champion.json")
    file = options.file.resolve()
    assert file.exists(), f"{file=} doesn't exist"

    config = get_config(file)
    config.data_root = file.parent

    if config.experiment != options.experiment:
        logging.info(f"Evaluating on {options.experiment} instead of {config.experiment}")
        config.experiment = options.experiment

    genome, data = Genotype.from_file(file)
    fitness, stats = data["fitness"], data.get("stats", {})

    logging.info(f"Fitness from file: {fitness}")

    Evaluator.initialize(config=config, options=options, verbose=False)

    try:
        result = Evaluator.evaluate(genome)
        if performance_compare(result,
                               EvaluationResult(fitness, stats),
                               verbosity=2):
            logging.error(f"Re-evaluation gave different results")
    except Exception as e:
        print("Stuff did not work:", e)
        raise e


if __name__ == "__main__":
    main()
