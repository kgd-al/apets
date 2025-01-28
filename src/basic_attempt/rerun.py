"""Rerun a robot with given body and parameters."""
import argparse
import logging
from pathlib import Path

from revolve2.experimentation.logging import setup_logging

from abrain.neat.evolver import EvaluationResult, Evolver
from config import ExperimentType
from evaluator import Evaluator, Options, performance_compare
from genotype import Genotype


def get_config(file: Path):
    root = file.parent.resolve()
    while not (config := root.joinpath('evolution.json')).exists():
        root = root.parent
        if root == Path('/') or root == Path('.'):
            raise ValueError(f"Could not find evolution.json in any parent"
                             f" directory to '{file}'")
    config, data = Evolver.load_config(config)
    assert config is not None
    assert data is not None
    return config, data["data"]


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
    parser.add_argument("--render", type=str, nargs='?',
                        help="Render the genotype to file with provided"
                             " extension (or png)",
                        const="png", default=None)

    options = parser.parse_args(namespace=Options())

    options.rerun = True
    print(options)

    if options.file == Path("last"):
        options.file = Path("tmp/last/champion.json")
    file = options.file.resolve()
    assert file.exists(), f"{file=} doesn't exist"

    config, static_data = get_config(file)
    config.data_root = file.parent

    if options.experiment is not None and config.experiment != options.experiment:
        logging.info(f"Evaluating on {options.experiment} instead of {config.experiment}")
        config.experiment = options.experiment

    genome, data = Genotype.from_file(file)
    fitness, stats = data["fitness"], data.get("stats", {})

    if options.render:
        genome.to_dot(file, options.render, static_data)

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
