import argparse
import math
import pickle
from pathlib import Path

from apets.hack.cma_es.evolve import bco
from apets.hack.tensor_brain import TensorBrainFactory
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic


def cpg_brain(weights, body, neighborhood):
    body, cpg_network_structure, output_mapping = bco(body, neighborhood)
    return BrainCpgNetworkStatic.uniform_from_params(
        params=weights,
        cpg_network_structure=cpg_network_structure,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=output_mapping,
    )


def mlp_brain(self, weights):
    return TensorBrainFactory(
        body=self._body,
        width=self._width, depth=self._depth, weights=weights
    )


def main():
    parser = argparse.ArgumentParser("Extracts mlp or cpg controller from"
                                     " cma-es archive or stable-baselines3"
                                     " model")
    parser.add_argument("file", type=Path,
                        help="path to the file to process")
    parser.add_argument("--body", choices=["spider"],
                        help="robot morphology", required=True)

    args = parser.parse_args()

    file = args.file.name
    if file == "cma-es.pkl":
        extract_cma(args)

    elif file == "model.zip":
        print("Extracting mlp controller from stable-baselines3")
        extract_sb3(args)


def extract_cma(args):
    trainer, reward, arch, run, _ = args.file.parts[-5:]
    assert trainer == "cma"
    arch, arch_params = arch.split("-")
    print(arch, arch_params)
    print(f"Extracting {arch} controller from cma-es")
    with open(args.file, "rb") as f:
        es = pickle.loads(f.read())

        print(es.result.xbest)
        if arch == "cpg":
            neighborhood = int(arch_params[0])
            brain = cpg_brain(es.result.xbest, args.body, neighborhood)

        with open(args.file.with_name("revolve_brain.pkl"), "wb") as f:
            pickle.dump(brain, f)


if __name__ == "__main__":
    main()