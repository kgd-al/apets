import argparse
import math
import pickle
from pathlib import Path

import pandas as pd

from apets.hack.cma_es.evolve import bco
from apets.hack.tensor_brain import TensorBrainFactory
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic
from revolve2.standards import modular_robots_v2


def cpg_brain(weights, body, neighborhood):
    body, cpg_network_structure, output_mapping = bco(body, neighborhood)
    return BrainCpgNetworkStatic.uniform_from_params(
        params=weights,
        cpg_network_structure=cpg_network_structure,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=output_mapping,
    )


def mlp_brain(body, depth, width, weights):
    body = modular_robots_v2.get(body)
    return TensorBrainFactory(
        body=body,
        width=int(width), depth=int(depth), weights=weights
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
        brain = extract_cma(args)

    elif file == "model.zip":
        print("Extracting mlp controller from stable-baselines3")
        brain = extract_sb3(args)

    else:
        raise RuntimeError(f"What the hell?")

    if (joint_data := args.file.with_name("joints_data.csv")).exists():
        joints_df = pd.read_csv(joint_data, index_col=0)
        brain.simu_data = joints_df

    out_file = args.file.with_name("revolve_brain.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(brain, f)

    print("Saved revolve controller to", out_file, end='')
    if "joints_df" in locals():
        print(" (with simulated-ground-truth data)")
    print()


def extract_cma(args):
    trainer, reward, arch, run, _ = args.file.parts[-5:]
    assert trainer == "cma"
    arch, *arch_params = arch.split("-")
    print(arch, arch_params)
    print(f"Extracting {arch} controller from cma-es")
    with open(args.file, "rb") as f:
        es = pickle.loads(f.read())

        weights = es.result.xbest
        print(weights)
        if arch == "cpg":
            neighborhood = int(arch_params[0])
            brain = cpg_brain(es.result.xbest, args.body, neighborhood)

        elif arch == "mlp":
            depth, width = arch_params
            brain = mlp_brain(args.body, depth, width, weights)

    return brain


if __name__ == "__main__":
    main()