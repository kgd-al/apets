import argparse
import math
import pickle
from pathlib import Path

import graphviz
import numpy as np
import pandas as pd

from apets.hack.body import compute_positions
from apets.hack.cma_es.evolve import bco
from apets.hack.tensor_brain import TensorBrainFactory
from revolve2.modular_robot.body.base import ActiveHinge, Core, Brick
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic
from revolve2.standards import modular_robots_v2


def cpg_to_dot(body, brain, cpg_network_structure, filename):
    om = {h: i for i, h in brain._output_mapping}
    # om_inv = {i: h for h, i in om.items()}
    modules_pos = {
        m: p for m, p in compute_positions(body).items()
    }

    dot = graphviz.Graph(
        "CPG", "connectivity pattern",
        engine="neato",
        graph_attr=dict(overlap="scale", outputorder="edgesfirst", splines="true"),
        node_attr=dict(style="filled", fillcolor="white"),
        # edge_attr=dict(dir="both")
    )

    for i, (m, p) in enumerate(modules_pos.items()):
        pos = f"{p.x},{p.y}!"
        name, style = None, None

        if isinstance(m, Core):
            name = "Core"
            style = dict(shape="rectangle")

        elif isinstance(m, ActiveHinge):
            name = f"H_{om[m]}"
            style = dict(shape="circle")

        elif isinstance(m, Brick):
            name = f"B_{i}"
            style = dict(shape="rectangle")

        else:
            raise RuntimeError(f"Unknown module type {type(m)}")

        if "_" in name:
            tokens = name.split("_")
            label = "".join(["<", tokens[0], "<sub>", tokens[1], "</sub>>"])

        else:
            label = name

        dot.node(name=name, label=label, pos=pos, **style)

    # for c in cpg_network_structure.cpgs:
    #     pos = modules_pos[om_inv[c.index]]
    #     dot.node(name=str(c.index), pos=f"{pos.x},{pos.y}!")

    for c in cpg_network_structure.connections:
        print()
        dot.edge(f"H_{c.cpg_index_lowest.index}", f"H_{c.cpg_index_highest.index}")

    print(filename)
    print(dot.source)
    dot.render(outfile=filename.with_name("brain.pdf"), cleanup=True)

    with np.printoptions(precision=1, linewidth=200):
        print(brain._weight_matrix)
        print(modules_pos)


def cpg_brain(weights, body, neighborhood, file):
    body, cpg_network_structure, output_mapping = bco(body, neighborhood)

    brain = BrainCpgNetworkStatic.uniform_from_params(
        params=weights,
        cpg_network_structure=cpg_network_structure,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=output_mapping,
    )

    cpg_to_dot(body, brain, cpg_network_structure, file)

    return brain


def mlp_brain(body, depth, width, weights):
    body = modular_robots_v2.get(body)
    return TensorBrainFactory(
        body=body,
        width=int(width), depth=int(depth), weights=weights
    )


def main(args):
    file = args.file.name
    if file == "cma-es.pkl":
        brain = extract_cma(args)

    elif file == "model.zip":
        if not args.quiet:
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

    if args.quiet < 2:
        print("Saved revolve controller to", out_file, end='')
        if "joints_df" in locals():
            print(" (with simulated-ground-truth data)")
        print()

    return out_file


def extract_cma(args):
    trainer, reward, arch, run, _ = args.file.parts[-5:]
    assert trainer == "cma"
    arch, *arch_params = arch.split("-")
    if not args.quiet:
        print(arch, arch_params)
        print(f"Extracting {arch} controller from cma-es")
    with open(args.file, "rb") as f:
        es = pickle.loads(f.read())

        weights = es.result.xbest
        if not args.quiet:
            print(weights)

        if arch == "cpg":
            neighborhood = int(arch_params[0])
            brain = cpg_brain(es.result.xbest, args.body, neighborhood, args.file)

        elif arch == "mlp":
            depth, width = arch_params
            brain = mlp_brain(args.body, depth, width, weights)

    return brain


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extracts mlp or cpg controller from"
                                     " cma-es archive or stable-baselines3"
                                     " model")
    parser.add_argument("file", type=Path,
                        help="path to the file to process")
    parser.add_argument("--body", choices=["spider"],
                        help="robot morphology", required=True)
    parser.add_argument("--quiet", "-q",
                        default=0, action="count",
                        help="I'll be quiet, sure")

    args = parser.parse_args()

    main(args)
