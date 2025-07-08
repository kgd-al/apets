"""Main script for the example."""
import argparse
import json
import pickle
import pprint
import shutil
import time
from pathlib import Path

import cma
import graphviz
import matplotlib
import numpy as np
import pandas as pd

from apets.hack.body import compute_positions
from apets.hack.cma_es.env import Environment
from apets.hack.evaluator import MoveForwardFitness
from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.base import ActiveHinge, Core, Brick, AttachmentFace
from revolve2.modular_robot.body.v2 import BodyV2, ActiveHingeV2
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.simulation.simulator import RecordSettings
from revolve2.standards import modular_robots_v2
from revolve2.standards.simulation_parameters import STANDARD_CONTROL_FREQUENCY

folder = Path("tmp/cpg_study/")
folder.mkdir(parents=True, exist_ok=True)
print(folder.absolute())


def __fixed_neighbors(self, within_range: int) -> list[Module]:
    """
    Get the neighbours of this module with a certain range of the module tree.

    :param within_range: The range in which modules are considered a neighbour. Minimum is 1.
    :returns: The neighbouring modules.
    """
    queue, seen = [(self, 0)], set()

    while queue:
        module, depth = queue.pop()
        seen.add(module)

        if depth < within_range:
            if isinstance(module, AttachmentFace) and isinstance(module.parent, Core):
                modules = [
                    m
                    for face in module.parent.attachment_faces.values()
                    for m in face.children.values()
                    if face is not module
                ]
            else:
                modules = list(module.children.values()) + [module.parent]

            queue.extend([
                (mod, depth+1)
                for mod in modules
                if mod is not None and mod not in seen
            ])
    seen.remove(self)
    return list(seen)
Module.neighbours = __fixed_neighbors


def real_snake(l=3):
    body = BodyV2()
    module = body.core_v2.back_face.bottom = ActiveHingeV2(0.0)
    for _ in range(l-1):
        module.attachment = ActiveHingeV2(0.0)
        module = module.attachment
    return body


def to_dot(modules_pos, active_hinges, cpg_network_structure, filename):
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
            style = dict(shape="rectangle", color="red")
        elif isinstance(m, ActiveHinge):
            name = f"H_{active_hinges.index(m)}"
            style = dict(shape="circle")
        elif isinstance(m, Brick):
            name = f"B_{i}"
            style = dict(shape="rectangle", color="blue")
        if "_" in name:
            tokens = name.split("_")
            label = "".join(["<", tokens[0], "<sub>", tokens[1], "</sub>>"])
        else:
            label = name
        dot.node(name=name, label=label, pos=pos, **style)

    # for c in cpg_network_structure.cpgs:
    #     pos = modules_pos[active_hinges[c.index]]
    #     dot.node(name=str(c.index), pos=f"{pos.x},{pos.y}!")
    for c in cpg_network_structure.connections:
        dot.edge(f"H_{c.cpg_index_lowest.index}", f"H_{c.cpg_index_highest.index}")

    print(dot.source)
    dot.render(outfile=filename, cleanup=True)


def bco(body_name, neighborhood):
    if body_name == "real_snake":
        body = real_snake(5)
    else:
        body = modular_robots_v2.get(body_name)

    modules_pos = {
        m: p for m, p in compute_positions(body).items()
    }
    active_hinges = [m for m in modules_pos if isinstance(m, ActiveHinge)]

    # Create a structure for the CPG network from these hinges.
    # This also returns a mapping between active hinges and the index of there corresponding cpg in the network.
    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges, neighborhood)

    _s = cpg_network_structure
    # print(f"> {_s.num_cpgs=}, {_s.num_connections=}, {_s.num_states=}")
    # to_dot(modules_pos, active_hinges, cpg_network_structure, folder.joinpath(f"{body_name}-n{neighborhood}.png"))

    return body, cpg_network_structure, output_mapping


# # for body in ["spider"]:#["real_snake", "snake", "spider"]:
# #     for i in [1, 2]:#range(6):
# for body in ["real_snake", "snake", "spider"]:
#     for i in range(7):
#         # print(f"{body}-{i}")
#         _b, _s, _m = bco(body, i)
#         print(f"{body}-{i}: {_s.num_connections}")
# exit(42)


def evolve(args):
    folder = args.output_folder
    folder_str = str(folder) + "/"

    with open(folder.joinpath("config.json"), "wt") as f:
        config = vars(args).copy()
        config["output_folder"] = str(args.output_folder)
        print("Configuration:", pprint.pformat(config))
        f.write(json.dumps(config))

    body, cpg_network_structure, output_mapping = bco(args.body, args.neighborhood)
    evaluator = Environment(
        body=body,
        reward=args.reward,
        cpg_network_structure=cpg_network_structure,
        output_mapping=output_mapping,

        simulation_time=args.simulation_time,
        rotated=args.rotated,
    )

    # Initial parameter values for the brain.
    initial_mean = cpg_network_structure.num_connections * [0.5]

    # We use the CMA-ES optimizer from the cma python package.
    options = cma.CMAOptions()
    options.set("verb_filenameprefix", folder_str)
    options.set("bounds", [-1.0, 1.0])
    options.set("seed", args.seed)
    es = cma.CMAEvolutionStrategy(initial_mean, args.initial_std, options)
    args.threads = 0
    es.optimize(evaluator.evaluate, maxfun=args.budget, n_jobs=args.threads, verb_disp=1)
    with open(folder.joinpath("cma-es.pkl"), "wb") as f:
        f.write(es.pickle_dumps())

    res = es.result_pretty()
    matplotlib.use("agg")
    cma.plot(folder_str, abscissa=1)
    # plt.tight_layout()
    cma.s.figsave(folder.joinpath('plot.png'), bbox_inches='tight')  # save current figure
    cma.s.figsave(folder.joinpath('plot.pdf'), bbox_inches='tight')  # save current figure

    rerun_fitness = evaluator.evaluate(res.xbest)
    if rerun_fitness != res.fbest:
        print("Different fitness value on rerun:")
        print(res.fbest, res.xbest)
        print("Rerun:", rerun_fitness)

    args.evolution_simulation_time = args.simulation_time
    args.simulation_time = 15
    args.movie = True
    rerun(args)


def rerun(args):
    folder = args.output_folder
    body, cpg_network_structure, output_mapping = bco(args.body, args.neighborhood)

    env_kwargs = dict(
        body=body,
        reward=args.reward,
        cpg_network_structure=cpg_network_structure,
        output_mapping=output_mapping,

        simulation_time=args.simulation_time,
        rotated=args.rotated,

        rerun=True,
        render=not args.headless or args.movie, headless=args.headless,
        log_trajectory=True, log_reward=True,
        plot_path=folder,

        return_ff=True,
    )
    if args.movie:
        env_kwargs["record_settings"] = RecordSettings(
            video_directory=folder,
            overwrite=True,
            fps=25,
            width=480, height=480,

            camera_id=2
        )

    evaluator = Environment(**env_kwargs)

    with open(folder.joinpath("cma-es.pkl"), "rb") as f:
        es = pickle.loads(f.read())

    start_time = time.time()
    fitness_function = evaluator.evaluate(es.result.xbest)
    fitness = -fitness_function.fitness

    steps_per_episode = (getattr(args, "evolution_simulation_time", args.simulation_time)
                         * STANDARD_CONTROL_FREQUENCY)

    summary = {
        "arch": "cpg",
        "budget": args.budget * steps_per_episode,
        "neighborhood": args.neighborhood,
        "reward": args.reward,
        "run": args.seed,
        "body": args.body + ("45" if args.rotated else ""),
        "params": cpg_network_structure.num_connections,
        "tps": steps_per_episode / (time.time() - start_time),
    }
    summary.update(fitness_function.infos)
    summary = pd.DataFrame.from_dict({k: [v] for k, v in summary.items()})
    summary.index = [folder]

    summary.to_csv(folder.joinpath("summary.csv"))
    print(summary.to_string())

    return fitness


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Generic")
    group.add_argument("-T", "--simulation-time", default=10, type=float)
    group.add_argument("-R", "--reward", required=True,
                       type=lambda _str: _str.lower(),
                       choices=[v.name.lower() for v in MoveForwardFitness.RewardType])

    group = parser.add_argument_group("Evolution")
    group.add_argument("--body", default="spider",
                       help=f"Morphology to use for the robot", choices=["spider"])
    group.add_argument("-r", "--rotated", default=False, action="store_true",
                       help="Whether the front of the robot is rotated by 45 degrees")
    group.add_argument("-b", "--budget", default=20, type=int,
                       help="Maximal number of evaluations")
    group.add_argument("-s", "--seed", default=0, type=int,
                       help="Seed for the RNG Gods")
    group.add_argument("-o", "--output-folder", default=None, type=Path,
                       help="Where to store the model and associated data")
    group.add_argument("--threads", default=None, type=int,
                       help="Number of parallel environments to use")
    group.add_argument("--overwrite", default=False, action="store_true", )

    group.add_argument("--neighborhood", type=int, default=2,
                       help="Neighborhood size for CPG network connectivity")
    group.add_argument("--initial-std", type=float, default=.5,
                       help="Initial standard deviation for CMA-ES")

    group = parser.add_argument_group("Evaluation")
    group.add_argument("--rerun", type=str, default=None,
                       const='', nargs='?')
    group.add_argument("--movie", default=False, action="store_true", )
    group.add_argument("--headless", default=False, action="store_true", )

    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = Path("tmp/cma/").joinpath(f"run-{args.seed}")
    folder = args.output_folder

    _rerun = folder.exists() and not args.overwrite
    if folder.exists():
        if args.overwrite:
            shutil.rmtree(folder)
        elif args.rerun is None:
            raise ValueError(f"Log folder '{folder}' already exists and overwriting was not requested")
    folder.mkdir(parents=True, exist_ok=True)

    if _rerun:
        rerun(args)
    else:
        assert args.headless is True
        evolve(args)


if __name__ == "__main__":
    main()
