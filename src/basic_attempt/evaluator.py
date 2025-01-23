"""Evaluator class."""
import json
import logging
import math
import numbers
import os
import pprint
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Optional, Annotated, ClassVar, Tuple, Dict

from colorama import Style, Fore
from revolve2.experimentation.evolution.abstract_elements import Evaluator as Eval
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BrickV2
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    simulate_scenes,
)
from revolve2.simulation.simulator import RecordSettings
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import fitness_functions, terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

import abrain
from abrain.neat.evolver import EvaluationResult
from brain import ABrainInstance
from config import Config, ConfigBase
from genotype import Genotype


@dataclass
class Options(ConfigBase):
    rerun: Annotated[bool, "Is this a rerun or are we evolving"] = False
    headless: Annotated[bool, "Do you think you need a GUI?"] = True
    movie: Annotated[bool, "Do you want a nice video of the robot?"] = False
    start_paused: Annotated[bool, "Should we let you a good look at the robot first?"] = False
    file: Annotated[Optional[Path], "If a rerun, path to the investigated robot's genome"] = None


class Evaluator(Eval):
    """Provides evaluation of robots."""

    config: ClassVar[Config] = None
    options: ClassVar[Options] = None

    _log: ClassVar[logging.Logger] = None

    @classmethod
    def initialize(cls, config: Config, options: Optional[Options] = None,
                   verbose=True):
        cls.config = config

        options = options or Options()
        cls.options = options
        # if options.rerun:
        #     options.rerun = False
        #     config.num_simulators = None
        #     options.headless = True

        if options.rerun:
            config.threads = 1
        elif config.threads is None:
            config.threads = len(os.sched_getaffinity(0))

        cls._data_folder = config.data_root
        if options and options.file is not None:
            cls._data_folder = options.file.parent

        cls._log = getattr(config, "logger", None)
        if cls._log and verbose:
            cls._log.info(f"Configuration:\n"
                          f"{pprint.pformat(cls.config)}\n"
                          f"{pprint.pformat(cls.options)}")

    @classmethod
    def evaluate(cls, genotype: Genotype) -> EvaluationResult:
        """
        Evaluate a *single* robot.

        Fitness is the distance traveled on the xy plane.

        :param genotype: The genotype to develop into a robot and then simulate.
        :returns: Fitness of the robot.
        """

        config, options = cls.config, cls.options

        try:
            simulator = LocalSimulator(
                headless=options.headless,
                num_simulators=1,
                start_paused=options.start_paused,
                # viewer_type="native"
            )
            terrain = terrains.flat()

            robot = genotype.develop(config)

            controller: ABrainInstance = robot.brain.make_instance()
            brain: abrain.ANN3D = controller.brain
            if not brain.empty():
                brain.render3D().write_html(config.data_root.joinpath("brain.html"))

            # Create the scenes.
            scene = ModularRobotScene(terrain=terrain)
            scene.add_robot(robot)

            # Simulate all scenes.
            scene_states = simulate_scenes(
                simulator=simulator,
                batch_parameters=make_standard_batch_parameters(
                    simulation_time=config.simulation_duration,
                ),
                scenes=[scene],
                record_settings=(
                    None
                    if not options.rerun else
                    RecordSettings(
                        video_directory=str(config.data_root),
                        overwrite=True,
                        width=512, height=512
                    )
                )
            )

            # Calculate the xy displacements.
            xy_displacement = cls.fitness(robot, scene_states[0])

            return EvaluationResult(fitness=xy_displacement)

        except Exception as e:
            f = config.data_root.joinpath("failures")
            f.mkdir(parents=True, exist_ok=True)
            f = f.joinpath(f"{genotype.id()}.json")
            if not options.rerun:
                cls._log.error(
                    f"Evaluation failed: {e.__class__.__name__}({e})."
                    f" Guilty genotype written to {f}.")
                cls._log.exception("Stack trace:")

            else:
                raise RuntimeError("Evaluation failed") from e

            return EvaluationResult()

    @staticmethod
    def fitness(robot, states):
        modules = {
            t: len(robot.body.find_modules_of_type(t))
            for t in [BrickV2, ActiveHingeV2]
        }

        i = int(.8*(len(states)-1))
        def pos(_i): return states[_i].get_modular_robot_simulation_state(robot)
        return (
                .01 * fitness_functions.xy_displacement(pos(0), pos(i-1))
                + fitness_functions.xy_displacement(pos(i), pos(-1))
        ) / max(1, modules[BrickV2] + 2 * modules[ActiveHingeV2])


def performance_compare(lhs: EvaluationResult, rhs: EvaluationResult, verbosity):
    width = 20
    key_width = max(len(k) for keys in
                    [["fitness"], lhs.stats or [], rhs.stats or []]
                    # [lhs.fitnesses, lhs.stats, rhs.fitnessess, rhs.stats]
                    for k in keys) + 1

    def s_format(s=''): return f"{s:{width}}"

    def f_format(f):
        if isinstance(f, numbers.Number):
            # return f"{f}"[:width-3] + "..."
            return s_format(f"{f:g}")
        else:
            return "\n" + pprint.pformat(f, width=width)

    def map_compare(lhs_d: Dict[str, float], rhs_d: Dict[str, float]):
        output, code = "", 0
        lhs_keys, rhs_keys = set(lhs_d.keys()), set(rhs_d.keys())
        all_keys = sorted(lhs_keys.union(rhs_keys))
        for k in all_keys:
            output += f"{k:>{key_width}}: "
            lhs_v, rhs_v = lhs_d.get(k), rhs_d.get(k)
            if lhs_v is None:
                output += f"{Fore.YELLOW}{s_format()} > {f_format(rhs_v)}"
            elif rhs_v is None:
                output += f"{Fore.YELLOW}{f_format(lhs_v)} <"
            else:
                if lhs_v != rhs_v:
                    lhs_str, rhs_str = f_format(lhs_v), f_format(rhs_v)
                    if isinstance(lhs_v, numbers.Number):
                        diff = rhs_v - lhs_v
                        ratio = math.inf if lhs_v == 0 else diff/math.fabs(lhs_v)
                        output += f"{Fore.RED}{lhs_str} | {rhs_str}" \
                                  f"\t({diff}, {100*ratio:.2f}%)"
                    else:
                        output += "\n"
                        for lhs_item, rhs_item in zip(lhs_str.split('\n'), rhs_str.split('\n')):
                            if lhs_item != rhs_item:
                                output += Fore.RED
                            output += f"{lhs_item:{width}s} | {rhs_item:{width}s}"
                            if lhs_item != rhs_item:
                                output += Style.RESET_ALL
                            output += "\n"
                    code = 1
                else:
                    output += f"{Fore.GREEN}{f_format(lhs_v)}"

            output += f"{Style.RESET_ALL}\n"
        return output, code

    def json_compliant(obj): return json.loads(json.dumps(obj))

    f_str, f_code = map_compare({"fitness": lhs.fitness},
                                {"fitness": rhs.fitness})
    # d_str, d_code = map_compare(lhs.descriptors,
    #                             json_compliant(rhs.descriptors))
    s_str, s_code = map_compare(lhs.stats or {},
                                json_compliant(rhs.stats or {}))
    max_width = max(len(line) for text in [f_str, s_str] for line in text.split('\n'))
    if verbosity == 1:
        summary = []
        codes = {0: Fore.GREEN, 1: Fore.RED}
        for _code, name in [(f_code, "fitness"),
                            # (d_code, "descriptors"),
                            (s_code, "stats")]:
            summary.append(f"{codes[_code]}{name}{Style.RESET_ALL}")
        print(f"Performance summary: {lhs.fitness} ({' '.join(summary)})")
        # print(f"Performance summary: {lhs.fitnesses} ({' '.join(summary)})")

    elif verbosity > 1:
        def header(): print("-"*max_width)
        print("Performance summary:")
        header()
        print(f_str, end='')
        header()
        # print(d_str, end='')
        # header()
        print(s_str, end='')
        header()
        print()

    # return max([f_code, d_code, s_code])
    return max([f_code, s_code])
