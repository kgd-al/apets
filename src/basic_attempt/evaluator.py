"""Evaluator class."""
import json
import logging
import math
import numbers
import os
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated, ClassVar, Dict, Callable

from colorama import Style, Fore
from pyrr import Vector3
from revolve2.experimentation.evolution.abstract_elements import Evaluator as Eval
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BrickV2
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    simulate_scenes,
)
from revolve2.simulation.scene import Pose
from revolve2.simulation.simulator import RecordSettings
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.interactive_objects import Ball
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

from abrain import ANN3D
from abrain.neat.config import ConfigBase
from abrain.neat.evolver import EvaluationResult
from brain import ABrainInstance
from config import Config, ExperimentType, EXPERIMENT_DURATIONS
from genotype import Genotype


@dataclass
class Options(ConfigBase):
    rerun: Annotated[bool, "Is this a rerun or are we evolving"] = False
    headless: Annotated[bool, "Do you think you need a GUI?"] = True
    movie: Annotated[bool, "Do you want a nice video of the robot?"] = False
    start_paused: Annotated[bool, "Should we let you a good look at the robot first?"] = False

    file: Annotated[Optional[Path], "Path to the genome under re-evaluation."] = None
    duration: Annotated[Optional[int], ("Simulation duration in seconds."
                                        " Overwrites the value used in evolution")] = None


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

        if config.simulation_duration is None:
            if options.rerun and options.duration is not None:
                config.simulation_duration = options.duration
            else:
                config.simulation_duration = EXPERIMENT_DURATIONS[config.experiment]

        cls._data_folder = config.data_root

        cls._log = getattr(config, "logger", None)
        if cls._log and verbose:
            cls._log.info(f"Configuration:\n"
                          f"{pprint.pformat(cls.config)}\n"
                          f"{pprint.pformat(cls.options)}")

        cls._fitness = getattr(cls, f"fitness_{config.experiment.lower()}")

    @classmethod
    def evaluate(cls, genotype: Genotype) -> EvaluationResult:
        """
        Evaluate a *single* robot.

        Fitness is the distance traveled on the xy plane.

        :param genotype: The genotype to develop into a robot and then simulate.
        :returns: Fitness of the robot.
        """

        config, options = cls.config, cls.options

        simulator = LocalSimulator(
            headless=options.headless,
            num_simulators=1,
            start_paused=options.start_paused,
            # viewer_type="native"
        )
        terrain = terrains.flat()

        robots = [genotype.develop(config)]

        if options.rerun:
            controller: ABrainInstance = robots[0].brain.make_instance()
            brain: ANN3D = controller.brain
            if not brain.empty():
                brain.render3D().write_html(config.data_root.joinpath("brain.html"))

        # Create the scenes.
        scene = ModularRobotScene(terrain=terrain)
        for robot in robots:
            scene.add_robot(robot)

        objects = []
        if config.experiment is not ExperimentType.LOCOMOTION:
            ball = Ball(radius=0.05, mass=0.05,
                        pose=Pose(Vector3([.25, .25, 0.])))
            scene.add_interactive_object(ball)
            objects.append(ball)
            # scene.mark_starting_position()

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=simulator,
            batch_parameters=make_standard_batch_parameters(
                simulation_time=options.duration or config.simulation_duration,
            ),
            scenes=scene,
            record_settings=(
                None
                if not options.movie else
                RecordSettings(
                    video_directory=str(config.data_root),
                    overwrite=True,
                    width=512, height=512
                )
            )
        )

        fitness = cls._fitness(robots, objects, scene_states)
        try:
            assert not math.isnan(fitness) and not math.isinf(fitness), f"{fitness=}"
        except Exception as e:
            raise RuntimeError(f"{fitness=}") from e
        return EvaluationResult(fitness=fitness)

    @staticmethod
    def rename_movie(genome_file: Path):
        genome_file.parent.joinpath("0.mp4").rename(genome_file.with_suffix(".mp4"))

    @staticmethod
    def fitness_locomotion(robots, objects, states):
        # Get as far as possible with as few modules as possible
        # Hinges cost more.

        assert len(robots) == 1, "This experiment is not meant for multiple robots"
        assert len(objects) == 0, "This experiment does not require interactive objects"
        robot = robots[0]
        modules = {
            t: len(robot.body.find_modules_of_type(t))
            for t in [BrickV2, ActiveHingeV2]
        }

        i = int(.8*(len(states)-1))
        pos = _robot_pos(robot, states)
        return (
                .01 * euclidian_distance(pos(0), pos(i-1))
                + euclidian_distance(pos(i), pos(-1))
        ) / max(1, modules[BrickV2] + 2 * modules[ActiveHingeV2])

    @staticmethod
    def fitness_punch_once(robots, objects, states):
        # Punch the ball as far as possible
        # If you do not touch the ball at least try to get close to it
        assert len(robots) == 1, "This experiment is not meant for multiple robots."
        robot = robots[0]
        assert len(objects) == 1, "This experiment requires a single object."
        ball = objects[0]
        assert isinstance(ball, Ball), "This experiment's interactive object should be a Ball."

        robot_pos = _robot_pos(robot, states)
        ball_pos = _ball_pos(ball, states)

        ball_dist = euclidian_distance(ball_pos(0), ball_pos(-1))
        if ball_dist > 0:
            return ball_dist
        else:
            return -euclidian_distance(ball_pos(-1), robot_pos(-1))

    @staticmethod
    def fitness_punch_ahead(robots, objects, states):
        # Punch the ball *forward* as far as possible
        # If you do not touch the ball at least try to get close to it
        assert len(robots) == 1, "This experiment is not meant for multiple robots."
        robot = robots[0]
        assert len(objects) == 1, "This experiment requires a single object."
        ball = objects[0]
        assert isinstance(ball, Ball), "This experiment's interactive object should be a Ball."

        robot_pos = _robot_pos(robot, states)
        ball_pos = _ball_pos(ball, states)

        b0, b1 = ball_pos(0), ball_pos(-1)
        ball_dist = euclidian_distance(b0, b1)
        if ball_dist > 0:
            return clip(-5, b1.y - b0.y, 5) - clip(0, abs(b1.x - b0.x), 5)
        else:
            return -10 - .1 * euclidian_distance(ball_pos(-1), robot_pos(-1))


def clip(low, value, high): return max(low, min(value, high))


def _robot_pos(robot, states) -> Callable[[int], Vector3]:
    return lambda i: states[i].get_modular_robot_simulation_state(robot).get_pose().position


def _ball_pos(ball, states) -> Callable[[int], Vector3]:
    return lambda i: states[i]._simulation_state.get_multi_body_system_pose(ball).position


def euclidian_distance(start, finish):
    return math.sqrt((finish.x - start.x)**2 + (finish.x - start.x)**2)


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
