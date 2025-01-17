"""Evaluator class."""
import logging
import os
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated, ClassVar

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
                   verbose = True):
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
    def evaluate(cls, genotype: Genotype) -> list[float]:
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
                    video_directory=config.data_root,
                    overwrite=True,
                    width=512, height=512
                )
            )
        )

        # Calculate the xy displacements.
        xy_displacement = cls.fitness(robot, scene_states[0])

        return xy_displacement

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
