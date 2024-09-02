"""Evaluator class."""
import logging
import os
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated

import abrain
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2, ActiveHingeV2, BrickV2
from revolve2.simulation.simulator import RecordSettings
from revolve2.simulators.mujoco_simulator.viewers import ViewerType

from genotype import Genotype

from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.experimentation.evolution.abstract_elements import Evaluator as Eval
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator

from config import Config, ConfigBase
from brain import ABrainInstance


@dataclass
class Options(ConfigBase):
    rerun: Annotated[bool, "Is this a rerun or are we evolving"] = False
    headless: Annotated[bool, "Do you think you need a GUI?"] = True
    movie: Annotated[bool, "Do you want a nice video of the robot?"] = False
    start_paused: Annotated[bool, "Should we let you a good look at the robot first?"] = False
    file: Annotated[Optional[Path], "If a rerun, path to the investigated robot's genome"] = None


class Evaluator(Eval):
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain

    def __init__(self, config: Config, options: Optional[Options] = None):
        options = options or Options()
        if options.rerun:
            config.num_simulators = 1
        elif config.num_simulators is None:
            config.num_simulators = len(os.sched_getaffinity(0))

        self._config = config
        self._options = options

        self._data_folder = config.data_root
        if options and options.file is not None:
            self._data_folder = options.file.parent

        self._log = hasattr(config, "logger")
        if self._log:
            self._config.logger.info(f"Configuration:\n"
                                     f"{pprint.pformat(self._config)}\n"
                                     f"{pprint.pformat(self._options)}")

        self._simulator = LocalSimulator(
            headless=options.headless, num_simulators=config.num_simulators,
            start_paused=options.start_paused,
            # viewer_type="native"
        )
        self._terrain = terrains.flat()

    def evaluate(
        self,
        population: list[Genotype],
    ) -> list[float]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param population: The robots to simulate.
        :returns: Fitnesses of the robots.
        """

        if self._log:
            self._config.logger.debug(f"Starting evaluation of {len(population)} robots.")
        robots = [genotype.develop() for genotype in population]

        if len(robots) == 1:
            controller: ABrainInstance = robots[0].brain.make_instance()
            brain: abrain.ANN3D = controller.brain
            if not brain.empty():
                brain.render3D().write_html(self._config.data_root.joinpath("brain.html"))

        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(
                simulation_time=self._config.simulation_duration,
            ),
            scenes=scenes,
            record_settings=(
                None if (len(population) > 1 or
                         self._simulator._viewer_type is ViewerType.NATIVE) else
                RecordSettings(
                    video_directory=self._data_folder,
                    overwrite=True,
                    width=512, height=512
                )
            )
        )

        # Calculate the xy displacements.
        xy_displacements = [
            self.fitness(robot, states)
            for robot, states in zip(robots, scene_states)
        ]
        if self._log:
            self._config.logger.debug(f"Finished evaluation of {len(population)} robots."
                                      f" Fitnesses:\n{pprint.pformat(xy_displacements)}")

        return xy_displacements

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
