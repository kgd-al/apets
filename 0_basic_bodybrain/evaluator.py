"""Evaluator class."""
import logging
import pprint

import abrain
from revolve2.simulation.simulator import RecordSettings

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

from brain import ABrainInstance


class Evaluator(Eval):
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators,
            # start_paused=(num_simulators == 1),
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
        logging.info(f"Starting evaluation of {len(population)} robots.")
        robots = [genotype.develop() for genotype in population]

        if len(robots) == 1:
            controller: ABrainInstance = robots[0].brain.make_instance()
            brain: abrain.ANN3D = controller.brain
            if not brain.empty():
                brain.render3D().write_html("brain.html")

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
                simulation_time=5
            ),
            scenes=scenes,
            record_settings=(
                None if len(population) > 1 else
                RecordSettings(
                    video_directory="foo",
                    overwrite=True,
                    width=512, height=512
                )
            )
        )

        # Calculate the xy displacements.
        xy_displacements = [
            fitness_functions.xy_displacement(
                states[0].get_modular_robot_simulation_state(robot),
                states[-1].get_modular_robot_simulation_state(robot),
            )
            for robot, states in zip(robots, scene_states)
        ]
        logging.info(f"Finished evaluation of {len(population)} robots."
                     f" Fitnesses:\n{pprint.pformat(xy_displacements)}")

        return xy_displacements
