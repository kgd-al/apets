import functools
import math
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np

from apets.hack.evaluator import MoveForwardFitness, Evaluator
from apets.hack.local_simulator import LocalSimulator, StepwiseFitnessFunction
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic
from revolve2.modular_robot_simulation import ModularRobotScene


class Environment:
    def __init__(self, body, reward,
                 cpg_network_structure, output_mapping,
                 **kwargs):
        self._body = body

        self._cpg_network_structure = cpg_network_structure
        self._output_mapping = output_mapping

        self.rerun = kwargs.get('rerun', False)
        self.rotated = kwargs.pop('rotated', False)

        self.evaluate_one = functools.partial(
            self._evaluate, reward=reward, **kwargs)

    def evaluate(self, weights: np.ndarray) -> float | StepwiseFitnessFunction:
        robot = ModularRobot(
            body=self._body,
            brain=BrainCpgNetworkStatic.uniform_from_params(
                params=weights,
                cpg_network_structure=self._cpg_network_structure,
                initial_state_uniform=math.sqrt(2) * 0.5,
                output_mapping=self._output_mapping,
            ),
        )

        return self.evaluate_one(
            Evaluator.scene(robot, rerun=self.rerun, rotation=self.rotated)
        )

    @staticmethod
    def _evaluate(scene: ModularRobotScene, reward, **kwargs):
        render = kwargs.pop("render", False)
        fitness_function = MoveForwardFitness(
            reward=reward,
            rerun=kwargs.pop("rerun", False),
            render=render,
            log_trajectory=kwargs.pop("log_trajectory", False), backup_trajectory=False,
            log_reward=kwargs.pop("log_reward", False),
        )

        return_ff = kwargs.pop("return_ff", False)

        plot_path: Optional[Path] = kwargs.pop("plot_path", None)
        simulator = LocalSimulator(scene=scene, fitness_function=fitness_function, **kwargs)
        simulator.run(render)

        if plot_path is not None and plot_path.exists():
            fitness_function.do_plots(plot_path)

        # Minimize negative value
        if return_ff:
            return fitness_function
        else:
            return -fitness_function.fitness
