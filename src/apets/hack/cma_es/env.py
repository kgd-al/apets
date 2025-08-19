import functools
import math
import pprint
from hashlib import sha1
from pathlib import Path
from typing import Optional

import numpy as np

from apets.hack.body import compute_positions
from apets.hack.evaluator import MoveForwardFitness, Evaluator
from apets.hack.local_simulator import LocalSimulator, StepwiseFitnessFunction
from apets.hack.tensor_brain import TensorBrainFactory, mlp_structure
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic
from revolve2.modular_robot_simulation import ModularRobotScene


class Environment:
    def __init__(self, body, reward, **kwargs):
        self._body = body

        match (arch := kwargs.pop("arch")):
            case "mlp":
                self._arch = arch
                self._width = kwargs.pop("mlp_width")
                self._depth = kwargs.pop("mlp_depth")
                self._params = sum(
                    p.numel() for p in
                    mlp_structure(
                        len(self._body.find_modules_of_type(ActiveHinge)),
                        self._width, self._depth).parameters()
                )
                self._brain_factory = self.mlp_brain

            case "cpg":
                self._arch = arch
                self._cpg_network_structure = kwargs.pop("cpg_network_structure")
                self._output_mapping = kwargs.pop("output_mapping")
                self._params = self._cpg_network_structure.num_connections
                self._brain_factory = self.cpg_brain

        self.rerun = kwargs.get('rerun', False)
        self.rotated = kwargs.pop('rotated', False)

        self.evaluate_one = functools.partial(
            self._evaluate, reward=reward, **kwargs)

    @property
    def num_parameters(self): return self._params

    def cpg_brain(self, weights):
        return BrainCpgNetworkStatic.uniform_from_params(
            params=weights,
            cpg_network_structure=self._cpg_network_structure,
            initial_state_uniform=math.sqrt(2) * 0.5,
            output_mapping=self._output_mapping,
        )

    def mlp_brain(self, weights):
        return TensorBrainFactory(
            body=self._body,
            width=self._width, depth=self._depth, weights=weights
        )

    def evaluate(self, weights: np.ndarray) -> float | StepwiseFitnessFunction:
        robot = ModularRobot(body=self._body, brain=self._brain_factory(weights))

        res = self.evaluate_one(
            Evaluator.scene(robot, rerun=self.rerun, rotation=self.rotated)
        )

        # weights_np_str = np.array2string(
        #     robot.brain.make_instance()._weight_matrix,
        #     precision=2, separator=',', suppress_small=True,
        #     floatmode="maxprec", max_line_width=1000
        # )
        # print(f"-- {res} - {sha1(weights).hexdigest()}:\n{weights_np_str}")

        return res

    @staticmethod
    def _evaluate(scene: ModularRobotScene, reward, **kwargs):
        render = kwargs.pop("render", False)
        fitness_function = MoveForwardFitness(
            reward=reward,
            rerun=kwargs.pop("rerun", False),
            render=render,
            log_trajectory=kwargs.pop("log_trajectory", False),
            backup_trajectory=False,
            log_reward=kwargs.pop("log_reward", False),
            introspective=kwargs.pop("introspective", False),
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
