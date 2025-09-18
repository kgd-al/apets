from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import Body, ActiveHinge
from revolve2.modular_robot.brain import Brain as BrainFactory
from revolve2.modular_robot.brain import BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot_physical import UUIDKey


def mlp_structure(hinges: int, width: int, depth: int) -> nn.Sequential:
    sizes = [hinges] + [width] * depth + [hinges]
    seq = nn.Sequential()
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])

        seq.append(linear)
        seq.append(nn.Tanh())

    return seq


def mlp_weights(sequence: nn.Sequential, weights: np.ndarray):
    _wi, _wn = 0, 0
    for m in sequence.children():
        if isinstance(m, nn.Linear):
            _sn = m.weight.data.size()
            _wn = int(np.prod(_sn))
            m.weight.data[:] = torch.from_numpy(weights[_wi:_wi + _wn].reshape(_sn))
            _wi += _wn

            _wn = int(np.prod(m.bias.data.size()))
            m.bias.data[:] = torch.from_numpy(weights[_wi:_wi + _wn])
            _wi += _wn

    assert _wi == len(weights), f"Error: {len(weights)-_wi} unused weights"


class TensorBrain(BrainInstance):
    def __init__(self, hinges: list[ActiveHinge],
                 depth: int, width: int,
                 weights: np.ndarray,
                 simu_data: Optional[pd.DataFrame] = None):

        self._hinges = hinges
        self._n = len(hinges)

        self._modules = mlp_structure(self._n, width, depth)
        mlp_weights(self._modules, weights)

        self._step, self._time = 0, 0

        self._simu_data = None
        if simu_data is not None:
            self._simu_data = simu_data[[c for c in simu_data.columns if c.endswith("-pos")]]
            self._get_observation = self.recall_state
        else:
            self._get_observation = self.perceive_state

    def perceive_state(self, sensor_state: ModularRobotSensorState):
        return [
            sensor_state.get_active_hinge_sensor_state(hinge.sensors.active_hinge_sensor).position
            for hinge in self._hinges
        ]

    def recall_state(self, _):
        l = self._simu_data.iloc[self._step, :]
        l = [float(l[h.mujoco_name + "-pos"]) for h in self._hinges]
        print(f"Recalling: ~~{l}~~")
        return l

    def control(self, dt: float,
                sensor_state: ModularRobotSensorState,
                control_interface: ModularRobotControlInterface) -> None:

        state = self._get_observation(sensor_state)
        action = self._modules(torch.tensor(state)).detach().numpy()

        # print([h.name for h in self._hinges])
        # print(self._time, [" ".join(f"{x:.2g}" for x in a) for a in [np.array(state), action * 1.047197551]])

        for i, hinge in enumerate(self._hinges):
            control_interface.set_active_hinge_target(
                hinge, float(action[i] * hinge.range)
            )

        self._time += dt
        self._step += 1


class TensorBrainFactory(BrainFactory):
    def __init__(self, body: Body,
                 depth: int, width: int,
                 weights: np.ndarray):
        self._hinges = body.find_modules_of_type(ActiveHinge)
        self.depth, self.width = depth, width
        self._weights = weights

        self._total_recall = False

    def set_total_recall(self, recall: bool):
        self._total_recall = recall
        if recall and not hasattr(self, "simu_data"):
            raise RuntimeError("Requesting total recall mode without simulation ground truth."
                               " See extract_controller for details")

    def make_instance(self) -> TensorBrain:
        return TensorBrain(
            hinges=self._hinges,
            depth=self.depth, width=self.width, weights=self._weights,
            simu_data=None if not self._total_recall else getattr(self, "simu_data", None))
