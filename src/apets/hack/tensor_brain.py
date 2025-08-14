import logging
import logging
import math
from random import Random

import numpy as np
import torch
from torch import nn

from abrain import Genome, CPPN
from apets.hack.body import compute_positions
from apets.hack.config import Config
from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import Body, ActiveHinge
from revolve2.modular_robot.brain import Brain as BrainFactory
from revolve2.modular_robot.brain import BrainInstance
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighbor, BrainCpgInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState


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
                 weights: np.ndarray):

        self._hinges = hinges
        self._n = len(hinges)

        self._modules = mlp_structure(self._n, width, depth)
        mlp_weights(self._modules, weights)

    def control(self, dt: float,
                sensor_state: ModularRobotSensorState,
                control_interface: ModularRobotControlInterface) -> None:

        state = [
            sensor_state.get_active_hinge_sensor_state(hinge.sensors.active_hinge_sensor).position
            for hinge in self._hinges
        ]

        action = self._modules(torch.tensor(state)).detach().numpy()

        for i, hinge in enumerate(self._hinges):
            control_interface.set_active_hinge_target(
                hinge, action[i] * hinge.range
            )


class TensorBrainFactory(BrainFactory):
    def __init__(self, body: Body,
                 depth: int, width: int,
                 weights: np.ndarray):
        self._hinges = body.find_modules_of_type(ActiveHinge)
        self.depth, self.width = depth, width
        self._weights = weights

    def make_instance(self) -> TensorBrain:
        return TensorBrain(
            hinges=self._hinges,
            depth=self.depth, width=self.width, weights=self._weights)
