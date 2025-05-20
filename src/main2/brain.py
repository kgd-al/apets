import functools
import logging
import math
import pprint
from random import Random
from typing import List, Tuple, Optional

import cv2
import numpy as np
from abrain import Genome, Point3D as Point, ANN3D as ANN, CPPN
from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import Body, ActiveHinge
from revolve2.modular_robot.body.sensors import CameraSensor
from revolve2.modular_robot.brain import Brain as BrainFactory
from revolve2.modular_robot.brain import BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState

from body import compute_positions
from _retina_mapping import x_aligned as retina_mapper_x, ternary_1d as retina_mapper_rg
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighbor, BrainCpgInstance


class BrainSensoryCpgInstance(BrainCpgInstance):
    _output_mapping: list[tuple[int, tuple[ActiveHinge, int]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pprint.pprint(self._output_mapping)

    def control(self,
                dt: float,
                sensor_state: ModularRobotSensorState,
                control_interface: ModularRobotControlInterface) -> None:
        self._state = self._rk45(self._state, self._weight_matrix, dt)

        # Set active hinge targets to match newly calculated state.
        for state_index, (active_hinge, side) in self._output_mapping:
            control_interface.set_active_hinge_target(
                active_hinge, float(self._state[state_index]) * active_hinge.range
            )


class _BrainCpgNetworkNeighbor(BrainCpgNetworkNeighbor):
    def __init__(self, dna: Genome, body: Body, hinges_coordinates):
        self.dna = dna
        super().__init__(body)

        pprint.pprint(hinges_coordinates)
        self._output_mapping_with_side = [
            (index, (hinge, self._sign(hinges_coordinates[hinge].y)))
            for index, hinge in self._output_mapping
        ]

    @staticmethod
    def _sign(x):
        if round(x, 3) == 0:
            return 0
        else:
            return math.copysign(1, x)

    def _make_weights(
        self,
        active_hinges: list[ActiveHinge],
        connections: list[tuple[ActiveHinge, ActiveHinge]],
        body: Body,
    ) -> tuple[list[float], list[float]]:
        cppn = CPPN(self.dna)

        internal_weights = [
            cppn(0,
                 pos.x, pos.y, pos.z,
                 pos.x, pos.y, pos.z)
            for pos in [
                body.grid_position(active_hinge) for active_hinge in active_hinges
            ]
        ]

        external_weights = [
            cppn(0,
                 pos1.x, pos1.y, pos1.z,
                 pos2.x, pos2.y, pos2.z)
            for (pos1, pos2) in [
                (body.grid_position(active_hinge1), body.grid_position(active_hinge2))
                for (active_hinge1, active_hinge2) in connections
            ]
        ]

        return internal_weights, external_weights

    def make_instance(self) -> BrainSensoryCpgInstance:
        return BrainSensoryCpgInstance(
            initial_state=self._initial_state,
            weight_matrix=self._weight_matrix,
            output_mapping=self._output_mapping_with_side,
        )


class HackInstance(BrainInstance):
    def __init__(self, cpg_network: BrainSensoryCpgInstance):
        self.stem = cpg_network

        self._step, self._time = 0, 0

        self.__brain_dead = False
        if self.__brain_dead:
            logging.warning("Brain dead!")

        self.__fake_brain = False
        if self.__fake_brain:
            logging.warning("Fake brain!")
            self.rng = Random(0)

    def reset(self):
        self.brain.reset()
        self._step, self._time = 0, 0

    def control(self, dt: float,
                sensor_state: ModularRobotSensorState,
                control_interface: ModularRobotControlInterface) -> None:

        if self.__brain_dead:
            return

        self.stem.control(dt, sensor_state, control_interface)


class HackFactory(BrainFactory):
    def __init__(self,
                 body: Body, stem: Genome, brain: Genome,
                 with_labels=False, _id: int = 0):
        logging.debug(f"Creating a brain factory for {brain.id()}")

        h_coords = {m: p for m, p in compute_positions(body).items() if isinstance(m, ActiveHinge)}

        self._cpg_network = _BrainCpgNetworkNeighbor(stem, body, h_coords)
        self._brain = brain
        self._id = _id

    def make_instance(self) -> HackInstance:
        return HackInstance(
            cpg_network=self._cpg_network.make_instance()
        )


def develop(body: Body, stem: Genome, brain: Genome, with_labels=False, _id: int = 0):
    return HackFactory(body, stem, brain, with_labels, _id)
