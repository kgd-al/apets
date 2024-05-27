import logging
import math
import pprint
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from abrain import Genome, Point3D as Point, ANN3D as ANN
from pyrr import Vector3, Quaternion
from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.base import Body, ActiveHinge, Core
from revolve2.modular_robot.body.base._body import _GridMaker
from revolve2.modular_robot.brain import Brain as BrainFactory
from revolve2.modular_robot.brain import BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState


class ABrainInstance(BrainInstance):
    def __init__(self, genome: Genome,
                 inputs: List[Point], outputs: List[Point],
                 mapping: List[Tuple[ActiveHinge, int]]):
        self.brain = ANN.build(inputs, outputs, genome)
        self.i_buffer, self.o_buffer = self.brain.buffers()
        self._mapping = mapping

        self._step = 0
        logging.info(f"Created a brain instance for {genome.id()}")

    def reset(self):
        self.brain.reset()
        self._step = 0
    #
    # def __getstate__(self):
    #     logging.info("Pretending to return a state")
    #     return {}
    #
    # def __setstate__(self, state):
    #     logging.info("Pretending to load from a state")
    #     pass

    def control(self, dt: float,
                sensor_state: ModularRobotSensorState,
                control_interface: ModularRobotControlInterface) -> None:
        # off = len(data.sensors)
        # self.i_buffer[:off] = [pos for pos in data.sensors]
        #
        # if self.vision is not None:
        #     if Config.debug_retina_brain > 1:
        #         img = self._debug_retina_image()
        #         self.vision.img = img
        #     else:
        #         img = self.vision.process(data.model, data.data)
        #     self.i_buffer[off:] = [x / 255 for x in img.flat]

        self._step += 1

        self.brain.__call__(self.i_buffer, self.o_buffer)

        for hinge, o_index in self._mapping:
            control_interface.set_active_hinge_target(
                hinge, self.o_buffer[o_index] * hinge.range)

        # pprint.pprint([n.value for n in self.brain.neurons()])


class ABrainFactory(BrainFactory):
    def __init__(self, dna: Genome, body: Body, with_labels=False):
        logging.info(f"Creating a brain factory for {dna.id()}")
        self._dna = dna
        self._labels = {} if with_labels else None
        self._inputs, self._outputs, self._mapping = [], [], []

        hinges = body.find_modules_of_type(ActiveHinge)
        if len(hinges) == 0:
            return

        h_coords = {h: body.grid_position(h) for h in hinges}

        bounds = np.zeros((2, 3), dtype=int)
        np.quantile([c.tolist() for c in h_coords.values()], [0, 1], axis=0, out=bounds)

        # if bounds[0][2] != bounds[1][2]:
        #     raise NotImplementedError("Can only handle planar robots (with z=0 for all modules)")

        x_min, x_max = bounds[0][0], bounds[1][0]
        xrange = max(abs(x_min), abs(x_max), 1)
        y_min, y_max = bounds[0][1], bounds[1][1]
        yrange = max(abs(y_min), abs(y_max), 1)
        z_min, z_max = bounds[0][2], bounds[1][2]
        zrange = max(abs(z_min), abs(z_max), 1)

        hinges_pos = {
            m: (c.x / xrange, c.y / yrange, c.z / zrange)
            for m, c in h_coords.items()
        }

        if len(hinges_pos) != len(set(hinges_pos.values())):
            logging.warning("Duplicate hinge positions detected."
                            " Patching with small variations.")
            _duplicates = {}
            for m, p in hinges_pos.items():
                _duplicates.setdefault(p, [])
                _duplicates[p].append(m)
            _duplicates = {k: v for k, v in _duplicates.items() if len(v) > 1}

            d = 0.001
            def _shift(p, s): return __shift(p[0], s, xrange), __shift(p[1], s, yrange), __shift(p[2], s, zrange)
            def __shift(x, s, r): return max(-r, min(x+s*d, r))
            for p, ms in _duplicates.items():
                assert len(ms) == 2
                hinges_pos[ms[0]] = _shift(p, +1)
                hinges_pos[ms[1]] = _shift(p, -1)
            pprint.pprint(_duplicates)

        for i, (hinge, p) in enumerate(hinges_pos.items()):
            y = .05 * (p[2] + 1)
            ip = Point(p[1], -1 + y, p[0])
            self._inputs.append(ip)
            op = Point(p[1], 1 - y, p[0])
            self._outputs.append(op)
            self._mapping.append((hinge, i))

            if with_labels:
                self._labels[ip] = f"P{i}"
                self._labels[op] = f"M{i}"
        #
        # if self.brain_dna.with_vision():
        #     mapper = retina_mapper()
        #     w, h = self.brain_dna.vision
        #     for j in range(h):
        #         for i in range(w):
        #             for k, c in enumerate("RGB"):
        #                 p = mapper(i, j, k, w, h)
        #                 inputs.append(p)
        #
        #                 if self.with_labels:
        #                     labels[p] = f"{c}[{i},{j}]"

        # Ensure no duplicates
        try:
            # assert all(len(set(io_)) == len(io_) for io_ in [inputs, outputs])
            assert len({p for io in [self._inputs, self._outputs] for p in io}) \
                   == (len(self._inputs) + len(self._outputs))
        except AssertionError as e:
            duplicates = {}
            for io in [self._inputs, self._outputs]:
                for p in io:
                    duplicates[p] = duplicates.get(p, 0) + 1
            duplicates = {k: v for k, v in duplicates.items() if v > 1}
            raise ValueError(f"Found duplicates: {pprint.pformat(duplicates)}") from e

    def make_instance(self) -> ABrainInstance:

        c = ABrainInstance(self._dna,
                           self._inputs, self._outputs,
                           self._mapping)

        if self._labels:
            c.labels = self._labels

        return c


def develop(genome: Genome, body: Body, with_labels=False):
    return ABrainFactory(genome, body, with_labels)
