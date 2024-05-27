import abc
import pprint
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from typing import Any, List

import numpy as np
from abrain import Genome, CPPN
from numpy.typing import NDArray
from pyrr import Quaternion, Vector3
from revolve2.modular_robot.body import AttachmentPoint, Module
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2


_DEBUG = True


class __BodyPlan(abc.ABC):
    @dataclass
    class _Module:
        position: Vector3[np.int_]
        forward: Vector3[np.int_]
        up: Vector3[np.int_]
        chain_length: int
        module_reference: Module

    @classmethod
    def _add_child(
            cls,
            cppn: CPPN,
            module: _Module,
            attachment_point_tuple: tuple[int, AttachmentPoint],
            grid: NDArray[np.uint8],
    ) -> _Module | None:
        attachment_index, attachment_point = attachment_point_tuple

        """Here we adjust the forward facing direction, and the position for
         the new potential module."""
        # forward = cls.__rotate(module.forward, module.up,
        #                        attachment_point.orientation)
        forward = attachment_point.offset
        if _DEBUG:
            print(module.position, forward)
        position = (module.position + forward)
        chain_length = module.chain_length + 1

        """If grid cell is occupied, we don't make a child."""
        if _DEBUG:
            print(f"grid[{tuple(position)}]={grid[tuple(position)]}")
        if grid[tuple(position)] > 0:
            return None

        """Now we adjust the position for the potential new module to fit the
         attachment point of the parent, additionally we query the CPPN for
          child type and angle of the child."""
        new_pos = np.array(np.round(position + attachment_point.offset),
                           dtype=np.int64)
        child_type, angle = cls._evaluate_cppn(cppn, new_pos, chain_length)

        """Here we check whether the CPPN evaluated to place a module and if
         the module can be set on the parent."""
        can_set = module.module_reference.can_set_child(attachment_index)
        if (child_type is None) or (not can_set):
            return None  # No module will be placed.

        """Now we know we want a child on the parent and we instantiate it, add
         the position to the grid and adjust the up direction for the new
          module."""
        child = child_type(angle)
        grid[tuple(position)] += 1
        up = cls.__rotate(module.up, forward,
                          Quaternion.from_eulers([angle, 0, 0]))
        module.module_reference.set_child(child, attachment_index)

        return cls._Module(
            position,
            forward,
            up,
            chain_length,
            child,
        )

    @staticmethod
    def __rotate(a: Vector3, b: Vector3, rotation: Quaternion) -> Vector3:
        """
        Rotates vector a, a given angle around b.

        :param a: Vector a
        :param b: Vector b.
        :param rotation: The quaternion for rotation.
        :returns: A rotated copy of `a`.
        """
        cos_angle: int = int(round(np.cos(rotation.angle)))
        sin_angle: int = int(round(np.sin(rotation.angle)))

        vec: Vector3 = (
                a * cos_angle
                + sin_angle * b.cross(a)
                + (1 - cos_angle) * b.dot(a) * b
        )
        return vec

    @staticmethod
    def __vec3_int(vector: Vector3) -> Vector3[np.int_]:
        """
        Cast a Vector3 object to an integer only Vector3.

        :param vector: The vector.
        :return: The integer vector.
        """
        return Vector3(list(map(lambda v: int(round(v)), vector)),
                       dtype=np.int64)

    @classmethod
    @abc.abstractmethod
    def develop(cls, genotype: Genome) -> BodyV2:
        pass

    @classmethod
    @abc.abstractmethod
    def _evaluate_cppn(cls, cppn: CPPN, *args, **kwargs):
        pass


class DefaultBodyPlan(__BodyPlan):
    @classmethod
    def develop(cls, genotype: Genome) -> BodyV2:
        assert genotype.inputs - genotype.bias == 4, f"{genotype.inputs} != 4"
        assert genotype.outputs == 2, f"{genotype.outputs} != 2"

        max_parts = 20  # Determine the maximum parts available for a robots body.
        cppn = CPPN(genotype)

        to_explore: List[cls._Module] = []
        grid = defaultdict(int)

        body = BodyV2()

        core_position = Vector3([0, 0, 0], dtype=np.int_)
        grid[tuple(core_position)] = 1
        part_count = 1

        for attachment_face in body.core_v2.attachment_faces.values():
            to_explore.append(
                cls._Module(
                    core_position,
                    attachment_face._child_offset,
                    Vector3([0, 0, 1]),
                    0,
                    attachment_face,
                )
            )
        # pprint.pprint(to_explore)

        while len(to_explore) > 0:
            module = to_explore.pop(0)

            for attachment_point_tuple \
                    in module.module_reference.attachment_points.items():
                if _DEBUG:
                    print(attachment_point_tuple)
                if part_count < max_parts:
                    child = cls._add_child(cppn, module,
                                           attachment_point_tuple, grid)
                    if child is not None:
                        to_explore.append(child)
                        part_count += 1
        return body

    @classmethod
    def _evaluate_cppn(
            cls,
            cppn: CPPN,
            position: Vector3[np.int_],
            chain_length: int,
    ) -> tuple[Any, float]:
        x, y, z = position
        assert isinstance(
            x, np.int_
        ), f"Error: The position is not of type int. Type: {type(x)}."
        outputs = cppn.obuffer()
        cppn(outputs, x, y, z, chain_length)

        """We select the module type for the current position using the first output of the CPPN network."""
        types = [None, BrickV2, ActiveHingeV2]
        target_idx = max(0, int(outputs[0] * len(types) - 1e-6))
        module_type = types[target_idx]

        """
        Here we get the rotation of the module from the second output of the CPPN network.
    
        The output ranges between [0,1] and we have 4 rotations available (0, 90, 180, 270).
        """
        angle = max(0, int(outputs[0] * 4 - 1e-6)) * (np.pi / 2.0)

        return module_type, angle

