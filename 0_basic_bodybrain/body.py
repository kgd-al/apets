import abc
import itertools
import math
import pprint
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from random import Random
from typing import Any, List

import numpy as np
from abrain import Genome, CPPN
from numpy.typing import NDArray
from pyrr import Quaternion, Vector3
from revolve2.modular_robot.body import AttachmentPoint, Module
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2, CoreV2
from revolve2.modular_robot.body.v2._attachment_face_core_v2 import AttachmentFaceCoreV2

_DEBUG = True


def vec(*args): return Vector3([*args])


class _Grid:
    def __init__(self):
        self._data = defaultdict(int)

    def query(self, key: Vector3):
        # print(f"query({key}):\n", pprint.pformat({k: v for k, v in self._iter(key)}))
        return any(v > 0 for _, v in self._iter(key))

    def collisions(self, key: Vector3):
        return {
            k: v for k, v in self._iter(key) if v > 0
        }

    def register(self, key: Vector3):
        # print(f"register({key}):\n", pprint.pformat({k: v for k, v in self._iter(key)}))
        for k, _ in self._iter(key):
            self._data[k] += 1

    def _iter(self, key: Vector3):
        g_key = (2 * key).astype(np.int_)
        # print(tuple(key), tuple(g_key))
        for dx, dy, dz in itertools.product(*[[-1, 0]] * 3):
            k = tuple(g_key + Vector3([dx, dy, dz]))
            yield k, self._data[k]

    def __repr__(self):
        return pprint.pformat(self._data)


class __BodyPlan(abc.ABC):
    @dataclass
    class _Module:
        position: Vector3[np.int_]
        rotation: Quaternion
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

        __prefix = f"{'  ' * module.chain_length}"
        # if _DEBUG:
        #     print(__prefix, "----")

        if not module.module_reference.can_set_child(attachment_index):
            print(__prefix, f">> Attachment {attachment_index} unavailable")
            return None

        def _debug_rotation(r: Quaternion):
            return f"({r.axis}:{180*r.angle/math.pi})"

        """Here we adjust the forward facing direction, and the position for
         the new potential module."""
        rotation = attachment_point.orientation * module.rotation
        print(__prefix, ">>> Rotation:", rotation, _debug_rotation(rotation))
        print(__prefix, "            =",
              _debug_rotation(module.rotation), "*",
              _debug_rotation(attachment_point.orientation))
        print(__prefix, ">>> Position:", module.position)
        forward = cls._rotate(rotation, vec(1.0, 0, 0))
        print(__prefix, ">>>  Forward:", forward)

        print(__prefix, ">>>   Offset:", attachment_point.offset)
        offset = cls._vec3_sign(attachment_point.offset)
        print(__prefix, "             ", offset)
        if isinstance(module.module_reference, AttachmentFaceCoreV2):
            offset.x += .5
            offset.y /= 2
            offset.z /= 2
        print(__prefix, "             ", offset)
        offset = cls._rotate(rotation, offset)
        print(__prefix, "             ", offset)

        position = (module.position + offset)
        if _DEBUG and False:
            def _fmt(__v, __w=10): return f"{str(__v):{__w}}"
            print(__prefix, _fmt(module.position), _fmt(forward, 15),
                  _fmt(offset, 20), _fmt(position, 20),
                  attachment_index, module.module_reference.__class__.__name__)
        chain_length = module.chain_length + 1

        """If grid cell is occupied, we don't make a child."""
        if grid.query(position):
            print(__prefix, ">> Collision(s)\n", pprint.pformat(grid.collisions(position)))
            print(__prefix, ">> ===")
            return None

        """Now we adjust the position for the potential new module to fit the
         attachment point of the parent, additionally we query the CPPN for
          child type and angle of the child."""
        child_type, angle = BrickV2, 0.0 #cls._evaluate_cppn(cppn, new_pos, chain_length)
        # return None

        ## DEBUG ##
        if ((isinstance(module.module_reference, AttachmentFaceCoreV2)
                and attachment_index != 0) and
                not isinstance(module.module_reference, BrickV2)):
            return None
        ###########

        """Here we check whether the CPPN evaluated to place a module"""
        if child_type is None:
            return None  # No module will be placed.

        """Now we know we want a child on the parent and we instantiate it, add
         the position to the grid and adjust the up direction for the new
          module."""
        child = child_type(angle)

        grid.register(position)

        module.module_reference.set_child(child, attachment_index)

        if _DEBUG:
            print(__prefix, f"> Added module at {position}")

        return cls._Module(
            position,
            rotation,
            chain_length,
            child
        )

    @staticmethod
    def _vec3_int(vector: Vector3) -> Vector3[np.int_]:
        """
        Cast a Vector3 object to an integer only Vector3.

        :param vector: The vector.
        :return: The integer vector.
        """
        return Vector3(list(map(lambda v: int(round(v)), vector)),
                       dtype=np.int64)

    @staticmethod
    def _vec3_sign(vector: Vector3) -> Vector3[np.int_]:
        def sign(x): return 1 if x > 0 else -1 if x < 0 else 0
        return Vector3(list(map(lambda v: sign(v), vector)),
                       dtype=np.float64)

    @staticmethod
    def _rotate(rotation: Quaternion, vector: Vector3):
        return (rotation * vector).round(1)

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

        rng = Random(0)

        max_parts = 11
        max_depth = 5

        def limit(__n, __d):
            if max_parts is not None:
                return __n >= max_parts
            return __d >= max_depth

        cppn = CPPN(genotype)

        to_explore: List[cls._Module] = []
        grid = _Grid()

        body = BodyV2()

        core_position = Vector3([0, 0, 0], dtype=np.int_)
        for dx, dy, dz in itertools.product(*[[-.5, .5]]*3):
            grid.register(core_position+Vector3([dx, dy, dz]))
        part_count = 1
        # pprint.pprint(grid)

        faces = body.core_v2.attachment_faces.values()
        # for attachment_face in faces:
        for attachment_face in list(faces)[3:4]:
        # for attachment_face in rng.sample(list(faces), k=len(faces)):
            to_explore.append(
                cls._Module(
                    core_position,
                    Quaternion.from_axis_rotation(vec(0, 0, 1), math.pi),
                    # Quaternion(),
                    0,
                    attachment_face
                )
            )
        # pprint.pprint(to_explore)

        print("== Process start ====")
        while len(to_explore) > 0:
            module = to_explore.pop(0)

            attachments = module.module_reference.attachment_points.items()
            for attachment_point_tuple in rng.sample(attachments,
                                                     k=len(attachments)):
            # for attachment_point_tuple in attachments:
                # if _DEBUG:
                #     print(attachment_point_tuple)
                if not limit(part_count, module.chain_length):
                    child = cls._add_child(cppn, module,
                                           attachment_point_tuple, grid)
                    if child is not None:
                        to_explore.append(child)
                        part_count += 1

            print("== Module processed ====")
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

