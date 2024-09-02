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
from revolve2.modular_robot.body import AttachmentPoint, Module, RightAngles
from revolve2.modular_robot.body.base import Body
from revolve2.modular_robot.body.sensors import ActiveHingeSensor
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2, CoreV2
from revolve2.modular_robot.body.v2._attachment_face_core_v2 import AttachmentFaceCoreV2
from revolve2.simulation.scene import _aabb, Pose

_DEBUG = -1


def correctBrickV2Init(brick: BrickV2, rotation: float | RightAngles):
    w, h, d = 0.075, 0.075, 0.075
    super(BrickV2, brick).__init__(
        rotation=rotation,
        bounding_box=Vector3([d, w, h]),
        mass=0.06043,
        child_offset=d / 2.0,
        sensors=[],
    )
    if _DEBUG >= 0:
        print("[kgd-debug] Fixed BrickV2 initialization")


# TODO This is not working well enough:
# - Child is too far apart from parent
# - Dimensions ratios are complete guesses without actual measurement (except the bb)
def correctActiveHingeV2Init(hinge: ActiveHingeV2, rotation: float | RightAngles):
    w, h, d = 0.052, 0.052, 0.074
    frame_offset = 0.04495  # Was 0.04525
    servo_offset = 0.0239   # Was 0.0299
    joint_offset = 0.0119   # not changed
    frame_bounding_box = Vector3([0.018, 0.052, 0.0165891])
    servo1_bounding_box = Vector3([0.05125, 0.0512, 0.020])
    servo2_bounding_box = Vector3([0.002, 0.052, 0.052])
    super(ActiveHingeV2, hinge).__init__(
        rotation=rotation,
        range=1.047197551,
        effort=0.948013269,
        velocity=6.338968228,
        # frame_bounding_box=Vector3([0.018, 0.053, 0.0165891]),
        frame_bounding_box=frame_bounding_box,
        # frame_bounding_box=Vector3([d, w, h]),
        # frame_offset=0.04525,
        frame_offset=frame_offset,
        # servo1_bounding_box=Vector3([0.0583, 0.0512, 0.020]),
        servo1_bounding_box=servo1_bounding_box,
        servo2_bounding_box=servo2_bounding_box,
        frame_mass=0.01632,
        servo1_mass=0.058,
        servo2_mass=0.025,
        servo_offset=servo_offset,
        # servo_offset=0.0274,
        joint_offset=joint_offset,
        static_friction=1.0,
        dynamic_friction=1.0,
        # armature=0.002,
        armature=0.2,
        pid_gain_p=5.0,
        # pid_gain_p=10000.0,
        # pid_gain_d=0.05,
        pid_gain_d=0.1,
        # child_offset=d - frame_offset/2 - servo_offset,
        child_offset=servo1_bounding_box[0]/2 + 0.002 + 0.01,
        sensors=[
            ActiveHingeSensor()
        ],
    )
    hinge.bounding_box = Vector3([d, w, h])
    if _DEBUG >= 0:
        print("[kgd-debug] (Poorly) Fixed ActiveHingeV2 initialization")

    if _DEBUG > 0 and True:
        print("[kgd-debug]", "Testing bounding box == sum(geometries)")

        servo_pos = vec(.5 * (hinge.servo1_bounding_box[0] + hinge.servo2_bounding_box[0]), 0, 0)
        frame_pos = vec(.5 * hinge.frame_offset, 0, 0)
        frame_pos_real = vec(.5 * hinge.frame_bounding_box[0], 0, 0)
        servo_body_pos = frame_pos + vec(hinge.servo_offset, 0, 0)
        joint_pos = frame_pos + vec(hinge.joint_offset, 0, 0)

        def _aabb(pos, extent): return AABB(pos - .5 * extent, pos + .5 * extent)

        frame_geom = _aabb(frame_pos_real, hinge.frame_bounding_box)

        servo1_geom = _aabb(servo_body_pos, hinge.servo1_bounding_box)
        servo2_geom = _aabb(servo_body_pos + servo_pos, hinge.servo2_bounding_box)

        print(f"[kgd-debug]", "="*40)
        total = AABB(vec(0, 0, 0), vec(0, 0, 0))
        for aabb in [frame_geom, servo1_geom, servo2_geom]:
            total.min = vec([min(a, b) for a, b in zip(total.min, aabb.min)])
            total.max = vec([max(a, b) for a, b in zip(total.max, aabb.max)])

        print(f"[kgd-debug]  frame_geom={frame_geom}")
        print(f"[kgd-debug] servo1_geom={servo1_geom}")
        print(f"[kgd-debug] servo2_geom={servo2_geom}")
        print(f"[kgd-debug]       total={total}")
        print(f"[kgd-debug]", "-"*40)
        print(f"[kgd-debug]        aabb={hinge.bounding_box}")
        bb = total.max - total.min
        assert bb == hinge.bounding_box, f"{bb=} != {hinge.bounding_box=}"


def correctCoreV2Init(
        core: CoreV2,
        rotation: float | RightAngles,
        num_batteries: int = 1,
        ):
    mass = (
        num_batteries * core._BATTERY_MASS + core._FRAME_MASS
    )  # adjust if multiple batteries are installed

    core._attachment_faces = {
        core.FRONT: AttachmentFaceCoreV2(
            horizontal_offset=core._horizontal_offset,
            vertical_offset=core._vertical_offset,
            face_rotation=0.0,
        ),
        core.BACK: AttachmentFaceCoreV2(
            horizontal_offset=core._horizontal_offset,
            vertical_offset=core._vertical_offset,
            face_rotation=math.pi,
        ),
        core.RIGHT: AttachmentFaceCoreV2(
            horizontal_offset=core._horizontal_offset,
            vertical_offset=core._vertical_offset,
            face_rotation=-math.pi / 2.0,
        ),
        core.LEFT: AttachmentFaceCoreV2(
            horizontal_offset=core._horizontal_offset,
            vertical_offset=core._vertical_offset,
            face_rotation=math.pi / 2.0,
        ),
    }
    super(CoreV2, core).__init__(
        rotation=rotation,
        mass=mass,
        bounding_box=Vector3([0.15, 0.15, 0.15]),
        child_offset=0.0,
        sensors=[],
    )

    core.front = core.attachment_faces[core.FRONT]
    core.back = core.attachment_faces[core.BACK]
    core.right = core.attachment_faces[core.RIGHT]
    core.left = core.attachment_faces[core.LEFT]

    if _DEBUG >= 0:
        print("[kgd-debug] Fixed CoreV2 initialization")


BrickV2.__init__ = correctBrickV2Init
ActiveHingeV2.__init__ = correctActiveHingeV2Init
CoreV2.__init__ = correctCoreV2Init


def vec(*args): return Vector3([*args])


@dataclass
class AABB:
    min: Vector3
    max: Vector3


class _Grid:
    def __init__(self, scaling: Vector3, debug=False):
        self._data = defaultdict(list)
        self._scaling = scaling
        self._debug_data = defaultdict(list) if debug else None

    def query(self, position: Vector3, rotation: Quaternion, module: Module):
        aabb = self._aabb(position, rotation, module)
        if _DEBUG > 1:
            self.__debug_print("query", position, aabb)
        return any(
            self._collision(aabb, _aabb)
            for _, aabbs in self._iter(aabb)
            for _aabb in aabbs
        )

    def debug_collisions(self, position: Vector3, rotation: Quaternion,
                         module: Module):
        aabb = self._aabb(position, rotation, module)
        dct = {
            k: v for k, v in self._iter(aabb)
            for _aabb in v if self._collision(aabb, _aabb)
        }
        msg = (f"Between {aabb} and\n"
               f"{pprint.pformat(dct)}")
        return msg

    def register(self, position: Vector3, rotation: Quaternion,
                 module: Module, debug_info):
        aabb = self._aabb(position, rotation, module)
        for k, _ in self._iter(aabb):
            self._data[k].append(aabb)
            if self._debug_data is not None:
                self._debug_data[k].append(debug_info)
        if _DEBUG > 1:
            self.__debug_print("register", position, aabb)

    def _aabb(self, position: Vector3, rotation: Quaternion, module: Module):
        mbb = abs(rotation * module.bounding_box) / 2
        aabb = AABB(self._scaled(position - mbb),
                    self._scaled(position + mbb))
        return aabb

    @staticmethod
    def _collision(lhs: AABB, rhs: AABB) -> bool:
        return (lhs.min.x < rhs.max.x and rhs.min.x < lhs.max.x
                and lhs.min.y < rhs.max.y and rhs.min.y < lhs.max.y
                and lhs.min.z < rhs.max.z and rhs.min.z < lhs.max.z)

    def _iter(self, aabb: AABB):
        for x, y, z in itertools.product(
                *[range(math.floor(lower), math.ceil(upper))
                  for lower, upper in zip(aabb.min, aabb.max)]):
            k = (x, y, z)
            yield k, self._data[k]

    def __repr__(self):
        return pprint.pformat(self._data)

    def _scaled(self, v: Vector3):
        return Vector3([round(a * b, 6) for a, b in zip(v, self._scaling)])
        # return Vector3([a * b for a, b in zip(v, self._scaling)])

    def __debug_print(self, name, position, aabb):
        print(f"{name}({position}:\n {aabb=}\n",
              pprint.pformat({k: {_k: _v for _k, _v in
                                  zip(self._debug_data[k], self._data[k])}
                              for k, v in self._iter(aabb)}))


class __BodyPlan(abc.ABC):
    @dataclass
    class _Module:
        position: Vector3[np.int_]
        rotation: Quaternion
        chain_length: int
        module_reference: Module
        id: int

    @classmethod
    def _add_child(
            cls,
            cppn: CPPN,
            module: _Module,
            attachment_point_tuple: tuple[int, AttachmentPoint],
            grid: NDArray[np.uint8],
            _id: int
    ) -> _Module | None:
        attachment_index, attachment_point = attachment_point_tuple

        __prefix = f"{'  ' * module.chain_length}"
        # if _DEBUG:
        #     print(__prefix, "----")

        if not module.module_reference.can_set_child(attachment_index):
            if _DEBUG > 0:
                print(__prefix, f">> Attachment {attachment_index} unavailable")
            return None

        def _debug_rotation(r: Quaternion):
            return f"({r.axis}:{180*r.angle/math.pi})"

        """Here we adjust the forward facing direction, and the position for
         the new potential module."""
        rotation = module.rotation * attachment_point.orientation
        if _DEBUG > 0:
            print(__prefix, ">>>   Attach:", f"{module.id}[{attachment_index}]")
            print(__prefix, ">>> Rotation:", rotation, _debug_rotation(rotation))
            print(__prefix, "            =",
                  _debug_rotation(module.rotation), "*",
                  _debug_rotation(attachment_point.orientation))
            forward = rotation * vec(1.0, 0, 0)
            print(__prefix, ">>>  Forward:", forward)

            print(__prefix, ">>>   Offset:", attachment_point.offset)
        offset = attachment_point.offset
        offset = rotation * offset
        if _DEBUG > 0:
            print(__prefix, "             ", offset)

            print(__prefix, ">>> Position:", module.position)
        position = (module.position + offset)
        if _DEBUG > 0:
            print(__prefix, "             ", module.position, "+", offset)
            print(__prefix, "             ", position)

        chain_length = module.chain_length + 1

        """Now we adjust the position for the potential new module to fit the
         attachment point of the parent, additionally we query the CPPN for
          child type and angle of the child."""
        child_type, angle = cls._evaluate_cppn(cppn, position, chain_length)
        # child_type, angle = BrickV2, 0.0
        # child_type, angle = ActiveHingeV2, 0.0
        # child_type, angle = [BrickV2, ActiveHingeV2][_id%2], 0.0
        # return None

        if _DEBUG > 0:
            print(__prefix, ">>>     CPPN:",
                  child_type.__name__,
                  f"{angle/math.pi:g}PI")

        ## DEBUG ##
        # if ((isinstance(module.module_reference, AttachmentFaceCoreV2)
        #         and attachment_index != 0) and
        #         not isinstance(module.module_reference, BrickV2)):
        #     print("< Debug reject")
        #     return None
        ###########

        """Here we check whether the CPPN evaluated to place a module"""
        if child_type is None:
            if _DEBUG > 0:
                print(__prefix, "< No child reject")
            return None  # No module will be placed.

        """Now we know we want a child on the parent and we instantiate it, add
         the position to the grid and adjust the up direction for the new
          module."""
        child = child_type(angle)

        """ Update the rotation according to cppn output """
        if _DEBUG > 0:
            print(__prefix, ">>> Rotation:",
                  _debug_rotation(rotation), "*",
                  _debug_rotation(child.orientation))
        rotation = rotation * child.orientation
        if _DEBUG > 0:
            print(__prefix, "            =", rotation, _debug_rotation(rotation))

        """ Shift along the main axis to get to the center """
        center_offset = rotation * vec(.5 * child.bounding_box.x, 0, 0)
        if _DEBUG > 0:
            print(__prefix, ">>> Position:", position, "+", center_offset)
        position += center_offset
        if _DEBUG > 0:
            print(__prefix, ">>>          ", position)

        """Now we have the child, check if its AABB fits in the current grid"""
        if grid.query(position, rotation, child):
            if _DEBUG > 0:
                print(__prefix, ">> Collision(s)\n",
                      grid.debug_collisions(position, rotation, child))
                print(__prefix, "< Collision reject")
            return None

        grid.register(position, rotation, child, _id)

        module.module_reference.set_child(child, attachment_index)

        if _DEBUG > 0:
            print(__prefix,
                  f"> Added {child.uuid} at {(100 * position).round(1)} cm")

        return cls._Module(
            position,
            rotation,
            chain_length,
            child,
            -1
        )

    @classmethod
    @abc.abstractmethod
    def develop(cls, genotype: Genome) -> BodyV2:
        pass

    @classmethod
    @abc.abstractmethod
    def _evaluate_cppn(cls, cppn: CPPN, *args, **kwargs):
        pass


def torso_body() -> BodyV2:
    print("Forcing torso body")
    body = BodyV2()
    body.core_v2.left_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.left_face.bottom.attachment = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.left_face.bottom.attachment.attachment = BrickV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom.attachment.attachment = BrickV2(RightAngles.DEG_0)
    return body


def gecko_body() -> BodyV2:
    def add_arms(m: Module):
        m.left = ActiveHingeV2(math.pi / 2.0)
        m.left.attachment = ActiveHingeV2(math.pi / 2.0)
        m.left.attachment.attachment = BrickV2(0.0)
        m.right = ActiveHingeV2(math.pi / 2.0)
        m.right.attachment = ActiveHingeV2(math.pi / 2.0)
        m.right.attachment.attachment = BrickV2(0.0)

    body = BodyV2()
    body.core.front.bottom = BrickV2(0.0)
    add_arms(body.core.front.bottom)
    body.core.back.bottom = BrickV2(0.0)
    add_arms(body.core.back.bottom)

    return body


class DefaultBodyPlan(__BodyPlan):
    @classmethod
    def develop(cls, genotype: Genome) -> BodyV2:
        # return gecko_body()
        # return torso_body()

        assert genotype.inputs - genotype.bias == 4, f"{genotype.inputs} != 4"
        assert genotype.outputs == 2, f"{genotype.outputs} != 2"

        persistent_rng = Random(0)

        # def rng(): return Random(persistent_rng.random())
        def rng(): return Random(0)

        max_parts = None
        max_depth = 3

        def limit(__n, __d):
            if max_parts is not None:
                return __n >= max_parts
            return __d >= max_depth

        cppn = CPPN(genotype)

        to_explore: List[cls._Module] = []

        body = BodyV2()
        grid = _Grid(2 / body.core_v2.bounding_box, debug=True)

        core_position = Vector3([0, 0, 0], dtype=np.int_)
        grid.register(core_position, Quaternion(), body.core_v2, 0)
        part_count = 1
        # pprint.pprint(grid)

        faces = body.core_v2.attachment_faces.values()
        # for attachment_face in faces:
        # for attachment_face in list(faces)[2:3]:
        for attachment_face in rng().sample(list(faces), k=len(faces)):
            to_explore.append(
                cls._Module(
                    core_position,
                    # Quaternion.from_eulers([0, 0, -math.pi]),
                    Quaternion(),
                    0,
                    attachment_face,
                    0
                )
            )
        # pprint.pprint(to_explore)

        if _DEBUG >= 0:
            print("== Process start ====")
        while len(to_explore) > 0:
            module = to_explore.pop(0)

            attachments = module.module_reference.attachment_points.items()
            for attachment_point_tuple in rng().sample(attachments,
                                                       k=len(attachments)):
            # for attachment_point_tuple in attachments:
                # if _DEBUG:
                #     print(attachment_point_tuple)
                if not limit(part_count, module.chain_length):
                    if _DEBUG > 0:
                        print("> Module", part_count)
                    child = cls._add_child(cppn, module,
                                           attachment_point_tuple, grid,
                                           part_count)
                    if child is not None:
                        if _DEBUG > 0:
                            print("< Module", part_count, "added")
                        child.id = part_count
                        to_explore.append(child)
                        part_count += 1

            # print("== Module processed ====")
        return body

    @classmethod
    def _evaluate_cppn(
            cls,
            cppn: CPPN,
            position: Vector3,
            chain_length: int,
    ) -> tuple[Any, float]:
        x, y, z = position
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


def compute_positions(body: Body):
    positions: dict[Module, Vector3] = {body.core: vec(0, 0, 0)}

    def process(m: Module, p: Vector3, q: Quaternion, depth):
        for i, child in m.children.items():
            a = m.attachment_points[i]

            rotation = q * a.orientation
            offset = rotation * a.offset
            position = (p + offset +
                        rotation * vec(.5 * child.bounding_box.x, 0, 0))
            positions[child] = position

            process(child, position, rotation, depth+1)

    if isinstance(body.core, CoreV2):
        core: CoreV2 = body.core
        for face in core.attachment_faces.values():
            process(face, vec(0, 0, 0), Quaternion(), 0)
    else:
        process(body.core, vec(0, 0, 0), Quaternion(), 0)

    # pprint.pprint({m.uuid: p for m, p in positions.items()})

    return positions
