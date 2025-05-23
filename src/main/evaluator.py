"""Evaluator class."""
import json
import logging
import math
import numbers
import pprint
import time
from dataclasses import dataclass
from enum import auto
from pathlib import Path
from typing import Optional, Annotated, ClassVar, Dict, Callable, List

import cv2
import glfw
import mujoco
import numpy as np
import yaml
from colorama import Style, Fore
from mujoco import MjModel, MjData
from mujoco_viewer import MujocoViewer
from pyrr import Vector3, Quaternion
from strenum import StrEnum

from abrain import ANN3D
from abrain.core.ann import ANNMonitor
from abrain.neat.config import ConfigBase
from abrain.neat.evolver import EvaluationResult
from config import Config, ExperimentType, EXPERIMENT_DURATIONS
from genotype import Genotype
from brain import ABrainInstance
from revolve2.experimentation.evolution.abstract_elements import Evaluator as Eval
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BrickV2
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    simulate_scenes, Terrain,
)
from revolve2.modular_robot_simulation._modular_robot_simulation_handler import ModularRobotSimulationHandler
from revolve2.simulation.scene import Pose, Color, UUIDKey, AABB
from revolve2.simulation.scene.geometry import GeometryBox, GeometryPlane
from revolve2.simulation.scene.geometry.textures import Texture, MapType
from revolve2.simulation.scene.vector2 import Vector2
from revolve2.simulation.simulator import RecordSettings
from revolve2.simulation.simulator._simulator import Callback
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.simulators.mujoco_simulator._abstraction_to_mujoco_mapping import AbstractionToMujocoMapping
from revolve2.simulators.mujoco_simulator.textures import Flat, Checker
from revolve2.simulators.mujoco_simulator.viewers import CustomMujocoViewer
from revolve2.simulators.mujoco_simulator.viewers._custom_mujoco_viewer import _MujocoViewerBackend
from revolve2.standards.interactive_objects import Ball
from revolve2.standards.simulation_parameters import make_standard_batch_parameters


@dataclass
class Options(ConfigBase):
    rerun: Annotated[bool, "Is this a rerun or are we evolving"] = False
    headless: Annotated[bool, "Do you think you need a GUI?"] = False
    movie: Annotated[bool, "Do you want a nice video of the robot?"] = False
    start_paused: Annotated[bool, "Should we let you a good look at the robot first?"] = False

    file: Annotated[Optional[Path], "Path to the genome under re-evaluation."] = None
    duration: Annotated[Optional[int], ("Simulation duration in seconds."
                                        " Overwrites the value used in evolution")] = None

    ann_dynamics: Annotated[bool, "Whether to record precise dynamics of the neural network"] = False


def vec(x, y, z): return Vector3([x, y, z], dtype=float)


def make_custom_terrain() -> Terrain:
    # terrain = terrains.flat(
    #     size=Vector2([8, 8]),
    #     texture=Texture(
    #         # base_color=Color(128, 128, 196, 255),
    #         reference=TextureReference(
    #             builtin="checker"
    #         ),
    #         base_color=Color(255, 255, 255, 255),
    #         primary_color=Color(128, 128, 160, 255),
    #         secondary_color=Color(64, 64, 80, 255),
    #         repeat=(2, 2),
    #         size=(1024, 1024),
    #         map_type=MapType.MAP2D
    #     ))
    s = 8
    return Terrain(
        static_geometry=[
            GeometryPlane(
                pose=Pose(position=Vector3(), orientation=Quaternion()),
                mass=0.0,
                size=Vector3([2*s, 2*s, 0]),
                texture=Checker(
                    primary_color=Color(128, 128, 160, 255),
                    secondary_color=Color(64, 64, 80, 255),
                    map_type=MapType.MAP2D,
                    repeat=(2, 2),
                    size=(1024, 1024)
                ),
            )
        ] + [
            GeometryBox(
                pose=Pose(position=Vector3([x*s, y*s, .5]),
                          orientation=Quaternion.from_z_rotation(math.pi*r)),
                mass=0.0,
                texture=Flat(primary_color=Color(96, 96, 120, 255)),
                aabb=AABB(size=Vector3([.1, 2*s - .1, 1])),
            )
            for x, y, r in [(1, 0, 0), (0, 1, .5), (-1, 0, 1), (0, -1, 1.5)]
        ]
    )


X_OFFSET = .25
Y_OFFSET = .25


def robot_position(config: Config):
    if config.experiment in [ExperimentType.FOLLOW, ExperimentType.PUNCH_FOLLOW]:
        p = vec(0, 0, 0)
    else:
        if config.centered_ball:
            p = vec(0, 0, 0)
        else:
            p = vec(-X_OFFSET, Y_OFFSET, 0)
        if config.experiment is ExperimentType.PUNCH_TOGETHER:
            p.x -= 1
    return p


def ball_position(config: Config, r: float):
    exp = config.experiment
    x = 0
    if exp in [ExperimentType.PUNCH_ONCE, ExperimentType.PUNCH_AHEAD,
               ExperimentType.PUNCH_THRICE, ExperimentType.PUNCH_TOGETHER]:
        x = 2 * X_OFFSET if config.centered_ball else 0
    elif exp in [ExperimentType.PUNCH_BACK]:
        x = 2
    if exp is ExperimentType.PUNCH_TOGETHER:
        x -= 1
    return vec(x, 0, r)


def camera_position(exp: ExperimentType):
    pos = [0, 0, 1]
    if exp in [ExperimentType.LOCOMOTION, ExperimentType.FOLLOW, ExperimentType.PUNCH_FOLLOW,
               ExperimentType.PUNCH_ONCE, ExperimentType.PUNCH_AHEAD]:
        pos[0] = -1
    elif exp in [ExperimentType.PUNCH_BACK, ExperimentType.PUNCH_THRICE]:
        pos[0] = -2
    elif exp in [ExperimentType.PUNCH_TOGETHER]:
        pos[1] = 3
    return pos


class FitnessData:
    def __init__(self, robots, objects, **kwargs):
        self.robots = robots
        self.objects = objects
        self.states = None

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def _2d_pos(data: MjData, obj_id: int):
        return Vector2(data.xpos[obj_id][:2].copy())

    @staticmethod
    def _2d_vel(data: MjData, obj_id: int):
        return Vector2(data.cvel[obj_id][3:5].copy())


class BackAndForthFitnessData(FitnessData):
    # __magic_bullet_ball_pos = slice(-7, -5)
    __magic_bullet_ball_vel = slice(-6, -4)

    velocity = 10.
    target_distance = 2

    __debug = False

    def __init__(self, robots, objects, render=False):
        super().__init__(robots, objects)

        self.mapping: Optional[AbstractionToMujocoMapping] = None
        self.ball_id = None
        self.robots_ix = None

        self.state = 1 if len(robots) == 1 else -1
        self.last_contact = "mbs1"
        self.exchanges = 0
        self.pos, self.vel = None, None
        self.fitness = 0

        self.render = render
        if render:
            self.p = None
            self.u, self.v = None, None

    def ball_pos_and_vel(self, data: MjData):
        bid = self.ball_id
        return self._2d_pos(data, bid), self._2d_vel(data, bid)

    def set_ball_velocity(self, data: MjData, vel: Vector2):
        data.qvel[self.__magic_bullet_ball_vel] = vel

    def robots_pos(self, data: MjData):
        robot0 = self._2d_pos(data, self.robots_ix[0])
        robot1 = (
            self._2d_pos(data, self.robots_ix[1])
            if len(self.robots) > 1 else
            Vector2([self.target_distance, 0])
        )
        return robot0, robot1

    def start(self, model: MjModel, data: MjData,
              mapping: AbstractionToMujocoMapping,
              handler: ModularRobotSimulationHandler):
        self.mapping = mapping
        self.ball_id = mapping.multi_body_system[UUIDKey(self.objects[0])].id
        # Ugly but working (with two(?) robots)
        i = 2
        self.robots_ix = []
        for r in self.robots:
            self.robots_ix.append(i)
            i += len(r.body.find_modules_of_type(ActiveHingeV2)) + 1

    def before_step(self, model: MjModel, data: MjData):
        self.pos, self.vel = self.ball_pos_and_vel(data)
        # print("Before step:", self.pos, self.vel)

    def after_step(self, model: MjModel, data: MjData):
        pos, vel = self.ball_pos_and_vel(data)
        delta_pos, delta_vel = [
            end - start for end, start in zip([pos, vel], [self.pos, self.vel])
        ]
        # print("After step:", pos, vel)
        # print("> Delta", delta_pos, delta_vel)
        old_state = self.state
        new_vel = vel

        # Compute instant fitness
        p0, p1 = self.robots_pos(data)
        b = self._2d_pos(data, self.ball_id)
        if self.state > 0:  # Ball goes towards second robot / fixed point
            x, y = (p1 - b).normalized
        else:  # Ball goes towards first robot
            x, y = (p0 - b).normalized
        u, v = Vector2([x, y]), Vector2([-y, x])
        self.fitness += self.exchanges * delta_pos.dot(u) - abs(delta_pos.dot(v))

        # Check if we changed state (and if we need to punch the ball)
        if len(self.robots) == 2:
            contacts = set()
            for c in data.contact:
                name1, name2 = data.geom(c.geom1).name, data.geom(c.geom2).name
                if ("mbs" in name1 and "mbs" in name2
                        and (parent1 := name1.split("/")[0]) != (parent2 := name2.split("/")[0])):
                    contacts.add(min(int(_p) for _p in [parent1[-1], parent2[-1]]))
            if len(contacts) > 0 and (new_contact := next(iter(contacts))) != self.last_contact:
                self.state *= -1
                self.last_contact = new_contact
                # print(contacts)

        elif len(self.robots) == 1:
            if pos.x >= self.target_distance and vel.x > 0:
                # new_vel = -vel
                # new_vel = vel.length * (p0 - b).normalized
                new_vel = - max(2., vel.length) * vel.normalized
                self.state = -1
            else:
                changed_direction = (self.vel.x * vel.x < 0)
                if self.__debug and pos.x <= 0 and vel.x <= 0:
                    new_vel = self.velocity * u
                    self.state = 1
                elif not self.__debug and vel.x > 0 and changed_direction:
                    self.state = 1

        if self.state != old_state:
            self.exchanges += 1
            # print(">> New state:", self.state)

        if new_vel != vel:
            self.set_ball_velocity(data, new_vel)
            # print(">> New vel:", new_vel)

        if self.render:
            self.u, self.v = u, v

        self.pos, self.vel = pos, vel

    def pre_render(self, model: MjModel, data: MjData, viewer: CustomMujocoViewer):
        p3d = Vector3([*self.pos, 0])
        mat = Quaternion.from_y_rotation(math.pi/2).matrix33
        u, v = self.u, self.v
        if u is not None:
            mat *= Quaternion.from_z_rotation(theta=math.atan2(u[1], u[0]))
        args = dict(
            pos=p3d,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[.005, .005, 1],
            mat=mat,
            rgba=[0, 1, 0, .5]
        )
        viewer._viewer_backend.add_marker(**args)
        mat = mat * Quaternion.from_z_rotation(math.pi/2)
        args["mat"] = mat
        args["rgba"] = [1, 0, 0, .5]
        viewer._viewer_backend.add_marker(**args)


class FollowFitnessData(FitnessData):
    def __init__(self, robots, objects, **kwargs):
        super().__init__(robots, objects, **kwargs)


class PunchFollowFitnessData(FitnessData):
    def __init__(self, robots, objects, **kwargs):
        super().__init__(robots, objects, **kwargs)


class DynamicsMonitor:
    def __init__(self, path: Path, dt: float):
        self.path, self.dt = path, dt
        self.monitors: List[ANNMonitor] = []

    def start(self, model: MjModel, data: MjData,
              mapping: AbstractionToMujocoMapping,
              handler: ModularRobotSimulationHandler):

        def path(_i):
            p = self.path.with_suffix("")
            if len(handler._brains) > 1:
                p = p.joinpath(f"brain_{_i}")
            return p

        self.monitors = [
            ANNMonitor(ann=b.brain, labels=b.labels, folder=path(i), dt=self.dt)
            for i, (b, _) in enumerate(handler._brains)
        ]

    def step(self, model: MjModel, data: MjData):
        for monitor in self.monitors:
            monitor.step()

    def end(self, model: MjModel, data: MjData):
        self.step(model, data)
        for monitor in self.monitors:
            monitor.close()


class MultiCameraOverlay:
    class Mode(StrEnum):  # Inset configuration
        NONE = auto()
        PRETTY_RGB = auto()  # rgb coloring and comfortable size
        RGB = auto()         # rgb coloring and accurate size
        PRETTY = auto()      # color-mapped vision and comfortable size
        ACCURATE = auto()    # color-mapped vision and accurate size (WYSWYG)

    PRETTY_CAMERA_WIDTH_RATIO = .25

    def __init__(self, vision, fd_cm, inv_cm):
        self.viewer = None
        self.cameras = None
        self.vision = vision
        self.mode = self.Mode.ACCURATE

        self.vopt, self.scene, self.ctx = None, None, None

        self.camera_buffer = np.zeros(shape=(vision[1], vision[0], 3), dtype=np.uint8)
        self.window_buffer = np.zeros(0)

        self.fd_cm, self.inv_cm = fd_cm, inv_cm

    def start(self, model: MjModel, data: MjData, viewer: CustomMujocoViewer):
        self.viewer = viewer
        self.cameras = []

        for name_index in model.name_camadr:
            terminator = model.names.find(b'\x00', name_index)
            name = model.names[name_index:terminator].decode("ascii")
            if "mbs" in name:
                camera = mujoco.MjvCamera()
                camera.fixedcamid = model.camera(name).id
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.cameras.append(camera)

        if len(self.cameras) > 0:
            # create options, camera, scene, context
            self.vopt = mujoco.MjvOption()
            self.scene = mujoco.MjvScene(model, maxgeom=10000)

            self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

            viewer._viewer_backend.add_callback(
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                "Camera inset",
                glfw.KEY_PAGE_UP,
                lambda: self.mode.name.capitalize().replace("_", " "),
                self.next_mode
            )

            viewer._viewer_backend.add_callback(None, None, glfw.KEY_PAGE_DOWN, None, self.prev_mode)

    def next_mode(self, step=1):
        modes = list(self.Mode)
        index = (modes.index(self.mode) + step) % len(modes)
        print(f"{self.mode} -> {modes}[{index}]")
        self.mode = modes[index]
        print(f">>", self.mode)

    def prev_mode(self):
        self.next_mode(-1)

    def process(self, model: MjModel, data: MjData, viewer: _MujocoViewerBackend):
        if not viewer.is_alive or self.mode is self.Mode.NONE:
            return

        start_0 = time.time()

        vw, vh = self.vision
        width, height = viewer.viewport.width, viewer.viewport.height
        for camera in self.cameras:
            inset_width = int(width * self.PRETTY_CAMERA_WIDTH_RATIO)
            inset_height = int(inset_width * vh / vw)
            if self.mode in [self.Mode.PRETTY, self.Mode.PRETTY_RGB]:
                camera_width, camera_height = inset_width, inset_height
            else:
                camera_width, camera_height = self.vision

            mujoco.mjv_updateScene(model, data, self.vopt, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)

            viewport_inset = mujoco.MjrRect(width - inset_width, height - inset_height, inset_width, inset_height)

            mujoco.mjr_rectangle(mujoco.MjrRect(viewport_inset.left - 1, viewport_inset.bottom - 1,
                                                viewport_inset.width + 2, viewport_inset.height + 2),
                                 1, 0, 0, 1)

            if self.mode is self.Mode.PRETTY_RGB:
                viewport_render = mujoco.MjrRect(width - camera_width, height - camera_height,
                                                 camera_width, camera_height)
                mujoco.mjr_render(viewport_render, self.scene, self.ctx)

            else:
                viewport_render = mujoco.MjrRect(0, 0, camera_width, camera_height)
                # View has been rendered, now grab and tweak it

                rescale = (self.mode in [self.Mode.RGB, self.Mode.ACCURATE])

                if rescale:
                    buffer = self.camera_buffer
                else:
                    size = inset_width * inset_height * 3
                    if len(self.window_buffer) != size:
                        self.window_buffer = np.ones(shape=(inset_height, inset_width, 3), dtype=np.uint8)
                    buffer = self.window_buffer

                start_rn = time.time()
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
                mujoco.mjr_render(viewport_render, self.scene, self.ctx)
                mujoco.mjr_readPixels(buffer, None, viewport_render, self.ctx)
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
                print(" grab:", time.time() - start_rn)

                if self.mode in [self.Mode.PRETTY, self.Mode.ACCURATE]:
                    start_cm = time.time()
                    buffer = self.inv_cm(buffer)
                    print(" cmap:", time.time() - start_cm)

                if rescale:
                    start_rs = time.time()
                    buffer = cv2.resize(buffer, (inset_width, inset_height),
                                        interpolation=cv2.INTER_NEAREST)
                    print("scale:", time.time() - start_rs)

                start_dp = time.time()
                mujoco.mjr_drawPixels(rgb=buffer.flatten(), depth=None, viewport=viewport_inset, con=self.ctx)
                print(" draw:", time.time() - start_dp)

        print(f"[{self.mode}] Rendering in", time.time() - start_0)


class PersistentViewerOptions:
    @classmethod
    def persistent_storage(cls, viewer):
        path = cls.backend(viewer).CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def backend(cls, viewer) -> MujocoViewer:
        backend = viewer._viewer_backend
        assert isinstance(backend, MujocoViewer)
        return backend

    @classmethod
    def window(cls, viewer):
        return cls.backend(viewer).window

    @classmethod
    def start(cls, model: MjModel, data: MjData, viewer: CustomMujocoViewer):
        backend = cls.backend(viewer)

        backend.original_close = backend.close
        def monkey_patch():
            cls.end(viewer)
            backend.original_close()
        backend.close = monkey_patch

        try:
            with open(cls.persistent_storage(viewer), "r") as f:
                if (config := yaml.safe_load(f)) is not None:
                    # print("[kgd-debug] restored viewer config: ", config)
                    window = PersistentViewerOptions.window(viewer)
                    glfw.restore_window(window)
                    glfw.set_window_pos(window, *config["pos"])
                    glfw.set_window_size(window, *config["size"])

        except FileNotFoundError:
            pass

    @classmethod
    def end(cls, viewer: CustomMujocoViewer):
        if not cls.backend(viewer).is_alive:
            return
        with open(cls.persistent_storage(viewer), "w") as f:
            window = cls.window(viewer)
            yaml.safe_dump(
                dict(
                    pos=glfw.get_window_pos(window),
                    size=glfw.get_window_size(window),
                ),
                f
            )


class Evaluator(Eval):
    """Provides evaluation of robots."""

    config: ClassVar[Config] = None
    options: ClassVar[Options] = None

    _log: ClassVar[logging.Logger] = None

    @classmethod
    def initialize(cls, config: Config, options: Optional[Options] = None,
                   verbose=True):
        cls.config = config

        options = options or Options(headless=True)
        cls.options = options

        if config.simulation_duration is None:
            if options.rerun and options.duration is not None:
                config.simulation_duration = options.duration
            else:
                config.simulation_duration = EXPERIMENT_DURATIONS[config.experiment]

        cls._data_folder = config.data_root

        cls._log = getattr(config, "logger", None)
        if cls._log and verbose:
            cls._log.info(f"Configuration:\n"
                          f"{pprint.pformat(cls.config)}\n"
                          f"{pprint.pformat(cls.options)}")

        cls._fitness = getattr(cls, f"fitness_{config.experiment.lower()}")

    @classmethod
    def evaluate(cls, genotype: Genotype) -> EvaluationResult:
        """
        Evaluate a *single* robot.

        Fitness is the distance traveled on the xy plane.

        :param genotype: The genotype to develop into a robot and then simulate.
        :returns: Fitness of the robot.
        """

        config, options = cls.config, cls.options

        batch_parameters = make_standard_batch_parameters(
                simulation_time=options.duration or config.simulation_duration)

        simulator = LocalSimulator(
            headless=options.headless,
            num_simulators=1,
            start_paused=options.start_paused,
            viewer_type="custom"
        )

        terrain = make_custom_terrain()

        ann_labels = options.ann_dynamics or options.rerun

        robots = [genotype.develop(config, with_labels=ann_labels, _id=i)
                  for i in range(1 + (config.experiment is ExperimentType.PUNCH_TOGETHER))]

        if options.ann_dynamics:
            dynamics_monitor = DynamicsMonitor(options.file, 1/batch_parameters.control_frequency)
            simulator.register_callback(Callback.START, dynamics_monitor.start)
            simulator.register_callback(Callback.POST_CONTROL, dynamics_monitor.step)
            simulator.register_callback(Callback.END, dynamics_monitor.end)

        if not options.headless:
            simulator.register_callback(Callback.RENDER_START, PersistentViewerOptions.start)
            if config.vision is not None:
                multiview = MultiCameraOverlay(config.vision, ABrainInstance.forward_colormap, ABrainInstance.inverse_colormap)
                simulator.register_callback(Callback.RENDER_START, multiview.start)
                simulator.register_callback(Callback.POST_RENDER, multiview.process)

        if options.rerun:
            def write_brain(model, data, mapping, handler):
                brain: ANN3D = handler._brains[0][0].brain
                if not brain.empty():
                    brain.render3D().write_html(config.data_root.joinpath("brain.html"))
            simulator.register_callback(Callback.START, write_brain)

        # Create the scenes.
        scene = ModularRobotScene(terrain=terrain)
        for i, robot in enumerate(robots):
            pose = Pose(position=robot_position(config))
            if i > 0:
                pose.position = Vector3([
                    -pose.position.x,
                    -pose.position.y,
                    pose.position.z,
                ])
                pose.orientation = pose.orientation * Quaternion.from_eulers([math.pi, 0, 0])
            scene.add_robot(robot, pose)

            if options.rerun:
                scene.add_site(
                    parent=f"mbs{i+1}", parent_tag="attachment_frame",
                    name=f"robot_fwd_stem", pos=[0, 0, 0.075],
                    size=[.05, .005, .001],
                    rgba=[.5, 0, 0, 1],
                    type="box"
                )
                scene.add_site(
                    parent=f"mbs{i+1}", parent_tag="attachment_frame",
                    name=f"robot_fwd_head", pos=[0.05, 0, 0.075], quat=Quaternion.from_x_rotation(math.pi / 4),
                    size=[.01, .01, .001],
                    rgba=[.5, 0, 0, 1],
                    type="box"
                )

                scene.add_site(
                    parent=None,
                    name=f"robot_start_{i}", pos=pose.position,
                    size=[.1, .1, .005],
                    rgba=[.1, 0., 0, 1.],
                    type="ellipsoid"
                )

        objects = []
        if config.experiment not in [ExperimentType.LOCOMOTION,
                                     ExperimentType.FOLLOW]:
            r = .05
            pose = Pose(ball_position(config, r))
            ball = Ball(radius=r, mass=0.05, pose=pose,
                        texture=Texture(base_color=Color(0, 255, 0, 255)))
            scene.add_interactive_object(ball)
            objects.append(ball)

            if options.rerun:
                scene.add_site(
                    parent=None,
                    name="ball_start", pos=[pose.position.x, pose.position.y, 0],
                    size=[r, r, .001],
                    rgba=[0., .1, 0, 1.],
                    type="ellipsoid"
                )

        if config.experiment in [ExperimentType.PUNCH_BACK]:
            simulator.register_callback(Callback.START, cls.push_ball)

        if options.rerun:
            scene.add_camera(
                name="tracking-camera",
                mode="targetbody",
                target=f"mbs{len(robots)+len(objects)}/",
                pos=camera_position(config.experiment)
            )

        if config.experiment is ExperimentType.FOLLOW:
            scene.add_body(
                parent=None,
                name="human",
                mocap=True
            )

        if config.experiment in [ExperimentType.PUNCH_BACK,
                                 ExperimentType.PUNCH_THRICE]:
            scene.add_site(
                parent=None,
                name="puncher",
                pos=[BackAndForthFitnessData.target_distance, 0, .075],
                size=[.075, .075, .075],
                rgba=[1., 0., 0., .5],
                type="box"
            )

        if config.experiment in [ExperimentType.PUNCH_THRICE,
                                 ExperimentType.PUNCH_TOGETHER]:
            fd_class = BackAndForthFitnessData
        elif config.experiment is ExperimentType.FOLLOW:
            fd_class = FollowFitnessData
        elif config.experiment is ExperimentType.PUNCH_FOLLOW:
            fd_class = PunchFollowFitnessData
        else:
            fd_class = FitnessData
        fd = fd_class(robots, objects, render=not options.headless)

        if config.experiment in [ExperimentType.PUNCH_THRICE,
                                 ExperimentType.PUNCH_TOGETHER]:
            simulator.register_callback(Callback.START, fd.start)
            simulator.register_callback(Callback.PRE_STEP, fd.before_step)
            simulator.register_callback(Callback.POST_STEP, fd.after_step)
            if not options.headless:
                simulator.register_callback(Callback.PRE_RENDER, fd.pre_render)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=simulator,
            batch_parameters=batch_parameters,
            scenes=scene,
            record_settings=(
                None
                if not options.movie else
                RecordSettings(
                    video_directory=str(config.data_root),
                    overwrite=True,
                    width=640, height=480,
                    camera_type="fixed", camera_id=0
                )
            )
        )

        fd.update(states=scene_states)
        fitness = cls._fitness(fd)
        try:
            assert not math.isnan(fitness) and not math.isinf(fitness), f"{fitness=}"
        except Exception as e:
            raise RuntimeError(f"{fitness=}") from e
        return EvaluationResult(fitness=fitness)

    @staticmethod
    def rename_movie(genome_file: Path):
        src = genome_file.parent.joinpath("0.mp4")
        assert src.exists(), f"{src=} does not exist"
        dst = genome_file.parent.joinpath(genome_file.stem).with_suffix(".mp4")
        src.rename(dst)
        assert dst.exists(), f"{dst=} does not exist"
        logging.info(f"Renamed {src=} to {dst=}")

    @staticmethod
    def fitness_locomotion(fd: FitnessData):
        # Get as far as possible with as few modules as possible
        # Hinges cost more.

        assert len(fd.robots) == 1, "This experiment is not meant for multiple robots"
        assert len(fd.objects) == 0, "This experiment does not require interactive objects"
        robot = fd.robots[0]
        modules = {
            t: len(robot.body.find_modules_of_type(t))
            for t in [BrickV2, ActiveHingeV2]
        }

        states = fd.states
        i = int(.8*(len(states)-1))
        pos = _robot_pos(robot, fd)
        return (
                .01 * euclidian_distance(pos(0), pos(i-1))
                + euclidian_distance(pos(i), pos(-1))
        ) / max(1, modules[BrickV2] + 2 * modules[ActiveHingeV2])

    @staticmethod
    def fitness_punch_once(fd: FitnessData):
        # Punch the ball as far as possible
        # If you do not touch the ball at least try to get close to it
        robot, ball = single_robot_and_ball(fd)
        robot_pos = _robot_pos(robot, fd)
        ball_pos = _ball_pos(ball, fd)

        ball_dist = euclidian_distance(ball_pos(0), ball_pos(-1))
        if ball_dist > 0:
            return ball_dist
        else:
            return -euclidian_distance(ball_pos(-1), robot_pos(-1))

    @staticmethod
    def fitness_punch_ahead(fd: FitnessData):
        # Punch the ball *forward* as far as possible
        # If you do not touch the ball at least try to get close to it
        robot, ball = single_robot_and_ball(fd)
        robot_pos = _robot_pos(robot, fd)
        ball_pos = _ball_pos(ball, fd)

        b0, b1 = ball_pos(0), ball_pos(-1)
        ball_dist = euclidian_distance(b0, b1)
        if ball_dist > 0:
            return (b1.x - b0.x) - abs(b1.y - b0.y)
        else:
            return -10 - .1 * euclidian_distance(ball_pos(-1), robot_pos(-1))

    @staticmethod
    def fitness_punch_back(fd: FitnessData):
        # Punch the ball back. It's coming right for you!
        _, ball = single_robot_and_ball(fd)
        ball_pos = _ball_pos(ball, fd)
        b0, b1 = ball_pos(0), ball_pos(-1)
        return (b1.x - b0.x) - abs(b1.y - b0.y)

    @staticmethod
    def fitness_punch_thrice(fd: BackAndForthFitnessData):
        if fd.fitness == 0:
            robot, ball = single_robot_and_ball(fd)
            return -100-euclidian_distance(_ball_pos(ball, fd)(-1),
                                           _robot_pos(robot, fd)(-1))
        else:
            return fd.fitness

    @staticmethod
    def fitness_punch_together(fd: BackAndForthFitnessData):
        if fd.fitness == 0:
            robots, ball = robots_and_ball(fd, n=2)
            return -100-min(
                euclidian_distance(_ball_pos(ball, fd)(-1),
                                   _robot_pos(r, fd)(-1))
                for r in robots)
        else:
            return fd.fitness

    @staticmethod
    def fitness_follow(fd: FollowFitnessData):
        return fd.fitness

    @staticmethod
    def fitness_punch_follow(fd: PunchFollowFitnessData):
        return fd.fitness

    @staticmethod
    def push_ball(model: MjModel, data: MjData,
                  mapping: AbstractionToMujocoMapping,
                  handler: ModularRobotSimulationHandler):
        data.qvel[-6] -= [BackAndForthFitnessData.velocity]


def single_robot_and_ball(fd: FitnessData):
    assert len(fd.robots) == 1, "This experiment is not meant for multiple robots."
    robot = fd.robots[0]
    assert len(fd.objects) == 1, "This experiment requires a single object."
    ball = fd.objects[0]
    assert isinstance(ball, Ball), "This experiment's interactive object should be a Ball."
    return robot, ball


def robots_and_ball(fd: FitnessData, n=2):
    assert len(fd.robots) == n, f"This experiment is meant for {n} robots."
    robots = fd.robots[0:n]
    assert len(fd.objects) == 1, "This experiment requires a single object."
    ball = fd.objects[0]
    assert isinstance(ball, Ball), "This experiment's interactive object should be a Ball."
    return robots, ball


def clip(low, value, high): return max(low, min(value, high))


def _robot_pos(robot, fd: FitnessData) -> Callable[[int], Vector3]:
    return lambda i: fd.states[i].get_modular_robot_simulation_state(robot).get_pose().position


def _ball_pos(ball, fd: FitnessData) -> Callable[[int], Vector3]:
    return lambda i: fd.states[i]._simulation_state.get_multi_body_system_pose(ball).position


def euclidian_distance(start, finish):
    return math.sqrt((finish.x - start.x)**2 + (finish.x - start.x)**2)


def performance_compare(lhs: EvaluationResult, rhs: EvaluationResult, verbosity):
    width = 20
    key_width = max(len(k) for keys in
                    [["fitness"], lhs.stats or [], rhs.stats or []]
                    # [lhs.fitnesses, lhs.stats, rhs.fitnessess, rhs.stats]
                    for k in keys) + 1

    def s_format(s=''): return f"{s:{width}}"

    def f_format(f):
        if isinstance(f, numbers.Number):
            # return f"{f}"[:width-3] + "..."
            return s_format(f"{f:g}")
        else:
            return "\n" + pprint.pformat(f, width=width)

    def map_compare(lhs_d: Dict[str, float], rhs_d: Dict[str, float]):
        output, code = "", 0
        lhs_keys, rhs_keys = set(lhs_d.keys()), set(rhs_d.keys())
        all_keys = sorted(lhs_keys.union(rhs_keys))
        for k in all_keys:
            output += f"{k:>{key_width}}: "
            lhs_v, rhs_v = lhs_d.get(k), rhs_d.get(k)
            if lhs_v is None:
                output += f"{Fore.YELLOW}{s_format()} > {f_format(rhs_v)}"
            elif rhs_v is None:
                output += f"{Fore.YELLOW}{f_format(lhs_v)} <"
            else:
                if lhs_v != rhs_v:
                    lhs_str, rhs_str = f_format(lhs_v), f_format(rhs_v)
                    if isinstance(lhs_v, numbers.Number):
                        diff = rhs_v - lhs_v
                        ratio = math.inf if lhs_v == 0 else diff/math.fabs(lhs_v)
                        output += f"{Fore.RED}{lhs_str} | {rhs_str}" \
                                  f"\t({diff}, {100*ratio:.2f}%)"
                    else:
                        output += "\n"
                        for lhs_item, rhs_item in zip(lhs_str.split('\n'), rhs_str.split('\n')):
                            if lhs_item != rhs_item:
                                output += Fore.RED
                            output += f"{lhs_item:{width}s} | {rhs_item:{width}s}"
                            if lhs_item != rhs_item:
                                output += Style.RESET_ALL
                            output += "\n"
                    code = 1
                else:
                    output += f"{Fore.GREEN}{f_format(lhs_v)}"

            output += f"{Style.RESET_ALL}\n"
        return output, code

    def json_compliant(obj): return json.loads(json.dumps(obj))

    f_str, f_code = map_compare({"fitness": lhs.fitness},
                                {"fitness": rhs.fitness})
    # d_str, d_code = map_compare(lhs.descriptors,
    #                             json_compliant(rhs.descriptors))
    s_str, s_code = map_compare(lhs.stats or {},
                                json_compliant(rhs.stats or {}))

    error = max([f_code, s_code])
    verbosity += error

    max_width = max(len(line) for text in [f_str, s_str] for line in text.split('\n'))
    if verbosity == 1:
        summary = []
        codes = {0: Fore.GREEN, 1: Fore.RED}
        for _code, name in [(f_code, "fitness"),
                            # (d_code, "descriptors"),
                            (s_code, "stats")]:
            summary.append(f"{codes[_code]}{name}{Style.RESET_ALL}")
        print(f"Performance summary: {lhs.fitness} ({' '.join(summary)})")
        # print(f"Performance summary: {lhs.fitnesses} ({' '.join(summary)})")

    elif verbosity > 1:
        def header(): print("-"*max_width)
        print("Performance summary:")
        header()
        print(f_str, end='')
        header()
        # print(d_str, end='')
        # header()
        print(s_str, end='')
        header()
        print()

    # return max([f_code, d_code, s_code])
    return
