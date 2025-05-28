"""Evaluator class."""
import json
import logging
import math
import numbers
import pprint
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated, ClassVar, Dict, Any

import glfw
import yaml
from colorama import Style, Fore
from mujoco import MjModel, MjData
from mujoco_viewer import MujocoViewer
from pyrr import Vector3, Quaternion

from abrain.neat.config import ConfigBase
from abrain.neat.evolver import EvaluationResult
from config import Config, TaskType
from genotype import Genotype
from local_simulator import LocalSimulator, StepwiseFitnessFunction
from revolve2.experimentation.evolution.abstract_elements import Evaluator as Eval
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
)
from revolve2.simulation.scene import Pose, Color
from revolve2.simulation.scene.geometry import GeometryPlane
from revolve2.simulation.scene.geometry.textures import MapType
from revolve2.simulation.simulator import RecordSettings
from revolve2.simulators.mujoco_simulator.textures import Checker
from revolve2.simulators.mujoco_simulator.viewers import CustomMujocoViewer


@dataclass
class Options(ConfigBase):
    rerun: Annotated[bool, "Is this a rerun or are we evolving"] = False
    headless: Annotated[bool, "Do you need a GUI?"] = False
    movie: Annotated[bool, "Do you want a nice video of the robot?"] = False
    start_paused: Annotated[bool, "Should we let you a good look at the robot first?"] = False

    file: Annotated[Optional[Path], "Path to the genome under re-evaluation."] = None
    duration: Annotated[Optional[int], ("Simulation duration in seconds."
                                        " Overwrites the value used in evolution")] = None

    # ann_dynamics: Annotated[bool, "Whether to record precise dynamics of the neural network"] = False


def vec(x, y, z): return Vector3([x, y, z], dtype=float)


def make_custom_terrain() -> Terrain:
    s = 8
    geometries = [
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
    ]
    # geometries += [
    #     GeometryBox(
    #         pose=Pose(position=Vector3([x*s, y*s, .5]),
    #                   orientation=Quaternion.from_z_rotation(math.pi*r)),
    #         mass=0.0,
    #         texture=Flat(primary_color=Color(96, 96, 120, 255)),
    #         aabb=AABB(size=Vector3([.1, 2*s - .1, 1])),
    #     )
    #     for x, y, r in [(1, 0, 0), (0, 1, .5), (-1, 0, 1), (0, -1, 1.5)]
    # ]
    return Terrain(static_geometry=geometries)


class SubTaskFitnessData(StepwiseFitnessFunction):
    __debug = False

    def __init__(self, robot, state, render=False):
        self.robot = robot
        self.state = state

        self.robot_ix = None

        self._fitness, self._reward = 0, 0

        self.render = render

    @property
    def fitness(self): return self._fitness

    @property
    def cumulative_reward(self): return self.fitness

    @property
    def reward(self): return self._reward

    def reset(self, model: MjModel, data: MjData) -> None:
        self._fitness, self._reward = 0, 0
        self.robot_ix = model.body("mbs1/").id

    def robot_pos(self, data: MjData):
        return Vector3(data.xpos[self.robot_ix][:3].copy())

    def robot_ort(self, data: MjData):
        return data.xquat[self.robot_ix].copy()

    def _robot_xy_angle(self, data: MjData):
        v = Quaternion(self.robot_ort(data)) * vec(1, 0, 0)
        return math.atan2(v.y, v.x)

    def before_step(self, model: MjModel, data: MjData):
        pass

    def after_step(self, model: MjModel, data: MjData):
        pass

    def pre_render(self, model: MjModel, data: MjData, viewer: CustomMujocoViewer):
        pass


class MoveFitness(SubTaskFitnessData):
    def __init__(self, robot, forward=True, render=False):
        super().__init__(robot=robot,
                         state=1 if forward else -1,
                         render=render)
        self.prev_pos, self.prev_angle = None, None

    def before_step(self, model: MjModel, data: MjData):
        self.prev_pos = self.robot_pos(data)
        self.prev_angle = self._robot_xy_angle(data)
        # print("[kgd-debug] Before step:", self.prev_pos)

    def after_step(self, model: MjModel, data: MjData):
        pos = self.robot_pos(data)
        delta = pos - self.prev_pos

        x = self._robot_xy_angle(data)
        delta_angle = x - self.prev_angle
        # print("[kgd-debug] After step:", pos)
        # print("[kgd-debug] > Delta", delta_pos)

        self._reward = self.state * data.time * delta.x - .25 * (abs(delta.y) + abs(delta_angle))
        self._fitness += self._reward


class MoveForwardFitness(MoveFitness):
    def __init__(self, robot, render=False):
        super().__init__(robot=robot, forward=True, render=render)


class MoveBackwardFitness(MoveFitness):
    def __init__(self, robot, render=False):
        super().__init__(robot=robot, forward=False, render=render)


class RotateFitness(SubTaskFitnessData):
    def __init__(self, robot, rotate_direct, render=False):
        super().__init__(robot=robot,
                         state=1 if rotate_direct else -1,
                         render=render)
        self.prev_angle = None

    def before_step(self, model: MjModel, data: MjData):
        self.prev_angle = self._robot_xy_angle(data)
        # print("[kgd-debug:Rotate] Before step:", self.prev_angle)

    def after_step(self, model: MjModel, data: MjData):
        angle = self._robot_xy_angle(data)
        # print("[kgd-debug:Rotate] After step:", angle)
        delta = self.prev_angle - angle
        # print("[kgd-debug:Rotate] > Delta", delta)

        self._reward = self.state * delta
        self._fitness += self._reward


class RotateDirectFitness(RotateFitness):
    def __init__(self, robot, render=False):
        super().__init__(robot, True, render)


class RotateIndirectFitness(RotateFitness):
    def __init__(self, robot, render=False):
        super().__init__(robot, False, render)


TASKS = {
    TaskType.FORWARD: MoveForwardFitness,
    TaskType.BACKWARD: MoveBackwardFitness,
    TaskType.ROTATE_LEFT: RotateDirectFitness,
    TaskType.ROTATE_RIGHT: RotateIndirectFitness,
}


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
                config.simulation_duration = 5

        cls._data_folder = config.data_root

        cls._log = getattr(config, "logger", None)
        if cls._log and verbose:
            cls._log.info(f"Configuration:\n"
                          f"{pprint.pformat(cls.config)}\n"
                          f"{pprint.pformat(cls.options)}")

    @classmethod
    def scene(cls, robot, rerun, rotated):
        scene = ModularRobotScene(terrain=make_custom_terrain())

        rotation = Quaternion.from_x_rotation(math.pi / 4) if rotated else Quaternion()
        pose = Pose(orientation=rotation)
        scene.add_robot(robot, pose=pose)

        if rerun:
            scene.add_site(
                parent=f"mbs1", parent_tag="attachment_frame",
                name=f"robot_fwd_stem", pos=[0, 0, 0.075],
                quat=pose.orientation.inverse,
                size=[.05, .005, .001],
                rgba=[.5, 0, 0, 1],
                type="box"
            )
            scene.add_site(
                parent=f"mbs1", parent_tag="attachment_frame",
                name=f"robot_fwd_head", pos=pose.orientation.inverse * Vector3([0.05, 0, 0.075]),
                quat=Quaternion.from_x_rotation(math.pi / 4),
                size=[.01, .01, .001],
                rgba=[.5, 0, 0, 1],
                type="box"
            )

            scene.add_site(
                parent=None,
                name=f"robot_start", pos=pose.position,
                size=[.1, .1, .005],
                rgba=[.1, 0., 0, 1.],
                type="ellipsoid"
            )

        if rerun:
            scene.add_camera(
                name="tracking-camera",
                mode="targetbody",
                target=f"mbs1/",
                pos=pose.position + vec(-2, 0, 2)
            )

        return scene

    @classmethod
    def evaluate(cls, genotype: Genotype) -> EvaluationResult:
        """
        Evaluate a *single* robot.

        Fitness is the distance traveled on the xy plane.

        :param genotype: The genotype to develop into a robot and then simulate.
        :returns: Fitness of the robot.
        """

        config, options = cls.config, cls.options

        start = time.time()

        tasks = TASKS if not config.monotask else {TaskType.FORWARD: TASKS[TaskType.FORWARD]}
        duration = options.duration or config.simulation_duration
        # duration *= len(TASKS) / len(tasks)

        robot = genotype.develop(config)

        fitness = 0
        stats: Dict[str, Any] = dict(fitnesses=dict())

        for task, fd_type in tasks.items():
            # if not options.headless:
            #     simulator.register_callback(Callback.RENDER_START, PersistentViewerOptions.start)
                # if config.vision is not None:
                #     multiview = MultiCameraOverlay(config.vision, ABrainInstance.forward_colormap, ABrainInstance.inverse_colormap)
                #     simulator.register_callback(Callback.RENDER_START, multiview.start)
                #     simulator.register_callback(Callback.POST_RENDER, multiview.process)

            # if options.rerun:
            #     def write_brain(model, data, mapping, handler):
            #         brain: ANN3D = handler._brains[0][0].brain
            #         if not brain.empty():
            #             brain.render3D().write_html(config.data_root.joinpath("brain.html"))
            #     simulator.register_callback(Callback.START, write_brain)

            # Create the scenes.
            scene = cls.scene(robot, options.rerun, rotated=True)

            fd = fd_type(robot, render=not options.headless)
            simulator = LocalSimulator(
                scene=scene,
                fitness_function=fd,

                headless=True,#options.headless,
                start_paused=options.start_paused,
                simulation_time=duration,
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

            # simulator.register_callback(Callback.START, fd.start)
            # simulator.register_callback(Callback.PRE_STEP, fd.before_step)
            # simulator.register_callback(Callback.POST_STEP, fd.after_step)
            # if not options.headless:
            #     simulator.register_callback(Callback.PRE_RENDER, fd.pre_render)

            while not simulator.done:
                simulator.step()

            subfitness = fd.fitness
            stats["fitnesses"][task.name.lower()] = subfitness
            try:
                assert not math.isnan(subfitness) and not math.isinf(subfitness), f"{subfitness=}"
            except Exception as e:
                raise RuntimeError(f"{subfitness=}") from e

            fitness += subfitness
        fitness /= len(tasks)

        stats["time"] = time.time() - start
        # print(f"{genotype.id()=}", "time =", stats["time"])

        return EvaluationResult(
            fitness=fitness,
            stats=stats,
        )

    @staticmethod
    def rename_movie(genome_file: Path):
        src = genome_file.parent.joinpath("0.mp4")
        assert src.exists(), f"{src=} does not exist"
        dst = genome_file.parent.joinpath(genome_file.stem).with_suffix(".mp4")
        src.rename(dst)
        assert dst.exists(), f"{dst=} does not exist"
        logging.info(f"Renamed {src=} to {dst=}")


def performance_compare(lhs: EvaluationResult, rhs: EvaluationResult,
                        verbosity, ignore_fields=None):
    if ignore_fields is None:
        ignore_fields = []
    if "time" not in ignore_fields:
        ignore_fields.append("time")
    ignore_fields = set(ignore_fields)

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
        all_keys = sorted(lhs_keys.union(rhs_keys).difference(ignore_fields))
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
    verbosity = max(verbosity, error)

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
    return error
