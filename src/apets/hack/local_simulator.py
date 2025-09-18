import dataclasses
import os
import pathlib
import pprint
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable

import cv2
import glfw
import mujoco
import yaml
from gymnasium.envs.mujoco import MujocoRenderer
from mujoco import MjModel, MjData, Renderer, MjvCamera
from mujoco.viewer import Handle, launch, launch_passive
from mujoco_viewer import MujocoViewer

from revolve2.modular_robot_simulation import ModularRobotScene
from revolve2.modular_robot_simulation._sensor_state_impl import ModularRobotSensorStateImpl
from revolve2.simulation.simulator import RecordSettings
from revolve2.simulators.mujoco_simulator._abstraction_to_mujoco_mapping import AbstractionToMujocoMapping
from revolve2.simulators.mujoco_simulator._control_interface_impl import ControlInterfaceImpl
from revolve2.simulators.mujoco_simulator._render_backend import RenderBackend
from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model
from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl
from revolve2.simulators.mujoco_simulator.viewers import CustomMujocoViewer
from revolve2.standards.simulation_parameters import STANDARD_SIMULATION_TIME, \
    STANDARD_SIMULATION_TIMESTEP, STANDARD_CONTROL_FREQUENCY


class StepwiseFitnessFunction(ABC):
    @abstractmethod
    def reset(self, model: MjModel, data: MjData) -> None: ...

    @abstractmethod
    def before_step(self, model: MjModel, data: MjData) -> None: ...

    @abstractmethod
    def after_step(self, model: MjModel, data: MjData) -> None: ...

    def render(self, model: MjModel, data: MjData, viewer) -> None:
        pass

    @property
    @abstractmethod
    def fitness(self) -> float: ...

    @property
    @abstractmethod
    def reward(self) -> float: ...

    @property
    def invalid(self) -> bool: return False

    @property
    def infos(self) -> dict: return dict()


class LocalSimulator:
    @dataclass(frozen=True)
    class Parameters:
        simulation_time: int = STANDARD_SIMULATION_TIME
        simulation_timestep: float = STANDARD_SIMULATION_TIMESTEP
        control_frequency: float = STANDARD_CONTROL_FREQUENCY

        label: str = None

        headless: bool = True
        start_paused: bool = False
        cast_shadows: bool = False
        fast_sim: bool = False

        control_step: float = None

        record_settings: RecordSettings = None

    def __init__(
            self,
            scene: Optional[ModularRobotScene],
            fitness_function: StepwiseFitnessFunction,
            **kwargs
    ):
        self._parameters = self.Parameters()

        self._fitness_function = fitness_function
        self._model: MjModel | None = None
        self._data: MjData | None = None
        self._mapping: AbstractionToMujocoMapping | None = None

        self._handler = None
        self._control_interface: ControlInterfaceImpl | None = None
        self._next_control_time = 0

        self._canceled = False

        LocalSimulator.reset(self, scene, **kwargs)

        viewer = not self._parameters.headless
        record = self._parameters.record_settings is not None
        self._visu = (
            _Visualizer(self._model, self._data, self._parameters, self._fitness_function.render)
            if viewer or record
            else None
        )

    def reset(self, scene: ModularRobotScene | None, **parameters):
        if scene is not None:
            _scene, _mapping = scene.to_simulation_scene()
            self._handler = _scene.handler

            self._parameters = dataclasses.replace(self._parameters, **parameters)
            self._parameters = dataclasses.replace(
                self._parameters, control_step=1.0 / self._parameters.control_frequency)

            if self._parameters.record_settings is not None:
                os.makedirs(
                    self._parameters.record_settings.video_directory,
                    exist_ok=self._parameters.record_settings.overwrite,
                )

            self._model, self._data, self._mapping = scene_to_model(
                _scene, self._parameters.simulation_timestep,
                cast_shadows=self._parameters.cast_shadows,
                fast_sim=self._parameters.fast_sim,
            )

            self._control_interface = ControlInterfaceImpl(
                data=self._data, abstraction_to_mujoco_mapping=self._mapping
            )

        else:
            mujoco.mj_resetData(self._model, self._data)

        self._canceled = False

        mujoco.mj_forward(self._model, self._data)

        self._fitness_function.reset(self._model, self._data)

    def run(self, render: bool = False):
        if render:
            self.render()

        while not self.done:
            self.step()
            if render:
                self.render()

    def step(self):
        if self._visu is not None and not self._visu.alive:
            self._canceled = True
            return

        self._fitness_function.before_step(self._model, self._data)

        # Do control
        # print("[Control]", self._data.time)
        self._handler.handle(
            SimulationStateImpl(
                data=self._data,
                abstraction_to_mujoco_mapping=self._mapping,
                camera_views={}  # TODO Missing cameras
            ),
            self._control_interface, self._parameters.control_step)

        # ==========================
        # = Debugging proprioception
        # ==========================

        body_to_multi_body_system_mapping = self._handler._brains[0][1]
        sensor_state = ModularRobotSensorStateImpl(
            simulation_state=SimulationStateImpl(
                data=self._data,
                abstraction_to_mujoco_mapping=self._mapping,
                camera_views={}  # TODO Missing cameras
            ),
            body_to_multi_body_system_mapping=body_to_multi_body_system_mapping,
        )

        print("Mujoco data")
        print(self._data.time, [x for x in self._data.qpos])
        print(f"{'mbs1/pos + mbs1/quat':>25s} {self._data.body('mbs1/').xpos} {self._data.body('mbs1/').xquat}")
        for i in range(self._model.njnt):
            joint = self._data.joint(i)
            print(f"{joint.name:>25s} {[x for x in joint.qpos]}")
        print("Revolve proprioceptive data")
        print(self._data.time, [
            sensor_state.get_active_hinge_sensor_state(hinge._value).position
            for hinge in body_to_multi_body_system_mapping.active_hinge_sensor_to_joint_hinge.keys()
        ])
        print()

        # ==========================

        # Could diverge
        substeps = self._parameters.control_step / self._model.opt.timestep
        assert substeps == round(substeps)
        substeps = round(substeps)

        # Fast-forward to next control timestep
        mujoco.mj_step(self._model, self._data, nstep=substeps)

        if not self.done:
            self._fitness_function.after_step(self._model, self._data)
        self._canceled |= self._fitness_function.invalid

    def render(self):
        assert self._visu is not None, "Visualization not requested at initialization"
        self._canceled = self._canceled or not self._visu.render()

    @property
    def parameters(self): return self._parameters

    @property
    def fitness_function(self): return self._fitness_function

    @property
    def timeout(self): return self._data.time >= self._parameters.simulation_time

    @property
    def canceled(self): return self._canceled

    @property
    def done(self):
        return self.timeout or self.canceled

    @property
    def mj_model(self): return self._model

    @property
    def mj_data(self): return self._data


LATEST_MUJOCO = 332
OLD_MUJOCO = mujoco.mj_version() < LATEST_MUJOCO
if OLD_MUJOCO:
    print(f"WARNING: Your mujoco version is old {mujoco.mj_version()} < {LATEST_MUJOCO}")


def persistent_settings_custom_mujoco_viewer(viewer):
    close = viewer.close

    storage = viewer.CONFIG_PATH
    storage.parent.mkdir(parents=True, exist_ok=True)

    def monkey_patch():
        if not viewer.is_alive:
            return
        with open(storage, "w") as f:
            window = viewer.window
            yaml.safe_dump(
                dict(
                    pos=glfw.get_window_pos(window),
                    size=glfw.get_window_size(window),
                    menus=viewer._hide_menus,
                    speed=viewer._run_speed,
                ),
                f
            )

        close()

    viewer.close = monkey_patch

    try:
        with open(storage, "r") as f:
            if (config := yaml.safe_load(f)) is not None:
                # print("[kgd-debug] restored viewer config: ", config)
                window = viewer.window
                glfw.restore_window(window)
                if (pos := config.get("pos")) is not None:
                    glfw.set_window_pos(window, *pos)
                if (size := config.get("size")) is not None:
                    glfw.set_window_size(window, *size)
                viewer._hide_menus = config.get("menus", False)
                viewer._run_speed = config.get("speed", 1)

    except FileNotFoundError:
        pass


class _Visualizer:
    def __init__(self,
                 model: MjModel, data: MjData,
                 parameters: LocalSimulator.Parameters,
                 render_callback: Callable = None):

        self._records = parameters.record_settings is not None
        self._with_viewer = not parameters.headless and not self._records
        self._render_callback = render_callback

        self._renderer, self._viewer = None, None
        self._model, self._data = model, data

        self.geoms = []

        self.recorder = None

        self.camera = "mbs1/tracking-camera"

        if not self._with_viewer and self._records:  # offscreen
            rs = parameters.record_settings
            self._renderer = Renderer(model, width=rs.width, height=rs.height)

            filename = str(rs.video_directory)
            if not filename.endswith(".mp4"):
                filename += "/movie.mp4"
            self._video_writer = cv2.VideoWriter(
                filename=filename,
                fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
                fps=rs.fps,
                frameSize=(rs.width, rs.height)
            )

            self._close = lambda: self._video_writer.release()

        else:
            if OLD_MUJOCO:
                self._viewer = CustomMujocoViewer(model, data,
                                                  backend=RenderBackend.GLFW,
                                                  start_paused=parameters.start_paused)
                self._viewer_alive = self._viewer.is_alive
                self._close = self._viewer.close_viewer
                backend = self._viewer._viewer_backend
                persistent_settings_custom_mujoco_viewer(backend)
                camera = backend.cam

            else:
                self._paused = True#parameters.start_paused
                self._step = False

                def key_event(key):
                    if chr(key) == ' ':
                        self._paused = not self._paused
                    elif chr(key) == 'N':
                        self._step = True

                self._viewer = launch_passive(
                    model, data,
                    key_callback=key_event)
                self._viewer_alive = self._viewer.is_running
                self._close = self._viewer.close
                camera = self._viewer.cam
                self._last_step = time.time()
                self._target_fps = 1/60.

            camera.fixedcamid = data.camera(self.camera).id
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    @property
    def alive(self):
        return self.is_recording or (self.is_viewer and self._viewer_alive())

    @property
    def is_viewer(self): return self._with_viewer

    @property
    def is_recording(self): return self._records

    @classmethod
    def persistent_storage(cls):
        path = pathlib.Path.joinpath(pathlib.Path.home(), ".config/mujoco/viewer.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def render(self):
        if self._with_viewer and self._render_callback is not None:
            self._render_callback(self._model, self._data, self._viewer)

        if self._renderer is not None:
            self._renderer.update_scene(self._data, camera=self.camera)
            img = self._renderer.render()
            self._video_writer.write(img[:, :, ::-1])

        if self._viewer is not None:
            if not self.alive:
                self.close()
                return False

            if OLD_MUJOCO:
                self._viewer.render()
            else:
                self._viewer.sync()

                while self._paused and not self._step and self.alive:
                    time.sleep(self._target_fps)
                else:
                    # Rudimentary time keeping, will drift relative to wall clock.
                    time_until_next_step = self._target_fps - (time.time() - self._last_step)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                self._step = False
                self._last_step = time.time()

            if not self.alive:
                self.close()
                return False

        return True

    def close(self):
        # self._save()
        self._close()

    # def _restore(self):
    #     print(f"[kgd-debug] _restore({self.persistent_storage()})")
    #     try:
    #         with open(self.persistent_storage(), "r") as f:
    #             if (config := yaml.safe_load(f)) is not None:
    #                 print("[kgd-debug] restored viewer config: ", config)
    #                 window = self._window
    #                 glfw.restore_window(window)
    #                 glfw.set_window_pos(window, *config["pos"])
    #                 glfw.set_window_size(window, *config["size"])
    #
    #     except FileNotFoundError:
    #         pass
    #
    # def _save(self):
    #     print(f"[kgd-debug] _save({self.persistent_storage()})")
    #     with open(self.persistent_storage(), "w") as f:
    #         window = self._window
    #         yaml.safe_dump(
    #             dict(
    #                 pos=glfw.get_window_pos(window),
    #                 size=glfw.get_window_size(window),
    #             ),
    #             f
    #         )
