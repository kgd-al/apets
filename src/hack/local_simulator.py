import dataclasses
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import mujoco
from mujoco import MjModel, MjData

from revolve2.modular_robot_simulation import ModularRobotScene
from revolve2.simulation.simulator import BatchParameters, RecordSettings
from revolve2.simulators.mujoco_simulator._control_interface_impl import ControlInterfaceImpl
from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model
from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl
from revolve2.standards.simulation_parameters import make_standard_batch_parameters, STANDARD_SIMULATION_TIME, \
    STANDARD_SIMULATION_TIMESTEP, STANDARD_CONTROL_FREQUENCY


class RewardMetric(ABC):
    @abstractmethod
    def before_step(self, model: MjModel, data: MjData) -> None: ...

    @abstractmethod
    def after_step(self, model: MjModel, data: MjData) -> None: ...

    @abstractmethod
    def fitness(self) -> float: ...

    @abstractmethod
    def reward(self) -> float: ...

    @abstractmethod
    def reset(self) -> None: ...

class LocalSimulator:
    @dataclass
    class Parameters:
        simulation_time: int = STANDARD_SIMULATION_TIME
        simulation_timestep: float = STANDARD_SIMULATION_TIMESTEP
        control_frequency: float = STANDARD_CONTROL_FREQUENCY

        headless: bool = False
        start_paused: bool = False
        cast_shadows: bool = False
        fast_sim: bool = False

        control_step: float = None

        record_settings: RecordSettings = None

    def __init__(
            self,
            scene: Optional[ModularRobotScene],
            reward_metric: RewardMetric,
            **kwargs
    ):
        self._parameters = self.Parameters()

        self._reward_metric = reward_metric
        self._model, self._data, self._mapping = None, None, None

        self._handler = None
        self._control_interface, self._next_control_time = None, 0

        self._renderer = None

        self.reset(scene, **kwargs)

    def reset(self, scene: ModularRobotScene | None, **parameters):
        if scene is not None:
            _scene, _mapping = scene.to_simulation_scene()
            self._handler = _scene.handler

            self._parameters = dataclasses.replace(self._parameters, **parameters)
            self._parameters.control_step = 1.0 / self._parameters.control_frequency

            if self._parameters.record_settings is not None:
                os.makedirs(
                    self._parameters.record_settings.video_directory,
                    exist_ok=self._parameters.record_settings.overwrite,
                )

            self._model, self._mapping = scene_to_model(
                _scene, self._parameters.simulation_timestep,
                cast_shadows=self._parameters.cast_shadows,
                fast_sim=self._parameters.fast_sim,
            )
            self._data = mujoco.MjData(self._model)

            self._control_interface = ControlInterfaceImpl(
                data=self._data, abstraction_to_mujoco_mapping=self._mapping
            )

            if not self.headless or self.offscreen_render:
                self._renderer = _Renderer()

        else:
            mujoco.mj_resetData(self._model, self._data)

        """Define some additional control variables."""
        self._next_control_time = 0.0

        self._reward_metric.reset()

        mujoco.mj_forward(self._model, self._data)

    def step(self):
        self._reward_metric.before_step(self._model, self._data)

        # Do control
        # print("[Control]", self._data.time)
        self._handler.handle(
            SimulationStateImpl(
                data=self._data,
                abstraction_to_mujoco_mapping=self._mapping,
                camera_views={}  # TODO Missing cameras
            ),
            self._control_interface, self._parameters.control_step)

        self._next_control_time += self._parameters.control_step

        # Fast-forward to next control timestep
        while self._data.time < self._next_control_time:
            mujoco.mj_step(self._model, self._data)
            # print("[MjStep]", self._data.time)

        self._reward_metric.after_step(self._model, self._data)

        return (self._reward_metric.reward,
                self._data.time >= self._parameters.simulation_time)

    def render(self):
        pass

    @property
    def headless(self): return self._parameters.headless

    @property
    def record(self): return self._parameters.record_settings is not None

    @property
    def offscreen_render(self): return self.headless and self.record

    @property
    def renderer(self): return self._renderer


class _Renderer:
    def __init__(self):
        raise NotImplementedError()
