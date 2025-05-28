import dataclasses
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from mujoco import MjModel, MjData
from pyparsing import actions

from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.brain import BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot_simulation import ModularRobotScene
from revolve2.modular_robot_simulation._build_multi_body_systems import BodyToMultiBodySystemMapping
from revolve2.modular_robot_simulation._sensor_state_impl import ModularRobotSensorStateImpl
from revolve2.simulation.simulator import RecordSettings
from revolve2.simulators.mujoco_simulator._abstraction_to_mujoco_mapping import AbstractionToMujocoMapping
from revolve2.simulators.mujoco_simulator._control_interface_impl import ControlInterfaceImpl
from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model
from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl
from revolve2.standards.simulation_parameters import STANDARD_SIMULATION_TIME, \
    STANDARD_SIMULATION_TIMESTEP, STANDARD_CONTROL_FREQUENCY


class StepwiseFitnessFunction(ABC):
    @abstractmethod
    def reset(self, model: MjModel, data: MjData) -> None: ...

    @abstractmethod
    def before_step(self, model: MjModel, data: MjData) -> None: ...

    @abstractmethod
    def after_step(self, model: MjModel, data: MjData) -> None: ...

    @abstractmethod
    def fitness(self) -> float: ...

    @abstractmethod
    def reward(self) -> float: ...


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

        self._renderer = None

        LocalSimulator.reset(self, scene, **kwargs)

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

        self._fitness_function.reset(self._model, self._data)

        mujoco.mj_forward(self._model, self._data)

    def step(self):
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

        self._next_control_time += self._parameters.control_step

        # Fast-forward to next control timestep
        while self._data.time < self._next_control_time:
            mujoco.mj_step(self._model, self._data)
            # print("[MjStep]", self._data.time)

        self._fitness_function.after_step(self._model, self._data)

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

    @property
    def done(self): return self._data.time >= self._parameters.simulation_time


class _Renderer:
    def __init__(self):
        raise NotImplementedError()


class PassthroughBrain(BrainInstance):
    def __init__(self, mapping: BodyToMultiBodySystemMapping):
        super().__init__()
        self._mapping = mapping
        self._actions = np.zeros(len(mapping.active_hinge_to_joint_hinge))

    def set_action(self, action):
        assert action.shape == self._actions.shape
        self._actions[:] = action

    def control(self, dt, sensor_state, control_interface) -> None:
        for action, hinge in zip(self._actions, self._mapping.active_hinge_to_joint_hinge):
            control_interface.set_active_hinge_target(
                hinge.value, action * hinge.value.range
            )


class GymSimulator(LocalSimulator, gym.Env):
    metadata = {'render.modes': ['human'], "render_fps": 60}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hinges = len(self._mapping.hinge_joint)
        inputs = hinges  # TODO Only counts hinges
        outputs = hinges
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(inputs,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(outputs,))

        assert len(self.agents()) == 1, f"Painful enough with one agent. Not bothering with more"

        for i, (old_brain, mapping) in enumerate(self.agents()):
            self.agents()[i] = (PassthroughBrain(mapping), mapping)

    def agents(self): return self._handler._brains

    def observations(self):
        data = []
        data.extend(self._data.sensordata) # Sensors first (only IMU??)

        simulation_state = SimulationStateImpl(
            data=self._data,
            abstraction_to_mujoco_mapping=self._mapping,
            camera_views={}  # TODO Missing cameras
        )

        for brain, bmb_mapping in self.agents():
            sensor_state = ModularRobotSensorStateImpl(
                simulation_state=simulation_state,
                body_to_multi_body_system_mapping=bmb_mapping,
            )
            data.extend([
                sensor_state.get_active_hinge_sensor_state(hinge.sensors.active_hinge_sensor).position
                for i, hinge in enumerate(bmb_mapping.active_hinge_to_joint_hinge)
            ])
        return np.array(data)

    def infos(self): return dict()

    def reset(self, seed=None, options=None):
        super().reset(scene=None, **options)
        return self.observations(), self.infos()

    def step(self, action):
        self.agents()[0].set_action(action)

        super().step()

        return self.observations(), self._fitness_function.reward(), self.done, False, self.infos()
