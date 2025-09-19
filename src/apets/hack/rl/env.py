import copy
import logging

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.env_util import make_vec_env

from apets.hack.evaluator import Evaluator
from apets.hack.local_simulator import LocalSimulator
from revolve2.modular_robot.brain import BrainInstance
from revolve2.modular_robot_simulation._build_multi_body_systems import BodyToMultiBodySystemMapping
from revolve2.modular_robot_simulation._sensor_state_impl import ModularRobotSensorStateImpl
from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl


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
            assert -1 <= action <= 1
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
        data.extend(self._data.sensordata)  # Sensors first (only IMU??)

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
            # TODO: this does NOT smell good
            data.extend(np.clip([
                sensor_state.get_active_hinge_sensor_state(hinge.value.sensors.active_hinge_sensor).position
                / hinge.value.range
                for i, hinge in enumerate(bmb_mapping.active_hinge_to_joint_hinge)
            ], -1, 1))
        assert all(-1 <= x <= 1 for x in data)
        return np.array(data, dtype=np.float32)

    def infos(self): return dict(**self.reward_function.infos)

    def previous_infos(self): return dict(**self.reward_function.previous_infos)

    @property
    def reward_function(self): return self.fitness_function

    @property
    def reward(self) -> float: return self._fitness_function.reward

    @property
    def cumulative_reward(self) -> float: return self._fitness_function.fitness

    @property
    def truncated(self) -> bool: return self._canceled

    def reset(self, seed=None, options=None):
        # print(f"reset at time {self._data.time}")
        super().reset(scene=None, **(options or {}))
        return self.observations(), self.infos()

    def step(self, action):
        self.agents()[0][0].set_action(action)

        super().step()
        reward = self.reward

        return self.observations(), reward, self.done, self._fitness_function.invalid, self.infos()

    def plot_trajectory(self):
        return self.reward_function.plot_trajectory(column="R")

    def plot_trajectory_as_image(self):
        try:
            fig = self.plot_trajectory()
            canvas = fig.canvas
            canvas.draw()
            data = np.asarray(canvas.buffer_rgba())
            plt.close()
            return data
        except Exception as e:
            logging.error(f"Failed to plot trajectory: {e}")
            return None


def make(robot, rotated, reward_function, name=None,
         rerun=False, render=False,
         **kwargs):
    scene = Evaluator.scene(robot, rerun, rotated)
    options = dict(
        headless=not render,
        start_paused=False,
        simulation_time=10,
        label=name,
        record_settings=None,
    )
    options.update(kwargs)
    simulator = GymSimulator(
        scene=scene,
        fitness_function=copy.deepcopy(reward_function),
        **options
    )

    return simulator


def make_vec(n,
             *,
             robot, rotated, reward_function,
             vec_env_cls,
             name=None,
             **kwargs):
    return make_vec_env(
        env_id=make,
        n_envs=n,
        env_kwargs=dict(
            robot=robot,
            name=name,
            rotated=rotated,
            reward_function=reward_function,
            **kwargs
        ),
        vec_env_cls=vec_env_cls,
    )
