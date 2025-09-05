"""An example on how to remote control a physical modular robot."""
import argparse
import itertools
import math
import random
import subprocess
from enum import Enum, auto
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from revolve2.modular_robot import ModularRobot, ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.v2 import BodyV2, ActiveHingeV2, BrickV2
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot_physical import Config, UUIDKey
from revolve2.modular_robot_physical.remote import run_remote

wifi = subprocess.check_output("iwgetid -r".split()).decode()
assert "ThymioNet" in wifi, f"WRONG NETWORK: {wifi}"


rng = random.Random(0)


def gecko_v2() -> BodyV2:
    """
    Sample robot with new HW config.

    :returns: the robot
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(
        0.0
    )
    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = (
        BrickV2(0.0)
    )

    return body


def spider_v2() -> BodyV2:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.front_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.front_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(0.0)

    return body


BODIES = {
    "gecko": gecko_v2,
    "spider": spider_v2,
}


class HingePinDebuggingBrainFactory(Brain):
    @staticmethod
    def periodic(fn, t, f): return fn(f * 2 * math.pi * t)

    @classmethod
    def sinusoidal(cls, t): return cls.periodic(fn=math.sin, f=.2, t=t)

    @classmethod
    def random(cls, t): return rng.random() * 2 - 1

    @classmethod
    def square(cls, t):
        pi = math.pi
        t = t%(2*pi)
        return (
            -1 if t < pi/2 else
            0 if t <= pi else
            1 if t < 3 * pi / 2 else
            0
        )

    class DebugTypes(Enum):
        SINUSOIDAL = auto()
        SQUARE = auto()
        RANDOM = auto()

    def __init__(self, hinges, index, d_type):
        self._target_hinge = hinges[index]
        if isinstance(d_type, str):
            d_type = self.DebugTypes[d_type]
        self._type = d_type

    def make_instance(self) -> BrainInstance:
        fn_mapping = {
            self.DebugTypes.SINUSOIDAL: self.sinusoidal,
            self.DebugTypes.SQUARE: self.square,
            self.DebugTypes.RANDOM: self.random,
        }
        return self.HingePinDebuggingBrain(self._target_hinge, fn_mapping[self._type])

    class HingePinDebuggingBrain(BrainInstance):
        def __init__(self, target_hinge, function):
            self._target_hinge = target_hinge
            self._function = function
            self._time = 0

        def control(
            self,
            dt: float,
            sensor_state: ModularRobotSensorState,
            control_interface: ModularRobotControlInterface,
        ) -> None:
            self._time += dt
            target = self._function(self._time) * self._target_hinge.range
            print(self._time, sensor_state.get_active_hinge_sensor_state(self._target_hinge), ">>", target)
            control_interface.set_active_hinge_target(self._target_hinge, target)


class RLBrainFactory(Brain):
    def __init__(self, path: Path, hinges, callback):
        self._model = PPO.load(path, device="cpu")
        self._hinges = hinges
        self._callback = callback

    def make_instance(self) -> BrainInstance:
        return self.RLBrainInstance(self._model, self._hinges, self._callback)

    class RLBrainInstance(BrainInstance):
        def __init__(self, model, hinges, callback):
            self._model = model
            self._hinges = hinges
            self._callback = callback
            self._time = 0

        def control(
            self,
            dt: float,
            sensor_state: ModularRobotSensorState,
            control_interface: ModularRobotControlInterface,
        ) -> None:
            self._time += dt
            print(self._time)
            obs = np.clip([
                sensor_state.get_active_hinge_sensor_state(hinge.sensors.active_hinge_sensor).position
                / hinge.range
                for i, hinge in enumerate(self._hinges)
            ], -1, 1)

            actions, _states = self._model.predict(obs, deterministic=True)
            self._callback(time=self._time, observations=obs, actions=actions)

            for action, hinge in zip(actions, self._hinges):
                assert -1 <= action <= 1
                control_interface.set_active_hinge_target(
                    hinge, float(action * hinge.range)
                )


def main() -> None:
    parser = argparse.ArgumentParser("Main entry point for robot hardware control")
    parser.add_argument("--ip", required=True, help="ip address of the robot")
    parser.add_argument("--body", required=True, help="robot morphology")
    parser.add_argument("--rotated", default=False, help="whether the robot is front-facing or rotated by 45 degrees")
    parser.add_argument("--duration", default=15, type=float, help="how long to run for (in seconds)")

    group = parser.add_mutually_exclusive_group()

    group.add_argument("--debug-hinge", type=int, default=None,
                        help="Index of the hinge to debug")
    parser.add_argument("--debug-type", choices=[e.name for e in HingePinDebuggingBrainFactory.DebugTypes],
                        help="Specify the type of function used to send signals")

    group.add_argument("--model", type=Path)

    args = parser.parse_args()

    debugging = args.debug_hinge is not None

    """Remote control a physical modular robot."""
    """
    Create a modular robot, similar to what was done in the 1a_simulate_single_robot example.
    Of course, you can replace this with your own robot, such as one you have optimized using an evolutionary algorithm.
    """
    if args.body.endswith("45"):
        args.body = args.body[:-2]
        args.rotated = True
    body = BODIES[args.body]()
    hinges = body.find_modules_of_type(ActiveHinge)

    w, h = matplotlib.rcParams["figure.figsize"]
    fig, axes = plt.subplots(len(hinges), 2,
                             sharex=True, sharey=True,
                             figsize=(3 * w, 2 * h))

    plot_data = [[] for _ in range(1+2*len(hinges))]
    def plot_callback(time, observations, actions):
        for i, x in enumerate(itertools.chain([time], observations, actions)):
            plot_data[i].append(x)

    # brain = BrainCpgNetworkNeighborRandom(body=body, rng=make_rng_time_seed())
    if debugging:
        brain = HingePinDebuggingBrainFactory(
            hinges=hinges, index=args.debug_hinge, d_type=args.debug_type)
    elif args.model is not None:
        brain = RLBrainFactory(args.model, hinges, plot_callback)

    robot = ModularRobot(body, brain)

    """
    Some important notes to understand:
    - Hinge mappings are specific to each robot, so they have to be created new for each type of body. 
    - The pin`s id`s can be found on th physical robots HAT.
    - The order of the pin`s is crucial for a correct translation into the physical robot.
    - Each ActiveHinge needs one corresponding pin to be able to move. 
    - If the mapping is faulty check the simulators behavior versus the physical behavior and adjust the mapping iteratively.
    
    For a concrete implementation look at the following example of mapping the robots`s hinges:
    """
    h1, h2, h3, h4, h5, h6, h7, h8 = hinges
    hinge_mapping = {
        UUIDKey(h1): 0,
        UUIDKey(h2): 1,
        # UUIDKey(h3): 15,
        # UUIDKey(h4): 14,
        # UUIDKey(h5): 16,
        # UUIDKey(h6): 17,
        UUIDKey(h3): 16,
        UUIDKey(h4): 17,
        UUIDKey(h5): 15,
        UUIDKey(h6): 14,
        UUIDKey(h7): 31,
        UUIDKey(h8): 30,
    }

    def on_prepared() -> None:
        if debugging:
            print("Ready to run. Doing so now")
        else:
            print("Done. Press enter to start the brain.")
            input()

    """
    A configuration consists of the follow parameters:
    - modular_robot: The ModularRobot object, exactly as you would use it in simulation.
    - hinge_mapping: This maps active hinges to GPIO pins on the physical modular robot core.
    - run_duration: How long to run the robot for in seconds.
    - control_frequency: Frequency at which to call the brain control functions in seconds. If you also ran the robot in simulation, this must match your setting there.
    - initial_hinge_positions: Initial positions for the active hinges. In Revolve2 the simulator defaults to 0.0.
    - inverse_servos: Sometimes servos on the physical robot are mounted backwards by accident. Here you inverse specific servos in software. Example: {13: True} would inverse the servo connected to GPIO pin 13.
    """
    config = Config(
        modular_robot=robot,
        hinge_mapping=hinge_mapping,
        run_duration=5 if debugging else args.duration,
        control_frequency=20,
        initial_hinge_positions={UUIDKey(active_hinge): 0.0 for active_hinge in hinges},
        inverse_servos={
            hinge_mapping[UUIDKey(h)]: True
            for h in [h2, h4, h6, h8]
        },
    )

    """
    Create a Remote for the physical modular robot.
    Make sure to target the correct hardware type and fill in the correct IP and credentials.
    The debug flag is turned on. If the remote complains it cannot keep up, turning off debugging might improve performance.
    If you want to display the camera view, set display_camera_view to True.
    """
    print("Initializing robot..")
    run_remote(
        config=config,
        hostname=args.ip,  # "Set the robot IP here.
        debug=False,
        on_prepared=on_prepared,
        display_camera_view=False,
        manual_mode=False
    )
    """
    Note that theoretically if you want the robot to be self controlled and not dependant on a external remote, you can run this script on the robot locally.
    """

    print("Plotting")
    for i in range(len(hinges)):
        for j in range(2):
            axes[i][j].plot(plot_data[0], plot_data[1+j*len(hinges)+i])
    fig.tight_layout()
    fig.savefig("hinges.pdf")


if __name__ == "__main__":
    main()
