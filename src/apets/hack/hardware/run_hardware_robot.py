"""An example on how to remote control a physical modular robot."""
import argparse
import itertools
import math
import os
import pickle
import pprint
import random
import subprocess
import time
from enum import Enum, auto
from pathlib import Path
from typing import Tuple, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from torch import nn

from apets.hack.config import Config
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import Body, ActiveHinge
from revolve2.modular_robot.body.v2 import BodyV2, ActiveHingeV2, BrickV2
from revolve2.modular_robot.brain import Brain as BrainFactory
from revolve2.modular_robot.brain import BrainInstance
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic
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


def spider_v2() -> Tuple[BodyV2, List[ActiveHinge]]:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = BodyV2()
    hinges = list()

    def hinge(angle, name):
        h = ActiveHingeV2(angle)
        h.name = name
        hinges.append(h)
        return h

    body.core_v2.front_face.bottom = hinge(np.pi / 2.0, "FRH")
    body.core_v2.front_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment.front = hinge(0.0, "FRK")
    body.core_v2.front_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = hinge(np.pi / 2.0, "BLH")
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = hinge(0.0, "BLK")
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.left_face.bottom = hinge(np.pi / 2.0, "FLH")
    body.core_v2.left_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front = hinge(0.0, "FLK")
    body.core_v2.left_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = hinge(np.pi / 2.0, "BRH")
    body.core_v2.right_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.front = hinge(0.0, "BRK")
    body.core_v2.right_face.bottom.attachment.front.attachment = BrickV2(0.0)

    return body, hinges


BODIES = {
    "gecko": gecko_v2,
    "spider": spider_v2,
}


class DummyBrain(BrainFactory):
    def make_instance(self) -> BrainInstance:
        return self.DummyBrainInstance()

    class DummyBrainInstance(BrainInstance):
        def control(
            self,
            dt: float,
            sensor_state: ModularRobotSensorState,
            control_interface: ModularRobotControlInterface,
        ) -> None:
            pass


class HingePinDebuggingBrainFactory(BrainFactory):
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

    def __init__(self, hinges, args):
        if args.debug_hinge is not None:
            try:
                hinge = hinges[int(args.debug_hinge)]
            except ValueError:
                hinge = {h.name: h for h in hinges}[args.debug_hinge]

            self._target_hinge = lambda _: hinge

        elif args.debug_hinges:
            n = len(hinges)
            self._target_hinge = lambda t: hinges[min(n, int(n * t / args.duration))]

        d_type = args.debug_type
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
            self._last_hinge = None

        def control(
            self,
            dt: float,
            sensor_state: ModularRobotSensorState,
            control_interface: ModularRobotControlInterface,
        ) -> None:
            self._time += dt
            hinge = self._target_hinge(self._time)
            if self._last_hinge is not None and self._last_hinge != hinge:
                control_interface.set_active_hinge_target(self._last_hinge, 0)

            target = self._function(self._time) * hinge.range
            print(self._time, sensor_state.get_active_hinge_sensor_state(hinge.sensors.active_hinge_sensor).position, ">>", target)
            control_interface.set_active_hinge_target(hinge, target)
            self._last_hinge = hinge


class RLBrainFactory(BrainFactory):
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
            print(self._time, dt)
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


class RobotControllerTrackerFactory(BrainFactory):
    def __init__(self, brain: BrainFactory, hinges: List[ActiveHinge]):
        self.brain = brain
        self.hinges = hinges

        names = [h.name for h in hinges]
        print("[kgd-debug] hinges:", names)

        self.plot_header = ["time"] + [f"{n}-obs" for n in names] + [f"{n}-ctrl" for n in names]
        self.plot_data = [[] for _ in range(1 + 2 * len(hinges))]

    def make_instance(self) -> BrainInstance:
        return self.Instance(self.brain.make_instance(), self.hinges, self.plot_data)

    class Instance(BrainInstance):
        def __init__(self, brain: BrainInstance, hinges, plot_data):
            self._brain = brain
            self._hinges = hinges
            self._plot_data = plot_data
            self._time = 0

            # Debug variable
            self.single_hinge = os.environ.get("DEBUG_HINGE", None)
            self.single_hinge = int(self.single_hinge) if self.single_hinge is not None else None

            print("[kgd-debug] hinges:", [h.name for h in self._hinges])

        def control(
            self,
            dt: float,
            sensor_state: ModularRobotSensorState,
            control_interface: ModularRobotControlInterface,
        ) -> None:
            self._time += dt
            print(f"[kgd-debug] {self._time}, {dt}")
            print("[kgd-debug] hinges:", [(i, h.name) for i, h in enumerate(self._hinges)])
            obs = [
                sensor_state.get_active_hinge_sensor_state(hinge.sensors.active_hinge_sensor).position
                / hinge.range
                for i, hinge in enumerate(self._hinges)
            ]

            self._brain.control(dt, sensor_state, control_interface)

            if self.single_hinge is not None:
                control_interface._set_active_hinges = [
                    (key, value) for key, value in control_interface._set_active_hinges
                    if key == UUIDKey(self._hinges[self.single_hinge])
                ]

            hinges_dict = dict(control_interface._set_active_hinges)
            actions = [hinges_dict.get(UUIDKey(h), float("nan")) for h in self._hinges]
            # print(f"[kgd-debug] {actions=}")

            # print("[kgd-debug] randomized action buffer")
            # random.Random(0).shuffle(actions)

            for _i, x in enumerate(itertools.chain([self._time], obs, actions)):
                self._plot_data[_i].append(x)
            print("[kgd-debug] last plot data line:", [f"{d[-1]:.2f}" for d in self._plot_data])


def main() -> None:
    parser = argparse.ArgumentParser("Main entry point for robot hardware control")
    parser.add_argument("--ip", required=True, help="ip address of the robot")
    parser.add_argument("--body", required=True, help="robot morphology")
    parser.add_argument("--rotated", default=False, help="whether the robot is front-facing or rotated by 45 degrees")
    parser.add_argument("--duration", default=15, type=float, help="how long to run for (in seconds)")

    group = parser.add_mutually_exclusive_group()

    group.add_argument("--debug-hinge", default=None,
                       help="Index of the hinge to debug")
    group.add_argument("--debug-hinges", default=False, action="store_true",
                       help="Debug all hinges in sequence")
    parser.add_argument("--debug-type", choices=[e.name for e in HingePinDebuggingBrainFactory.DebugTypes],
                        help="Specify the type of function used to send signals")
    parser.add_argument("--no-plots", default=False, action="store_true",
                        help="Disables plotting (e.g. for many sequential tests)")

    group.add_argument("--brain", type=Path, help="Pre-processed controller for the robot. See `extract_controller.py`")
    group.add_argument("--reset", default=False, action="store_true", help="Reset hinges to 0")

    args = parser.parse_args()

    debugging = args.debug_hinge is not None or args.debug_hinges

    """Remote control a physical modular robot."""
    """
    Create a modular robot, similar to what was done in the 1a_simulate_single_robot example.
    Of course, you can replace this with your own robot, such as one you have optimized using an evolutionary algorithm.
    """
    if args.body.endswith("45"):
        args.body = args.body[:-2]
        args.rotated = True
    body, hinges = BODIES[args.body]()
    hinge_names = {h.name: h for h in hinges}

    # brain = BrainCpgNetworkNeighborRandom(body=body, rng=make_rng_time_seed())
    if args.reset:
        brain = DummyBrain()
    elif debugging:
        brain = HingePinDebuggingBrainFactory(hinges=hinges, args=args)
    elif args.brain is not None:
        # brain = RLBrainFactory(args.model, hinges, plot_callback)
        with open(args.brain, "rb") as f:
            brain = pickle.load(f)

            hinges_order = ['FRH', 'FRK', 'BLH', 'BLK', 'BRH', 'BRK', 'FLH', 'FLK']

            if isinstance(brain, BrainCpgNetworkStatic):
                # hinges[4:6], hinges[6:8] = hinges[6:8], hinges[4:6]
                # print("[kgd-debug] randomized hinges in _output_mapping")
                # random.Random(1).shuffle(hinges)
                brain._output_mapping = [(i, hinge_names[h_name]) for i, h_name in enumerate(hinges_order)]
                pprint.pprint([(i, h.name) for i, h in brain._output_mapping])

    brain = RobotControllerTrackerFactory(brain, hinges)

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
    assert args.body == "spider"
    hinge_mapping = {
        UUIDKey(hinge_names[name]): pin
        for name, pin in [
            # Verified mapping (not working?!? inverted?)
            # ("FRH", 0), ("FRK", 1),
            # ("BLH", 16), ("BLK", 17),
            # ("FLH", 15), ("FLK", 14),
            # ("BRH", 31), ("BRK", 30),

            # Reverse mapping (to check)
            # ("FRH", 16), ("FRK", 17),
            # ("BLH", 0), ("BLK", 1),
            # ("FLH", 31), ("FLK", 30),
            # ("BRH", 15), ("BRK", 14),

            # Has kind worked (but with a 90 rotation)
            # ("FLH", 0), ("FLK", 1),
            # ("BRH", 16), ("BRK", 17),
            # ("FRH", 15), ("FRK", 14),
            # ("BLH", 31), ("BLK", 30),

            # Has kind of worked. A bit too much of a far-righter though.
            # ("FLH", 31), ("FLK", 30),
            # ("FRH", 0), ("FRK", 1),
            # ("BRH", 15), ("BRK", 14),
            # ("BLH", 16), ("BLK", 17),

            # Tested with name hinges
            ("FLH", 15), ("FLK", 14),
            ("FRH", 0), ("FRK", 1),
            ("BRH", 31), ("BRK", 30),
            ("BLH", 16), ("BLK", 17),
        ]
    }

    def on_prepared() -> None:
        if args.reset:
            print("Resetting hinges")
        elif debugging:
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
    if args.reset:
        duration = 0
    elif args.debug_hinge is not None:
        duration = 5
    elif args.debug_hinges:
        duration = 5 * len(hinges)
    else:
        duration = args.duration
    config = Config(
        modular_robot=robot,
        hinge_mapping=hinge_mapping,
        run_duration=duration,
        control_frequency=20,
        initial_hinge_positions={UUIDKey(h): 0.0 for h in hinges},
        inverse_servos={},
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
        hostname=args.ip,
        debug=False,
        on_prepared=on_prepared,
        display_camera_view=False,
        manual_mode=False
    )
    """
    Note that theoretically if you want the robot to be self controlled and not dependant on a external remote, you can run this script on the robot locally.
    """

    if args.reset or args.no_plots:
        return

    print("Plotting")
    w, h = matplotlib.rcParams["figure.figsize"]
    fig, axes = plt.subplots(len(hinges), 2,
                             sharex=True, sharey=True,
                             figsize=(3 * w, 2 * h))

    plot_data = brain.plot_data

    try:
        simu_data: Optional[pd.DataFrame] = brain.brain.simu_data
    except AttributeError:
        simu_data = None

    x = np.array(plot_data[0])
    for i in range(len(hinges)):
        for j in range(2):
            ax = axes[i][j]
            ix = 1 + j * len(hinges) + i

            art_hard, = ax.plot(x, plot_data[ix], zorder=1)
            title = brain.plot_header[ix]

            if simu_data is not None:
                art_soft, = ax.plot(simu_data.index, simu_data.iloc[:, ix-1], zorder=-1)
                title += " | " + simu_data.columns[ix-1]

            ax.set_ylim(-1.2, 1.2)
            # ax.set_xlim(-1, 16)

            ax.set_title(title)

    if simu_data is not None:
        fig.legend(handles=[art_soft, art_hard], labels=["Simulation", "Real-world"], loc="upper center",
                   ncols=2, title=time.ctime())

    fig.tight_layout()
    fig.savefig("hinges.pdf", bbox_inches="tight")


if __name__ == "__main__":
    print("Start")
    main()
