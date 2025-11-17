"""An example on how to remote control a physical modular robot."""
import argparse
import itertools
import math
import os
import pickle
import random
import time
from enum import Enum, auto
from pathlib import Path
from typing import Tuple, List, Optional

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from apets.hack.tensor_brain import TensorBrainFactory
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.v2 import BodyV2, ActiveHingeV2, BrickV2
from revolve2.modular_robot.brain import Brain as BrainFactory
from revolve2.modular_robot.brain import BrainInstance
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, BrainCpgInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot_physical import Config, UUIDKey
from revolve2.modular_robot_physical.remote import run_remote

# wifi = subprocess.check_output("iwgetid -r".split()).decode()
# assert "ThymioNet" in wifi, f"WRONG NETWORK: {wifi}"


def gecko_v2() -> Tuple[BodyV2, List[ActiveHinge]]:
    """
    Sample robot with new HW config.

    :returns: the robot
    """
    body = BodyV2()
    hinges = list()

    def hinge(angle, name, mujoco_postfix):
        h = ActiveHingeV2(angle)
        h.name = name
        h.mujoco_name = f"mbs1/actuator_position_mbs1_{mujoco_postfix}"
        hinges.append(h)
        return h

    body.core_v2.right_face.bottom = hinge(0.0, "FR", "joint2")
    body.core_v2.right_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.left_face.bottom = hinge(0.0, "FL", "joint1")
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = hinge(np.pi / 2.0, "SS", "joint0")
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = hinge(np.pi / 2.0, "SE", "link0_joint1")
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left = hinge(0.0, "BL", "link0_link1_joint2")
    body.core_v2.back_face.bottom.attachment.front.attachment.right = hinge(0.0, "BR", "link0_link1_joint1")
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = BrickV2(0.0)

    return body, hinges


def spider_v2() -> Tuple[BodyV2, List[ActiveHinge]]:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = BodyV2()
    hinges = list()

    def hinge(angle, name, mujoco_postfix):
        h = ActiveHingeV2(angle)
        h.name = name
        h.mujoco_name = f"mbs1/actuator_position_mbs1_{mujoco_postfix}"
        hinges.append(h)
        return h

    body.core_v2.front_face.bottom = hinge(np.pi / 2.0, "FRH", "joint0")
    body.core_v2.front_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment.front = hinge(0.0, "FRK", "link0_joint1")
    body.core_v2.front_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = hinge(np.pi / 2.0, "BLH", "joint1")
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = hinge(0.0, "BLK", "link1_joint1")
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.left_face.bottom = hinge(np.pi / 2.0, "FLH", "joint2")
    body.core_v2.left_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front = hinge(0.0, "FLK", "link2_joint1")
    body.core_v2.left_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = hinge(np.pi / 2.0, "BRH", "joint3")
    body.core_v2.right_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.front = hinge(0.0, "BRK", "link3_joint1")
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
            self._target_hinge = lambda t: hinges[min(n-1, int(n * t / args.duration))]

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

#
# class RLBrainFactory(BrainFactory):
#     def __init__(self, path: Path, hinges, callback):
#         self._model = PPO.load(path, device="cpu")
#         self._hinges = hinges
#         self._callback = callback
#
#     def make_instance(self) -> BrainInstance:
#         return self.RLBrainInstance(self._model, self._hinges, self._callback)
#
#     class RLBrainInstance(BrainInstance):
#         def __init__(self, model, hinges, callback):
#             self._model = model
#             self._hinges = hinges
#             self._callback = callback
#             self._time = 0
#
#         def control(
#             self,
#             dt: float,
#             sensor_state: ModularRobotSensorState,
#             control_interface: ModularRobotControlInterface,
#         ) -> None:
#             self._time += dt
#             print(self._time, dt)
#             obs = np.clip([
#                 sensor_state.get_active_hinge_sensor_state(hinge.sensors.active_hinge_sensor).position
#                 / hinge.range
#                 for i, hinge in enumerate(self._hinges)
#             ], -1, 1)
#
#             actions, _states = self._model.predict(obs, deterministic=True)
#             self._callback(time=self._time, observations=obs, actions=actions)
#
#             for action, hinge in zip(actions, self._hinges):
#                 assert -1 <= action <= 1
#                 control_interface.set_active_hinge_target(
#                     hinge, float(action * hinge.range)
#                 )
#


class RobotControllerTrackerFactory(BrainFactory):
    def __init__(self, brain_factory: BrainFactory, hinges: List[ActiveHinge]):
        self.brain_factory = brain_factory
        self.hinges = hinges

        names = [h.name for h in hinges]
        # print("[kgd-debug] hinges:", names)

        self.plot_header = ["time"] + [f"{n}-obs" for n in names] + [f"{n}-ctrl" for n in names]
        self.plot_data = [[] for _ in range(1 + 2 * len(hinges))]

    def make_instance(self) -> BrainInstance:
        return self.Instance(self.brain_factory.make_instance(), self.hinges, self.plot_data)

    class Instance(BrainInstance):
        def __init__(self, brain_instance: BrainInstance, hinges, plot_data):
            self._brain_instance = brain_instance
            self._hinges = hinges
            self._plot_data = plot_data
            self._time = 0

            self._names = {h.name: h for h in self._hinges}

            # Debug variable
            self.single_hinge = os.environ.get("DEBUG_HINGE", None)
            if self.single_hinge is not None:
                try:
                    self.single_hinge = int(self.single_hinge)
                except ValueError:
                    self.single_hinge = self._hinges.index(self._names[self.single_hinge])

            # print("[kgd-debug|Monitor::__init__] hinges:", self._names.keys())

        def control(
            self,
            dt: float,
            sensor_state: ModularRobotSensorState,
            control_interface: ModularRobotControlInterface,
        ) -> None:
            # print(f"[kgd-debug|Monitor::control] {self._time}, {dt}")
            # print("[kgd-debug|Monitor::control] hinges:", [(i, h.name) for i, h in enumerate(self._hinges)])

            if (sense_fn := getattr(self._brain_instance, "_get_observation", None)) is not None:
                obs = sense_fn(sensor_state)
            else:
                obs = [
                    sensor_state.get_active_hinge_sensor_state(hinge.sensors.active_hinge_sensor).position
                    for i, hinge in enumerate(self._hinges)
                ]

            self._brain_instance.control(dt, sensor_state, control_interface)

            if self.single_hinge is not None:
                control_interface._set_active_hinges = [
                    (key, value) for key, value in control_interface._set_active_hinges
                    if key == UUIDKey(self._hinges[self.single_hinge])
                ]
                print([
                    (key.value.name, key.value.mujoco_name, value) for key, value in
                    control_interface._set_active_hinges
                ])

            hinges_dict = dict(control_interface._set_active_hinges)
            actions = [hinges_dict.get(UUIDKey(h), float("nan")) for h in self._hinges]
            # print(f"[kgd-debug] {actions=}")

            # print("[kgd-debug] randomized action buffer")
            # random.Random(0).shuffle(actions)

            for _i, x in enumerate(itertools.chain([self._time], obs, actions)):
                self._plot_data[_i].append(x)
            # print("[kgd-debug] last plot data line:", [f"{d[-1]:.2f}" for d in self._plot_data])

            if False:
                control_interface._set_active_hinges = []

            self._time += dt


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

    group.add_argument("--run-when-ready", default=True,
                       dest='start_paused', action="store_false",
                       help="Whether to request user input before running")

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
        with open(args.brain, "rb") as f:
            brain = pickle.load(f)
            print(brain)

            if isinstance(brain, BrainCpgNetworkStatic):
                if args.body == "spider":
                    hinges_order = ['FRH', 'FRK', 'BLH', 'BLK', 'BRH', 'BRK', 'FLH', 'FLK']
                elif args.body == "gecko":
                    hinges_order = ["SS", "SE", "BR", "BL", "FR", "FL"]
                    # hinges_order = ["FR", "FL", "SS", "SE", "BR", "BL"]
                else:
                    raise RuntimeError(f"Unsupported body type '{args.body}'")

                brain._output_mapping = [(i, hinge_names[h_name]) for i, h_name in enumerate(hinges_order)]
                print("CPG output mapping", [(i, h.name) for i, h in brain._output_mapping])

            elif isinstance(brain, TensorBrainFactory):
                if args.body == "spider":
                    hinges_order = ['FRH', 'FRK', 'BLH', 'BLK', 'BRH', 'BRK', 'FLH', 'FLK']
                elif args.body == "gecko":
                    raise NotImplementedError
                else:
                    raise RuntimeError(f"Unsupported body type '{args.body}'")

                brain._hinges = [hinge_names[h_name] for h_name in hinges_order]
                brain.set_total_recall(True)

            else:
                raise NotImplementedError

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
    inverse_servos = {}
    if args.body == "spider":
        hinge_mapping = {
            UUIDKey(hinge_names[name]): pin
            for name, pin in [
                ("FLH", 15), ("FLK", 14),
                ("FRH", 0), ("FRK", 1),
                ("BRH", 31), ("BRK", 30),
                ("BLH", 16), ("BLK", 17),
            ]
        }

    elif args.body == "gecko":
        hinge_mapping = {
            UUIDKey(hinge_names[name]): pin
            for name, pin in [
                ("FL", 0), ("FR", 31),
                ("SS", 2), ("SE", 29),
                ("BL", 1), ("BR", 30),
            ]
        }
        inverse_servos = {
            hinge_mapping[UUIDKey(hinge_names[name])]: True
            for name in ["SS", "SE", "BR", "FR", "FL"]
        }
        print(inverse_servos)
        # hinge_mapping = {
        #     UUIDKey(hinge_names[name]): pin
        #     for name, pin in [
        #         ("FL", 16), ("FR", 15),
        #         ("SS", 1), ("SE", 13),
        #         ("BL", 17), ("BR", 14),
        #     ]
        # }

    else:
        raise RuntimeError

    def on_prepared() -> None:
        if args.reset:
            print("Resetting hinges")
        elif debugging:
            print("Ready to run. Doing so now")
        else:
            if args.start_paused:
                print("Done. Press enter to start the brain.")
                _in = input()
                print(_in, len(_in))
                if _in[:2].lower() != "go" and len(_in) > 0:
                    print("Exiting.")
                    exit(2)
        time.sleep(.5)

    """
    A configuration consists of the follow parameters:
    - modular_robot: The ModularRobot object, exactly as you would use it in simulation.
    - hinge_mapping: This maps active hinges to GPIO pins on the physical modular robot core.
    - run_duration: How long to run the robot for in seconds.
    - control_frequency: Frequency at which to call the brain control functions in seconds. If you also ran the robot in simulation, this must match your setting there.
    - initial_hinge_positions: Initial positions for the active hinges. In Revolve2 the simulator defaults to 0.0.
    - inverse_servos: Sometimes servos on the physical robot are mounted backwards by accident. Here you inverse specific servos in software. Example: {13: True} would inverse the servo connected to GPIO pin 13.
    """
    plot_filename = "hinges"
    if args.reset:
        args.duration = .5
    elif args.debug_hinge is not None:
        args.duration = 5
        plot_filename = f"debug_hinge_{args.debug_hinge}"
    elif args.debug_hinges:
        args.duration = 5 * len(hinges)
        plot_filename = f"debug_hinges"
    config = Config(
        modular_robot=robot,
        hinge_mapping=hinge_mapping,
        run_duration=args.duration,
        control_frequency=20,
        initial_hinge_positions={UUIDKey(h): 0.0 for h in hinges},
        inverse_servos=inverse_servos,
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

    filename = Path(os.environ.get("PLOT_NAME", Path(args.brain or plot_filename).with_suffix(".pdf")))
    print("Plotting", filename)

    plot_data = brain.plot_data

    plot_data = pd.DataFrame({header: data for header, data in zip(brain.plot_header, plot_data)}).set_index("time")
    plot_data.to_csv(filename.with_suffix(".csv"))

    w, h = matplotlib.rcParams["figure.figsize"]
    fig, axes = plt.subplots(len(hinges), 2,
                             sharex=True, sharey=True,
                             figsize=(3 * w, 2 * h))

    x = plot_data.index
    samples = len(x)

    try:
        simu_data: Optional[pd.DataFrame] = brain.brain_factory.simu_data
        print(simu_data.shape)
        print(simu_data.columns)
        samples = min(samples, len(simu_data.index))
        sx = simu_data.index[:samples]

    except AttributeError:
        simu_data = None
    print(plot_data.columns)
    print(f"{samples=}")

    x = plot_data.index[:samples]
    for i in range(len(hinges)):
        for j, ptype in enumerate(["pos", "ctrl"]):
            ax = axes[i][j]
            ax.grid()
            ix = j * len(hinges) + i

            c = plot_data.columns[ix]
            art_hard, = ax.plot(x, plot_data[c].head(samples), zorder=1)
            title = c

            if simu_data is not None:
                sc = f"{hinges[i].mujoco_name}-{ptype}"
                art_soft, = ax.plot(sx, simu_data[sc].head(samples), zorder=-1)
                title += " | " + sc

            ax.set_title(title)

    if simu_data is not None:
        fig.legend(handles=[art_soft, art_hard], labels=["Simulation", "Real-world"], loc="upper center",
                   ncols=2, title=time.ctime())

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    print("Start")
    main()
