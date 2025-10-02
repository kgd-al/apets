import argparse
import time
from pathlib import Path

import numpy as np
import threading

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TABLEAU_COLORS
from scipy.spatial.transform import Rotation as R

import natnet
import logging
from natnet.protocol.MocapFrameMessage import LabelledMarker, RigidBody
from natnet.comms import TimestampAndLatency


class RealSensor:
    """Real sensor interface for OptiTrack motion capture system."""

    def __init__(self, args):
        self.ip_address = args.server
        self.robot_id = args.robot
        self.verbose = args.verbose

        # Data handler for NatNet callbacks
        self.data_handler = NatNetDataHandler(args)

        # Thread for running NatNet client
        self.natnet_thread = None
        self.is_running = False

        # Setup NatNet logger
        self.natnet_logger = logging.getLogger('natnet')
        self.natnet_logger.setLevel(logging.INFO)  # Reverted to INFO
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # Reverted to INFO
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.natnet_logger.addHandler(ch)

    def start(self):
        """Start the NatNet client in a separate thread."""
        if self.is_running:
            print("[RealSensor] Sensor is already running.")
            return

        print(f"[RealSensor] Starting OptiTrack sensor on {self.ip_address}...")
        self.is_running = True
        self.natnet_thread = threading.Thread(
            target=self._run_natnet_client_in_thread,
            daemon=True
        )
        self.natnet_thread.start()

    def stop(self):
        """Stop the NatNet client."""
        print("[RealSensor] Stopping OptiTrack sensor...")
        self.is_running = False
        if self.natnet_thread:
            self.natnet_thread.join(timeout=2.0)

    def get_relative_pose(self):
        """Get the latest relative pose of end-effector with respect to table."""
        with self.data_handler.data_lock:
            return self.data_handler.latest_relative_pos, self.data_handler.latest_relative_quat

    def is_connected(self):
        """Check if the sensor is connected to OptiTrack."""
        return self.data_handler.is_connected

    def data_available(self):
        """Check if new data has been received."""
        return self.data_handler.data_received_flag

    def _run_natnet_client_in_thread(self):
        """Run the NatNet client in the thread."""
        print(f'[RealSensor] Attempting to connect to {self.ip_address}...')
        print(f'[RealSensor] NatNet client connection timeout set to 10 seconds.')
        try:
            client = natnet.Client.connect(self.ip_address, logger=self.natnet_logger, timeout=10)
            print('[RealSensor] Client connected successfully.')
            print(f'[RealSensor] NatNet client details: {client}')
            self.data_handler.is_connected = True

            # Create callback that uses the configured rigid body IDs
            def callback(rigid_bodies, markers, timing):
                self.data_handler.natnet_callback(rigid_bodies, markers, timing, self.robot_id)

            client.set_callback(callback)
            print('[RealSensor] Callback set. Starting data acquisition loop (blocking).')

            client.spin()  # This will block until the client is stopped or an error occurs

        except natnet.DiscoveryError as e:
            print(f"[RealSensor] Error: Failed to connect to {self.ip_address}: {e}")
            print(
                "[RealSensor] Ensure Motive/OptiTrack software is running, NatNet streaming is enabled, and IP/firewall settings are correct.")
        except Exception as e:
            print(f"\n[RealSensor] An unexpected error occurred: {e}")
        finally:
            print("[RealSensor] Client thread exiting.")
            self.data_handler.is_connected = False


class NatNetDataHandler:
    """Data handler for processing NatNet data."""

    def __init__(self, args):
        self.latest_rigid_bodies = []
        self.latest_markers = []
        self.latest_timing = None
        self.latest_relative_pos = None
        self.latest_relative_quat = None
        self.data_received_flag = False
        self.verbose = args.verbose
        self.is_connected = False
        self.data_lock = threading.Lock()

        self.data_columns = ["t", "x", "y", "z", "roll", "pitch", "yaw"]
        self.data = [[] for _ in range(len(self.data_columns))]

        self.initial_pos, self.initial_quat = None, None

    def natnet_callback(self, rigid_bodies: list[RigidBody], markers: list[LabelledMarker],
                        timing: TimestampAndLatency, target_id: int):
        """
        Callback function to handle incoming data from NatNet.
        """
        # Update raw data, protected by lock for thread-safety
        with self.data_lock:
            self.latest_rigid_bodies = rigid_bodies
            self.latest_markers = markers
            self.latest_timing = timing
            self.data_received_flag = True

        if self.verbose:
            print(f"[RealSensor] Callback triggered! Markers count: {len(markers)}, Timestamp: {timing.timestamp:.4f}s")

        target_rb = None

        # Find the specific rigid bodies by their IDs
        if self.verbose:
            print(f"[RealSensor] Rigid Bodies:", [rb.id_ for rb in self.latest_rigid_bodies])
        for rb in rigid_bodies:
            if rb.id_ == target_id:
                target_rb = rb
                break

        if target_rb and target_rb.tracking_valid:
            # Get positions (x, y, z)
            pos = np.array(target_rb.position)

            # Get orientations (qx, qy, qz, qw) - NatNet uses (x, y, z, w) order
            quat = R.from_quat(target_rb.orientation)

            if self.initial_pos is None:
                self.initial_pos = pos

            if self.initial_quat is None:
                self.initial_quat = quat

            # 1. Calculate Relative Position
            relative_pos = self.initial_quat.inv().apply(self.initial_pos - pos)

            # 2. Calculate Relative Orientation
            relative_rot = self.initial_quat.inv() * quat
            relative_quat = relative_rot.as_quat()  # Get the quaternion (x, y, z, w)
            relative_euler = relative_rot.as_euler('xyz', degrees=True)

            if self.verbose:
                print(f"[RealSensor] \n--- Relative Pose (Timestamp: {timing.timestamp:.4f}s) ---")
                print(f"[RealSensor] Target ({target_id}):  ({'valid' if target_rb.tracking_valid else 'invalid'})")
                print(
                    f"[RealSensor] Position (x,y,z): {relative_pos[0]:.4f}, {relative_pos[1]:.4f}, {relative_pos[2]:.4f} meters")
                print(
                    f"[RealSensor] Orientation (Quat x,y,z,w): {relative_quat[0]:.4f}, {relative_quat[1]:.4f}, {relative_quat[2]:.4f}, {relative_quat[3]:.4f}")
                print(
                    f"[RealSensor] Orientation (Euler deg Roll,Pitch,Yaw): {relative_euler[0]:.4f}, {relative_euler[1]:.4f}, {relative_euler[2]:.4f} degrees")
            else:
                # print(time.time(), timing.timestamp, timing)
                print(".", end='', flush=True)

            # Update relative pose data
            with self.data_lock:
                for i, x in enumerate([
                    time.time(),
                    *relative_pos,
                    *relative_euler
                ]):
                    self.data[i].append(x)

                self.latest_relative_pos = relative_pos
                self.latest_relative_quat = relative_quat

        elif self.verbose:
            if not target_rb:
                print(f"[RealSensor] Warning: Target RB with ID {target_id} not found in this frame.")
            print("[RealSensor] Cannot calculate relative pose without rigid body.")


def main(args):
    assert not args.live, "Live mode not really working, sorry"

    sensor = RealSensor(args)
    sensor.start()

    start = time.time()
    while time.time() - start < 10 and not sensor.is_connected():
        time.sleep(0.1)

    # Wait for actual data frames to start flowing. This is crucial for valid observations.
    data_flow_start = time.time()
    max_wait_for_data = 5.0  # Max wait for first frame after connection
    while not sensor.data_available() and (time.time() - data_flow_start < max_wait_for_data):
        time.sleep(0.1)
    if not sensor.data_available():
        print("[RL Env] Warning: NatNet client connected but no data received within timeout. Check OptiTrack streaming and rigid body visibility.")
    else:
        print("[RL Env] NatNet client connected and initial data received.")

    if args.live:
        plt.ion()
        data = sensor.data_handler.data
        names = sensor.data_handler.data_columns
        color = list(TABLEAU_COLORS.values())[0]
        fig, ax = plt.subplots(nrows=2, ncols=3)

        window = 100

        fig.show()

        def step():
            if len(data[0]) == 0:
                return

            with sensor.data_handler.data_lock:
                print(data)
                t = np.array(data[0]) - data[0][0]
                for _ax, (i, column) in zip(ax.flat, enumerate(names[1:])):
                    _ax.plot(t[-window:], data[i+1][-window:], color=color)
                    _ax.set_title(column)
                fig.tight_layout()
                plt.pause(.1)

    else:
        def step():
            time.sleep(0.1)

    run = True
    while run:
        try:
            step()

        except KeyboardInterrupt:
            run = False
            print("Stopping")

    sensor.stop()

    if not args.live:
        folder = args.output
        data = pd.DataFrame(np.array(sensor.data_handler.data).T,
                            columns=sensor.data_handler.data_columns)
        print(data)
        data["t"] -= data["t"][0]
        data.to_csv(folder.joinpath("data.csv"))

        if not folder.exists():
            folder.mkdir(parents=True)

        with PdfPages(folder.joinpath("plots.pdf")) as pdf:
            fig, ax = plt.subplots(nrows=2, ncols=3)
            for ax, column in zip(ax.flat, data.columns[1:]):
                ax.plot(data["t"], data[column])
                ax.set_title(column)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(data["x"], data["y"])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from NatNet.")
    parser.add_argument("--server", default="192.168.1.2",
                        help="Which server to connect to.")
    parser.add_argument("--robot", default=4, type=int,
                        help="Which robot (rigid body) to track")
    parser.add_argument("--live", default=False, action="store_true",
                        help="Plot data live (default just makes a file)")
    parser.add_argument("--output", default="optitrack", type=Path,
                        help="Folder name for output files")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Whether to talk to much")

    main(parser.parse_args())
