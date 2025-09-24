import time
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R

import natnet
import logging
from natnet.protocol.MocapFrameMessage import LabelledMarker, RigidBody
from natnet.comms import TimestampAndLatency


class RealSensor:
    """Real sensor interface for OptiTrack motion capture system."""

    def __init__(self, ip_address="127.0.0.1", robot_id=11, verbose=True):
        self.ip_address = ip_address
        self.robot_id = robot_id
        self.verbose = verbose

        # Data handler for NatNet callbacks
        self.data_handler = NatNetDataHandler(verbose=verbose)

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
        with self.data_handler._data_lock:
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


class RealSensorWithoutLock:
    """Real sensor interface for OptiTrack motion capture system."""

    def __init__(self, ip_address="127.0.0.1", end_effector_rb_id=12, table_rb_id=11, verbose=True):
        self.ip_address = ip_address
        self.end_effector_rb_id = end_effector_rb_id
        self.table_rb_id = table_rb_id
        self.verbose = verbose

        # Data handler for NatNet callbacks
        self.data_handler = NatNetDataHandlerWithoutLock(verbose=verbose)

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
        # with self.data_handler._data_lock:
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
                self.data_handler.natnet_callback(rigid_bodies, markers, timing,
                                                  self.end_effector_rb_id, self.table_rb_id)

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

    def __init__(self, verbose: bool = True):
        self.latest_rigid_bodies = []
        self.latest_markers = []
        self.latest_timing = None
        self.latest_relative_pos = None
        self.latest_relative_quat = None
        self.data_received_flag = False
        self.verbose = verbose
        self.is_connected = False
        self._data_lock = threading.Lock()

        self.initial_pos, self.initial_quat = None, None

    def natnet_callback(self, rigid_bodies: list[RigidBody], markers: list[LabelledMarker],
                        timing: TimestampAndLatency, target_id: int):
        """
        Callback function to handle incoming data from NatNet.
        """
        # Update raw data, protected by lock for thread-safety
        with self._data_lock:
            self.latest_rigid_bodies = rigid_bodies
            self.latest_markers = markers
            self.latest_timing = timing
            self.data_received_flag = True

        if self.verbose:
            print(f"[RealSensor] Callback triggered! Markers count: {len(markers)}, Timestamp: {timing.timestamp:.4f}s")

        target_rb = None

        # Find the specific rigid bodies by their IDs
        print(f"[RealSensor] Rigid Bodies:", [rb.id_ for rb in self.latest_rigid_bodies])
        for rb in rigid_bodies:
            if rb.id_ == target_id:
                target_rb = rb
                break

        if target_rb:
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
                print(f"[RealSensor] Target ({target_id}):")
                print(
                    f"[RealSensor] Position (x,y,z): {relative_pos[0]:.4f}, {relative_pos[1]:.4f}, {relative_pos[2]:.4f} meters")
                print(
                    f"[RealSensor] Orientation (Quat x,y,z,w): {relative_quat[0]:.4f}, {relative_quat[1]:.4f}, {relative_quat[2]:.4f}, {relative_quat[3]:.4f}")
                print(
                    f"[RealSensor] Orientation (Euler deg Roll,Pitch,Yaw): {relative_euler[0]:.4f}, {relative_euler[1]:.4f}, {relative_euler[2]:.4f} degrees")

            # Update relative pose data
            with self._data_lock:
                self.latest_relative_pos = relative_pos
                self.latest_relative_quat = relative_quat

        else:
            if not target_rb:
                print(f"[RealSensor] Warning: Target RB with ID {target_id} not found in this frame.")
            print("[RealSensor] Cannot calculate relative pose without rigid body.")


class NatNetDataHandlerWithoutLock:
    """Data handler for processing NatNet data."""

    def __init__(self, verbose: bool = True):
        self.latest_rigid_bodies = []
        self.latest_markers = []
        self.latest_timing = None
        self.latest_relative_pos = np.zeros(3)
        self.latest_relative_quat = np.zeros(4)
        self.data_received_flag = False
        self.verbose = verbose
        self.is_connected = False
        # self._data_lock = threading.Lock()
        self.optitrack_timestamp = time.time()  # Timestamp for the last received data

    def natnet_callback(self, rigid_bodies: list[RigidBody], markers: list[LabelledMarker],
                        timing: TimestampAndLatency, end_effector_rb_id: int, table_rb_id: int):
        """
        Callback function to handle incoming data from NatNet.
        """
        # Update raw data, protected by lock for thread-safety
        # with self._data_lock:
        self.optitrack_timestamp = time.time()
        self.latest_rigid_bodies = rigid_bodies
        self.latest_markers = markers
        self.latest_timing = timing
        self.data_received_flag = True

        if self.verbose:
            print(f"[RealSensor] Callback triggered! Markers count: {len(markers)}, Timestamp: {timing.timestamp:.4f}s")

        end_effector_rb = None
        table_rb = None

        # Find the specific rigid bodies by their IDs
        for rb in rigid_bodies:
            if rb.id_ == end_effector_rb_id:
                end_effector_rb = rb
            elif rb.id_ == table_rb_id:
                table_rb = rb
            # Break early if both are found
            if end_effector_rb and table_rb:
                break

        if end_effector_rb and table_rb:
            # Get positions (x, y, z) OptiTrack uses (z-x-y) order so change it to (x, y, z):
            pos_ee_world = np.array(
                [end_effector_rb.position[2], end_effector_rb.position[0], end_effector_rb.position[1]])
            pos_table_world = np.array([table_rb.position[2], table_rb.position[0], table_rb.position[1]])

            # Get orientations (qx, qy, qz, qw) - NatNet uses (x, y, z, w) order
            # NatNet uses (x, y, z, w) order for quaternions.
            # If OptiTrack's 3D position uses (x, z, y) and we want (x, y, z),
            # then the quaternion components corresponding to OptiTrack's Y and Z axes
            # (which are the 2nd and 3rd components in (x,y,z,w) respectively)
            # need to be swapped to align with the new (x,y,z) coordinate system.
            ee_orientation = end_effector_rb.orientation
            table_orientation = table_rb.orientation

            # Swap y (index 1) and z (index 2) components of the quaternion
            quat_ee_world = R.from_quat([ee_orientation[2], ee_orientation[0], ee_orientation[1], ee_orientation[3]])
            quat_table_world = R.from_quat(
                [table_orientation[2], table_orientation[0], table_orientation[1], table_orientation[3]])

            # 1. Calculate Relative Position
            vec_ee_from_table_world = pos_ee_world - pos_table_world
            relative_pos = quat_table_world.inv().apply(vec_ee_from_table_world)

            # 2. Calculate Relative Orientation
            relative_rot = quat_table_world.inv() * quat_ee_world
            relative_quat = relative_rot.as_quat()  # Get the quaternion (x, y, z, w)
            relative_euler = relative_rot.as_euler('xyz', degrees=True)

            if self.verbose:
                print(f"[RealSensor] \n--- Relative Pose (Timestamp: {timing.timestamp:.4f}s) ---")
                print(f"[RealSensor] End-effector ({end_effector_rb_id}) relative to Table ({table_rb_id}):")
                print(
                    f"[RealSensor] Position (x,y,z): {relative_pos[0]:.4f}, {relative_pos[1]:.4f}, {relative_pos[2]:.4f} meters")
                print(
                    f"[RealSensor] Orientation (Quat x,y,z,w): {relative_quat[0]:.4f}, {relative_quat[1]:.4f}, {relative_quat[2]:.4f}, {relative_quat[3]:.4f}")
                print(
                    f"[RealSensor] Orientation (Euler deg Roll,Pitch,Yaw): {relative_euler[0]:.4f}, {relative_euler[1]:.4f}, {relative_euler[2]:.4f} degrees")

            # Update relative pose data
            # with self._data_lock:
            self.latest_relative_pos = relative_pos
            self.latest_relative_quat = relative_quat

        else:
            if not end_effector_rb:
                print(f"[RealSensor] Warning: End-effector RB with ID {end_effector_rb_id} not found in this frame.")
            if not table_rb:
                print(f"[RealSensor] Warning: Table RB with ID {table_rb_id} not found in this frame.")
            print("[RealSensor] Cannot calculate relative pose without both rigid bodies.")


if __name__ == "__main__":
    print("Hello")
    sensor = RealSensor(ip_address="192.168.1.2")
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

    while True:
        print(sensor.get_relative_pose())
        time.sleep(0.1)

    sensor.stop()