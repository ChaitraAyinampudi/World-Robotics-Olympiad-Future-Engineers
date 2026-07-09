#!/usr/bin/env python3
"""WRO 2026 recording and autonomous-driving controller.

Modes:
    --recording            Record camera and PS4 control data
    --driving              Run autonomous driving with the parking exit
    --driving-view         Run autonomous driving with a camera window
    --driving-skip         Run autonomous driving without the parking exit
    --driving-view-skip    Skip the parking exit and show the camera window

The program supports DonkeyCar model driving, DYNAMIXEL steering and throttle,
PS4 recording, camera preview, gyro-based stopping, and obstacle parking exit.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys
import time
import signal
import re
import math
import threading
from pathlib import Path
from typing import Tuple
from sense_hat import SenseHat
from time import sleep

# Third-party
import donkeycar as dk
import numpy as np
import pygame
from dynamixel_sdk import *
from donkeycar.parts.transform import Lambda
from donkeycar.parts.tub_v2 import TubWriter, TubWiper
from donkeycar.parts.datastore_v2 import Catalog

try:
    from donkeycar.parts.keras import KerasInterpreter, KerasLinear
except ImportError:
    sys.exit("Keras/TensorFlow not available - install DonkeyCar with AI support.")


FREERUN_MODEL_PATH_CW    = "~/WRO_FE_2026/models/fcwm/fcwm001-006/mypilot.h5"
FREERUN_MODEL_PATH_CCW   = "~/WRO_FE_2026/models/fccwm/fccwm001-006/mypilot.h5"
OBSTACLE_MODEL_PATH_CW   = "~/WRO_FE_2026/models/ocwm/ocwm001-017/mypilot.h5"
OBSTACLE_MODEL_PATH_CCW  = "~/WRO_FE_2026/models/occwm/occwm001-017/mypilot.h5"


def model_path_for_drive_mode(drive_mode: str) -> str:
    """Return the configured default model path for one drive mode."""
    paths = {
        "FCW": FREERUN_MODEL_PATH_CW,
        "FCCW": FREERUN_MODEL_PATH_CCW,
        "OCW": OBSTACLE_MODEL_PATH_CW,
        "OCCW": OBSTACLE_MODEL_PATH_CCW,
    }
    try:
        return os.path.expanduser(paths[drive_mode.upper()])
    except KeyError:
        raise ValueError(f"Unknown drive mode: {drive_mode}")


def infer_drive_mode_from_path(model_path: str):
    """Best-effort mode detection for manually supplied model paths."""
    name = str(model_path).lower()
    for token, mode in (
        ("occw", "OCCW"),
        ("fccw", "FCCW"),
        ("ocw", "OCW"),
        ("fcw", "FCW"),
    ):
        if token in name:
            return mode
    return None


def select_model_path_with_sensehat() -> Tuple[str, str]:
    """Ask for FCW/FCCW/OCW/OCCW only when a driving mode needs it."""
    sense = SenseHat()
    sense.set_rotation(180)
    sense.low_light = True
    sense.clear()

    def flash(msg, seconds=0.7):
        sense.show_message(msg, scroll_speed=0.10, text_colour=[255, 255, 255])
        sleep(seconds)

    drive_mode = "FCW"
    flash(drive_mode)
    print("Driving mode selected.")
    print("Move Sense HAT joystick to choose model, press middle to confirm.")

    while True:
        for ev in sense.stick.get_events():
            if ev.action != "pressed":
                continue
            if ev.direction == "left":
                drive_mode = "FCW";  flash("FCW")
            elif ev.direction == "right":
                drive_mode = "FCCW"; flash("FCCW")
            elif ev.direction == "up":
                drive_mode = "OCW";  flash("OCW")
            elif ev.direction == "down":
                drive_mode = "OCCW"; flash("OCCW")
            elif ev.direction == "middle":
                flash(drive_mode)
                sense.clear()
                print("Selection finished.")
                break
        else:
            continue
        break

    model_path = model_path_for_drive_mode(drive_mode)
    print("Mode:", drive_mode)
    print("Model:", model_path)
    return model_path, drive_mode


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAPTURE_W, CAPTURE_H = (176, 132)      # camera capture
CROP_W, CROP_H       = (160, 120)      # crop sent to network / tub

DATA_PATH          = os.path.expanduser("~/WRO_FE_2026/data")

DRIVE_LOOP_HZ      = 20
JOYSTICK_DEADZONE  = 0.05
RECORD_THRESHOLD   = 0.05               # throttle magnitude to start recording
MAX_SPEED_PERCENT  = 100
STEERING_MAX_SPEED = 100
angle_offset       = 0.7

TUB_INPUTS = [
    "cam/image_array",
    "user/angle",
    "user/throttle",
    "user/mode",
]
TUB_TYPES  = ["image_array", "float", "float", "str"]


# ---------------------------------------------------------------------------
# DYNAMIXEL / XL330 motor configuration
# ---------------------------------------------------------------------------
DXL_PORT = "/dev/ttyUSB0"
DXL_BAUDRATE = 57600
DXL_PROTOCOL_VERSION = 2.0

DXL_STEER_ID = 2
DXL_THROTTLE_IDS = [1]
DXL_THROTTLE_DIRECTIONS = [1]

# Steering calibration
DXL_STEER_CENTER_TICKS = 3060
DXL_STEER_LEFT_DEG = -60
DXL_STEER_RIGHT_DEG = 60
DXL_STEER_DIRECTION = 1          # change to -1 if steering is backwards
DXL_TICKS_PER_DEG = 4096 / 360

DXL_VELOCITY_UNIT_RPM = 0.229
DXL_THROTTLE_MAX_RPM = 100
DXL_STEER_PROFILE_ACCEL = 200
DXL_THROTTLE_PROFILE_ACCEL = 200

# XL330 / X-series Protocol 2.0 control table addresses.
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104
ADDR_PROFILE_ACCEL = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

MODE_VELOCITY = 1
MODE_POSITION = 3
MODE_EXTENDED_POSITION = 4
TORQUE_OFF = 0
TORQUE_ON = 1


# ---------------------------------------------------------------------------
# run_for_degrees() accuracy and safety settings
# ---------------------------------------------------------------------------
RUN_FOR_DEGREES_TOLERANCE_TICKS = 15
RUN_FOR_DEGREES_POLL_SECONDS = 0.02
RUN_FOR_DEGREES_TIMEOUT_MARGIN = 3.0

CAMERA_VIEW_W = 240
CAMERA_VIEW_H = 160

GYRO_TARGET_DEG = 1440.0
FREE_RUN_FINISH_SECONDS = 2.0
OBSTACLE_RUN_FINISH_SECONDS = 0.0

# Raw Z-axis gyro setup. Keep the robot motionless during startup calibration.
GYRO_CALIBRATION_SECONDS = 2.0
GYRO_CALIBRATION_SAMPLE_SECONDS = 0.01
GYRO_RATE_DEADBAND_DEG_PER_SEC = 1.5
GYRO_MAX_VALID_RATE_DEG_PER_SEC = 500.0

# The gyro is sampled independently from DRIVE_LOOP_HZ. A 100 Hz sampling
# thread captures fast turns more consistently than the 20 Hz model loop.
GYRO_SAMPLE_HZ = 100.0
GYRO_SAMPLE_SECONDS = 1.0 / GYRO_SAMPLE_HZ
GYRO_MAX_INTEGRATION_DT_SECONDS = 0.05
GYRO_STATUS_PRINT_SECONDS = 0.20


def _to_uint32(value: int) -> int:
    """Convert signed int to the unsigned 32-bit value expected by the SDK."""
    return int(value) & 0xFFFFFFFF


def _rpm_to_velocity_lsb(rpm: float) -> int:
    return int(round(float(rpm) / DXL_VELOCITY_UNIT_RPM))


class DynamixelBus:
    """Shared DYNAMIXEL bus for steering and throttle parts."""
    def __init__(self, port=DXL_PORT, baudrate=DXL_BAUDRATE):
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(DXL_PROTOCOL_VERSION)
        self.ref_count = 0
        self.closed = False

        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open DYNAMIXEL port: {port}")
        if not self.port_handler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set DYNAMIXEL baudrate: {baudrate}")

        print(f"DYNAMIXEL bus opened on {port} at {baudrate} baud")

    def register_part(self):
        self.ref_count += 1

    def release_part(self):
        self.ref_count = max(0, self.ref_count - 1)
        if self.ref_count == 0:
            self.close()

    def _check(self, dxl_id, result, error, action):
        if result != COMM_SUCCESS:
            print(f"[DYNAMIXEL ID {dxl_id}] {action} failed: "
                  f"{self.packet_handler.getTxRxResult(result)}")
            return False
        if error != 0:
            print(f"[DYNAMIXEL ID {dxl_id}] {action} error: "
                  f"{self.packet_handler.getRxPacketError(error)}")
            return False
        return True

    def write1(self, dxl_id, addr, value, action="write1"):
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, addr, int(value)
        )
        return self._check(dxl_id, result, error, action)

    def write4(self, dxl_id, addr, value, action="write4"):
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, addr, _to_uint32(value)
        )
        return self._check(dxl_id, result, error, action)

    def read4(self, dxl_id, addr, action="read4"):
        value, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, dxl_id, addr
        )
        if not self._check(dxl_id, result, error, action):
            return None
        return int(value)

    def configure_motor(self, dxl_id, mode, profile_accel=200, profile_velocity=None):
        # Operating mode can only be changed with torque off.
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
        time.sleep(0.02)
        self.write1(dxl_id, ADDR_OPERATING_MODE, mode, "set operating mode")
        time.sleep(0.02)
        self.write4(dxl_id, ADDR_PROFILE_ACCEL, profile_accel, "set profile accel")
        if profile_velocity is not None:
            self.write4(dxl_id, ADDR_PROFILE_VELOCITY, profile_velocity,
                        "set profile velocity")
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON, "torque on")
        time.sleep(0.02)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.port_handler.closePort()
            print("DYNAMIXEL bus closed")
        except Exception:
            pass


_DXL_BUS = None


def get_dxl_bus() -> DynamixelBus:
    global _DXL_BUS
    if _DXL_BUS is None or _DXL_BUS.closed:
        _DXL_BUS = DynamixelBus()
    return _DXL_BUS



def _from_uint32(value: int) -> int:
    """Interpret an unsigned SDK position value as a signed 32-bit integer."""
    value = int(value) & 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _motor_ids(value):
    """Allow run_for_degrees() to receive one motor ID or several motor IDs."""
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    return [int(value)]


def _throttle_direction_for_id(motor_id: int) -> int:
    """Return the configured physical-forward multiplier for a throttle motor."""
    try:
        index = list(DXL_THROTTLE_IDS).index(int(motor_id))
    except ValueError:
        return 1

    if index < len(DXL_THROTTLE_DIRECTIONS):
        return -1 if DXL_THROTTLE_DIRECTIONS[index] < 0 else 1
    return 1


def run_position(motor_id, speed, angle, bus=None):
    """Move the steering motor to an angle relative to center.

    Args:
        motor_id: DYNAMIXEL steering ID.
        speed: profile speed percentage from 1 to 100.
        angle: steering angle in physical motor degrees. Positive is right.
        bus: optional shared DynamixelBus.
    """
    bus = bus or get_dxl_bus()
    speed_percent = max(1.0, min(abs(float(speed)), 100.0))
    profile_rpm = DXL_THROTTLE_MAX_RPM * speed_percent / 100.0
    profile_velocity = max(1, _rpm_to_velocity_lsb(profile_rpm))

    bus.configure_motor(
        int(motor_id),
        MODE_POSITION,
        profile_accel=DXL_STEER_PROFILE_ACCEL,
        profile_velocity=profile_velocity,
    )

    goal = DXL_STEER_CENTER_TICKS + (
        DXL_STEER_DIRECTION * float(angle) * DXL_TICKS_PER_DEG
    )
    goal = int(max(0, min(4095, round(goal))))
    bus.write4(int(motor_id), ADDR_GOAL_POSITION, goal, "run_position goal")
    return goal


def run_for_degrees(motor_id, speed, degrees, bus=None):
    """Rotate one or more throttle motors by a relative number of degrees.

    Positive degrees move forward according to DXL_THROTTLE_DIRECTIONS.
    Negative degrees move in reverse. A negative speed also reverses movement;
    its magnitude sets the speed. The function blocks until all selected motors
    reach their targets or the timeout expires.
    """
    bus = bus or get_dxl_bus()
    ids = _motor_ids(motor_id)
    if not ids:
        return True

    speed_value = float(speed)
    speed_percent = max(1.0, min(abs(speed_value), 100.0))
    signed_degrees = float(degrees) * (-1.0 if speed_value < 0 else 1.0)

    profile_rpm = DXL_THROTTLE_MAX_RPM * speed_percent / 100.0
    profile_velocity = max(1, _rpm_to_velocity_lsb(profile_rpm))

    starts = {}
    targets = {}

    # Extended position mode allows relative movements that continue across the
    # normal 0..4095 single-turn boundary.
    for dxl_id in ids:
        bus.configure_motor(
            dxl_id,
            MODE_EXTENDED_POSITION,
            profile_accel=DXL_THROTTLE_PROFILE_ACCEL,
            profile_velocity=profile_velocity,
        )

        raw_position = bus.read4(
            dxl_id, ADDR_PRESENT_POSITION, "read starting position"
        )
        if raw_position is None:
            raise RuntimeError(
                f"Could not read starting position from DYNAMIXEL ID {dxl_id}"
            )

        current_position = _from_uint32(raw_position)
        physical_direction = _throttle_direction_for_id(dxl_id)
        delta_ticks = int(round(
            signed_degrees * DXL_TICKS_PER_DEG * physical_direction
        ))
        target_position = current_position + delta_ticks

        starts[dxl_id] = current_position
        targets[dxl_id] = target_position

    # Send all targets together before waiting, so multi-motor drive systems
    # begin moving nearly simultaneously.
    for dxl_id in ids:
        bus.write4(
            dxl_id,
            ADDR_GOAL_POSITION,
            targets[dxl_id],
            "run_for_degrees goal",
        )

    degrees_per_second = max(1.0, profile_rpm * 6.0)
    expected_seconds = abs(signed_degrees) / degrees_per_second
    timeout = max(1.0, expected_seconds + RUN_FOR_DEGREES_TIMEOUT_MARGIN)
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        all_reached = True
        for dxl_id in ids:
            raw_position = bus.read4(
                dxl_id, ADDR_PRESENT_POSITION, "read movement position"
            )
            if raw_position is None:
                all_reached = False
                continue

            current_position = _from_uint32(raw_position)
            if abs(targets[dxl_id] - current_position) > RUN_FOR_DEGREES_TOLERANCE_TICKS:
                all_reached = False

        if all_reached:
            return True

        time.sleep(RUN_FOR_DEGREES_POLL_SECONDS)

    for dxl_id in ids:
        raw_position = bus.read4(dxl_id, ADDR_PRESENT_POSITION, "read timeout position")
        current_position = None if raw_position is None else _from_uint32(raw_position)
        print(
            f"run_for_degrees timeout on ID {dxl_id}: "
            f"start={starts[dxl_id]}, target={targets[dxl_id]}, "
            f"current={current_position}",
            flush=True,
        )
    return False


def restore_throttle_velocity_mode(bus=None):
    """Return all drive motors to the velocity mode used by the model."""
    bus = bus or get_dxl_bus()
    for dxl_id in DXL_THROTTLE_IDS:
        bus.configure_motor(
            int(dxl_id),
            MODE_VELOCITY,
            profile_accel=DXL_THROTTLE_PROFILE_ACCEL,
            profile_velocity=None,
        )
        bus.write4(int(dxl_id), ADDR_GOAL_VELOCITY, 0, "restore throttle stop")


def obstacle_start_program(drive_mode: str):
    """Exit obstacle parking, then hand control to the trained model.

    The maneuver intentionally uses the reusable motor functions directly:
        run_position(motor_id, speed, angle)
        run_for_degrees(motor_id, speed, degrees)

    OCW and OCCW use different travel distances. OCCW is shorter so the car
    stays behind the traffic-signal intersection line before model takeover.
    """
    mode = str(drive_mode).upper()
    if mode not in ("OCW", "OCCW"):
        return False

    bus = get_dxl_bus()
    print(f"{mode}: parking-exit start program running.", flush=True)
    time.sleep(0.50)

    try:
        if mode == "OCW":
            # Turn right and leave the parking space.
            run_position(DXL_STEER_ID, 50, 65, bus=bus)
            time.sleep(0.15)
            run_for_degrees(DXL_THROTTLE_IDS, 50, 400, bus=bus)

            # Turn left to straighten the car.
            run_position(DXL_STEER_ID, 50, -50, bus=bus)
            time.sleep(0.15)
            run_for_degrees(DXL_THROTTLE_IDS, 50, 400, bus=bus)

            # Center the steering before model control begins.
            run_position(DXL_STEER_ID, 50, 0, bus=bus)
            time.sleep(0.15)

        else:  # OCCW
            # Shorter conservative exit to remain behind the traffic-signal line.
            run_position(DXL_STEER_ID, 50, -65, bus=bus)
            time.sleep(0.15)
            run_for_degrees(DXL_THROTTLE_IDS, 50, 430, bus=bus)

            run_position(DXL_STEER_ID, 50, 60, bus=bus)
            time.sleep(0.15)
            run_for_degrees(DXL_THROTTLE_IDS, 50, 450, bus=bus)

            run_position(DXL_STEER_ID, 50, -7, bus=bus)
            time.sleep(0.15)
            run_for_degrees(DXL_THROTTLE_IDS, -50, 1500, bus=bus)
            
            run_position(DXL_STEER_ID, 50, 15, bus=bus)
            time.sleep(0.15)
            run_for_degrees(DXL_THROTTLE_IDS, -50, 600, bus=bus)

    finally:
        # The model drives with velocity mode after the position-based startup.
        restore_throttle_velocity_mode(bus=bus)
        run_position(DXL_STEER_ID, 100, 0, bus=bus)

    time.sleep(0.15)
    print("Obstacle parking exit complete. Model now controls the car.", flush=True)
    return True


# ---------------------------------------------------------------------------
# Camera parameters
# ---------------------------------------------------------------------------
CAMERA_FRAMERATE             = 30
PICAMERA_AWB_MODE            = 'off'
PICAMERA_EXPOSURE_MODE       = 'off'
PICAMERA_ISO                 = 100
PICAMERA_SHUTTER_SPEED       = 15000
PICAMERA_AWB_GAINS           = (1.5, 1.2)
PICAMERA_EXPOSURE_COMPENSATION = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def center_crop(img, tw=CROP_W, th=CROP_H):
    h, w = img.shape[:2]
    x0 = (w - tw) // 2
    y0 = (h - th) // 2 - 6
    return img[y0:y0 + th, x0:x0 + tw]


# ---------------------------------------------------------------------------
# Hardware parts
# ---------------------------------------------------------------------------
class DynamixelSteering:
    """Single steering motor using DYNAMIXEL / XL330 position mode."""
    def __init__(self, motor_id=DXL_STEER_ID,
                 left=DXL_STEER_LEFT_DEG,
                 right=DXL_STEER_RIGHT_DEG):
        self.bus = get_dxl_bus()
        self.bus.register_part()
        self.id = int(motor_id)
        self.left = float(left)
        self.right = float(right)
        self.prev_goal = None
        self.closed = False

        steer_profile_velocity = _rpm_to_velocity_lsb(STEERING_MAX_SPEED)
        self.bus.configure_motor(
            self.id,
            MODE_POSITION,
            profile_accel=DXL_STEER_PROFILE_ACCEL,
            profile_velocity=steer_profile_velocity
        )
        time.sleep(0.1)
        self._write_goal(DXL_STEER_CENTER_TICKS)

    def _angle_to_goal_ticks(self, angle: float) -> int:
        angle = angle * angle_offset
        angle = max(min(angle, 1.0), -1.0)

        # Normalized input: -1 is left, 0 is center, and +1 is right.
        steer_deg = self.left + (angle + 1) * (self.right - self.left) / 2
        goal = DXL_STEER_CENTER_TICKS + DXL_STEER_DIRECTION * steer_deg * DXL_TICKS_PER_DEG

        # XL330 normal position mode range is 0 to 4095.
        return int(max(0, min(4095, round(goal))))

    def _write_goal(self, goal_ticks: int):
        if goal_ticks == self.prev_goal:
            return
        self.bus.write4(self.id, ADDR_GOAL_POSITION, goal_ticks, "set steering goal")
        self.prev_goal = goal_ticks

    def run(self, angle: float):
        goal_ticks = self._angle_to_goal_ticks(angle)
        self._write_goal(goal_ticks)

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._write_goal(DXL_STEER_CENTER_TICKS)
            time.sleep(0.25)
            self.bus.write1(self.id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "steering torque off")
        finally:
            self.bus.release_part()


class DynamixelThrottle:
    """Drive one or more DYNAMIXEL / XL330 motors using velocity mode."""
    def __init__(self, motor_ids=DXL_THROTTLE_IDS, max_speed=MAX_SPEED_PERCENT):
        self.bus = get_dxl_bus()
        self.bus.register_part()
        self.ids = [int(i) for i in motor_ids]
        self.max_speed = int(max(min(max_speed, 100), 0))
        self.last_speed = None
        self.closed = False

        for dxl_id in self.ids:
            self.bus.configure_motor(
                dxl_id,
                MODE_VELOCITY,
                profile_accel=DXL_THROTTLE_PROFILE_ACCEL,
                profile_velocity=None
            )
        self._stop()

    def _direction_for_index(self, i: int) -> int:
        if i < len(DXL_THROTTLE_DIRECTIONS):
            return -1 if DXL_THROTTLE_DIRECTIONS[i] < 0 else 1
        return 1

    def _stop(self):
        print('run stop!')
        for dxl_id in self.ids:
            self.bus.write4(dxl_id, ADDR_GOAL_VELOCITY, 0, "stop throttle")

    def run(self, throttle: float):
        throttle = max(min(throttle, 1.0), -1.0)
        speed = int(throttle * self.max_speed)
        speed = int(round(speed / 10.0) * 10)

        if self.last_speed == 0 and speed == 0:
            return

        if self.last_speed != 0 and speed == 0:
            self._stop()
            self.last_speed = speed
            return

        if speed != 0 and speed == self.last_speed:
            return

        if speed != 0 and speed != self.last_speed:
            rpm = DXL_THROTTLE_MAX_RPM * (speed / 100.0)
            velocity_lsb = _rpm_to_velocity_lsb(rpm)
            for i, dxl_id in enumerate(self.ids):
                goal_velocity = velocity_lsb * self._direction_for_index(i)
                self.bus.write4(dxl_id, ADDR_GOAL_VELOCITY, goal_velocity,
                                "set throttle velocity")

        self.last_speed = speed

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._stop()
            time.sleep(0.05)
            for dxl_id in self.ids:
                self.bus.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF,
                                "throttle torque off")
        finally:
            self.bus.release_part()


class PS4Joystick:
    """Read PS4 controls and debounce the Triangle erase button."""

    TRIANGLE_BUTTON = 2
    STOP_BUTTON = 4
    TRIANGLE_RELEASE_DEBOUNCE = 0.25

    def __init__(self, deadzone=JOYSTICK_DEADZONE):
        pygame.init(); pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No PS4 controller detected.")
        self.js = pygame.joystick.Joystick(0); self.js.init()
        self.dz = deadzone

        self._triangle_armed = True
        self._triangle_release_started = None

        print(f"Connected joystick: {self.js.get_name()}", flush=True)

    def _dz(self, v):
        return 0.0 if abs(v) < self.dz else float(v)

    def run(self) -> Tuple[float, float, str]:
        pygame.event.pump()
        angle = self._dz(self.js.get_axis(0))
        throttle = -self._dz(self.js.get_axis(4))
        now = time.monotonic()

        if self.js.get_button(self.STOP_BUTTON):
            raise KeyboardInterrupt

        triangle_down = bool(self.js.get_button(self.TRIANGLE_BUTTON))

        if triangle_down:
            self._triangle_release_started = None

            if self._triangle_armed:
                self._triangle_armed = False
                print(
                    "Triangle detected: deleting the newest 100 records...",
                    flush=True,
                )
                return angle, 0.0, "erase"

            return angle, 0.0, "erase_hold"

        if not self._triangle_armed:
            if self._triangle_release_started is None:
                self._triangle_release_started = now
            elif now - self._triangle_release_started >= self.TRIANGLE_RELEASE_DEBOUNCE:
                self._triangle_armed = True
                self._triangle_release_started = None

        return angle, throttle, "user"

    def shutdown(self):
        pygame.quit()


# ---------------------------------------------------------------------------
# Display and control parts for driving modes
# ---------------------------------------------------------------------------
class ConsoleDisplay:
    def __init__(self):
        self.last_t = 0

    def run(self, angle: float, throttle: float):
        t = time.monotonic()
        if t - self.last_t >= 1.0:
            angle = 0.0 if angle is None else float(angle)
            throttle = 0.0 if throttle is None else float(throttle)
            print(f"Pred -> angle {angle:+.2f}  thr {throttle:+.2f}")
            self.last_t = t


class CameraViewer:
    """Show the cropped camera input during recording or view-enabled driving."""
    def __init__(self, width=CAMERA_VIEW_W, height=CAMERA_VIEW_H):
        pygame.init()
        self.size = (int(width), int(height))
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("WRO FE 2026 - Live Driving Camera")
        self.closed = False

    def run(self, image):
        if image is None or self.closed:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return

        frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        # pygame.surfarray expects width x height x channels.
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.smoothscale(surface, self.size)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        try:
            pygame.display.quit()
        except Exception:
            pass


class ObstacleStartController:
    """Run the parking-exit start program once before model control begins."""
    def __init__(self, drive_mode: str):
        self.drive_mode = str(drive_mode).upper()
        self.required = self.drive_mode in ("OCW", "OCCW")
        self.finished = False

        if self.required:
            print(
                f"{self.drive_mode}: parking-exit start program armed."
            )
        else:
            print(f"{self.drive_mode}: no parking-exit program required.")

    def run(self):
        if self.finished or not self.required:
            return 0.0, 0.0, False

        # This deliberately blocks the vehicle loop while run_for_degrees()
        # performs the complete physical startup maneuver. The camera capture
        # thread remains alive; model control begins immediately afterward.
        obstacle_start_program(self.drive_mode)
        self.finished = True
        return 0.0, 0.0, False


class DriveControlMux:
    """Use startup controls while active, otherwise pass through model controls."""
    def run(self, start_angle, start_throttle, start_active,
            pilot_angle, pilot_throttle):
        if bool(start_active):
            return float(start_angle), float(start_throttle)

        angle = 0.0 if pilot_angle is None else float(pilot_angle)
        throttle = 0.0 if pilot_throttle is None else float(pilot_throttle)
        return angle, throttle


class GyroThreeLapController:
    """Stop autonomous driving after the gyro target and finish delay.

    A dedicated thread samples the Sense HAT raw Z-axis gyro at GYRO_SAMPLE_HZ.
    Sampling begins after the parking-exit sequence so that maneuver is not
    included in lap progress. Valid samples are bias-corrected and integrated
    with the trapezoidal rule.
    """

    def __init__(self, drive_mode: str):
        self.drive_mode = str(drive_mode).upper()
        self.sense = SenseHat()

        self.gyro_bias_rad_per_sec = 0.0
        self.total_rotation_deg = 0.0
        self.current_rate_deg_per_sec = 0.0
        self.finish_deadline = None
        self.stopped = False
        self.last_print_time = 0.0
        self.read_error_printed = False

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sample_thread = None
        self._sampling_started = False

        if self.drive_mode in ("OCW", "OCCW"):
            self.finish_seconds = OBSTACLE_RUN_FINISH_SECONDS
            run_name = "obstacle"
        else:
            self.finish_seconds = FREE_RUN_FINISH_SECONDS
            run_name = "free"

        self._calibrate_raw_z_gyro()

        print(
            f"Sense HAT raw Z gyro armed for {run_name} run: "
            f"sampling at {GYRO_SAMPLE_HZ:.0f} Hz, stop after "
            f"{GYRO_TARGET_DEG:.0f} degrees plus "
            f"{self.finish_seconds:.1f} seconds.",
            flush=True,
        )

    def _read_raw_z(self):
        """Read raw Z-axis angular velocity in radians per second."""
        try:
            raw = self.sense.get_gyroscope_raw()
            self.read_error_printed = False
            return float(raw["z"])
        except Exception as exc:
            if not self.read_error_printed:
                print(f"Sense HAT raw gyro read failed: {exc}", flush=True)
                self.read_error_printed = True
            return None

    def _calibrate_raw_z_gyro(self):
        """Measure stationary raw-Z bias before autonomous driving begins."""
        print(
            f"Calibrating raw Z gyro for {GYRO_CALIBRATION_SECONDS:.1f} seconds. "
            "Keep the robot completely still...",
            flush=True,
        )

        samples = []
        deadline = time.monotonic() + GYRO_CALIBRATION_SECONDS
        while time.monotonic() < deadline:
            value = self._read_raw_z()
            if value is not None and math.isfinite(value):
                samples.append(value)
            time.sleep(GYRO_CALIBRATION_SAMPLE_SECONDS)

        if samples:
            self.gyro_bias_rad_per_sec = sum(samples) / len(samples)
            print(
                f"Raw Z gyro calibrated from {len(samples)} samples: "
                f"bias={self.gyro_bias_rad_per_sec:+.6f} rad/s.",
                flush=True,
            )
        else:
            self.gyro_bias_rad_per_sec = 0.0
            print(
                "Warning: no valid gyro calibration samples; using zero bias.",
                flush=True,
            )

    def _start_sampling(self):
        """Start the high-rate gyro sampler once model control is active."""
        if self._sampling_started:
            return

        self._sampling_started = True
        self._stop_event.clear()
        self._sample_thread = threading.Thread(
            target=self._sample_loop,
            name="sensehat-gyro-100hz",
            daemon=True,
        )
        self._sample_thread.start()
        print(
            f"Raw gyro sampling started at {GYRO_SAMPLE_HZ:.0f} Hz.",
            flush=True,
        )

    def _sample_loop(self):
        """Continuously sample and integrate the raw Z-axis gyro."""
        previous_time = None
        previous_rate_deg_per_sec = None
        next_sample_time = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()
            raw_z = self._read_raw_z()

            if raw_z is not None and math.isfinite(raw_z):
                rate_deg_per_sec = math.degrees(
                    raw_z - self.gyro_bias_rad_per_sec
                )

                if abs(rate_deg_per_sec) < GYRO_RATE_DEADBAND_DEG_PER_SEC:
                    rate_deg_per_sec = 0.0

                if abs(rate_deg_per_sec) <= GYRO_MAX_VALID_RATE_DEG_PER_SEC:
                    if previous_time is not None:
                        dt = now - previous_time
                        if 0.0 < dt <= GYRO_MAX_INTEGRATION_DT_SECONDS:
                            if previous_rate_deg_per_sec is None:
                                delta_deg = rate_deg_per_sec * dt
                            else:
                                delta_deg = (
                                    previous_rate_deg_per_sec + rate_deg_per_sec
                                ) * 0.5 * dt

                            with self._lock:
                                self.total_rotation_deg += delta_deg

                    previous_time = now
                    previous_rate_deg_per_sec = rate_deg_per_sec
                    with self._lock:
                        self.current_rate_deg_per_sec = rate_deg_per_sec
                else:
                    print(
                        f"Ignored unusual raw gyro rate of "
                        f"{rate_deg_per_sec:+.1f} deg/s.",
                        flush=True,
                    )
                    # Reset timing so an invalid gap cannot become one large
                    # integration interval on the next valid sample.
                    previous_time = None
                    previous_rate_deg_per_sec = None
            else:
                previous_time = None
                previous_rate_deg_per_sec = None

            next_sample_time += GYRO_SAMPLE_SECONDS
            sleep_seconds = next_sample_time - time.monotonic()
            if sleep_seconds > 0.0:
                self._stop_event.wait(sleep_seconds)
            else:
                # If a read takes longer than one period, restart the schedule
                # from now instead of trying to execute many catch-up samples.
                next_sample_time = time.monotonic()

    def run(self, angle, throttle):
        angle = 0.0 if angle is None else float(angle)
        throttle = 0.0 if throttle is None else float(throttle)
        now = time.monotonic()

        if not self._sampling_started:
            self._start_sampling()

        if self.stopped:
            return 0.0, 0.0

        with self._lock:
            total_rotation_deg = self.total_rotation_deg
            current_rate = self.current_rate_deg_per_sec

        progress = abs(total_rotation_deg)

        if now - self.last_print_time >= GYRO_STATUS_PRINT_SECONDS:
            print(
                f"Raw gyro: {current_rate:+.1f} deg/s | "
                f"lap progress: {progress:.1f} / "
                f"{GYRO_TARGET_DEG:.0f} degrees",
                flush=True,
            )
            self.last_print_time = now

        if self.finish_deadline is None and progress >= GYRO_TARGET_DEG:
            self.finish_deadline = now + self.finish_seconds
            print(
                f"Gyro target detected at {progress:.1f} degrees. "
                f"Continuing model driving for {self.finish_seconds:.1f} seconds.",
                flush=True,
            )

        if self.finish_deadline is not None and now >= self.finish_deadline:
            self.stopped = True
            self._stop_event.set()
            print(
                "Gyro-target finish delay complete. "
                "Steering centered and throttle stopped.",
                flush=True,
            )
            return 0.0, 0.0

        return angle, throttle

    def shutdown(self):
        """Stop the high-rate sampling thread during vehicle shutdown."""
        self._stop_event.set()
        thread = self._sample_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Reusable camera initialization for recording and driving
# ---------------------------------------------------------------------------
def add_camera(car: dk.vehicle.Vehicle):
    from donkeycar.parts.camera import PiCamera

    cam = PiCamera(
        image_w = CAPTURE_W,
        image_h = CAPTURE_H,
        image_d = 3
    )

    time.sleep(2)

    cam.camera.framerate           = CAMERA_FRAMERATE
    cam.camera.exposure_mode       = PICAMERA_EXPOSURE_MODE
    cam.camera.awb_mode            = PICAMERA_AWB_MODE
    cam.camera.iso                 = PICAMERA_ISO
    cam.camera.shutter_speed       = PICAMERA_SHUTTER_SPEED
    cam.camera.exposure_compensation = PICAMERA_EXPOSURE_COMPENSATION
    cam.camera.awb_gains           = PICAMERA_AWB_GAINS

    car.add(cam, outputs=["cam/raw"], threaded=True)
    car.add(Lambda(center_crop), inputs=["cam/raw"], outputs=["cam/image_array"])


# ---------------------------------------------------------------------------
# Vehicle builders
# ---------------------------------------------------------------------------
def build_vehicle_recording() -> dk.vehicle.Vehicle:
    car = dk.vehicle.Vehicle()
    add_camera(car)

    car.add(CameraViewer(), inputs=["cam/image_array"], outputs=[])

    car.add(PS4Joystick(), outputs=["user/angle", "user/throttle", "user/mode"])
    car.add(DynamixelSteering(), inputs=["user/angle"])
    car.add(DynamixelThrottle(), inputs=["user/throttle"])

    # Write only during normal user driving. Do not record while the
    # controller is in erase mode, even if the throttle stick is moved.
    car.add(Lambda(lambda t, mode: mode == "user" and abs(t) > RECORD_THRESHOLD),
            inputs=["user/throttle", "user/mode"], outputs=["recording"])

    tub = TubWriter(base_path=DATA_PATH, inputs=TUB_INPUTS, types=TUB_TYPES)
    car.add(tub, inputs=TUB_INPUTS, outputs=["tub/num_records"], run_condition="recording")

    class RecordCounter:
        """Display total sample count every 10 new records."""
        def __init__(self):
            self.last_ten = -1
        def run(self, n):
            if n is None:
                return
            ten = n // 10
            if ten != self.last_ten:
                print(f"Recorded samples: {n}")
                self.last_ten = ten

    car.add(RecordCounter(), inputs=["tub/num_records"], outputs=[])

    class PromptWiper:
        """Physically remove the newest records from a DonkeyCar tub.

        The deletion removes matching data files, truncates catalog files,
        updates catalog metadata, and lowers manifest.current_index.
        """
        def __init__(self, tub, num_records=100):
            self.tub = tub
            self.num = int(num_records)
            self.last_delete_time = float("-inf")
            self.delete_cooldown = 1.0

        def _tub_path(self):
            for attr in ("base_path", "path", "tub_path", "dir"):
                value = getattr(self.tub, attr, None)
                if value:
                    return Path(value).expanduser()

            data_root = Path(DATA_PATH).expanduser()
            tub_dirs = []
            if data_root.exists():
                for child in data_root.iterdir():
                    if child.is_dir() and ((child / "images").exists() or (child / "manifest.json").exists()):
                        tub_dirs.append(child)
            if tub_dirs:
                return max(tub_dirs, key=lambda x: x.stat().st_mtime)

            return data_root

        @staticmethod
        def _index_from_name(path):
            """Return the first integer in a DonkeyCar file name."""
            match = re.search(r"(\d+)", path.stem)
            if not match:
                return None
            try:
                return int(match.group(1))
            except ValueError:
                return None

        @staticmethod
        def _catalog_manifest_path(catalog_path):
            return catalog_path.with_name(f"{catalog_path.stem}.catalog_manifest")

        @staticmethod
        def _safe_unlink(path):
            try:
                path.unlink()
                return 1
            except FileNotFoundError:
                return 0
            except OSError as e:
                print(f"Could not delete {path}: {e}")
                return 0

        def _delete_data_files_for_range(self, tub_path, start_index, end_index):
            """Delete physical image/json files whose index is in [start, end)."""
            deleted = 0
            image_dirs = []

            if (tub_path / "images").exists():
                image_dirs.append(tub_path / "images")
            if tub_path.exists():
                image_dirs.append(tub_path)

            image_exts = {".jpg", ".jpeg", ".png", ".npy"}
            seen = set()
            for folder in image_dirs:
                for path in folder.iterdir():
                    if not path.is_file() or path in seen:
                        continue
                    seen.add(path)
                    if path.suffix.lower() not in image_exts:
                        continue
                    idx = self._index_from_name(path)
                    if idx is not None and start_index <= idx < end_index:
                        deleted += self._safe_unlink(path)

            # Delete indexed JSON records when present.
            protected_json = {"manifest.json"}
            if tub_path.exists():
                for path in tub_path.rglob("*.json"):
                    if not path.is_file():
                        continue
                    if path.name in protected_json:
                        continue
                    idx = self._index_from_name(path)
                    if idx is not None and start_index <= idx < end_index:
                        deleted += self._safe_unlink(path)

            return deleted

        def _truncate_catalogs(self, tub_path, delete_start, old_current_index):
            manifest = self.tub.manifest
            old_catalog_paths = list(getattr(manifest, "catalog_paths", []))
            new_catalog_paths = []
            removed_catalog_files = 0

            # Close the active catalog before physically rewriting catalog files.
            try:
                if manifest.current_catalog:
                    manifest.current_catalog.close()
            except Exception as e:
                print(f"Warning: could not close active catalog before erase: {e}")

            for rel_path in old_catalog_paths:
                catalog_path = tub_path / rel_path

                if not catalog_path.exists():
                    continue

                try:
                    cat = Catalog(catalog_path.as_posix(), read_only=False)
                    start_index = int(cat.manifest.start_index())
                    line_count = int(cat.seekable.lines())
                    end_index = start_index + line_count

                    if start_index >= delete_start:
                        # Keep one empty catalog when all records are deleted.
                        if delete_start == 0 and not new_catalog_paths:
                            cat.seekable.truncate_until_end(0)
                            cat.manifest.update_line_lengths([])
                            new_catalog_paths.append(rel_path)
                            cat.close()
                        else:
                            cat.close()
                            removed_catalog_files += self._safe_unlink(catalog_path)
                            removed_catalog_files += self._safe_unlink(self._catalog_manifest_path(catalog_path))

                    elif start_index < delete_start < end_index:
                        # Keep rows before the deletion range.
                        keep_count = max(0, delete_start - start_index)
                        cat.seekable.truncate_until_end(keep_count)
                        cat.manifest.update_line_lengths(cat.seekable.line_lengths)
                        new_catalog_paths.append(rel_path)
                        cat.close()

                    else:
                        new_catalog_paths.append(rel_path)
                        cat.close()

                except Exception as e:
                    print(f"Could not rewrite catalog {catalog_path}: {e}")

            if not new_catalog_paths:
                rel_path = "catalog_0.catalog"
                catalog_path = tub_path / rel_path
                try:
                    cat = Catalog(catalog_path.as_posix(), read_only=False, start_index=0)
                    cat.seekable.truncate_until_end(0)
                    cat.manifest.update_line_lengths([])
                    cat.close()
                    new_catalog_paths = [rel_path]
                except Exception as e:
                    print(f"Could not recreate empty catalog: {e}")

            manifest.catalog_paths = new_catalog_paths
            manifest.current_index = delete_start
            manifest.deleted_indexes = set(i for i in manifest.deleted_indexes if i < delete_start)
            manifest._update_catalog_metadata(update=True)

            try:
                last_catalog_path = tub_path / manifest.catalog_paths[-1]
                manifest.current_catalog = Catalog(last_catalog_path.as_posix(), read_only=False)
            except Exception as e:
                print(f"Warning: could not reopen active catalog after erase: {e}")

            return removed_catalog_files

        def _hard_delete_last_n_records(self):
            manifest = getattr(self.tub, "manifest", None)
            if manifest is None:
                print("Hard delete failed: tub has no manifest object.")
                return 0, 0, 0

            old_current_index = int(getattr(manifest, "current_index", 0))
            if old_current_index <= 0:
                print("No tub records to delete.")
                return 0, 0, 0

            delete_start = max(0, old_current_index - self.num)
            actual_deleted_records = old_current_index - delete_start
            tub_path = self._tub_path()

            deleted_data_files = self._delete_data_files_for_range(
                tub_path, delete_start, old_current_index
            )
            removed_catalog_files = self._truncate_catalogs(
                tub_path, delete_start, old_current_index
            )

            return actual_deleted_records, deleted_data_files, removed_catalog_files

        def run(self, mode):
            if mode != "erase":
                return

            now = time.monotonic()
            if now - self.last_delete_time < self.delete_cooldown:
                return
            self.last_delete_time = now

            print("Erase request accepted. Updating tub files...", flush=True)
            try:
                deleted_records, deleted_data_files, removed_catalog_files = (
                    self._hard_delete_last_n_records()
                )
            except Exception as e:
                print(f"Hard delete failed: {e}", flush=True)
                return

            print(
                f"Hard-deleted {deleted_records} records, "
                f"{deleted_data_files} image/json files, "
                f"and {removed_catalog_files} old catalog files.",
                flush=True,
            )

    wiper = PromptWiper(tub.tub, num_records=100)
    car.add(wiper, inputs=["user/mode"], outputs=[])
    return car


def build_vehicle_driving(model_path: str, drive_mode: str,
                          show_camera: bool = False,
                          skip_parking: bool = False) -> dk.vehicle.Vehicle:
    """Build one of the autonomous driving variants.

    show_camera:
        False for --driving / --driving-skip
        True for --driving-view / --driving-view-skip

    skip_parking:
        False runs obstacle_start_program() for OCW/OCCW.
        True connects the model directly to the DYNAMIXEL motors immediately.
    """
    car = dk.vehicle.Vehicle()
    add_camera(car)

    interpreter = KerasInterpreter()
    pilot = KerasLinear(interpreter=interpreter, input_shape=(CROP_H, CROP_W, 3))
    pilot.load(model_path)

    car.add(pilot, inputs=["cam/image_array"],
            outputs=["pilot/angle", "pilot/throttle"])

    car.add(ConsoleDisplay(), inputs=["pilot/angle", "pilot/throttle"], outputs=[])

    if show_camera:
        car.add(CameraViewer(), inputs=["cam/image_array"], outputs=[])

    if skip_parking:
        print("Parking-exit program skipped. Model controls the car immediately.")
        steering_input = "pilot/angle"
        throttle_input = "pilot/throttle"
    else:
        car.add(
            ObstacleStartController(drive_mode),
            outputs=["startup/angle", "startup/throttle", "startup/active"]
        )
        car.add(
            DriveControlMux(),
            inputs=[
                "startup/angle", "startup/throttle", "startup/active",
                "pilot/angle", "pilot/throttle",
            ],
            outputs=["drive/angle", "drive/throttle"]
        )
        steering_input = "drive/angle"
        throttle_input = "drive/throttle"

    # Stop after the configured gyro target and finish delay.
    car.add(
        GyroThreeLapController(drive_mode),
        inputs=[steering_input, throttle_input],
        outputs=["final/angle", "final/throttle"],
    )

    car.add(DynamixelSteering(), inputs=["final/angle"])
    car.add(DynamixelThrottle(), inputs=["final/throttle"])
    return car


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    print(
        f"manage26V17 starting | gyro target: {GYRO_TARGET_DEG:.0f} degrees | "
        f"gyro sample rate: {GYRO_SAMPLE_HZ:.0f} Hz",
        flush=True,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Record data or run one of four autonomous driving variants, "
            "with optional camera view, optional parking-exit skip, and "
            "Sense HAT gyro-target stopping"
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--recording", action="store_true",
        help="Manual PS4 driving, live camera view, and data logging"
    )
    group.add_argument(
        "--driving", action="store_true",
        help="Autonomous model driving without a camera window"
    )
    group.add_argument(
        "--driving-view", action="store_true",
        help="Autonomous driving with parking exit and a live camera window"
    )
    group.add_argument(
        "--driving-skip", action="store_true",
        help="Autonomous model driving that skips the parking-exit program"
    )
    group.add_argument(
        "--driving-view-skip", action="store_true",
        help="Skip the parking exit and show the live camera window"
    )
    parser.add_argument(
        "--model", default=None,
        help="Optional .h5 model path for any driving variant"
    )
    parser.add_argument(
        "--drive-mode", choices=["FCW", "FCCW", "OCW", "OCCW"], default=None,
        help="Optional mode override, especially useful with --model"
    )
    args = parser.parse_args()

    if args.recording:
        print("Recording mode selected with live camera view. Skipping model selection.")
        vehicle = build_vehicle_recording()
    else:
        show_camera = bool(args.driving_view or args.driving_view_skip)
        skip_parking = bool(args.driving_skip or args.driving_view_skip)

        if show_camera and skip_parking:
            print("Driving with camera view selected; parking exit will be skipped.")
        elif show_camera:
            print("Driving with camera view selected; parking exit remains enabled.")
        elif skip_parking:
            print("Driving-skip selected; parking exit will be skipped.")
        else:
            print("Normal driving selected; parking exit remains enabled.")

        if args.model:
            model_path = os.path.expanduser(args.model)
            drive_mode = args.drive_mode or infer_drive_mode_from_path(model_path)
            if drive_mode is None:
                print(
                    "Warning: drive mode could not be inferred from the model path. "
                    "The obstacle parking-exit program will be disabled. Use "
                    "--drive-mode OCW or --drive-mode OCCW when needed."
                )
                drive_mode = "UNKNOWN"
        elif args.drive_mode:
            drive_mode = args.drive_mode
            model_path = model_path_for_drive_mode(drive_mode)
        else:
            model_path, drive_mode = select_model_path_with_sensehat()

        print("Mode:", drive_mode)
        print("Model:", model_path)

        mpath = Path(model_path).expanduser()
        if not mpath.is_file():
            sys.exit(f"Model file not found: {mpath}")

        vehicle = build_vehicle_driving(
            str(mpath),
            drive_mode=drive_mode,
            show_camera=show_camera,
            skip_parking=skip_parking,
        )

    def _sigterm(_s, _f):
        print("\nSIGTERM - shutting down...")
        (vehicle.shutdown() if hasattr(vehicle, "shutdown") else vehicle.stop())
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    try:
        vehicle.start(rate_hz=DRIVE_LOOP_HZ)
    except KeyboardInterrupt:
        print("\nCtrl-C / window close - shutting down...")
    finally:
        (vehicle.shutdown() if hasattr(vehicle, "shutdown") else vehicle.stop())
