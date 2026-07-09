from time import sleep, monotonic
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# Configuration
PORT = "/dev/ttyUSB0"
BAUD = 57600
PROTOCOL = 2.0

ID_THROTTLE = 1
ID_STEER = 2

STEER_CENTER = 3126
MAX_STEER_DEG = 180

# XL330 control-table addresses
ADDR_OPERATING_MODE = 11
ADDR_VELOCITY_LIMIT = 44
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

VELOCITY_MODE = 1
POSITION_MODE = 3
EXTENDED_POSITION_MODE = 4

TORQUE_OFF = 0
TORQUE_ON = 1

TICKS_PER_REVOLUTION = 4096
RPM_PER_UNIT = 0.229

port = PortHandler(PORT)
packet = PacketHandler(PROTOCOL)
current_modes = {}


def check(result, error, action):
    """Raise an error when a DYNAMIXEL command fails."""
    if result != COMM_SUCCESS:
        raise RuntimeError(f"{action}: {packet.getTxRxResult(result)}")
    if error:
        raise RuntimeError(f"{action}: {packet.getRxPacketError(error)}")


def write1(motor_id, address, value, action):
    result, error = packet.write1ByteTxRx(port, motor_id, address, value)
    check(result, error, action)


def write4(motor_id, address, value, action):
    result, error = packet.write4ByteTxRx(
        port, motor_id, address, value & 0xFFFFFFFF
    )
    check(result, error, action)


def read4(motor_id, address, action):
    value, result, error = packet.read4ByteTxRx(port, motor_id, address)
    check(result, error, action)
    return value


def signed32(value):
    return value - 0x100000000 if value & 0x80000000 else value


def set_mode(motor_id, mode):
    """Change operating mode only when necessary."""
    if current_modes.get(motor_id) == mode:
        return

    write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_OFF,
           f"Disable torque on ID {motor_id}")
    write1(motor_id, ADDR_OPERATING_MODE, mode,
           f"Set operating mode on ID {motor_id}")
    write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_ON,
           f"Enable torque on ID {motor_id}")
    current_modes[motor_id] = mode


def speed_percent_to_raw(motor_id, speed_percent):
    """Convert 0-100% into the motor's configured velocity units."""
    speed_percent = max(0.0, min(100.0, abs(float(speed_percent))))
    velocity_limit = read4(
        motor_id, ADDR_VELOCITY_LIMIT, f"Read velocity limit from ID {motor_id}"
    )
    return max(1, round(velocity_limit * speed_percent / 100.0))


def run_position(motor_id, speed, angle):
    """
    Move to an angle relative to STEER_CENTER.

    speed: 0-100 percent
    angle: -MAX_STEER_DEG to +MAX_STEER_DEG

    This command is non-blocking, so the steering can move while another
    function controls the throttle motor.
    """
    angle = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, float(angle)))
    goal = round(STEER_CENTER + angle * TICKS_PER_REVOLUTION / 360.0)
    goal = max(0, min(TICKS_PER_REVOLUTION - 1, goal))

    set_mode(motor_id, POSITION_MODE)
    raw_speed = speed_percent_to_raw(motor_id, speed)

    write4(motor_id, ADDR_PROFILE_ACCELERATION, 200,
           f"Set position acceleration on ID {motor_id}")
    write4(motor_id, ADDR_PROFILE_VELOCITY, raw_speed,
           f"Set position speed on ID {motor_id}")
    write4(motor_id, ADDR_GOAL_POSITION, goal,
           f"Set goal position on ID {motor_id}")


def run_for_seconds(motor_id, speed, seconds):
    """
    Run continuously for a set time.

    speed: -100 to +100 percent; negative reverses direction
    seconds: movement duration
    """
    speed = max(-100.0, min(100.0, float(speed)))
    seconds = max(0.0, float(seconds))

    set_mode(motor_id, VELOCITY_MODE)
    if speed == 0:
        write4(motor_id, ADDR_GOAL_VELOCITY, 0, f"Stop ID {motor_id}")
        sleep(seconds)
        return

    raw_speed = speed_percent_to_raw(motor_id, speed)
    raw_speed = raw_speed if speed > 0 else -raw_speed

    write4(motor_id, ADDR_PROFILE_ACCELERATION, 200,
           f"Set velocity acceleration on ID {motor_id}")
    write4(motor_id, ADDR_GOAL_VELOCITY, raw_speed,
           f"Start velocity movement on ID {motor_id}")

    try:
        sleep(seconds)
    finally:
        write4(motor_id, ADDR_GOAL_VELOCITY, 0,
               f"Stop ID {motor_id}")


def run_for_degrees(motor_id, speed, degrees):
    """
    Rotate by a relative number of degrees.

    speed: 0-100 percent
    degrees: positive or negative; its sign chooses direction
    """
    degrees = float(degrees)
    if degrees == 0:
        return
    if float(speed) == 0:
        raise ValueError("run_for_degrees speed must be greater than 0.")

    set_mode(motor_id, EXTENDED_POSITION_MODE)
    raw_speed = speed_percent_to_raw(motor_id, speed)

    current = signed32(
        read4(motor_id, ADDR_PRESENT_POSITION,
              f"Read present position from ID {motor_id}")
    )
    movement_ticks = round(degrees * TICKS_PER_REVOLUTION / 360.0)
    goal = current + movement_ticks

    write4(motor_id, ADDR_PROFILE_ACCELERATION, 200,
           f"Set degree-move acceleration on ID {motor_id}")
    write4(motor_id, ADDR_PROFILE_VELOCITY, raw_speed,
           f"Set degree-move speed on ID {motor_id}")
    write4(motor_id, ADDR_GOAL_POSITION, goal,
           f"Set relative goal position on ID {motor_id}")

    # Stop waiting if the motor does not reach the target in time.
    rpm = max(raw_speed * RPM_PER_UNIT, 0.229)
    expected_seconds = abs(degrees) / (rpm * 6.0)
    deadline = monotonic() + expected_seconds * 2.0 + 2.0

    while monotonic() < deadline:
        present = signed32(
            read4(motor_id, ADDR_PRESENT_POSITION,
                  f"Check present position on ID {motor_id}")
        )
        if abs(goal - present) <= 10:
            return
        sleep(0.02)

    raise TimeoutError(
        f"ID {motor_id} did not finish the {degrees:g}-degree movement."
    )


def stop_motor(motor_id):
    """Stop a motor if it is currently in velocity mode."""
    if current_modes.get(motor_id) == VELOCITY_MODE:
        write4(motor_id, ADDR_GOAL_VELOCITY, 0, f"Stop ID {motor_id}")


def connect():
    if not port.openPort():
        raise RuntimeError(f"Could not open {PORT}")
    if not port.setBaudRate(BAUD):
        port.closePort()
        raise RuntimeError(f"Could not set baud rate to {BAUD}")

    for motor_id in (ID_THROTTLE, ID_STEER):
        _, result, error = packet.ping(port, motor_id)
        check(result, error, f"Ping ID {motor_id}")


def shutdown():
    """Stop movement, center steering, disable torque, and close the port."""
    try:
        stop_motor(ID_THROTTLE)

        if current_modes.get(ID_STEER) is not None:
            run_position(ID_STEER, 30, 0)
            sleep(0.5)

        for motor_id in (ID_THROTTLE, ID_STEER):
            try:
                write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_OFF,
                       f"Disable torque on ID {motor_id}")
            except RuntimeError:
                pass
    finally:
        port.closePort()


# Example controlled program
if __name__ == "__main__":
    connect()

    try:
        run_position(ID_STEER, 35, 45)
        run_for_seconds(ID_THROTTLE, 50, 2)

        run_position(ID_STEER, 35, -45)
        run_for_degrees(ID_THROTTLE, 40, -720)

        run_position(ID_STEER, 35, 0)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        shutdown()
