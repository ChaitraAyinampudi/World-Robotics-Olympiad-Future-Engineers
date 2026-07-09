from machine import Pin, I2C
import time

from vl53l0x import VL53L0X


# Pico I2C0 wiring: SDA=GP2, SCL=GP3
I2C_ID = 0
SDA_PIN = 2
SCL_PIN = 3
I2C_FREQUENCY = 400000

# A, B, and C use unique addresses. Sensor D stays disabled on GP21.
SENSOR_CONFIG = (
    ("A", 18, 0x2A),
    ("B", 19, 0x2B),
    ("C", 20, 0x2C),
)

D_XSHUT_PIN = 21

STARTUP_DELAY_MS = 10
LOOP_DELAY_MS = 0
INVALID_READING = -1.0


def initialize_sensors(i2c):
    """Start A, B, and C one at a time and assign unique addresses."""
    xshut_pins = {}

    # Hold every sensor in shutdown before assigning addresses.
    for name, pin_number, _address in SENSOR_CONFIG:
        pin = Pin(pin_number, Pin.OUT)
        pin.value(0)
        xshut_pins[name] = pin

    d_xshut = Pin(D_XSHUT_PIN, Pin.OUT)
    d_xshut.value(0)

    time.sleep_ms(50)

    sensors = {}

    for name, _pin_number, new_address in SENSOR_CONFIG:
        # Start only one sensor at the default 0x29 address.
        xshut_pins[name].value(1)
        time.sleep_ms(STARTUP_DELAY_MS)

        sensor = VL53L0X(i2c, address=0x29)
        sensor.set_address(new_address)
        sensors[name] = sensor

        time.sleep_ms(STARTUP_DELAY_MS)

    return sensors


def read_distance(sensor):
    """Return one distance in millimeters or -1.0 after a read failure."""
    try:
        distance = sensor.read()

        if distance <= 0 or distance >= 8191:
            return INVALID_READING

        return float(distance)

    except OSError:
        return INVALID_READING


def main():
    i2c = I2C(
        I2C_ID,
        sda=Pin(SDA_PIN),
        scl=Pin(SCL_PIN),
        freq=I2C_FREQUENCY,
    )

    sensors = initialize_sensors(i2c)

    while True:
        a = read_distance(sensors["A"])
        b = read_distance(sensors["B"])
        c = read_distance(sensors["C"])

        # Serial output used by the Raspberry Pi parking program.
        print("A={:.1f},B={:.1f},C={:.1f}".format(a, b, c))

        if LOOP_DELAY_MS > 0:
            time.sleep_ms(LOOP_DELAY_MS)


main()
