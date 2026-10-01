"""虚拟位移传感器测试。"""

import pytest

from sensors.virtual_displacement_sensor import (
    DisplacementSensorParameters,
    VirtualDisplacementSensor,
)


def test_bias_should_be_added_to_displacement() -> None:
    """测量值应包含固定零偏。"""

    sensor = VirtualDisplacementSensor(
        DisplacementSensorParameters(
            measurement_range=1.0,
            resolution_bits=24,
            noise_standard_deviation=0.0,
            constant_bias=0.01,
            dropout_probability=0.0,
        )
    )

    measurement = sensor.read(
        true_displacement=0.02,
        timestamp=0.0,
    )

    assert measurement.valid is True
    assert measurement.displacement == pytest.approx(
        0.03,
        abs=1e-5,
    )


def test_measurement_should_be_limited_by_range() -> None:
    """超过量程的位移应被限幅。"""

    sensor = VirtualDisplacementSensor(
        DisplacementSensorParameters(
            measurement_range=0.05,
            noise_standard_deviation=0.0,
            constant_bias=0.0,
            dropout_probability=0.0,
        )
    )

    measurement = sensor.read(
        true_displacement=0.10,
        timestamp=0.0,
    )

    assert measurement.saturated is True
    assert measurement.displacement == pytest.approx(0.05)


def test_dropout_should_return_last_measurement() -> None:
    """丢样时应返回上一次测量并标记无效。"""

    sensor = VirtualDisplacementSensor(
        DisplacementSensorParameters(
            dropout_probability=1.0,
        )
    )

    measurement = sensor.read(
        true_displacement=0.03,
        timestamp=0.0,
    )

    assert measurement.valid is False
    assert measurement.displacement == 0.0


def test_same_seed_should_generate_same_sequence() -> None:
    """相同随机种子应产生相同测量序列。"""

    parameters = DisplacementSensorParameters(
        noise_standard_deviation=0.001,
        dropout_probability=0.1,
        seed=2027,
    )

    first_sensor = VirtualDisplacementSensor(parameters)
    second_sensor = VirtualDisplacementSensor(parameters)

    first_sequence = [first_sensor.read(0.01, index * 0.001) for index in range(100)]

    second_sequence = [second_sensor.read(0.01, index * 0.001) for index in range(100)]

    assert first_sequence == second_sequence


def test_reset_should_restore_random_sequence() -> None:
    """复位后应重新产生相同测量序列。"""

    sensor = VirtualDisplacementSensor(
        DisplacementSensorParameters(
            noise_standard_deviation=0.001,
            seed=2027,
        )
    )

    first_sequence = [sensor.read(0.01, index * 0.001) for index in range(20)]

    sensor.reset()

    second_sequence = [sensor.read(0.01, index * 0.001) for index in range(20)]

    assert first_sequence == second_sequence


def test_invalid_range_should_raise_value_error() -> None:
    """位移量程必须大于零。"""

    with pytest.raises(
        ValueError,
        match="measurement_range",
    ):
        DisplacementSensorParameters(
            measurement_range=0.0,
        )
