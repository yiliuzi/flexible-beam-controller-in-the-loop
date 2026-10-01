"""虚拟IMU测试。"""

import pytest

from sensors.virtual_imu import VirtualIMU, VirtualIMUParameters


def test_bias_should_be_added_to_measurement() -> None:
    """测量结果应包含配置的固定零偏。"""

    imu = VirtualIMU(
        VirtualIMUParameters(
            measurement_range=100.0,
            resolution_bits=24,
            noise_standard_deviation=0.0,
            constant_bias=0.5,
            dropout_probability=0.0,
        )
    )

    measurement = imu.read(
        true_acceleration=2.0,
        timestamp=0.0,
    )

    assert measurement.valid is True
    assert measurement.acceleration == pytest.approx(2.5, abs=1e-4)


def test_measurement_should_be_limited_by_sensor_range() -> None:
    """超过量程的测量值应被限幅。"""

    imu = VirtualIMU(
        VirtualIMUParameters(
            measurement_range=10.0,
            resolution_bits=16,
            noise_standard_deviation=0.0,
            constant_bias=0.0,
            dropout_probability=0.0,
        )
    )

    measurement = imu.read(
        true_acceleration=20.0,
        timestamp=0.0,
    )

    assert measurement.saturated is True
    assert measurement.acceleration == pytest.approx(10.0)


def test_dropout_should_mark_measurement_as_invalid() -> None:
    """丢样时应将测量标记为无效。"""

    imu = VirtualIMU(
        VirtualIMUParameters(
            dropout_probability=1.0,
        )
    )

    measurement = imu.read(
        true_acceleration=5.0,
        timestamp=0.0,
    )

    assert measurement.valid is False
    assert measurement.acceleration == 0.0


def test_same_seed_should_generate_same_measurement_sequence() -> None:
    """相同种子应生成相同的噪声和丢样序列。"""

    parameters = VirtualIMUParameters(
        noise_standard_deviation=0.2,
        dropout_probability=0.1,
        seed=2026,
    )

    first_imu = VirtualIMU(parameters)
    second_imu = VirtualIMU(parameters)

    first_sequence = [first_imu.read(1.0, index * 0.001) for index in range(100)]

    second_sequence = [second_imu.read(1.0, index * 0.001) for index in range(100)]

    assert first_sequence == second_sequence


def test_reset_should_restore_original_random_sequence() -> None:
    """复位后应重新产生相同测量序列。"""

    imu = VirtualIMU(
        VirtualIMUParameters(
            noise_standard_deviation=0.2,
            seed=2026,
        )
    )

    first_sequence = [imu.read(1.0, index * 0.001) for index in range(20)]

    imu.reset()

    second_sequence = [imu.read(1.0, index * 0.001) for index in range(20)]

    assert first_sequence == second_sequence


def test_invalid_dropout_probability_should_raise_value_error() -> None:
    """丢样概率必须位于零到一之间。"""

    with pytest.raises(ValueError, match="dropout_probability"):
        VirtualIMUParameters(
            dropout_probability=1.5,
        )
