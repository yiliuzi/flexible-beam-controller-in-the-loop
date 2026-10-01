"""IMU数据预处理模块测试。"""

import pytest

from sensors.imu_processor import (
    IMUProcessor,
    IMUProcessorParameters,
)
from sensors.virtual_imu import IMUMeasurement


def create_measurement(
    acceleration: float,
    timestamp: float = 0.0,
    valid: bool = True,
    saturated: bool = False,
) -> IMUMeasurement:
    """创建测试用IMU测量值。"""

    return IMUMeasurement(
        timestamp=timestamp,
        acceleration=acceleration,
        valid=valid,
        saturated=saturated,
    )


def create_calibrated_processor() -> IMUProcessor:
    """创建完成零偏校准的处理器。"""

    processor = IMUProcessor(
        IMUProcessorParameters(
            sample_time=0.001,
            cutoff_frequency=30.0,
            calibration_sample_count=3,
            maximum_consecutive_dropouts=3,
        )
    )

    for index in range(3):
        processor.update_calibration(
            create_measurement(
                acceleration=0.2,
                timestamp=index * 0.001,
            )
        )

    return processor


def test_calibration_should_estimate_constant_bias() -> None:
    """校准结果应等于静止测量值的平均值。"""

    processor = create_calibrated_processor()

    assert processor.calibrated is True
    assert processor.bias == pytest.approx(0.2)


def test_invalid_measurement_should_not_be_used_for_calibration() -> None:
    """无效样本不应进入零偏计算。"""

    processor = IMUProcessor(
        IMUProcessorParameters(
            calibration_sample_count=2,
        )
    )

    processor.update_calibration(
        create_measurement(
            acceleration=100.0,
            valid=False,
        )
    )

    processor.update_calibration(create_measurement(acceleration=0.1))

    assert processor.calibrated is False

    processor.update_calibration(create_measurement(acceleration=0.3))

    assert processor.calibrated is True
    assert processor.bias == pytest.approx(0.2)


def test_processing_before_calibration_should_raise_error() -> None:
    """未完成校准时不得输出处理结果。"""

    processor = IMUProcessor()

    with pytest.raises(RuntimeError, match="calibrated"):
        processor.process(create_measurement(acceleration=1.0))


def test_bias_should_be_removed_after_calibration() -> None:
    """处理器应从测量值中减去估计零偏。"""

    processor = create_calibrated_processor()

    result = processor.process(create_measurement(acceleration=1.2))

    assert result.acceleration == pytest.approx(1.0)
    assert result.sensor_fault is False


def test_low_pass_filter_should_smooth_step_input() -> None:
    """低通滤波后的输出应平滑接近输入。"""

    processor = create_calibrated_processor()

    first_result = processor.process(create_measurement(acceleration=0.2))

    second_result = processor.process(
        create_measurement(
            acceleration=10.2,
            timestamp=0.001,
        )
    )

    assert first_result.acceleration == pytest.approx(0.0)
    assert 0.0 < second_result.acceleration < 10.0


def test_invalid_measurement_should_hold_previous_output() -> None:
    """发生丢样时应保持上一次滤波输出。"""

    processor = create_calibrated_processor()

    valid_result = processor.process(create_measurement(acceleration=1.2))

    dropout_result = processor.process(
        create_measurement(
            acceleration=0.0,
            timestamp=0.001,
            valid=False,
        )
    )

    assert dropout_result.input_valid is False
    assert dropout_result.acceleration == pytest.approx(valid_result.acceleration)
    assert dropout_result.consecutive_dropouts == 1


def test_consecutive_dropouts_should_trigger_sensor_fault() -> None:
    """连续丢样达到阈值时应触发传感器故障。"""

    processor = create_calibrated_processor()

    processor.process(
        create_measurement(
            acceleration=0.0,
            valid=False,
        )
    )

    processor.process(
        create_measurement(
            acceleration=0.0,
            timestamp=0.001,
            valid=False,
        )
    )

    result = processor.process(
        create_measurement(
            acceleration=0.0,
            timestamp=0.002,
            valid=False,
        )
    )

    assert result.sensor_fault is True
    assert result.consecutive_dropouts == 3


def test_valid_measurement_should_clear_dropout_counter() -> None:
    """恢复有效测量后应清除连续丢样计数。"""

    processor = create_calibrated_processor()

    processor.process(
        create_measurement(
            acceleration=0.0,
            valid=False,
        )
    )

    result = processor.process(
        create_measurement(
            acceleration=0.2,
            timestamp=0.001,
            valid=True,
        )
    )

    assert result.consecutive_dropouts == 0
    assert result.sensor_fault is False


def test_reset_should_clear_all_processor_states() -> None:
    """复位应清除校准及故障状态。"""

    processor = create_calibrated_processor()

    processor.process(
        create_measurement(
            acceleration=0.0,
            valid=False,
        )
    )

    processor.reset()

    assert processor.calibrated is False
    assert processor.bias == 0.0
    assert processor.consecutive_dropouts == 0
