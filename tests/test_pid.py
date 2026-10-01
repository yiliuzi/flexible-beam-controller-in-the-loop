"""PID控制器测试。"""

import pytest

from controllers.pid import PIDController, PIDParameters


@pytest.fixture
def controller() -> PIDController:
    """创建通用测试控制器。"""

    return PIDController(
        PIDParameters(
            proportional_gain=2.0,
            integral_gain=1.0,
            derivative_gain=0.5,
            output_minimum=-5.0,
            output_maximum=5.0,
            integral_minimum=-1.0,
            integral_maximum=1.0,
        )
    )


def test_zero_error_should_generate_zero_output(
    controller: PIDController,
) -> None:
    """零误差应产生零控制量。"""

    output = controller.update(
        reference=0.0,
        measurement=0.0,
        time_step=0.001,
    )

    assert output == pytest.approx(0.0)


def test_positive_measurement_should_generate_negative_output(
    controller: PIDController,
) -> None:
    """位移为正且目标为零时，应产生负方向控制力。"""

    output = controller.update(
        reference=0.0,
        measurement=0.1,
        time_step=0.001,
    )

    assert output < 0.0


def test_output_should_respect_upper_limit() -> None:
    """控制输出不得超过上限。"""

    controller = PIDController(
        PIDParameters(
            proportional_gain=100.0,
            integral_gain=0.0,
            derivative_gain=0.0,
            output_minimum=-2.0,
            output_maximum=2.0,
        )
    )

    output = controller.update(
        reference=1.0,
        measurement=0.0,
        time_step=0.001,
    )

    assert output == pytest.approx(2.0)


def test_output_should_respect_lower_limit() -> None:
    """控制输出不得低于下限。"""

    controller = PIDController(
        PIDParameters(
            proportional_gain=100.0,
            integral_gain=0.0,
            derivative_gain=0.0,
            output_minimum=-2.0,
            output_maximum=2.0,
        )
    )

    output = controller.update(
        reference=-1.0,
        measurement=0.0,
        time_step=0.001,
    )

    assert output == pytest.approx(-2.0)


def test_integral_should_remain_inside_configured_limits(
    controller: PIDController,
) -> None:
    """积分状态不得超过配置范围。"""

    for _ in range(10000):
        controller.update(
            reference=0.5,
            measurement=0.0,
            time_step=0.001,
        )

    assert -1.0 <= controller.integral <= 1.0


def test_reset_should_clear_internal_state(
    controller: PIDController,
) -> None:
    """复位后应清除积分和历史误差。"""

    controller.update(
        reference=1.0,
        measurement=0.0,
        time_step=0.1,
    )

    controller.reset()

    assert controller.integral == 0.0
    assert controller.previous_error == 0.0
    assert controller.initialized is False


def test_invalid_time_step_should_raise_value_error(
    controller: PIDController,
) -> None:
    """控制周期必须大于零。"""

    with pytest.raises(ValueError, match="time_step"):
        controller.update(
            reference=0.0,
            measurement=0.0,
            time_step=0.0,
        )
