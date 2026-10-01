"""外部扰动模型测试。"""

import pytest

from plant.disturbances import (
    RandomDisturbance,
    impulse_disturbance,
    sinusoidal_disturbance,
)


def test_impulse_should_only_exist_inside_configured_interval() -> None:
    """冲击扰动只应在指定区间内存在。"""

    assert impulse_disturbance(0.49, 0.50, 0.05, 1.5) == 0.0
    assert impulse_disturbance(0.50, 0.50, 0.05, 1.5) == 1.5
    assert impulse_disturbance(0.54, 0.50, 0.05, 1.5) == 1.5
    assert impulse_disturbance(0.55, 0.50, 0.05, 1.5) == 0.0


def test_invalid_impulse_duration_should_raise_value_error() -> None:
    """冲击持续时间必须大于零。"""

    with pytest.raises(ValueError, match="duration"):
        impulse_disturbance(0.0, 0.0, 0.0, 1.0)


def test_sinusoidal_disturbance_should_start_from_zero() -> None:
    """正弦扰动起始点应为零。"""

    force = sinusoidal_disturbance(
        time=1.0,
        amplitude=2.0,
        frequency=5.0,
        start_time=1.0,
    )

    assert force == pytest.approx(0.0, abs=1e-12)


def test_sinusoidal_disturbance_should_be_inactive_outside_interval() -> None:
    """正弦扰动在有效区间外应为零。"""

    assert sinusoidal_disturbance(0.5, 1.0, 2.0, start_time=1.0) == 0.0

    assert (
        sinusoidal_disturbance(
            3.0,
            1.0,
            2.0,
            start_time=1.0,
            end_time=3.0,
        )
        == 0.0
    )


def test_random_disturbance_should_be_repeatable_with_same_seed() -> None:
    """相同种子必须产生相同的随机扰动序列。"""

    first = RandomDisturbance(
        standard_deviation=0.1,
        seed=2026,
    )

    second = RandomDisturbance(
        standard_deviation=0.1,
        seed=2026,
    )

    first_values = [first(index * 0.001) for index in range(10)]
    second_values = [second(index * 0.001) for index in range(10)]

    assert first_values == second_values
