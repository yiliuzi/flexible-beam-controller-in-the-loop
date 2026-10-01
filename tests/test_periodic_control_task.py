"""嵌入式周期控制任务测试。"""

import pytest

from embedded.periodic_control_task import (
    ControlTaskParameters,
    PeriodicControlTask,
)


def proportional_controller(
    measurement: float,
    time_step: float,
) -> float:
    """测试使用的简单比例控制器。"""

    del time_step
    return -2.0 * measurement


def test_invalid_task_parameters_should_raise_error() -> None:
    with pytest.raises(ValueError):
        ControlTaskParameters(control_period=0.0)

    with pytest.raises(ValueError):
        ControlTaskParameters(computation_delay=-0.001)

    with pytest.raises(ValueError):
        ControlTaskParameters(deadline=0.0)


def test_task_should_execute_at_configured_period() -> None:
    task = PeriodicControlTask(
        controller=proportional_controller,
        parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.0,
            deadline=0.005,
        ),
    )

    first_output = task.update(
        timestamp=0.0,
        measurement=1.0,
    )

    intermediate_output = task.update(
        timestamp=0.001,
        measurement=2.0,
    )

    second_output = task.update(
        timestamp=0.005,
        measurement=2.0,
    )

    assert first_output.task_executed
    assert not intermediate_output.task_executed
    assert second_output.task_executed

    assert task.execution_count == 2


def test_zero_order_hold_should_keep_previous_command() -> None:
    task = PeriodicControlTask(
        controller=proportional_controller,
        parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.0,
        ),
    )

    first_output = task.update(
        timestamp=0.0,
        measurement=1.0,
    )

    held_output = task.update(
        timestamp=0.001,
        measurement=5.0,
    )

    assert first_output.applied_command == pytest.approx(-2.0)

    assert held_output.applied_command == pytest.approx(-2.0)


def test_computation_delay_should_delay_new_command() -> None:
    task = PeriodicControlTask(
        controller=proportional_controller,
        parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.002,
        ),
    )

    release_output = task.update(
        timestamp=0.0,
        measurement=1.0,
    )

    before_activation = task.update(
        timestamp=0.001,
        measurement=1.0,
    )

    activation_output = task.update(
        timestamp=0.002,
        measurement=1.0,
    )

    assert release_output.applied_command == pytest.approx(0.0)

    assert before_activation.applied_command == pytest.approx(0.0)

    assert activation_output.applied_command == pytest.approx(-2.0)

    assert activation_output.command_activated


def test_deadline_miss_should_be_counted() -> None:
    task = PeriodicControlTask(
        controller=proportional_controller,
        parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.006,
            deadline=0.005,
        ),
    )

    output = task.update(
        timestamp=0.0,
        measurement=1.0,
    )

    assert output.deadline_missed
    assert output.deadline_miss_count == 1
    assert task.deadline_miss_count == 1


def test_decreasing_timestamp_should_raise_error() -> None:
    task = PeriodicControlTask(
        controller=proportional_controller,
    )

    task.update(
        timestamp=0.010,
        measurement=1.0,
    )

    with pytest.raises(ValueError):
        task.update(
            timestamp=0.005,
            measurement=1.0,
        )


def test_reset_should_clear_task_state() -> None:
    task = PeriodicControlTask(
        controller=proportional_controller,
        parameters=ControlTaskParameters(
            computation_delay=0.0,
        ),
    )

    task.update(
        timestamp=0.0,
        measurement=1.0,
    )

    task.reset()

    assert task.execution_count == 0
    assert task.deadline_miss_count == 0
    assert task.applied_command == pytest.approx(0.0)
    assert task.next_release_time == pytest.approx(0.0)
