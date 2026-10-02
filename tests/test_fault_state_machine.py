"""故障安全状态机测试。"""

import pytest

from safety.fault_state_machine import (
    FaultStateMachine,
    SafetyInput,
    SafetyParameters,
    SafetyState,
)


def send_valid_frames(
    state_machine: FaultStateMachine,
    count: int,
) -> None:
    """向状态机连续输入有效报文。"""

    for _ in range(count):
        state_machine.update(
            SafetyInput(
                valid_frame_received=True,
            )
        )


def test_invalid_parameters_should_raise_error() -> None:
    with pytest.raises(ValueError):
        SafetyParameters(degraded_failure_count=0)

    with pytest.raises(ValueError):
        SafetyParameters(
            degraded_failure_count=5,
            safe_stop_failure_count=2,
        )

    with pytest.raises(ValueError):
        SafetyParameters(degraded_force_scale=1.1)


def test_system_should_start_in_recovery_state() -> None:
    state_machine = FaultStateMachine()

    output = state_machine.update(SafetyInput())

    assert output.state is SafetyState.RECOVERY
    assert output.actuator_enabled
    assert output.force_scale == pytest.approx(0.25)


def test_valid_frames_should_enter_normal_state() -> None:
    state_machine = FaultStateMachine()

    send_valid_frames(
        state_machine,
        state_machine.parameters.normal_valid_frame_count,
    )

    assert state_machine.state is SafetyState.NORMAL


def test_repeated_failures_should_enter_degraded_state() -> None:
    state_machine = FaultStateMachine()

    send_valid_frames(
        state_machine,
        state_machine.parameters.normal_valid_frame_count,
    )

    for _ in range(state_machine.parameters.degraded_failure_count):
        output = state_machine.update(SafetyInput(sensor_fault=True))

    assert output.state is SafetyState.DEGRADED
    assert output.actuator_enabled
    assert output.force_scale == pytest.approx(0.50)


def test_continuous_failures_should_enter_safe_stop() -> None:
    state_machine = FaultStateMachine()

    send_valid_frames(
        state_machine,
        state_machine.parameters.normal_valid_frame_count,
    )

    for _ in range(state_machine.parameters.safe_stop_failure_count):
        output = state_machine.update(
            SafetyInput(
                communication_timed_out=True,
            )
        )

    assert output.state is SafetyState.SAFE_STOP
    assert not output.actuator_enabled
    assert output.force_scale == pytest.approx(0.0)


def test_emergency_stop_should_stop_immediately() -> None:
    state_machine = FaultStateMachine()

    send_valid_frames(
        state_machine,
        state_machine.parameters.normal_valid_frame_count,
    )

    output = state_machine.update(
        SafetyInput(
            emergency_stop_requested=True,
        )
    )

    assert output.state is SafetyState.SAFE_STOP
    assert not output.actuator_enabled


def test_safe_stop_should_require_recovery_frames() -> None:
    state_machine = FaultStateMachine()

    state_machine.update(
        SafetyInput(
            emergency_stop_requested=True,
        )
    )

    send_valid_frames(
        state_machine,
        state_machine.parameters.recovery_valid_frame_count,
    )

    assert state_machine.state is SafetyState.RECOVERY

    send_valid_frames(
        state_machine,
        state_machine.parameters.normal_valid_frame_count,
    )

    assert state_machine.state is SafetyState.NORMAL


def test_reset_should_restore_recovery_state() -> None:
    state_machine = FaultStateMachine()

    send_valid_frames(
        state_machine,
        state_machine.parameters.normal_valid_frame_count,
    )

    state_machine.reset()

    assert state_machine.state is SafetyState.RECOVERY
    assert state_machine.consecutive_failures == 0
    assert state_machine.consecutive_valid_frames == 0
