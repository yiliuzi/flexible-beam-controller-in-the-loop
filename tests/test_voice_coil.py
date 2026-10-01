"""虚拟音圈执行器测试。"""

import pytest

from actuators.voice_coil import (
    VoiceCoilActuator,
    VoiceCoilParameters,
)


def test_invalid_parameters_should_raise_error() -> None:
    with pytest.raises(ValueError):
        VoiceCoilParameters(maximum_force=0.0)

    with pytest.raises(ValueError):
        VoiceCoilParameters(time_constant=0.0)

    with pytest.raises(ValueError):
        VoiceCoilParameters(
            maximum_force=2.0,
            dead_zone=2.0,
        )


def test_force_inside_dead_zone_should_be_zero() -> None:
    actuator = VoiceCoilActuator(VoiceCoilParameters(dead_zone=0.1))

    output = actuator.update(
        requested_force=0.05,
        time_step=0.01,
    )

    assert output.inside_dead_zone
    assert output.target_force == pytest.approx(0.0)
    assert output.applied_force == pytest.approx(0.0)


def test_force_should_be_limited_by_maximum_force() -> None:
    actuator = VoiceCoilActuator(
        VoiceCoilParameters(
            maximum_force=2.0,
            dead_zone=0.0,
        )
    )

    output = actuator.update(
        requested_force=5.0,
        time_step=0.01,
    )

    assert output.saturated
    assert output.target_force == pytest.approx(2.0)
    assert 0.0 < output.applied_force < 2.0


def test_response_should_not_change_instantaneously() -> None:
    actuator = VoiceCoilActuator(
        VoiceCoilParameters(
            maximum_force=2.0,
            time_constant=0.1,
            dead_zone=0.0,
        )
    )

    output = actuator.update(
        requested_force=1.0,
        time_step=0.01,
    )

    assert 0.0 < output.applied_force < 1.0


def test_response_should_converge_to_target_force() -> None:
    actuator = VoiceCoilActuator(
        VoiceCoilParameters(
            maximum_force=2.0,
            time_constant=0.01,
            dead_zone=0.0,
        )
    )

    output = None

    for _ in range(100):
        output = actuator.update(
            requested_force=1.5,
            time_step=0.01,
        )

    assert output is not None
    assert output.applied_force == pytest.approx(
        1.5,
        abs=1e-6,
    )


def test_negative_force_should_be_supported() -> None:
    actuator = VoiceCoilActuator(VoiceCoilParameters(dead_zone=0.0))

    output = actuator.update(
        requested_force=-1.0,
        time_step=0.01,
    )

    assert output.target_force == pytest.approx(-1.0)
    assert output.applied_force < 0.0


def test_disabled_actuator_should_decay_toward_zero() -> None:
    actuator = VoiceCoilActuator(VoiceCoilParameters(dead_zone=0.0))

    for _ in range(20):
        actuator.update(
            requested_force=1.0,
            time_step=0.01,
        )

    previous_force = actuator.applied_force

    output = actuator.update(
        requested_force=1.0,
        time_step=0.01,
        enabled=False,
    )

    assert not output.enabled
    assert output.target_force == pytest.approx(0.0)
    assert abs(output.applied_force) < abs(previous_force)


def test_emergency_stop_should_immediately_clear_force() -> None:
    actuator = VoiceCoilActuator(VoiceCoilParameters(dead_zone=0.0))

    actuator.update(
        requested_force=1.0,
        time_step=0.01,
    )

    assert actuator.applied_force > 0.0

    actuator.emergency_stop()

    assert actuator.applied_force == pytest.approx(0.0)
