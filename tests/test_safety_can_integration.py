"""CAN通信与安全状态机集成测试。"""

import numpy as np

from communication.virtual_can_bus import (
    VirtualCanParameters,
)
from experiments.can_fault_control import (
    CanFaultConfiguration,
    run_can_control_simulation,
)
from plant.beam_sdof import BeamParameters
from simulation.config import SimulationConfig


def zero_disturbance(current_time: float) -> float:
    """返回零扰动。"""

    del current_time
    return 0.0


def create_beam_parameters() -> BeamParameters:
    """创建统一柔性梁参数。"""

    return BeamParameters(
        mass=0.08,
        damping=0.12,
        stiffness=18.0,
    )


def test_healthy_can_should_reach_normal_state() -> None:
    """健康通信建立后系统应进入正常状态。"""

    result = run_can_control_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=SimulationConfig(
            time_step=0.001,
            duration=0.10,
        ),
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
            drop_probability=0.0,
            corruption_probability=0.0,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.020,
        ),
        disturbance=zero_disturbance,
    )

    assert "RECOVERY" in result.safety_state
    assert "NORMAL" in result.safety_state
    assert result.safety_state[-1] == "NORMAL"


def test_long_disconnection_should_enter_safe_stop() -> None:
    """长时间CAN断连应触发安全停机。"""

    result = run_can_control_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=SimulationConfig(
            time_step=0.001,
            duration=0.20,
        ),
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.010,
            disconnect_start=0.06,
            disconnect_duration=0.06,
        ),
        disturbance=zero_disturbance,
    )

    assert "DEGRADED" in result.safety_state
    assert "SAFE_STOP" in result.safety_state
    assert np.any(~result.bus_connected)
    assert np.any(result.communication_timeout)


def test_system_should_recover_after_bus_reconnection() -> None:
    """CAN恢复后系统应经过恢复状态返回正常。"""

    result = run_can_control_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=SimulationConfig(
            time_step=0.001,
            duration=0.25,
        ),
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.010,
            disconnect_start=0.06,
            disconnect_duration=0.06,
        ),
        disturbance=zero_disturbance,
    )

    safe_stop_indices = np.flatnonzero(result.safety_state == "SAFE_STOP")

    assert len(safe_stop_indices) > 0

    last_safe_stop_index = int(safe_stop_indices[-1])

    states_after_stop = result.safety_state[last_safe_stop_index + 1 :]

    assert "RECOVERY" in states_after_stop
    assert "NORMAL" in states_after_stop
    assert result.safety_state[-1] == "NORMAL"


def test_force_scale_should_match_safety_state() -> None:
    """每个安全状态应使用对应的控制力比例。"""

    result = run_can_control_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=SimulationConfig(
            time_step=0.001,
            duration=0.20,
        ),
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.010,
            disconnect_start=0.06,
            disconnect_duration=0.06,
        ),
        disturbance=zero_disturbance,
    )

    expected_scales = {
        "NORMAL": 1.0,
        "DEGRADED": 0.50,
        "SAFE_STOP": 0.0,
        "RECOVERY": 0.25,
    }

    for state_name, expected_scale in expected_scales.items():
        state_mask = result.safety_state == state_name

        if np.any(state_mask):
            assert np.allclose(
                result.safety_force_scale[state_mask],
                expected_scale,
            )


def test_safe_stop_should_request_zero_force() -> None:
    """安全停机状态下请求控制力必须为零。"""

    result = run_can_control_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=SimulationConfig(
            time_step=0.001,
            duration=0.20,
        ),
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.010,
            disconnect_start=0.06,
            disconnect_duration=0.06,
        ),
        disturbance=zero_disturbance,
    )

    safe_stop_mask = result.safety_state == "SAFE_STOP"

    assert np.any(safe_stop_mask)

    assert np.allclose(
        result.requested_force[safe_stop_mask],
        0.0,
    )

    assert np.max(np.abs(result.simulation.control_force)) <= 2.0
