"""CAN通信故障闭环实验测试。"""

import numpy as np

from communication.virtual_can_bus import (
    VirtualCanParameters,
)
from experiments.can_fault_control import (
    CanFaultConfiguration,
    run_can_control_simulation,
)
from plant.beam_sdof import BeamParameters
from plant.disturbances import impulse_disturbance
from simulation.config import SimulationConfig


def create_beam_parameters() -> BeamParameters:
    """创建统一柔性梁参数。"""

    return BeamParameters(
        mass=0.08,
        damping=0.12,
        stiffness=18.0,
    )


def create_disturbance(
    current_time: float,
) -> float:
    """创建短时间冲击扰动。"""

    return impulse_disturbance(
        time=current_time,
        start_time=0.03,
        duration=0.02,
        amplitude=1.50,
    )


def test_healthy_can_should_deliver_valid_frames() -> None:
    """健康CAN应正常发送且无丢包和损坏。"""

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
        disturbance=create_disturbance,
    )

    assert result.transmitted_frames == 21
    assert result.delivered_frames == 20
    assert result.dropped_frames == 0
    assert result.corrupted_frames == 0
    assert result.invalid_frames == 0
    assert result.lost_frames == 0

    assert np.any(result.valid_frame_received)


def test_startup_delay_should_create_initial_timeout() -> None:
    """第一帧到达前应短暂处于通信超时状态。"""

    result = run_can_control_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=SimulationConfig(
            time_step=0.001,
            duration=0.05,
        ),
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.020,
        ),
        disturbance=create_disturbance,
    )

    assert result.communication_timeout[0]
    assert result.communication_timeout[1]
    assert not result.communication_timeout[2]
    assert result.timeout_samples == 2


def test_disconnection_should_trigger_timeout() -> None:
    """总线断连时间超过阈值后必须触发超时。"""

    result = run_can_control_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=SimulationConfig(
            time_step=0.001,
            duration=0.15,
        ),
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.010,
            disconnect_start=0.05,
            disconnect_duration=0.05,
        ),
        disturbance=create_disturbance,
    )

    assert result.dropped_frames > 0
    assert result.timeout_samples > 2
    assert np.any(~result.bus_connected)
    assert np.any(result.communication_timeout)


def test_corrupted_frames_should_be_rejected() -> None:
    """CRC损坏报文不能用于控制。"""

    result = run_can_control_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=SimulationConfig(
            time_step=0.001,
            duration=0.05,
        ),
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.0,
            drop_probability=0.0,
            corruption_probability=1.0,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.010,
        ),
        disturbance=create_disturbance,
    )

    assert result.corrupted_frames > 0
    assert result.invalid_frames > 0
    assert result.corrupted_frames == result.invalid_frames

    assert not np.any(result.valid_frame_received)
    assert np.all(result.communication_timeout)


def test_control_force_should_remain_inside_safe_limit() -> None:
    """存在通信故障时驱动力仍不能超过安全限制。"""

    result = run_can_control_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=SimulationConfig(
            time_step=0.001,
            duration=0.30,
        ),
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
            drop_probability=0.10,
            corruption_probability=0.05,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.020,
            disconnect_start=0.15,
            disconnect_duration=0.05,
        ),
        disturbance=create_disturbance,
    )

    maximum_applied_force = float(np.max(np.abs(result.simulation.control_force)))

    assert maximum_applied_force <= 2.0

    assert np.all(np.isfinite(result.simulation.displacement))

    assert np.all(np.isfinite(result.simulation.velocity))
