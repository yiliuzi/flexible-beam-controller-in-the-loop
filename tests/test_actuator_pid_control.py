"""非理想音圈执行器闭环实验测试。"""

import numpy as np

from experiments.actuator_pid_control import (
    run_actuator_pid_simulation,
)
from plant.beam_sdof import BeamParameters, SingleDegreeBeam
from plant.disturbances import impulse_disturbance
from simulation.config import SimulationConfig
from simulation.runner import run_simulation


def create_test_configuration() -> tuple[
    BeamParameters,
    SimulationConfig,
]:
    """创建闭环测试使用的统一参数。"""

    beam_parameters = BeamParameters(
        mass=0.08,
        damping=0.12,
        stiffness=18.0,
    )

    config = SimulationConfig(
        time_step=0.001,
        duration=1.5,
    )

    return beam_parameters, config


def disturbance(current_time: float) -> float:
    """生成测试使用的冲击扰动。"""

    return impulse_disturbance(
        time=current_time,
        start_time=0.20,
        duration=0.05,
        amplitude=1.50,
    )


def test_actuator_pid_result_should_have_valid_shape() -> None:
    """闭环仿真结果长度应保持一致。"""

    beam_parameters, config = create_test_configuration()

    (
        result,
        requested_force,
        estimated_displacement,
        estimated_velocity,
        actuator_saturated,
        estimated_sensor_bias,
    ) = run_actuator_pid_simulation(
        beam_parameters=beam_parameters,
        config=config,
        disturbance=disturbance,
    )

    expected_sample_count = int(round(config.duration / config.time_step)) + 1

    assert len(result.time) == expected_sample_count
    assert len(result.displacement) == expected_sample_count
    assert len(result.control_force) == expected_sample_count
    assert len(requested_force) == expected_sample_count

    assert len(estimated_displacement) == expected_sample_count
    assert len(estimated_velocity) == expected_sample_count
    assert len(actuator_saturated) == expected_sample_count

    assert np.isfinite(estimated_sensor_bias)


def test_applied_force_should_respect_actuator_limit() -> None:
    """实际驱动力不能超过音圈执行器上限。"""

    beam_parameters, config = create_test_configuration()

    result, requested_force, _, _, actuator_saturated, _ = run_actuator_pid_simulation(
        beam_parameters=beam_parameters,
        config=config,
        disturbance=disturbance,
    )

    assert np.max(np.abs(result.control_force)) <= 2.0

    assert np.max(np.abs(requested_force)) <= 5.0

    assert np.any(np.abs(requested_force - result.control_force) > 1e-6)

    assert actuator_saturated.dtype == np.bool_


def test_actuator_pid_should_reduce_beam_vibration() -> None:
    """非理想执行器PID应降低扰动后的位移均方根。"""

    beam_parameters, config = create_test_configuration()

    open_loop_result = run_simulation(
        beam=SingleDegreeBeam(
            parameters=beam_parameters,
        ),
        config=config,
        disturbance=disturbance,
    )

    actuator_result, _, _, _, _, _ = run_actuator_pid_simulation(
        beam_parameters=beam_parameters,
        config=config,
        disturbance=disturbance,
    )

    evaluation_mask = open_loop_result.time >= 0.20

    open_loop_rms = float(
        np.sqrt(np.mean(np.square(open_loop_result.displacement[evaluation_mask])))
    )

    actuator_pid_rms = float(
        np.sqrt(np.mean(np.square(actuator_result.displacement[evaluation_mask])))
    )

    assert actuator_pid_rms < open_loop_rms


def test_state_estimates_should_remain_finite() -> None:
    """传感器丢包情况下状态估计不能发散。"""

    beam_parameters, config = create_test_configuration()

    (
        _,
        _,
        estimated_displacement,
        estimated_velocity,
        _,
        _,
    ) = run_actuator_pid_simulation(
        beam_parameters=beam_parameters,
        config=config,
        disturbance=disturbance,
    )

    assert np.all(np.isfinite(estimated_displacement))
    assert np.all(np.isfinite(estimated_velocity))

    assert np.max(np.abs(estimated_displacement)) < 0.10
    assert np.max(np.abs(estimated_velocity)) < 5.0
