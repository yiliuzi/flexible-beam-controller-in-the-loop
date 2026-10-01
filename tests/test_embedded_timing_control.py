"""嵌入式时序闭环实验测试。"""

import numpy as np

from embedded.periodic_control_task import (
    ControlTaskParameters,
)
from experiments.embedded_timing_control import (
    run_embedded_timing_simulation,
)
from plant.beam_sdof import BeamParameters
from plant.disturbances import impulse_disturbance
from simulation.config import SimulationConfig


def disturbance(current_time: float) -> float:
    """生成测试使用的冲击扰动。"""

    return impulse_disturbance(
        time=current_time,
        start_time=0.05,
        duration=0.02,
        amplitude=1.50,
    )


def create_beam_parameters() -> BeamParameters:
    """创建统一的柔性梁参数。"""

    return BeamParameters(
        mass=0.08,
        damping=0.12,
        stiffness=18.0,
    )


def calculate_rms(values: np.ndarray) -> float:
    """计算信号均方根。"""

    return float(
        np.sqrt(
            np.mean(
                np.square(values),
            )
        )
    )


def test_control_task_execution_count_should_be_correct() -> None:
    """控制任务执行次数应符合设定周期。"""

    config = SimulationConfig(
        time_step=0.001,
        duration=0.10,
    )

    result = run_embedded_timing_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=config,
        task_parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.002,
            deadline=0.005,
        ),
        disturbance=disturbance,
    )

    assert result.execution_count == 21
    assert np.count_nonzero(result.task_executed) == 21


def test_normal_delay_should_not_miss_deadline() -> None:
    """计算延迟小于截止期时不能记录超时。"""

    config = SimulationConfig(
        time_step=0.001,
        duration=0.10,
    )

    result = run_embedded_timing_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=config,
        task_parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.002,
            deadline=0.005,
        ),
        disturbance=disturbance,
    )

    assert result.deadline_miss_count == 0
    assert not np.any(result.deadline_missed)


def test_excessive_delay_should_miss_every_deadline() -> None:
    """计算延迟超过截止期时每次任务均应超时。"""

    config = SimulationConfig(
        time_step=0.001,
        duration=0.10,
    )

    result = run_embedded_timing_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=config,
        task_parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.006,
            deadline=0.005,
        ),
        disturbance=disturbance,
    )

    assert result.execution_count == 21
    assert result.deadline_miss_count == 21

    assert np.count_nonzero(result.deadline_missed) == result.execution_count


def test_control_force_should_respect_actuator_limit() -> None:
    """周期控制情况下实际驱动力仍应满足限幅。"""

    config = SimulationConfig(
        time_step=0.001,
        duration=0.30,
    )

    result = run_embedded_timing_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=config,
        task_parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.002,
            deadline=0.005,
        ),
        disturbance=disturbance,
    )

    maximum_force = float(
        np.max(
            np.abs(
                result.simulation.control_force,
            )
        )
    )

    assert maximum_force <= 2.0
    assert np.all(
        np.isfinite(
            result.simulation.displacement,
        )
    )


def test_longer_delay_should_increase_vibration_rms() -> None:
    """持续超时应使振动抑制性能下降。"""

    config = SimulationConfig(
        time_step=0.001,
        duration=1.0,
    )

    normal_result = run_embedded_timing_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=config,
        task_parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.002,
            deadline=0.005,
        ),
        disturbance=disturbance,
    )

    delayed_result = run_embedded_timing_simulation(
        beam_parameters=create_beam_parameters(),
        simulation_config=config,
        task_parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.006,
            deadline=0.005,
        ),
        disturbance=disturbance,
    )

    evaluation_mask = normal_result.simulation.time >= 0.05

    normal_rms = calculate_rms(normal_result.simulation.displacement[evaluation_mask])

    delayed_rms = calculate_rms(delayed_result.simulation.displacement[evaluation_mask])

    assert delayed_rms > normal_rms
