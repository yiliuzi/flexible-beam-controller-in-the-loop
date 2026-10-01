"""Luenberger状态观测器测试。"""

import pytest

from controllers.state_observer import (
    LuenbergerObserver,
    ObserverParameters,
    ObserverState,
)
from plant.beam_sdof import BeamParameters, BeamState, SingleDegreeBeam


def test_invalid_mass_should_raise_value_error() -> None:
    """观测器质量参数必须大于零。"""

    with pytest.raises(ValueError, match="mass"):
        ObserverParameters(mass=0.0)


def test_zero_state_should_remain_at_zero() -> None:
    """零状态和零测量应保持在零。"""

    observer = LuenbergerObserver()

    for _ in range(1000):
        observer.update(
            measured_displacement=0.0,
            applied_control_force=0.0,
            time_step=0.001,
        )

    assert observer.state.displacement == pytest.approx(0.0)
    assert observer.state.velocity == pytest.approx(0.0)


def test_positive_measurement_should_correct_estimated_state() -> None:
    """正位移测量应推动估计状态向正方向修正。"""

    observer = LuenbergerObserver()

    result = observer.update(
        measured_displacement=0.01,
        applied_control_force=0.0,
        time_step=0.001,
    )

    assert result.displacement > 0.0
    assert result.velocity > 0.0
    assert observer.last_innovation == pytest.approx(0.01)


def test_invalid_measurement_should_use_model_prediction_only() -> None:
    """测量无效时应仅使用模型完成状态预测。"""

    observer = LuenbergerObserver(
        initial_state=ObserverState(
            displacement=0.01,
            velocity=0.0,
        )
    )

    result = observer.update(
        measured_displacement=1.0,
        applied_control_force=0.0,
        time_step=0.001,
        measurement_valid=False,
    )

    assert observer.last_innovation == 0.0
    assert result.displacement == pytest.approx(0.01)
    assert result.velocity < 0.0


def test_observer_should_converge_to_true_state() -> None:
    """使用理想位移测量时，估计状态应收敛到真实状态。"""

    parameters = BeamParameters(
        mass=0.08,
        damping=0.12,
        stiffness=18.0,
    )

    beam = SingleDegreeBeam(
        parameters=parameters,
        initial_state=BeamState(
            displacement=0.05,
            velocity=0.0,
        ),
    )

    observer = LuenbergerObserver(
        ObserverParameters(
            mass=parameters.mass,
            damping=parameters.damping,
            stiffness=parameters.stiffness,
            position_gain=58.5,
            velocity_gain=562.25,
        )
    )

    for _ in range(3000):
        observer.update(
            measured_displacement=beam.state.displacement,
            applied_control_force=0.0,
            time_step=0.001,
        )

        beam.step(
            time_step=0.001,
        )

    assert observer.state.displacement == pytest.approx(
        beam.state.displacement,
        abs=1e-4,
    )

    assert observer.state.velocity == pytest.approx(
        beam.state.velocity,
        abs=1e-3,
    )


def test_reset_should_clear_estimated_state() -> None:
    """复位应清除观测器内部状态。"""

    observer = LuenbergerObserver(
        initial_state=ObserverState(
            displacement=1.0,
            velocity=2.0,
        )
    )

    observer.reset()

    assert observer.state.displacement == 0.0
    assert observer.state.velocity == 0.0
    assert observer.last_innovation == 0.0
