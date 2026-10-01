"""单自由度柔性梁模型测试。"""

import pytest

from plant.beam_sdof import BeamParameters, BeamState, SingleDegreeBeam


def test_invalid_mass_should_raise_value_error() -> None:
    """质量必须大于零。"""

    with pytest.raises(ValueError, match="mass"):
        BeamParameters(mass=0.0)


def test_zero_state_without_force_should_remain_at_equilibrium() -> None:
    """无初始状态、无外力时，系统应保持静止。"""

    beam = SingleDegreeBeam()

    for _ in range(1000):
        beam.step(time_step=0.001)

    assert beam.state.displacement == pytest.approx(0.0)
    assert beam.state.velocity == pytest.approx(0.0)


def test_external_force_should_generate_expected_acceleration() -> None:
    """静止状态下的加速度应满足牛顿第二定律。"""

    parameters = BeamParameters(
        mass=0.5,
        damping=0.0,
        stiffness=10.0,
    )

    beam = SingleDegreeBeam(parameters=parameters)

    acceleration = beam.acceleration(disturbance_force=2.0)

    assert acceleration == pytest.approx(4.0)


def test_positive_displacement_should_generate_restoring_acceleration() -> None:
    """正位移应产生方向相反的恢复加速度。"""

    beam = SingleDegreeBeam(
        initial_state=BeamState(
            displacement=0.1,
            velocity=0.0,
        )
    )

    acceleration = beam.acceleration()

    assert acceleration < 0.0


def test_damping_should_reduce_mechanical_energy() -> None:
    """存在阻尼时，系统机械能应随时间衰减。"""

    beam = SingleDegreeBeam(
        parameters=BeamParameters(
            mass=0.08,
            damping=0.12,
            stiffness=18.0,
        ),
        initial_state=BeamState(
            displacement=0.1,
            velocity=0.0,
        ),
    )

    initial_energy = beam.mechanical_energy()

    for _ in range(5000):
        beam.step(time_step=0.001)

    final_energy = beam.mechanical_energy()

    assert final_energy < initial_energy
    assert final_energy < initial_energy * 0.01
