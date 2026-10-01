"""柔性梁单自由度等效动力学模型。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BeamParameters:
    """单自由度柔性梁等效参数。"""

    mass: float = 0.08
    damping: float = 0.12
    stiffness: float = 18.0

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be greater than zero")

        if self.damping < 0:
            raise ValueError("damping must not be negative")

        if self.stiffness <= 0:
            raise ValueError("stiffness must be greater than zero")


@dataclass(slots=True)
class BeamState:
    """柔性梁状态。"""

    displacement: float = 0.0
    velocity: float = 0.0


class SingleDegreeBeam:
    """单自由度柔性梁数字孪生模型。

    动力学方程：

        m * y_ddot + c * y_dot + k * y = u + d

    其中：
        y     表示梁端等效位移；
        y_dot 表示梁端等效速度；
        u     表示控制器输出力；
        d     表示外部扰动力。
    """

    def __init__(
        self,
        parameters: BeamParameters | None = None,
        initial_state: BeamState | None = None,
    ) -> None:
        self.parameters = parameters or BeamParameters()
        self.state = initial_state or BeamState()

    def acceleration(
        self,
        state: BeamState | None = None,
        control_force: float = 0.0,
        disturbance_force: float = 0.0,
    ) -> float:
        """计算当前状态下的加速度。"""

        current_state = state or self.state
        total_force = control_force + disturbance_force

        return (
            total_force
            - self.parameters.damping * current_state.velocity
            - self.parameters.stiffness * current_state.displacement
        ) / self.parameters.mass

    def step(
        self,
        time_step: float,
        control_force: float = 0.0,
        disturbance_force: float = 0.0,
    ) -> BeamState:
        """使用四阶Runge-Kutta方法推进一个仿真周期。"""

        if time_step <= 0:
            raise ValueError("time_step must be greater than zero")

        initial_displacement = self.state.displacement
        initial_velocity = self.state.velocity
        total_force = control_force + disturbance_force

        def derivatives(displacement: float, velocity: float) -> tuple[float, float]:
            acceleration = (
                total_force
                - self.parameters.damping * velocity
                - self.parameters.stiffness * displacement
            ) / self.parameters.mass

            return velocity, acceleration

        k1_displacement, k1_velocity = derivatives(
            initial_displacement,
            initial_velocity,
        )

        k2_displacement, k2_velocity = derivatives(
            initial_displacement + 0.5 * time_step * k1_displacement,
            initial_velocity + 0.5 * time_step * k1_velocity,
        )

        k3_displacement, k3_velocity = derivatives(
            initial_displacement + 0.5 * time_step * k2_displacement,
            initial_velocity + 0.5 * time_step * k2_velocity,
        )

        k4_displacement, k4_velocity = derivatives(
            initial_displacement + time_step * k3_displacement,
            initial_velocity + time_step * k3_velocity,
        )

        self.state.displacement = (
            initial_displacement
            + time_step
            * (k1_displacement + 2.0 * k2_displacement + 2.0 * k3_displacement + k4_displacement)
            / 6.0
        )

        self.state.velocity = (
            initial_velocity
            + time_step * (k1_velocity + 2.0 * k2_velocity + 2.0 * k3_velocity + k4_velocity) / 6.0
        )

        return BeamState(
            displacement=self.state.displacement,
            velocity=self.state.velocity,
        )

    def mechanical_energy(self) -> float:
        """计算当前等效机械能。"""

        kinetic_energy = 0.5 * self.parameters.mass * self.state.velocity * self.state.velocity

        potential_energy = (
            0.5 * self.parameters.stiffness * self.state.displacement * self.state.displacement
        )

        return kinetic_energy + potential_energy

    def reset(self, state: BeamState | None = None) -> None:
        """恢复到指定状态或静止状态。"""

        new_state = state or BeamState()

        self.state = BeamState(
            displacement=new_state.displacement,
            velocity=new_state.velocity,
        )
