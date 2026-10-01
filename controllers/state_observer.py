"""柔性梁Luenberger状态观测器。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObserverParameters:
    """状态观测器参数。"""

    mass: float = 0.08
    damping: float = 0.12
    stiffness: float = 18.0
    position_gain: float = 58.5
    velocity_gain: float = 562.25

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be greater than zero")

        if self.damping < 0:
            raise ValueError("damping must not be negative")

        if self.stiffness <= 0:
            raise ValueError("stiffness must be greater than zero")

        if self.position_gain <= 0:
            raise ValueError("position_gain must be greater than zero")

        if self.velocity_gain <= 0:
            raise ValueError("velocity_gain must be greater than zero")


@dataclass(slots=True)
class ObserverState:
    """状态观测器估计结果。"""

    displacement: float = 0.0
    velocity: float = 0.0


class LuenbergerObserver:
    """根据位移测量估计柔性梁位移和速度。"""

    def __init__(
        self,
        parameters: ObserverParameters | None = None,
        initial_state: ObserverState | None = None,
    ) -> None:
        self.parameters = parameters or ObserverParameters()

        initial = initial_state or ObserverState()

        self.state = ObserverState(
            displacement=initial.displacement,
            velocity=initial.velocity,
        )

        self.last_innovation = 0.0

    def update(
        self,
        measured_displacement: float,
        applied_control_force: float,
        time_step: float,
        measurement_valid: bool = True,
    ) -> ObserverState:
        """推进一个观测器周期。"""

        if time_step <= 0:
            raise ValueError("time_step must be greater than zero")

        estimated_displacement = self.state.displacement
        estimated_velocity = self.state.velocity

        if measurement_valid:
            innovation = measured_displacement - estimated_displacement
        else:
            innovation = 0.0

        predicted_acceleration = (
            applied_control_force
            - self.parameters.damping * estimated_velocity
            - self.parameters.stiffness * estimated_displacement
        ) / self.parameters.mass

        displacement_derivative = estimated_velocity + self.parameters.position_gain * innovation

        velocity_derivative = predicted_acceleration + self.parameters.velocity_gain * innovation

        self.state.displacement = estimated_displacement + time_step * displacement_derivative

        self.state.velocity = estimated_velocity + time_step * velocity_derivative

        self.last_innovation = innovation

        return ObserverState(
            displacement=self.state.displacement,
            velocity=self.state.velocity,
        )

    def reset(
        self,
        state: ObserverState | None = None,
    ) -> None:
        """复位观测器状态。"""

        new_state = state or ObserverState()

        self.state = ObserverState(
            displacement=new_state.displacement,
            velocity=new_state.velocity,
        )

        self.last_innovation = 0.0
