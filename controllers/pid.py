"""可移植PID控制器。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PIDParameters:
    """PID控制器参数。"""

    proportional_gain: float
    integral_gain: float
    derivative_gain: float
    output_minimum: float = -5.0
    output_maximum: float = 5.0
    integral_minimum: float = -1.0
    integral_maximum: float = 1.0

    def __post_init__(self) -> None:
        if self.output_minimum >= self.output_maximum:
            raise ValueError("output_minimum must be smaller than output_maximum")

        if self.integral_minimum >= self.integral_maximum:
            raise ValueError("integral_minimum must be smaller than integral_maximum")


class PIDController:
    """带输出限幅和抗积分饱和的PID控制器。"""

    def __init__(self, parameters: PIDParameters) -> None:
        self.parameters = parameters
        self.integral = 0.0
        self.previous_error = 0.0
        self.initialized = False

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))

    def update(
        self,
        reference: float,
        measurement: float,
        time_step: float,
    ) -> float:
        """计算一个控制周期的输出。"""

        if time_step <= 0:
            raise ValueError("time_step must be greater than zero")

        error = reference - measurement

        derivative = 0.0
        if self.initialized:
            derivative = (error - self.previous_error) / time_step

        candidate_integral = self._clamp(
            self.integral + error * time_step,
            self.parameters.integral_minimum,
            self.parameters.integral_maximum,
        )

        candidate_output = (
            self.parameters.proportional_gain * error
            + self.parameters.integral_gain * candidate_integral
            + self.parameters.derivative_gain * derivative
        )

        limited_output = self._clamp(
            candidate_output,
            self.parameters.output_minimum,
            self.parameters.output_maximum,
        )

        output_is_saturated = limited_output != candidate_output
        error_drives_positive_saturation = (
            limited_output >= self.parameters.output_maximum and error > 0
        )
        error_drives_negative_saturation = (
            limited_output <= self.parameters.output_minimum and error < 0
        )

        if not output_is_saturated or not (
            error_drives_positive_saturation or error_drives_negative_saturation
        ):
            self.integral = candidate_integral

        output = (
            self.parameters.proportional_gain * error
            + self.parameters.integral_gain * self.integral
            + self.parameters.derivative_gain * derivative
        )

        self.previous_error = error
        self.initialized = True

        return self._clamp(
            output,
            self.parameters.output_minimum,
            self.parameters.output_maximum,
        )

    def reset(self) -> None:
        """清除控制器内部状态。"""

        self.integral = 0.0
        self.previous_error = 0.0
        self.initialized = False
