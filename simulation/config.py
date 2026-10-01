"""仿真参数配置。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """固定步长仿真配置。"""

    time_step: float = 0.001
    duration: float = 5.0

    def __post_init__(self) -> None:
        if self.time_step <= 0:
            raise ValueError("time_step must be greater than zero")

        if self.duration <= 0:
            raise ValueError("duration must be greater than zero")

        if self.time_step > self.duration:
            raise ValueError("time_step must not be greater than duration")
