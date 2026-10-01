"""柔性梁外部扰动模型。"""

from dataclasses import dataclass, field
from math import pi, sin

import numpy as np


def impulse_disturbance(
    time: float,
    start_time: float,
    duration: float,
    amplitude: float,
) -> float:
    """生成有限持续时间的冲击扰动。"""

    if duration <= 0:
        raise ValueError("duration must be greater than zero")

    if start_time <= time < start_time + duration:
        return amplitude

    return 0.0


def sinusoidal_disturbance(
    time: float,
    amplitude: float,
    frequency: float,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> float:
    """生成指定时间区间内的正弦扰动。"""

    if frequency < 0:
        raise ValueError("frequency must not be negative")

    if time < start_time:
        return 0.0

    if end_time is not None and time >= end_time:
        return 0.0

    relative_time = time - start_time

    return amplitude * sin(2.0 * pi * frequency * relative_time)


@dataclass(slots=True)
class RandomDisturbance:
    """具有固定随机种子的高斯白噪声扰动。"""

    standard_deviation: float
    start_time: float = 0.0
    end_time: float | None = None
    seed: int = 42
    _generator: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.standard_deviation < 0:
            raise ValueError("standard_deviation must not be negative")

        self._generator = np.random.default_rng(self.seed)

    def __call__(self, time: float) -> float:
        """返回当前时刻的随机扰动力。"""

        if time < self.start_time:
            return 0.0

        if self.end_time is not None and time >= self.end_time:
            return 0.0

        return float(self._generator.normal(0.0, self.standard_deviation))
