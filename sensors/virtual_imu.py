"""包含典型非理想因素的虚拟IMU。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VirtualIMUParameters:
    """虚拟IMU参数。"""

    measurement_range: float = 16.0 * 9.80665
    resolution_bits: int = 16
    noise_standard_deviation: float = 0.05
    constant_bias: float = 0.10
    dropout_probability: float = 0.01
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.measurement_range <= 0:
            raise ValueError("measurement_range must be greater than zero")

        if not 2 <= self.resolution_bits <= 32:
            raise ValueError("resolution_bits must be between 2 and 32")

        if self.noise_standard_deviation < 0:
            raise ValueError("noise_standard_deviation must not be negative")

        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be between zero and one")


@dataclass(frozen=True, slots=True)
class IMUMeasurement:
    """单次IMU测量结果。"""

    timestamp: float
    acceleration: float
    valid: bool
    saturated: bool


class VirtualIMU:
    """模拟噪声、零偏、量化、饱和和丢样的IMU。"""

    def __init__(
        self,
        parameters: VirtualIMUParameters | None = None,
    ) -> None:
        self.parameters = parameters or VirtualIMUParameters()
        self._generator = np.random.default_rng(self.parameters.seed)
        self._last_acceleration = 0.0

    @property
    def quantization_step(self) -> float:
        """返回传感器量化步长。"""

        level_count = 2**self.parameters.resolution_bits - 1

        return 2.0 * self.parameters.measurement_range / level_count

    def read(
        self,
        true_acceleration: float,
        timestamp: float,
    ) -> IMUMeasurement:
        """产生一次虚拟IMU测量。"""

        sample_is_dropped = self._generator.random() < self.parameters.dropout_probability

        if sample_is_dropped:
            return IMUMeasurement(
                timestamp=timestamp,
                acceleration=self._last_acceleration,
                valid=False,
                saturated=False,
            )

        noise = self._generator.normal(
            0.0,
            self.parameters.noise_standard_deviation,
        )

        unbounded_acceleration = true_acceleration + self.parameters.constant_bias + float(noise)

        saturated = abs(unbounded_acceleration) > self.parameters.measurement_range

        limited_acceleration = float(
            np.clip(
                unbounded_acceleration,
                -self.parameters.measurement_range,
                self.parameters.measurement_range,
            )
        )

        quantized_acceleration = (
            round(
                (limited_acceleration + self.parameters.measurement_range) / self.quantization_step
            )
            * self.quantization_step
            - self.parameters.measurement_range
        )

        self._last_acceleration = quantized_acceleration

        return IMUMeasurement(
            timestamp=timestamp,
            acceleration=quantized_acceleration,
            valid=True,
            saturated=saturated,
        )

    def reset(self) -> None:
        """复位随机发生器和历史测量值。"""

        self._generator = np.random.default_rng(self.parameters.seed)
        self._last_acceleration = 0.0
