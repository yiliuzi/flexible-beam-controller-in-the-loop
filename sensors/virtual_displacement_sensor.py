"""包含典型非理想因素的虚拟位移传感器。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DisplacementSensorParameters:
    """虚拟位移传感器参数。"""

    measurement_range: float = 0.10
    resolution_bits: int = 16
    noise_standard_deviation: float = 0.00005
    constant_bias: float = 0.00020
    dropout_probability: float = 0.005
    seed: int = 2027

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
class DisplacementMeasurement:
    """单次位移传感器测量结果。"""

    timestamp: float
    displacement: float
    valid: bool
    saturated: bool


class VirtualDisplacementSensor:
    """模拟噪声、零偏、量化、饱和和丢样的位移传感器。"""

    def __init__(
        self,
        parameters: DisplacementSensorParameters | None = None,
    ) -> None:
        self.parameters = parameters or DisplacementSensorParameters()

        self._generator = np.random.default_rng(self.parameters.seed)

        self._last_displacement = 0.0

    @property
    def quantization_step(self) -> float:
        """返回位移传感器量化步长。"""

        level_count = 2**self.parameters.resolution_bits - 1

        return 2.0 * self.parameters.measurement_range / level_count

    def read(
        self,
        true_displacement: float,
        timestamp: float,
    ) -> DisplacementMeasurement:
        """产生一次虚拟位移测量。"""

        sample_is_dropped = self._generator.random() < self.parameters.dropout_probability

        if sample_is_dropped:
            return DisplacementMeasurement(
                timestamp=timestamp,
                displacement=self._last_displacement,
                valid=False,
                saturated=False,
            )

        noise = self._generator.normal(
            0.0,
            self.parameters.noise_standard_deviation,
        )

        unbounded_displacement = true_displacement + self.parameters.constant_bias + float(noise)

        saturated = abs(unbounded_displacement) > self.parameters.measurement_range

        limited_displacement = float(
            np.clip(
                unbounded_displacement,
                -self.parameters.measurement_range,
                self.parameters.measurement_range,
            )
        )

        quantized_displacement = (
            round(
                (limited_displacement + self.parameters.measurement_range) / self.quantization_step
            )
            * self.quantization_step
            - self.parameters.measurement_range
        )

        self._last_displacement = quantized_displacement

        return DisplacementMeasurement(
            timestamp=timestamp,
            displacement=quantized_displacement,
            valid=True,
            saturated=saturated,
        )

    def reset(self) -> None:
        """复位随机发生器和历史测量。"""

        self._generator = np.random.default_rng(self.parameters.seed)

        self._last_displacement = 0.0
