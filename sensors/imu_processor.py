"""面向嵌入式部署的IMU数据预处理模块。"""

from dataclasses import dataclass
from math import pi

from sensors.virtual_imu import IMUMeasurement


@dataclass(frozen=True, slots=True)
class IMUProcessorParameters:
    """IMU预处理器参数。"""

    sample_time: float = 0.001
    cutoff_frequency: float = 30.0
    calibration_sample_count: int = 1000
    maximum_consecutive_dropouts: int = 5

    def __post_init__(self) -> None:
        if self.sample_time <= 0:
            raise ValueError("sample_time must be greater than zero")

        if self.cutoff_frequency <= 0:
            raise ValueError("cutoff_frequency must be greater than zero")

        if self.calibration_sample_count <= 0:
            raise ValueError("calibration_sample_count must be greater than zero")

        if self.maximum_consecutive_dropouts <= 0:
            raise ValueError("maximum_consecutive_dropouts must be greater than zero")


@dataclass(frozen=True, slots=True)
class ProcessedIMUMeasurement:
    """IMU预处理结果。"""

    timestamp: float
    acceleration: float
    input_valid: bool
    calibrated: bool
    sensor_fault: bool
    consecutive_dropouts: int


class IMUProcessor:
    """执行零偏校准、滤波、丢样保持及故障检测。"""

    def __init__(
        self,
        parameters: IMUProcessorParameters | None = None,
    ) -> None:
        self.parameters = parameters or IMUProcessorParameters()

        self._calibration_sum = 0.0
        self._calibration_count = 0
        self._bias = 0.0
        self._calibrated = False

        self._filter_initialized = False
        self._filtered_acceleration = 0.0
        self._consecutive_dropouts = 0

    @property
    def calibrated(self) -> bool:
        """返回校准是否完成。"""

        return self._calibrated

    @property
    def bias(self) -> float:
        """返回估计得到的传感器零偏。"""

        return self._bias

    @property
    def consecutive_dropouts(self) -> int:
        """返回当前连续丢样次数。"""

        return self._consecutive_dropouts

    @property
    def filter_coefficient(self) -> float:
        """计算一阶低通滤波器系数。"""

        time_constant = 1.0 / (2.0 * pi * self.parameters.cutoff_frequency)

        return self.parameters.sample_time / (time_constant + self.parameters.sample_time)

    def update_calibration(
        self,
        measurement: IMUMeasurement,
    ) -> bool:
        """使用静止状态的有效样本更新零偏校准。"""

        if self._calibrated:
            return True

        if not measurement.valid or measurement.saturated:
            return False

        self._calibration_sum += measurement.acceleration
        self._calibration_count += 1

        if self._calibration_count >= self.parameters.calibration_sample_count:
            self._bias = self._calibration_sum / self._calibration_count
            self._calibrated = True

        return self._calibrated

    def process(
        self,
        measurement: IMUMeasurement,
    ) -> ProcessedIMUMeasurement:
        """处理一次IMU测量。"""

        if not self._calibrated:
            raise RuntimeError("IMU processor must be calibrated before processing")

        if not measurement.valid:
            self._consecutive_dropouts += 1

            return ProcessedIMUMeasurement(
                timestamp=measurement.timestamp,
                acceleration=self._filtered_acceleration,
                input_valid=False,
                calibrated=True,
                sensor_fault=(
                    self._consecutive_dropouts >= self.parameters.maximum_consecutive_dropouts
                ),
                consecutive_dropouts=self._consecutive_dropouts,
            )

        self._consecutive_dropouts = 0
        corrected_acceleration = measurement.acceleration - self._bias

        if not self._filter_initialized:
            self._filtered_acceleration = corrected_acceleration
            self._filter_initialized = True
        else:
            coefficient = self.filter_coefficient

            self._filtered_acceleration += coefficient * (
                corrected_acceleration - self._filtered_acceleration
            )

        return ProcessedIMUMeasurement(
            timestamp=measurement.timestamp,
            acceleration=self._filtered_acceleration,
            input_valid=True,
            calibrated=True,
            sensor_fault=measurement.saturated,
            consecutive_dropouts=0,
        )

    def reset(self) -> None:
        """清除校准、滤波及故障状态。"""

        self._calibration_sum = 0.0
        self._calibration_count = 0
        self._bias = 0.0
        self._calibrated = False

        self._filter_initialized = False
        self._filtered_acceleration = 0.0
        self._consecutive_dropouts = 0
