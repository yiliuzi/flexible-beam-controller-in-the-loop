"""虚拟IMU校准、滤波及故障检测实验。"""

from pathlib import Path

import numpy as np
import pandas as pd

from plant.beam_sdof import BeamParameters, SingleDegreeBeam
from plant.disturbances import impulse_disturbance
from sensors.imu_processor import (
    IMUProcessor,
    IMUProcessorParameters,
)
from sensors.virtual_imu import (
    IMUMeasurement,
    VirtualIMU,
    VirtualIMUParameters,
)
from simulation.config import SimulationConfig
from simulation.runner import run_open_loop_simulation
from visualization.plots import plot_imu_processing_comparison


def calculate_rmse(
    reference: np.ndarray,
    measurement: np.ndarray,
) -> float:
    """计算均方根误差。"""

    error = measurement - reference

    return float(np.sqrt(np.mean(np.square(error))))


def main() -> None:
    """运行完整虚拟IMU预处理实验。"""

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "imu_processing_comparison.csv"
    figure_path = project_root / "results" / "imu_processing_comparison.png"

    config = SimulationConfig(
        time_step=0.001,
        duration=5.0,
    )

    beam = SingleDegreeBeam(
        parameters=BeamParameters(
            mass=0.08,
            damping=0.12,
            stiffness=18.0,
        )
    )

    def disturbance(current_time: float) -> float:
        return impulse_disturbance(
            time=current_time,
            start_time=0.50,
            duration=0.05,
            amplitude=1.50,
        )

    beam_result = run_open_loop_simulation(
        beam=beam,
        config=config,
        disturbance=disturbance,
    )

    imu = VirtualIMU(
        VirtualIMUParameters(
            measurement_range=16.0 * 9.80665,
            resolution_bits=16,
            noise_standard_deviation=0.08,
            constant_bias=0.12,
            dropout_probability=0.01,
            seed=2026,
        )
    )

    processor = IMUProcessor(
        IMUProcessorParameters(
            sample_time=config.time_step,
            cutoff_frequency=30.0,
            calibration_sample_count=1000,
            maximum_consecutive_dropouts=5,
        )
    )

    calibration_timestamp = 0.0

    while not processor.calibrated:
        calibration_measurement = imu.read(
            true_acceleration=0.0,
            timestamp=calibration_timestamp,
        )

        processor.update_calibration(calibration_measurement)

        calibration_timestamp += config.time_step

    sample_count = len(beam_result.time)

    raw_acceleration = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    processed_acceleration = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    valid_samples = np.zeros(
        sample_count,
        dtype=np.bool_,
    )

    sensor_faults = np.zeros(
        sample_count,
        dtype=np.bool_,
    )

    consecutive_dropouts = np.zeros(
        sample_count,
        dtype=np.int32,
    )

    for sample_index, current_time in enumerate(beam_result.time):
        measurement = imu.read(
            true_acceleration=float(beam_result.acceleration[sample_index]),
            timestamp=float(current_time),
        )

        fault_injection_active = 2.000 <= current_time < 2.008

        if fault_injection_active:
            measurement = IMUMeasurement(
                timestamp=float(current_time),
                acceleration=measurement.acceleration,
                valid=False,
                saturated=False,
            )

        processed = processor.process(measurement)

        raw_acceleration[sample_index] = measurement.acceleration

        processed_acceleration[sample_index] = processed.acceleration

        valid_samples[sample_index] = processed.input_valid

        sensor_faults[sample_index] = processed.sensor_fault

        consecutive_dropouts[sample_index] = processed.consecutive_dropouts

    raw_rmse = calculate_rmse(
        beam_result.acceleration[valid_samples],
        raw_acceleration[valid_samples],
    )

    processed_rmse = calculate_rmse(
        beam_result.acceleration,
        processed_acceleration,
    )

    rmse_improvement = (raw_rmse - processed_rmse) / raw_rmse * 100.0

    dropout_count = int(np.count_nonzero(~valid_samples))

    sensor_fault_count = int(np.count_nonzero(sensor_faults))

    dropout_rate = dropout_count / sample_count * 100.0

    output_table = pd.DataFrame(
        {
            "time_s": beam_result.time,
            "true_acceleration_m_s2": (beam_result.acceleration),
            "raw_acceleration_m_s2": raw_acceleration,
            "processed_acceleration_m_s2": (processed_acceleration),
            "input_valid": valid_samples,
            "sensor_fault": sensor_faults,
            "consecutive_dropouts": consecutive_dropouts,
        }
    )

    output_table.to_csv(
        data_path,
        index=False,
    )

    plot_imu_processing_comparison(
        time=beam_result.time,
        true_acceleration=beam_result.acceleration,
        raw_acceleration=raw_acceleration,
        processed_acceleration=processed_acceleration,
        valid_samples=valid_samples,
        sensor_faults=sensor_faults,
        output_path=figure_path,
    )

    print("IMU processing experiment completed")
    print(f"Estimated bias:       {processor.bias:.6f} m/s^2")
    print(f"Configured bias:      {0.12:.6f} m/s^2")
    print(f"Raw IMU RMSE:         {raw_rmse:.6f} m/s^2")
    print(f"Processed IMU RMSE:   {processed_rmse:.6f} m/s^2")
    print(f"RMSE improvement:     {rmse_improvement:.2f}%")
    print(f"Dropped samples:      {dropout_count}")
    print(f"Dropout rate:         {dropout_rate:.3f}%")
    print(f"Fault-active samples: {sensor_fault_count}")
    print(f"Output data:          {data_path}")
    print(f"Output figure:        {figure_path}")


if __name__ == "__main__":
    main()
