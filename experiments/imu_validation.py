"""虚拟IMU测量验证实验。"""

from pathlib import Path

import numpy as np
import pandas as pd

from plant.beam_sdof import BeamParameters, SingleDegreeBeam
from plant.disturbances import impulse_disturbance
from sensors.virtual_imu import VirtualIMU, VirtualIMUParameters
from simulation.config import SimulationConfig
from simulation.runner import run_open_loop_simulation
from visualization.plots import plot_imu_measurement_comparison


def main() -> None:
    """运行虚拟IMU验证实验。"""

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "imu_measurement.csv"
    figure_path = project_root / "results" / "imu_measurement_comparison.png"

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

    result = run_open_loop_simulation(
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

    measured_acceleration = np.zeros_like(result.acceleration)
    valid_samples = np.zeros(len(result.time), dtype=np.bool_)
    saturated_samples = np.zeros(len(result.time), dtype=np.bool_)

    for sample_index, current_time in enumerate(result.time):
        measurement = imu.read(
            true_acceleration=float(result.acceleration[sample_index]),
            timestamp=float(current_time),
        )

        measured_acceleration[sample_index] = measurement.acceleration
        valid_samples[sample_index] = measurement.valid
        saturated_samples[sample_index] = measurement.saturated

    valid_error = measured_acceleration[valid_samples] - result.acceleration[valid_samples]

    measurement_rmse = float(np.sqrt(np.mean(np.square(valid_error))))

    dropout_count = int(np.count_nonzero(~valid_samples))
    saturation_count = int(np.count_nonzero(saturated_samples))
    dropout_rate = dropout_count / len(result.time) * 100.0

    output_table = pd.DataFrame(
        {
            "time_s": result.time,
            "true_acceleration_m_s2": result.acceleration,
            "measured_acceleration_m_s2": measured_acceleration,
            "valid": valid_samples,
            "saturated": saturated_samples,
        }
    )

    output_table.to_csv(
        data_path,
        index=False,
    )

    plot_imu_measurement_comparison(
        time=result.time,
        true_acceleration=result.acceleration,
        measured_acceleration=measured_acceleration,
        valid_samples=valid_samples,
        output_path=figure_path,
    )

    print("Virtual IMU validation completed")
    print(f"Total samples:       {len(result.time)}")
    print(f"Measurement RMSE:    {measurement_rmse:.6f} m/s^2")
    print(f"Dropped samples:     {dropout_count}")
    print(f"Dropout rate:        {dropout_rate:.3f}%")
    print(f"Saturated samples:   {saturation_count}")
    print(f"Measurement data:    {data_path}")
    print(f"Comparison figure:   {figure_path}")


if __name__ == "__main__":
    main()
