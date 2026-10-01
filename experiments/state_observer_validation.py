"""Luenberger状态观测器验证实验。"""

from pathlib import Path

import numpy as np
import pandas as pd

from controllers.state_observer import (
    LuenbergerObserver,
    ObserverParameters,
)
from plant.beam_sdof import BeamParameters, SingleDegreeBeam
from plant.disturbances import impulse_disturbance
from sensors.virtual_displacement_sensor import (
    DisplacementSensorParameters,
    VirtualDisplacementSensor,
)
from simulation.config import SimulationConfig
from simulation.runner import run_open_loop_simulation
from visualization.plots import plot_state_observer_validation


def calculate_rmse(
    reference: np.ndarray,
    estimation: np.ndarray,
) -> float:
    """计算估计结果的均方根误差。"""

    error = estimation - reference

    return float(np.sqrt(np.mean(np.square(error))))


def main() -> None:
    """运行带噪位移测量下的状态观测器实验。"""

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "state_observer_validation.csv"
    figure_path = project_root / "results" / "state_observer_validation.png"

    config = SimulationConfig(
        time_step=0.001,
        duration=5.0,
    )

    beam_parameters = BeamParameters(
        mass=0.08,
        damping=0.12,
        stiffness=18.0,
    )

    beam = SingleDegreeBeam(
        parameters=beam_parameters,
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

    sensor = VirtualDisplacementSensor(
        DisplacementSensorParameters(
            measurement_range=0.10,
            resolution_bits=16,
            noise_standard_deviation=0.00005,
            constant_bias=0.00020,
            dropout_probability=0.005,
            seed=2027,
        )
    )

    calibration_values: list[float] = []

    while len(calibration_values) < 1000:
        measurement = sensor.read(
            true_displacement=0.0,
            timestamp=0.0,
        )

        if measurement.valid and not measurement.saturated:
            calibration_values.append(measurement.displacement)

    estimated_sensor_bias = float(np.mean(calibration_values))

    observer = LuenbergerObserver(
        ObserverParameters(
            mass=beam_parameters.mass,
            damping=beam_parameters.damping,
            stiffness=beam_parameters.stiffness,
            position_gain=58.5,
            velocity_gain=562.25,
        )
    )

    sample_count = len(result.time)

    measured_displacement = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    estimated_displacement = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    estimated_velocity = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    valid_measurements = np.zeros(
        sample_count,
        dtype=np.bool_,
    )

    for sample_index, current_time in enumerate(result.time):
        measurement = sensor.read(
            true_displacement=float(result.displacement[sample_index]),
            timestamp=float(current_time),
        )

        corrected_measurement = measurement.displacement - estimated_sensor_bias

        observer_state = observer.update(
            measured_displacement=corrected_measurement,
            applied_control_force=0.0,
            time_step=config.time_step,
            measurement_valid=measurement.valid,
        )

        measured_displacement[sample_index] = corrected_measurement

        estimated_displacement[sample_index] = observer_state.displacement

        estimated_velocity[sample_index] = observer_state.velocity

        valid_measurements[sample_index] = measurement.valid

    evaluation_mask = result.time >= 0.60

    displacement_rmse = calculate_rmse(
        result.displacement[evaluation_mask],
        estimated_displacement[evaluation_mask],
    )

    velocity_rmse = calculate_rmse(
        result.velocity[evaluation_mask],
        estimated_velocity[evaluation_mask],
    )

    maximum_displacement_error = float(np.max(np.abs(estimated_displacement - result.displacement)))

    dropout_count = int(np.count_nonzero(~valid_measurements))

    output_table = pd.DataFrame(
        {
            "time_s": result.time,
            "true_displacement_m": result.displacement,
            "measured_displacement_m": (measured_displacement),
            "estimated_displacement_m": (estimated_displacement),
            "true_velocity_m_s": result.velocity,
            "estimated_velocity_m_s": estimated_velocity,
            "measurement_valid": valid_measurements,
        }
    )

    output_table.to_csv(
        data_path,
        index=False,
    )

    plot_state_observer_validation(
        time=result.time,
        true_displacement=result.displacement,
        estimated_displacement=estimated_displacement,
        true_velocity=result.velocity,
        estimated_velocity=estimated_velocity,
        valid_measurements=valid_measurements,
        output_path=figure_path,
    )

    print("State observer validation completed")
    print(f"Estimated sensor bias:       {estimated_sensor_bias * 1000.0:.4f} mm")
    print(f"Displacement RMSE:           {displacement_rmse * 1000.0:.4f} mm")
    print(f"Velocity RMSE:               {velocity_rmse:.6f} m/s")
    print(f"Maximum displacement error:  {maximum_displacement_error * 1000.0:.4f} mm")
    print(f"Dropped measurements:        {dropout_count}")
    print(f"Output data:                 {data_path}")
    print(f"Output figure:               {figure_path}")


if __name__ == "__main__":
    main()
