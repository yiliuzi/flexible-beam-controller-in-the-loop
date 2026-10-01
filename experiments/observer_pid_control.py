"""传感器、状态观测器和PID完整闭环实验。"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from controllers.pid import PIDController, PIDParameters
from controllers.state_observer import (
    LuenbergerObserver,
    ObserverParameters,
)
from plant.beam_sdof import (
    BeamParameters,
    BeamState,
    SingleDegreeBeam,
)
from plant.disturbances import impulse_disturbance
from sensors.virtual_displacement_sensor import (
    DisplacementSensorParameters,
    VirtualDisplacementSensor,
)
from simulation.config import SimulationConfig
from simulation.runner import SimulationResult, run_simulation
from visualization.plots import plot_observer_pid_comparison

DisturbanceFunction = Callable[[float], float]


def calculate_rms(values: np.ndarray) -> float:
    """计算信号均方根。"""

    return float(np.sqrt(np.mean(np.square(values))))


def estimate_sensor_bias(
    sensor: VirtualDisplacementSensor,
    sample_count: int = 1000,
) -> float:
    """使用静止样本估计位移传感器零偏。"""

    calibration_values: list[float] = []
    timestamp = 0.0

    while len(calibration_values) < sample_count:
        measurement = sensor.read(
            true_displacement=0.0,
            timestamp=timestamp,
        )

        if measurement.valid and not measurement.saturated:
            calibration_values.append(measurement.displacement)

        timestamp += 0.001

    return float(np.mean(calibration_values))


def run_observer_pid_simulation(
    beam_parameters: BeamParameters,
    config: SimulationConfig,
    disturbance: DisturbanceFunction,
) -> tuple[
    SimulationResult,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """运行基于传感器和状态观测器的PID闭环。"""

    beam = SingleDegreeBeam(
        parameters=beam_parameters,
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

    estimated_sensor_bias = estimate_sensor_bias(sensor)

    observer = LuenbergerObserver(
        ObserverParameters(
            mass=beam_parameters.mass,
            damping=beam_parameters.damping,
            stiffness=beam_parameters.stiffness,
            position_gain=58.5,
            velocity_gain=562.25,
        )
    )

    controller = PIDController(
        PIDParameters(
            proportional_gain=35.0,
            integral_gain=0.0,
            derivative_gain=1.2,
            output_minimum=-5.0,
            output_maximum=5.0,
            integral_minimum=-0.5,
            integral_maximum=0.5,
        )
    )

    sample_count = int(round(config.duration / config.time_step)) + 1

    time_values = np.arange(sample_count, dtype=np.float64) * config.time_step

    displacement_values = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    velocity_values = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    acceleration_values = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    control_values = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    disturbance_values = np.zeros(
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

    previously_applied_control = 0.0

    for sample_index, current_time in enumerate(time_values):
        disturbance_force = disturbance(float(current_time))

        measurement = sensor.read(
            true_displacement=beam.state.displacement,
            timestamp=float(current_time),
        )

        corrected_displacement = measurement.displacement - estimated_sensor_bias

        observer_state = observer.update(
            measured_displacement=corrected_displacement,
            applied_control_force=previously_applied_control,
            time_step=config.time_step,
            measurement_valid=measurement.valid,
        )

        control_force = controller.update(
            reference=0.0,
            measurement=observer_state.displacement,
            time_step=config.time_step,
        )

        displacement_values[sample_index] = beam.state.displacement

        velocity_values[sample_index] = beam.state.velocity

        acceleration_values[sample_index] = beam.acceleration(
            control_force=control_force,
            disturbance_force=disturbance_force,
        )

        control_values[sample_index] = control_force
        disturbance_values[sample_index] = disturbance_force

        estimated_displacement[sample_index] = observer_state.displacement

        estimated_velocity[sample_index] = observer_state.velocity

        valid_measurements[sample_index] = measurement.valid

        if sample_index < sample_count - 1:
            beam.step(
                time_step=config.time_step,
                control_force=control_force,
                disturbance_force=disturbance_force,
            )

        previously_applied_control = control_force

    result = SimulationResult(
        time=time_values,
        displacement=displacement_values,
        velocity=velocity_values,
        acceleration=acceleration_values,
        control_force=control_values,
        disturbance_force=disturbance_values,
    )

    return (
        result,
        estimated_displacement,
        estimated_velocity,
        valid_measurements,
        estimated_sensor_bias,
    )


def main() -> None:
    """运行三种控制模式的公平对比实验。"""

    project_root = Path(__file__).resolve().parents[1]

    open_loop_data_path = project_root / "data" / "observer_comparison_open_loop.csv"

    ideal_pid_data_path = project_root / "data" / "observer_comparison_ideal_pid.csv"

    observer_pid_data_path = project_root / "data" / "observer_pid_control_response.csv"

    figure_path = project_root / "results" / "observer_pid_control_comparison.png"

    config = SimulationConfig(
        time_step=0.001,
        duration=5.0,
    )

    beam_parameters = BeamParameters(
        mass=0.08,
        damping=0.12,
        stiffness=18.0,
    )

    def disturbance(current_time: float) -> float:
        return impulse_disturbance(
            time=current_time,
            start_time=0.50,
            duration=0.05,
            amplitude=1.50,
        )

    open_loop_result = run_simulation(
        beam=SingleDegreeBeam(
            parameters=beam_parameters,
        ),
        config=config,
        disturbance=disturbance,
    )

    ideal_controller = PIDController(
        PIDParameters(
            proportional_gain=35.0,
            integral_gain=0.0,
            derivative_gain=1.2,
            output_minimum=-5.0,
            output_maximum=5.0,
            integral_minimum=-0.5,
            integral_maximum=0.5,
        )
    )

    def ideal_pid_callback(
        _current_time: float,
        state: BeamState,
    ) -> float:
        return ideal_controller.update(
            reference=0.0,
            measurement=state.displacement,
            time_step=config.time_step,
        )

    ideal_pid_result = run_simulation(
        beam=SingleDegreeBeam(
            parameters=beam_parameters,
        ),
        config=config,
        disturbance=disturbance,
        controller=ideal_pid_callback,
    )

    (
        observer_pid_result,
        estimated_displacement,
        estimated_velocity,
        valid_measurements,
        estimated_sensor_bias,
    ) = run_observer_pid_simulation(
        beam_parameters=beam_parameters,
        config=config,
        disturbance=disturbance,
    )

    evaluation_mask = open_loop_result.time >= 0.50

    open_loop_rms = calculate_rms(open_loop_result.displacement[evaluation_mask])

    ideal_pid_rms = calculate_rms(ideal_pid_result.displacement[evaluation_mask])

    observer_pid_rms = calculate_rms(observer_pid_result.displacement[evaluation_mask])

    ideal_suppression = (open_loop_rms - ideal_pid_rms) / open_loop_rms * 100.0

    observer_suppression = (open_loop_rms - observer_pid_rms) / open_loop_rms * 100.0

    observer_performance_penalty = (observer_pid_rms - ideal_pid_rms) / ideal_pid_rms * 100.0

    estimation_rmse = calculate_rms(
        (estimated_displacement - observer_pid_result.displacement)[evaluation_mask]
    )

    maximum_observer_control = float(np.max(np.abs(observer_pid_result.control_force)))

    dropout_count = int(np.count_nonzero(~valid_measurements))

    open_loop_result.to_dataframe().to_csv(
        open_loop_data_path,
        index=False,
    )

    ideal_pid_result.to_dataframe().to_csv(
        ideal_pid_data_path,
        index=False,
    )

    observer_table = observer_pid_result.to_dataframe()

    observer_table["estimated_displacement_m"] = estimated_displacement

    observer_table["estimated_velocity_m_s"] = estimated_velocity

    observer_table["measurement_valid"] = valid_measurements

    observer_table.to_csv(
        observer_pid_data_path,
        index=False,
    )

    plot_observer_pid_comparison(
        open_loop_result=open_loop_result,
        ideal_pid_result=ideal_pid_result,
        observer_pid_result=observer_pid_result,
        estimated_displacement=estimated_displacement,
        estimated_velocity=estimated_velocity,
        output_path=figure_path,
    )

    print("Observer-based PID experiment completed")
    print(f"Estimated sensor bias:        {estimated_sensor_bias * 1000.0:.4f} mm")
    print(f"Open-loop RMS:                {open_loop_rms * 1000.0:.4f} mm")
    print(f"Ideal PID RMS:                {ideal_pid_rms * 1000.0:.4f} mm")
    print(f"Observer PID RMS:             {observer_pid_rms * 1000.0:.4f} mm")
    print(f"Ideal PID suppression:        {ideal_suppression:.2f}%")
    print(f"Observer PID suppression:     {observer_suppression:.2f}%")
    print(f"Observer performance penalty: {observer_performance_penalty:.2f}%")
    print(f"Observer displacement RMSE:   {estimation_rmse * 1000.0:.4f} mm")
    print(f"Maximum observer control:     {maximum_observer_control:.4f} N")
    print(f"Dropped measurements:         {dropout_count}")
    print(f"Output figure:                {figure_path}")


if __name__ == "__main__":
    main()
