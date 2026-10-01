"""非理想音圈执行器PID闭环对比实验。"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from actuators.voice_coil import (
    VoiceCoilActuator,
    VoiceCoilParameters,
)
from controllers.pid import PIDController, PIDParameters
from controllers.state_observer import (
    LuenbergerObserver,
    ObserverParameters,
)
from experiments.observer_pid_control import (
    calculate_rms,
    estimate_sensor_bias,
    run_observer_pid_simulation,
)
from plant.beam_sdof import BeamParameters, SingleDegreeBeam
from plant.disturbances import impulse_disturbance
from sensors.virtual_displacement_sensor import (
    DisplacementSensorParameters,
    VirtualDisplacementSensor,
)
from simulation.config import SimulationConfig
from simulation.runner import SimulationResult, run_simulation
from visualization.plots import plot_actuator_pid_comparison

DisturbanceFunction = Callable[[float], float]


def run_actuator_pid_simulation(
    beam_parameters: BeamParameters,
    config: SimulationConfig,
    disturbance: DisturbanceFunction,
) -> tuple[
    SimulationResult,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """运行包含非理想音圈执行器的观测器PID闭环。"""

    beam = SingleDegreeBeam(parameters=beam_parameters)

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

    actuator = VoiceCoilActuator(
        VoiceCoilParameters(
            maximum_force=2.0,
            time_constant=0.01,
            dead_zone=0.02,
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
    applied_force_values = np.zeros(
        sample_count,
        dtype=np.float64,
    )
    requested_force_values = np.zeros(
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
    actuator_saturated = np.zeros(
        sample_count,
        dtype=np.bool_,
    )
    inside_dead_zone = np.zeros(
        sample_count,
        dtype=np.bool_,
    )

    previously_applied_force = 0.0

    for sample_index, current_time in enumerate(time_values):
        disturbance_force = disturbance(float(current_time))

        measurement = sensor.read(
            true_displacement=beam.state.displacement,
            timestamp=float(current_time),
        )

        corrected_displacement = measurement.displacement - estimated_sensor_bias

        observer_state = observer.update(
            measured_displacement=corrected_displacement,
            applied_control_force=previously_applied_force,
            time_step=config.time_step,
            measurement_valid=measurement.valid,
        )

        requested_force = controller.update(
            reference=0.0,
            measurement=observer_state.displacement,
            time_step=config.time_step,
        )

        actuator_output = actuator.update(
            requested_force=requested_force,
            time_step=config.time_step,
        )

        applied_force = actuator_output.applied_force

        displacement_values[sample_index] = beam.state.displacement
        velocity_values[sample_index] = beam.state.velocity

        acceleration_values[sample_index] = beam.acceleration(
            control_force=applied_force,
            disturbance_force=disturbance_force,
        )

        requested_force_values[sample_index] = requested_force
        applied_force_values[sample_index] = applied_force
        disturbance_values[sample_index] = disturbance_force

        estimated_displacement[sample_index] = observer_state.displacement
        estimated_velocity[sample_index] = observer_state.velocity

        actuator_saturated[sample_index] = actuator_output.saturated
        inside_dead_zone[sample_index] = actuator_output.inside_dead_zone

        if sample_index < sample_count - 1:
            beam.step(
                time_step=config.time_step,
                control_force=applied_force,
                disturbance_force=disturbance_force,
            )

        previously_applied_force = applied_force

    result = SimulationResult(
        time=time_values,
        displacement=displacement_values,
        velocity=velocity_values,
        acceleration=acceleration_values,
        control_force=applied_force_values,
        disturbance_force=disturbance_values,
    )

    return (
        result,
        requested_force_values,
        estimated_displacement,
        estimated_velocity,
        actuator_saturated,
        estimated_sensor_bias,
    )


def main() -> None:
    """运行理想执行器和非理想执行器对比实验。"""

    project_root = Path(__file__).resolve().parents[1]

    data_path = project_root / "data" / "actuator_pid_control_response.csv"

    figure_path = project_root / "results" / "actuator_pid_control_comparison.png"

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
        beam=SingleDegreeBeam(parameters=beam_parameters),
        config=config,
        disturbance=disturbance,
    )

    (
        ideal_actuator_result,
        _,
        _,
        _,
        _,
    ) = run_observer_pid_simulation(
        beam_parameters=beam_parameters,
        config=config,
        disturbance=disturbance,
    )

    (
        nonideal_actuator_result,
        requested_force,
        estimated_displacement,
        estimated_velocity,
        actuator_saturated,
        estimated_sensor_bias,
    ) = run_actuator_pid_simulation(
        beam_parameters=beam_parameters,
        config=config,
        disturbance=disturbance,
    )

    evaluation_mask = nonideal_actuator_result.time >= 0.50

    open_loop_rms = calculate_rms(open_loop_result.displacement[evaluation_mask])

    ideal_actuator_rms = calculate_rms(ideal_actuator_result.displacement[evaluation_mask])

    nonideal_actuator_rms = calculate_rms(nonideal_actuator_result.displacement[evaluation_mask])

    suppression = (open_loop_rms - nonideal_actuator_rms) / open_loop_rms * 100.0

    actuator_penalty = (nonideal_actuator_rms - ideal_actuator_rms) / ideal_actuator_rms * 100.0

    force_tracking_rmse = calculate_rms(
        (requested_force - nonideal_actuator_result.control_force)[evaluation_mask]
    )

    maximum_requested_force = float(np.max(np.abs(requested_force)))

    maximum_applied_force = float(np.max(np.abs(nonideal_actuator_result.control_force)))

    saturation_count = int(np.count_nonzero(actuator_saturated))

    result_table = nonideal_actuator_result.to_dataframe()

    result_table["requested_force_n"] = requested_force
    result_table["estimated_displacement_m"] = estimated_displacement
    result_table["estimated_velocity_m_s"] = estimated_velocity
    result_table["actuator_saturated"] = actuator_saturated

    result_table.to_csv(
        data_path,
        index=False,
    )

    plot_actuator_pid_comparison(
        open_loop_result=open_loop_result,
        ideal_actuator_result=ideal_actuator_result,
        nonideal_actuator_result=nonideal_actuator_result,
        requested_force=requested_force,
        actuator_saturated=actuator_saturated,
        output_path=figure_path,
    )

    print("Nonideal actuator experiment completed")
    print(f"Estimated sensor bias:       {estimated_sensor_bias * 1000.0:.4f} mm")
    print(f"Open-loop RMS:               {open_loop_rms * 1000.0:.4f} mm")
    print(f"Ideal actuator RMS:          {ideal_actuator_rms * 1000.0:.4f} mm")
    print(f"Nonideal actuator RMS:       {nonideal_actuator_rms * 1000.0:.4f} mm")
    print(f"Nonideal suppression:        {suppression:.2f}%")
    print(f"Actuator performance penalty: {actuator_penalty:.2f}%")
    print(f"Force tracking RMSE:         {force_tracking_rmse:.4f} N")
    print(f"Maximum requested force:     {maximum_requested_force:.4f} N")
    print(f"Maximum applied force:       {maximum_applied_force:.4f} N")
    print(f"Saturated samples:           {saturation_count}")
    print(f"Output data:                 {data_path}")
    print(f"Output figure:               {figure_path}")


if __name__ == "__main__":
    main()
