"""嵌入式控制周期和计算延迟闭环实验。"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from actuators.voice_coil import (
    VoiceCoilActuator,
    VoiceCoilParameters,
)
from controllers.pid import PIDController, PIDParameters
from controllers.state_observer import (
    LuenbergerObserver,
    ObserverParameters,
)
from embedded.periodic_control_task import (
    ControlTaskParameters,
    PeriodicControlTask,
)
from experiments.observer_pid_control import (
    calculate_rms,
    estimate_sensor_bias,
)
from plant.beam_sdof import BeamParameters, SingleDegreeBeam
from plant.disturbances import impulse_disturbance
from sensors.virtual_displacement_sensor import (
    DisplacementSensorParameters,
    VirtualDisplacementSensor,
)
from simulation.config import SimulationConfig
from simulation.runner import SimulationResult
from visualization.plots import (
    plot_embedded_timing_comparison,
)

DisturbanceFunction = Callable[[float], float]


@dataclass(slots=True)
class EmbeddedTimingResult:
    """一次嵌入式时序闭环实验数据。"""

    simulation: SimulationResult
    requested_force: NDArray[np.float64]
    task_executed: NDArray[np.bool_]
    command_activated: NDArray[np.bool_]
    deadline_missed: NDArray[np.bool_]
    execution_count: int
    deadline_miss_count: int


def run_embedded_timing_simulation(
    beam_parameters: BeamParameters,
    simulation_config: SimulationConfig,
    task_parameters: ControlTaskParameters,
    disturbance: DisturbanceFunction,
) -> EmbeddedTimingResult:
    """运行带周期调度和计算延迟的完整闭环。"""

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

    def controller_callback(
        measurement: float,
        control_period: float,
    ) -> float:
        return controller.update(
            reference=0.0,
            measurement=measurement,
            time_step=control_period,
        )

    control_task = PeriodicControlTask(
        controller=controller_callback,
        parameters=task_parameters,
    )

    actuator = VoiceCoilActuator(
        VoiceCoilParameters(
            maximum_force=2.0,
            time_constant=0.01,
            dead_zone=0.02,
        )
    )

    sample_count = int(round(simulation_config.duration / simulation_config.time_step)) + 1

    time_values = np.arange(sample_count, dtype=np.float64) * simulation_config.time_step

    displacement_values = np.zeros(sample_count)
    velocity_values = np.zeros(sample_count)
    acceleration_values = np.zeros(sample_count)
    requested_force_values = np.zeros(sample_count)
    applied_force_values = np.zeros(sample_count)
    disturbance_values = np.zeros(sample_count)

    task_executed_values = np.zeros(
        sample_count,
        dtype=np.bool_,
    )
    command_activated_values = np.zeros(
        sample_count,
        dtype=np.bool_,
    )
    deadline_missed_values = np.zeros(
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
            time_step=simulation_config.time_step,
            measurement_valid=measurement.valid,
        )

        task_output = control_task.update(
            timestamp=float(current_time),
            measurement=observer_state.displacement,
        )

        actuator_output = actuator.update(
            requested_force=task_output.applied_command,
            time_step=simulation_config.time_step,
        )

        applied_force = actuator_output.applied_force

        displacement_values[sample_index] = beam.state.displacement
        velocity_values[sample_index] = beam.state.velocity

        acceleration_values[sample_index] = beam.acceleration(
            control_force=applied_force,
            disturbance_force=disturbance_force,
        )

        requested_force_values[sample_index] = task_output.applied_command
        applied_force_values[sample_index] = applied_force
        disturbance_values[sample_index] = disturbance_force

        task_executed_values[sample_index] = task_output.task_executed
        command_activated_values[sample_index] = task_output.command_activated
        deadline_missed_values[sample_index] = task_output.deadline_missed

        if sample_index < sample_count - 1:
            beam.step(
                time_step=simulation_config.time_step,
                control_force=applied_force,
                disturbance_force=disturbance_force,
            )

        previously_applied_force = applied_force

    simulation_result = SimulationResult(
        time=time_values,
        displacement=displacement_values,
        velocity=velocity_values,
        acceleration=acceleration_values,
        control_force=applied_force_values,
        disturbance_force=disturbance_values,
    )

    return EmbeddedTimingResult(
        simulation=simulation_result,
        requested_force=requested_force_values,
        task_executed=task_executed_values,
        command_activated=command_activated_values,
        deadline_missed=deadline_missed_values,
        execution_count=control_task.execution_count,
        deadline_miss_count=(control_task.deadline_miss_count),
    )


def main() -> None:
    """运行不同控制周期与延迟的对比实验。"""

    project_root = Path(__file__).resolve().parents[1]

    figure_path = project_root / "results" / "embedded_timing_comparison.png"

    data_path = project_root / "data" / "embedded_timing_response.csv"

    simulation_config = SimulationConfig(
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

    fast_loop = run_embedded_timing_simulation(
        beam_parameters=beam_parameters,
        simulation_config=simulation_config,
        task_parameters=ControlTaskParameters(
            control_period=0.001,
            computation_delay=0.0,
            deadline=0.001,
        ),
        disturbance=disturbance,
    )

    embedded_loop = run_embedded_timing_simulation(
        beam_parameters=beam_parameters,
        simulation_config=simulation_config,
        task_parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.002,
            deadline=0.005,
        ),
        disturbance=disturbance,
    )

    missed_deadline_loop = run_embedded_timing_simulation(
        beam_parameters=beam_parameters,
        simulation_config=simulation_config,
        task_parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.006,
            deadline=0.005,
        ),
        disturbance=disturbance,
    )

    evaluation_mask = fast_loop.simulation.time >= 0.50

    fast_rms = calculate_rms(fast_loop.simulation.displacement[evaluation_mask])

    embedded_rms = calculate_rms(embedded_loop.simulation.displacement[evaluation_mask])

    missed_deadline_rms = calculate_rms(
        missed_deadline_loop.simulation.displacement[evaluation_mask]
    )

    embedded_penalty = (embedded_rms - fast_rms) / fast_rms * 100.0

    deadline_penalty = (missed_deadline_rms - fast_rms) / fast_rms * 100.0

    result_table = embedded_loop.simulation.to_dataframe()

    result_table["requested_force_n"] = embedded_loop.requested_force
    result_table["task_executed"] = embedded_loop.task_executed
    result_table["command_activated"] = embedded_loop.command_activated
    result_table["deadline_missed"] = embedded_loop.deadline_missed

    result_table.to_csv(
        data_path,
        index=False,
    )

    plot_embedded_timing_comparison(
        fast_result=fast_loop.simulation,
        embedded_result=embedded_loop.simulation,
        missed_deadline_result=(missed_deadline_loop.simulation),
        embedded_requested_force=(embedded_loop.requested_force),
        embedded_task_executed=(embedded_loop.task_executed),
        output_path=figure_path,
    )

    print("Embedded timing experiment completed")
    print(f"Fast-loop RMS:             {fast_rms * 1000.0:.4f} mm")
    print(f"Embedded-loop RMS:         {embedded_rms * 1000.0:.4f} mm")
    print(f"Missed-deadline RMS:       {missed_deadline_rms * 1000.0:.4f} mm")
    print(f"Embedded timing penalty:   {embedded_penalty:.2f}%")
    print(f"Deadline timing penalty:   {deadline_penalty:.2f}%")
    print(f"Embedded task executions:  {embedded_loop.execution_count}")
    print(f"Embedded deadline misses:  {embedded_loop.deadline_miss_count}")
    print(f"Delayed task executions:   {missed_deadline_loop.execution_count}")
    print(f"Delayed deadline misses:   {missed_deadline_loop.deadline_miss_count}")
    print(f"Output data:               {data_path}")
    print(f"Output figure:             {figure_path}")


if __name__ == "__main__":
    main()
