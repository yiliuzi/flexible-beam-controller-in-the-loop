"""CAN通信故障下的柔性梁控制闭环实验。"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from actuators.voice_coil import (
    VoiceCoilActuator,
    VoiceCoilParameters,
)
from communication.can_protocol import (
    CanFrameMonitor,
    CanStateMessage,
    encode_state_frame,
)
from communication.virtual_can_bus import (
    VirtualCanBus,
    VirtualCanParameters,
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
from safety.fault_state_machine import (
    FaultStateMachine,
    SafetyInput,
)
from sensors.virtual_displacement_sensor import (
    DisplacementSensorParameters,
    VirtualDisplacementSensor,
)
from simulation.config import SimulationConfig
from simulation.runner import SimulationResult
from visualization.plots import (
    plot_can_fault_comparison,
    plot_safety_state_timeline,
)

DisturbanceFunction = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class CanFaultConfiguration:
    """CAN通信闭环故障参数。"""

    transmission_period: float = 0.005
    communication_timeout: float = 0.020
    disconnect_start: float | None = None
    disconnect_duration: float = 0.0

    def __post_init__(self) -> None:
        if self.transmission_period <= 0:
            raise ValueError("transmission_period must be greater than zero")

        if self.communication_timeout <= 0:
            raise ValueError("communication_timeout must be greater than zero")

        if self.disconnect_start is not None:
            if self.disconnect_start < 0:
                raise ValueError("disconnect_start must not be negative")

            if self.disconnect_duration <= 0:
                raise ValueError("disconnect_duration must be greater than zero")


@dataclass(slots=True)
class CanControlResult:
    """CAN通信闭环实验结果。"""

    simulation: SimulationResult
    requested_force: NDArray[np.float64]
    communication_timeout: NDArray[np.bool_]
    valid_frame_received: NDArray[np.bool_]
    bus_connected: NDArray[np.bool_]
    transmitted_frames: int
    delivered_frames: int
    dropped_frames: int
    corrupted_frames: int
    invalid_frames: int
    lost_frames: int
    timeout_samples: int
    safety_state: NDArray[np.str_]
    safety_force_scale: NDArray[np.float64]
    safety_state_changed: NDArray[np.bool_]


def run_can_control_simulation(
    beam_parameters: BeamParameters,
    simulation_config: SimulationConfig,
    bus_parameters: VirtualCanParameters,
    fault_configuration: CanFaultConfiguration,
    disturbance: DisturbanceFunction,
) -> CanControlResult:
    """运行经过虚拟CAN总线的控制器在环仿真。"""

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
        parameters=ControlTaskParameters(
            control_period=0.005,
            computation_delay=0.002,
            deadline=0.005,
        ),
    )

    actuator = VoiceCoilActuator(
        VoiceCoilParameters(
            maximum_force=2.0,
            time_constant=0.01,
            dead_zone=0.02,
        )
    )

    can_bus = VirtualCanBus(bus_parameters)
    frame_monitor = CanFrameMonitor()
    safety_state_machine = FaultStateMachine()

    sample_count = int(round(simulation_config.duration / simulation_config.time_step)) + 1

    time_values = np.arange(sample_count, dtype=np.float64) * simulation_config.time_step

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

    requested_force_values = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    applied_force_values = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    disturbance_values = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    communication_timeout_values = np.zeros(
        sample_count,
        dtype=np.bool_,
    )

    valid_frame_received_values = np.zeros(
        sample_count,
        dtype=np.bool_,
    )

    bus_connected_values = np.zeros(
        sample_count,
        dtype=np.bool_,
    )

    safety_state_values = np.empty(
        sample_count,
        dtype="<U16",
    )

    safety_force_scale_values = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    safety_state_changed_values = np.zeros(
        sample_count,
        dtype=np.bool_,
    )

    next_transmission_time = 0.0
    sequence = 0

    latest_remote_displacement = 0.0
    previously_applied_force = 0.0
    previously_actuator_saturated = False
    previously_deadline_missed = False

    for sample_index, current_time in enumerate(time_values):
        timestamp = float(current_time)

        disturbance_force = disturbance(timestamp)

        measurement = sensor.read(
            true_displacement=beam.state.displacement,
            timestamp=timestamp,
        )

        corrected_displacement = measurement.displacement - estimated_sensor_bias

        observer_state = observer.update(
            measured_displacement=corrected_displacement,
            applied_control_force=previously_applied_force,
            time_step=simulation_config.time_step,
            measurement_valid=measurement.valid,
        )

        disconnected = False

        if fault_configuration.disconnect_start is not None:
            disconnect_end = (
                fault_configuration.disconnect_start + fault_configuration.disconnect_duration
            )

            disconnected = fault_configuration.disconnect_start <= timestamp < disconnect_end

        can_bus.set_connected(not disconnected)

        if timestamp + 1e-12 >= next_transmission_time:
            message = CanStateMessage(
                displacement=observer_state.displacement,
                velocity=observer_state.velocity,
                sequence=sequence,
                measurement_valid=measurement.valid,
                sensor_fault=not measurement.valid,
                actuator_saturated=(previously_actuator_saturated),
                deadline_missed=(previously_deadline_missed),
            )

            can_bus.send(
                payload=encode_state_frame(message),
                timestamp=timestamp,
            )

            sequence = (sequence + 1) % 256

            next_transmission_time += fault_configuration.transmission_period

        valid_frame_received = False
        received_sensor_fault = False

        for payload in can_bus.receive(timestamp=timestamp):
            receive_result = frame_monitor.receive(
                payload=payload,
                timestamp=timestamp,
            )

            if receive_result.valid and receive_result.message is not None:
                latest_remote_displacement = receive_result.message.displacement

                received_sensor_fault = receive_result.message.sensor_fault

                valid_frame_received = True

        communication_timed_out = frame_monitor.is_timed_out(
            timestamp=timestamp,
            timeout=(fault_configuration.communication_timeout),
        )

        task_output = control_task.update(
            timestamp=timestamp,
            measurement=latest_remote_displacement,
        )

        safety_output = safety_state_machine.update(
            SafetyInput(
                communication_timed_out=(communication_timed_out),
                valid_frame_received=(valid_frame_received),
                sensor_fault=received_sensor_fault,
                deadline_missed=(task_output.deadline_missed),
            )
        )

        requested_force = task_output.applied_command * safety_output.force_scale

        actuator_output = actuator.update(
            requested_force=requested_force,
            time_step=simulation_config.time_step,
            enabled=safety_output.actuator_enabled,
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

        communication_timeout_values[sample_index] = communication_timed_out

        valid_frame_received_values[sample_index] = valid_frame_received

        bus_connected_values[sample_index] = can_bus.connected

        safety_state_values[sample_index] = safety_output.state.name

        safety_force_scale_values[sample_index] = safety_output.force_scale

        safety_state_changed_values[sample_index] = safety_output.state_changed

        if sample_index < sample_count - 1:
            beam.step(
                time_step=simulation_config.time_step,
                control_force=applied_force,
                disturbance_force=disturbance_force,
            )

        previously_applied_force = applied_force

        previously_actuator_saturated = actuator_output.saturated

        previously_deadline_missed = task_output.deadline_missed

    result = SimulationResult(
        time=time_values,
        displacement=displacement_values,
        velocity=velocity_values,
        acceleration=acceleration_values,
        control_force=applied_force_values,
        disturbance_force=disturbance_values,
    )

    return CanControlResult(
        simulation=result,
        requested_force=requested_force_values,
        communication_timeout=(communication_timeout_values),
        valid_frame_received=(valid_frame_received_values),
        bus_connected=bus_connected_values,
        transmitted_frames=(can_bus.transmitted_frame_count),
        delivered_frames=(can_bus.delivered_frame_count),
        dropped_frames=(can_bus.dropped_frame_count),
        corrupted_frames=(can_bus.corrupted_frame_count),
        invalid_frames=(frame_monitor.invalid_frame_count),
        lost_frames=(frame_monitor.total_lost_frames),
        timeout_samples=int(np.count_nonzero(communication_timeout_values)),
        safety_state=safety_state_values,
        safety_force_scale=(safety_force_scale_values),
        safety_state_changed=(safety_state_changed_values),
    )


def create_disturbance(
    current_time: float,
) -> float:
    """创建统一冲击扰动。"""

    return impulse_disturbance(
        time=current_time,
        start_time=0.50,
        duration=0.05,
        amplitude=1.50,
    )


def main() -> None:
    """运行健康CAN与故障CAN闭环对比实验。"""

    project_root = Path(__file__).resolve().parents[1]

    healthy_data_path = project_root / "data" / "healthy_can_control.csv"

    faulty_data_path = project_root / "data" / "faulty_can_control.csv"

    figure_path = project_root / "results" / "can_fault_control_comparison.png"

    safety_figure_path = project_root / "results" / "safety_state_timeline.png"

    simulation_config = SimulationConfig(
        time_step=0.001,
        duration=5.0,
    )

    beam_parameters = BeamParameters(
        mass=0.08,
        damping=0.12,
        stiffness=18.0,
    )

    healthy_result = run_can_control_simulation(
        beam_parameters=beam_parameters,
        simulation_config=simulation_config,
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
            drop_probability=0.0,
            corruption_probability=0.0,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.020,
        ),
        disturbance=create_disturbance,
    )

    faulty_result = run_can_control_simulation(
        beam_parameters=beam_parameters,
        simulation_config=simulation_config,
        bus_parameters=VirtualCanParameters(
            transmission_delay=0.002,
            drop_probability=0.08,
            corruption_probability=0.04,
            seed=2028,
        ),
        fault_configuration=CanFaultConfiguration(
            transmission_period=0.005,
            communication_timeout=0.020,
            disconnect_start=2.50,
            disconnect_duration=0.15,
        ),
        disturbance=create_disturbance,
    )

    evaluation_mask = healthy_result.simulation.time >= 0.50

    healthy_rms = calculate_rms(healthy_result.simulation.displacement[evaluation_mask])

    faulty_rms = calculate_rms(faulty_result.simulation.displacement[evaluation_mask])

    communication_penalty = (faulty_rms - healthy_rms) / healthy_rms * 100.0

    healthy_table = healthy_result.simulation.to_dataframe()

    healthy_table["requested_force_n"] = healthy_result.requested_force

    healthy_table["communication_timeout"] = healthy_result.communication_timeout

    healthy_table["valid_frame_received"] = healthy_result.valid_frame_received

    healthy_table["bus_connected"] = healthy_result.bus_connected

    healthy_table["safety_state"] = healthy_result.safety_state

    healthy_table["safety_force_scale"] = healthy_result.safety_force_scale

    healthy_table["safety_state_changed"] = healthy_result.safety_state_changed

    healthy_table.to_csv(
        healthy_data_path,
        index=False,
    )

    faulty_table = faulty_result.simulation.to_dataframe()

    faulty_table["requested_force_n"] = faulty_result.requested_force

    faulty_table["communication_timeout"] = faulty_result.communication_timeout

    faulty_table["valid_frame_received"] = faulty_result.valid_frame_received

    faulty_table["bus_connected"] = faulty_result.bus_connected

    faulty_table["safety_state"] = faulty_result.safety_state

    faulty_table["safety_force_scale"] = faulty_result.safety_force_scale

    faulty_table["safety_state_changed"] = faulty_result.safety_state_changed

    faulty_table.to_csv(
        faulty_data_path,
        index=False,
    )

    plot_can_fault_comparison(
        healthy_result=healthy_result.simulation,
        faulty_result=faulty_result.simulation,
        healthy_requested_force=(healthy_result.requested_force),
        faulty_requested_force=(faulty_result.requested_force),
        faulty_bus_connected=(faulty_result.bus_connected),
        faulty_timeout=(faulty_result.communication_timeout),
        faulty_valid_frame=(faulty_result.valid_frame_received),
        output_path=figure_path,
    )

    plot_safety_state_timeline(
        time=faulty_result.simulation.time,
        safety_state=faulty_result.safety_state,
        force_scale=faulty_result.safety_force_scale,
        communication_timeout=(faulty_result.communication_timeout),
        bus_connected=faulty_result.bus_connected,
        output_path=safety_figure_path,
    )

    normal_samples = int(np.count_nonzero(faulty_result.safety_state == "NORMAL"))

    degraded_samples = int(np.count_nonzero(faulty_result.safety_state == "DEGRADED"))

    safe_stop_samples = int(np.count_nonzero(faulty_result.safety_state == "SAFE_STOP"))

    recovery_samples = int(np.count_nonzero(faulty_result.safety_state == "RECOVERY"))

    safety_transitions = int(np.count_nonzero(faulty_result.safety_state_changed))

    print("CAN fault control experiment completed")
    print(f"Healthy CAN RMS:          {healthy_rms * 1000.0:.4f} mm")
    print(f"Faulty CAN RMS:           {faulty_rms * 1000.0:.4f} mm")
    print(f"Communication penalty:    {communication_penalty:.2f}%")
    print(f"Healthy transmitted:      {healthy_result.transmitted_frames}")
    print(f"Healthy dropped:          {healthy_result.dropped_frames}")
    print(f"Healthy timeout samples:  {healthy_result.timeout_samples}")
    print(f"Faulty transmitted:       {faulty_result.transmitted_frames}")
    print(f"Faulty delivered:         {faulty_result.delivered_frames}")
    print(f"Faulty dropped:           {faulty_result.dropped_frames}")
    print(f"Faulty corrupted:         {faulty_result.corrupted_frames}")
    print(f"Faulty invalid:           {faulty_result.invalid_frames}")
    print(f"Faulty detected loss:     {faulty_result.lost_frames}")
    print(f"Faulty timeout samples:   {faulty_result.timeout_samples}")
    print(f"Normal samples:            {normal_samples}")
    print(f"Degraded samples:          {degraded_samples}")
    print(f"Safe-stop samples:         {safe_stop_samples}")
    print(f"Recovery samples:          {recovery_samples}")
    print(f"Safety state transitions:  {safety_transitions}")
    print(f"Healthy data:             {healthy_data_path}")
    print(f"Faulty data:              {faulty_data_path}")
    print(f"Output figure:            {figure_path}")
    print(f"Safety figure:            {safety_figure_path}")


if __name__ == "__main__":
    main()
