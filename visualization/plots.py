"""柔性梁实验结果可视化。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from simulation.runner import SimulationResult


def plot_free_vibration_response(
    result: SimulationResult,
    output_path: str | Path,
) -> None:
    """绘制冲击扰动及柔性梁自由振动响应。"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        result.time,
        result.displacement * 1000.0,
        color="#2563EB",
        linewidth=1.5,
        label="Beam-tip displacement",
    )
    axes[0].axhline(0.0, color="#64748B", linewidth=0.8)
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title("Flexible Beam Free-Vibration Response")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        result.time,
        result.disturbance_force,
        color="#DC2626",
        linewidth=1.5,
        label="External disturbance",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Force (N)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_pid_comparison(
    open_loop_result: SimulationResult,
    controlled_result: SimulationResult,
    output_path: str | Path,
) -> None:
    """绘制无控制与PID控制的响应对比。"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        open_loop_result.time,
        open_loop_result.displacement * 1000.0,
        color="#DC2626",
        linewidth=1.4,
        label="Without control",
    )

    axes[0].plot(
        controlled_result.time,
        controlled_result.displacement * 1000.0,
        color="#2563EB",
        linewidth=1.4,
        label="PID control",
    )

    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title("Flexible Beam Vibration Control Comparison")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        controlled_result.time,
        controlled_result.control_force,
        color="#16A34A",
        linewidth=1.2,
        label="PID control force",
    )
    axes[1].set_ylabel("Control force (N)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        controlled_result.time,
        controlled_result.disturbance_force,
        color="#9333EA",
        linewidth=1.2,
        label="External disturbance",
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Disturbance (N)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_imu_measurement_comparison(
    time: NDArray[np.float64],
    true_acceleration: NDArray[np.float64],
    measured_acceleration: NDArray[np.float64],
    valid_samples: NDArray[np.bool_],
    output_path: str | Path,
) -> None:
    """绘制真实加速度与虚拟IMU测量值对比。"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    measurement_error = measured_acceleration - true_acceleration

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        time,
        true_acceleration,
        color="#1D4ED8",
        linewidth=1.4,
        label="True acceleration",
    )

    axes[0].plot(
        time,
        measured_acceleration,
        color="#F97316",
        linewidth=0.9,
        alpha=0.75,
        label="Virtual IMU measurement",
    )

    invalid_indices = ~valid_samples

    if np.any(invalid_indices):
        axes[0].scatter(
            time[invalid_indices],
            measured_acceleration[invalid_indices],
            color="#DC2626",
            marker="x",
            s=18,
            label="Dropped sample",
        )

    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title("Virtual IMU Measurement Validation")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        time,
        measurement_error,
        color="#7C3AED",
        linewidth=0.9,
        label="Measurement error",
    )

    axes[1].axhline(
        0.0,
        color="#64748B",
        linewidth=0.8,
    )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (m/s²)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_imu_processing_comparison(
    time: NDArray[np.float64],
    true_acceleration: NDArray[np.float64],
    raw_acceleration: NDArray[np.float64],
    processed_acceleration: NDArray[np.float64],
    valid_samples: NDArray[np.bool_],
    sensor_faults: NDArray[np.bool_],
    output_path: str | Path,
) -> None:
    """绘制IMU原始测量与预处理结果。"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    raw_error = raw_acceleration - true_acceleration
    processed_error = processed_acceleration - true_acceleration

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        time,
        true_acceleration,
        color="#1D4ED8",
        linewidth=1.5,
        label="True acceleration",
    )

    axes[0].plot(
        time,
        raw_acceleration,
        color="#F97316",
        linewidth=0.8,
        alpha=0.55,
        label="Raw IMU",
    )

    axes[0].plot(
        time,
        processed_acceleration,
        color="#16A34A",
        linewidth=1.2,
        label="Processed IMU",
    )

    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title("Virtual IMU Processing and Fault Detection")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        time,
        raw_error,
        color="#F97316",
        linewidth=0.8,
        alpha=0.7,
        label="Raw measurement error",
    )

    axes[1].plot(
        time,
        processed_error,
        color="#7C3AED",
        linewidth=1.0,
        label="Processed measurement error",
    )

    axes[1].axhline(
        0.0,
        color="#64748B",
        linewidth=0.8,
    )

    axes[1].set_ylabel("Error (m/s²)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    invalid_samples = ~valid_samples

    axes[2].step(
        time,
        invalid_samples.astype(float),
        where="post",
        color="#DC2626",
        linewidth=1.2,
        label="Invalid input",
    )

    axes[2].step(
        time,
        sensor_faults.astype(float),
        where="post",
        color="#7C3AED",
        linewidth=1.5,
        label="Sensor fault",
    )

    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Fault state")
    axes[2].set_yticks([0.0, 1.0])
    axes[2].set_yticklabels(["Normal", "Active"])
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_state_observer_validation(
    time: NDArray[np.float64],
    true_displacement: NDArray[np.float64],
    estimated_displacement: NDArray[np.float64],
    true_velocity: NDArray[np.float64],
    estimated_velocity: NDArray[np.float64],
    valid_measurements: NDArray[np.bool_],
    output_path: str | Path,
) -> None:
    """绘制状态观测器估计结果。"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    displacement_error = estimated_displacement - true_displacement

    velocity_error = estimated_velocity - true_velocity

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        time,
        true_displacement * 1000.0,
        color="#1D4ED8",
        linewidth=1.5,
        label="True displacement",
    )

    axes[0].plot(
        time,
        estimated_displacement * 1000.0,
        color="#F97316",
        linewidth=1.1,
        linestyle="--",
        label="Estimated displacement",
    )

    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title("Luenberger Observer Validation")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        time,
        true_velocity,
        color="#16A34A",
        linewidth=1.5,
        label="True velocity",
    )

    axes[1].plot(
        time,
        estimated_velocity,
        color="#7C3AED",
        linewidth=1.1,
        linestyle="--",
        label="Estimated velocity",
    )

    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        time,
        displacement_error * 1000.0,
        color="#DC2626",
        linewidth=1.0,
        label="Displacement error",
    )

    axes[2].plot(
        time,
        velocity_error,
        color="#0891B2",
        linewidth=1.0,
        label="Velocity error",
    )

    invalid_indices = ~valid_measurements

    if np.any(invalid_indices):
        axes[2].scatter(
            time[invalid_indices],
            np.zeros(np.count_nonzero(invalid_indices)),
            color="#111827",
            marker="x",
            s=16,
            label="Dropped measurement",
        )

    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Estimation error")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_observer_pid_comparison(
    open_loop_result: SimulationResult,
    ideal_pid_result: SimulationResult,
    observer_pid_result: SimulationResult,
    estimated_displacement: NDArray[np.float64],
    estimated_velocity: NDArray[np.float64],
    output_path: str | Path,
) -> None:
    """绘制无控制、理想PID及观测器PID对比。"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        open_loop_result.time,
        open_loop_result.displacement * 1000.0,
        color="#DC2626",
        linewidth=1.2,
        label="Without control",
    )

    axes[0].plot(
        ideal_pid_result.time,
        ideal_pid_result.displacement * 1000.0,
        color="#2563EB",
        linewidth=1.3,
        label="Ideal-state PID",
    )

    axes[0].plot(
        observer_pid_result.time,
        observer_pid_result.displacement * 1000.0,
        color="#16A34A",
        linewidth=1.3,
        label="Observer-based PID",
    )

    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title("Ideal-State and Observer-Based PID Comparison")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        observer_pid_result.time,
        observer_pid_result.displacement * 1000.0,
        color="#1D4ED8",
        linewidth=1.4,
        label="True displacement",
    )

    axes[1].plot(
        observer_pid_result.time,
        estimated_displacement * 1000.0,
        color="#F97316",
        linewidth=1.1,
        linestyle="--",
        label="Estimated displacement",
    )

    axes[1].plot(
        observer_pid_result.time,
        estimated_velocity,
        color="#7C3AED",
        linewidth=0.9,
        alpha=0.8,
        label="Estimated velocity",
    )

    axes[1].set_ylabel("Estimated state")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        ideal_pid_result.time,
        ideal_pid_result.control_force,
        color="#2563EB",
        linewidth=1.2,
        label="Ideal PID force",
    )

    axes[2].plot(
        observer_pid_result.time,
        observer_pid_result.control_force,
        color="#16A34A",
        linewidth=1.2,
        label="Observer PID force",
    )

    axes[2].plot(
        observer_pid_result.time,
        observer_pid_result.disturbance_force,
        color="#9333EA",
        linewidth=1.0,
        label="External disturbance",
    )

    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Force (N)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_actuator_pid_comparison(
    open_loop_result: SimulationResult,
    ideal_actuator_result: SimulationResult,
    nonideal_actuator_result: SimulationResult,
    requested_force: NDArray[np.float64],
    actuator_saturated: NDArray[np.bool_],
    output_path: str | Path,
) -> None:
    """绘制理想执行器与非理想音圈执行器的闭环对比。"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    force_tracking_error = requested_force - nonideal_actuator_result.control_force

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 12),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        open_loop_result.time,
        open_loop_result.displacement * 1000.0,
        color="#DC2626",
        linewidth=1.1,
        label="Without control",
    )

    axes[0].plot(
        ideal_actuator_result.time,
        ideal_actuator_result.displacement * 1000.0,
        color="#2563EB",
        linewidth=1.3,
        label="Ideal actuator",
    )

    axes[0].plot(
        nonideal_actuator_result.time,
        nonideal_actuator_result.displacement * 1000.0,
        color="#16A34A",
        linewidth=1.3,
        label="Nonideal voice-coil actuator",
    )

    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title("Observer PID with Ideal and Nonideal Actuator")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        nonideal_actuator_result.time,
        requested_force,
        color="#F97316",
        linewidth=1.1,
        label="PID requested force",
    )

    axes[1].plot(
        nonideal_actuator_result.time,
        nonideal_actuator_result.control_force,
        color="#16A34A",
        linewidth=1.3,
        label="Actuator applied force",
    )

    axes[1].plot(
        nonideal_actuator_result.time,
        nonideal_actuator_result.disturbance_force,
        color="#9333EA",
        linewidth=1.0,
        label="External disturbance",
    )

    axes[1].set_ylabel("Force (N)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        nonideal_actuator_result.time,
        force_tracking_error,
        color="#DC2626",
        linewidth=1.1,
        label="Force tracking error",
    )

    axes[2].axhline(
        0.0,
        color="#64748B",
        linewidth=0.8,
    )

    axes[2].set_ylabel("Force error (N)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    axes[3].step(
        nonideal_actuator_result.time,
        actuator_saturated.astype(float),
        where="post",
        color="#B91C1C",
        linewidth=1.2,
        label="Actuator saturation",
    )

    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Saturation")
    axes[3].set_yticks([0.0, 1.0])
    axes[3].set_yticklabels(["Inactive", "Active"])
    axes[3].grid(alpha=0.25)
    axes[3].legend()

    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_embedded_timing_comparison(
    fast_result: SimulationResult,
    embedded_result: SimulationResult,
    missed_deadline_result: SimulationResult,
    embedded_requested_force: NDArray[np.float64],
    embedded_task_executed: NDArray[np.bool_],
    output_path: str | Path,
) -> None:
    """绘制不同控制周期和计算延迟下的闭环响应。"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 12),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        fast_result.time,
        fast_result.displacement * 1000.0,
        color="#2563EB",
        linewidth=1.2,
        label="1 ms period, no delay",
    )

    axes[0].plot(
        embedded_result.time,
        embedded_result.displacement * 1000.0,
        color="#16A34A",
        linewidth=1.3,
        label="5 ms period, 2 ms delay",
    )

    axes[0].plot(
        missed_deadline_result.time,
        missed_deadline_result.displacement * 1000.0,
        color="#DC2626",
        linewidth=1.1,
        label="5 ms period, 6 ms delay",
    )

    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title("Embedded Control Period and Computation Delay")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        fast_result.time,
        fast_result.control_force,
        color="#2563EB",
        linewidth=1.0,
        label="Fast-loop applied force",
    )

    axes[1].plot(
        embedded_result.time,
        embedded_result.control_force,
        color="#16A34A",
        linewidth=1.2,
        label="Embedded-loop applied force",
    )

    axes[1].plot(
        missed_deadline_result.time,
        missed_deadline_result.control_force,
        color="#DC2626",
        linewidth=1.0,
        label="Missed-deadline applied force",
    )

    axes[1].set_ylabel("Applied force (N)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        embedded_result.time,
        embedded_requested_force,
        color="#F97316",
        linewidth=1.0,
        label="Embedded PID command",
    )

    axes[2].plot(
        embedded_result.time,
        embedded_result.control_force,
        color="#16A34A",
        linewidth=1.2,
        label="Actuator force",
    )

    axes[2].set_ylabel("Force (N)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    axes[3].step(
        embedded_result.time,
        embedded_task_executed.astype(float),
        where="post",
        color="#7C3AED",
        linewidth=1.0,
        label="Control task executed",
    )

    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Task state")
    axes[3].set_yticks([0.0, 1.0])
    axes[3].set_yticklabels(["Idle", "Executed"])
    axes[3].grid(alpha=0.25)
    axes[3].legend()

    figure.savefig(destination, dpi=180)
    plt.close(figure)
