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
