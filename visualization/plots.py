"""柔性梁实验结果可视化。"""

from pathlib import Path

import matplotlib.pyplot as plt

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
