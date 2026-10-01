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
