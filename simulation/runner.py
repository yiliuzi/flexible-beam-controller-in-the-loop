"""柔性梁固定步长仿真运行器。"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from plant.beam_sdof import SingleDegreeBeam
from simulation.config import SimulationConfig

DisturbanceFunction = Callable[[float], float]


@dataclass(slots=True)
class SimulationResult:
    """单次柔性梁仿真实验结果。"""

    time: NDArray[np.float64]
    displacement: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    disturbance_force: NDArray[np.float64]

    def to_dataframe(self) -> pd.DataFrame:
        """转换为便于保存和分析的数据表。"""

        return pd.DataFrame(
            {
                "time_s": self.time,
                "displacement_m": self.displacement,
                "velocity_m_s": self.velocity,
                "acceleration_m_s2": self.acceleration,
                "disturbance_force_n": self.disturbance_force,
            }
        )


def run_open_loop_simulation(
    beam: SingleDegreeBeam,
    config: SimulationConfig,
    disturbance: DisturbanceFunction | None = None,
) -> SimulationResult:
    """运行无控制器的柔性梁开环仿真。"""

    sample_count = int(round(config.duration / config.time_step)) + 1

    time_values = np.arange(sample_count, dtype=np.float64) * config.time_step
    displacement_values = np.zeros(sample_count, dtype=np.float64)
    velocity_values = np.zeros(sample_count, dtype=np.float64)
    acceleration_values = np.zeros(sample_count, dtype=np.float64)
    disturbance_values = np.zeros(sample_count, dtype=np.float64)

    for sample_index, current_time in enumerate(time_values):
        disturbance_force = disturbance(float(current_time)) if disturbance else 0.0

        displacement_values[sample_index] = beam.state.displacement
        velocity_values[sample_index] = beam.state.velocity
        disturbance_values[sample_index] = disturbance_force
        acceleration_values[sample_index] = beam.acceleration(disturbance_force=disturbance_force)

        if sample_index < sample_count - 1:
            beam.step(
                time_step=config.time_step,
                disturbance_force=disturbance_force,
            )

    return SimulationResult(
        time=time_values,
        displacement=displacement_values,
        velocity=velocity_values,
        acceleration=acceleration_values,
        disturbance_force=disturbance_values,
    )
