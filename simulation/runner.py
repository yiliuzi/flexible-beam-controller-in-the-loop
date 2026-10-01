"""柔性梁固定步长仿真运行器。"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from plant.beam_sdof import BeamState, SingleDegreeBeam
from simulation.config import SimulationConfig

DisturbanceFunction = Callable[[float], float]
ControllerFunction = Callable[[float, BeamState], float]


@dataclass(slots=True)
class SimulationResult:
    """单次柔性梁仿真实验结果。"""

    time: NDArray[np.float64]
    displacement: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    control_force: NDArray[np.float64]
    disturbance_force: NDArray[np.float64]

    def to_dataframe(self) -> pd.DataFrame:
        """转换为便于保存和分析的数据表。"""

        return pd.DataFrame(
            {
                "time_s": self.time,
                "displacement_m": self.displacement,
                "velocity_m_s": self.velocity,
                "acceleration_m_s2": self.acceleration,
                "control_force_n": self.control_force,
                "disturbance_force_n": self.disturbance_force,
            }
        )


def run_simulation(
    beam: SingleDegreeBeam,
    config: SimulationConfig,
    disturbance: DisturbanceFunction | None = None,
    controller: ControllerFunction | None = None,
) -> SimulationResult:
    """运行固定步长柔性梁仿真。"""

    sample_count = int(round(config.duration / config.time_step)) + 1

    time_values = np.arange(sample_count, dtype=np.float64) * config.time_step
    displacement_values = np.zeros(sample_count, dtype=np.float64)
    velocity_values = np.zeros(sample_count, dtype=np.float64)
    acceleration_values = np.zeros(sample_count, dtype=np.float64)
    control_values = np.zeros(sample_count, dtype=np.float64)
    disturbance_values = np.zeros(sample_count, dtype=np.float64)

    for sample_index, current_time in enumerate(time_values):
        state = BeamState(
            displacement=beam.state.displacement,
            velocity=beam.state.velocity,
        )

        disturbance_force = disturbance(float(current_time)) if disturbance else 0.0
        control_force = controller(float(current_time), state) if controller else 0.0

        displacement_values[sample_index] = state.displacement
        velocity_values[sample_index] = state.velocity
        control_values[sample_index] = control_force
        disturbance_values[sample_index] = disturbance_force
        acceleration_values[sample_index] = beam.acceleration(
            state=state,
            control_force=control_force,
            disturbance_force=disturbance_force,
        )

        if sample_index < sample_count - 1:
            beam.step(
                time_step=config.time_step,
                control_force=control_force,
                disturbance_force=disturbance_force,
            )

    return SimulationResult(
        time=time_values,
        displacement=displacement_values,
        velocity=velocity_values,
        acceleration=acceleration_values,
        control_force=control_values,
        disturbance_force=disturbance_values,
    )


def run_open_loop_simulation(
    beam: SingleDegreeBeam,
    config: SimulationConfig,
    disturbance: DisturbanceFunction | None = None,
) -> SimulationResult:
    """运行无控制器的开环仿真。"""

    return run_simulation(
        beam=beam,
        config=config,
        disturbance=disturbance,
        controller=None,
    )
