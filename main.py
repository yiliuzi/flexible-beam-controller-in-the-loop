"""柔性梁PID主动振动控制实验。"""

from pathlib import Path

import numpy as np

from controllers.pid import PIDController, PIDParameters
from plant.beam_sdof import BeamParameters, BeamState, SingleDegreeBeam
from plant.disturbances import impulse_disturbance
from simulation.config import SimulationConfig
from simulation.runner import run_simulation
from visualization.plots import plot_pid_comparison


def calculate_rms(values: np.ndarray) -> float:
    """计算信号均方根。"""

    return float(np.sqrt(np.mean(np.square(values))))


def main() -> None:
    """对比无控制与PID控制的柔性梁响应。"""

    project_root = Path(__file__).resolve().parent

    open_loop_data_path = project_root / "data" / "open_loop_response.csv"
    controlled_data_path = project_root / "data" / "pid_control_response.csv"
    figure_path = project_root / "results" / "pid_control_comparison.png"

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

    open_loop_beam = SingleDegreeBeam(
        parameters=beam_parameters,
    )

    open_loop_result = run_simulation(
        beam=open_loop_beam,
        config=config,
        disturbance=disturbance,
    )

    pid_controller = PIDController(
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

    controlled_beam = SingleDegreeBeam(
        parameters=beam_parameters,
    )

    def controller(
        _current_time: float,
        state: BeamState,
    ) -> float:
        return pid_controller.update(
            reference=0.0,
            measurement=state.displacement,
            time_step=config.time_step,
        )

    controlled_result = run_simulation(
        beam=controlled_beam,
        config=config,
        disturbance=disturbance,
        controller=controller,
    )

    open_loop_result.to_dataframe().to_csv(
        open_loop_data_path,
        index=False,
    )

    controlled_result.to_dataframe().to_csv(
        controlled_data_path,
        index=False,
    )

    plot_pid_comparison(
        open_loop_result=open_loop_result,
        controlled_result=controlled_result,
        output_path=figure_path,
    )

    evaluation_mask = open_loop_result.time >= 0.50

    open_loop_rms = calculate_rms(open_loop_result.displacement[evaluation_mask])

    controlled_rms = calculate_rms(controlled_result.displacement[evaluation_mask])

    suppression_rate = (open_loop_rms - controlled_rms) / open_loop_rms * 100.0

    open_loop_peak = float(np.max(np.abs(open_loop_result.displacement)))

    controlled_peak = float(np.max(np.abs(controlled_result.displacement)))

    maximum_control_force = float(np.max(np.abs(controlled_result.control_force)))

    print("PID vibration-control experiment completed")
    print(f"Open-loop RMS:          {open_loop_rms * 1000.0:.4f} mm")
    print(f"PID-controlled RMS:     {controlled_rms * 1000.0:.4f} mm")
    print(f"RMS suppression rate:   {suppression_rate:.2f}%")
    print(f"Open-loop peak:         {open_loop_peak * 1000.0:.4f} mm")
    print(f"PID-controlled peak:    {controlled_peak * 1000.0:.4f} mm")
    print(f"Maximum control force:  {maximum_control_force:.4f} N")
    print(f"Comparison figure:      {figure_path}")


if __name__ == "__main__":
    main()
