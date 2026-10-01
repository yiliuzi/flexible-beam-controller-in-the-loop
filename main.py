"""柔性梁控制器在环平台入口。"""

from pathlib import Path

from plant.beam_sdof import BeamParameters, SingleDegreeBeam
from plant.disturbances import impulse_disturbance
from simulation.config import SimulationConfig
from simulation.runner import run_open_loop_simulation
from visualization.plots import plot_free_vibration_response


def main() -> None:
    """运行冲击扰动下的柔性梁自由振动实验。"""

    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "free_vibration_response.csv"
    figure_path = project_root / "results" / "free_vibration_response.png"

    config = SimulationConfig(
        time_step=0.001,
        duration=5.0,
    )

    beam = SingleDegreeBeam(
        parameters=BeamParameters(
            mass=0.08,
            damping=0.12,
            stiffness=18.0,
        )
    )

    def disturbance(current_time: float) -> float:
        return impulse_disturbance(
            time=current_time,
            start_time=0.50,
            duration=0.05,
            amplitude=1.50,
        )

    result = run_open_loop_simulation(
        beam=beam,
        config=config,
        disturbance=disturbance,
    )

    result.to_dataframe().to_csv(
        data_path,
        index=False,
    )

    plot_free_vibration_response(
        result=result,
        output_path=figure_path,
    )

    maximum_displacement = float(abs(result.displacement).max())
    maximum_acceleration = float(abs(result.acceleration).max())

    print("Simulation completed")
    print(f"Samples:              {len(result.time)}")
    print(f"Maximum displacement: {maximum_displacement * 1000.0:.3f} mm")
    print(f"Maximum acceleration: {maximum_acceleration:.3f} m/s^2")
    print(f"CSV result:           {data_path}")
    print(f"Response figure:      {figure_path}")


if __name__ == "__main__":
    main()
