"""柔性梁控制器在环平台入口。"""

from plant.beam_sdof import BeamParameters, SingleDegreeBeam


def main() -> None:
    """运行单自由度柔性梁自由振动演示。"""

    time_step = 0.001
    simulation_time = 5.0
    total_steps = int(simulation_time / time_step)

    beam = SingleDegreeBeam(
        parameters=BeamParameters(
            mass=0.08,
            damping=0.12,
            stiffness=18.0,
        )
    )

    maximum_displacement = 0.0

    for step_index in range(total_steps):
        current_time = step_index * time_step

        disturbance_force = 1.5 if 0.50 <= current_time < 0.55 else 0.0

        state = beam.step(
            time_step=time_step,
            disturbance_force=disturbance_force,
        )

        maximum_displacement = max(
            maximum_displacement,
            abs(state.displacement),
        )

    print("Simulation completed")
    print(f"Maximum displacement: {maximum_displacement:.6f} m")
    print(f"Final displacement:   {beam.state.displacement:.6f} m")
    print(f"Final velocity:       {beam.state.velocity:.6f} m/s")
    print(f"Final energy:         {beam.mechanical_energy():.8f} J")


if __name__ == "__main__":
    main()
