"""A short dipole radiating into open space, in a TE slice.

The source is a short pulse, so anything left in the domain at the end shows what
the PML failed to absorb. Run with the viz extra:

    uv run --extra viz python examples/dipole_2d.py
    uv run --extra viz python examples/dipole_2d.py --output dipole.png
"""

import argparse

import matplotlib.pyplot as plt

import fdtd
from fdtd import C0, CPML, Grid, Mode, Simulation
from fdtd.viz import animate_snapshot, plot_frames

WAVELENGTH = 245.0  # m
POINTS_PER_WAVELENGTH = 50
DOMAIN = 1000.0  # m
STEPS = 2000
FRAME_INTERVAL = 10


def simulate() -> tuple[Grid, fdtd.Snapshot]:
    frequency = C0 / WAVELENGTH
    dx = WAVELENGTH / POINTS_PER_WAVELENGTH
    nodes = round(DOMAIN / dx) + 1
    grid = Grid((nodes, nodes), dx, Mode.TE, pml_cells=20)

    source = fdtd.PointSource(
        "Ey", (DOMAIN / 2, DOMAIN / 2), fdtd.ModulatedGaussian(frequency, 1 / frequency)
    )
    snapshot = fdtd.Snapshot(("Ex",), every=FRAME_INTERVAL)
    sim = Simulation(grid, sources=[source], monitors=[snapshot], pml=CPML(frequency))

    print(f"Grid {grid.dims}, dx = {dx:.2f} m, dt = {grid.dt * 1e9:.2f} ns")
    sim.run(STEPS)
    return grid, snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="save selected frames here instead of animating")
    args = parser.parse_args()

    grid, snapshot = simulate()
    if args.output:
        count = STEPS // FRAME_INTERVAL
        plot_frames(snapshot, grid, "Ex", [count // 4, count // 2, count - 1]).savefig(
            args.output, dpi=120
        )
    else:
        animation = animate_snapshot(snapshot, grid, "Ex")
        plt.show()
        del animation


if __name__ == "__main__":
    main()
