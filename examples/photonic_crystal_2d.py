"""Light turning a sharp corner in a photonic crystal, in a TM slice.

A square lattice of dielectric rods has a band gap, so light at a frequency inside it cannot enter
the crystal. Removing rods along an L-shaped path leaves a channel that guides the light instead.
Run with the viz extra to watch the field evolve, or save the animation or its final frame:

    uv run --extra viz python examples/photonic_crystal_2d.py
    uv run --extra viz python examples/photonic_crystal_2d.py --output crystal.gif
    uv run --extra viz python examples/photonic_crystal_2d.py --output crystal.png
"""

import argparse
from itertools import product

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

import fdtd
from fdtd import C0, CPML, Grid, Materials, Mode, Simulation

LATTICE = 1e-6  # lattice constant (m)
ROD_RADIUS = 0.18 * LATTICE
ROD_EPS = 11.56
FREQUENCY = 0.353 * C0 / LATTICE  # inside the TM band gap of this lattice

RODS = 21  # rods along each side of the crystal
MARGIN = 2  # air between the crystal and the PML (lattice constants)
CELLS_PER_LATTICE = 20
STEPS = 1200
STEPS_PER_FRAME = 4  # about 5 frames per optical period
FIELD_LIMIT = 3e-7  # colour range (V/m), a third of the peak for the 1 A/m^2 source
GIF_DPI = 60

CORNER = RODS // 2


def make_grid() -> Grid:
    nodes = (RODS + 2 * MARGIN) * CELLS_PER_LATTICE + 1
    return Grid((nodes, nodes), LATTICE / CELLS_PER_LATTICE, Mode.TM, pml_cells=20)


def lattice_site(i: int, j: int) -> tuple[float, float]:
    """Centre (m) of the rod at column ``i`` and row ``j``."""
    return (MARGIN + i + 0.5) * LATTICE, (MARGIN + j + 0.5) * LATTICE


def in_waveguide(i: int, j: int) -> bool:
    """True along the path in from the left, then up and out of the top."""
    return (j == CORNER and i <= CORNER) or (i == CORNER and j >= CORNER)


def crystal_permittivity(grid: Grid, with_waveguide: bool = True) -> torch.Tensor:
    x, y = torch.meshgrid(*grid.coords("Ez"), indexing="ij")
    eps = torch.ones_like(x)
    for i, j in product(range(RODS), repeat=2):
        if with_waveguide and in_waveguide(i, j):
            continue
        cx, cy = lattice_site(i, j)
        eps[(x - cx) ** 2 + (y - cy) ** 2 <= ROD_RADIUS**2] = ROD_EPS
    return eps


def make_figure(grid: Grid, eps: torch.Tensor):
    """Figure with the rods outlined, and an empty image ready to receive Ez."""
    x, y = (axis.numpy() / LATTICE for axis in grid.coords("Ez"))
    figure, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        torch.zeros(grid.shape).T.numpy(),
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="RdBu_r",
        vmin=-FIELD_LIMIT,
        vmax=FIELD_LIMIT,
    )
    ax.contour(x, y, eps.T.numpy(), levels=[0.5 * (1 + ROD_EPS)], colors="black", linewidths=0.5)
    figure.colorbar(image, label="Ez (V/m)")
    ax.set(xlabel="x / a", ylabel="y / a")
    return figure, ax, image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="save here instead of showing: .png or .gif")
    args = parser.parse_args()

    grid = make_grid()
    eps = crystal_permittivity(grid)
    source = fdtd.PointSource(
        "Ez", lattice_site(2, CORNER), fdtd.Sinusoid(FREQUENCY, ramp=10 / FREQUENCY)
    )
    sim = Simulation(grid, Materials(eps=eps), [source], pml=CPML(FREQUENCY))

    print(f"Grid {grid.dims}, dt = {grid.dt * 1e15:.3f} fs")
    figure, ax, image = make_figure(grid, eps)

    def advance(_frame: int) -> None:
        """Step the simulation live, so no frames need to be held in memory."""
        sim.run(STEPS_PER_FRAME)
        image.set_data(sim.field("Ez").T.numpy())
        ax.set_title(f"Ez at t = {sim.time * 1e15:.0f} fs")

    frames = STEPS // STEPS_PER_FRAME
    if args.output and args.output.endswith(".png"):
        for frame in range(frames):
            advance(frame)
        figure.savefig(args.output, dpi=120)
        return

    animation = FuncAnimation(
        figure, advance, frames=frames, interval=30, repeat=True, cache_frame_data=False
    )
    if args.output:
        animation.save(args.output, fps=30, dpi=GIF_DPI)
    else:
        plt.show()


if __name__ == "__main__":
    main()
