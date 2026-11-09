"""Field of a vertical dipole over the earth, swept over the earth's conductivity in one batch.

A vertical current at a quarter wavelength above flat ground radiates continuously, in a TE slice
(so it is strictly a line source, uniform along z). A receiver at the same height, a few
wavelengths away, measures |Ey| once the field has settled. Every conductivity is one element of a
single batched simulation, along with two references: no ground at all (free space), and a ground
conductive enough to act as a perfect conductor.

    uv run --extra viz python examples/dipole_over_earth.py
    uv run --extra viz python examples/dipole_over_earth.py --output earth.png --device cuda

The highest conductivities have a skin depth near one cell, so their surface is only roughly
resolved. A finer DX resolves it at the cost of a longer run.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

import fdtd
from fdtd import C0, CPML, Grid, Materials, Mode, Simulation

FREQUENCY = 10e6  # Hz
WAVELENGTH = C0 / FREQUENCY
DX = WAVELENGTH / 100

EARTH_EPS = 10.0  # relative permittivity of moist soil
CONDUCTIVITIES = np.logspace(-4, 0, 15)  # S/m, dry ground to wet
CONDUCTOR = 1e4  # S/m, a perfect conductor for this grid

EARTH_DEPTH = 0.5 * WAVELENGTH  # earth above the bottom PML
AIR_HEIGHT = 1.0 * WAVELENGTH
MARGIN = 0.5 * WAVELENGTH  # air beside each antenna, before the PML
ANTENNA_HEIGHT = 0.25 * WAVELENGTH  # transmitter and receiver
DISTANCE = 3.0 * WAVELENGTH  # transmitter to receiver

PERIODS = 20  # long enough for the field at the receiver to settle
MEASURED_PERIODS = 2  # the amplitude is the peak over the last few periods

SERIES_COLOR = "#2a78d6"
REFERENCE_COLOR = "#52514e"
TEXT_COLOR = "#0b0b0b"


def make_grid() -> Grid:
    nx = round((2 * MARGIN + DISTANCE) / DX) + 1
    ny = round((EARTH_DEPTH + AIR_HEIGHT) / DX) + 1
    return Grid((nx, ny), DX, Mode.TE, pml_cells=20)


def earth_materials(grid: Grid, conductivities: list[float]) -> Materials:
    """One element per conductivity, with ``None`` standing for no earth at all."""
    batch = len(conductivities)
    eps = torch.ones(batch, *grid.shape, dtype=torch.float64)
    sigma = torch.zeros(batch, *grid.shape, dtype=torch.float64)
    surface = round(EARTH_DEPTH / DX)  # first row of air
    for element, conductivity in enumerate(conductivities):
        if conductivity is not None:
            eps[element, :, :surface] = EARTH_EPS
            sigma[element, :, :surface] = conductivity
    return Materials(eps=eps, sigma=sigma)


def simulate(device: str) -> np.ndarray:
    """Peak |Ey| at the receiver for every element: free space, conductor, then the sweep."""
    grid = make_grid()
    conductivities = [None, CONDUCTOR, *CONDUCTIVITIES.tolist()]
    height = EARTH_DEPTH + ANTENNA_HEIGHT
    source = fdtd.PointSource("Ey", (MARGIN, height), fdtd.Sinusoid(FREQUENCY, 3 / FREQUENCY))
    probe = fdtd.Probe("Ey", (MARGIN + DISTANCE, height))
    sim = Simulation(
        grid,
        earth_materials(grid, conductivities),
        [source],
        [probe],
        CPML(FREQUENCY),
        device=device,
        batch_size=len(conductivities),
    )
    print(f"Grid {grid.dims} x {len(conductivities)} runs, dt = {grid.dt * 1e9:.3f} ns")
    sim.run(duration=PERIODS / FREQUENCY)

    values = np.abs(np.array(probe.values))  # (steps, batch)
    window = round(MEASURED_PERIODS / FREQUENCY / grid.dt)
    amplitude = values[-window:].max(axis=0)
    previous = values[-2 * window : -window].max(axis=0)
    drift = np.abs(20 * np.log10(amplitude / previous)).max()
    print(f"Largest change over the last {MEASURED_PERIODS} periods: {drift:.2f} dB")
    return amplitude


def plot(amplitude: np.ndarray):
    """|Ey| against conductivity, in dB relative to free space."""
    level = 20 * np.log10(amplitude / amplitude[0])
    conductor, sweep = level[1], level[2:]

    figure, ax = plt.subplots(figsize=(7, 4.2))
    for value, label in ((0.0, "free space"), (conductor, "perfect conductor")):
        ax.axhline(value, color=REFERENCE_COLOR, linestyle="--", linewidth=1)
        ax.annotate(
            label,
            (CONDUCTIVITIES[0], value),
            xytext=(0, 4),
            textcoords="offset points",
            color=REFERENCE_COLOR,
            fontsize=9,
        )
    ax.semilogx(CONDUCTIVITIES, sweep, color=SERIES_COLOR, linewidth=2, marker="o", markersize=5)

    ax.set_xlabel("Earth conductivity (S/m)", color=TEXT_COLOR)
    ax.set_ylabel("|Ey| relative to free space (dB)", color=TEXT_COLOR)
    ax.set_title(
        f"Vertical dipole over earth (εr = {EARTH_EPS:g}), {FREQUENCY / 1e6:g} MHz, "
        f"receiver {DISTANCE / WAVELENGTH:g} λ away",
        color=TEXT_COLOR,
        fontsize=11,
    )
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="save the plot here instead of showing it")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cuda")
    args = parser.parse_args()

    figure = plot(simulate(args.device))
    if args.output:
        figure.savefig(args.output, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
