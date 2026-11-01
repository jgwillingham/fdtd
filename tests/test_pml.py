"""Validation of the CPML against reference runs with no boundary in reach."""

import numpy as np
import pytest
import torch

import fdtd
from fdtd import C0, CPML, Grid, Materials, Mode, Simulation

CELLS_PER_WAVELENGTH = 20


def run(mode, shape, cells, offset, steps, dx=1.0, pml=True, **cpml_options):
    """Probe a source and its nearby point, with both placed relative to the domain's origin."""
    frequency = C0 / (CELLS_PER_WAVELENGTH * dx)
    component = {Mode.TE: "Ey", Mode.TM: "Ez", Mode.FULL_3D: "Ez"}[mode]
    grid = Grid(shape, dx, mode, pml_cells=cells)
    position = [n // 2 * dx for n in shape]
    source = fdtd.PointSource(
        component, tuple(position), fdtd.ModulatedGaussian(frequency, 1 / frequency)
    )
    position[0] += offset * dx
    probe = fdtd.Probe(component, tuple(position))
    sim = Simulation(
        grid,
        sources=[source],
        monitors=[probe],
        pml=CPML(frequency, **cpml_options) if pml else None,
        dtype=torch.float64,
    )
    sim.run(steps)
    return np.array(probe.values)


def reflection_db(mode, shape, cells, reference_shape, steps, **cpml_options):
    """Peak difference from the reference at a probe 2 cells from the PML, relative to its peak."""
    probe_offset = shape[0] // 2 - 3
    absorbed = run(mode, shape, cells, probe_offset, steps, **cpml_options)
    reference = run(mode, reference_shape, 0, probe_offset, steps, pml=False)
    return 20 * np.log10(np.abs(absorbed - reference).max() / np.abs(reference).max())


@pytest.mark.parametrize("mode", [Mode.TE, Mode.TM])
def test_reflection_2d(mode):
    assert reflection_db(mode, (60, 60), 20, (560, 560), 700) < -60


def test_reflection_3d():
    assert reflection_db(Mode.FULL_3D, (30, 30, 30), 20, (240, 240, 240), 300) < -60


def test_pml_and_grid_must_agree():
    with pytest.raises(ValueError, match="together"):
        Simulation(Grid((10, 10), 1.0, Mode.TE, pml_cells=4))
    with pytest.raises(ValueError, match="together"):
        Simulation(Grid((10, 10), 1.0, Mode.TE), pml=CPML(1e6))


def test_reset_clears_pml_memory():
    grid = Grid((30, 30), 1.0, Mode.TM, pml_cells=10)
    frequency = C0 / 20
    source = fdtd.PointSource("Ez", (15.0, 15.0), fdtd.ModulatedGaussian(frequency, 1 / frequency))
    sim = Simulation(grid, sources=[source], pml=CPML(frequency), dtype=torch.float64)
    sim.run(300)
    first = sim.field("Ez").clone()
    sim.reset()
    sim.run(300)
    assert torch.equal(first, sim.field("Ez"))


@pytest.mark.slow
@pytest.mark.parametrize("touching_dielectric", [False, True])
def test_long_run_stability(touching_dielectric):
    """No late-time growth, including with a dielectric running into the layer."""
    shape, dx = (60, 60), 1.0
    frequency = C0 / (CELLS_PER_WAVELENGTH * dx)
    eps = torch.ones(shape, dtype=torch.float64)
    if touching_dielectric:
        eps[40:] = 4.0
    energy = fdtd.Energy(every=1000)
    source = fdtd.PointSource("Ey", (30.0, 30.0), fdtd.ModulatedGaussian(frequency, 1 / frequency))
    sim = Simulation(
        Grid(shape, dx, Mode.TE, pml_cells=20),
        Materials(eps=eps),
        [source],
        [energy],
        CPML(frequency),
        dtype=torch.float64,
    )
    sim.run(100_000)
    values = np.array(energy.values)
    assert np.isfinite(values).all()
    assert values[-1] < 1e-6 * values.max()
    assert values[50:].max() <= 10 * values[50]
