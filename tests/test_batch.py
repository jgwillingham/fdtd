"""A batch must evolve exactly as its elements would run one at a time."""

import numpy as np
import pytest
import torch

import fdtd
from fdtd import C0, CPML, Grid, Materials, Mode, Simulation

FREQUENCY = C0 / 20


def pulse(width_periods=1.0):
    return fdtd.ModulatedGaussian(FREQUENCY, width_periods / FREQUENCY)


def run_alone(grid, materials, sources, steps):
    sim = Simulation(
        grid,
        materials,
        sources,
        pml=CPML(FREQUENCY) if grid.pml_cells else None,
        dtype=torch.float64,
    )
    sim.run(steps)
    return sim


def assert_matches(batch, singles):
    for name in batch.grid.components:
        for element, single in enumerate(singles):
            expected = single.fields[name]
            scale = max(expected.abs().max().item(), 1e-300)
            torch.testing.assert_close(
                batch.fields[name][element], expected, rtol=0, atol=1e-12 * scale
            )


@pytest.mark.parametrize(
    ("mode", "shape", "components"),
    [(Mode.TE, (30, 24), ("Ey", "Ex")), (Mode.FULL_3D, (14, 12, 10), ("Ez", "Ex"))],
)
def test_different_sources_per_element(mode, shape, components):
    grid = Grid(shape, 1.0, mode, pml_cells=6)
    centre = tuple(n / 2 for n in shape)
    corner = tuple(n / 4 for n in shape)
    per_element = [
        [fdtd.PointSource(components[0], centre, pulse())],
        [  # two sources, one of them on another component, and a different waveform
            fdtd.PointSource(components[1], corner, pulse(0.5), amplitude=-2.0),
            fdtd.PointSource(components[0], centre, fdtd.Sinusoid(FREQUENCY, 3 / FREQUENCY)),
        ],
        [],  # no sources at all
    ]
    batch = Simulation(
        grid, sources=per_element, pml=CPML(FREQUENCY), dtype=torch.float64, batch_size=3
    )
    batch.run(150)
    assert_matches(batch, [run_alone(grid, None, s, 150) for s in per_element])
    assert batch.fields[components[0]][2].abs().max() == 0


def test_shared_sources_with_per_element_materials():
    shape = (32, 28)
    grid = Grid(shape, 1.0, Mode.TM, pml_cells=6)
    source = fdtd.PointSource("Ez", (10.0, 14.0), pulse())

    slab = torch.ones(shape, dtype=torch.float64)
    slab[18:] = 4.0
    uniform = torch.tensor([1.0, 2.25, 9.0], dtype=torch.float64)
    sigma = torch.tensor([0.0, 0.0, 1e-3])
    spatial = torch.stack([torch.ones(shape, dtype=torch.float64), slab, 2 * slab])
    cases = [
        (
            Materials(eps=uniform, sigma=sigma),
            [
                Materials(eps=e, sigma=s)
                for e, s in zip(uniform.tolist(), sigma.tolist(), strict=True)
            ],
        ),
        (Materials(eps=spatial, mu=slab), [Materials(eps=e, mu=slab) for e in spatial]),
    ]
    for batched_materials, singles in cases:
        batch = Simulation(
            grid,
            batched_materials,
            [source],
            pml=CPML(FREQUENCY),
            dtype=torch.float64,
            batch_size=3,
        )
        batch.run(150)
        assert_matches(batch, [run_alone(grid, m, [source], 150) for m in singles])


def test_batched_monitors_record_every_element():
    grid = Grid((24, 24), 1.0, Mode.TE, pml_cells=6)
    per_element = [[fdtd.PointSource("Ey", (12.0, 12.0), pulse(), amplitude=a)] for a in (1.0, 2.0)]
    probe = fdtd.Probe("Ey", (14.0, 12.0))
    energy = fdtd.Energy(every=5)
    snapshot = fdtd.Snapshot(("Hz",), every=50)
    sim = Simulation(
        grid,
        sources=per_element,
        monitors=[probe, energy, snapshot],
        pml=CPML(FREQUENCY),
        dtype=torch.float64,
        batch_size=2,
    )
    sim.run(100)

    values = np.array(probe.values)
    assert values.shape == (100, 2)
    np.testing.assert_allclose(values[:, 1], 2 * values[:, 0], rtol=1e-12, atol=0)
    energies = np.array(energy.values)
    assert energies.shape == (20, 2)
    np.testing.assert_allclose(energies[:, 1], 4 * energies[:, 0], rtol=1e-12)
    assert snapshot.frames["Hz"][0].shape == (2, *grid.shape)


def test_unbatched_fields_have_no_batch_axis():
    grid = Grid((10, 12), 1.0, Mode.TM)
    sim = Simulation(grid)
    assert sim.fields["Ez"].shape == grid.dims
    assert sim.field("Ez").shape == grid.shape
    assert isinstance(sim.energy(), float)
    batched = Simulation(grid, batch_size=4)
    assert batched.fields["Ez"].shape == (4, *grid.dims)
    assert batched.field("Ez").shape == (4, *grid.shape)
    assert len(batched.energy()) == 4


def test_source_table_spans_chunks():
    """Runs longer than one tabulated chunk, and split runs, give the same result."""
    grid = Grid((20, 20), 1.0, Mode.TE)
    source = fdtd.PointSource("Ey", (10.0, 10.0), fdtd.Sinusoid(FREQUENCY, 5 / FREQUENCY))
    whole = Simulation(grid, sources=[source], dtype=torch.float64)
    whole.run(fdtd.simulation.SOURCE_CHUNK + 100)
    split = Simulation(grid, sources=[source], dtype=torch.float64)
    for steps in (7, fdtd.simulation.SOURCE_CHUNK, 93):
        split.run(steps)
    assert torch.equal(whole.fields["Ey"], split.fields["Ey"])


def test_invalid_batches_rejected():
    grid = Grid((10, 10), 1.0, Mode.TE)
    source = fdtd.PointSource("Ey", (5.0, 5.0), pulse())
    with pytest.raises(ValueError, match="one per element"):
        Simulation(grid, sources=[[source]], batch_size=2)
    with pytest.raises(ValueError, match="one per element"):
        Simulation(grid, sources=[source, [source]], batch_size=2)
    with pytest.raises(ValueError, match="unbatched"):
        Simulation(grid, sources=[[source]])
    with pytest.raises(ValueError, match="Material shape"):
        Simulation(grid, Materials(eps=torch.ones(3)), batch_size=2)
    with pytest.raises(ValueError, match="positive"):
        Simulation(grid, batch_size=0)
