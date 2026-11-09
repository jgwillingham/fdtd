# fdtd

A finite-difference time-domain solver for Maxwell's equations, written in PyTorch. It supports 2D TE, 2D TM, and full 3D simulations, with a convolutional PML (CPML) for open boundaries.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/).

```
uv sync --extra viz --extra hdf5
uv run pytest                     # fast tests
uv run pytest -o addopts="" -m slow   # long stability and 3D reflection tests
uv run python benchmarks/bench_step.py
uv run --extra viz python examples/dipole_2d.py
uv run --extra viz python examples/dipole_over_earth.py   # batched sweep of earth conductivity
```

The default wheels are CPU-only on Windows. Pass `device="cuda"` to `Simulation` once a CUDA build of torch is installed.

## Usage

```python
import fdtd
from fdtd import C0, CPML, Grid, Mode, Simulation

frequency = 3e9
dx = C0 / frequency / 20
grid = Grid((200, 200), dx, Mode.TM, pml_cells=20)

source = fdtd.PointSource("Ez", (100 * dx, 100 * dx), fdtd.ModulatedGaussian(frequency, 1 / frequency))
snapshot = fdtd.Snapshot(("Ez",), every=10)

sim = Simulation(grid, sources=[source], monitors=[snapshot], pml=CPML(frequency))
sim.run(steps=2000)

x, y = grid.coords("Ez")          # node coordinates of snapshot.frames["Ez"]
times = snapshot.times["Ez"]
```

### Batches

`batch_size` steps several independent runs together on one grid, so every update launches once for the whole batch. The grid, the time step, the step count, and the CPML are shared. Sources and materials can differ per element:

```python
sources = [
    [fdtd.PointSource("Ez", (50 * dx, 100 * dx), pulse)],
    [fdtd.PointSource("Ez", (150 * dx, 60 * dx), other_pulse, amplitude=2.0)],
    [],
]
eps = torch.ones(3, 200, 200)     # or shape (3,) for a uniform medium per element
sim = Simulation(grid, Materials(eps=eps), sources, pml=CPML(frequency), batch_size=3)
```

A flat list of sources is shared by every element. Each material parameter may be a scalar, a tensor of `grid.shape`, a tensor of shape `(batch_size,)`, or a tensor of shape `(batch_size, *grid.shape)`. Fields, snapshot frames, probe values, and energies gain a leading batch axis, while an unbatched simulation has none.

### Saving to HDF5

`fdtd.io.HDF5Writer` (needs the `hdf5` extra) streams results to disk as the simulation runs, so long runs never hold their snapshots in memory. The monitors you pass it choose what is saved, and each becomes a named group:

```python
from fdtd.io import HDF5Writer

outputs = {
    "fields": fdtd.Snapshot(("Ez", "Hx", "Hy"), every=10),
    "receiver": fdtd.Probe("Ez", (150 * dx, 100 * dx)),
    "energy": fdtd.Energy(every=50),
}
with HDF5Writer("run.h5", sim, outputs, dtype=np.float32, compression="gzip"):
    sim.run(steps=2000)
```

The file also records everything needed to interpret and reproduce the data: the grid, dtype, device, and batch size; the materials as given; the CPML settings; and every source, with its batch element and its waveform's parameters as JSON. Each saved quantity sits at `/outputs/<name>/<key>/` with `values` of shape `(samples, [batch], ...)`, `time`, `step`, and, for snapshots, the node coordinates `x`, `y`, `z` of that component. The docstring of `fdtd/io.py` lists the full layout.

## Conventions

- **Units:** SI throughout. `eps` and `mu` are relative, `sigma` is in S/m, and sources are current densities in A/m².
- **Staggering:** `OFFSETS` in `grid.py` is the only definition. Node `i` of a component sits at `(i + offset) * dx`, with H half a cell ahead of E along the axes of its curl. Always take coordinates from `Grid.coords`.
- **Time:** after `n` steps, E is at `n * dt` and H at `(n - 1/2) * dt`. `Simulation.field_time` and the monitors account for this.
- **Boundaries:** every outer face is PEC. The CPML adds `pml_cells` outside the physical domain, and monitors report only the physical domain.
- **Modes:** TE is `Ex, Ey, Hz` and TM is `Ez, Hx, Hy`, both uniform along z.

## CPML settings

`CPML` needs a `reference_frequency`, usually the lowest frequency of interest. It sets the alpha term, which has no safe fixed value: if alpha is much larger than the simulated angular frequency, the layer stops absorbing.

Reflection measured against a large reference domain, with a probe 2 cells from the layer, at 20 cells per wavelength:

| PML cells | 5 | 10 | 20 | 30 |
|---|---|---|---|---|
| TE reflection (dB) | -50 | -89 | -108 | -119 |

At 20 cells, TM gives -107 dB and 3D gives -109 dB.

`ModulatedGaussian` defaults to a delay of five widths. Shorter delays truncate the pulse and leave a net charge behind, whose static field no boundary can absorb.

## Performance

Measured in Mcells/s on 4 CPU threads with torch 2.15, on live fields:

| Case | float32 | float64 | float32 + 20-cell PML |
|---|---|---|---|
| TE 205² | 179 | 203 | 82 |
| TE 1000² | 541 | 217 | 299 |
| TE 4000² | 356 | 181 | 322 |
| 3D 100³ | 153 | 75 | 58 |
| 3D 256³ | 119 | 60 | 69 |

PML throughput counts the layer's cells, so it depends on how much of the grid the layer occupies.

Batching pays off most on small grids, where each update is too short to cover its fixed cost. TE 101² with a 20-cell PML, float32, counting every element's cells:

| Batch size | 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| Mcells/s | 33 | 65 | 140 | 176 |

## Layout

| Module | Purpose |
|---|---|
| `grid.py` | Geometry, staggering, stability limit, and the table of curl terms |
| `materials.py` | Material parameters and update coefficients |
| `fields.py` | Field tensors and scratch space |
| `kernels.py` | Stencils built from the curl terms, shared by all modes |
| `pml.py` | CPML profiles and slab updates |
| `sources.py` | Waveforms and point sources |
| `monitors.py` | Probe, Snapshot, and Energy recorders |
| `simulation.py` | Time stepping |
| `io.py` | Optional HDF5 output, with the simulation's metadata |
| `viz.py` | Optional plotting helpers |
