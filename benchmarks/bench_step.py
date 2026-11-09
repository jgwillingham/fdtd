"""Measure stepping throughput in Mcells/s.

uv run python benchmarks/bench_step.py
"""

import time

import torch

from fdtd import C0, CPML, Grid, Mode, ModulatedGaussian, PointSource, Simulation

CASES = [
    ("TE 205^2", (205, 205), Mode.TE, 3000),
    ("TE 1000^2", (1000, 1000), Mode.TE, 300),
    ("TE 4000^2", (4000, 4000), Mode.TE, 30),
    ("3D 100^3", (100, 100, 100), Mode.FULL_3D, 100),
    ("3D 256^3", (256, 256, 256), Mode.FULL_3D, 10),
]


BATCH_CASE = ("TE 101^2", (101, 101), Mode.TE, 2000)
BATCH_SIZES = (1, 4, 16, 64)


def throughput(shape, mode, steps, pml_cells, dtype, batch_size=None) -> float:
    """Mcells/s, counting every cell of every batch element."""
    grid = Grid(shape, 1.0, mode, pml_cells=pml_cells)
    pml = CPML(C0 / 20) if pml_cells else None
    component = "Ey" if mode is Mode.TE else "Ez"
    centre = tuple(n / 2 for n in shape)
    source = PointSource(component, centre, ModulatedGaussian(C0 / 20, 20 / C0))
    sim = Simulation(grid, sources=[source], pml=pml, dtype=dtype, batch_size=batch_size)
    start = time.perf_counter()
    sim.run(steps)
    elapsed = time.perf_counter() - start
    cells = torch.tensor(grid.dims, dtype=torch.float64).prod().item() * (batch_size or 1)
    return cells * steps / elapsed / 1e6


def main() -> None:
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    print(f"{'case':<12}{'float32':>10}{'float64':>10}{'f32+PML':>10}")
    for name, shape, mode, steps in CASES:
        results = [
            throughput(shape, mode, steps, 0, torch.float32),
            throughput(shape, mode, steps, 0, torch.float64),
            throughput(shape, mode, steps, 20, torch.float32),
        ]
        print(f"{name:<12}" + "".join(f"{value:10.1f}" for value in results))

    name, shape, mode, steps = BATCH_CASE
    print(f"\n{name}, float32 + 20-cell PML, by batch size")
    print(f"{'batch':<12}{'Mcells/s':>10}")
    for batch_size in BATCH_SIZES:
        value = throughput(shape, mode, steps // batch_size, 20, torch.float32, batch_size)
        print(f"{batch_size:<12}{value:10.1f}")


if __name__ == "__main__":
    main()
