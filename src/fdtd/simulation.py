"""Time stepping."""

from collections.abc import Sequence

import torch

from fdtd.constants import EPS0, MU0
from fdtd.fields import Fields
from fdtd.grid import Grid
from fdtd.kernels import build_stencils
from fdtd.materials import Materials, coefficient_region, expand_materials, update_coefficients
from fdtd.monitors import Monitor
from fdtd.pml import CPML, build_pml_terms
from fdtd.sources import Injector, PointSource, sources_per_element

SOURCE_CHUNK = 1024  # steps of source currents evaluated at once


class Simulation:
    """Leapfrog FDTD simulation of one grid, or of a batch of independent runs on the same grid.

    After ``n`` steps, E is known at time ``n * dt`` and H at ``(n - 1/2) * dt``.

    Parameters
    ----------
    grid : Grid
        Shared by every batch element, along with the time step.
    materials : Materials, optional
        Defaults to vacuum. See ``Materials`` for per-element values.
    sources : sequence
        ``PointSource``s shared by every batch element or, when batched, one sequence of
        ``PointSource`` per element.
    monitors : sequence of Monitor
    pml : CPML, optional
        Required when ``grid.pml_cells > 0``. Without a PML, all outer faces are PEC. Shared by
        every batch element.
    device : torch device
    dtype : torch dtype
        ``torch.float32`` or ``torch.float64``.
    batch_size : int, optional
        Number of independent runs stepped together. When given, every field carries a leading
        batch axis, and the monitors record every element. When omitted, nothing has one.
    """

    def __init__(
        self,
        grid: Grid,
        materials: Materials | None = None,
        sources: Sequence[PointSource] | Sequence[Sequence[PointSource]] = (),
        monitors: Sequence[Monitor] = (),
        pml: CPML | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        batch_size: int | None = None,
    ):
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if (pml is None) != (grid.pml_cells == 0):
            raise ValueError("A CPML config and Grid.pml_cells > 0 must be given together")

        self.grid = grid
        self.batch_size = batch_size
        self.materials = materials if materials is not None else Materials()
        self.monitors = list(monitors)
        self.fields = Fields(grid, device, dtype, batch_size)
        self.step_count = 0

        self._expanded = expand_materials(self.materials, grid, batch_size or 1, device, dtype)
        coefficients = update_coefficients(self._expanded, grid)
        self._e_decay = coefficients.e_decay
        self._lossless = not isinstance(self._e_decay, torch.Tensor) and self._e_decay == 1.0
        self._e_fields = [self.fields.tensors[name] for name in grid.components if name[0] == "E"]

        self._pml_terms = []
        self._h_updates = build_stencils(grid, self.fields, "H", coefficients.h_curl)
        self._e_updates = build_stencils(grid, self.fields, "E", coefficients.e_curl)
        if pml is not None:
            h_terms = build_pml_terms(
                grid, self.fields, pml, "H", coefficients.h_curl, device, dtype
            )
            e_terms = build_pml_terms(
                grid, self.fields, pml, "E", coefficients.e_curl, device, dtype
            )
            self._pml_terms = h_terms + e_terms
            self._h_updates += h_terms
            self._e_updates += e_terms

        self._injector = Injector(
            sources_per_element(sources, batch_size),
            grid,
            self.fields,
            coefficients.e_source,
            device,
            dtype,
        )

    @property
    def time(self) -> float:
        """Time of the E field (s)."""
        return self.step_count * self.grid.dt

    def field_time(self, component: str) -> float:
        """Time at which a component is currently known (s)."""
        return self.time - (0.5 * self.grid.dt if component[0] == "H" else 0.0)

    def field(self, component: str) -> torch.Tensor:
        """View of a component over the physical domain, with the batch axis first if batched."""
        return self.fields[component][(..., *self.grid.physical_slice)]

    def energy(self) -> float | list[float]:
        """Electromagnetic energy in the physical domain (J, or J/m in 2D), per batch element."""
        region = self.grid.physical_slice
        total = 0.0
        for name in self.grid.components:
            if name[0] == "E":
                weight = EPS0 * coefficient_region(self._expanded.eps, region)
            else:
                weight = MU0 * coefficient_region(self._expanded.mu, region)
            values = self.fields.tensors[name][(slice(None), *region)]
            total = total + (weight * values.square()).flatten(1).sum(1)
        total = 0.5 * total * self.grid.dx**self.grid.ndim
        return total.tolist() if self.fields.batched else total[0].item()

    def run(self, steps: int | None = None, duration: float | None = None) -> None:
        """Advance by a number of steps, or by a duration rounded to the nearest step."""
        if (steps is None) == (duration is None):
            raise ValueError("Pass exactly one of steps or duration")
        if steps is None:
            steps = round(duration / self.grid.dt)
        with torch.inference_mode():
            remaining = steps
            while remaining > 0:
                chunk = min(remaining, SOURCE_CHUNK)
                self._injector.tabulate(self.step_count, chunk)
                for _ in range(chunk):
                    self._step()
                remaining -= chunk

    def reset(self) -> None:
        """Return to the initial conditions and clear the monitors."""
        self.fields.zero_()
        for term in self._pml_terms:
            term.psi.zero_()
        self.step_count = 0
        for monitor in self.monitors:
            monitor.reset()

    def _step(self) -> None:
        for update in self._h_updates:
            update.apply()

        if not self._lossless:
            for e_field in self._e_fields:
                e_field.mul_(self._e_decay)
        for update in self._e_updates:
            update.apply()
        self._injector.apply()

        self.step_count += 1
        for monitor in self.monitors:
            if self.step_count % monitor.every == 0:
                monitor.record(self)
