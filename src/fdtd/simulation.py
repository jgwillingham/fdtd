"""Time stepping."""

from collections.abc import Sequence

import torch

from fdtd.fields import Fields
from fdtd.grid import Grid
from fdtd.kernels import build_stencils
from fdtd.materials import Materials, update_coefficients
from fdtd.monitors import Monitor
from fdtd.pml import CPML, build_pml_terms
from fdtd.sources import PointSource


class Simulation:
    """Leapfrog FDTD simulation.

    After ``n`` steps, E is known at time ``n * dt`` and H at ``(n - 1/2) * dt``.

    Parameters
    ----------
    grid : Grid
    materials : Materials, optional
        Defaults to vacuum.
    sources : sequence of PointSource
    monitors : sequence of Monitor
    pml : CPML, optional
        Required when ``grid.pml_cells > 0``. Without a PML, all outer faces are PEC.
    device : torch device
    dtype : torch dtype
        ``torch.float32`` or ``torch.float64``.
    """

    def __init__(
        self,
        grid: Grid,
        materials: Materials | None = None,
        sources: Sequence[PointSource] = (),
        monitors: Sequence[Monitor] = (),
        pml: CPML | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.grid = grid
        self.materials = materials if materials is not None else Materials()
        self.monitors = list(monitors)
        self.fields = Fields(grid, device, dtype)
        self.step_count = 0

        if (pml is None) != (grid.pml_cells == 0):
            raise ValueError("A CPML config and Grid.pml_cells > 0 must be given together")

        coefficients = update_coefficients(self.materials, grid, device, dtype)
        self._e_decay = coefficients.e_decay
        self._e_source = coefficients.e_source
        self._lossless = not isinstance(self._e_decay, torch.Tensor) and self._e_decay == 1.0
        self._e_fields = [self.fields[name] for name in grid.components if name[0] == "E"]

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

        self._sources = []
        for s in sources:
            index = grid.nearest_index(s.component, s.position)
            scale = self._e_source
            if isinstance(scale, torch.Tensor):
                scale = scale[index].item()  # read once here, not every step
            self._sources.append((s, self.fields[s.component], index, scale))

    @property
    def time(self) -> float:
        """Time of the E field (s)."""
        return self.step_count * self.grid.dt

    def field_time(self, component: str) -> float:
        """Time at which a component is currently known (s)."""
        return self.time - (0.5 * self.grid.dt if component[0] == "H" else 0.0)

    def field(self, component: str) -> torch.Tensor:
        """View of a component over the physical domain."""
        return self.fields[component][self.grid.physical_slice]

    def run(self, steps: int | None = None, duration: float | None = None) -> None:
        """Advance by a number of steps, or by a duration rounded to the nearest step."""
        if (steps is None) == (duration is None):
            raise ValueError("Pass exactly one of steps or duration")
        if steps is None:
            steps = round(duration / self.grid.dt)
        with torch.inference_mode():
            for _ in range(steps):
                self._step()

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
        self._apply_sources()

        self.step_count += 1
        for monitor in self.monitors:
            if self.step_count % monitor.every == 0:
                monitor.record(self)

    def _apply_sources(self) -> None:
        t = (self.step_count + 0.5) * self.grid.dt  # J sits between E(n) and E(n + 1)
        for source, field, index, scale in self._sources:
            field[index] -= scale * source.amplitude * source.waveform(t)
