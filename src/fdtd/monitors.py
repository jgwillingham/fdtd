"""Observers that record simulation data while it runs."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from fdtd.constants import EPS0, MU0

if TYPE_CHECKING:
    from fdtd.simulation import Simulation


class Monitor(Protocol):
    """Called after every step for which ``step % every == 0``."""

    every: int

    def record(self, sim: "Simulation") -> None: ...

    def reset(self) -> None: ...


@dataclass
class Probe:
    """Time series of one component at one point."""

    component: str
    position: tuple[float, ...]
    every: int = 1
    times: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def record(self, sim: "Simulation") -> None:
        index = sim.grid.nearest_index(self.component, self.position)
        self.times.append(sim.field_time(self.component))
        self.values.append(sim.fields[self.component][index].item())

    def reset(self) -> None:
        self.times.clear()
        self.values.clear()


@dataclass
class Snapshot:
    """Copies of whole components over the physical domain.

    ``frames[name]`` is a list of arrays, sampled at ``times[name]`` on the nodes given by
    ``Grid.coords(name)``. E and H are half a time step apart.
    """

    components: tuple[str, ...]
    every: int = 1
    times: dict[str, list[float]] = field(default_factory=dict)
    frames: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def record(self, sim: "Simulation") -> None:
        for name in self.components:
            self.times.setdefault(name, []).append(sim.field_time(name))
            self.frames.setdefault(name, []).append(sim.field(name).cpu().numpy().copy())

    def reset(self) -> None:
        self.times.clear()
        self.frames.clear()


@dataclass
class Energy:
    """Electromagnetic energy in the physical domain (J, or J/m in 2D).

    E and H are half a step apart, so this oscillates slightly around the conserved value.
    """

    every: int = 1
    times: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def record(self, sim: "Simulation") -> None:
        def weighted_sum(prefix: str, weight) -> float:
            names = (name for name in sim.grid.components if name[0] == prefix)
            return sum((weight * sim.field(name).square()).sum().item() for name in names)

        electric = EPS0 * weighted_sum("E", sim.materials.eps)
        magnetic = MU0 * weighted_sum("H", sim.materials.mu)
        self.times.append(sim.time)
        self.values.append(0.5 * (electric + magnetic) * sim.grid.dx**sim.grid.ndim)

    def reset(self) -> None:
        self.times.clear()
        self.values.clear()
