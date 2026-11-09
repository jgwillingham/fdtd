"""Observers that record simulation data while it runs."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from fdtd.simulation import Simulation


class Monitor(Protocol):
    """Called after every step for which ``step % every == 0``."""

    every: int

    def record(self, sim: "Simulation") -> None: ...

    def reset(self) -> None: ...


@dataclass
class Probe:
    """Time series of one component at one point.

    In a batched simulation, each entry of ``values`` lists every element's value, so
    ``np.array(values)`` has shape ``(len(times), batch_size)``.
    """

    component: str
    position: tuple[float, ...]
    every: int = 1
    times: list[float] = field(default_factory=list)
    values: list[float] | list[list[float]] = field(default_factory=list)

    def record(self, sim: "Simulation") -> None:
        index = sim.grid.nearest_index(self.component, self.position)
        self.times.append(sim.field_time(self.component))
        self.values.append(sim.fields[self.component][(..., *index)].tolist())

    def reset(self) -> None:
        self.times.clear()
        self.values.clear()


@dataclass
class Snapshot:
    """Copies of whole components over the physical domain.

    ``frames[name]`` is a list of arrays, sampled at ``times[name]`` on the nodes given by
    ``Grid.coords(name)``. E and H are half a time step apart. In a batched simulation, each frame
    has the batch axis first.
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

    E and H are half a step apart, so this oscillates slightly around the conserved value. In a
    batched simulation, each entry of ``values`` lists every element's energy.
    """

    every: int = 1
    times: list[float] = field(default_factory=list)
    values: list[float] | list[list[float]] = field(default_factory=list)

    def record(self, sim: "Simulation") -> None:
        self.times.append(sim.time)
        self.values.append(sim.energy())

    def reset(self) -> None:
        self.times.clear()
        self.values.clear()
