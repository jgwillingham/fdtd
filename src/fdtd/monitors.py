"""Observers that record simulation data while it runs.

Each monitor defines what to measure through ``sample``, and keeps what it measures in memory
through ``record``. ``fdtd.io.HDF5Writer`` streams the same samples to a file instead.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

if TYPE_CHECKING:
    from fdtd.simulation import Simulation


class Sample(NamedTuple):
    """One measurement: ``values`` of quantity ``key`` at ``time`` (s).

    ``values`` has the batch axis first in a batched simulation, and may share memory with the
    simulation's fields, so copy it before the next step if it must be kept.
    """

    key: str
    time: float
    values: np.ndarray


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

    def sample(self, sim: "Simulation") -> list[Sample]:
        index = sim.grid.nearest_index(self.component, self.position)
        value = sim.fields[self.component][(..., *index)].cpu().numpy()
        return [Sample(self.component, sim.field_time(self.component), value)]

    def record(self, sim: "Simulation") -> None:
        for _, time, values in self.sample(sim):
            self.times.append(time)
            self.values.append(values.tolist())

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

    def sample(self, sim: "Simulation") -> list[Sample]:
        return [
            Sample(name, sim.field_time(name), sim.field(name).cpu().numpy())
            for name in self.components
        ]

    def record(self, sim: "Simulation") -> None:
        for name, time, values in self.sample(sim):
            self.times.setdefault(name, []).append(time)
            self.frames.setdefault(name, []).append(values.copy())

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

    def sample(self, sim: "Simulation") -> list[Sample]:
        return [Sample("energy", sim.time, np.asarray(sim.energy()))]

    def record(self, sim: "Simulation") -> None:
        for _, time, values in self.sample(sim):
            self.times.append(time)
            self.values.append(values.tolist())

    def reset(self) -> None:
        self.times.clear()
        self.values.clear()
