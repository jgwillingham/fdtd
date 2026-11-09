"""Time-dependent current sources."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from fdtd.fields import Fields
from fdtd.grid import Grid
from fdtd.materials import Coefficient, coefficient_at

type Waveform = Callable[[float], float]


@dataclass(frozen=True)
class ModulatedGaussian:
    """Zero-mean pulse ``sin(w(t - delay)) * exp(-((t - delay) / width)^2)``.

    The delay defaults to five widths. Any less truncates the pulse enough to leave a net charge
    behind, whose static field no boundary can absorb: 3 widths leaves it at -83 dB of the peak
    energy, 4 at -146 dB, and 5 at round-off.
    """

    frequency: float
    width: float
    delay: float | None = None

    def __call__(self, t: float) -> float:
        delay = 5 * self.width if self.delay is None else self.delay
        phase = 2 * math.pi * self.frequency * (t - delay)
        return math.sin(phase) * math.exp(-(((t - delay) / self.width) ** 2))


@dataclass(frozen=True)
class Sinusoid:
    """Continuous wave ``sin(wt)``, smoothly ramped on over ``ramp`` seconds."""

    frequency: float
    ramp: float = 0.0

    def __call__(self, t: float) -> float:
        envelope = 1.0 if t >= self.ramp else math.sin(0.5 * math.pi * t / self.ramp) ** 2
        return envelope * math.sin(2 * math.pi * self.frequency * t)


@dataclass(frozen=True)
class PointSource:
    """Current density ``amplitude * waveform(t)`` (A/m^2) in one cell of an E component.

    ``position`` is in metres from the origin of the physical domain.
    """

    component: str
    position: tuple[float, ...]
    waveform: Waveform
    amplitude: float = 1.0


def sources_per_element(sources: Sequence, batch_size: int | None) -> list[list[PointSource]]:
    """The sources of each batch element.

    ``sources`` is a sequence of ``PointSource`` shared by every element or, for a batched
    simulation, one sequence of ``PointSource`` per element.
    """
    sources = list(sources)
    if all(isinstance(s, PointSource) for s in sources):
        return [sources] * (batch_size or 1)
    if batch_size is None:
        raise ValueError("An unbatched simulation takes a flat sequence of PointSource")
    if len(sources) != batch_size or any(isinstance(s, PointSource) for s in sources):
        raise ValueError(
            f"Batched sources must be PointSources shared by every element, or {batch_size} "
            "sequences of PointSource, one per element"
        )
    return [list(element) for element in sources]


@dataclass(slots=True)
class _Injection:
    """The sources driving one component, across the whole batch."""

    field: torch.Tensor  # flat view of the batched component
    index: torch.Tensor  # flat index of each source, on the device
    column: torch.Tensor  # each source's column of the waveform table, on the CPU
    weight: torch.Tensor  # -e_source * amplitude of each source, float64 on the CPU
    values: torch.Tensor | None = None  # increments per step of the current chunk


class Injector:
    """Adds the source currents of every batch element to E, with one kernel per component.

    The CPU evaluates the waveforms ahead of time, a chunk of steps at once, so stepping needs
    neither Python work per source nor a transfer from the CPU.
    """

    def __init__(
        self,
        per_element: list[list[PointSource]],
        grid: Grid,
        fields: Fields,
        e_source: Coefficient,
        device,
        dtype,
    ):
        self._dt = grid.dt
        self._device, self._dtype = device, dtype
        self._waveforms: list[Waveform] = []
        columns: dict[int, int] = {}  # one column per distinct waveform object

        grouped: dict[str, tuple[list[int], list[int], list[float]]] = {}
        for element, sources in enumerate(per_element):
            for source in sources:
                tensor = fields.tensors[source.component]
                index = grid.nearest_index(source.component, source.position)
                flat = sum(i * s for i, s in zip((element, *index), tensor.stride(), strict=True))
                if id(source.waveform) not in columns:
                    columns[id(source.waveform)] = len(self._waveforms)
                    self._waveforms.append(source.waveform)
                scale = coefficient_at(e_source, element, index)
                indices, cols, weights = grouped.setdefault(source.component, ([], [], []))
                indices.append(flat)
                cols.append(columns[id(source.waveform)])
                weights.append(-scale * source.amplitude)

        self._injections = [
            _Injection(
                field=fields.tensors[component].view(-1),
                index=torch.tensor(indices, device=device),
                column=torch.tensor(cols),
                weight=torch.tensor(weights, dtype=torch.float64),
            )
            for component, (indices, cols, weights) in grouped.items()
        ]
        self._next = 0

    def tabulate(self, first_step: int, steps: int) -> None:
        """Evaluate the increments of the next ``steps`` steps, starting after ``first_step``."""
        if not self._injections:
            return
        # J sits between E(n) and E(n + 1)
        times = [(first_step + k + 0.5) * self._dt for k in range(steps)]
        table = torch.tensor([[w(t) for w in self._waveforms] for t in times], dtype=torch.float64)
        for injection in self._injections:
            increments = table[:, injection.column] * injection.weight
            injection.values = increments.to(self._device, self._dtype)
        self._next = 0

    def apply(self) -> None:
        """Add the next tabulated step's increments."""
        if not self._injections:
            return
        for injection in self._injections:
            injection.field.index_add_(0, injection.index, injection.values[self._next])
        self._next += 1
