"""Time-dependent current sources."""

import math
from collections.abc import Callable
from dataclasses import dataclass

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
