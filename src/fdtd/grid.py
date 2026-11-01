"""Yee grid geometry. This module is the single source of truth for staggering.

Every component has one array of ``Grid.dims`` nodes, and node ``i`` of a component sits at
``(i + offset) * dx`` along each axis, with the offsets in ``OFFSETS``. H is half a cell ahead of
E along the axes of its curl, so nodes at index ``N - 1`` of an offset axis lie outside the domain
and stay unused. Tangential E is zero on every outer face (PEC).
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch

from fdtd.constants import C0

OFFSETS = {
    "Ex": (0.5, 0.0, 0.0),
    "Ey": (0.0, 0.5, 0.0),
    "Ez": (0.0, 0.0, 0.5),
    "Hx": (0.0, 0.5, 0.5),
    "Hy": (0.5, 0.0, 0.5),
    "Hz": (0.5, 0.5, 0.0),
}


class Mode(Enum):
    """Field polarisation. TE and TM are the 2D modes, uniform along z."""

    TE = ("Ex", "Ey", "Hz")
    TM = ("Ez", "Hx", "Hy")
    FULL_3D = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class CurlTerm(NamedTuple):
    """One term of the curl equations: ``d(target)/dt ~ sign * d(source)/d(axis)``."""

    target: str
    source: str
    axis: int
    sign: float


_CURL_TERMS = (
    CurlTerm("Hx", "Ez", 1, -1.0),
    CurlTerm("Hx", "Ey", 2, +1.0),
    CurlTerm("Hy", "Ex", 2, -1.0),
    CurlTerm("Hy", "Ez", 0, +1.0),
    CurlTerm("Hz", "Ey", 0, -1.0),
    CurlTerm("Hz", "Ex", 1, +1.0),
    CurlTerm("Ex", "Hz", 1, +1.0),
    CurlTerm("Ex", "Hy", 2, -1.0),
    CurlTerm("Ey", "Hx", 2, +1.0),
    CurlTerm("Ey", "Hz", 0, -1.0),
    CurlTerm("Ez", "Hy", 0, +1.0),
    CurlTerm("Ez", "Hx", 1, -1.0),
)


@dataclass(frozen=True)
class Grid:
    """Uniform Yee grid.

    Parameters
    ----------
    shape : tuple of int
        Nodes of the physical domain along each axis (2 for TE/TM, 3 for 3D).
    dx : float
        Lattice spacing (m).
    mode : Mode
        Polarisation. Must be TE or TM for a 2D shape.
    courant : float, optional
        Courant number. Defaults to 99% of the stability limit.
    pml_cells : int
        CPML thickness in cells, added outside the physical domain on every face.
    """

    shape: tuple[int, ...]
    dx: float
    mode: Mode = Mode.FULL_3D
    courant: float | None = None
    pml_cells: int = 0

    def __post_init__(self):
        expected = 3 if self.mode is Mode.FULL_3D else 2
        if len(self.shape) != expected:
            raise ValueError(f"{self.mode.name} needs a {expected}D shape, got {self.shape}")
        if min(self.shape) < 2:
            raise ValueError(f"Every axis needs at least 2 nodes, got {self.shape}")
        if self.dx <= 0:
            raise ValueError(f"dx must be positive, got {self.dx}")
        if self.pml_cells < 0:
            raise ValueError(f"pml_cells must be non-negative, got {self.pml_cells}")

        limit = 1.0 / math.sqrt(self.ndim)
        if self.courant is None:
            object.__setattr__(self, "courant", 0.99 * limit)
        elif not 0 < self.courant <= limit:
            raise ValueError(f"Unstable Courant number {self.courant}: max is {limit:.4f}")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dims(self) -> tuple[int, ...]:
        """Total nodes along each axis, including the PML."""
        return tuple(n + 2 * self.pml_cells for n in self.shape)

    @property
    def dt(self) -> float:
        return self.courant * self.dx / C0

    @property
    def components(self) -> tuple[str, ...]:
        return self.mode.value

    @property
    def physical_slice(self) -> tuple[slice, ...]:
        """Selects the physical domain out of a component array."""
        return tuple(slice(self.pml_cells, self.pml_cells + n) for n in self.shape)

    def offsets(self, component: str) -> tuple[float, ...]:
        """Stagger of a component in cells, along the active axes."""
        return OFFSETS[component][: self.ndim]

    def coords(self, component: str) -> tuple[torch.Tensor, ...]:
        """Physical-domain coordinates (m) of a component's nodes, one 1D tensor per axis."""
        return tuple(
            (torch.arange(n, dtype=torch.float64) + offset) * self.dx
            for n, offset in zip(self.shape, self.offsets(component), strict=True)
        )

    def nearest_index(self, component: str, position: tuple[float, ...]) -> tuple[int, ...]:
        """Array index, including the PML, of the component node nearest to ``position`` (m)."""
        if len(position) != self.ndim:
            raise ValueError(f"Expected a {self.ndim}D position, got {position}")
        index = []
        for x, offset, n in zip(position, self.offsets(component), self.shape, strict=True):
            i = round(x / self.dx - offset)
            if not 0 <= i < n:
                raise ValueError(f"Position {position} is outside the physical domain")
            index.append(i + self.pml_cells)
        return tuple(index)

    def curl_terms(self, field: str) -> tuple[CurlTerm, ...]:
        """Curl terms updating the ``"E"`` or ``"H"`` components of this grid's mode."""
        return tuple(
            term
            for term in _CURL_TERMS
            if term.target[0] == field
            and term.axis < self.ndim
            and term.target in self.components
            and term.source in self.components
        )
