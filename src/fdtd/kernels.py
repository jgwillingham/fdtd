"""Yee update stencils.

Each ``CurlTerm`` of a grid becomes one ``Stencil``, whose slices follow from the stagger offsets
alone, so TE, TM, and 3D share the same code. Views are built once, which keeps the stepping loop
free of slicing and allocation.
"""

from dataclasses import dataclass

import torch

from fdtd.fields import Fields
from fdtd.grid import CurlTerm, Grid
from fdtd.materials import Coefficient


@dataclass(slots=True)
class Stencil:
    """Applies ``target += coefficient * (ahead - behind)`` in place."""

    target: torch.Tensor
    ahead: torch.Tensor
    behind: torch.Tensor
    coefficient: Coefficient
    scratch: torch.Tensor

    def apply(self) -> None:
        torch.sub(self.ahead, self.behind, out=self.scratch)
        self.scratch.mul_(self.coefficient)
        self.target.add_(self.scratch)


def derivative_range(grid: Grid, term: CurlTerm) -> range:
    """Target nodes along ``term.axis`` that have a difference of ``term.source``."""
    n = grid.dims[term.axis]
    return range(0, n - 1) if term.target[0] == "H" else range(1, n - 1)


def term_slices(
    grid: Grid, term: CurlTerm, nodes: range | None = None
) -> tuple[tuple[slice, ...], tuple[slice, ...], tuple[slice, ...]]:
    """Slices of the target and of the two source nodes around it.

    ``nodes`` restricts the target along ``term.axis`` (default: every node with a difference).
    H sits half a cell ahead of the E node of the same index, so its difference is forward, and E
    half a cell ahead of the H node one index below, so its difference is backward. Transverse to
    the derivative, E skips its PEC walls.
    """
    nodes = nodes if nodes is not None else derivative_range(grid, term)
    source_shift = 0 if term.target[0] == "H" else -1
    target, ahead, behind = [], [], []
    for axis, offset in enumerate(grid.offsets(term.target)):
        if axis == term.axis:
            first = nodes.start + source_shift
            last = nodes.stop + source_shift
            target.append(slice(nodes.start, nodes.stop))
            ahead.append(slice(first + 1, last + 1))
            behind.append(slice(first, last))
        else:
            if offset:
                transverse = slice(None, -1)
            else:
                transverse = slice(None) if term.target[0] == "H" else slice(1, -1)
            target.append(transverse)
            ahead.append(transverse)
            behind.append(transverse)
    return tuple(target), tuple(ahead), tuple(behind)


def signed_coefficient(coefficient: Coefficient, sign: float, region) -> Coefficient:
    """The coefficient over ``region``, or the scalar itself if it is uniform."""
    if isinstance(coefficient, torch.Tensor):
        return sign * coefficient[region]
    return sign * coefficient


def build_stencils(
    grid: Grid, fields: Fields, field: str, coefficient: Coefficient
) -> list[Stencil]:
    """Stencils for the ``"E"`` or ``"H"`` update, with ``coefficient`` scaling the curl."""
    stencils = []
    for term in grid.curl_terms(field):
        target, ahead, behind = term_slices(grid, term)
        view = fields[term.target][target]
        stencils.append(
            Stencil(
                target=view,
                ahead=fields[term.source][ahead],
                behind=fields[term.source][behind],
                coefficient=signed_coefficient(coefficient, term.sign, target),
                scratch=fields.scratch(view.shape),
            )
        )
    return stencils
