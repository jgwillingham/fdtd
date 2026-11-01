"""Convolutional perfectly matched layer (Roden & Gedney, 2000).

Inside the layer each derivative is stretched by ``s = kappa + sigma / (alpha + jw)``, with sigma
and alpha in 1/s. In the time domain this becomes ``d/du -> (1 / kappa) d/du + psi``, where psi
is a recursive convolution of the derivative.

The main stencils have already applied ``d/du`` everywhere, so a ``PMLTerm`` only has to add the
remainder ``(1 / kappa - 1) d/du + psi`` over its slab. There is one term per curl term per face.
Each term samples the profiles at its own target nodes, which differ by half a cell between the
E and H updates.
"""

import math
from dataclasses import dataclass

import torch

from fdtd.constants import EPS0, ETA0
from fdtd.fields import Fields
from fdtd.grid import CurlTerm, Grid
from fdtd.kernels import derivative_range, signed_coefficient, term_slices
from fdtd.materials import Coefficient


@dataclass(frozen=True)
class CPML:
    """CPML grading. The thickness is ``Grid.pml_cells``.

    With ``d`` running from 0 at the inner edge to 1 at the outer wall:
    ``sigma = sigma_max * d^grading``, ``kappa = 1 + (kappa_max - 1) * d^grading``, and
    ``alpha = alpha_scale * 2 pi reference_frequency * (1 - d)``.

    Parameters
    ----------
    reference_frequency : float
        Typically the lowest frequency of interest (Hz). There is deliberately no default: alpha
        must stay small against the simulated frequencies, or the layer stops absorbing.
    grading : int
        Polynomial order of the profiles.
    sigma_scale : float
        Multiple of the optimal ``sigma_max = 0.8 (grading + 1) / (eta0 dx)``.
    kappa_max : float
    alpha_scale : float
    """

    reference_frequency: float
    grading: int = 3
    sigma_scale: float = 1.0
    kappa_max: float = 5.0
    alpha_scale: float = 0.1

    def profiles(self, depth: torch.Tensor, dx: float):
        """``(sigma, kappa, alpha)`` at relative depths ``d`` into the layer."""
        sigma_max = self.sigma_scale * 0.8 * (self.grading + 1) / (ETA0 * dx) / EPS0
        alpha_max = self.alpha_scale * 2 * math.pi * self.reference_frequency
        graded = depth**self.grading
        return sigma_max * graded, 1 + (self.kappa_max - 1) * graded, alpha_max * (1 - depth)


@dataclass(slots=True)
class PMLTerm:
    """Applies the stretched-coordinate remainder of one curl term over one slab."""

    target: torch.Tensor
    ahead: torch.Tensor
    behind: torch.Tensor
    coefficient: Coefficient
    decay: torch.Tensor
    gain: torch.Tensor
    inverse_kappa_excess: torch.Tensor
    psi: torch.Tensor
    scratch: torch.Tensor

    def apply(self) -> None:
        derivative = torch.sub(self.ahead, self.behind, out=self.scratch)
        self.psi.mul_(self.decay).addcmul_(derivative, self.gain)
        derivative.mul_(self.inverse_kappa_excess).add_(self.psi).mul_(self.coefficient)
        self.target.add_(derivative)


def _slab_nodes(grid: Grid, term: CurlTerm) -> tuple[list[int], list[int]]:
    """Target nodes along ``term.axis`` inside the low and high layers."""
    offset = grid.offsets(term.target)[term.axis]
    inner_low = grid.pml_cells
    inner_high = grid.dims[term.axis] - 1 - grid.pml_cells
    nodes = derivative_range(grid, term)
    low = [i for i in nodes if i + offset < inner_low]
    high = [i for i in nodes if i + offset > inner_high]
    return low, high


def _build_term(
    grid: Grid, fields: Fields, config: CPML, term: CurlTerm, nodes: list[int], curl, device, dtype
) -> PMLTerm:
    target, ahead, behind = term_slices(grid, term, range(nodes[0], nodes[-1] + 1))
    view = fields[term.target][target]

    offset = grid.offsets(term.target)[term.axis]
    position = torch.tensor(nodes, device=device, dtype=torch.float64) + offset
    inner_high = grid.dims[term.axis] - 1 - grid.pml_cells
    outside = torch.maximum(grid.pml_cells - position, position - inner_high)
    sigma, kappa, alpha = config.profiles(outside / grid.pml_cells, grid.dx)

    decay = torch.exp(-(sigma / kappa + alpha) * grid.dt)
    gain = sigma * (decay - 1) / (kappa * (sigma + kappa * alpha))

    def as_field(profile: torch.Tensor) -> torch.Tensor:
        return profile.to(dtype).reshape([-1 if a == term.axis else 1 for a in range(grid.ndim)])

    return PMLTerm(
        target=view,
        ahead=fields[term.source][ahead],
        behind=fields[term.source][behind],
        coefficient=signed_coefficient(curl, term.sign, target),
        decay=as_field(decay),
        gain=as_field(gain),
        inverse_kappa_excess=as_field(1 / kappa - 1),
        psi=torch.zeros_like(view),
        scratch=fields.scratch(view.shape),
    )


def build_pml_terms(
    grid: Grid, fields: Fields, config: CPML, field: str, curl: Coefficient, device, dtype
) -> list[PMLTerm]:
    """PML terms for the ``"E"`` or ``"H"`` update, with ``curl`` scaling the curl."""
    terms = []
    for term in grid.curl_terms(field):
        for nodes in _slab_nodes(grid, term):
            terms.append(_build_term(grid, fields, config, term, nodes, curl, device, dtype))
    return terms
