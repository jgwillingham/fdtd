"""Material parameters and the update coefficients derived from them.

A ``Coefficient`` is either a float, uniform over every batch element and node, or a tensor with
one leading batch axis and one axis per grid axis. Each axis has its full length or length 1, the
latter broadcasting, so a uniform medium per batch element costs one value per element.
"""

from dataclasses import dataclass

import torch

from fdtd.constants import EPS0, MU0
from fdtd.grid import Grid

type Coefficient = float | torch.Tensor


@dataclass(frozen=True)
class Materials:
    """Relative permittivity, relative permeability, and conductivity (S/m).

    Each is one of:

    - a scalar, shared by every node and batch element;
    - a tensor of ``Grid.shape``, shared by every batch element;
    - a tensor of shape ``(batch_size,)``, uniform within each batch element;
    - a tensor of shape ``(batch_size, *Grid.shape)``.

    Tensors are sampled at the same index for every component (staircased), and their edge values
    extend through the PML.
    """

    eps: Coefficient = 1.0
    mu: Coefficient = 1.0
    sigma: Coefficient = 0.0


@dataclass(frozen=True)
class UpdateCoefficients:
    """``E <- e_decay * E + e_curl * curl(H)`` and ``H <- H - h_curl * curl(E)``."""

    e_decay: Coefficient
    e_curl: Coefficient
    h_curl: Coefficient
    e_source: Coefficient  # scales a current density (A/m^2) into an E increment


def _expand(value, grid: Grid, batch_size: int, device, dtype) -> Coefficient:
    """Convert a material parameter to a float, or to a coefficient tensor over ``Grid.dims``."""
    if isinstance(value, int | float):
        return float(value)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    shape = tuple(tensor.shape)
    if shape == grid.shape:
        tensor = tensor.unsqueeze(0)
    elif shape == (batch_size,):
        tensor = tensor.reshape(batch_size, *(1,) * grid.ndim)
    elif shape != (batch_size, *grid.shape):
        raise ValueError(
            f"Material shape {shape} must be {grid.shape}, ({batch_size},), "
            f"or ({batch_size}, *{grid.shape})"
        )
    for axis, n in enumerate(grid.shape):
        if tensor.shape[axis + 1] == 1:
            continue
        physical_index = torch.arange(n + 2 * grid.pml_cells, device=device) - grid.pml_cells
        tensor = tensor.index_select(axis + 1, physical_index.clamp(0, n - 1))
    return tensor


def expand_materials(materials: Materials, grid: Grid, batch_size: int, device, dtype) -> Materials:
    """The materials as coefficients over ``Grid.dims``, including the PML."""
    return Materials(
        eps=_expand(materials.eps, grid, batch_size, device, dtype),
        mu=_expand(materials.mu, grid, batch_size, device, dtype),
        sigma=_expand(materials.sigma, grid, batch_size, device, dtype),
    )


def coefficient_region(coefficient: Coefficient, region: tuple[slice, ...]) -> Coefficient:
    """A coefficient over the spatial ``region`` of every batch element.

    Axes of length 1 broadcast, so they are left whole.
    """
    if not isinstance(coefficient, torch.Tensor):
        return coefficient
    spatial = tuple(
        part if n > 1 else slice(None)
        for part, n in zip(region, coefficient.shape[1:], strict=True)
    )
    return coefficient[(slice(None), *spatial)]


def coefficient_at(coefficient: Coefficient, element: int, index: tuple[int, ...]) -> float:
    """A coefficient's value at one node of one batch element."""
    if not isinstance(coefficient, torch.Tensor):
        return coefficient
    full_index = (element, *index)
    return coefficient[
        tuple(i if n > 1 else 0 for i, n in zip(full_index, coefficient.shape, strict=True))
    ].item()


def update_coefficients(expanded: Materials, grid: Grid) -> UpdateCoefficients:
    """Update coefficients from materials already passed through ``expand_materials``."""
    eps, mu, sigma = expanded.eps, expanded.mu, expanded.sigma
    loss = sigma * grid.dt / (2 * EPS0 * eps)
    e_source = grid.dt / (EPS0 * eps) / (1 + loss)
    return UpdateCoefficients(
        e_decay=(1 - loss) / (1 + loss),
        e_curl=e_source / grid.dx,
        h_curl=grid.dt / (MU0 * mu * grid.dx),
        e_source=e_source,
    )
