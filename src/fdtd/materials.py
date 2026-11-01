"""Material parameters and the update coefficients derived from them."""

from dataclasses import dataclass

import torch

from fdtd.constants import EPS0, MU0
from fdtd.grid import Grid

type Coefficient = float | torch.Tensor


@dataclass(frozen=True)
class Materials:
    """Relative permittivity, relative permeability, and conductivity (S/m).

    Each is a scalar or a tensor of ``Grid.shape``. Tensors are sampled at the same index for every
    component (staircased), and their edge values extend through the PML.
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


def _expand(value, grid: Grid, device, dtype) -> Coefficient:
    """Convert a material parameter to a float, or to a tensor of ``Grid.dims``."""
    if isinstance(value, int | float):
        return float(value)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tuple(tensor.shape) != grid.shape:
        raise ValueError(f"Material shape {tuple(tensor.shape)} must match grid {grid.shape}")
    for axis, n in enumerate(grid.shape):
        physical_index = torch.arange(n + 2 * grid.pml_cells, device=device) - grid.pml_cells
        tensor = tensor.index_select(axis, physical_index.clamp(0, n - 1))
    return tensor


def update_coefficients(materials: Materials, grid: Grid, device, dtype) -> UpdateCoefficients:
    eps = _expand(materials.eps, grid, device, dtype)
    mu = _expand(materials.mu, grid, device, dtype)
    sigma = _expand(materials.sigma, grid, device, dtype)

    loss = sigma * grid.dt / (2 * EPS0 * eps)
    e_source = grid.dt / (EPS0 * eps) / (1 + loss)
    return UpdateCoefficients(
        e_decay=(1 - loss) / (1 + loss),
        e_curl=e_source / grid.dx,
        h_curl=grid.dt / (MU0 * mu * grid.dx),
        e_source=e_source,
    )
