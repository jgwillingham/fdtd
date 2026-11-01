"""Electromagnetic field storage."""

import torch

from fdtd.grid import Grid


class Fields:
    """Field components, stored as one tensor of ``Grid.dims`` each.

    The tensors are updated in place, so views taken of them stay valid for the whole simulation.
    ``scratch`` hands out views of one shared buffer for short-lived intermediate results.
    """

    def __init__(self, grid: Grid, device, dtype):
        self.components = {
            name: torch.zeros(grid.dims, device=device, dtype=dtype) for name in grid.components
        }
        self._buffer = torch.zeros(grid.dims, device=device, dtype=dtype)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.components[name]

    def scratch(self, shape: torch.Size) -> torch.Tensor:
        return self._buffer[tuple(slice(n) for n in shape)]

    def zero_(self) -> None:
        for tensor in self.components.values():
            tensor.zero_()
