"""Electromagnetic field storage."""

import torch

from fdtd.grid import Grid


class Fields:
    """Field components, stored as one tensor of ``(batch_size, *Grid.dims)`` each.

    An unbatched simulation stores a batch of one, and indexing hides that axis, so
    ``fields["Ez"]`` has shape ``Grid.dims``. A batched simulation's components keep the batch
    axis first. ``tensors`` always holds the full batched storage.

    The tensors are updated in place, so views taken of them stay valid for the whole simulation.
    ``scratch`` hands out views of one shared buffer for short-lived intermediate results.
    """

    def __init__(self, grid: Grid, device, dtype, batch_size: int | None = None):
        self.batched = batch_size is not None
        shape = (batch_size or 1, *grid.dims)
        self.tensors = {
            name: torch.zeros(shape, device=device, dtype=dtype) for name in grid.components
        }
        self._buffer = torch.zeros(shape, device=device, dtype=dtype)

    def __getitem__(self, name: str) -> torch.Tensor:
        tensor = self.tensors[name]
        return tensor if self.batched else tensor[0]

    def scratch(self, shape: torch.Size) -> torch.Tensor:
        return self._buffer[tuple(slice(n) for n in shape)]

    def zero_(self) -> None:
        for tensor in self.tensors.values():
            tensor.zero_()
