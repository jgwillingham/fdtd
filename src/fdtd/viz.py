"""Plotting helpers for 2D simulations. Requires the ``viz`` extra."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation

from fdtd.grid import Grid
from fdtd.monitors import Snapshot


def _image(ax, frame: np.ndarray, grid: Grid, component: str, floor_db: float, peak: float):
    """Draw a frame as dB relative to ``peak``, with x horizontal."""
    x, y = (axis.numpy() for axis in grid.coords(component))
    level = 20 * np.log10(np.abs(frame) / peak + 1e-300)
    return ax.imshow(
        level.T,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="turbo",
        vmin=floor_db,
        vmax=0.0,
        animated=True,
    )


def animate_snapshot(
    snapshot: Snapshot, grid: Grid, component: str, floor_db: float = -80.0
) -> ArtistAnimation:
    """Animate a recorded component in dB relative to its overall peak."""
    frames = snapshot.frames[component]
    peak = max(np.abs(frame).max() for frame in frames)
    figure, ax = plt.subplots()
    images = [[_image(ax, frame, grid, component, floor_db, peak)] for frame in frames]
    figure.colorbar(images[0][0], label=f"|{component}| (dB)")
    ax.set(xlabel="x (m)", ylabel="y (m)", title=component)
    return ArtistAnimation(figure, images, interval=30, blit=True)


def plot_frames(
    snapshot: Snapshot, grid: Grid, component: str, indices: list[int], floor_db: float = -80.0
):
    """Plot selected frames of a recorded component side by side."""
    frames = snapshot.frames[component]
    peak = max(np.abs(frame).max() for frame in frames)
    figure, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices) + 1, 3.6), squeeze=False)
    for ax, index in zip(axes[0], indices, strict=True):
        image = _image(ax, frames[index], grid, component, floor_db, peak)
        ax.set(xlabel="x (m)", title=f"{component}, t = {snapshot.times[component][index]:.3g} s")
    axes[0][0].set_ylabel("y (m)")
    figure.colorbar(image, ax=axes[0].tolist(), label=f"|{component}| (dB)")
    return figure
