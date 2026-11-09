"""HDF5 output. Requires the ``hdf5`` extra.

``HDF5Writer`` streams monitor samples to disk as the simulation runs, next to everything needed to
interpret and reproduce them. The monitors passed to it choose what is saved; they are used only as
specifications, so their in-memory lists stay empty. Layout, with ``[B]`` present only when
batched and ``T`` the number of samples:

    /                       attrs: format, format_version, fdtd_version, torch_version, created
    /grid                   attrs: mode, components, shape, dims, dx, dt, courant, pml_cells
    /simulation             attrs: batched, batch_size, dtype, device, first_step, steps, time
    /materials/{eps,mu,sigma}   as given to Materials: a scalar, or an array
    /pml                    attrs: the CPML settings (absent without a PML)
    /sources/{element,component,position,index,amplitude,waveform}
                            one row per source; waveform is JSON
    /outputs/<name>         attrs: monitor, every (and the probe's position and node)
    /outputs/<name>/<key>/values    (T, [B], ...) samples; attrs: units
    /outputs/<name>/<key>/time      (T,) seconds
    /outputs/<name>/<key>/step      (T,) step counts
    /outputs/<name>/<key>/{x,y,z}   node coordinates of a snapshot component (m)
"""

import dataclasses
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import torch

from fdtd.monitors import Energy, Monitor, Probe, Sample, Snapshot
from fdtd.simulation import Simulation

FORMAT_VERSION = 1
CHUNK_BYTES = 1 << 20  # target size of one HDF5 chunk


def _units(key: str, ndim: int) -> str:
    if key == "energy":
        return "J" if ndim == 3 else "J/m"
    return {"E": "V/m", "H": "A/m"}.get(key[0], "")


def _waveform_json(waveform) -> str:
    """A waveform's type and parameters, or its repr if it is not a dataclass."""
    description = {"type": type(waveform).__qualname__}
    if dataclasses.is_dataclass(waveform):
        description.update(dataclasses.asdict(waveform))
    else:
        description["repr"] = repr(waveform)
    return json.dumps(description)


def _set_attrs(item: h5py.HLObject, **values) -> None:
    """Set attributes, storing strings and lists of strings as HDF5 variable-length strings."""
    for name, value in values.items():
        if isinstance(value, str):
            value = str(value)  # plain str, not a subclass such as torch's version string
        elif isinstance(value, list | tuple) and value and isinstance(value[0], str):
            value = np.array([str(v) for v in value], dtype=h5py.string_dtype())
        item.attrs[name] = value


def _as_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class HDF5Writer:
    """Saves monitor samples and the simulation's metadata to an HDF5 file.

    Small samples, such as probe values, are buffered and written a chunk at a time, so call
    ``flush`` before reading the file while the writer is open. Use it as a context manager, which
    flushes, closes the file, and detaches the writer on exit::

        outputs = {"fields": Snapshot(("Ez",), every=10), "receiver": Probe("Ez", (x, y))}
        with HDF5Writer("run.h5", sim, outputs):
            sim.run(2000)

    Parameters
    ----------
    path : str or Path
    sim : Simulation
        The writer adds itself to ``sim.monitors``.
    outputs : mapping or sequence of monitors
        What to save, by group name. A sequence is named ``<type>_<n>``, e.g. ``probe_0``. Any
        monitor with a ``sample`` method works; ``Probe``, ``Snapshot``, and ``Energy`` also save
        their positions or coordinates.
    dtype : numpy dtype, optional
        Store samples in this type, e.g. ``np.float32`` to halve a float64 run's file. Defaults to
        the simulation's dtype.
    compression : str, optional
        HDF5 filter for the samples: ``"gzip"`` (portable) or ``"lzf"`` (fast, h5py only).
    overwrite : bool
        Replace an existing file instead of raising ``FileExistsError``.
    """

    every = 1

    def __init__(
        self,
        path: str | Path,
        sim: Simulation,
        outputs: Mapping[str, Monitor] | Sequence[Monitor],
        *,
        dtype=None,
        compression: str | None = None,
        overwrite: bool = False,
    ):
        if not isinstance(outputs, Mapping):
            counts: dict[str, int] = {}
            named = {}
            for monitor in outputs:
                kind = type(monitor).__name__.lower()
                named[f"{kind}_{counts.get(kind, 0)}"] = monitor
                counts[kind] = counts.get(kind, 0) + 1
            outputs = named
        for name, monitor in outputs.items():
            if not hasattr(monitor, "sample"):
                raise TypeError(f"Output {name!r} has no sample method")

        self._sim = sim
        self._outputs = dict(outputs)
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self._compression = compression
        self._pending: dict[str, tuple[list, list, list]] = {}  # samples not yet written
        self._file = h5py.File(path, "w" if overwrite else "w-")
        self._write_metadata()
        sim.monitors.append(self)

    def __enter__(self) -> "HDF5Writer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def file(self) -> h5py.File:
        return self._file

    def record(self, sim: Simulation) -> None:
        if sim is not self._sim:
            raise ValueError("This writer belongs to another simulation")
        for name, monitor in self._outputs.items():
            if sim.step_count % monitor.every == 0:
                for sample in monitor.sample(sim):
                    self._append(name, sample)

    def reset(self) -> None:
        """Discard every saved sample, following ``Simulation.reset``."""

        def truncate(_, item):
            if isinstance(item, h5py.Dataset) and item.maxshape[0] is None:
                item.resize(0, axis=0)

        self._pending.clear()
        self._file["outputs"].visititems(truncate)
        self._file["simulation"].attrs["first_step"] = self._sim.step_count
        self.flush()

    def flush(self) -> None:
        """Write buffered samples, record how far the simulation has run, and flush to disk."""
        for series_name in list(self._pending):
            self._write_pending(series_name)
        attrs = self._file["simulation"].attrs
        attrs["steps"] = self._sim.step_count
        attrs["time"] = self._sim.time
        self._file.flush()

    def close(self) -> None:
        """Finish the file, and stop recording."""
        if not self._file:
            return
        self.flush()
        self._file.close()
        if self in self._sim.monitors:
            self._sim.monitors.remove(self)

    def _write_metadata(self) -> None:
        sim, grid, f = self._sim, self._sim.grid, self._file
        _set_attrs(
            f,
            format="fdtd",
            format_version=FORMAT_VERSION,
            fdtd_version=version("fdtd"),
            torch_version=torch.__version__,
            created=datetime.now(UTC).isoformat(),
        )
        _set_attrs(
            f.create_group("grid"),
            mode=grid.mode.name,
            components=list(grid.components),
            shape=grid.shape,
            dims=grid.dims,
            dx=grid.dx,
            dt=grid.dt,
            courant=grid.courant,
            pml_cells=grid.pml_cells,
        )
        _set_attrs(
            f.create_group("simulation"),
            batched=sim.batch_size is not None,
            batch_size=sim.batch_size or 1,
            dtype=str(sim.dtype).removeprefix("torch."),
            device=str(sim.device),
            first_step=sim.step_count,
            steps=sim.step_count,
            time=sim.time,
        )

        materials = f.create_group("materials")
        for name in ("eps", "mu", "sigma"):
            data = _as_array(getattr(sim.materials, name))
            options = {"compression": self._compression} if data.ndim else {}
            materials.create_dataset(name, data=data, **options)
        _set_attrs(materials, eps="relative", mu="relative", sigma="S/m")

        if sim.pml is not None:
            _set_attrs(f.create_group("pml"), **dataclasses.asdict(sim.pml))

        self._write_sources(f.create_group("sources"))
        f.create_group("outputs")
        for name in self._outputs:  # so the file records its configuration even if never sampled
            self._output_group(name)

    def _write_sources(self, group: h5py.Group) -> None:
        grid = self._sim.grid
        rows = [
            (element, source)
            for element, sources in enumerate(self._sim.element_sources)
            for source in sources
        ]
        strings = h5py.string_dtype()
        group.create_dataset("element", data=np.array([e for e, _ in rows], dtype=np.int64))
        group.create_dataset(
            "component", data=np.array([s.component for _, s in rows], dtype=object), dtype=strings
        )
        group.create_dataset(
            "position",
            data=np.array([s.position for _, s in rows], dtype=np.float64).reshape(-1, grid.ndim),
        )
        group.create_dataset(
            "index",
            data=np.array(
                [grid.nearest_index(s.component, s.position) for _, s in rows], dtype=np.int64
            ).reshape(-1, grid.ndim),
        )
        group.create_dataset(
            "amplitude", data=np.array([s.amplitude for _, s in rows], dtype=np.float64)
        )
        group.create_dataset(
            "waveform",
            data=np.array([_waveform_json(s.waveform) for _, s in rows], dtype=object),
            dtype=strings,
        )
        _set_attrs(group, position="m, from the physical domain's origin", amplitude="A/m^2")

    def _output_group(self, name: str) -> h5py.Group:
        outputs = self._file["outputs"]
        if name in outputs:
            return outputs[name]
        group = outputs.create_group(name)
        monitor = self._outputs[name]
        _set_attrs(group, monitor=type(monitor).__name__, every=monitor.every)
        if isinstance(monitor, Probe):
            grid = self._sim.grid
            index = grid.nearest_index(monitor.component, monitor.position)
            node = [
                (i - grid.pml_cells + offset) * grid.dx
                for i, offset in zip(index, grid.offsets(monitor.component), strict=True)
            ]
            _set_attrs(
                group,
                component=monitor.component,
                position=monitor.position,
                node=node,
                index=index,
            )
        elif isinstance(monitor, Snapshot):
            _set_attrs(group, components=list(monitor.components))
        elif isinstance(monitor, Energy):
            _set_attrs(group, region="physical domain")
        return group

    def _append(self, name: str, sample: Sample) -> None:
        """Buffer a sample, and write the buffer once it fills one chunk."""
        group = self._output_group(name)
        values = sample.values if self._dtype is None else sample.values.astype(self._dtype)
        if sample.key not in group:
            self._create_series(group, name, sample.key, values)
        series = group[sample.key]
        pending = self._pending.setdefault(series.name, ([], [], []))
        pending[0].append(values.copy())
        pending[1].append(sample.time)
        pending[2].append(self._sim.step_count)
        if len(pending[0]) >= series["values"].chunks[0]:
            self._write_pending(series.name)

    def _write_pending(self, series_name: str) -> None:
        values, times, steps = self._pending.pop(series_name, ([], [], []))
        if not values:
            return
        series = self._file[series_name]
        count = series["time"].shape[0]
        for key, block in (("values", np.stack(values)), ("time", times), ("step", steps)):
            series[key].resize(count + len(values), axis=0)
            series[key][count:] = block

    def _create_series(self, group: h5py.Group, name: str, key: str, values: np.ndarray) -> None:
        grid = self._sim.grid
        series = group.create_group(key)
        rows = max(1, min(4096, CHUNK_BYTES // max(1, values.nbytes)))
        series.create_dataset(
            "values",
            shape=(0, *values.shape),
            maxshape=(None, *values.shape),
            chunks=(rows, *values.shape),
            dtype=values.dtype,
            compression=self._compression,
        )
        series["values"].attrs["units"] = _units(key, grid.ndim)
        for key_, dtype in (("time", np.float64), ("step", np.int64)):
            series.create_dataset(
                key_, shape=(0,), maxshape=(None,), chunks=(max(rows, 256),), dtype=dtype
            )
        series["time"].attrs["units"] = "s"
        if isinstance(self._outputs[name], Snapshot):
            for axis, coords in zip("xyz", grid.coords(key), strict=False):
                series.create_dataset(axis, data=coords.numpy())
                series[axis].attrs["units"] = "m"
