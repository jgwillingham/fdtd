"""HDF5 output must hold what the in-memory monitors record, and the metadata to interpret it."""

import json

import numpy as np
import pytest
import torch

import fdtd
from fdtd import C0, CPML, Grid, Materials, Mode, Simulation

h5py = pytest.importorskip("h5py")
from fdtd.io import HDF5Writer  # noqa: E402

FREQUENCY = C0 / 20


def batched_sim():
    shape = (24, 20)
    grid = Grid(shape, 1.0, Mode.TE, pml_cells=6)
    eps = torch.ones(2, *shape)
    eps[1, 12:] = 4.0
    sources = [
        [fdtd.PointSource("Ey", (12.0, 10.0), fdtd.ModulatedGaussian(FREQUENCY, 1 / FREQUENCY))],
        [
            fdtd.PointSource("Ex", (6.0, 5.0), fdtd.Sinusoid(FREQUENCY, 2 / FREQUENCY), 2.0),
            fdtd.PointSource("Ey", (8.0, 8.0), lambda t: 0.0),
        ],
    ]
    return Simulation(
        grid, Materials(eps=eps, sigma=0.01), sources, pml=CPML(FREQUENCY), batch_size=2
    )


def test_file_matches_in_memory_monitors(tmp_path):
    sim = batched_sim()
    memory = {
        "fields": fdtd.Snapshot(("Ey", "Hz"), every=10),
        "receiver": fdtd.Probe("Ey", (15.0, 10.0)),
        "energy": fdtd.Energy(every=5),
    }
    sim.monitors.extend(memory.values())
    specs = {
        "fields": fdtd.Snapshot(("Ey", "Hz"), every=10),
        "receiver": fdtd.Probe("Ey", (15.0, 10.0)),
        "energy": fdtd.Energy(every=5),
    }
    path = tmp_path / "run.h5"
    with HDF5Writer(path, sim, specs):
        sim.run(60)
    assert sim.monitors == list(memory.values())  # the writer detached itself
    assert specs["receiver"].values == []  # specifications only

    with h5py.File(path, "r") as f:
        outputs = f["outputs"]
        for name in ("Ey", "Hz"):
            series = outputs["fields"][name]
            np.testing.assert_array_equal(
                series["values"][:], np.stack(memory["fields"].frames[name])
            )
            np.testing.assert_array_equal(series["time"][:], memory["fields"].times[name])
            np.testing.assert_array_equal(series["step"][:], np.arange(10, 61, 10))
            for axis, coords in zip("xy", sim.grid.coords(name), strict=True):
                np.testing.assert_array_equal(series[axis][:], coords.numpy())
        assert outputs["fields"]["Ey"]["values"].shape == (6, 2, 24, 20)
        assert outputs["fields"]["Hz"]["values"].attrs["units"] == "A/m"

        receiver = outputs["receiver"]
        np.testing.assert_array_equal(receiver["Ey"]["values"][:], memory["receiver"].values)
        assert receiver["Ey"]["values"].shape == (60, 2)
        assert receiver.attrs["monitor"] == "Probe"
        np.testing.assert_allclose(receiver.attrs["node"], (15.0, 10.5))

        energy = outputs["energy"]["energy"]
        np.testing.assert_array_equal(energy["values"][:], memory["energy"].values)
        assert energy["values"].attrs["units"] == "J/m"


def test_metadata(tmp_path):
    sim = batched_sim()
    path = tmp_path / "run.h5"
    with HDF5Writer(path, sim, [fdtd.Energy(every=10)]):
        sim.run(30)

    with h5py.File(path, "r") as f:
        assert f.attrs["format"] == "fdtd"
        grid = f["grid"].attrs
        assert grid["mode"] == "TE"
        assert tuple(grid["shape"]) == (24, 20) and grid["pml_cells"] == 6
        assert grid["dt"] == sim.grid.dt
        simulation = f["simulation"].attrs
        assert simulation["batched"] and simulation["batch_size"] == 2
        assert simulation["steps"] == 30 and simulation["time"] == sim.time
        assert simulation["dtype"] == "float32"

        np.testing.assert_array_equal(f["materials"]["eps"][:], sim.materials.eps.numpy())
        assert f["materials"]["sigma"][()] == 0.01
        assert f["pml"].attrs["reference_frequency"] == FREQUENCY

        sources = f["sources"]
        np.testing.assert_array_equal(sources["element"][:], [0, 1, 1])
        assert [c.decode() for c in sources["component"][:]] == ["Ey", "Ex", "Ey"]
        np.testing.assert_array_equal(sources["position"][1], (6.0, 5.0))
        np.testing.assert_array_equal(sources["amplitude"][:], [1.0, 2.0, 1.0])
        waveforms = [json.loads(w) for w in sources["waveform"][:]]
        assert waveforms[0] == {
            "type": "ModulatedGaussian",
            "frequency": FREQUENCY,
            "width": 1 / FREQUENCY,
            "delay": None,
        }
        assert waveforms[2]["type"] == "function"
        assert "energy_0" in f["outputs"]


def test_unbatched_shapes_and_float32_storage(tmp_path):
    grid = Grid((16, 16), 1.0, Mode.TM)
    sim = Simulation(grid, dtype=torch.float64)
    sim.fields["Ez"][8, 8] = 1.0
    path = tmp_path / "run.h5"
    outputs = [fdtd.Snapshot(("Ez",), every=4), fdtd.Probe("Ez", (8.0, 8.5))]
    with HDF5Writer(path, sim, outputs, dtype=np.float32, compression="gzip"):
        sim.run(8)
    with h5py.File(path, "r") as f:
        frames = f["outputs"]["snapshot_0"]["Ez"]["values"]
        assert frames.shape == (2, 16, 16) and frames.dtype == np.float32
        assert frames.compression == "gzip"
        assert f["outputs"]["probe_0"]["Ez"]["values"].shape == (8,)
        assert not f["simulation"].attrs["batched"]
        assert f["sources"]["position"].shape == (0, 2)


def test_reset_discards_samples(tmp_path):
    grid = Grid((16, 16), 1.0, Mode.TM)
    sim = Simulation(grid)
    path = tmp_path / "run.h5"
    with HDF5Writer(path, sim, [fdtd.Energy()]) as writer:
        sim.run(5)
        writer.flush()
        sim.reset()
        sim.run(3)
        writer.flush()
        assert writer.file["outputs"]["energy_0"]["energy"]["time"].shape == (3,)


def test_existing_file_kept_unless_overwrite(tmp_path):
    path = tmp_path / "run.h5"
    path.write_bytes(b"")
    sim = Simulation(Grid((8, 8), 1.0, Mode.TM))
    with pytest.raises(FileExistsError):
        HDF5Writer(path, sim, [])
    assert sim.monitors == []
    with HDF5Writer(path, sim, [], overwrite=True):
        pass
