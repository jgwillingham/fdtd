"""Validation of the bare Yee engine against analytic results."""

import math

import numpy as np
import pytest
import torch
from scipy.special import hankel2

import fdtd
from fdtd import C0, EPS0, Grid, Materials, Mode, Simulation


def dominant_frequency(values, dt):
    """Frequency of the strongest spectral peak, refined by parabolic interpolation."""
    signal = np.asarray(values) - np.mean(values)
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window, 8 * len(signal)))
    peak = int(np.argmax(spectrum))
    a, b, c = np.log(spectrum[peak - 1 : peak + 2])
    shift = 0.5 * (a - c) / (a - 2 * b + c)
    return (peak + shift) / (8 * len(signal) * dt)


def tm_cavity(n, dtype=torch.float64):
    """Square PEC cavity of side 1 m with Ez seeded in the (1, 1) mode."""
    grid = Grid((n + 1, n + 1), 1.0 / n, Mode.TM, courant=0.5)
    sim = Simulation(grid, dtype=dtype)
    x = torch.arange(n + 1, dtype=dtype) / n
    mode = torch.sin(math.pi * x)[:, None] * torch.sin(math.pi * x)[None, :]
    sim.fields["Ez"].copy_(mode)
    return grid, sim


def test_courant_limit_rejected():
    with pytest.raises(ValueError, match="Unstable"):
        Grid((10, 10), 1.0, Mode.TE, courant=0.72)


@pytest.mark.parametrize(
    ("mode", "shape", "components"),
    [
        (Mode.TE, (6, 7), ("Ex", "Ey", "Hz")),
        (Mode.TM, (6, 7), ("Ez", "Hx", "Hy")),
        (Mode.FULL_3D, (6, 7, 8), ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")),
    ],
)
def test_curl_stencils_are_centred_on_their_target(mode, shape, components):
    """Every difference must straddle the target node, half a cell either side."""
    grid = Grid(shape, 0.5, mode)
    assert grid.components == components
    for field in "EH":
        for term in grid.curl_terms(field):
            target, ahead, behind = fdtd.kernels.term_slices(grid, term)
            target_at = grid.offsets(term.target)
            source_at = grid.offsets(term.source)
            for axis in range(grid.ndim):
                t = range(*target[axis].indices(grid.dims[axis]))
                a = range(*ahead[axis].indices(grid.dims[axis]))
                b = range(*behind[axis].indices(grid.dims[axis]))
                assert len(t) == len(a) == len(b)
                if axis == term.axis:
                    assert t[0] + target_at[axis] == pytest.approx(a[0] + source_at[axis] - 0.5)
                    assert t[0] + target_at[axis] == pytest.approx(b[0] + source_at[axis] + 0.5)
                else:
                    assert t[0] + target_at[axis] == pytest.approx(a[0] + source_at[axis])


def test_coords_follow_the_stagger():
    grid = Grid((4, 5), 2.0, Mode.TE, pml_cells=3)
    x, y = grid.coords("Hz")
    assert x[0] == 1.0 and y[0] == 1.0
    assert grid.nearest_index("Hz", (1.0, 3.0)) == (3, 4)
    assert grid.coords("Ey")[1][0] == 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cavity_resonance_2d(dtype):
    grid, sim = tm_cavity(40, dtype)
    probe = fdtd.Probe("Ez", (0.5, 0.5))
    sim.monitors.append(probe)
    sim.run(4000)
    analytic = C0 * math.sqrt(2) / 2
    measured = dominant_frequency(probe.values, grid.dt)
    assert measured == pytest.approx(analytic, rel=3e-3)


def test_cavity_resonance_converges_second_order():
    errors = []
    for n in (20, 40):
        grid, sim = tm_cavity(n)
        probe = fdtd.Probe("Ez", (0.5, 0.5))
        sim.monitors.append(probe)
        sim.run(100 * n)
        measured = dominant_frequency(probe.values, grid.dt)
        errors.append(abs(measured / (C0 * math.sqrt(2) / 2) - 1))
    assert errors[0] / errors[1] == pytest.approx(4.0, abs=0.8)


def test_cavity_resonance_3d():
    """TE_101-like mode of a cube, with Ey = sin(pi x) sin(pi z)."""
    n = 16
    grid = Grid((n + 1, n + 1, n + 1), 1.0 / n, courant=0.4)
    sim = Simulation(grid, dtype=torch.float64)
    x = torch.arange(n + 1, dtype=torch.float64) / n
    sim.fields["Ey"].copy_(
        torch.sin(math.pi * x)[:, None, None] * torch.sin(math.pi * x)[None, None, :]
    )
    probe = fdtd.Probe("Ey", (0.5, 0.5 + 0.5 / n, 0.5))
    sim.monitors.append(probe)
    sim.run(1500)
    measured = dominant_frequency(probe.values, grid.dt)
    assert measured == pytest.approx(C0 * math.sqrt(2) / 2, rel=5e-3)


@pytest.mark.parametrize("mode", [Mode.TE, Mode.TM])
def test_lossless_cavity_energy_is_conserved(mode):
    component = "Ey" if mode is Mode.TE else "Ez"
    source = fdtd.PointSource(component, (0.4, 0.3), fdtd.ModulatedGaussian(2e9, 4e-10))
    grid = Grid((60, 60), 0.02, mode)
    sim = Simulation(grid, sources=[source], dtype=torch.float64)
    sim.run(400)

    def staggered_energy():
        """The exact invariant of the leapfrog: eps E(n)^2 + mu H(n - 1/2) . H(n + 1/2)."""
        e_names = [name for name in grid.components if name[0] == "E"]
        h_names = [name for name in grid.components if name[0] == "H"]
        h_before = [sim.fields[name].clone() for name in h_names]
        electric = sum(sim.fields[name].square().sum().item() for name in e_names)
        sim.run(1)
        magnetic = sum(
            (old * sim.fields[name]).sum().item()
            for old, name in zip(h_before, h_names, strict=True)
        )
        return EPS0 * electric + fdtd.MU0 * magnetic

    invariant = np.array([staggered_energy() for _ in range(1000)])
    assert invariant.mean() > 0
    assert (invariant.max() - invariant.min()) / invariant.mean() < 1e-9


def test_conductive_decay():
    """A uniform field in a conductor decays as exp(-sigma t / eps)."""
    eps, steps = 4.0, 15
    grid = Grid((40, 40), 1.0, Mode.TM)
    sigma = eps * EPS0 / (steps * grid.dt)  # one e-folding, with a small loss per step
    sim = Simulation(grid, Materials(eps=eps, sigma=sigma), dtype=torch.float64)
    sim.fields["Ez"][1:-1, 1:-1] = 1.0
    sim.run(steps)  # the wall disturbance has not reached the centre yet
    assert sim.fields["Ez"][20, 20].item() == pytest.approx(math.exp(-1), rel=1e-3)


def reflection_error(refinement, n_index=2.0):
    """Error of the reflection from a half space at normal incidence, against Fresnel."""
    r = refinement
    nx, ny, dx = 1100 * r, 8, 2e-3 / r
    source_x, probe_x, interface_x = 300 * r, 400 * r, 700 * r
    pulse = fdtd.ModulatedGaussian(6e9, 3e-10, delay=9e-10)
    eps = torch.ones(nx, ny, dtype=torch.float64)
    eps[interface_x:] = n_index**2

    def probe_series(materials):
        grid = Grid((nx, ny), dx, Mode.TE, courant=0.5)
        probe = fdtd.Probe("Ey", (probe_x * dx, 4.5 * dx))
        # A sheet of Jy launches a plane wave, which the PEC y walls leave undisturbed.
        sheet = [
            fdtd.PointSource("Ey", (source_x * dx, (j + 0.5) * dx), pulse) for j in range(ny - 1)
        ]
        sim = Simulation(grid, materials, sheet, [probe], dtype=torch.float64)
        sim.run(1940 * r)  # the reflection has passed, and no wall echo has arrived
        return np.array(probe.values)

    incident = probe_series(Materials())
    reflected = probe_series(Materials(eps=eps)) - incident
    measured = np.abs(reflected).max() / np.abs(incident[: 740 * r]).max()
    return abs(measured - abs((1 - n_index) / (1 + n_index)))


def test_dielectric_interface_reflection_converges_to_fresnel():
    coarse, fine = reflection_error(1), reflection_error(2)
    assert fine < 0.005
    assert coarse / fine == pytest.approx(4.0, abs=0.8)


def test_line_source_amplitude():
    """A z-directed line current I radiates |Ez| = (w mu0 I / 4) |H0^(2)(k r)|."""
    n, dx, r_cells = 600, 1e-3, 40
    frequency, current = 3e9, 2.0
    grid = Grid((n, n), dx, Mode.TM, courant=0.5)
    centre = n // 2 * dx
    r = r_cells * dx
    source = fdtd.PointSource(
        "Ez", (centre, centre), fdtd.Sinusoid(frequency, ramp=2 / frequency), current / dx**2
    )
    probe = fdtd.Probe("Ez", (centre + r, centre))
    sim = Simulation(grid, sources=[source], monitors=[probe], dtype=torch.float64)
    # A cylindrical wave settles slowly, so stop only just before the wall echo.
    sim.run(int(0.95 * (n - r_cells) * dx / C0 / grid.dt))
    period_steps = round(1 / frequency / grid.dt)
    measured = np.abs(probe.values[-period_steps:]).max()
    omega = 2 * math.pi * frequency
    analytic = omega * fdtd.MU0 * current / 4 * abs(hankel2(0, omega / C0 * r))
    assert measured == pytest.approx(analytic, rel=2e-3)
