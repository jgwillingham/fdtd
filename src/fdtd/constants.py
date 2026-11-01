"""Physical constants in SI units."""

import math

C0 = 299_792_458.0  # speed of light in vacuum (m/s)
MU0 = 1.25663706212e-6  # vacuum permeability (H/m)
EPS0 = 1.0 / (MU0 * C0**2)  # vacuum permittivity (F/m)
ETA0 = math.sqrt(MU0 / EPS0)  # impedance of free space (ohm)
