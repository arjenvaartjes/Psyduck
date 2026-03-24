"""Noise models for open spin system dynamics (Lindblad collapse operators)."""

import numpy as np
from psyduck.operations import get_spin_operators


def get_collapse_operators(I, T2_star_m, exponent_m, T2_star_e=None, exponent_e=None):
    """Build time-dependent Lindblad collapse operators for dephasing noise.

    Magnetic noise (Iz) is always included. Electric/quadrupole noise (Iz²)
    is included only for I > 1/2 and when T2_star_e and exponent_e are provided.

    The noise profile reproduces a stretched-exponential coherence decay
    exp(-(t/T2*)^alpha) via a time-dependent collapse operator amplitude
    sqrt(rate) * t^((alpha-1)/2).

    Parameters
    ----------
    I : float
        Spin quantum number.
    T2_star_m : float
        Characteristic dephasing time for magnetic (Iz) noise.
    exponent_m : float
        Exponent for magnetic noise.
    T2_star_e : float, optional
        Characteristic dephasing time for electric (Iz^2) noise.
    exponent_e : float, optional
        Exponent for electric noise.

    Returns
    -------
    list
        QuTiP time-dependent collapse operator list suitable for mesolve.
    """
    _, _, Iz = get_spin_operators(I)

    rate_m = (2 / T2_star_m) ** exponent_m

    def magnetic_profile(t, args):
        return np.sqrt(rate_m) * t ** ((exponent_m - 1) / 2)

    c_ops = [[Iz, magnetic_profile]]

    if I > 0.5 and T2_star_e is not None and exponent_e is not None:
        rate_e = (2 / T2_star_e) ** exponent_e

        def electric_profile(t, args):
            return np.sqrt(rate_e) * t ** ((exponent_e - 1) / 2)

        c_ops.append([Iz * Iz, electric_profile])

    return c_ops
