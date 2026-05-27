"""Unit tests for psyduck.noise."""

import numpy as np
import pytest
import qutip as qt

from psyduck.noise import get_collapse_operators
from psyduck.operations import get_spin_operators


# ---------------------------------------------------------------------------
# Magnetic-only noise (default)
# ---------------------------------------------------------------------------
class TestMagneticOnly:
    def test_returns_single_pair_when_only_magnetic(self):
        c_ops = get_collapse_operators(7 / 2, T2_star_m=1e-3, exponent_m=1.0)
        assert len(c_ops) == 1
        operator, profile = c_ops[0]
        assert isinstance(operator, qt.Qobj)
        assert callable(profile)

    def test_operator_is_Iz(self):
        I = 7 / 2
        _, _, Iz = get_spin_operators(I)
        c_ops = get_collapse_operators(I, T2_star_m=1e-3, exponent_m=1.0)
        operator, _ = c_ops[0]
        assert (operator - Iz).norm() == pytest.approx(0.0, abs=1e-12)

    def test_profile_has_correct_amplitude_at_t1_for_alpha_one(self):
        # alpha=1 → time-independent profile = sqrt(2/T2*).
        T2 = 1e-3
        c_ops = get_collapse_operators(7 / 2, T2_star_m=T2, exponent_m=1.0)
        _, profile = c_ops[0]
        expected = np.sqrt(2 / T2)
        assert profile(1.0, None) == pytest.approx(expected, rel=1e-10)
        # Independent of t for alpha=1.
        assert profile(0.5, None) == pytest.approx(expected, rel=1e-10)

    def test_stretched_exponent_scales_with_t(self):
        # alpha=2 → profile = sqrt(rate) * t^(0.5).
        T2 = 1e-3
        c_ops = get_collapse_operators(7 / 2, T2_star_m=T2, exponent_m=2.0)
        _, profile = c_ops[0]
        rate = (2 / T2) ** 2
        for t in [0.1, 0.5, 1.0]:
            assert profile(t, None) == pytest.approx(np.sqrt(rate) * np.sqrt(t))

    def test_no_electric_when_T2e_none(self):
        c_ops = get_collapse_operators(
            7 / 2, T2_star_m=1e-3, exponent_m=1.0, T2_star_e=None, exponent_e=2.0
        )
        assert len(c_ops) == 1


# ---------------------------------------------------------------------------
# Magnetic + electric noise
# ---------------------------------------------------------------------------
class TestMagneticPlusElectric:
    def test_returns_two_pairs(self):
        c_ops = get_collapse_operators(
            7 / 2, T2_star_m=1e-3, exponent_m=1.0, T2_star_e=2e-3, exponent_e=2.0
        )
        assert len(c_ops) == 2

    def test_electric_operator_is_Iz_squared(self):
        I = 7 / 2
        _, _, Iz = get_spin_operators(I)
        c_ops = get_collapse_operators(
            I, T2_star_m=1e-3, exponent_m=1.0, T2_star_e=2e-3, exponent_e=2.0
        )
        electric_op, _ = c_ops[1]
        assert (electric_op - Iz * Iz).norm() == pytest.approx(0.0, abs=1e-12)

    def test_electric_profile_alpha_two(self):
        T2_e = 2e-3
        c_ops = get_collapse_operators(
            7 / 2, T2_star_m=1e-3, exponent_m=1.0, T2_star_e=T2_e, exponent_e=2.0
        )
        _, electric_profile = c_ops[1]
        rate_e = (2 / T2_e) ** 2
        assert electric_profile(0.5, None) == pytest.approx(
            np.sqrt(rate_e) * np.sqrt(0.5)
        )


# ---------------------------------------------------------------------------
# I=1/2 special case (electric noise suppressed)
# ---------------------------------------------------------------------------
class TestSpinHalfSuppressesElectric:
    def test_electric_dropped_for_spin_half(self):
        # Even with electric params provided, I=1/2 has no rank-2 quadrupole
        # interaction → only magnetic returned.
        c_ops = get_collapse_operators(
            1 / 2, T2_star_m=1e-3, exponent_m=1.0, T2_star_e=2e-3, exponent_e=2.0
        )
        assert len(c_ops) == 1


# ---------------------------------------------------------------------------
# Integration with mesolve
# ---------------------------------------------------------------------------
class TestIntegrationWithMesolve:
    def test_runs_with_mesolve(self):
        I = 7 / 2
        d = int(2 * I + 1)
        c_ops = get_collapse_operators(
            I, T2_star_m=1e-2, exponent_m=1.0, T2_star_e=2e-2, exponent_e=2.0
        )
        # Coherent superposition along x (so Iz dephasing has something to act on).
        psi0 = (qt.basis(d, 0) + qt.basis(d, d - 1)).unit()
        H = qt.qzero(d)
        result = qt.mesolve(H, psi0, np.linspace(0, 5e-3, 11), c_ops=c_ops)
        rho_final = result.states[-1]
        purity = (rho_final * rho_final).tr().real
        # Pure dephasing on a coherent superposition reduces purity below 1.
        assert purity < 1.0
        assert purity > 0.0
