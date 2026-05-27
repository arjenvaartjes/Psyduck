"""Unit tests for psyduck.hamiltonians."""

import numpy as np
import pytest
import qutip as qt

from psyduck.hamiltonians import (
    Hz_order,
    drive_hamiltonian,
    get_quadrupole_splittings,
    hyperfine_hamiltonian,
    ner1_hamiltonian,
    ner2_hamiltonian,
    nmr1_hamiltonian,
    quadrupole_hamiltonian,
    quadrupole_hamiltonian_from_Vab,
    zeeman_hamiltonian,
)
from psyduck.operations import get_spin_operators


# ---------------------------------------------------------------------------
# zeeman_hamiltonian
# ---------------------------------------------------------------------------
class TestZeemanHamiltonian:
    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_default_is_minus_gamma_B0_Iz(self, I):
        gamma, B0 = 2.5, 1.7
        H = zeeman_hamiltonian(I, B0, gamma)
        _, _, Iz = get_spin_operators(I)
        assert (H - (-gamma * B0 * Iz)).norm() == pytest.approx(0.0, abs=1e-12)

    def test_field_along_x(self):
        I = 7 / 2
        gamma, B0 = 1.0, 1.0
        H = zeeman_hamiltonian(I, B0, gamma, theta=np.pi / 2, phi=0.0)
        Ix, _, _ = get_spin_operators(I)
        assert (H - (-gamma * B0 * Ix)).norm() == pytest.approx(0.0, abs=1e-10)

    def test_hermitian(self):
        H = zeeman_hamiltonian(7 / 2, B0=1.0, gamma=1.5, theta=0.5, phi=1.2)
        assert (H - H.dag()).norm() == pytest.approx(0.0, abs=1e-12)

    def test_eigenvalues_for_default_orientation(self):
        # Field along z: eigenvalues are -gamma*B0*m for m in {-I, ..., I}.
        I = 7 / 2
        gamma, B0 = 2.0, 1.0
        H = zeeman_hamiltonian(I, B0, gamma)
        evals = np.sort(np.real(H.eigenenergies()))
        ms = np.arange(-I, I + 1)
        expected = np.sort(-gamma * B0 * ms)
        np.testing.assert_allclose(evals, expected, atol=1e-12)

    def test_tensor_product_form(self):
        # Two spins (I=1/2 and I=7/2): result lives in the (2*8)-dim Hilbert space.
        H = zeeman_hamiltonian([1 / 2, 7 / 2], B0=1.0, gamma=[1.0, 2.0])
        assert H.shape == (2 * 8, 2 * 8)


# ---------------------------------------------------------------------------
# quadrupole_hamiltonian
# ---------------------------------------------------------------------------
class TestQuadrupoleHamiltonian:
    def test_axial_matches_Iz_squared_form(self):
        # Axial PAF aligned with z (theta=0, eta=0): expect H = (f_q/3)*(3Iz²-I²)
        # diagonalised, but since Q_ab = (f_q/3)*diag([-0.5, -0.5, 1]),
        # H = Q_ab[0,0]*Ix² + Q_ab[1,1]*Iy² + Q_ab[2,2]*Iz²
        #   = (f_q/3)*(-0.5 Ix² - 0.5 Iy² + Iz²) = (f_q/3)*(Iz² - I(I+1)/2 + Iz²/... )
        # Easier: just check it's diagonal with eigenvalues we expect by reconstruction.
        I = 7 / 2
        f_q = 25e3
        H = quadrupole_hamiltonian(I, f_q, eta=0.0)
        Ix, Iy, Iz = get_spin_operators(I)
        expected = (f_q / 3) * (-0.5 * Ix * Ix - 0.5 * Iy * Iy + Iz * Iz)
        assert (H - expected).norm() == pytest.approx(0.0, abs=1e-8)

    def test_hermitian(self):
        H = quadrupole_hamiltonian(7 / 2, f_q=25e3, eta=0.3, theta=0.5, phi=1.2)
        assert (H - H.dag()).norm() == pytest.approx(0.0, abs=1e-8)

    def test_zero_fq_is_zero(self):
        H = quadrupole_hamiltonian(7 / 2, f_q=0.0, eta=0.5)
        assert H.norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# quadrupole_hamiltonian_from_Vab
# ---------------------------------------------------------------------------
class TestQuadrupoleHamiltonianFromVab:
    def test_single_tensor_returns_qobj(self):
        V_ab = np.diag([-1.0, -1.0, 2.0])
        H = quadrupole_hamiltonian_from_Vab(7 / 2, V_ab, Q=2e-29)
        assert isinstance(H, qt.Qobj)
        assert H.shape == (8, 8)

    def test_batched_returns_ndarray(self):
        V_batch = np.array([np.diag([-1.0, -1.0, 2.0]), np.diag([0.5, -2.0, 1.5])])
        out = quadrupole_hamiltonian_from_Vab(7 / 2, V_batch, Q=2e-29)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 8, 8)

    def test_linear_in_Vab(self):
        V = np.diag([-1.0, -1.0, 2.0])
        H1 = quadrupole_hamiltonian_from_Vab(7 / 2, V, Q=2e-29)
        H2 = quadrupole_hamiltonian_from_Vab(7 / 2, 3 * V, Q=2e-29)
        np.testing.assert_allclose(H2.full(), 3 * H1.full(), atol=1e-12)


# ---------------------------------------------------------------------------
# hyperfine_hamiltonian
# ---------------------------------------------------------------------------
class TestHyperfineHamiltonian:
    def test_dimensions(self):
        # S=1/2 (dim 2), I=7/2 (dim 8) → joint dim 16.
        H = hyperfine_hamiltonian(S=1 / 2, I=7 / 2, A=1.0)
        assert H.shape == (2 * 8, 2 * 8)

    def test_hermitian(self):
        H = hyperfine_hamiltonian(S=1 / 2, I=7 / 2, A=2.5e6)
        assert (H - H.dag()).norm() == pytest.approx(0.0, abs=1e-8)

    def test_proportional_to_A(self):
        H1 = hyperfine_hamiltonian(S=1 / 2, I=7 / 2, A=1.0)
        H2 = hyperfine_hamiltonian(S=1 / 2, I=7 / 2, A=3.5)
        assert (H2 - 3.5 * H1).norm() == pytest.approx(0.0, abs=1e-12)

    def test_zero_A_is_zero(self):
        H = hyperfine_hamiltonian(S=1 / 2, I=7 / 2, A=0.0)
        assert H.norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# nmr1_hamiltonian
# ---------------------------------------------------------------------------
class TestNmr1Hamiltonian:
    def test_scalar_B1_matches_minus_gamma_B1_Ix(self):
        I = 7 / 2
        gamma, B1 = 2.0, 1.5
        H = nmr1_hamiltonian(I, B1=B1, axis="x", gamma=gamma)
        Ix, _, _ = get_spin_operators(I)
        assert (H - (-gamma * B1 * Ix)).norm() == pytest.approx(0.0, abs=1e-12)

    def test_axis_y(self):
        I = 7 / 2
        H = nmr1_hamiltonian(I, B1=1.0, axis="y", gamma=1.0)
        _, Iy, _ = get_spin_operators(I)
        assert (H - (-1.0 * Iy)).norm() == pytest.approx(0.0, abs=1e-12)

    def test_per_transition_array_shape(self):
        I = 7 / 2
        d = int(2 * I + 1)
        B1 = np.linspace(0.5, 1.5, int(2 * I))  # one entry per Δm=1 transition
        H = nmr1_hamiltonian(I, B1=B1, axis="x", gamma=1.0)
        assert H.shape == (d, d)
        assert (H - H.dag()).norm() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# ner1_hamiltonian
# ---------------------------------------------------------------------------
class TestNer1Hamiltonian:
    def test_scalar_returns_qobj(self):
        H = ner1_hamiltonian(7 / 2, dQxz=1.0, dQyz=0.0)
        assert isinstance(H, qt.Qobj)
        assert H.shape == (8, 8)

    def test_hermitian_scalar(self):
        H = ner1_hamiltonian(7 / 2, dQxz=0.7, dQyz=-0.3)
        assert (H - H.dag()).norm() == pytest.approx(0.0, abs=1e-10)

    def test_per_transition_array(self):
        I = 7 / 2
        n = int(2 * I)
        H = ner1_hamiltonian(
            I, dQxz=np.ones(n), dQyz=np.zeros(n), coupling=1.0
        )
        assert H.shape == (8, 8)
        assert (H - H.dag()).norm() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# ner2_hamiltonian
# ---------------------------------------------------------------------------
class TestNer2Hamiltonian:
    def test_scalar_returns_qobj(self):
        H = ner2_hamiltonian(7 / 2, dQxx_yy=1.0, dQxy=0.0)
        assert isinstance(H, qt.Qobj)
        assert H.shape == (8, 8)

    def test_hermitian(self):
        H = ner2_hamiltonian(7 / 2, dQxx_yy=0.5, dQxy=0.7)
        assert (H - H.dag()).norm() == pytest.approx(0.0, abs=1e-10)

    def test_per_transition_array(self):
        I = 7 / 2
        n = int(2 * I) - 1
        H = ner2_hamiltonian(
            I, dQxx_yy=np.ones(n), dQxy=np.zeros(n), coupling=1.0
        )
        assert H.shape == (8, 8)
        assert (H - H.dag()).norm() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Hz_order
# ---------------------------------------------------------------------------
class TestHzOrder:
    def test_order_one_is_kappa_Iz(self):
        # H = kappa * Iz^1 / (1 * I^0) = kappa * Iz.
        I = 7 / 2
        kappa = 0.7
        H = Hz_order(kappa, order=1, spin_I=I)
        Iz = qt.jmat(I, "z")
        assert (H - kappa * Iz).norm() == pytest.approx(0.0, abs=1e-12)

    def test_order_two_formula(self):
        I = 7 / 2
        kappa = 1.3
        H = Hz_order(kappa, order=2, spin_I=I)
        Iz = qt.jmat(I, "z")
        expected = kappa * (Iz ** 2) / (2 * I)
        assert (H - expected).norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# drive_hamiltonian
# ---------------------------------------------------------------------------
class TestDriveHamiltonian:
    def test_scalar_time_returns_qobj(self):
        I = 7 / 2
        n_trans = int(2 * I)
        n_freqs = 3
        amps = np.ones(n_freqs, dtype=complex)
        freqs = np.array([1.0, 2.0, 3.0])
        rf_freqs = np.linspace(0.5, 3.5, n_trans)
        H = drive_hamiltonian(I, time_array=0.0, drive_amplitudes=amps,
                              drive_frequencies=freqs,
                              rotating_frame_frequencies=rf_freqs)
        assert isinstance(H, qt.Qobj)
        assert H.shape == (8, 8)

    def test_array_time_returns_list_of_pairs(self):
        I = 7 / 2
        n_trans = int(2 * I)
        n_freqs = 2
        T = 5
        t = np.linspace(0, 1e-3, T)
        amps = np.ones((n_freqs, T), dtype=complex)
        freqs = np.array([1.0, 2.0])
        rf_freqs = np.linspace(0.5, 2.5, n_trans)
        H_drive = drive_hamiltonian(I, time_array=t, drive_amplitudes=amps,
                                    drive_frequencies=freqs,
                                    rotating_frame_frequencies=rf_freqs)
        # One [op, coeff] pair per (transition, x-or-y).
        assert isinstance(H_drive, list)
        assert len(H_drive) == 2 * n_trans

    def test_assertion_on_rf_freq_length(self):
        with pytest.raises(AssertionError):
            drive_hamiltonian(
                I=7 / 2,
                time_array=0.0,
                drive_amplitudes=np.ones(2, dtype=complex),
                drive_frequencies=np.array([1.0, 2.0]),
                rotating_frame_frequencies=np.array([1.0, 2.0]),  # wrong length
            )

    def test_cross_coupling_cutoff_zero(self):
        # A tiny cutoff means no drive components match → H is zero (but still
        # the right shape).
        I = 7 / 2
        n_trans = int(2 * I)
        amps = np.ones(2, dtype=complex)
        H = drive_hamiltonian(
            I,
            time_array=0.0,
            drive_amplitudes=amps,
            drive_frequencies=np.array([100.0, 200.0]),
            rotating_frame_frequencies=np.linspace(0.5, 3.5, n_trans),
            cross_coupling_cutoff=1e-6,
        )
        assert H.norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# get_quadrupole_splittings — light smoke test (compute on a tiny grid)
# ---------------------------------------------------------------------------
class TestQuadrupoleSplittings:
    def test_shape_and_finiteness(self):
        thetas = np.linspace(0, np.pi / 2, 3)
        phis = np.linspace(0, np.pi, 4)
        V_ab = np.diag([-1e20, -1e20, 2e20])
        fq1, fq2 = get_quadrupole_splittings(
            V_ab, I=7 / 2, B0=1.0, gamma=2e7, Q=2e-29,
            thetas=thetas, phis=phis,
        )
        assert fq1.shape == (3, 4)
        assert fq2.shape == (3, 4)
        assert np.all(np.isfinite(fq1))
        assert np.all(np.isfinite(fq2))
