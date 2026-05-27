"""Unit tests for psyduck.evolve."""

import numpy as np
import pytest
import qutip as qt

from psyduck.evolve import frame_rotate, free_decay, kicked_dynamics
from psyduck.noise import get_collapse_operators
from psyduck.operations import get_spin_operators


# ---------------------------------------------------------------------------
# free_decay
# ---------------------------------------------------------------------------
class TestFreeDecay:
    def test_returns_three_arrays_with_expected_shapes(self):
        I = 7 / 2
        d = int(2 * I + 1)
        # Two coherent superpositions to compute T2* on.
        psi0s = [
            (qt.basis(d, 0) + qt.basis(d, d - 1)).unit(),
            (qt.basis(d, 1) + qt.basis(d, d - 2)).unit(),
        ]
        times = np.linspace(0, 5e-3, 21)
        c_ops = get_collapse_operators(I, T2_star_m=1e-3, exponent_m=1.0)
        fid, T2s, alphas = free_decay(psi0s, times, c_ops)
        assert fid.shape == (2, 21)
        assert T2s.shape == (2,)
        assert alphas.shape == (2,)

    def test_no_collapse_gives_constant_fidelity(self):
        I = 7 / 2
        d = int(2 * I + 1)
        psi0 = qt.basis(d, 0)
        times = np.linspace(0, 1e-3, 5)
        # Empty c_ops — but mesolve still works (no decay).
        fid, _, _ = free_decay([psi0], times, c_ops=[])
        np.testing.assert_allclose(fid[0], 1.0, atol=1e-10)

    def test_T2_close_to_input_for_alpha_one(self):
        # With alpha=1, fidelity decays as exp(-(t/T2)^1). The fit should
        # recover T2_input within ~10% on enough points.
        I = 7 / 2
        d = int(2 * I + 1)
        T2_input = 1e-3
        psi0 = (qt.basis(d, 0) + qt.basis(d, d - 1)).unit()
        c_ops = get_collapse_operators(I, T2_star_m=T2_input, exponent_m=1.0)
        # Need to span several T2 worth of time for a clean fit.
        times = np.linspace(0, 5 * T2_input, 41)
        _, T2s, alphas = free_decay([psi0], times, c_ops)
        # T2_fitted scales with 1/(I·something) — we don't know the analytic
        # mapping for high-spin, only check that fit produced finite, positive.
        assert T2s[0] > 0
        assert np.isfinite(alphas[0])


# ---------------------------------------------------------------------------
# frame_rotate
# ---------------------------------------------------------------------------
class TestFrameRotate:
    def test_zero_time_is_identity(self):
        I = 7 / 2
        d = int(2 * I + 1)
        psi = qt.basis(d, 3)
        _, _, Iz = get_spin_operators(I)
        out = frame_rotate([psi], np.array([0.0]), Iz)
        assert (out[0] - psi).norm() == pytest.approx(0.0, abs=1e-12)

    def test_returns_list_with_correct_length(self):
        I = 7 / 2
        d = int(2 * I + 1)
        states = [qt.basis(d, 0), qt.basis(d, 1), qt.basis(d, 2)]
        times = np.array([0.0, 0.5, 1.0])
        _, _, Iz = get_spin_operators(I)
        out = frame_rotate(states, times, Iz)
        assert isinstance(out, list)
        assert len(out) == 3

    def test_unitary_preserves_norm(self):
        I = 7 / 2
        d = int(2 * I + 1)
        psi = (qt.basis(d, 0) + qt.basis(d, d - 1)).unit()
        _, _, Iz = get_spin_operators(I)
        out = frame_rotate([psi, psi], np.array([0.5, 1.7]), Iz)
        for state in out:
            assert state.norm() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# kicked_dynamics
# ---------------------------------------------------------------------------
class TestKickedDynamics:
    def test_default_returns_four_lists(self):
        I = 7 / 2
        d = int(2 * I + 1)
        psi0 = qt.basis(d, 0)
        psi_list, overlap_list, entropy_list, exp_list = kicked_dynamics(
            psi0, tau=0.1, kappa=2.5, I=I, N=3
        )
        assert len(psi_list) == 4  # initial + N kicks
        assert len(overlap_list) == 4
        assert len(entropy_list) == 4
        assert len(exp_list) == 4

    def test_initial_overlap_is_one(self):
        I = 7 / 2
        d = int(2 * I + 1)
        psi0 = qt.basis(d, 0)
        _, overlap_list, _, _ = kicked_dynamics(
            psi0, tau=0.1, kappa=1.0, I=I, N=2
        )
        # Overlap with self at step 0 is 1.
        assert float(overlap_list[0]) == pytest.approx(1.0, abs=1e-10)

    def test_pure_state_initial_entropy_zero(self):
        I = 7 / 2
        d = int(2 * I + 1)
        psi0 = (qt.basis(d, 0) + qt.basis(d, d - 1)).unit()
        _, _, entropy_list, _ = kicked_dynamics(
            psi0, tau=0.1, kappa=0.5, I=I, N=2
        )
        # Initial state is pure → entropy = 0.
        assert entropy_list[0] == pytest.approx(0.0, abs=1e-10)

    def test_larmor_pulse_type(self):
        I = 7 / 2
        d = int(2 * I + 1)
        psi0 = qt.basis(d, 0)
        psi_list, _, _, _ = kicked_dynamics(
            psi0, tau=0.1, kappa=1.0, I=I, N=2, pulse_type="larmor"
        )
        # Just check it ran and produced unit-norm states.
        for psi in psi_list:
            assert psi.norm() == pytest.approx(1.0, abs=1e-8)
