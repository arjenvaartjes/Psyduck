"""Unit tests for psyduck.spin_composite.SpinComposite."""

import numpy as np
import pytest
import qutip as qt

from psyduck import Spin, SpinComposite


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_default_spins_are_half_and_seven_half(self):
        # Per __init__: defaults are [Spin(1/2), Spin(7/2)].
        sc = SpinComposite()
        assert len(sc.spins) == 2
        assert [s.I for s in sc.spins] == [1 / 2, 7 / 2]
        assert sc.I == [1 / 2, 7 / 2]  # convenience attribute

    def test_state_is_tensor_product_of_inputs(self):
        s1 = Spin(1 / 2)
        s2 = Spin(7 / 2)
        sc = SpinComposite([s1, s2])
        expected = qt.tensor(s1.state, s2.state)
        assert (sc.state - expected).norm() == pytest.approx(0.0, abs=1e-12)

    def test_custom_state_overrides_default(self):
        d_total = 2 * 8
        custom = qt.basis(d_total, 5)
        custom.dims = [[2, 8], [1, 1]]
        sc = SpinComposite([Spin(1 / 2), Spin(7 / 2)], state=custom)
        assert (sc.state - custom).norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Ix / Iy / Iz with and without spin_idx
# ---------------------------------------------------------------------------
class TestSpinComponents:
    def test_total_iz_is_sum_of_subsystem_iz(self):
        # Both spins start in default state (basis 0 → +I), so total
        # <Iz_total> = I_1 + I_2 = 1/2 + 7/2 = 4.
        sc = SpinComposite()
        assert sc.Iz() == pytest.approx(1 / 2 + 7 / 2, abs=1e-12)

    def test_iz_per_subsystem(self):
        sc = SpinComposite()
        assert sc.Iz(spin_idx=0) == pytest.approx(1 / 2, abs=1e-12)
        assert sc.Iz(spin_idx=1) == pytest.approx(7 / 2, abs=1e-12)

    def test_ix_default_zero(self):
        sc = SpinComposite()
        assert abs(sc.Ix()) == pytest.approx(0.0, abs=1e-12)

    def test_iy_default_zero(self):
        sc = SpinComposite()
        assert abs(sc.Iy()) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------
class TestParity:
    def test_default_parity(self):
        # Both at basis 0 (even index) → parity is +1 ⊗ +1 = +1.
        sc = SpinComposite()
        assert sc.parity() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# global_rotate
# ---------------------------------------------------------------------------
class TestGlobalRotate:
    def test_global_pi_y_inverts_total_iz(self):
        sc = SpinComposite()
        sc.global_rotate(np.pi, "y")
        assert sc.Iz() == pytest.approx(-(1 / 2 + 7 / 2), abs=1e-10)

    def test_global_pi_y_on_one_subsystem(self):
        sc = SpinComposite()
        sc.global_rotate(np.pi, "y", spin_idx=1)
        # Only spin 1 (the I=7/2 one) flips: Iz = +1/2 - 7/2.
        assert sc.Iz() == pytest.approx(1 / 2 - 7 / 2, abs=1e-10)
        # Spin 0 unchanged.
        assert sc.Iz(spin_idx=0) == pytest.approx(1 / 2, abs=1e-10)


# ---------------------------------------------------------------------------
# subspace_rotate (single subsystem)
# ---------------------------------------------------------------------------
class TestSubspaceRotate:
    def test_pi_x_swaps_top_two_levels_of_chosen_spin(self):
        sc = SpinComposite()  # state = |+1/2> ⊗ |+7/2>  (basis index 0 of each)
        I_target = 7 / 2
        sc.subspace_rotate(np.pi, "x", (I_target, I_target - 1), spin_idx=1)
        # Spin 1's Iz should now be 5/2 (one step down).
        assert sc.Iz(spin_idx=1) == pytest.approx(5 / 2, abs=1e-10)
        # Spin 0 unchanged.
        assert sc.Iz(spin_idx=0) == pytest.approx(1 / 2, abs=1e-10)


# ---------------------------------------------------------------------------
# shift
# ---------------------------------------------------------------------------
class TestShift:
    def test_dim_applications_returns_to_initial(self):
        sc = SpinComposite()
        before = sc.state.copy()
        d = sc.spins[0].dim  # default spin_idx=0 → I=1/2 → dim=2
        for _ in range(d):
            sc.shift()
        assert qt.fidelity(before, sc.state) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# copy / repr / state_labels
# ---------------------------------------------------------------------------
class TestDunderAndCopy:
    def test_copy_independent_state(self):
        sc = SpinComposite()
        c = sc.copy()
        c.global_rotate(np.pi, "y", spin_idx=1)
        # Original unchanged.
        assert sc.Iz(spin_idx=1) == pytest.approx(7 / 2, abs=1e-10)
        # Copy flipped.
        assert c.Iz(spin_idx=1) == pytest.approx(-7 / 2, abs=1e-10)

    def test_state_labels_returns_list_per_subsystem(self):
        sc = SpinComposite()
        labels = sc.state_labels()
        assert len(labels) == 2
        assert labels[0] == ["|1/2>", "|-1/2>"]
        assert len(labels[1]) == 8

    def test_repr_contains_class_name(self):
        sc = SpinComposite()
        assert "SpinComposite" in repr(sc)
