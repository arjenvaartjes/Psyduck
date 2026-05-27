"""Unit tests for psyduck.operations."""

import numpy as np
import pytest
import qutip as qt

from psyduck.operations import (
    euler_rotation,
    get_spin_operators,
    get_transition_operators,
    global_pi,
    global_rotation,
    parity_operator,
    permutation_operator,
    shift_operator,
    snap,
    subspace_rotation,
)


# ---------------------------------------------------------------------------
# get_spin_operators
# ---------------------------------------------------------------------------
class TestGetSpinOperators:
    @pytest.mark.parametrize("I", [1 / 2, 1, 3 / 2, 7 / 2])
    def test_returns_three_qobjs_of_correct_shape(self, I):
        d = int(2 * I + 1)
        Ix, Iy, Iz = get_spin_operators(I)
        for op in (Ix, Iy, Iz):
            assert isinstance(op, qt.Qobj)
            assert op.shape == (d, d)

    @pytest.mark.parametrize("I", [1 / 2, 7 / 2])
    def test_matches_qt_jmat(self, I):
        Ix, Iy, Iz = get_spin_operators(I)
        assert (Ix - qt.jmat(I, "x")).norm() == pytest.approx(0.0, abs=1e-12)
        assert (Iy - qt.jmat(I, "y")).norm() == pytest.approx(0.0, abs=1e-12)
        assert (Iz - qt.jmat(I, "z")).norm() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_iz_eigenvalues_descending(self, I):
        _, _, Iz = get_spin_operators(I)
        evals = np.sort(np.real(Iz.eigenenergies()))[::-1]
        expected = np.arange(I, -I - 1, -1)
        np.testing.assert_allclose(evals, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# get_transition_operators
# ---------------------------------------------------------------------------
class TestGetTransitionOperators:
    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_lengths_match_2I(self, I):
        Tx, Ty = get_transition_operators(I)
        assert len(Tx) == int(2 * I)
        assert len(Ty) == int(2 * I)

    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_sum_recovers_Ix_Iy(self, I):
        Ix, Iy, _ = get_spin_operators(I)
        Tx, Ty = get_transition_operators(I)
        assert (sum(Tx) - Ix).norm() == pytest.approx(0.0, abs=1e-12)
        assert (sum(Ty) - Iy).norm() == pytest.approx(0.0, abs=1e-12)

    def test_each_transition_is_two_level(self):
        Tx, _ = get_transition_operators(7 / 2)
        for k, tx in enumerate(Tx):
            arr = tx.full()
            mask = np.zeros_like(arr, dtype=bool)
            mask[k, k + 1] = True
            mask[k + 1, k] = True
            # Only those two off-diagonal entries are non-zero.
            assert np.all(arr[~mask] == 0)


# ---------------------------------------------------------------------------
# euler_rotation
# ---------------------------------------------------------------------------
class TestEulerRotation:
    def test_identity_at_zero_angles(self):
        np.testing.assert_allclose(euler_rotation(0, 0, 0), np.eye(3), atol=1e-12)

    def test_orthogonal(self):
        R = euler_rotation(0.7, 1.3, -0.2)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_determinant_one(self):
        R = euler_rotation(0.7, 1.3, -0.2)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    def test_z_only_rotation(self):
        R = euler_rotation(np.pi / 2, 0.0, 0.0)
        # ZYZ with theta=0, psi=0 → pure rotation around z by phi.
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# global_rotation
# ---------------------------------------------------------------------------
class TestGlobalRotation:
    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    @pytest.mark.parametrize("I", [1 / 2, 7 / 2])
    def test_identity_at_zero_angle(self, I, axis):
        U = global_rotation(I, 0.0, axis)
        d = int(2 * I + 1)
        assert (U - qt.qeye(d)).norm() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_unitary(self, I, axis):
        U = global_rotation(I, 0.7, axis)
        d = int(2 * I + 1)
        assert (U.dag() * U - qt.qeye(d)).norm() == pytest.approx(0.0, abs=1e-10)

    def test_ndarray_axis_matches_string_axis(self):
        U_str = global_rotation(7 / 2, 0.5, "x")
        U_arr = global_rotation(7 / 2, 0.5, np.array([1.0, 0.0, 0.0]))
        assert (U_str - U_arr).norm() == pytest.approx(0.0, abs=1e-12)

    def test_axis_normalisation(self):
        # An axis with magnitude != 1 is internally normalised, so direction
        # is what matters.
        U_unit = global_rotation(7 / 2, 0.5, np.array([1.0, 0.0, 0.0]))
        U_long = global_rotation(7 / 2, 0.5, np.array([5.0, 0.0, 0.0]))
        assert (U_unit - U_long).norm() == pytest.approx(0.0, abs=1e-12)

    def test_invalid_string_axis_raises(self):
        with pytest.raises(ValueError, match="axis must be"):
            global_rotation(7 / 2, 0.5, "q")

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            global_rotation(7 / 2, 0.5, np.array([0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# subspace_rotation
# ---------------------------------------------------------------------------
class TestSubspaceRotation:
    def test_identity_outside_subspace(self):
        # A π pulse on (m=I, m=I-1) leaves the diagonal entries for indices
        # 2..d-1 unchanged at 1 (identity block) on the unitary.
        I = 7 / 2
        U = subspace_rotation(I, np.pi, "x", (I, I - 1))
        d = int(2 * I + 1)
        for i in range(2, d):
            assert U[i, i] == pytest.approx(1.0, abs=1e-12)
            for j in range(d):
                if i != j:
                    assert U[i, j] == pytest.approx(0.0, abs=1e-12)

    def test_swaps_population_in_subspace(self):
        I = 7 / 2
        U = subspace_rotation(I, np.pi, "x", (I, I - 1))
        d = int(2 * I + 1)
        ket0 = qt.basis(d, 0)
        rotated = U * ket0
        pops = np.abs(rotated.full().flatten()) ** 2
        assert pops[1] == pytest.approx(1.0, abs=1e-10)
        assert pops[0] == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_unitary(self, axis):
        I = 7 / 2
        U = subspace_rotation(I, 0.7, axis, (I, I - 1))
        d = int(2 * I + 1)
        assert (U.dag() * U - qt.qeye(d)).norm() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# snap
# ---------------------------------------------------------------------------
class TestSnap:
    def test_default_dim(self):
        U = snap(np.zeros(8))
        assert U.shape == (8, 8)
        assert (U - qt.qeye(8)).norm() == pytest.approx(0.0, abs=1e-12)

    def test_phases_applied_diagonally(self):
        phases = np.array([0.1, 0.2, 0.3, 0.4])
        U = snap(phases, dim=4)
        diag = np.diag(U.full())
        np.testing.assert_allclose(diag, np.exp(1j * phases), atol=1e-12)

    def test_off_diagonal_zero(self):
        U = snap([0.1, -0.2, 0.7], dim=3)
        arr = U.full()
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert arr[i, j] == pytest.approx(0.0, abs=1e-12)

    def test_unitary(self):
        U = snap([0.1, 0.7, -1.3, 2.0], dim=4)
        assert (U.dag() * U - qt.qeye(4)).norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# shift_operator
# ---------------------------------------------------------------------------
class TestShiftOperator:
    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_unitary(self, I):
        U = shift_operator(I)
        d = int(2 * I + 1)
        assert (U.dag() * U - qt.qeye(d)).norm() == pytest.approx(0.0, abs=1e-12)

    def test_shift_basis_zero(self):
        # U|0> = |d-1> per the implementation (U[d-1, 0] = 1).
        I = 7 / 2
        d = int(2 * I + 1)
        U = shift_operator(I)
        out = U * qt.basis(d, 0)
        expected = qt.basis(d, d - 1)
        assert qt.fidelity(out, expected) == pytest.approx(1.0, abs=1e-12)

    def test_shift_basis_three(self):
        # For column j ≥ 1, U|j> = |j-1>.
        I = 7 / 2
        d = int(2 * I + 1)
        U = shift_operator(I)
        out = U * qt.basis(d, 3)
        expected = qt.basis(d, 2)
        assert qt.fidelity(out, expected) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_dim_applications_is_identity(self, I):
        d = int(2 * I + 1)
        U = shift_operator(I)
        prod = U
        for _ in range(d - 1):
            prod = U * prod
        assert (prod - qt.qeye(d)).norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# permutation_operator
# ---------------------------------------------------------------------------
class TestPermutationOperator:
    def test_swaps_two_basis_states(self):
        I = 7 / 2
        d = int(2 * I + 1)
        U = permutation_operator(0, 3, I=I)
        assert qt.fidelity(U * qt.basis(d, 0), qt.basis(d, 3)) == pytest.approx(
            1.0, abs=1e-12
        )
        assert qt.fidelity(U * qt.basis(d, 3), qt.basis(d, 0)) == pytest.approx(
            1.0, abs=1e-12
        )

    def test_other_states_unchanged(self):
        I = 7 / 2
        d = int(2 * I + 1)
        U = permutation_operator(0, 3, I=I)
        for k in [1, 2, 4, 5, 6, 7]:
            assert qt.fidelity(U * qt.basis(d, k), qt.basis(d, k)) == pytest.approx(
                1.0, abs=1e-12
            )

    def test_self_inverse(self):
        I = 7 / 2
        d = int(2 * I + 1)
        U = permutation_operator(2, 5, I=I)
        assert (U * U - qt.qeye(d)).norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# parity_operator
# ---------------------------------------------------------------------------
class TestParityOperator:
    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_diagonal_alternating_sign(self, I):
        P = parity_operator(I).full()
        d = int(2 * I + 1)
        for i in range(d):
            assert P[i, i] == pytest.approx((-1) ** i)

    def test_squared_is_identity(self):
        P = parity_operator(7 / 2)
        assert (P * P - qt.qeye(8)).norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# global_pi
# ---------------------------------------------------------------------------
class TestGlobalPi:
    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_matches_global_rotation_pi(self, axis):
        U_pi = global_pi(7 / 2, axis=axis)
        U_ref = global_rotation(7 / 2, np.pi, axis)
        assert (U_pi - U_ref).norm() == pytest.approx(0.0, abs=1e-12)

    def test_default_axis_is_x(self):
        assert (global_pi(7 / 2) - global_rotation(7 / 2, np.pi, "x")).norm() == (
            pytest.approx(0.0, abs=1e-12)
        )
