"""Unit tests for psyduck.spin.Spin and inherited SpinInterface methods."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt

from psyduck import Spin, SpinSeries
from psyduck.operations import global_rotation


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_default_arguments(self):
        s = Spin()
        assert s.I == 7 / 2
        assert s.dim == 8
        assert isinstance(s.dim, int)

    @pytest.mark.parametrize("I", [1 / 2, 1, 3 / 2, 5 / 2, 7 / 2])
    def test_dim_matches_I(self, I):
        s = Spin(I)
        assert s.dim == int(2 * I + 1)
        assert isinstance(s.dim, int)

    def test_state_defaults_to_basis_zero(self, spin_sb):
        expected = qt.basis(spin_sb.dim, 0)
        assert qt.fidelity(spin_sb.state, expected) == pytest.approx(1.0)

    def test_custom_state_stored(self):
        psi = qt.basis(8, 3)
        s = Spin(7 / 2, state=psi)
        assert s.state == psi

    def test_dm_property_synced_for_ket(self, spin_sb):
        rho_expected = spin_sb.state * spin_sb.state.dag()
        assert (spin_sb.dm - rho_expected).norm() == pytest.approx(0.0, abs=1e-12)

    def test_dm_property_for_density_matrix_input(self):
        rho = qt.fock_dm(8, 0)
        s = Spin(7 / 2, state=rho)
        assert (s.dm - rho).norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------
class TestRepr:
    def test_repr_format(self, spin_sb):
        assert re.match(r"^Spin\(I=3\.5, dim=8\)$", repr(spin_sb))


# ---------------------------------------------------------------------------
# State labels
# ---------------------------------------------------------------------------
class TestStateLabels:
    def test_half_integer_labels(self, spin_half):
        assert spin_half.state_labels() == ["|1/2>", "|-1/2>"]

    def test_seven_half_labels(self, spin_sb):
        labels = spin_sb.state_labels()
        assert len(labels) == 8
        assert labels[0] == "|7/2>"
        assert labels[-1] == "|-7/2>"


# ---------------------------------------------------------------------------
# Spin operators
# ---------------------------------------------------------------------------
class TestSpinOperators:
    @pytest.mark.parametrize("I", [1 / 2, 7 / 2])
    def test_returns_three_qobjs(self, I):
        Ix, Iy, Iz = Spin(I).get_spin_operators()
        d = int(2 * I + 1)
        for op in (Ix, Iy, Iz):
            assert isinstance(op, qt.Qobj)
            assert op.shape == (d, d)

    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_hermitian(self, I):
        Ix, Iy, Iz = Spin(I).get_spin_operators()
        for op in (Ix, Iy, Iz):
            assert (op - op.dag()).norm() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("I", [1 / 2, 1, 7 / 2])
    def test_commutator_xyz(self, I):
        Ix, Iy, Iz = Spin(I).get_spin_operators()
        commutator = Ix * Iy - Iy * Ix
        assert (commutator - 1j * Iz).norm() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Expectation and Ix / Iy / Iz
# ---------------------------------------------------------------------------
class TestExpectationAndComponents:
    def test_default_iz_is_plus_I(self, spin_any):
        # qt.basis(dim, 0) is the highest-m eigenstate of Iz under QuTiP's
        # convention. (Spin.__init__'s docstring labels this "|I, -I>" but
        # make_eigenstate(I) maps to index 0, confirming this is |I, +I>.)
        assert spin_any.Iz() == pytest.approx(spin_any.I, abs=1e-12)

    def test_default_ix_is_zero(self, spin_any):
        assert abs(spin_any.Ix()) == pytest.approx(0.0, abs=1e-12)

    def test_default_iy_is_zero(self, spin_any):
        assert abs(spin_any.Iy()) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("I", [1 / 2, 7 / 2])
    def test_iz_for_each_eigenstate(self, I):
        s = Spin(I)
        for m in np.arange(-I, I + 1):
            s.make_eigenstate(m)
            assert s.Iz() == pytest.approx(m, abs=1e-12)

    def test_expectation_matches_iz(self, spin_sb):
        _, _, Iz = spin_sb.get_spin_operators()
        assert spin_sb.expectation(Iz) == pytest.approx(spin_sb.Iz(), abs=1e-12)

    def test_expectation_of_identity_is_one(self, spin_any):
        identity = qt.qeye(spin_any.dim)
        assert spin_any.expectation(identity) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# apply_operator
# ---------------------------------------------------------------------------
class TestApplyOperator:
    def test_identity_preserves_state(self, spin_sb):
        before = spin_sb.state.copy()
        spin_sb.apply_operator(qt.qeye(spin_sb.dim))
        assert qt.fidelity(before, spin_sb.state) == pytest.approx(1.0, abs=1e-12)

    def test_pauli_x_flips_qubit(self, spin_half):
        sigma_x = qt.sigmax()
        spin_half.apply_operator(sigma_x)
        assert spin_half.Iz() == pytest.approx(-1 / 2, abs=1e-12)

    def test_returns_self(self, spin_sb):
        result = spin_sb.apply_operator(qt.qeye(spin_sb.dim))
        assert isinstance(result, Spin)

    def test_apply_x_twice_is_identity(self, spin_half):
        before = spin_half.state.copy()
        sigma_x = qt.sigmax()
        spin_half.apply_operator(sigma_x)
        spin_half.apply_operator(sigma_x)
        assert qt.fidelity(before, spin_half.state) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------
class TestCopy:
    def test_copy_preserves_I(self, spin_sb):
        c = spin_sb.copy()
        assert c.I == spin_sb.I
        assert c.dim == spin_sb.dim

    def test_copy_independent_state(self, spin_sb):
        c = spin_sb.copy()
        c.apply_operator(global_rotation(c.I, np.pi, "y"))
        assert spin_sb.Iz() == pytest.approx(spin_sb.I, abs=1e-10)
        assert c.Iz() == pytest.approx(-c.I, abs=1e-10)

    def test_copy_state_is_distinct_object(self, spin_sb):
        c = spin_sb.copy()
        assert c.state is not spin_sb.state


# ---------------------------------------------------------------------------
# make_eigenstate
# ---------------------------------------------------------------------------
class TestMakeEigenstate:
    @pytest.mark.parametrize("I", [1 / 2, 7 / 2])
    def test_iz_matches(self, I):
        s = Spin(I)
        for m in np.arange(-I, I + 1):
            s.make_eigenstate(m)
            assert s.Iz() == pytest.approx(m, abs=1e-12)

    @pytest.mark.parametrize("I", [1 / 2, 7 / 2])
    def test_unit_norm(self, I):
        s = Spin(I)
        for m in np.arange(-I, I + 1):
            s.make_eigenstate(m)
            assert s.state.norm() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# make_zcat_state
# ---------------------------------------------------------------------------
class TestZcatState:
    def test_chaining_returns_self(self, spin_sb):
        assert spin_sb.make_zcat_state(0) is spin_sb

    def test_norm_one(self, spin_sb):
        spin_sb.make_zcat_state(0.7)
        assert spin_sb.state.norm() == pytest.approx(1.0, abs=1e-12)

    def test_amplitudes_phi_zero(self, spin_sb):
        spin_sb.make_zcat_state(0.0)
        amps = spin_sb.state.full().flatten()
        d = spin_sb.dim
        assert abs(amps[0]) == pytest.approx(1 / np.sqrt(2), abs=1e-12)
        assert abs(amps[d - 1]) == pytest.approx(1 / np.sqrt(2), abs=1e-12)
        for i in range(1, d - 1):
            assert abs(amps[i]) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("phi", [0.0, np.pi / 2, np.pi, 1.234])
    def test_iz_is_zero(self, spin_sb, phi):
        spin_sb.make_zcat_state(phi)
        assert abs(spin_sb.Iz()) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# make_xcat_state
# ---------------------------------------------------------------------------
class TestXcatState:
    def test_chaining_returns_self(self, spin_sb):
        assert spin_sb.make_xcat_state(0) is spin_sb

    def test_norm_one(self, spin_sb):
        spin_sb.make_xcat_state(0.7)
        assert spin_sb.state.norm() == pytest.approx(1.0, abs=1e-12)

    def test_iz_zero_and_ix_squared_large(self, spin_sb):
        # An x-cat lives at the antipodes |+x> and |-x>; the linear expectation
        # <Ix> averages to zero, but <Ix^2> ~ I^2 (variance peaked at the poles).
        spin_sb.make_xcat_state(0.0)
        Ix = qt.jmat(spin_sb.I, "x")
        assert abs(spin_sb.Iz()) == pytest.approx(0.0, abs=1e-10)
        assert spin_sb.Ix() == pytest.approx(0.0, abs=1e-10)
        ix_squared = spin_sb.expectation(Ix * Ix)
        assert ix_squared == pytest.approx(spin_sb.I**2, abs=1e-8)


# ---------------------------------------------------------------------------
# make_displaced_coherent_state
# ---------------------------------------------------------------------------
class TestDisplacedCoherentState:
    def test_theta_zero_matches_default(self, spin_sb):
        spin_sb.make_displaced_coherent_state(0.0, 0.0)
        assert spin_sb.Iz() == pytest.approx(spin_sb.I, abs=1e-12)

    def test_theta_pi_is_opposite_pole(self, spin_sb):
        spin_sb.make_displaced_coherent_state(np.pi, 0.0)
        assert spin_sb.Iz() == pytest.approx(-spin_sb.I, abs=1e-10)

    @pytest.mark.parametrize(
        "theta", [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    )
    def test_iz_follows_cos_theta(self, spin_sb, theta):
        spin_sb.make_displaced_coherent_state(theta, 0.0)
        assert spin_sb.Iz() == pytest.approx(spin_sb.I * np.cos(theta), abs=1e-10)

    def test_iz_independent_of_phi(self, spin_sb):
        ref = Spin(spin_sb.I)
        ref.make_displaced_coherent_state(np.pi / 3, 0.0)
        spin_sb.make_displaced_coherent_state(np.pi / 3, 1.7)
        assert spin_sb.Iz() == pytest.approx(ref.Iz(), abs=1e-10)

    def test_norm_preserved(self, spin_sb):
        spin_sb.make_displaced_coherent_state(np.pi / 3, 1.7)
        assert spin_sb.state.norm() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# evolve — closed system
# ---------------------------------------------------------------------------
class TestEvolveClosedSystem:
    def test_returns_spin_series(self, spin_sb):
        Iz = qt.jmat(spin_sb.I, "z")
        series = spin_sb.evolve(Iz, [0.0, 0.5, 1.0])
        assert isinstance(series, SpinSeries)
        assert len(series) == 3
        np.testing.assert_allclose(series.coords, [0.0, 0.5, 1.0])

    def test_single_float_promoted_to_two_point_grid(self, spin_sb):
        Iz = qt.jmat(spin_sb.I, "z")
        series = spin_sb.evolve(Iz, 0.5)
        np.testing.assert_allclose(series.coords, [0.0, 0.5])

    def test_unsorted_input_is_sorted(self, spin_sb):
        Iz = qt.jmat(spin_sb.I, "z")
        series = spin_sb.evolve(Iz, [0.0, 1.0, 0.5])
        np.testing.assert_allclose(series.coords, [0.0, 0.5, 1.0])

    def test_missing_zero_is_prepended(self, spin_sb):
        Iz = qt.jmat(spin_sb.I, "z")
        series = spin_sb.evolve(Iz, [0.5, 1.0])
        np.testing.assert_allclose(series.coords, [0.0, 0.5, 1.0])

    def test_state_updated_to_final(self, spin_sb):
        Iz = qt.jmat(spin_sb.I, "z")
        series = spin_sb.evolve(Iz, [0.0, 1.0])
        assert qt.fidelity(spin_sb.state, series.states[-1]) == pytest.approx(
            1.0, abs=1e-12
        )

    def test_iz_eigenstate_evolution_under_iz_preserves_iz(self, spin_sb):
        # |+I> is an eigenstate of Iz, so evolving under H = Iz only adds a
        # global phase — <Iz> is unchanged.
        Iz = qt.jmat(spin_sb.I, "z")
        spin_sb.evolve(Iz, [0.0, 1.7])
        assert spin_sb.Iz() == pytest.approx(spin_sb.I, abs=1e-10)

    def test_precession_under_iz(self, spin_sb):
        # Coherent state in the xy-plane precesses around z. <Iz> is conserved,
        # and Ix^2 + Iy^2 magnitude is conserved.
        spin_sb.make_displaced_coherent_state(np.pi / 2, 0.0)
        ix0, iy0 = spin_sb.Ix(), spin_sb.Iy()
        Iz = qt.jmat(spin_sb.I, "z")
        spin_sb.evolve(Iz, [0.0, np.pi / 4])
        assert abs(spin_sb.Iz()) == pytest.approx(0.0, abs=1e-8)
        r_after = np.hypot(spin_sb.Ix(), spin_sb.Iy())
        r_before = np.hypot(ix0, iy0)
        assert r_after == pytest.approx(r_before, abs=1e-8)

    def test_two_pi_evolution_under_iz_returns_initial(self, spin_sb):
        spin_sb.make_displaced_coherent_state(np.pi / 2, 0.0)
        before = spin_sb.state.copy()
        Iz = qt.jmat(spin_sb.I, "z")
        spin_sb.evolve(Iz, [0.0, 2 * np.pi])
        # Fidelity is insensitive to the global phase that arises for half-int I.
        assert qt.fidelity(before, spin_sb.state) == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# evolve — open system
# ---------------------------------------------------------------------------
class TestEvolveOpenSystem:
    def test_empty_c_ops_matches_closed(self, spin_sb):
        Iz = qt.jmat(spin_sb.I, "z")
        s_open = spin_sb.copy()
        spin_sb.evolve(Iz, [0.0, 1.0])
        s_open.evolve(Iz, [0.0, 1.0], c_ops=[])
        assert qt.fidelity(spin_sb.state, s_open.state) == pytest.approx(
            1.0, abs=1e-10
        )

    def test_dephasing_makes_state_mixed(self, spin_sb):
        spin_sb.make_displaced_coherent_state(np.pi / 2, 0.0)
        Iz = qt.jmat(spin_sb.I, "z")
        spin_sb.evolve(Iz, [0.0, 5.0], c_ops=[0.5 * Iz])
        rho = spin_sb.state
        purity = (rho * rho).tr().real
        assert purity < 1.0

    def test_open_evolution_state_is_density_matrix(self, spin_sb):
        Iz = qt.jmat(spin_sb.I, "z")
        spin_sb.evolve(Iz, [0.0, 1.0], c_ops=[0.1 * Iz])
        assert spin_sb.state.isoper


# ---------------------------------------------------------------------------
# plot_wigner
# ---------------------------------------------------------------------------
@pytest.fixture
def close_figures():
    yield
    plt.close("all")


class TestPlotWigner:
    # Return arities differ by projection (some wigner plots return
    # (fig, ax, pcm)), so unpack only the first two.

    def test_3d_projection_runs(self, spin_sb, close_figures):
        result = spin_sb.plot_wigner(projection="3d")
        assert result[0] is not None
        assert result[1] is not None

    def test_hammer_projection_runs(self, spin_sb, close_figures):
        result = spin_sb.plot_wigner(projection="hammer")
        assert result[0] is not None
        assert result[1] is not None

    def test_polar_projection_runs(self, spin_sb, close_figures):
        result = spin_sb.plot_wigner(projection="polar")
        assert result[0] is not None
        assert result[1] is not None

    def test_invalid_projection_raises(self, spin_sb):
        with pytest.raises(ValueError, match="projection must be"):
            spin_sb.plot_wigner(projection="invalid")


# ---------------------------------------------------------------------------
# Inherited rotations and shift
# ---------------------------------------------------------------------------
class TestInheritedRotations:
    def test_global_rotate_returns_self(self, spin_sb):
        assert spin_sb.global_rotate(0.1, "z") is spin_sb

    def test_2pi_z_rotation_preserves_state(self, spin_sb):
        before = spin_sb.state.copy()
        spin_sb.global_rotate(2 * np.pi, "z")
        assert qt.fidelity(before, spin_sb.state) == pytest.approx(1.0, abs=1e-10)

    def test_pi_y_rotation_inverts_iz(self, spin_sb):
        spin_sb.global_rotate(np.pi, "y")
        assert spin_sb.Iz() == pytest.approx(-spin_sb.I, abs=1e-10)

    def test_global_rotate_accepts_ndarray_axis(self, spin_sb):
        before_iz = spin_sb.Iz()
        spin_sb.global_rotate(np.pi / 3, np.array([1.0, 0.0, 0.0]))
        # Rotation around x by π/3 should change <Iz> from I to I·cos(π/3).
        assert spin_sb.Iz() == pytest.approx(spin_sb.I * np.cos(np.pi / 3), abs=1e-10)
        assert spin_sb.Iz() != pytest.approx(before_iz, abs=1e-3)

    def test_subspace_pi_x_swaps_top_two(self, spin_sb):
        # Default state has all population at index 0 (high-m). A π pulse
        # around x in the (I, I-1) subspace should move it entirely to index 1.
        I = spin_sb.I
        spin_sb.subspace_rotate(np.pi, "x", (I, I - 1))
        pops = np.abs(spin_sb.state.full().flatten()) ** 2
        assert pops[1] == pytest.approx(1.0, abs=1e-10)
        for i in [0] + list(range(2, spin_sb.dim)):
            assert pops[i] == pytest.approx(0.0, abs=1e-10)

    def test_shift_returns_self(self, spin_sb):
        assert spin_sb.shift() is spin_sb

    def test_shift_dim_times_is_identity(self, spin_sb):
        before = spin_sb.state.copy()
        for _ in range(spin_sb.dim):
            spin_sb.shift()
        assert qt.fidelity(before, spin_sb.state) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# parity, fidelity, linear_entropy
# ---------------------------------------------------------------------------
class TestParityAndFidelityAndEntropy:
    def test_parity_default_state(self, spin_sb):
        # Default state is basis-index 0 (even) → parity = +1.
        assert spin_sb.parity() == pytest.approx(1.0, abs=1e-12)

    def test_parity_odd_index_is_minus_one(self, spin_sb):
        # basis-index 1 ↔ eigenvalue m = I - 1.
        spin_sb.make_eigenstate(spin_sb.I - 1)
        assert spin_sb.parity() == pytest.approx(-1.0, abs=1e-12)

    def test_fidelity_with_self_is_one(self, spin_sb):
        assert spin_sb.fidelity(spin_sb.state) == pytest.approx(1.0, abs=1e-12)

    def test_fidelity_with_orthogonal_is_zero(self, spin_sb):
        other = qt.basis(spin_sb.dim, spin_sb.dim - 1)
        assert spin_sb.fidelity(other) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("m", [-7 / 2, -1 / 2, 7 / 2])
    def test_linear_entropy_pure_eigenstate_is_zero(self, spin_sb, m):
        spin_sb.make_eigenstate(m)
        assert spin_sb.linear_entropy() == pytest.approx(0.0, abs=1e-12)

    def test_linear_entropy_pure_coherent_is_zero(self, spin_sb):
        spin_sb.make_displaced_coherent_state(0.7, 1.3)
        assert spin_sb.linear_entropy() == pytest.approx(0.0, abs=1e-12)
