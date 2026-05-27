"""Unit tests for psyduck.cat_qubit.CatQubit."""

import numpy as np
import pytest
import qutip as qt

from psyduck import CatQubit


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_default(self):
        cq = CatQubit()
        assert cq.I == 7 / 2
        assert cq.dim == 8
        assert cq.axis == "Ix"
        assert cq.state.norm() == pytest.approx(1.0, abs=1e-12)

    def test_custom_axis(self):
        cq = CatQubit(I=7 / 2, axis="Iz")
        assert cq.axis == "Iz"

    @pytest.mark.parametrize("I", [3 / 2, 5 / 2, 7 / 2])
    def test_half_integer_I(self, I):
        cq = CatQubit(I=I)
        assert cq.dim == int(2 * I + 1)

    def test_integer_I_rejected(self):
        # Implementation asserts I is half-integer.
        with pytest.raises(AssertionError):
            CatQubit(I=1)

    def test_default_state_is_basis_zero(self):
        cq = CatQubit()
        assert qt.fidelity(cq.state, qt.basis(cq.dim, 0)) == pytest.approx(
            1.0, abs=1e-12
        )

    def test_state_list_initially_empty(self):
        assert CatQubit().state_list == []


# ---------------------------------------------------------------------------
# state / dm sync (inherited from SpinInterface)
# ---------------------------------------------------------------------------
class TestStateDmSync:
    def test_dm_synced_with_ket(self):
        cq = CatQubit()
        rho_expected = cq.state * cq.state.dag()
        assert (cq.dm - rho_expected).norm() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# spin_op dictionary
# ---------------------------------------------------------------------------
class TestSpinOpDict:
    def test_keys_present(self):
        cq = CatQubit()
        for k in ("Ix", "Iy", "Iz", "Ix^2", "Iy^2", "Iz^2", "Ip", "Im",
                  "I2AT", "parity", "Lx_z", "Ly_z", "Lz_z"):
            assert k in cq.spin_op

    def test_parity_diagonal_alternates(self):
        cq = CatQubit()
        diag = np.diag(cq.spin_op["parity"].full())
        for i, val in enumerate(diag):
            assert val == pytest.approx((-1) ** i)

    def test_Lx_z_is_anti_diagonal(self):
        cq = CatQubit()
        Lx_z = cq.spin_op["Lx_z"].full()
        # np.flip(eye, 1) → anti-diagonal of 1s.
        assert np.allclose(np.diag(np.flip(Lx_z, 1)), 1.0)


# ---------------------------------------------------------------------------
# initialize / save_state / clear_state_list
# ---------------------------------------------------------------------------
class TestInitializeAndStateList:
    def test_initialize_to_basis(self):
        cq = CatQubit()
        cq.initialize(mz=3)
        assert qt.fidelity(cq.state, qt.basis(cq.dim, 3)) == pytest.approx(
            1.0, abs=1e-12
        )

    def test_save_state(self):
        cq = CatQubit()
        cq.save_state()
        cq.initialize(mz=2)
        cq.save_state()
        assert len(cq.state_list) == 2

    def test_clear_state_list(self):
        cq = CatQubit()
        cq.save_state()
        cq.clear_state_list()
        assert cq.state_list == []


# ---------------------------------------------------------------------------
# encode / decode round-trip
# ---------------------------------------------------------------------------
class TestEncodeDecode:
    @pytest.mark.parametrize("axis", ["Ix", "Iy", "Iz"])
    def test_round_trip_preserves_state(self, axis):
        cq = CatQubit(axis=axis)
        before = cq.state.copy()
        cq.encode()
        cq.decode()
        assert qt.fidelity(before, cq.state) == pytest.approx(1.0, abs=1e-10)

    def test_iz_axis_no_rotation(self):
        cq = CatQubit(axis="Iz")
        before = cq.state.copy()
        cq.encode()
        # For Iz axis, encode does NOT rotate; only sets logical Paulis.
        assert qt.fidelity(before, cq.state) == pytest.approx(1.0, abs=1e-12)

    def test_logical_paulis_built_after_encode(self):
        cq = CatQubit(axis="Ix")
        cq.encode()
        for k in ("Lx", "Ly", "Lz"):
            assert k in cq.spin_op


# ---------------------------------------------------------------------------
# Logical gates
# ---------------------------------------------------------------------------
class TestLogicalGates:
    def test_LRx_zero_is_identity(self):
        cq = CatQubit()
        before = cq.state.copy()
        cq.LRx_gate(0.0)
        assert qt.fidelity(before, cq.state) == pytest.approx(1.0, abs=1e-12)

    def test_LRz_bias_preserving_zero_is_identity(self):
        cq = CatQubit()
        cq.encode()
        before = cq.state.copy()
        cq.LRz_gate(0.0, bias_preserving=True)
        assert qt.fidelity(before, cq.state) == pytest.approx(1.0, abs=1e-10)

    def test_LRz_non_bias_preserving_zero_is_identity(self):
        cq = CatQubit()
        before = cq.state.copy()
        cq.LRz_gate(0.0, bias_preserving=False)
        assert qt.fidelity(before, cq.state) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# state_populations
# ---------------------------------------------------------------------------
class TestStatePopulations:
    def test_z_basis_default_state(self):
        cq = CatQubit()
        pops = cq.state_populations(basis="z")
        assert pops[0] == pytest.approx(1.0, abs=1e-12)
        assert pops[1:].sum() == pytest.approx(0.0, abs=1e-12)

    def test_returns_real_array_summing_to_one(self):
        cq = CatQubit()
        for basis in ("x", "y", "z"):
            pops = cq.state_populations(basis=basis)
            assert pops.sum() == pytest.approx(1.0, abs=1e-10)

    def test_invalid_basis_raises(self):
        cq = CatQubit()
        with pytest.raises(ValueError, match="basis must be"):
            cq.state_populations(basis="q")


# ---------------------------------------------------------------------------
# logic_exp_values
# ---------------------------------------------------------------------------
class TestLogicExpValues:
    def test_returns_three_values(self):
        cq = CatQubit()
        cq.encode()
        vals = cq.logic_exp_values()
        assert len(vals) == 3

    def test_iz_basis_keys(self):
        cq = CatQubit()
        # Iz_basis=True uses Lx_z/Ly_z/Lz_z which are always present.
        vals = cq.logic_exp_values(Iz_basis=True)
        assert len(vals) == 3


# ---------------------------------------------------------------------------
# Phase-error gates
# ---------------------------------------------------------------------------
class TestPhaseError:
    def test_zero_phase_no_change(self):
        cq = CatQubit()
        before = cq.state.copy()
        cq.phase_error(Iz=0.0, Iz_sq=0.0)
        assert qt.fidelity(before, cq.state) == pytest.approx(1.0, abs=1e-12)

    def test_phase_error_preserves_norm(self):
        cq = CatQubit()
        cq.encode()
        cq.phase_error(Iz=0.5, Iz_sq=0.2)
        assert cq.state.norm() == pytest.approx(1.0, abs=1e-10)

    def test_random_phase_error_with_zero_range_no_change(self):
        cq = CatQubit()
        before = cq.state.copy()
        cq.random_phase_error(Iz_range=0.0, Iz_sq_range=0.0)
        assert qt.fidelity(before, cq.state) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Plotting dispatch
# ---------------------------------------------------------------------------
@pytest.fixture
def close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


class TestPlotWigner:
    def test_invalid_projection_raises(self):
        cq = CatQubit()
        with pytest.raises(ValueError, match="projection must be"):
            cq.plot_wigner(projection="invalid")

    def test_hammer_runs(self, close_figures):
        cq = CatQubit()
        result = cq.plot_wigner(projection="hammer")
        assert result[0] is not None  # fig
