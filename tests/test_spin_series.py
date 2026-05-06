"""Unit tests for psyduck.spin_series.SpinSeries."""

import numpy as np
import pytest
import qutip as qt

from psyduck import Spin, SpinSeries
from psyduck.operations import global_rotation


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_basic_construction(self):
        I = 7 / 2
        d = int(2 * I + 1)
        states = [qt.basis(d, k) for k in range(3)]
        coords = np.array([0.0, 1.0, 2.0])
        s = SpinSeries(states, I=I, coords=coords)
        assert s.I == I
        assert s.dim == d
        assert len(s.states) == 3
        np.testing.assert_array_equal(s.coords, coords)
        assert s.result is None

    def test_coords_none_yields_none(self):
        s = SpinSeries([qt.basis(8, 0)], I=7 / 2, coords=None)
        assert s.coords is None

    def test_states_copied_into_list(self):
        # Constructor wraps the input in list(...). Mutating the original
        # iterable should not affect the stored list.
        original = [qt.basis(8, 0), qt.basis(8, 1)]
        s = SpinSeries(original, I=7 / 2)
        original.append(qt.basis(8, 2))
        assert len(s.states) == 2

    def test_from_result(self):
        I = 7 / 2
        d = int(2 * I + 1)
        H = qt.qzero(d)
        psi0 = qt.basis(d, 0)
        times = np.linspace(0, 1.0, 4)
        result = qt.sesolve(H, psi0, times)
        s = SpinSeries.from_result(result, I=I, coords=times)
        assert isinstance(s, SpinSeries)
        assert s.result is result
        assert len(s.states) == len(times)


# ---------------------------------------------------------------------------
# expectation
# ---------------------------------------------------------------------------
class TestExpectation:
    def test_returns_ndarray_with_one_value_per_state(self):
        I = 7 / 2
        d = int(2 * I + 1)
        states = [qt.basis(d, k) for k in range(d)]
        s = SpinSeries(states, I=I, coords=np.arange(d))
        Iz = qt.jmat(I, "z")
        out = s.expectation(Iz)
        assert isinstance(out, np.ndarray)
        assert out.shape == (d,)

    def test_iz_eigenvalues_recovered(self):
        I = 7 / 2
        d = int(2 * I + 1)
        # basis(dim, k) is an eigenstate of Iz with eigenvalue I-k.
        states = [qt.basis(d, k) for k in range(d)]
        s = SpinSeries(states, I=I)
        Iz = qt.jmat(I, "z")
        np.testing.assert_allclose(s.expectation(Iz),
                                   [I - k for k in range(d)],
                                   atol=1e-12)


# ---------------------------------------------------------------------------
# fidelity
# ---------------------------------------------------------------------------
class TestFidelity:
    def test_self_fidelity_one(self):
        I = 7 / 2
        d = int(2 * I + 1)
        target = qt.basis(d, 0)
        states = [target] * 5
        s = SpinSeries(states, I=I)
        np.testing.assert_allclose(s.fidelity(target), 1.0, atol=1e-12)

    def test_orthogonal_fidelity_zero(self):
        I = 7 / 2
        d = int(2 * I + 1)
        states = [qt.basis(d, 0), qt.basis(d, d - 1)]
        s = SpinSeries(states, I=I)
        out = s.fidelity(qt.basis(d, 0))
        assert out[0] == pytest.approx(1.0, abs=1e-12)
        assert out[1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# apply_operator
# ---------------------------------------------------------------------------
class TestApplyOperator:
    def test_pauli_x_flips_each_state(self):
        s = SpinSeries([qt.basis(2, 0), qt.basis(2, 0)], I=1 / 2)
        s.apply_operator(qt.sigmax())
        for state in s.states:
            assert qt.fidelity(state, qt.basis(2, 1)) == pytest.approx(1.0, abs=1e-12)

    def test_state_count_preserved(self):
        s = SpinSeries([qt.basis(8, 0)] * 3, I=7 / 2)
        s.apply_operator(global_rotation(7 / 2, np.pi / 5, "z"))
        assert len(s.states) == 3


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------
class TestCopy:
    def test_independent_state_list(self):
        s = SpinSeries([qt.basis(8, 0), qt.basis(8, 1)], I=7 / 2,
                       coords=np.array([0.0, 1.0]))
        c = s.copy()
        c.states.append(qt.basis(8, 2))
        assert len(s.states) == 2
        assert len(c.states) == 3

    def test_independent_coords(self):
        s = SpinSeries([qt.basis(8, 0)], I=7 / 2, coords=np.array([0.0]))
        c = s.copy()
        c.coords[0] = 99.0
        assert s.coords[0] == 0.0


# ---------------------------------------------------------------------------
# populations
# ---------------------------------------------------------------------------
class TestPopulations:
    def test_kets_give_amplitude_squared(self):
        I = 7 / 2
        d = int(2 * I + 1)
        states = [qt.basis(d, 0), qt.basis(d, 3)]
        s = SpinSeries(states, I=I)
        pops = s.populations()
        assert pops.shape == (2, d)
        # First state: all population at index 0.
        assert pops[0, 0] == pytest.approx(1.0)
        # Second state: all population at index 3.
        assert pops[1, 3] == pytest.approx(1.0)
        assert pops[0, 1:].sum() == pytest.approx(0.0, abs=1e-12)

    def test_density_matrix_uses_diagonal(self):
        I = 7 / 2
        d = int(2 * I + 1)
        # Maximally mixed dm: every diagonal entry = 1/d.
        rho = qt.maximally_mixed_dm(d)
        s = SpinSeries([rho, rho], I=I)
        pops = s.populations()
        assert pops.shape == (2, d)
        np.testing.assert_allclose(pops, 1.0 / d, atol=1e-12)

    def test_populations_sum_to_one_per_row(self):
        I = 7 / 2
        d = int(2 * I + 1)
        states = [(qt.basis(d, 0) + qt.basis(d, d - 1)).unit()]
        s = SpinSeries(states, I=I)
        pops = s.populations()
        assert pops[0].sum() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# __len__, __getitem__, __repr__
# ---------------------------------------------------------------------------
class TestDunderMethods:
    def test_len(self):
        s = SpinSeries([qt.basis(8, 0)] * 4, I=7 / 2)
        assert len(s) == 4

    def test_getitem_returns_spin(self):
        I = 7 / 2
        d = int(2 * I + 1)
        states = [qt.basis(d, k) for k in range(d)]
        s = SpinSeries(states, I=I)
        spin_at_2 = s[2]
        assert isinstance(spin_at_2, Spin)
        assert spin_at_2.I == I
        # Iz expectation at index 2 corresponds to eigenvalue I - 2.
        assert spin_at_2.Iz() == pytest.approx(I - 2, abs=1e-12)

    def test_repr_contains_I_and_N(self):
        s = SpinSeries([qt.basis(8, 0)] * 3, I=7 / 2,
                       coords=np.array([0.0, 0.5, 1.0]))
        text = repr(s)
        assert "SpinSeries" in text
        assert "I=3.5" in text
        assert "N=3" in text
