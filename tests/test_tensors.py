"""Unit tests for psyduck.tensors."""

import numpy as np
import pytest

from psyduck.tensors import (
    Qab_to_Vab,
    Vab_to_Qab,
    get_Q_tensor,
    get_R_tensor,
    get_S_tensor,
    tensor_to_voigt,
    voigt_to_tensor,
)


# ---------------------------------------------------------------------------
# get_Q_tensor
# ---------------------------------------------------------------------------
class TestGetQTensor:
    def test_axial_paf_matches_textbook(self):
        f_q = 25e3
        Q = get_Q_tensor(f_q, eta=0.0)
        # In PAF and eta=0, V_PAF = diag([-0.5, -0.5, 1]), so Q = (f_q/3)*V_PAF.
        expected = (f_q / 3.0) * np.diag([-0.5, -0.5, 1.0])
        np.testing.assert_allclose(Q, expected, atol=1e-12)

    def test_traceless(self):
        Q = get_Q_tensor(25e3, eta=0.3, theta=0.7, phi=1.1, psi=-0.2)
        assert np.trace(Q) == pytest.approx(0.0, abs=1e-12)

    def test_symmetric(self):
        Q = get_Q_tensor(25e3, eta=0.5, theta=0.4, phi=2.0, psi=0.6)
        np.testing.assert_allclose(Q, Q.T, atol=1e-12)

    def test_eta_one_gives_distinct_principal_values(self):
        # eta=1: V_PAF = diag([0, -1, 1]) (entries -(1-eta)/2, -(1+eta)/2, 1).
        Q = get_Q_tensor(30e3, eta=1.0)
        expected = (30e3 / 3.0) * np.diag([0.0, -1.0, 1.0])
        np.testing.assert_allclose(Q, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# get_S_tensor / get_R_tensor (just shape + sanity)
# ---------------------------------------------------------------------------
class TestStaticTensors:
    def test_S_tensor_shape(self):
        assert get_S_tensor().shape == (6, 6)

    def test_S_tensor_default_values(self):
        # S11=2e22, S44=5.9e22 → top-left = S11/4 + S44.
        S = get_S_tensor()
        assert S[0, 0] == pytest.approx(2e22 / 4 + 5.9e22)

    def test_R_tensor_shape(self):
        assert get_R_tensor().shape == (6, 3)

    def test_R_tensor_default_R14(self):
        R = get_R_tensor()
        assert R[0, 1] == pytest.approx(-1.7e12)
        assert R[2, 1] == pytest.approx(1.7e12)


# ---------------------------------------------------------------------------
# voigt_to_tensor
# ---------------------------------------------------------------------------
class TestVoigtToTensor:
    def test_single_vector(self):
        v = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
        T = voigt_to_tensor(v)
        assert T.shape == (3, 3)
        # Diagonal: xx, yy, zz.
        assert T[0, 0] == 1.0
        assert T[1, 1] == 2.0
        assert T[2, 2] == 3.0
        # Off-diagonal: yz, xz, xy.
        assert T[1, 2] == T[2, 1] == 0.4
        assert T[0, 2] == T[2, 0] == 0.5
        assert T[0, 1] == T[1, 0] == 0.6

    def test_symmetric(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        T = voigt_to_tensor(v)
        np.testing.assert_array_equal(T, T.T)

    def test_batched(self):
        V = np.array([
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [4.0, 5.0, 6.0, 0.7, 0.8, 0.9],
        ])
        T = voigt_to_tensor(V)
        assert T.shape == (2, 3, 3)
        assert T[0, 0, 0] == 1.0
        assert T[1, 0, 1] == 0.9


# ---------------------------------------------------------------------------
# tensor_to_voigt
# ---------------------------------------------------------------------------
class TestTensorToVoigt:
    def test_round_trip_single(self):
        v = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
        v_back = tensor_to_voigt(voigt_to_tensor(v))
        np.testing.assert_allclose(v_back, v, atol=1e-12)

    def test_round_trip_batched(self):
        V = np.random.default_rng(0).normal(size=(5, 6))
        V_back = tensor_to_voigt(voigt_to_tensor(V))
        np.testing.assert_allclose(V_back, V, atol=1e-12)

    def test_shape_single(self):
        T = np.eye(3)
        assert tensor_to_voigt(T).shape == (6,)

    def test_shape_batched(self):
        T = np.zeros((4, 3, 3))
        assert tensor_to_voigt(T).shape == (4, 6)


# ---------------------------------------------------------------------------
# Vab_to_Qab and Qab_to_Vab
# ---------------------------------------------------------------------------
class TestVabQabConversion:
    def test_round_trip_single(self):
        rng = np.random.default_rng(1)
        V = rng.normal(size=(3, 3))
        V_back = Qab_to_Vab(Vab_to_Qab(V, I=7 / 2, Q=2e-29), I=7 / 2, Q=2e-29)
        np.testing.assert_allclose(V_back, V, atol=1e-12)

    def test_round_trip_batched(self):
        rng = np.random.default_rng(2)
        V = rng.normal(size=(3, 3, 3))
        V_back = Qab_to_Vab(Vab_to_Qab(V, I=7 / 2, Q=2e-29), I=7 / 2, Q=2e-29)
        np.testing.assert_allclose(V_back, V, atol=1e-12)

    def test_scaling_formula(self):
        # Q_ab = e * Q * V_ab / (2I (2I-1) h)
        I = 7 / 2
        Q = 2e-29
        e = 1.6e-19
        h = 6.626e-34
        V = np.eye(3)
        Q_ab = Vab_to_Qab(V, I, Q)
        scale = e * Q / (2 * I * (2 * I - 1) * h)
        np.testing.assert_allclose(Q_ab, scale * V, atol=1e-30)
