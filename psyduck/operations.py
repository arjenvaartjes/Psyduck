"""Quantum operators and gates for spin systems."""

import qutip as qt
import numpy as np
from typing import Union, Tuple
from numpy import ndarray


# ============================================================================
# Spin operator helpers
# ============================================================================

def get_spin_operators(I: float) -> Tuple[qt.Qobj, qt.Qobj, qt.Qobj]:
    """Get Ix, Iy, Iz spin operators for a given quantum number.

    :param I: Spin quantum number
    :return: Tuple of (Ix, Iy, Iz) operators
    """
    return qt.jmat(I, 'x'), qt.jmat(I, 'y'), qt.jmat(I, 'z')


def get_transition_operators(I: float) -> Tuple[list, list]:
    """Per-transition Ix and Iy operators for a spin-I system.

    Splits Ix and Iy into n = 2I individual transition operators Tx[k], Ty[k],
    each coupling only the k-th neighbouring level pair (level k ↔ level k+1
    in the computational basis, ordered from m = +I downward).

    By construction: sum(Tx) == Ix and sum(Ty) == Iy.

    These are the building blocks of the generalised rotating frame (GRF)
    Hamiltonian — see hamiltonians.drive_hamiltonian.

    :param I: Spin quantum number
    :return: (Tx, Ty) — two lists of length 2I, each element a qt.Qobj
    """
    Ix, Iy, _ = get_spin_operators(I)
    Ix_full = Ix.full()
    Iy_full = Iy.full()
    d = int(2 * I + 1)

    Tx, Ty = [], []
    for k in range(d - 1):
        tx = np.zeros((d, d), dtype=complex)
        tx[k, k + 1] = Ix_full[k, k + 1]
        tx[k + 1, k] = Ix_full[k + 1, k]
        Tx.append(qt.Qobj(tx))

        ty = np.zeros((d, d), dtype=complex)
        ty[k, k + 1] = Iy_full[k, k + 1]
        ty[k + 1, k] = Iy_full[k + 1, k]
        Ty.append(qt.Qobj(ty))

    return Tx, Ty


def euler_rotation(phi, theta, psi):
    """ZYZ Euler rotation matrix (3×3, acts on Cartesian vectors)."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    Rz1 = np.array([[ cph, -sph, 0],
                     [ sph,  cph, 0],
                     [   0,    0, 1]])
    Ry  = np.array([[ cth, 0, sth],
                     [   0, 1,   0],
                     [-sth, 0, cth]])
    Rz2 = np.array([[ cps, -sps, 0],
                     [ sps,  cps, 0],
                     [   0,    0, 1]])
    return Rz1 @ Ry @ Rz2


# ============================================================================
# Unitary operators / gates
# ============================================================================

def global_rotation(I: float, angle: float, axis: Union[str, ndarray]) -> qt.Qobj:
    """Rotation operator exp(-i * angle * I_axis) acting on the full Hilbert space.

    :param I: Spin quantum number
    :param angle: Rotation angle (rad)
    :param axis: 'x', 'y', 'z', or a 3-element array for an arbitrary axis
    :return: Rotation operator (Qobj)
    """
    if isinstance(axis, str):
        if axis not in ['x', 'y', 'z']:
            raise ValueError("axis must be 'x', 'y', 'z', or array-like")
        return (-1j * angle * qt.jmat(I, axis)).expm()

    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Rotation axis cannot be zero vector")
    axis = axis / norm
    Ix, Iy, Iz = get_spin_operators(I)
    I_axis = axis[0] * Ix + axis[1] * Iy + axis[2] * Iz
    return (-1j * angle * I_axis).expm()


def subspace_rotation(I: float, angle: float, axis: Union[str, ndarray], levels: tuple) -> qt.Qobj:
    """Rotation operator restricted to a multi-level subspace.

    Treats the selected levels as a spin-(k-1)/2 system and applies the
    corresponding rotation, leaving all other levels unchanged.
    The two-level case (k=2) is a Givens rotation.

    :param I: Spin quantum number
    :param angle: Rotation angle (rad)
    :param axis: 'x', 'y', 'z', or 3-element array
    :param levels: Tuple of magnetic quantum numbers defining the subspace
    :return: Rotation operator (Qobj)
    """
    d_full = int(2 * I + 1)
    indices = [int(I - m) for m in levels]
    k = len(indices)

    I_sub = (k - 1) / 2
    U_sub = global_rotation(I_sub, angle, axis).full()

    U = np.eye(d_full, dtype=np.complex128)
    U[np.ix_(indices, indices)] = U_sub
    return qt.Qobj(U)


def snap(phases, dim: int = 8) -> qt.Qobj:
    """Selective Number-dependent Arbitrary Phase (SNAP) gate.

    Applies an independent phase shift to each diagonal element.

    :param phases: Array of phase shifts (rad), length dim
    :param dim: Hilbert space dimension (default 8)
    :return: Diagonal unitary (Qobj)
    """
    U = np.eye(dim, dtype=np.complex128)
    for i, p in enumerate(phases):
        U[i, i] = np.exp(1j * p)
    return qt.Qobj(U)


def shift_operator(I: float) -> qt.Qobj:
    """Cyclic shift operator that maps |m⟩ → |m-1⟩ (mod d).

    :param I: Spin quantum number
    :return: Shift operator (Qobj)
    """
    d = int(2 * I + 1)
    U = np.diag(np.ones(d - 1), 1)
    U[d - 1, 0] = 1
    return qt.Qobj(U)


def permutation_operator(element1: int, element2: int, I: float = 7 / 2) -> qt.Qobj:
    """Permutation operator that swaps two basis states.

    :param element1: Index of the first basis state to swap
    :param element2: Index of the second basis state to swap
    :param I: Spin quantum number (default 7/2)
    :return: Permutation operator (Qobj)
    """
    d = int(2 * I + 1)
    U = np.eye(d)
    U[element1, element2] = 1
    U[element2, element1] = 1
    U[element1, element1] = 0
    U[element2, element2] = 0
    return qt.Qobj(U)


def parity_operator(I: float) -> qt.Qobj:
    """Parity operator: flips the sign of odd-indexed basis states.

    :param I: Spin quantum number
    :return: Parity operator (Qobj)
    """
    d = int(2 * I + 1)
    U = np.eye(d)
    for i in range(d):
        if i % 2 == 1:
            U[i, i] = -1
    return qt.Qobj(U)


def global_pi(I: float, axis: str = 'x') -> qt.Qobj:
    """Global π-pulse operator.

    :param I: Spin quantum number
    :param axis: Rotation axis ('x', 'y', or 'z', default 'x')
    :return: π-rotation operator (Qobj)
    """
    return global_rotation(I, np.pi, axis)
