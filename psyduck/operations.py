"""Quantum operations for spin systems (Hamiltonians, gates, etc.)."""

import qutip as qt
import numpy as np
from typing import Union, Tuple
from numpy import ndarray
# ============================================================================
# Helper Functions
# ============================================================================

def get_spin_operators(I: float) -> Tuple[qt.Qobj, qt.Qobj, qt.Qobj]:
    """Get Ix, Iy, Iz spin operators for a given quantum number.

    :param I: Spin quantum number
    :return: Tuple of (Ix, Iy, Iz) operators
    """
    return qt.jmat(I, 'x'), qt.jmat(I, 'y'), qt.jmat(I, 'z')


# ============================================================================
# Hamiltonian Functions
# ============================================================================

def zeeman_hamiltonian(I: float, B0: float, gamma: float = 1.0) -> qt.Qobj:
    """Zeeman Hamiltonian for a spin in a magnetic field.
    
    H = -gamma * B0 * Iz
    
    :param I: Spin quantum number
    :param B0: Magnetic field strength
    :param gamma: Gyromagnetic ratio (default 1.0)
    :return: Zeeman Hamiltonian
    """
    _, _, Iz = get_spin_operators(I) 
    return -gamma * B0 * Iz


def quadrupole_hamiltonian(I: float, V_ab: ndarray, Q: float,
                           e: float = 1.6e-19, h: float = 6.626e-34) -> qt.Qobj:
    """Quadrupole Hamiltonian for a nuclear spin.

    H = sum_{a,b} Q_ab[a,b] * I_a * I_b,  where Q_ab = e*Q*V_ab / (2I(2I-1)*h)

    :param I: Nuclear spin quantum number
    :param V_ab: 3x3 EFG tensor in SI units (V/m²)
    :param Q: Nuclear quadrupole moment (C·m²)
    :param e: Elementary charge (default 1.6e-19 C)
    :param h: Planck constant (default 6.626e-34 J·s)
    :return: Quadrupole Hamiltonian in Hz
    """
    Ix, Iy, Iz = get_spin_operators(I)
    I_vec = [Ix, Iy, Iz]
    Q_ab = e * Q * V_ab / (2 * I * (2 * I - 1)) / h
    H_quad = 0
    for alpha in range(3):
        for beta in range(3):
            H_quad += Q_ab[alpha, beta] * I_vec[alpha] * I_vec[beta]
    return qt.Qobj(H_quad)

def hyperfine_hamiltonian(S: float, I: float, A: float) -> qt.Qobj:
    """Hyperfine interaction Hamiltonian for electron and nuclear spins.
    
    H = A * S . I
    
    :param S: Electron spin quantum number
    :param I: Nuclear spin quantum number
    :param A: Hyperfine coupling constant
    :return: Hyperfine Hamiltonian
    """
    Sx, Sy, Sz = get_spin_operators(S)
    Ix, Iy, Iz = get_spin_operators(I)
    
    H_hyperfine = A * (qt.tensor(Sx, Ix) + qt.tensor(Sy, Iy) + qt.tensor(Sz, Iz))
    return H_hyperfine

def nmr1_hamiltonian(I: float, B1: Union[float, list, ndarray],
                   axis: str = 'x', gamma: float = 1.0) -> qt.Qobj:
    """Radiofrequency field Hamiltonian in the rotating frame.
    
    H = -gamma * B1 * I_axis * cos(omega_rf)
    
    Note: This returns the time-independent envelope for use with time-dependent solvers.
    
    :param I: Spin quantum number
    :param B1: RF field strength
    :param axis: Rotation axis ('x' or 'y')
    :param gamma: Gyromagnetic ratio (default 1.0)
    :return: RF Hamiltonian amplitude
    """
    I_axis = qt.jmat(I, axis)
    if isinstance(B1, float):
        return -gamma * B1 * I_axis
    else:
        H_nmr = np.zeros((8, 8), dtype=complex)
        for i in range(len(B1)):
            H_nmr[i, i+1] = -gamma * I_axis[i, i+1] * B1[i]
            H_nmr[i+1, i] = -gamma * I_axis[i+1, i] * B1[i]
        return qt.Qobj(H_nmr)


# ============================================================================
# Unitary Operators
# ============================================================================

def global_rotation(I: float, angle: float, axis: Union[str, ndarray]) -> qt.Qobj:
    """Rotation operator around arbitrary axis or principal axis.
    
    :param I: Spin quantum number
    :param angle: Rotation angle in radians
    :param axis: Rotation axis - 'x', 'y', 'z', or 3-element array
    :return: Rotation operator
    """
    if isinstance(axis, str):
        if axis not in ['x', 'y', 'z']:
            raise ValueError("axis must be 'x', 'y', 'z', or array-like")
        I_axis = qt.jmat(I, axis)
        U = (-1j * angle * I_axis).expm()
    else:
        # Arbitrary axis
        axis = np.array(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ValueError("Rotation axis cannot be zero vector")
        axis = axis / norm
        
        Ix, Iy, Iz = get_spin_operators(I)
        
        I_axis = axis[0]*Ix + axis[1]*Iy + axis[2]*Iz
        U = (-1j * angle * I_axis).expm()
    
    return U

def subspace_rotation(I: float, angle: float, axis: Union[str, ndarray], levels: tuple) -> qt.Qobj:
    """Rotation operator in a multi-level subspace.

    Treats the k selected levels as a spin-(k-1)/2 system and applies the
    corresponding rotation, leaving all other levels unchanged.
    The two-level case (k=2) is a Givens rotation.

    :param I: Spin quantum number
    :param angle: Rotation angle in radians
    :param axis: Rotation axis - 'x', 'y', 'z', or 3-element array
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

def snap(phases, dim: int=8):
    """
    Generates a unitary matrix that applies a phase shift to each diagonal element of the matrix.

    Parameters:
    phases (array-like): Array of phase shifts to be applied.
    d (int, optional): The dimension of the matrix. Default is 8.

    Returns:
    numpy.ndarray: A unitary matrix with the specified phase shifts applied.
    """
    U = np.eye(dim, dtype=np.complex128)
    for i, p in enumerate(phases):
        U[i, i] = np.exp(1j*p)
    return qt.Qobj(U)

def shift_operator(I: float):
    d = int(2 * I + 1)
    U = np.diag(np.ones(d-1), 1)
    U[d-1, 0] = 1
    return qt.Qobj(U)

def permutation_operator(element1: int, element2: int, I: float = 7/2) -> qt.Qobj:
    """
    Generates a permutation matrix that swaps two elements in a matrix.

    Parameters:
    element1 (int): Index of the first element to swap.
    element2 (int): Index of the second element to swap.
    d (int, optional): The dimension of the matrix. Default is 8.

    Returns:
    : A permutation matrix that swaps the elements at indices element1 and element2.
    """
    d = int(2 * I + 1)
    U = np.eye(d)
    U[element1, element2] = 1
    U[element2, element1] = 1
    U[element1, element1] = 0
    U[element2, element2] = 0
    return qt.Qobj(U)

def parity_operator(I: float) -> qt.Qobj:
    """
    Generates a parity operator that flips the sign of the odd elements in a matrix.

    Returns:
    : A parity operator that flips the sign of the odd elements in a matrix.
    """
    d = int(2 * I + 1)
    U = np.eye(d)
    for i in range(d):
        if i % 2 == 1:  # Flip sign for odd indices
            U[i, i] = -1
    return qt.Qobj(U)

def global_pi(I, axis='x'):
    """Generate a global pi pulse operator for a given spin and axis."""
    return global_rotation(I, np.pi, axis)
