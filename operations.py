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
    ""
    _, _, Iz = get_spin_operators(I)
    return -gamma * B0 * Iz """


def quadrupole_hamiltonian() -> None:
    """Quadrupole Hamiltonian."""
    pass


def rf_hamiltonian(I: float, B1: float, omega_rf: float, 
                   axis: str = 'x', gamma: float = 1.0) -> qt.Qobj:
    """Radiofrequency field Hamiltonian in the lab frame.
    
    H = -gamma * B1 * I_axis * cos(omega_rf * t)
    
    Note: This returns the time-independent envelope for use with time-dependent solvers.
    
    :param I: Spin quantum number
    :param B1: RF field strength
    :param omega_rf: RF frequency
    :param axis: Rotation axis ('x', 'y', or 'z')
    :param gamma: Gyromagnetic ratio (default 1.0)
    :return: RF Hamiltonian amplitude
    """
    I_axis = qt.jmat(I, axis)
    return -gamma * B1 * I_axis


# ============================================================================
# Unitary Operators
# ============================================================================

def pulse_operator(I: float, angle: float, axis: str = 'x') -> qt.Qobj:
    """RF pulse operator (rotation around specified axis).
    
    U = exp(-i * angle * I_axis)
    
    :param I: Spin quantum number
    :param angle: Rotation angle in radians
    :param axis: Rotation axis ('x', 'y', or 'z')
    :return: Pulse (rotation) operator
    """
    I_axis = qt.jmat(I, axis)
    U = (-1j * angle * I_axis).expm()
    return U


def rotation_operator(I: float, angle: float, axis: Union[str, ndarray]) -> qt.Qobj:
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

def subspace_rotation_operator(I: float, angle: float, axis: ndarray) -> qt.Qobj:
    """Subspace rotation operator around an arbitrary axis.
    
    :param I: Spin quantum number
    :param angle: Rotation angle in radians
    :param axis: Rotation axis as a 3-element array
    :return: Subspace rotation operator
    """
    d = int(2 * I + 1)
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Rotation axis cannot be zero vector")
    axis = axis / norm

    Ix, Iy, Iz = get_spin_operators(I)
    Rrot_exponent = 1j * angle * (axis[0] * Ix + axis[1] * Iy + axis[2] * Iz)
    Rrot = Rrot_exponent.expm()

    # Pad with zeros to fit an 8x8 matrix
    Rrot = np.pad(Rrot, (((8 - d) // 2, (8 - d) // 2), ((8 - d) // 2, (8 - d) // 2)), 
                  'constant', constant_values=0)
    return Rrot

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
    U = np.eye(d)
    U[element1, element2] = 1
    U[element2, element1] = 1
    U[element1, element1] = 0
    U[element2, element2] = 0
    return qt.Qobj(U)
