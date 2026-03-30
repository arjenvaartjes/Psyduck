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

def euler_rotation(phi, theta, psi):
    """ZYZ Euler rotation matrix."""
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
# Hamiltonian Functions
# ============================================================================

def zeeman_hamiltonian(I: float | list, B0: float, gamma: float | list = 1.0,
                       theta: float = 0.0, phi: float = 0.0) -> qt.Qobj:
    """Zeeman Hamiltonian for a spin in a magnetic field. If lists are given, make this return a tensor product sum

    H = -gamma * B0 * (sin(theta)*cos(phi)*Ix + sin(theta)*sin(phi)*Iy + cos(theta)*Iz)

    theta=0 -> field along +z. theta=pi/2, phi=0 -> field along +x.

    :param I: Spin quantum number (s)
    :param B0: Magnetic field strength
    :param gamma: Gyromagnetic ratio (default 1.0)
    :param theta: Polar angle from +z toward +x (default 0.0)
    :param phi: Azimuthal angle in x-y plane (default 0.0)
    :return: Zeeman Hamiltonian
    """
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    nx, ny, nz = st * cp, st * sp, ct

    if isinstance(I, list):
        # Tensor the spin systems
        H = qt.tensor(qt.qzero(2*i+1) for i in I)
        for inx in range(len(I)):
            Ix, Iy, Iz = get_spin_operators(I[inx])
            I_field = nx * Ix + ny * Iy + nz * Iz
            H += -gamma[inx] * B0 * qt.tensor(*(
                    [qt.qeye(2*I[i]+1) for i in range(0, inx)] +
                    [I_field] +
                    [qt.qeye(2*I[i]+1) for i in range(inx+1, len(I))]
            ))
        return H
    Ix, Iy, Iz = get_spin_operators(I)
    return -gamma * B0 * (nx * Ix + ny * Iy + nz * Iz)


def quadrupole_hamiltonian(I, f_q, eta=0.0, theta=0.0, phi=0.0, psi=0.0):
    """General quadrupole Hamiltonian with asymmetry and rotation.

    :param I: Nuclear spin quantum number
    :param f_q: Quadrupole splitting frequency (Hz)
    :param eta: Asymmetry parameter (0 to 1)
    :param theta: Polar tilt of PAF z-axis from B0 (rad)
    :param phi: Azimuthal angle of PAF z-axis (rad)
    :param psi: Twist around PAF z-axis (rad)
    :return: Quadrupole Hamiltonian as Qobj (Hz)
    """

    # Convert f_q to the prefactor for the double sum
    # f_q = 3*e*Q*V_zz / (2*I*(2I-1)*h), and the double sum uses
    # e*Q*V_ab / (2*I*(2I-1)*h), so the prefactor per V_ab element
    # relative to V_zz is f_q/3
    V_zz_normalised = 1.0  # work in units where V_zz = 1
    V_PAF = V_zz_normalised * np.diag([-(1-eta)/2, -(1+eta)/2, 1.0])

    R = euler_rotation(phi, theta, psi)
    V_lab = R @ V_PAF @ R.T

    # Scale so that the double sum gives the correct f_q
    Q_ab = (f_q / 3.0) * V_lab
    I_vec = [*get_spin_operators(I)]
    H = sum(Q_ab[a, b] * I_vec[a] * I_vec[b] for a in range(3) for b in range(3))
    return qt.Qobj(H)


def quadrupole_hamiltonian_from_Vab(I: float, V_ab: ndarray, Q: float,
                                    e: float = 1.6e-19, h: float = 6.626e-34):
    """Quadrupole Hamiltonian for a nuclear spin.

    H = sum_{a,b} Q_ab[a,b] * I_a * I_b,  where Q_ab = e*Q*V_ab / (2I(2I-1)*h)

    :param I: Nuclear spin quantum number
    :param V_ab: EFG tensor(s) in SI units (V/m²). Shape (3, 3) or (N, 3, 3).
    :param Q: Nuclear quadrupole moment (C·m²)
    :param e: Elementary charge (default 1.6e-19 C)
    :param h: Planck constant (default 6.626e-34 J·s)
    :return: qt.Qobj for a single tensor; np.ndarray of shape (N, d, d) for a batch.
    """
    V_ab = np.asarray(V_ab)
    scale = e * Q / (2 * I * (2 * I - 1) * h)

    Ix, Iy, Iz = get_spin_operators(I)
    I_vec = [Ix.full(), Iy.full(), Iz.full()]
    basis = np.array([I_vec[a] @ I_vec[b] for a in range(3) for b in range(3)])  # (9, d, d)

    if V_ab.ndim == 2:
        Q_ab_flat = (scale * V_ab).ravel()          # (9,)
        H = np.einsum('k,kij->ij', Q_ab_flat, basis)
        return qt.Qobj(H)
    else:
        Q_ab_flat = (scale * V_ab).reshape(len(V_ab), 9)   # (N, 9)
        return np.einsum('nk,kij->nij', Q_ab_flat, basis)  # (N, d, d)


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
    dim = int(2 * I + 1)
    I_axis = qt.jmat(I, axis)
    if isinstance(B1, float):
        return -gamma * B1 * I_axis
    else:
        H_nmr = np.zeros((dim, dim), dtype=complex)
        for i in range(len(B1)):
            H_nmr[i, i+1] = -gamma * I_axis[i, i+1] * B1[i]
            H_nmr[i+1, i] = -gamma * I_axis[i+1, i] * B1[i]
        return qt.Qobj(H_nmr)


def ner1_hamiltonian(I: float,
                     dQxz: Union[float, list, ndarray],
                     dQyz: Union[float, list, ndarray],
                     coupling: float = 1.0) -> qt.Qobj:
    """NER Δm=1 Hamiltonian in the rotating frame.

    Oscillating off-diagonal EFG components V_xz and V_yz drive Δm=1 transitions
    via the anticommutators {{Ix, Iz}} and {{Iy, Iz}}:

        H = coupling * (dQxz * {{Ix, Iz}} + dQyz * {{Iy, Iz}})

    where {{A, B}} = AB + BA.

    The two channels are physically independent:
      - dQxz controls the real (in-phase) part of the drive
      - dQyz controls the imaginary (quadrature) part of the drive

    :param I: Spin quantum number
    :param dQxz: Amplitude of the V_xz EFG component. Scalar for a global drive,
                 or length-(2I) array to set each Δm=1 transition independently.
    :param dQyz: Amplitude of the V_yz EFG component. Same shape convention as dQxz.
    :param coupling: Prefactor absorbing eQ/(2I(2I-1)h) (default 1.0)
    :return: NER Δm=1 Hamiltonian
    """
    Ix, Iy, Iz = get_spin_operators(I)
    IxIz = Ix * Iz + Iz * Ix   # {Ix, Iz}
    IyIz = Iy * Iz + Iz * Iy   # {Iy, Iz}

    if np.ndim(dQxz) == 0 and np.ndim(dQyz) == 0:
        return coupling * (float(dQxz) * IxIz + float(dQyz) * IyIz)

    dQxz = np.asarray(dQxz)
    dQyz = np.asarray(dQyz)
    dim = int(2 * I + 1)
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        elem = coupling * (dQxz[i] * IxIz[i, i + 1] + dQyz[i] * IyIz[i, i + 1])
        H[i, i + 1] = elem
        H[i + 1, i] = np.conj(elem)
    return qt.Qobj(H)


def ner2_hamiltonian(I: float,
                     dQxx_yy: Union[float, list, ndarray],
                     dQxy: Union[float, list, ndarray],
                     coupling: float = 1.0) -> qt.Qobj:
    """NER Δm=2 Hamiltonian in the rotating frame.

    Oscillating EFG components (V_xx - V_yy) and V_xy drive Δm=2 transitions
    via the operators (Ix²-Iy²) and {{Ix, Iy}}:

        H = coupling * (dQxx_yy * (Ix²-Iy²) + dQxy * {{Ix, Iy}})

    where {{A, B}} = AB + BA.

    The two channels are physically independent:
      - dQxx_yy controls the real (in-phase) part of the drive
      - dQxy controls the imaginary (quadrature) part of the drive

    :param I: Spin quantum number
    :param dQxx_yy: Amplitude of the (V_xx - V_yy) EFG component. Scalar for a global drive,
                    or length-(2I-1) array to set each Δm=2 transition independently.
    :param dQxy: Amplitude of the V_xy EFG component. Same shape convention as dQxx_yy.
    :param coupling: Prefactor absorbing eQ/(2I(2I-1)h) (default 1.0)
    :return: NER Δm=2 Hamiltonian
    """
    Ix, Iy, Iz = get_spin_operators(I)
    Ixx_yy = Ix * Ix - Iy * Iy          # Ix² - Iy² = (I+² + I-²)/2
    IxIy   = Ix * Iy + Iy * Ix          # {Ix, Iy}  = i(I-² - I+²)/2

    if np.ndim(dQxx_yy) == 0 and np.ndim(dQxy) == 0:
        return coupling * (float(dQxx_yy) * Ixx_yy + float(dQxy) * IxIy)

    dQxx_yy = np.asarray(dQxx_yy)
    dQxy    = np.asarray(dQxy)
    dim = int(2 * I + 1)
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 2):
        elem = coupling * (dQxx_yy[i] * Ixx_yy[i, i + 2] + dQxy[i] * IxIy[i, i + 2])
        H[i, i + 2] = elem
        H[i + 2, i] = np.conj(elem)
    return qt.Qobj(H)


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
