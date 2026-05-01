"""Hamiltonian constructors for spin systems."""

import qutip as qt
import numpy as np
from typing import Union
from numpy import ndarray

from psyduck.operations import get_spin_operators, get_transition_operators
from psyduck.tensors import Vab_to_Qab, get_Q_tensor


# ============================================================================
# Static Hamiltonians
# ============================================================================

def zeeman_hamiltonian(I: float | list, B0: float, gamma: float | list = 1.0,
                       theta: float = 0.0, phi: float = 0.0) -> qt.Qobj:
    """Zeeman Hamiltonian for a spin in a magnetic field.

    H = -gamma * B0 * (sin(theta)*cos(phi)*Ix + sin(theta)*sin(phi)*Iy + cos(theta)*Iz)

    theta=0 -> field along +z. theta=pi/2, phi=0 -> field along +x.

    If I and gamma are lists, returns a tensor-product sum over the listed spins.

    :param I: Spin quantum number, or list of spin quantum numbers
    :param B0: Magnetic field strength (T)
    :param gamma: Gyromagnetic ratio in Hz/T (default 1.0), or list matching I
    :param theta: Polar angle from +z toward +x (rad, default 0.0)
    :param phi: Azimuthal angle in x-y plane (rad, default 0.0)
    :return: Zeeman Hamiltonian (Hz)
    """
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    nx, ny, nz = st * cp, st * sp, ct

    if isinstance(I, list):
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
    """General quadrupole Hamiltonian with asymmetry and Euler rotation.

    The EFG principal axis frame (PAF) is rotated into the lab frame by ZYZ
    Euler angles (phi, theta, psi).

    :param I: Nuclear spin quantum number
    :param f_q: Quadrupole splitting frequency (Hz)
    :param eta: Asymmetry parameter (0 to 1)
    :param theta: Polar tilt of PAF z-axis from B0 (rad)
    :param phi: Azimuthal angle of PAF z-axis (rad)
    :param psi: Twist around PAF z-axis (rad)
    :return: Quadrupole Hamiltonian (Hz)
    """
    Q_ab = get_Q_tensor(f_q, eta, theta, phi, psi)
    I_vec = [*get_spin_operators(I)]
    H = sum(Q_ab[a, b] * I_vec[a] * I_vec[b] for a in range(3) for b in range(3))
    return qt.Qobj(H)


def quadrupole_hamiltonian_from_Vab(I: float, V_ab: ndarray, Q: float,
                                    e: float = 1.6e-19, h: float = 6.626e-34):
    """Quadrupole Hamiltonian built directly from the EFG tensor V_ab.

    H = sum_{a,b} Q_ab[a,b] * I_a * I_b,  where Q_ab = e*Q*V_ab / (2I(2I-1)*h)

    Accepts a single tensor or a batch.

    :param I: Nuclear spin quantum number
    :param V_ab: EFG tensor(s) in SI units (V/m²). Shape (3, 3) or (N, 3, 3).
    :param Q: Nuclear quadrupole moment (C·m²)
    :param e: Elementary charge (default 1.6e-19 C)
    :param h: Planck constant (default 6.626e-34 J·s)
    :return: qt.Qobj for a single tensor; np.ndarray of shape (N, d, d) for a batch.
    """
    V_ab = np.asarray(V_ab)
    Q_ab = Vab_to_Qab(V_ab, I, Q, e=e, h=h)

    Ix, Iy, Iz = get_spin_operators(I)
    I_vec = [Ix.full(), Iy.full(), Iz.full()]
    basis = np.array([I_vec[a] @ I_vec[b] for a in range(3) for b in range(3)])  # (9, d, d)

    if V_ab.ndim == 2:
        H = np.einsum('k,kij->ij', Q_ab.ravel(), basis)
        return qt.Qobj(H)
    else:
        H_batch = np.einsum('nk,kij->nij', Q_ab.reshape(len(Q_ab), 9), basis)
        return H_batch  # (N, d, d)


def get_quadrupole_stark_shift(V_ab: ndarray, E_vec: ndarray,
                               I: float, B0: float, gamma: float, Q: float,
                               thetas: ndarray, phis: ndarray,
                               delta: float = 1.0):
    """Quadrupole Stark shift: d(fq1)/dE and d(fq2)/dE over a (theta, phi) grid.

    Computes how the first- and second-order quadrupole splittings change per
    unit E-field (V/m) applied along the direction of E_vec, using a central
    finite difference.  The result maps out electric-noise sensitivity as a
    function of B-field orientation; zero crossings are sweet spots.

    :param V_ab: Static EFG tensor in SI units (V/m²), shape (3, 3).
    :param E_vec: E-field direction vector (e.g. [Ex, Ey, Ez] from NER fit).
                  Normalised internally; only the direction matters.
    :param I: Nuclear spin quantum number.
    :param B0: Static magnetic field (T).
    :param gamma: Nuclear gyromagnetic ratio (Hz/T).
    :param Q: Nuclear quadrupole moment (C·m²).
    :param thetas: 1-D array of polar angles (rad).
    :param phis: 1-D array of azimuthal angles (rad).
    :param delta: E-field step size for the finite difference (V/m, default 1.0).
    :return: (dfq1, dfq2) — arrays of shape (len(thetas), len(phis)), units Hz/(V/m).
    """
    from psyduck.tensors import voigt_to_tensor, get_R_tensor

    E_hat = np.asarray(E_vec, dtype=float)
    E_hat = E_hat / np.linalg.norm(E_hat)

    dV = voigt_to_tensor(get_R_tensor() @ (delta * E_hat))
    H_qp = quadrupole_hamiltonian_from_Vab(I, np.asarray(V_ab) + dV, Q)
    H_qm = quadrupole_hamiltonian_from_Vab(I, np.asarray(V_ab) - dV, Q)

    dfq1 = np.zeros((len(thetas), len(phis)))
    dfq2 = np.zeros((len(thetas), len(phis)))

    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            H_z = zeeman_hamiltonian(I, B0=B0, gamma=gamma, theta=theta, phi=phi)
            ep = (H_z + H_qp).eigenstates()[0]
            em = (H_z + H_qm).eigenstates()[0]
            dfq1[i, j] = (np.mean(np.diff(np.diff(ep))) - np.mean(np.diff(np.diff(em)))) / (2 * delta)
            dfq2[i, j] = (np.mean(np.diff(np.diff(np.diff(ep)))) - np.mean(np.diff(np.diff(np.diff(em))))) / (2 * delta)

    return dfq1, dfq2


def get_quadrupole_splittings(V_ab: ndarray, I: float, B0: float, gamma: float, Q: float,
                        thetas: ndarray, phis: ndarray):
    """Compute first- and second-order quadrupole splittings over a (theta, phi) grid.

    :param V_ab: EFG tensor in SI units (V/m²), shape (3, 3).
    :param I: Nuclear spin quantum number.
    :param B0: Static magnetic field (T).
    :param gamma: Nuclear gyromagnetic ratio (Hz/T).
    :param Q: Nuclear quadrupole moment (C·m²).
    :param thetas: 1-D array of polar angles (rad).
    :param phis: 1-D array of azimuthal angles (rad).
    :return: (fq1, fq2) — arrays of shape (len(thetas), len(phis)), units Hz.
    """
    H_q = quadrupole_hamiltonian_from_Vab(I, np.asarray(V_ab), Q)
    fq1 = np.zeros((len(thetas), len(phis)))
    fq2 = np.zeros((len(thetas), len(phis)))
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            evals = (zeeman_hamiltonian(I, B0=B0, gamma=gamma, theta=theta, phi=phi) + H_q).eigenstates()[0]
            fq1[i, j] = np.mean(np.diff(np.diff(evals)))
            fq2[i, j] = np.mean(np.diff(np.diff(np.diff(evals))))
    return fq1, fq2


def get_fq1(V_ab: ndarray, I: float, B0: float, gamma: float, Q: float,
            thetas: ndarray, phis: ndarray) -> ndarray:
    """First-order quadrupole splitting grid. See get_quadrupole_splittings."""
    return get_quadrupole_splittings(V_ab, I, B0, gamma, Q, thetas, phis)[0]


def get_fq2(V_ab: ndarray, I: float, B0: float, gamma: float, Q: float,
            thetas: ndarray, phis: ndarray) -> ndarray:
    """Second-order quadrupole splitting grid. See get_quadrupole_splittings."""
    return get_quadrupole_splittings(V_ab, I, B0, gamma, Q, thetas, phis)[1]


def hyperfine_hamiltonian(S: float, I: float, A: float) -> qt.Qobj:
    """Isotropic hyperfine interaction Hamiltonian.

    H = A * S · I

    :param S: Electron spin quantum number
    :param I: Nuclear spin quantum number
    :param A: Hyperfine coupling constant (Hz)
    :return: Hyperfine Hamiltonian (Hz)
    """
    Sx, Sy, Sz = get_spin_operators(S)
    Ix, Iy, Iz = get_spin_operators(I)
    return A * (qt.tensor(Sx, Ix) + qt.tensor(Sy, Iy) + qt.tensor(Sz, Iz))


def nmr1_hamiltonian(I: float, B1: Union[float, list, ndarray],
                     axis: str = 'x', gamma: float = 1.0) -> qt.Qobj:
    """NMR (Δm=1) RF drive Hamiltonian.

    Returns the static amplitude operator for use as the time-independent
    envelope in a QuTiP time-dependent Hamiltonian list.

    H = -gamma * B1 * I_axis

    :param I: Spin quantum number
    :param B1: RF field strength. Scalar for a global drive, or length-(2I)
               array to set each Δm=1 transition independently.
    :param axis: Rotation axis ('x' or 'y')
    :param gamma: Gyromagnetic ratio (default 1.0)
    :return: NMR drive amplitude operator (Hz)
    """
    dim = int(2 * I + 1)
    I_axis = qt.jmat(I, axis)
    if isinstance(B1, float):
        return -gamma * B1 * I_axis
    else:
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(len(B1)):
            H[i, i + 1] = -gamma * I_axis[i, i + 1] * B1[i]
            H[i + 1, i] = -gamma * I_axis[i + 1, i] * B1[i]
        return qt.Qobj(H)


def ner1_hamiltonian(I: float,
                     dQxz: Union[float, list, ndarray],
                     dQyz: Union[float, list, ndarray],
                     coupling: float = 1.0) -> qt.Qobj:
    """NER Δm=1 Hamiltonian in the rotating frame.

    Oscillating off-diagonal EFG components V_xz and V_yz drive Δm=1
    transitions via the anticommutators {Ix, Iz} and {Iy, Iz}:

        H = coupling * (dQxz * {Ix, Iz} + dQyz * {Iy, Iz})

    :param I: Spin quantum number
    :param dQxz: Amplitude of the V_xz EFG component. Scalar for a global
                 drive, or length-(2I) array per transition.
    :param dQyz: Amplitude of the V_yz EFG component. Same shape as dQxz.
    :param coupling: Prefactor absorbing eQ/(2I(2I-1)h) (default 1.0)
    :return: NER Δm=1 Hamiltonian (Hz)
    """
    Ix, Iy, Iz = get_spin_operators(I)
    IxIz = Ix * Iz + Iz * Ix
    IyIz = Iy * Iz + Iz * Iy

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

    Oscillating EFG components (V_xx - V_yy) and V_xy drive Δm=2 transitions:

        H = coupling * (dQxx_yy * (Ix²-Iy²) + dQxy * {Ix, Iy})

    :param I: Spin quantum number
    :param dQxx_yy: Amplitude of (V_xx - V_yy). Scalar or length-(2I-1) array.
    :param dQxy: Amplitude of V_xy. Same shape as dQxx_yy.
    :param coupling: Prefactor absorbing eQ/(2I(2I-1)h) (default 1.0)
    :return: NER Δm=2 Hamiltonian (Hz)
    """
    Ix, Iy, Iz = get_spin_operators(I)
    Ixx_yy = Ix * Ix - Iy * Iy
    IxIy   = Ix * Iy + Iy * Ix

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
# Time-dependent drive Hamiltonian
# ============================================================================

def drive_hamiltonian(
    I: float,
    time_array: Union[float, ndarray],
    drive_amplitudes: ndarray,
    drive_frequencies: ndarray,
    rotating_frame_frequencies: ndarray,
    cross_coupling_cutoff: float = None,
) -> Union[qt.Qobj, list]:
    """GRF drive Hamiltonian for multi-tone RF pulses.

    Builds a QuTiP time-dependent Hamiltonian list (drive part only) in the
    generalised rotating frame (GRF). For each spin transition k the drive
    signal is:

        signal_k(t) = sum_j { amp_j(t) * exp(2πi (f_j - f_k) t) }

    where the sum over frequency components j is restricted to those satisfying
    |f_j - f_k| < cross_coupling_cutoff (or unrestricted if None).

    When time_array is a 1-D array the result is a list of [operator,
    coefficient_array] pairs ready to splice into a QuTiP Hamiltonian:

        H = [2*pi*H_static, *drive_hamiltonian(...)]

    When time_array is a scalar float the result is a single static qt.Qobj
    (the instantaneous Hamiltonian at that time), and drive_amplitudes must
    have shape (n_freqs,).

    :param I: Spin quantum number
    :param time_array: 1-D array of time points (s), length T, or a single
        float for an instantaneous snapshot.
    :param drive_amplitudes: Complex drive amplitudes. Shape (n_freqs, T) when
        time_array is an array, or (n_freqs,) when time_array is a scalar.
        The imaginary part encodes the Y-phase (e.g. 1j * amp gives an Iy rotation).
    :param drive_frequencies: Drive frequencies (Hz), shape (n_freqs,)
    :param rotating_frame_frequencies: Rotating frame frequency for each
        transition (Hz), shape (n_transitions,) where n_transitions = 2I.
        Transition k couples level k to level k+1 in the computational basis.
    :param cross_coupling_cutoff: Frequency cutoff (Hz) for including
        off-resonant drive components. None includes all components (default).
        Set to a small value (e.g. 0.5) to suppress all cross-coupling.
    :return: Static qt.Qobj for scalar time, or list of [Qobj, ndarray] pairs
        for an array time.
    """
    drive_amplitudes = np.asarray(drive_amplitudes, dtype=complex)
    drive_frequencies = np.asarray(drive_frequencies)
    rotating_frame_frequencies = np.asarray(rotating_frame_frequencies)

    n_transitions = int(2 * I)
    assert len(rotating_frame_frequencies) == n_transitions, \
        f"rotating_frame_frequencies must have length 2I = {n_transitions}"

    Tx, Ty = get_transition_operators(I)
    scalar_input = np.ndim(time_array) == 0

    if scalar_input:
        t = float(time_array)
        assert drive_amplitudes.shape == (len(drive_frequencies),), \
            "For scalar time, drive_amplitudes must have shape (n_freqs,)"
        d = int(2 * I + 1)
        H = qt.Qobj(np.zeros((d, d), dtype=complex))
        for k in range(n_transitions):
            f_rf = rotating_frame_frequencies[k]
            signal = 0j
            for j, f in enumerate(drive_frequencies):
                if cross_coupling_cutoff is None or abs(f - f_rf) < cross_coupling_cutoff:
                    signal += drive_amplitudes[j] * np.exp(2j * np.pi * (f - f_rf) * t)
            H += 2 * np.pi * (np.real(signal) * Tx[k] + np.imag(signal) * Ty[k])
        return H

    assert drive_amplitudes.shape == (len(drive_frequencies), len(time_array)), \
        "drive_amplitudes must have shape (n_freqs, len(time_array))"

    H_drive = []
    for k in range(n_transitions):
        f_rf = rotating_frame_frequencies[k]
        signal = np.zeros(len(time_array), dtype=complex)
        for j, f in enumerate(drive_frequencies):
            if cross_coupling_cutoff is None or abs(f - f_rf) < cross_coupling_cutoff:
                signal += drive_amplitudes[j] * np.exp(2j * np.pi * (f - f_rf) * time_array)
        H_drive.append([Tx[k], 2 * np.pi * np.real(signal)])
        H_drive.append([Ty[k], 2 * np.pi * np.imag(signal)])
    return H_drive


def Hz_order(kappa, order, spin_I):
    """
    Create a higher-order Hamiltonian for the kicked-top.

    H_z^(order) = kappa * Iz^order / (order * I^(order-1))

    Parameters
    ----------
    kappa : float
        Kick strength parameter.
    order : int
        Order of the Hamiltonian (typically 2 or 3).
    spin_I : float
        Spin quantum number (e.g., 7/2 for a qutrit).

    Returns
    -------
    qutip.Qobj
        Higher-order Hamiltonian as a QuTiP operator.
    """
    Iz = qt.jmat(spin_I, 'z')
    H = kappa * (Iz ** order) / (order * (spin_I ** (order - 1)))
    return H
