"""
ChaosQKT Psyduck Helper Functions

This module provides wrapper functions and implementations for quantum chaos
simulations using the psyduck framework. It fills the gap between the psyduck
library and the full functionality needed for ChaosQKT tutorials.

Functions include:
- Spin state initialization and manipulation
- Kicked dynamics simulation
- Data fitting functions
- Utility functions for data processing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import qutip as qt
from qutip import jmat, basis, expect, ket2dm, Qobj
from psyduck import Spin
from psyduck.operations import global_rotation




# ============================================================================
# Trotterization Functions for Kicked-Top Hamiltonian
# ============================================================================

def build_trotter_step(spin_I, step_size, omega_drive, Hz_term):
    """
    Construct the first-order Trotter step for the kicked-top Hamiltonian.

    Parameters
    ----------
    spin_I : float
        Spin quantum number.
    step_size : float
        Trotter time step.
    omega_drive : float
        Y-drive strength in the Hamiltonian Hy = -omega_drive * Iy.
    Hz_term : qutip.Qobj
        Higher-order Z Hamiltonian term.

    Returns
    -------
    tuple[qutip.Qobj, qutip.Qobj, qutip.Qobj]
        (U_step, Uy_step, Uz_step)
    """
    Uy_step = global_rotation(spin_I, -omega_drive * step_size, 'y')
    Uz_step = (-1j * step_size * Hz_term).expm()
    U_step = Uz_step * Uy_step
    return U_step, Uy_step, Uz_step


def run_trotterized_evolution(psi_initial, U_step, n_steps):
    """
    Apply a single-step Trotter unitary repeatedly.

    Parameters
    ----------
    psi_initial : qutip.Qobj
        Initial ket.
    U_step : qutip.Qobj
        Single Trotter-step unitary.
    n_steps : int
        Number of repeated applications.

    Returns
    -------
    list[qutip.Qobj]
        State trajectory including the initial state.
    """
    trajectory = [psi_initial.copy()]
    for _ in range(n_steps):
        trajectory.append(U_step * trajectory[-1])
    return trajectory


def run_continuous_evolution(psi_initial, H_total, spin_I, total_time, n_steps):
    """
    Evolve a spin state continuously on a time grid.

    Parameters
    ----------
    psi_initial : qutip.Qobj
        Initial ket.
    H_total : qutip.Qobj
        Full continuous Hamiltonian.
    spin_I : float
        Spin quantum number.
    total_time : float
        Final evolution time.
    n_steps : int
        Number of intervals. The returned trajectory has n_steps + 1 states.

    Returns
    -------
    psyduck.spin_series.SpinSeries
        Continuous trajectory sampled on an equally spaced time grid.
    """
    times = np.linspace(0, total_time, n_steps + 1)
    nucleus = Spin(I=spin_I)
    nucleus.state = psi_initial.copy()
    return nucleus.evolve(H_total, times)


def state_populations(state):
    """
    Return basis-state populations for a ket or density matrix.

    Parameters
    ----------
    state : qutip.Qobj
        Quantum state.

    Returns
    -------
    numpy.ndarray
        Population vector over the computational basis.
    """
    if state.type == 'ket':
        return np.abs(state.full().ravel()) ** 2
    return np.real(state.diag())


def states_to_population_array(states):
    """
    Convert a list of states into a population array.

    Parameters
    ----------
    states : list[qutip.Qobj]
        List of kets or density matrices.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_states, dim).
    """
    return np.array([state_populations(state) for state in states])


###### OTOC ######
def probs_in_Hprime_basis(state, theta, phi, j=7/2):
    """
    Compute populations in the rotated H' basis used by the OTOC protocol.

    The basis rotation is implemented with the standard psyduck full-spin
    rotations R(phi, theta) = Rz(phi) Ry(theta).

    Parameters
    ----------
    state : qutip.Qobj
        Ket or density matrix.
    theta : float
        Colatitude angle defining the rotated basis.
    phi : float
        Azimuthal angle defining the rotated basis.
    j : float, optional
        Spin quantum number.

    Returns
    -------
    numpy.ndarray
        Population vector in the H' basis.
    """
    rho = state * state.dag() if state.isket else state
    rotation = global_rotation(j, phi, 'z') * global_rotation(j, theta, 'y')
    rho_prime = rotation.dag() * rho * rotation
    return np.real_if_close(np.array(rho_prime.diag(), dtype=complex)).real


def otoc_from_populations(m_vals, probs, eps):
    """
    Reconstruct the OTOC signal from H'-basis populations.

    Parameters
    ----------
    m_vals : array_like
        Magnetic quantum numbers ordered consistently with the populations.
    probs : array_like
        H'-basis populations.
    eps : float
        Small rotation angle in W_eps = exp(-i eps Jz).

    Returns
    -------
    tuple[float, complex]
        (F, W_expect), where F = |<W_eps>|^2.
    """
    phases = np.exp(-1j * eps * np.asarray(m_vals, dtype=float))
    W_expect = np.sum(np.asarray(probs, dtype=complex) * phases)
    F = float(np.abs(W_expect) ** 2)
    return F, W_expect


def otoc_trajectory(states, theta, phi, eps, j=7/2):
    """
    Compute the OTOC trajectory from a list of evolved states.

    Parameters
    ----------
    states : list[qutip.Qobj]
        State trajectory.
    theta : float
        Colatitude angle defining the measurement basis.
    phi : float
        Azimuthal angle defining the measurement basis.
    eps : float
        Small rotation angle in W_eps = exp(-i eps Jz).
    j : float, optional
        Spin quantum number.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        (C_values, F_values, populations, m_values)
    """
    m_vals = np.arange(j, -j - 1, -1, dtype=float)
    populations = []
    F_values = []
    C_values = []

    for state in states:
        probs = probs_in_Hprime_basis(state, theta, phi, j=j)
        F_value, _ = otoc_from_populations(m_vals, probs, eps)
        populations.append(probs)
        F_values.append(F_value)
        C_values.append(1.0 - F_value)

    return (
        np.array(C_values, dtype=float),
        np.array(F_values, dtype=float),
        np.array(populations, dtype=float),
        m_vals,
    )



def kappa_to_opx_phase_subspace(kappa, J, initial_state, order):
    """
    Calculate SNAP gate phases from kappa parameter.
    
    This function computes the diagonal phase angles for a SNAP gate that implements
    a higher-order Hamiltonian H = kappa * Iz^order. The phases are calculated for
    a specific subspace defined by initial_state and J.
    
    Parameters
    ----------
    kappa : float
        Kick strength parameter (phase accumulation per pulse).
    J : float
        Spin quantum number.
    initial_state : int
        Starting state (typically 0 for ground state).
    order : int
        Order of the kick (2 for quadratic, 3 for cubic, etc.).
        
    Returns
    -------
    list
        SNAP phases (in rotations, i.e., normalized by 2π) for each basis state.
        
    Notes
    -----
    The SNAP gate is defined as: SNAP = exp(-i * Iz^order * kappa / (order * J^(order-1)))
    This function extracts the diagonal phases from the SNAP unitary and maps them
    to the full Hilbert space with appropriate masking.
    """
    # Construct the tridiagonal matrix structure
    A = -np.tri(int(2*J), int(2*J), 0)
    A_inv = np.linalg.inv(A)
    
    # Get Iz operator
    Iz = jmat(J, 'z')
    
    # Construct SNAP unitary
    SNAP = (-1j * (Iz ** order) * kappa / (order * J ** (order - 1))).expm()
    
    # Extract diagonal phases (in radians)
    SNAP_angle = np.angle(SNAP.diag()) * 180 / np.pi
    
    # Reference to the first element
    SNAP_angle = SNAP_angle - SNAP_angle[0]
    
    # Invert to get optical dipole drive phases
    opx_phase = A_inv @ SNAP_angle[1:]
    
    # Normalize to [0, 1) (in units of full rotations)
    opx_phase_SNAP = ((opx_phase) / 360) % 1
    opx_phase_SNAP = np.round(opx_phase_SNAP, 6).tolist()
    
    # Map to full Hilbert space with mask
    mask_list = ([0] * int(initial_state) + 
                 [1] * int(J * 2) + 
                 [0] * int(int(2*J) + 1 - initial_state - int(J * 2)))
    
    result = []
    a_index = 0
    
    for m in mask_list:
        if m == 1:
            result.append(opx_phase_SNAP[a_index])
            a_index += 1
        else:
            result.append(0)
    
    return result
