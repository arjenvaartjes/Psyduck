"""Quantum evolution tools."""

import numpy as np
import qutip as qt
from typing import Union
from numpy import ndarray

from psyduck.fit_toolbox import ExponentialFit
from psyduck.hamiltonians import Hz_order
from psyduck.operations import get_spin_operators, global_rotation
from psyduck.spin import Spin

# Back-compat re-export: the classical kicked-top map lives in
# psyduck.classical_dynamics now.  Keep ``from psyduck.evolve import
# kicked_top_step`` working for older notebooks / scripts.
from psyduck.classical_dynamics import kicked_top_step  # noqa: F401


# ============================================================================
# Open-system evolution
# ============================================================================

def free_decay(psi0s, times, c_ops):
    """Simulate free decay for a list of initial states and fit T2* times.

    Parameters
    ----------
    psi0s : list of Qobj
        Initial states to simulate.
    times : array_like
        Time points for the simulation.
    c_ops : list
        Collapse operators (from get_collapse_operators).

    Returns
    -------
    fidelity : ndarray, shape (len(psi0s), len(times))
    T2s : ndarray, shape (len(psi0s),)
    alphas : ndarray, shape (len(psi0s),)
    """
    dim = psi0s[0].shape[0]
    H0 = qt.qzero(dim)

    fidelity = np.zeros([len(psi0s), len(times)])
    T2s = np.zeros(len(psi0s))
    alphas = np.zeros(len(psi0s))

    for p, psi0 in enumerate(psi0s):
        result = qt.mesolve(H0, psi0, times, c_ops)
        fidelity[p] = qt.expect(psi0 * psi0.dag(), result.states)
        fit = ExponentialFit(fidelity[p], xvals=times)
        T2s[p] = fit['tau']
        alphas[p] = fit['exponent_factor']

    return fidelity, T2s, alphas


# ============================================================================
# Frame transformations
# ============================================================================

def frame_rotate(states: list, times: ndarray, H_generator: qt.Qobj) -> list:
    """Apply a post-hoc rotating-frame transformation to a list of states.

    Applies the unitary S(t) = exp(i * 2π * H_generator * t) to each state,
    transforming from one rotating frame to another after simulation.

    A common use case is rotating GRF results back to the ZRF for comparison:
    given H_generator = f_q * Iz², this applies S(t) = exp(i 2π f_q Iz² t).

    :param states: List of Qobj states (kets or density matrices), one per
                   time point. Typically result.states from qt.sesolve.
    :param times: 1-D array of time points (s), same length as states.
    :param H_generator: Static Hamiltonian (Hz, no 2π factor) generating the
                        frame transformation.
    :return: List of rotated Qobj states.
    """
    rotated = []
    for state, t in zip(states, times):
        S = (1j * 2 * np.pi * H_generator * t).expm()
        rotated.append(S * state)
    return rotated

# ============================================================================
# Kicked Top Evolution
# ============================================================================

def kicked_dynamics(psi_initial, tau, kappa, I, N=1, order=2, pulse_type='pulse'):
    """
    Simulate stroboscopic kicked dynamics of a spin system (quantum kicked top).

    Implements the kicked-top Hamiltonian
        H(t) = -(pi / (2 tau)) Iy + (kappa / (2 I)) Iz^order * sum_n delta(t - n tau)
    i.e. a Iy rotation by H_rot = -(pi / (2 tau)) Iy between kicks, with an
    instantaneous nonlinear kick exp(-i * Hz_order(kappa, order, I)) at every
    multiple of tau.  The 1/tau factor in H_rot makes the per-period rotation
    a fixed pi/2 about Iy independently of tau, matching the standard textbook
    convention.

    Parameters
    ----------
    psi_initial : qutip.Qobj
        Initial quantum state (ket).
    tau : float
        Period between kicks.  Sets the 1/tau factor in H_rot; the per-period
        Iy rotation is pi/2 for any tau.
    kappa : float
        Kick strength parameter.
    I : float
        Total angular momentum quantum number (spin).
    N : int, optional
        Number of kicks to apply (default: 1).
    order : int, optional
        Order of the nonlinear kick operator (default: 2).
    pulse_type : str, optional
        Type of pulse to apply in each kick:
        - 'pulse' (default): Apply nonlinear kick Upulse
        - 'larmor': Apply Larmor precession Ularmor instead of Upulse

    Returns
    -------
    psi_list : list
        List of quantum states after each kick.
    overlap_list : list
        List of overlaps with the initial state.
    entropy_list : list
        List of linear entropy values.
    exp_list : list
        List of expectation value arrays [<Jx>, <Jy>, <Jz>].
    """

    # Create a Spin object to hold and track the state
    spin = Spin(I=I, state=psi_initial.copy())

    # Get spin operators using psyduck utilities
    Ix, Iy, Iz = get_spin_operators(I)

    # Rotation Hamiltonian H_rot = -pi / (2 tau) * Iy.
    # Evolving for time tau gives U_rot = exp(i (pi/2) Iy) regardless of tau.
    H_rot = -(np.pi / (2 * tau)) * Iy
    U_rot = (-1j * H_rot * tau).expm()

    # Larmor pulse (instantaneous Iz)
    Ularmor = (-1j * (kappa / 2) * Iz).expm()

    # Nonlinear kick unitary using Hz_order; for order=2 this is
    # H_kick = (kappa / (2 I)) Iz^2, matching the textbook kicked top.
    H_kick = Hz_order(kappa, order, I)
    Upulse = (-1j * H_kick).expm()
    
    # Store initial state for overlap calculation
    psi_initial_normalized = psi_initial / psi_initial.norm()
    
    # Evolve the state stroboscopically
    psi_list = [spin.state.copy()]
    overlap_list = [abs(psi_initial_normalized.dag() * spin.state)]
    entropy_list = [spin.linear_entropy()]
    exp_list = [[spin.expectation(Ix), spin.expectation(Iy), spin.expectation(Iz)]]
    
    for n in range(N):
        # Free evolution
        spin.apply_operator(U_rot)
        
        # Instantaneous pulse (choice between Larmor and nonlinear kick)
        if pulse_type.lower() == 'larmor':
            spin.apply_operator(Ularmor)
        else:  # Default to 'pulse'
            spin.apply_operator(Upulse)
        
        # Store results
        psi_list.append(spin.state.copy())
        overlap_list.append(abs(psi_initial_normalized.dag() * spin.state))
        entropy_list.append(spin.linear_entropy())
        exp_list.append([spin.expectation(Ix), spin.expectation(Iy), spin.expectation(Iz)])

    return psi_list, overlap_list, entropy_list, exp_list


def trotter_dynamics(psi_initial, delta_tau, omega_y, kappa, I, N=1, order=2):
    """First-order Trotter integrator for the continuous Hamiltonian
        H = -omega_y * Iy + Hz_order(kappa, order, I).

    Each step applies U_step = exp(-i * H_z * delta_tau) * exp(-i * H_y * delta_tau),
    so N steps cover a total time T = N * delta_tau.

    Returns the same four lists as kicked_dynamics so that downstream analysis
    (overlap traces, entropy, expectation values) is interchangeable between
    the two propagators.

    Parameters
    ----------
    psi_initial : qutip.Qobj
        Initial quantum state (ket).
    delta_tau : float
        Per-step time increment.
    omega_y : float
        Strength of the continuous Iy drive (H_y = -omega_y * Iy).
    kappa : float
        Strength of the continuous Iz^order term entering Hz_order.
    I : float
        Total angular momentum quantum number (spin).
    N : int, optional
        Number of Trotter steps (default: 1).
    order : int, optional
        Order of the nonlinear Iz term (default: 2).
    """
    spin = Spin(I=I, state=psi_initial.copy())
    Ix, Iy, Iz = get_spin_operators(I)

    H_y = -omega_y * Iy
    H_z = Hz_order(kappa, order, I)
    U_y_step = (-1j * H_y * delta_tau).expm()
    U_z_step = (-1j * H_z * delta_tau).expm()
    U_step   = U_z_step * U_y_step

    psi_initial_normalized = psi_initial / psi_initial.norm()

    psi_list     = [spin.state.copy()]
    overlap_list = [abs(psi_initial_normalized.dag() * spin.state)]
    entropy_list = [spin.linear_entropy()]
    exp_list     = [[spin.expectation(Ix), spin.expectation(Iy), spin.expectation(Iz)]]

    for _ in range(N):
        spin.apply_operator(U_step)
        psi_list.append(spin.state.copy())
        overlap_list.append(abs(psi_initial_normalized.dag() * spin.state))
        entropy_list.append(spin.linear_entropy())
        exp_list.append([spin.expectation(Ix), spin.expectation(Iy), spin.expectation(Iz)])

    return psi_list, overlap_list, entropy_list, exp_list


# ============================================================================
# Out-of-time-ordered correlator (OTOC) reconstruction
# ============================================================================

def otoc_trajectory(states, theta, phi, eps, j=7/2):
    """
    Compute the OTOC trajectory C(t) = 1 - F(t) from a forward-evolved state list.

    Implements the Blocher et al. protocol (Phys. Rev. A 106, 042429, Eqs. 6-7
    and Section III.E) for the choice V(0) = |theta, phi><theta, phi| and
    W_eps = R(theta, phi) exp(-i eps Jz) R†(theta, phi):

        F(t) = |<W_eps(t)>|^2 = | sum_m P_m(t) exp(-i eps m) |^2
        C(t) = 1 - F(t)

    where P_m(t) = |<J, m| R†(theta, phi) |psi(t)>|^2 are the populations of
    psi(t) in the eigenbasis of W_eps.

    Parameters
    ----------
    states : list[qutip.Qobj]
        Forward-evolved trajectory (kets or density matrices).
    theta, phi : float
        Polar/azimuthal angles defining both the initial spin-coherent state
        and the W_eps rotation axis.
    eps : float
        Rotation angle in W_eps = exp(-i eps n.J). Eigenvalues lambda_m =
        exp(-i eps m); eigenstates do not depend on eps.
    j : float, optional
        Total angular momentum quantum number (default 7/2).

    Returns
    -------
    C_values : ndarray
        OTOC growth signal C(t) = 1 - F(t).
    F_values : ndarray
        OTOC reconstruction F(t) = |<W_eps(t)>|^2.
    populations : ndarray, shape (n_states, dim)
        P_m(t) in the W_eps eigenbasis, m ordered +j -> -j.
    m_vals : ndarray
        Magnetic quantum numbers [+j, +j-1, ..., -j].
    """
    # --- One-time setup -----------------------------------------------------
    # R(theta, phi) = Rz(phi) Ry(theta) rotates |J, J> to the spin-coherent
    # state |theta, phi> and simultaneously diagonalizes W_eps. Building it
    # once and reusing across the trajectory is cheaper than rebuilding per
    # state.
    rotation = global_rotation(j, phi, 'z') * global_rotation(j, theta, 'y')
    R_dag = rotation.dag()

    # Magnetic quantum numbers ordered to match QuTiP's diag convention:
    # jmat(j, 'z') has eigenvalues from +j (index 0) down to -j (last index),
    # so populations[i] corresponds to m = j - i.
    m_vals = np.arange(j, -j - 1, -1, dtype=float)

    # Eigenvalues lambda_m = exp(-i eps m) of W_eps in the rotated basis.
    # These depend only on eps, so the same array is reused for every state —
    # a single trajectory yields F(t) for any eps via post-processing.
    phases = np.exp(-1j * eps * m_vals)

    populations = []
    F_values = []
    C_values = []

    for state in states:
        # --- Step 1: sample populations P_m(t) in the W_eps eigenbasis -----
        # For a ket, rotate the state vector directly: psi' = R† psi, then
        # P_m = |psi'_m|^2. For a density matrix, rho' = R† rho R, and
        # P_m = diag(rho')_m. The ket branch avoids forming the (d x d)
        # density matrix and is what we hit for sesolve/kicked_dynamics output.
        if state.isket:
            psi_prime = R_dag * state
            probs = np.abs(psi_prime.full().ravel()) ** 2
        else:
            rho_prime = R_dag * state * rotation
            probs = np.real_if_close(np.array(rho_prime.diag(), dtype=complex)).real

        # --- Step 2: reconstruct <W_eps>, F(t), and C(t) -------------------
        # <W_eps(t)> = sum_m P_m(t) lambda_m  (Blocher Eq. 7)
        # F(t)      = |<W_eps(t)>|^2          (Blocher Eq. 6)
        # C(t)      = 1 - F(t)                (growth signal, paper below Eq. 2)
        W_expect = np.sum(probs.astype(complex) * phases)
        F_value = float(np.abs(W_expect) ** 2)

        populations.append(probs)
        F_values.append(F_value)
        C_values.append(1.0 - F_value)

    return (
        np.array(C_values, dtype=float),
        np.array(F_values, dtype=float),
        np.array(populations, dtype=float),
        m_vals,
    )

