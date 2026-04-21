"""Quantum evolution tools."""

import numpy as np
import qutip as qt
from typing import Union
from numpy import ndarray

from psyduck.fit_toolbox import ExponentialFit


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
# Quantum Kicked Top Evolution
# ============================================================================

def kicked_dynamics(psi_initial, tau, kappa, I, N=1, order=2, pulse_type='pulse'):
    """
    Simulate stroboscopic kicked dynamics of a spin system (quantum kicked top).
    
    This implements the kicked top Hamiltonian with free evolution
    interleaved with instantaneous nonlinear kicks.
    
    Parameters
    ----------
    psi_initial : qutip.Qobj
        Initial quantum state (ket).
    tau : float
        Free evolution time between kicks.
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
    S = I
    dim = int(2 * S + 1)
    chi = kappa / 2
    
    # Create spin operators
    Jx = qt.jmat(S, 'x')
    Jy = qt.jmat(S, 'y')
    Jz = qt.jmat(S, 'z')
    
    # Free Hamiltonian H0
    H0 = (np.pi / 2) * (-Jy)
    
    # Precompute the free evolution operator
    U0 = (-1j * H0 * tau).expm()
    
    # Larmor pulse (instantaneous)
    Ularmor = (-1j * chi * Jz).expm()
    
    # Nonlinear kick unitary
    Upulse = (-1j * Jz**order * kappa / (order * S**(order - 1))).expm()
    
    # Helper function: calculate linear entropy
    def qudit_linear_entropy(psi):
        """Calculate linear entropy of a quantum state."""
        rho = psi * psi.dag() if psi.isket else psi
        return 1.0 - qt.expect(rho * rho, psi)
    
    # Helper function: calculate expectation values
    def qudit_exp(psi):
        """Calculate expectation values of angular momentum operators."""
        exp_x = qt.expect(Jx, psi)
        exp_y = qt.expect(Jy, psi)
        exp_z = qt.expect(Jz, psi)
        return [exp_x, exp_y, exp_z]
    
    # Evolve the state stroboscopically
    psi = psi_initial.copy()
    psi_list = [psi.copy()]
    overlap_list = [psi.overlap(psi_initial)]
    entropy_list = [qudit_linear_entropy(psi)]
    exp_list = [qudit_exp(psi)]
    
    for n in range(N):
        # Free evolution
        psi = U0 * psi
        
        # Instantaneous pulse (choice between Larmor and nonlinear kick)
        if pulse_type.lower() == 'larmor':
            psi = Ularmor * psi
        else:  # Default to 'pulse'
            psi = Upulse * psi
        
        # Store results
        psi_list.append(psi.copy())
        overlap_list.append(psi.overlap(psi_initial))
        entropy_list.append(qudit_linear_entropy(psi))
        exp_list.append(qudit_exp(psi))
    
    return psi_list, overlap_list, entropy_list, exp_list

