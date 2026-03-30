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
