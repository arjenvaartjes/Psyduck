"""Quantum evolution tools"""

import numpy as np
import qutip as qt
from psyduck.fit_toolbox import ExponentialFit


# ============================================================================
# Operation processes
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

