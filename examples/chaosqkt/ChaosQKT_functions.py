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
