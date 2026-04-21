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
# OTOC-related functions
# ============================================================================

def magnetic_quantum_numbers(j):
    """Return the ordered m-values for a spin-j system."""
    return np.arange(j, -j - 1, -1)


def spin_coherent_projector(j, theta, phi):
    """Create the projector onto a spin-coherent state."""
    spin = Spin(I=j)
    spin.make_displaced_coherent_state(theta=theta, phi=phi)
    return ket2dm(spin.state)


def _as_density_matrix(state):
    """Convert a ket or density matrix into a density matrix."""
    return ket2dm(state) if state.isket else state


def probs_in_Hprime_basis(state, theta, phi, j):
    """
    Calculate population probabilities in a tilted basis.
    
    Parameters
    ----------
    state : qutip.Qobj
        Quantum state as either a ket or a density matrix.
    theta : float
        Tilt angle (colatitude).
    phi : float
        Azimuthal angle.
    j : float
        Angular momentum quantum number.
        
    Returns
    -------
    np.ndarray
        Population probabilities in tilted basis.
    """
    rho = _as_density_matrix(state)

    # This rotation maps the z-basis onto the measurement axis n(theta, phi).
    rotation_axis = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float)
    rotation = global_rotation(j, theta, rotation_axis)
    rotated_rho = rotation.dag() * rho * rotation

    populations = np.asarray(np.real_if_close(np.diag(rotated_rho.full())), dtype=float)
    populations = np.clip(populations, 0.0, None)

    total_population = populations.sum()
    if total_population > 0:
        populations = populations / total_population

    return populations


def otoc_from_populations(m_vals, probs, eps):
    """
    Calculate OTOC from population distribution.
    
    OTOC is computed as F = sum_m P(m) * exp(i * eps * m)
    
    Parameters
    ----------
    m_vals : np.ndarray
        Eigenvalues of the angular momentum.
    probs : np.ndarray
        Population probabilities.
    eps : float
        Effective perturbation parameter.
        
    Returns
    -------
    complex
        Complex OTOC amplitude before taking the modulus squared.
    contrib : np.ndarray
        Individual contributions.
    """
    contrib = probs * np.exp(1j * eps * m_vals)
    F = np.sum(contrib)
    return F, contrib


def otoc_trajectory(states, theta, phi, eps, j):
    """
    Compute the OTOC growth and basis populations for a state trajectory.

    Parameters
    ----------
    states : Sequence[qutip.Qobj]
        List of quantum states or density matrices.
    theta, phi : float
        Spherical angles defining the measurement axis.
    eps : float
        Perturbation strength in the OTOC phase factor.
    j : float
        Angular momentum quantum number.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        OTOC growth values C(t) = 1 - |F(t)|^2, populations in the tilted basis,
        and the ordered magnetic quantum numbers.
    """
    m_vals = magnetic_quantum_numbers(j)
    population_history = []
    otoc_growth = []

    for state in states:
        probs = probs_in_Hprime_basis(state, theta, phi, j)
        F, _ = otoc_from_populations(m_vals, probs, eps)
        population_history.append(probs)
        growth = 1.0 - float(np.abs(F) ** 2)
        otoc_growth.append(float(np.clip(np.real_if_close(growth), 0.0, 1.0)))

    return np.asarray(otoc_growth), np.asarray(population_history), m_vals

# ============================================================================
# Utility Functions
# ============================================================================

def normalize_data(data):
    """Normalize data to [0, 1] range."""
    data = np.asarray(data)
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data, dtype=float)
    return (data - data_min) / (data_max - data_min)


def smooth_data(data, window_size=5):
    """Smooth data using moving average."""
    data = np.asarray(data)
    if window_size < 1:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def downsample_data(data, factor=5):
    """Downsample data by averaging."""
    data = np.asarray(data)
    new_len = len(data) // factor
    return np.mean(data[:new_len*factor].reshape(new_len, factor), axis=1)


def calculate_statistics(data):
    """Calculate basic statistics."""
    data = np.asarray(data)
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
    }


def find_peaks_simple(data, threshold=0.5):
    """Find peaks in data using simple threshold."""
    data = np.asarray(data)
    above_threshold = data > threshold
    diff = np.diff(above_threshold.astype(int))
    peaks = np.where(diff == 1)[0] + 1
    return peaks
