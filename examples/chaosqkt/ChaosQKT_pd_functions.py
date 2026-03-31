"""
ChaosQKT Psyduck Helper Functions

This module provides wrapper functions and implementations for quantum chaos
simulations using the psyduck framework. It fills the gap between the psyduck
library and the full functionality needed for ChaosQKT tutorials.

Functions include:
- Spin state initialization and manipulation
- Kicked dynamics simulation
- Wigner function and phase space visualization
- Data fitting functions
- Utility functions for data processing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy import stats
import qutip as qt
from qutip import jmat, basis, expect, ket2dm, Qobj


# ============================================================================
# State Operations
# ============================================================================

def create_spin_operators(j):
    """
    Create spin operators for a given quantum number.
    
    Parameters
    ----------
    j : float
        Angular momentum quantum number (spin).
        
    Returns
    -------
    dict
        Dictionary containing 'Jx', 'Jy', 'Jz' operators.
    """
    Jx = jmat(j, 'x')
    Jy = jmat(j, 'y')
    Jz = jmat(j, 'z')
    
    return {
        'Jx': Jx,
        'Jy': Jy,
        'Jz': Jz,
    }


def Displace(theta, phi, psi, j):
    """
    Apply displacement operator to a quantum state.
    
    Implements: D(theta, phi) = exp(theta/2 * (e^{i*phi} * J_- - e^{-i*phi} * J_+))
    
    Parameters
    ----------
    theta : float
        Displacement angle (colatitude).
    phi : float
        Azimuthal angle.
    psi : qutip.Qobj
        Initial quantum state (ket).
    j : float
        Angular momentum quantum number (spin).
        
    Returns
    -------
    psi_displaced : qutip.Qobj
        Displaced quantum state.
    """
    Ip = jmat(j, '+')
    Im = jmat(j, '-')
    
    D = (theta / 2 * (np.exp(1j * phi) * Im - np.exp(-1j * phi) * Ip)).expm()
    return D * psi


def initial_state(theta, phi, j):
    """
    Create an initial spin-coherent state with displacement.
    
    Parameters
    ----------
    theta : float
        Colatitude angle.
    phi : float
        Azimuthal angle.
    j : float
        Angular momentum quantum number.
        
    Returns
    -------
    psi : qutip.Qobj
        Spin-coherent state.
    """
    dim = int(2 * j + 1)
    psi0 = basis(dim, 0)
    psi_normal = Displace(theta, phi, psi0, j)
    return psi_normal


def qudit_exp(psi, j):
    """
    Calculate expectation values of angular momentum operators.
    
    Parameters
    ----------
    psi : qutip.Qobj
        Quantum state.
    j : float
        Angular momentum quantum number.
        
    Returns
    -------
    list
        [<Jx>, <Jy>, <Jz>]
    """
    Ix = jmat(j, 'x')
    Iy = jmat(j, 'y')
    Iz = jmat(j, 'z')
    
    exp_x = expect(Ix, psi)
    exp_y = expect(Iy, psi)
    exp_z = expect(Iz, psi)
    
    return [exp_x, exp_y, exp_z]


def qudit_linear_entropy(psi, j):
    """
    Calculate linear entropy of a quantum state.
    
    S = 1 - Tr(rho^2) (for pure states normalized to 0)
    
    Parameters
    ----------
    psi : qutip.Qobj
        Quantum state.
    j : float
        Angular momentum quantum number.
        
    Returns
    -------
    float
        Linear entropy value.
    """
    rho = ket2dm(psi)
    return 1.0 - expect(rho * rho, psi)


# ============================================================================
# Quantum Evolution - Kicked Dynamics
# ============================================================================

def kicked_dynamics(psi_initial, tau, kappa, j, N=1, order=2, pulse_type='pulse'):
    """
    Simulate stroboscopic kicked dynamics of a spin system.
    
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
    j : float
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
    S = j
    dim = int(2 * S + 1)
    chi = kappa / 2
    
    # Create spin operators
    Jx = jmat(S, 'x')
    Jy = jmat(S, 'y')
    Jz = jmat(S, 'z')
    
    # Free Hamiltonian H0
    H0 = (np.pi / 2) * (-Jy)
    
    # Precompute the free evolution operator
    U0 = (-1j * H0 * tau).expm()
    
    # Larmor pulse (instantaneous)
    Ularmor = (-1j * chi * Jz).expm()
    
    # Nonlinear kick unitary
    Upulse = (-1j * Jz**order * kappa / (order * S**(order - 1))).expm()
    
    # Evolve the state stroboscopically
    psi = psi_initial.copy()
    psi_list = [psi.copy()]
    overlap_list = [psi.overlap(psi_initial)]
    entropy_list = [qudit_linear_entropy(psi, j)]
    exp_list = [qudit_exp(psi, j)]
    
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
        entropy_list.append(qudit_linear_entropy(psi, j))
        exp_list.append(qudit_exp(psi, j))
    
    return psi_list, overlap_list, entropy_list, exp_list


# ============================================================================
# OTOC-related functions
# ============================================================================

def probs_in_Hprime_basis(rho, theta, phi, j):
    """
    Calculate population probabilities in a tilted basis.
    
    Parameters
    ----------
    rho : qutip.Qobj
        Density matrix.
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
    # Create rotation to new basis defined by (theta, phi)
    Ix = jmat(j, 'x')
    Iy = jmat(j, 'y')
    Iz = jmat(j, 'z')
    
    # Unit vector
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    
    # Total angular momentum along direction
    J_n = nx * Ix + ny * Iy + nz * Iz
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = J_n.eigenstates()
    
    # Calculate populations
    dim = int(2 * j + 1)
    probs = np.zeros(dim)
    
    for i in range(dim):
        psi_i = eigenvectors[i]
        probs[i] = expect(psi_i * psi_i.dag(), rho)
    
    return probs


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
    float
        OTOC (Loschmidt echo) value.
    contrib : np.ndarray
        Individual contributions.
    """
    contrib = probs * np.exp(1j * eps * m_vals)
    F = np.sum(contrib)
    return F, contrib


# ============================================================================
# Plotting Functions
# Note: wigner_plot and wigner_plot_hammer are provided by psyduck.plotting
# These custom plotting functions extend/complement psyduck functionality
# ============================================================================


def tomo_plot_3d(psi, kind='wigner', n_theta=50, n_phi=100, add_projections=True):
    """
    Plot 3D Bloch sphere with state tomography.
    
    Parameters
    ----------
    psi : qutip.Qobj
        Quantum state (ket).
    kind : str, optional
        Type: 'wigner' or 'husimi' (default: 'wigner').
    n_theta : int, optional
        Number of theta points (default: 50).
    n_phi : int, optional
        Number of phi points (default: 100).
    add_projections : bool, optional
        Whether to add projection planes (default: True).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        3D axes object.
    """
    theta = np.linspace(0, np.pi, num=n_theta, endpoint=True)
    phi = np.linspace(-np.pi, np.pi, num=n_phi, endpoint=True)
    
    if kind.lower() == 'wigner':
        data, theta_m, phi_m = qt.spin_wigner(psi, theta, phi)
    else:
        data, theta_m, phi_m = qt.spin_q_function(psi, theta, phi)
    
    # Create 3D surface
    r = 1
    x = r * np.cos(phi_m) * np.cos(theta_m - np.pi / 2)
    y = r * np.sin(phi_m) * np.cos(theta_m - np.pi / 2)
    z = r * np.sin(theta_m - np.pi / 2)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    cmap = mpl.cm.bwr
    norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())
    ax.plot_surface(x, y, z, facecolors=cmap(norm(data)), 
                    rstride=1, cstride=1, shade=False, alpha=0.9)
    
    # Add projection planes
    if add_projections:
        xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        # xy plane (z=0)
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')
        # xz plane (y=0)
        ax.plot_surface(xx, np.zeros_like(xx), yy, alpha=0.1, color='gray')
        # yz plane (x=0)
        ax.plot_surface(np.zeros_like(yy), xx, yy, alpha=0.1, color='gray')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    return fig, ax


def plot_phase_space_comparison(states, title_list=None, n_points=80):
    """
    Visualize multiple states side-by-side on Hammer projection.
    
    Parameters
    ----------
    states : list
        List of quantum states.
    title_list : list, optional
        List of titles for each subplot.
    n_points : int, optional
        Number of grid points (default: 80).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : np.ndarray
        Array of axes objects.
    """
    n_states = len(states)
    n_cols = min(n_states, 4)
    n_rows = int(np.ceil(n_states / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows),
                             subplot_kw=dict(projection='hammer'))
    
    if n_states == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    theta = np.linspace(0, np.pi, num=n_points, endpoint=True)
    phi = np.linspace(-np.pi, np.pi, num=2*n_points, endpoint=True)
    
    for i, state in enumerate(states):
        data, theta_m, phi_m = qt.spin_wigner(state, theta, phi)
        
        pc = axes[i].pcolormesh(phi_m, theta_m - np.pi / 2, data, cmap='bwr')
        axes[i].set_xticklabels([])
        axes[i].grid(False)
        
        if title_list is not None and i < len(title_list):
            axes[i].set_title(title_list[i], fontsize=11, fontweight='bold')
        else:
            axes[i].set_title(f'State {i}', fontsize=11, fontweight='bold')
        
        fig.colorbar(pc, ax=axes[i], label='Wigner')
    
    # Hide unused subplots
    for i in range(n_states, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    return fig, axes[:n_states]


# ============================================================================
# Fitting Functions
# ============================================================================

def rabi_oscillation(time, A, f, T2, d):
    """
    Rabi oscillation fit function with decay.
    
    Formula: A/2 * (1 - cos(2*pi*f*t) * exp(-(t/T2)^2)) + d
    
    Parameters
    ----------
    time : float or np.ndarray
        Time value(s).
    A : float
        Amplitude (visibility).
    f : float
        Rabi frequency (Hz).
    T2 : float
        Decay time constant.
    d : float
        Baseline offset.
        
    Returns
    -------
    float or np.ndarray
        Fitted value(s).
    """
    T2 = max(T2, 1e-12)
    return A / 2 * (1 - np.cos(2 * np.pi * f * time) * np.exp(-(time / T2)**2)) + d


def exponential_decay(time, amplitude, tau, offset=0):
    """
    Simple exponential decay function.
    
    Formula: amplitude * exp(-t/tau) + offset
    
    Parameters
    ----------
    time : float or np.ndarray
        Time value(s).
    amplitude : float
        Initial amplitude.
    tau : float
        Time constant.
    offset : float, optional
        Baseline offset (default: 0).
        
    Returns
    -------
    float or np.ndarray
        Fitted value(s).
    """
    return amplitude * np.exp(-time / tau) + offset


def gaussian(x, A, mu, sigma, offset=0):
    """
    Gaussian function.
    
    Formula: A * exp(-(x-mu)^2 / (2*sigma^2)) + offset
    
    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).
    A : float
        Amplitude.
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    offset : float, optional
        Vertical offset (default: 0).
        
    Returns
    -------
    float or np.ndarray
        Fitted value(s).
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset


def lorentzian(x, A, x0, gamma, offset=0):
    """
    Lorentzian function.
    
    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).
    A : float
        Amplitude.
    x0 : float
        Center.
    gamma : float
        FWHM/2 (half-width at half-max).
    offset : float, optional
        Vertical offset (default: 0).
        
    Returns
    -------
    float or np.ndarray
        Fitted value(s).
    """
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + offset


def fit_rabi_oscillation(time_data, signal_data, p0=None):
    """
    Fit data to Rabi oscillation model.
    
    Parameters
    ----------
    time_data : np.ndarray
        Time values.
    signal_data : np.ndarray
        Signal values.
    p0 : list, optional
        Initial parameter guess [A, f, T2, d].
        
    Returns
    -------
    dict or None
        Dictionary with keys 'params', 'fitted_curve', 'residuals', 'r_squared',
        or None if fit fails.
    """
    try:
        if p0 is None:
            p0 = [0.9, 1e5, np.max(time_data)/3, 0.05]
        
        popt, _ = curve_fit(rabi_oscillation, time_data, signal_data, p0=p0, maxfev=5000)
        fitted_curve = rabi_oscillation(time_data, *popt)
        residuals = signal_data - fitted_curve
        r_squared = 1 - np.sum(residuals**2) / np.sum((signal_data - np.mean(signal_data))**2)
        
        return {
            'params': popt,
            'fitted_curve': fitted_curve,
            'residuals': residuals,
            'r_squared': r_squared,
        }
    except Exception as e:
        print(f"Fit failed: {e}")
        return None


def fit_exponential_decay(time_data, signal_data, p0=None):
    """
    Fit data to exponential decay model.
    
    Parameters
    ----------
    time_data : np.ndarray
        Time values.
    signal_data : np.ndarray
        Signal values.
    p0 : list, optional
        Initial parameter guess [amplitude, tau, offset].
        
    Returns
    -------
    dict or None
        Dictionary with keys 'params', 'fitted_curve', 'residuals', 'r_squared',
        or None if fit fails.
    """
    try:
        if p0 is None:
            p0 = [signal_data[0], np.max(time_data)/2, 0.1]
        
        popt, _ = curve_fit(exponential_decay, time_data, signal_data, p0=p0, maxfev=5000)
        fitted_curve = exponential_decay(time_data, *popt)
        residuals = signal_data - fitted_curve
        r_squared = 1 - np.sum(residuals**2) / np.sum((signal_data - np.mean(signal_data))**2)
        
        return {
            'params': popt,
            'fitted_curve': fitted_curve,
            'residuals': residuals,
            'r_squared': r_squared,
        }
    except Exception as e:
        print(f"Fit failed: {e}")
        return None


def fit_gaussian(x_data, signal_data, p0=None):
    """
    Fit data to Gaussian model.
    
    Parameters
    ----------
    x_data : np.ndarray
        X values.
    signal_data : np.ndarray
        Signal values.
    p0 : list, optional
        Initial parameter guess [A, mu, sigma, offset].
        
    Returns
    -------
    dict or None
        Fit result dictionary.
    """
    try:
        if p0 is None:
            p0 = [np.max(signal_data), np.mean(x_data), np.std(x_data), 0]
        
        popt, _ = curve_fit(gaussian, x_data, signal_data, p0=p0, maxfev=5000)
        fitted_curve = gaussian(x_data, *popt)
        residuals = signal_data - fitted_curve
        r_squared = 1 - np.sum(residuals**2) / np.sum((signal_data - np.mean(signal_data))**2)
        
        return {
            'params': popt,
            'fitted_curve': fitted_curve,
            'residuals': residuals,
            'r_squared': r_squared,
        }
    except Exception as e:
        print(f"Fit failed: {e}")
        return None


def fit_lorentzian(x_data, signal_data, p0=None):
    """
    Fit data to Lorentzian model.
    
    Parameters
    ----------
    x_data : np.ndarray
        X values.
    signal_data : np.ndarray
        Signal values.
    p0 : list, optional
        Initial parameter guess [A, x0, gamma, offset].
        
    Returns
    -------
    dict or None
        Fit result dictionary.
    """
    try:
        if p0 is None:
            p0 = [np.max(signal_data), np.mean(x_data), np.std(x_data), 0]
        
        popt, _ = curve_fit(lorentzian, x_data, signal_data, p0=p0, maxfev=5000)
        fitted_curve = lorentzian(x_data, *popt)
        residuals = signal_data - fitted_curve
        r_squared = 1 - np.sum(residuals**2) / np.sum((signal_data - np.mean(signal_data))**2)
        
        return {
            'params': popt,
            'fitted_curve': fitted_curve,
            'residuals': residuals,
            'r_squared': r_squared,
        }
    except Exception as e:
        print(f"Fit failed: {e}")
        return None


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
