"""Helper functions for classical and quantum chaos simulations using psyduck framework.

This module provides utilities for:
- Classical kicked-top phase space dynamics
- Poincaré section generation
- Periodic orbit finding and classification
- Floquet analysis helpers
- Quantum scar detection
- Time-series analysis (FFT, fidelity traces)
"""

import numpy as np
import qutip as qt
from typing import Tuple, List, Callable


# ============================================================================
# Classical Kicked-Top Dynamics (Phase Space Helpers)
# ============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    return v / np.linalg.norm(v)


def sph_to_cart(theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates (theta, phi) to Cartesian (x, y, z)."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])


def cart_to_sph(v: np.ndarray) -> Tuple[float, float]:
    """Convert Cartesian coordinates (x, y, z) to spherical (theta, phi)."""
    v = normalize(v)
    theta = np.arccos(np.clip(v[2], -1, 1))
    phi = np.arctan2(v[1], v[0])
    return theta, phi


def spherical_angles(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract spherical angles from Cartesian vector(s).
    
    Parameters
    ----------
    v : ndarray
        Cartesian vector or array of vectors, shape (..., 3)
    
    Returns
    -------
    phi, theta : ndarray
        Azimuthal and polar angles
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return phi, theta


def geodesic_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Compute geodesic distance between two points on the unit sphere."""
    u = normalize(u)
    v = normalize(v)
    dot = np.clip(np.dot(u, v), -1, 1)
    return np.arccos(dot)


# ============================================================================
# SO(3) Rotation Helpers
# ============================================================================

def rotate_x(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector(s) about x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    x2 = x
    y2 = c * y - s * z
    z2 = s * y + c * z
    out = np.stack([x2, y2, z2], axis=-1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def rotate_y(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector(s) about y-axis."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    x2 = c * x + s * z
    y2 = y
    z2 = -s * x + c * z
    out = np.stack([x2, y2, z2], axis=-1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def rotate_z(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector(s) about z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    x2 = c * x - s * y
    y2 = s * x + c * y
    z2 = z
    out = np.stack([x2, y2, z2], axis=-1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


# ============================================================================
# Classical Kicked-Top Dynamics
# ============================================================================

def kicked_top_step(v: np.ndarray, kappa: float, alpha: float, order: int = 2) -> np.ndarray:
    """
    Single step of the generalized kicked-top map.
    
    Applies: v1 = R_z(-kappa*sign(z)*|z|^(order-1)) @ v
             v2 = R_y(-alpha) @ v1
    
    Parameters
    ----------
    v : ndarray
        Current state vector on unit sphere, shape (3,) or (N, 3)
    kappa : float
        Twist strength parameter
    alpha : float
        Rotation angle about y-axis
    order : int
        Nonlinearity order (typically 2 or 3)
    
    Returns
    -------
    v_new : ndarray
        Updated state vector(s), same shape as input
    """
    z = v[..., 2]
    angle = kappa * np.sign(z) * (np.abs(z) ** (order - 1))
    v1 = rotate_z(v, -angle)
    v2 = rotate_y(v1, -alpha)
    return v2


def generate_poincare_section(kappa: float, alpha: float, order: int = 2,
                             n_seeds_phi: int = 12, n_seeds_theta: int = 12,
                             n_iter: int = 1500, n_discard: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Poincaré section for the classical kicked top.
    
    Parameters
    ----------
    kappa : float
        Twist strength parameter
    alpha : float
        Rotation angle per kick
    order : int
        Nonlinearity order
    n_seeds_phi : int
        Number of azimuthal seeds
    n_seeds_theta : int
        Number of polar seeds
    n_iter : int
        Total number of iterations per seed
    n_discard : int
        Number of initial iterations to discard (transient)
    
    Returns
    -------
    phis, thetas : ndarray
        Phase space coordinates of trajectory points
    """
    phis = np.linspace(-np.pi, np.pi, n_seeds_phi, endpoint=False)
    thetas = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_seeds_theta)
    
    seeds = []
    for th in thetas:
        for ph in phis:
            seeds.append(sph_to_cart(th, ph))
    v = np.stack(seeds, axis=0)
    
    pts_phi, pts_theta = [], []
    for i in range(n_iter):
        v = np.array([kicked_top_step(vi, kappa, alpha, order) for vi in v])
        if i >= n_discard:
            for vi in v:
                ph, th = spherical_angles(vi)
                pts_phi.append(ph)
                pts_theta.append(th)
    
    return np.array(pts_phi), np.array(pts_theta)


# ============================================================================
# Periodic Orbit Finding
# ============================================================================

def floquet_map(v: np.ndarray, kappa: float, alpha: float, order: int, p: int) -> np.ndarray:
    """Apply the period-p Floquet map."""
    out = normalize(v)
    for _ in range(p):
        out = kicked_top_step(out, kappa, alpha, order)
    return normalize(out)


def residual_floquet(v: np.ndarray, kappa: float, alpha: float, order: int, p: int) -> np.ndarray:
    """Residual for periodic orbit finding: F_p(v) - v."""
    return floquet_map(v, kappa, alpha, order, p) - normalize(v)


def numerical_jacobian_floquet(v: np.ndarray, kappa: float, alpha: float, 
                               order: int, p: int, eps: float = 1e-6) -> np.ndarray:
    """Compute Jacobian of the Floquet map via finite differences."""
    J = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1
        f_plus = floquet_map(normalize(v + eps * e), kappa, alpha, order, p)
        f_minus = floquet_map(normalize(v - eps * e), kappa, alpha, order, p)
        J[:, j] = (f_plus - f_minus) / (2 * eps)
    return J


def refine_periodic_orbit(v0: np.ndarray, kappa: float, alpha: float, order: int, p: int,
                          maxit: int = 50, tol: float = 1e-12) -> Tuple[np.ndarray, bool]:
    """
    Refine a periodic orbit using Newton's method.
    
    Returns
    -------
    v_refined : ndarray
        Refined periodic orbit point
    converged : bool
        Whether the refinement converged
    """
    v = normalize(v0)
    for _ in range(maxit):
        r = residual_floquet(v, kappa, alpha, order, p)
        if np.linalg.norm(r) < tol:
            return normalize(v), True
        J = numerical_jacobian_floquet(v, kappa, alpha, order, p)
        A = J - np.eye(3)
        try:
            delta = np.linalg.solve(A, r)
        except np.linalg.LinAlgError:
            return normalize(v), False
        v = normalize(v - delta)
    return normalize(v), np.linalg.norm(residual_floquet(v, kappa, alpha, order, p)) < tol


def deduplicate_orbits(points: List[np.ndarray], angle_tol: float = 1e-3) -> List[np.ndarray]:
    """Remove duplicate periodic orbits based on geodesic distance."""
    uniq = []
    for v in points:
        if all(geodesic_angle(v, u) > angle_tol for u in uniq):
            uniq.append(v)
    return uniq


def find_period_p_orbits(kappa: float, alpha: float, order: int, p: int,
                        grid_th: int = 41, grid_ph: int = 81,
                        seed_tol: float = 5e-2, refine_tol: float = 1e-10) -> List[List[Tuple[float, float]]]:
    """
    Find all period-p orbits of the kicked top.
    
    Returns
    -------
    cycles : list of list of (theta, phi) tuples
        Each element is a periodic orbit represented in spherical coordinates
    """
    thetas = np.linspace(0, np.pi, grid_th)
    phis = np.linspace(-np.pi, np.pi, grid_ph)
    seeds = []
    for th in thetas:
        for ph in phis:
            v0 = sph_to_cart(th, ph)
            if np.linalg.norm(residual_floquet(v0, kappa, alpha, order, p)) < seed_tol:
                seeds.append(v0)
    
    refined = []
    for s in seeds:
        v_ref, ok = refine_periodic_orbit(s, kappa, alpha, order, p, tol=refine_tol)
        if ok:
            refined.append(v_ref)
    
    uniq = deduplicate_orbits(refined, angle_tol=1e-3)
    
    cycles = []
    used = [False] * len(uniq)
    for i, v in enumerate(uniq):
        if used[i]:
            continue
        orb = [v]
        cur = v
        for _ in range(1, p):
            cur = kicked_top_step(cur, kappa, alpha, order)
            j_best, ang_best = None, 1e9
            for j, u in enumerate(uniq):
                ang = geodesic_angle(cur, u)
                if ang < ang_best:
                    ang_best, j_best = ang, j
            if ang_best < 1e-2:
                orb.append(uniq[j_best])
                used[j_best] = True
        if len(orb) == p:
            cycles.append([[cart_to_sph(v) for v in orb]])
    
    return cycles


# ============================================================================
# Quantum Scar Analysis
# ============================================================================

def spin_coherent_state(I: float, theta: float, phi: float) -> qt.Qobj:
    """Generate a spin-coherent state |I, theta, phi>."""
    return qt.spin_coherent(I, theta, phi)


def overlap_with_classical_orbit(eigvec: qt.Qobj, I: float, orbit_points: List[Tuple[float, float]]) -> float:
    """
    Compute overlap of eigenvector with classical periodic orbit.
    
    Parameters
    ----------
    eigvec : qt.Qobj
        Quantum eigenstate
    I : float
        Spin quantum number
    orbit_points : list of (theta, phi) tuples
        Classical periodic orbit points
    
    Returns
    -------
    overlap : float
        Sum of squared overlaps with coherent states at orbit points
    """
    coh_states = [spin_coherent_state(I, th, ph) for th, ph in orbit_points]
    return sum(abs(coh.overlap(eigvec)) ** 2 for coh in coh_states)


def compute_scar_overlaps(eigvecs: List[qt.Qobj], I: float, orbit_points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute overlap of all eigenstates with a classical orbit.
    
    Returns
    -------
    overlaps : ndarray, shape (len(eigvecs),)
        Scar overlap for each eigenstate
    """
    return np.array([overlap_with_classical_orbit(eigvec, I, orbit_points) for eigvec in eigvecs])


# Note: Husimi Q-function is computed using QuTiP's spin_q_function from qutip.wigner
# See psyduck.plotting.wigner_plot_hammer and related functions for visualization


# ============================================================================
# Time-Series Analysis
# ============================================================================

def fft_fidelity_spectrum(fidelity: np.ndarray, times: np.ndarray, 
                          zero_pad_factor: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT spectrum of fidelity trace with zero-padding.
    
    Parameters
    ----------
    fidelity : ndarray
        Fidelity values F(t) = |<psi(0)|psi(t)>|^2
    times : ndarray
        Time points
    zero_pad_factor : int
        Zero-padding multiplier for better frequency resolution
    
    Returns
    -------
    freqs, spectrum : ndarray
        Frequency axis and normalized power spectrum
    """
    # Remove DC offset
    F_c = fidelity - np.mean(fidelity)
    
    # Zero-pad
    pad = (len(F_c) - 1) * (zero_pad_factor - 1)
    F_pad = np.pad(F_c, (0, pad), 'constant')
    
    # FFT
    fft_vals = np.fft.fft(F_pad)
    freqs = np.fft.fftfreq(len(F_pad), d=np.mean(np.diff(times)))
    
    # Keep positive frequencies and normalize
    mask = freqs > 0
    freqs = freqs[mask]
    spectrum = np.abs(fft_vals[mask])
    spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    
    return freqs, spectrum


# ============================================================================
# Floquet Eigenstate Classification
# ============================================================================

def classify_floquet_eigenphases(phases: np.ndarray, n_sectors: int = 4, 
                                 tolerance: float = 0.4) -> List[List[float]]:
    """
    Group Floquet eigenphases into symmetry sectors.
    
    Assumes phases are roughly equally spaced into n_sectors bins.
    
    Parameters
    ----------
    phases : ndarray
        Eigenphases (rad), shape (dim,)
    n_sectors : int
        Number of sectors to group into
    tolerance : float
        Tolerance for grouping (rad)
    
    Returns
    -------
    sectors : list of list
        Eigenphases grouped by sector
    """
    phases = np.mod(phases, 2 * np.pi)
    sorted_idx = np.argsort(phases)
    phases_sorted = phases[sorted_idx]
    
    groups = [[] for _ in range(n_sectors)]
    ref_phase = phases_sorted[0]
    
    for ph in phases_sorted:
        step = int(np.round(((ph - ref_phase) % (2 * np.pi)) / (2 * np.pi / n_sectors))) % n_sectors
        groups[step].append(ph)
    
    return groups


# ============================================================================
# State Trajectory Visualization Helpers
# ============================================================================

def generate_classical_trajectory(x0: np.ndarray, kappa: float, alpha: float, 
                                 order: int, n_steps: int) -> np.ndarray:
    """
    Generate a classical trajectory on the phase space.
    
    Returns
    -------
    trajectory : ndarray, shape (n_steps+1, 3)
        Cartesian coordinates of the classical trajectory
    """
    traj = [x0]
    v = x0.copy()
    for _ in range(n_steps):
        v = kicked_top_step(v, kappa, alpha, order)
        traj.append(v.copy())
    return np.array(traj)
