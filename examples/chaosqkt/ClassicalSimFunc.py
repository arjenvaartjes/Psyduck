"""Helper functions for classical and quantum chaos simulations on top of psyduck.

This module is a thin layer above ``psyduck``:

* Classical / phase-space helpers (kicked-top map, sphere geometry, periodic
  orbit finding) are kept here because they operate on plain ``numpy``
  3-vectors and have no analogue inside psyduck.
* Anything that builds a quantum state, propagates it, or analyses a Floquet
  spectrum delegates to ``psyduck`` rather than re-implementing the wheel.

Redundant wrappers that used to live in this file have been commented out and
labelled with the psyduck replacement to use.
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from typing import Tuple, List

from psyduck import Spin
from psyduck.hamiltonians import Hz_order


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
#
# These operate on plain numpy 3-vectors (NOT on QuTiP Qobj operators).
# ``psyduck.operations.global_rotation`` builds the QUANTUM rotation in a
# spin-J Hilbert space, which is a different object, so these are not
# redundant with anything in psyduck.
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
    """Single step of the generalized kicked-top map.

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


def iterate_kicked_top(seeds_cart: np.ndarray, kappa: float, alpha: float,
                       order: int, n_iter: int, n_discard: int) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate the kicked-top map on an array of seeds and return angle clouds.

    Used by Section 4 of the notebook (static Poincare section, animation,
    3-D sweep).  Replaces the per-cell ``run_map`` / ``_run_map_3d`` closures.

    Parameters
    ----------
    seeds_cart : ndarray, shape (n_seeds, 3)
        Unit vectors of the initial conditions.
    kappa, alpha, order : float, float, int
        Kicked-top parameters.
    n_iter, n_discard : int, int
        Total iterations and number of transient steps to drop.

    Returns
    -------
    phi_arr, theta_arr : ndarray, shape (n_iter - n_discard, n_seeds)
        Per-iteration azimuthal and polar angles for every seed.
    """
    v = seeds_cart.copy()
    n_seeds = v.shape[0]
    n_keep = n_iter - n_discard
    phi_arr = np.empty((n_keep, n_seeds))
    theta_arr = np.empty((n_keep, n_seeds))
    for i in range(n_iter):
        v = kicked_top_step(v, kappa, alpha, order)
        if i >= n_discard:
            phi_arr[i - n_discard], theta_arr[i - n_discard] = spherical_angles(v)
    return phi_arr, theta_arr


def generate_poincare_section(kappa: float, alpha: float, order: int = 2,
                              n_seeds_phi: int = 12, n_seeds_theta: int = 12,
                              n_iter: int = 1500, n_discard: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Poincare section for the classical kicked top.

    Thin wrapper around :func:`iterate_kicked_top` that also builds a default
    seed grid (uniform in phi, slightly inset in theta).

    Returns
    -------
    phis, thetas : ndarray
        Flattened phase-space coordinates of every recorded trajectory point.
    """
    phis = np.linspace(-np.pi, np.pi, n_seeds_phi, endpoint=False)
    thetas = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_seeds_theta)

    seeds_cart = np.stack(
        [sph_to_cart(th, ph) for th in thetas for ph in phis], axis=0
    )

    phi_arr, theta_arr = iterate_kicked_top(
        seeds_cart, kappa, alpha, order, n_iter, n_discard
    )
    return phi_arr.ravel(), theta_arr.ravel()


def make_poincare_axes(fig, projection: str = 'rectangular'):
    """Configure a matplotlib axes for a Poincare-section scatter.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Parent figure.
    projection : {'rectangular', 'hammer'}
        Axis style.  Rectangular plots (phi, theta) on the unit rectangle with
        theta=0 at the top; Hammer is the equal-area spherical projection.
    """
    if projection == 'hammer':
        ax = fig.add_subplot(111, projection='hammer')
        ax.set_xticks([])
    elif projection == 'rectangular':
        ax = fig.add_subplot(111)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(np.pi, 0)
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\theta$')
    else:
        raise ValueError(
            f"projection must be 'rectangular' or 'hammer', got {projection!r}"
        )
    ax.grid(True, alpha=0.3, linestyle=':')
    return ax


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
    """Refine a periodic orbit using Newton's method."""
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
                         seed_tol: float = 5e-2, refine_tol: float = 1e-10
                         ) -> List[List[Tuple[float, float]]]:
    """Find all period-p orbits of the kicked top."""
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
# Floquet spectrum helpers (use psyduck.hamiltonians.Hz_order under the hood)
# ============================================================================

# --- REMOVED: floquet_phases / level_spacings ------------------------------
# Inlined directly into ClassicalSimulations.ipynb (Section 3.3, kappa sweep
# cell).  Re-enable only if other notebooks need them.
#
# def floquet_phases(U_free: qt.Qobj, kappa: float, order: int, I: float) -> np.ndarray:
#     """Eigenphases of the kicked-top Floquet operator U_F = U_kick(kappa) @ U_free."""
#     H_kick = Hz_order(kappa, order, I)
#     U_kick = (-1j * H_kick).expm()
#     U_F = U_kick @ U_free
#     eigvals, _ = U_F.eigenstates()
#     eigvals = np.array(eigvals, dtype=complex)
#     return np.sort(np.mod(np.angle(eigvals), 2 * np.pi))
#
#
# def level_spacings(phases: np.ndarray) -> np.ndarray:
#     """Unfolded nearest-neighbour spacings s = Delta_omega / <Delta_omega>."""
#     gaps = np.diff(np.sort(phases))
#     return gaps / gaps.mean()


def classify_floquet_eigenphases(phases: np.ndarray, n_sectors: int = 4,
                                 tolerance: float = 0.4) -> List[List[float]]:
    """Group Floquet eigenphases into symmetry sectors."""
    phases = np.mod(phases, 2 * np.pi)
    sorted_idx = np.argsort(phases)
    phases_sorted = phases[sorted_idx]

    groups = [[] for _ in range(n_sectors)]
    ref_phase = phases_sorted[0]

    for ph in phases_sorted:
        step = int(
            np.round(((ph - ref_phase) % (2 * np.pi)) / (2 * np.pi / n_sectors))
        ) % n_sectors
        groups[step].append(ph)

    return groups


# ============================================================================
# Quantum Scar Analysis  (delegates state preparation to psyduck)
# ============================================================================

# --- REMOVED: redundant wrapper around qt.spin_coherent --------------------
# Use ``psyduck.Spin(I=I).make_displaced_coherent_state(theta, phi).state``
# instead -- it produces the same physical state (overlap = 1 to machine
# precision in the tests performed during this migration).
#
# def spin_coherent_state(I: float, theta: float, phi: float) -> qt.Qobj:
#     """Generate a spin-coherent state |I, theta, phi>."""
#     return qt.spin_coherent(I, theta, phi)


def overlap_with_classical_orbit(eigvec: qt.Qobj, I: float,
                                 orbit_points: List[Tuple[float, float]]) -> float:
    """Sum of |<coherent_k|eigvec>|^2 over each (theta, phi) on a classical orbit.

    Builds each coherent state via psyduck's ``Spin.make_displaced_coherent_state``
    so this module no longer calls ``qt.spin_coherent`` directly.
    """
    nucleus = Spin(I=I)
    total = 0.0
    for theta, phi in orbit_points:
        nucleus.make_displaced_coherent_state(theta, phi)
        total += abs(nucleus.state.overlap(eigvec)) ** 2
    return total


def compute_scar_overlaps(eigvecs: List[qt.Qobj], I: float,
                          orbit_points: List[Tuple[float, float]]) -> np.ndarray:
    """Scar-style overlap with a classical orbit, evaluated for every eigvec."""
    return np.array(
        [overlap_with_classical_orbit(e, I, orbit_points) for e in eigvecs]
    )


# Note: Husimi Q-function plotting lives in psyduck.plotting.wigner_plot
# (``wigner_plot_hammer`` with ``prob_function='husimi'`` and friends).


# ============================================================================
# Time-Series Analysis
# ============================================================================

def fft_fidelity_spectrum(fidelity: np.ndarray, times: np.ndarray,
                          zero_pad_factor: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """FFT spectrum of a fidelity trace with zero-padding."""
    F_c = fidelity - np.mean(fidelity)
    pad = (len(F_c) - 1) * (zero_pad_factor - 1)
    F_pad = np.pad(F_c, (0, pad), 'constant')

    fft_vals = np.fft.fft(F_pad)
    freqs = np.fft.fftfreq(len(F_pad), d=np.mean(np.diff(times)))

    mask = freqs > 0
    freqs = freqs[mask]
    spectrum = np.abs(fft_vals[mask])
    spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    return freqs, spectrum


# ============================================================================
# State Trajectory Visualization Helpers
# ============================================================================

def generate_classical_trajectory(x0: np.ndarray, kappa: float, alpha: float,
                                  order: int, n_steps: int) -> np.ndarray:
    """Generate a single-seed classical trajectory on the unit sphere."""
    traj = [x0]
    v = x0.copy()
    for _ in range(n_steps):
        v = kicked_top_step(v, kappa, alpha, order)
        traj.append(v.copy())
    return np.array(traj)
