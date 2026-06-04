"""Classical phase-space helpers for spin systems.

This module hosts the *classical* analogues of the quantum-state propagators
in :mod:`psyduck.evolve` -- in particular the stroboscopic kicked-top map and
its periodic-orbit finder.  Everything here operates on plain ``numpy``
3-vectors on the unit sphere and broadcasts over arbitrary leading batch
dimensions.  Nothing here touches ``qt.Qobj``.

Public API
----------
* :func:`kicked_top_step`     -- one stroboscopic step of the classical
  kicked-top map.
* :func:`find_period_p_orbits` -- enumerate every period-:math:`p` orbit at
  fixed kicked-top parameters (Newton search on top of a coarse grid).

See ``examples/chaosqkt/PERIODIC_ORBITS.md`` for the algorithm derivation
and a literature reading list.
"""

from typing import List, Tuple

import numpy as np
from numpy import ndarray


# ============================================================================
# Sphere rotations (numpy, batched)
# ============================================================================

def _rotate_y(v: ndarray, angle) -> ndarray:
    """Rotate vector(s) about y-axis. ``angle`` may be scalar or broadcastable to v.shape[:-1]."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    out = np.stack([c * x + s * z, y, -s * x + c * z], axis=-1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _rotate_z(v: ndarray, angle) -> ndarray:
    """Rotate vector(s) about z-axis. ``angle`` may be scalar or broadcastable to v.shape[:-1]."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    out = np.stack([c * x - s * y, s * x + c * y, z], axis=-1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


# ============================================================================
# Classical Kicked-Top Map
# ============================================================================

def kicked_top_step(v: ndarray, kappa: float, alpha: float, order: int = 2) -> ndarray:
    """Single step of the generalized classical kicked-top map.

    Applies the twist-then-rotate map
        v1 = R_z(-kappa * sign(z) * |z|^(order-1)) v
        v2 = R_y(-alpha) v1
    on the unit sphere.  Operates on plain numpy 3-vectors, broadcasting over
    leading batch dimensions.

    Parameters
    ----------
    v : ndarray
        Current state vector(s) on the unit sphere, shape (3,) or (..., 3).
    kappa : float
        Twist strength parameter.
    alpha : float
        Rotation angle about the y-axis.
    order : int, optional
        Nonlinearity order of the twist (default 2).

    Returns
    -------
    ndarray
        Updated state vector(s), same shape as ``v``.
    """
    z = v[..., 2]
    angle = kappa * np.sign(z) * (np.abs(z) ** (order - 1))
    v1 = _rotate_z(v, -angle)
    return _rotate_y(v1, -alpha)


# ============================================================================
# Periodic-orbit finding
# ============================================================================

def _kicked_top_residual(kappa: float, alpha: float, order: int, p: int):
    """Return F^p as a closure that broadcasts over leading axes.

    Used internally by :func:`find_period_p_orbits` for both the coarse-grid
    residual scan and the per-seed scipy.optimize.fsolve polish.  Kept private
    -- callers that need it can compose ``kicked_top_step`` directly.
    """

    def normalize(v: ndarray) -> ndarray:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def F_p(v: ndarray) -> ndarray:
        """Apply the kicked-top map p times.  Broadcasts over leading axes."""
        u = normalize(v)
        for _ in range(p):
            u = kicked_top_step(u, kappa, alpha, order)
        return normalize(u)

    return F_p, normalize


def _dedup_unit_vectors(vs: List[ndarray], angle_tol: float) -> List[ndarray]:
    """Cluster unit vectors by geodesic angle; keep one representative per cluster.

    Two unit vectors u, v are merged when arccos(u . v) < angle_tol.  The first
    representative encountered is kept.  Kept private because only the orbit
    finder uses it today -- promote if other callers need it.
    """
    uniq: List[ndarray] = []
    for v in vs:
        if all(np.arccos(np.clip(v @ u, -1.0, 1.0)) > angle_tol for u in uniq):
            uniq.append(v)
    return uniq


def find_period_p_orbits(kappa: float, alpha: float, order: int, p: int,
                         grid_th: int = 41, grid_ph: int = 81,
                         seed_tol: float = 5e-2, refine_tol: float = 1e-10,
                         dedup_angle_tol: float = 1e-3,
                         ) -> List[List[Tuple[float, float]]]:
    """Find every period-p orbit of the classical kicked top.

    A period-p orbit is a set of p unit vectors {v_1, ..., v_p} that maps
    cyclically into itself under the kicked-top map F:
        F(v_i) = v_{(i+1) mod p}.
    Equivalently, each v_i is a fixed point of F^p, the p-fold composition.

    Algorithm (three steps; see ``examples/chaosqkt/PERIODIC_ORBITS.md`` for
    the full derivation and references):

      1. **Coarse grid (batched)** -- sample the sphere on a (grid_th x grid_ph)
         (theta, phi) mesh, apply F^p to every seed in one batched call, and
         keep those with residual ||F^p(v) - v|| < seed_tol.
      2. **Polish (scipy.optimize.fsolve)** -- pass each surviving seed through
         fsolve on r(v) = F^p(v) - v_unit; scipy does the Newton iterations
         and the finite-difference Jacobian internally.
      3. **Dedup + assemble cycles** -- cluster refined fixed points by
         geodesic angle (dedup_angle_tol), then iterate F from each unused
         unique point to enumerate its p-cycle.

    Returns one list of (theta, phi) tuples per cycle; angles are in the
    canonical ranges theta in [0, pi] and phi in [-pi, pi] (arctan2 branch).
    Consecutive orbit points may straddle the phi = +/- pi seam -- use
    ``numpy.unwrap`` on the phi sequence if you want a continuous trace.
    """
    from scipy.optimize import fsolve

    F_p, normalize = _kicked_top_residual(kappa, alpha, order, p)

    # ----- 1. Coarse seed grid (vectorised) --------------------------------
    th, ph = np.meshgrid(np.linspace(0, np.pi, grid_th),
                         np.linspace(-np.pi, np.pi, grid_ph), indexing='ij')
    V = np.stack([np.sin(th) * np.cos(ph),
                  np.sin(th) * np.sin(ph),
                  np.cos(th)], axis=-1).reshape(-1, 3)
    seeds = V[np.linalg.norm(F_p(V) - V, axis=-1) < seed_tol]

    # ----- 2. Polish each seed with scipy.optimize.fsolve -------------------
    refined: List[ndarray] = []
    for s in seeds:
        v, _, ier, _ = fsolve(lambda x: F_p(x) - normalize(x),
                              s, xtol=refine_tol, full_output=True)
        v = normalize(v)
        if ier == 1 and np.linalg.norm(F_p(v) - v) < 1e2 * refine_tol:
            refined.append(v)

    # ----- 3. Deduplicate, then assemble cycles -----------------------------
    uniq = _dedup_unit_vectors(refined, dedup_angle_tol)

    def to_sph(c: ndarray) -> Tuple[float, float]:
        return float(np.arccos(np.clip(c[2], -1.0, 1.0))), float(np.arctan2(c[1], c[0]))

    cycles: List[List[Tuple[float, float]]] = []
    used = set()
    for i, v in enumerate(uniq):
        if i in used:
            continue
        # Enumerate the cycle by raw iteration: c_k = F^k(v) for k = 0..p-1.
        cycle = [v]
        for _ in range(p - 1):
            cycle.append(normalize(kicked_top_step(cycle[-1], kappa, alpha, order)))
        used.add(i)
        # Mark every iterate that coincides with another unique fixed point so
        # we don't restart the same cycle from a different head.
        for c in cycle[1:]:
            angs = np.array([np.arccos(np.clip(c @ u, -1.0, 1.0)) for u in uniq])
            j = int(angs.argmin())
            if angs[j] < 10 * dedup_angle_tol:
                used.add(j)
        cycles.append([to_sph(c) for c in cycle])

    return cycles
