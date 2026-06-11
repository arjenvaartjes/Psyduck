"""Helper functions for classical and quantum chaos simulations on top of psyduck.

This module is a thin layer above ``psyduck``:

* Classical / phase-space helpers (kicked-top map, sphere geometry, periodic
  orbit finding) are kept here because they operate on plain ``numpy``
  3-vectors and have no analogue inside psyduck.
* Anything that builds a quantum state, propagates it, or analyses a Floquet
  spectrum delegates to ``psyduck`` rather than re-implementing the wheel.
"""

import numpy as np
from typing import Tuple, List

from psyduck.evolve import kicked_top_step

# Imports below are only needed by the commented-out scar-analysis functions
# at the bottom of this file; uncomment together if you bring those back.
# import qutip as qt
# from psyduck import Spin

# ============================================================================
# Periodic Orbit Finding
# ============================================================================

def find_period_p_orbits(kappa: float, alpha: float, order: int, p: int,
                         grid_th: int = 41, grid_ph: int = 81,
                         seed_tol: float = 5e-2, refine_tol: float = 1e-10,
                         dedup_angle_tol: float = 1e-3,
                         ) -> List[List[Tuple[float, float]]]:
    """Find every period-p orbit of the classical kicked top.

    A period-p orbit is a set of p unit vectors {v_1, ..., v_p} that maps
    cyclically into itself under the kicked-top map F:  F(v_i) = v_{(i+1) mod p}.
    Equivalently, each v_i is a fixed point of F^p, the p-fold composition.

    Algorithm (three steps; see ``PERIODIC_ORBITS.md`` next to this file for
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

    def normalize(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def F_p(v: np.ndarray) -> np.ndarray:
        """Apply the kicked-top map p times.  Broadcasts over leading axes."""
        u = normalize(v)
        for _ in range(p):
            u = kicked_top_step(u, kappa, alpha, order)
        return normalize(u)

    # ----- 1. Coarse seed grid (vectorised) --------------------------------
    th, ph = np.meshgrid(np.linspace(0, np.pi, grid_th),
                         np.linspace(-np.pi, np.pi, grid_ph), indexing='ij')
    V = np.stack([np.sin(th) * np.cos(ph),
                  np.sin(th) * np.sin(ph),
                  np.cos(th)], axis=-1).reshape(-1, 3)
    seeds = V[np.linalg.norm(F_p(V) - V, axis=-1) < seed_tol]

    # ----- 2. Polish each seed with scipy.optimize.fsolve -------------------
    refined: List[np.ndarray] = []
    for s in seeds:
        v, _, ier, _ = fsolve(lambda x: F_p(x) - normalize(x),
                              s, xtol=refine_tol, full_output=True)
        v = normalize(v)
        if ier == 1 and np.linalg.norm(F_p(v) - v) < 1e2 * refine_tol:
            refined.append(v)

    # ----- 3. Deduplicate, then assemble cycles -----------------------------
    uniq: List[np.ndarray] = []
    for v in refined:
        if all(np.arccos(np.clip(v @ u, -1.0, 1.0)) > dedup_angle_tol for u in uniq):
            uniq.append(v)

    def to_sph(c: np.ndarray) -> Tuple[float, float]:
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
