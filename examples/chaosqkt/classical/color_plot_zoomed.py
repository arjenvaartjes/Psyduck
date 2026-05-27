import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import trange

import examples.chaosqkt.ClassicalSimFunc as cf

# ============================================================================
# CONFIGURATION  — edit these values to control every aspect of the run
# ============================================================================

PROJECTION = 'cartesian'   # 'cartesian'  or  'hammer'

# Zoom — set ZOOM=False to reproduce the full-sphere behaviour of color_plot.py.
# When ZOOM=True, seeds are placed inside the zoom rectangle and n_iter is
# auto-scaled so chaotic trajectories re-enter the region enough times.
# See the rationale block below for details.
ZOOM       = True

# Zoom in on the center dot
# ZOOM_PHI   = (-0.5, 0.5)   # φ range [rad],   full sphere ≈ (−π, π)
# ZOOM_THETA = (1.2,  1.9)   # θ range [rad],   full sphere ≈ (0, π)

# Zoom in on the fine structure
ZOOM_PHI   = (-0.7, 0.4)   # φ range [rad],   full sphere ≈ (−π, π)
ZOOM_THETA = (0.7,  1.2)   # θ range [rad],   full sphere ≈ (0, π)

# Base simulation parameters
n_seeds_phi   = 100
n_seeds_theta = 100
BASE_N_ITER   = 500   # iterations for full sphere; auto-scaled up when ZOOM=True
n_discard     = 100

kappa = 2.2
alpha = np.pi / 2
order = 2

# ============================================================================
# Zoom-sampling rationale
# ============================================================================
# The kicked-top map is area-preserving on S² (it preserves the uniform
# Liouville measure).  Birkhoff's ergodic theorem then guarantees that every
# chaotic trajectory visits any sub-region with time-average frequency equal
# to that region's solid-angle fraction
#
#     f_zoom = Δφ · (cos θ_lo − cos θ_hi) / [Δφ_full · (cos θ_lo_full − cos θ_hi_full)]
#
# Simply cropping the final scatter plot adds no new structure: it is
# equivalent to discarding ~(1 − f_zoom) of the data.  This script uses a
# genuinely denser sampling strategy with three components:
#
#   1. SEED INSIDE THE ZOOM RECTANGLE
#      Small KAM islands (period-n orbits) can only be traced faithfully if a
#      seed starts near them.  A global grid may place zero seeds inside a
#      tiny island's basin.  Concentrating all n_seeds inside the zoom window
#      ensures every sub-island is seeded, resolving fractal island chains.
#
#   2. AUTO-SCALE n_iter ∝ 1 / f_zoom
#      A chaotic seed started inside the zoom window leaves almost immediately
#      and returns every ~1/f_zoom kicks on average (ergodic return time).
#      Scaling n_iter up by 1/f_zoom gives each chaotic trajectory the same
#      expected number of plotted points as in the unzoomed run, filling the
#      chaotic sea uniformly within the window.
#
#   3. FILTER TRAJECTORY POINTS TO THE ZOOM WINDOW
#      After collection, trajectory points outside the rectangle are discarded
#      for all trajectory-level plots.  Seed-origin and Lyapunov plots already
#      live in seed space (which is the zoom window), so no filtering is needed.
#      In cartesian mode the axes are also explicitly clipped.  In hammer mode,
#      the Hammer projection cannot clip axes, so filtering alone achieves zoom.
# ============================================================================

# ---- Derived parameters ----

_FULL_PHI_RANGE   = (-np.pi, np.pi)
_FULL_THETA_RANGE = (0.15 * np.pi, 0.85 * np.pi)

if ZOOM:
    seed_phi_lo,   seed_phi_hi   = ZOOM_PHI
    seed_theta_lo, seed_theta_hi = ZOOM_THETA
    # Solid-angle of a φ-θ rectangle: ∫∫ sin θ dθ dφ = Δφ · (cos θ_lo − cos θ_hi)
    _sa_full = ((np.cos(_FULL_THETA_RANGE[0]) - np.cos(_FULL_THETA_RANGE[1]))
                * (_FULL_PHI_RANGE[1] - _FULL_PHI_RANGE[0]))
    _sa_zoom = ((np.cos(seed_theta_lo) - np.cos(seed_theta_hi))
                * (seed_phi_hi - seed_phi_lo))
    zoom_fraction = _sa_zoom / _sa_full
    n_iter = max(BASE_N_ITER, int(np.ceil(BASE_N_ITER / zoom_fraction)))
    print(f"Zoom solid-angle fraction : {zoom_fraction:.4f}")
    print(f"n_iter auto-scaled        : {BASE_N_ITER}  →  {n_iter}")
else:
    seed_phi_lo,   seed_phi_hi   = _FULL_PHI_RANGE
    seed_theta_lo, seed_theta_hi = _FULL_THETA_RANGE
    zoom_fraction = 1.0
    n_iter = BASE_N_ITER

# ---- Build seed grid ----

phis   = np.linspace(seed_phi_lo,   seed_phi_hi,   n_seeds_phi,   endpoint=False)
thetas = np.linspace(seed_theta_lo, seed_theta_hi, n_seeds_theta)

seeds_cart, seeds_theta0, seeds_phi0 = [], [], []
for th in thetas:
    for ph in phis:
        seeds_cart.append(cf.sph_to_cart(th, ph))
        seeds_theta0.append(th)
        seeds_phi0.append(ph)
v = np.stack(seeds_cart, axis=0)            # (n_seeds, 3)
seeds_theta0 = np.asarray(seeds_theta0)
seeds_phi0   = np.asarray(seeds_phi0)
n_seeds = v.shape[0]

# Per-seed diagonal colour gradient (same formula as color_plot.py)
c_seed = (seeds_theta0 / np.pi) + ((seeds_phi0 + np.pi) / (2 * np.pi))

# ---- Run trajectory — fully vectorised over all seeds ----
# kicked_top_step and spherical_angles both support batched (N, 3) input.

n_keep    = n_iter - n_discard
phi_arr   = np.empty((n_keep, n_seeds))
theta_arr = np.empty((n_keep, n_seeds))

for i in trange(n_iter, desc='Trajectory'):
    v = cf.kicked_top_step(v, kappa, alpha, order)
    if i >= n_discard:
        phi_arr[i - n_discard], theta_arr[i - n_discard] = cf.spherical_angles(v)

# ---- Zoom filter for trajectory-point plots ----

if ZOOM:
    in_zoom = ((phi_arr   >= ZOOM_PHI[0])   & (phi_arr   <= ZOOM_PHI[1]) &
               (theta_arr >= ZOOM_THETA[0]) & (theta_arr <= ZOOM_THETA[1]))
else:
    in_zoom = np.ones_like(phi_arr, dtype=bool)

# ============================================================================
# Plotting helpers — abstract over PROJECTION choice
# ============================================================================

_zoom_tag    = '  [zoom]' if ZOOM else ''
_phi_range   = ZOOM_PHI   if ZOOM else (-np.pi, np.pi)
_theta_range = ZOOM_THETA if ZOOM else (0, np.pi)


def _new_fig_ax(title='', figsize=(10, 5)):
    fig = plt.figure(figsize=figsize)
    if PROJECTION == 'hammer':
        ax = fig.add_subplot(111, projection='hammer')
        ax.set_xticks([])
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$\phi$ (azimuthal)', fontsize=12)
        ax.set_ylabel(r'$\theta$ (polar)',    fontsize=12)
        ax.set_xlim(*_phi_range)
        ax.set_ylim(*_theta_range)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(True, alpha=0.3, linestyle=':')
    return fig, ax


def _scatter(ax, phi, theta, **kwargs):
    """Scatter on the correct projection (Hammer needs lat = π/2 − θ)."""
    if PROJECTION == 'hammer':
        return ax.scatter(phi, np.pi / 2 - theta, **kwargs)
    return ax.scatter(phi, theta, **kwargs)


def _add_cbar(fig, sc, ax, label='', discrete_ticks=None, discrete_labels=None):
    if PROJECTION == 'hammer':
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                            fraction=0.05, pad=0.08, shrink=0.7)
    else:
        cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(label, fontsize=10)
    if discrete_ticks is not None:
        cbar.set_ticks(discrete_ticks)
        cbar.set_ticklabels(discrete_labels)
    return cbar


# ============================================================================
#%% Plot 1 — Poincaré section coloured by seed starting position
# ============================================================================

color_arr = np.broadcast_to(c_seed, phi_arr.shape)   # (n_keep, n_seeds), read-only

fig, ax = _new_fig_ax(
    title=f'Poincaré Section — seed colour  (κ={kappa}, order={order}){_zoom_tag}'
)
sc = _scatter(ax,
              phi_arr[in_zoom], theta_arr[in_zoom],
              c=color_arr[in_zoom], cmap='nipy_spectral', s=0.5, alpha=0.5)
_add_cbar(fig, sc, ax, label='Seed diagonal coordinate')
plt.tight_layout()
plt.show()


# ============================================================================
#%% Orbital period classification
# ============================================================================

max_p          = 20
recurrence_tol = 0.05
aspect_thresh  = 0.15

# Reconstruct Cartesian trajectory from stored angles
x_traj    = np.sin(theta_arr) * np.cos(phi_arr)
y_traj    = np.sin(theta_arr) * np.sin(phi_arr)
z_traj    = np.cos(theta_arr)
cart_traj = np.stack([x_traj, y_traj, z_traj], axis=-1)   # (n_keep, n_seeds, 3)

traj = cart_traj[n_keep // 2:]   # second half — drops any residual transient
n_t  = traj.shape[0]

# Recurrence test: period-p orbit → v[i] ≈ v[i+p], mean geodesic distance ≈ 0
period_arr = np.zeros(n_seeds, dtype=int)
for p in range(1, max_p + 1):
    v1   = traj[:-p]
    v2   = traj[p:]
    dots = np.clip(np.einsum('ijk,ijk->ij', v1, v2), -1.0, 1.0)
    mean_dist = np.mean(np.arccos(dots), axis=0)
    period_arr[(period_arr == 0) & (mean_dist < recurrence_tol)] = p

# PCA eigenvalue ratio: λ₂/λ₁ ≈ 0 → quasi-periodic (1-D curve)
#                                ≈ 1 → chaotic (2-D filled region)
aspect = np.zeros(n_seeds)
for s in range(n_seeds):
    cov       = np.cov(traj[:, s, :].T)
    eigs      = np.sort(np.linalg.eigvalsh(cov))[::-1]
    aspect[s] = eigs[1] / (eigs[0] + 1e-12)

period_code = period_arr.astype(float)
period_code[(period_arr == 0) & (aspect > aspect_thresh)] = -1.0

print(f"\nClassification  (aspect_thresh={aspect_thresh},  recurrence_tol={recurrence_tol} rad)")
for c in sorted(np.unique(period_code)):
    mask  = period_code == c
    label = 'Chaotic' if c < 0 else ('Quasi-periodic' if c == 0 else f'Period {int(c)}')
    print(f"  {label:15s}: {mask.sum():4d} seeds  "
          f"aspect = {aspect[mask].mean():.3f} ± {aspect[mask].std():.3f}")

# Discrete colormap — remap codes to consecutive integers for uniform band widths
unique_codes = sorted(np.unique(period_code))
n_codes      = len(unique_codes)
code_to_idx  = {c: i for i, c in enumerate(unique_codes)}

color_list = []
for c in unique_codes:
    if c < 0:
        color_list.append('#444444')
    elif c == 0:
        color_list.append('#aaaaaa')
    else:
        color_list.append(plt.cm.tab20.colors[int(c - 1) % 20])

cmap_disc   = mcolors.ListedColormap(color_list)
norm_disc   = mcolors.BoundaryNorm(np.arange(-0.5, n_codes), cmap_disc.N)
tick_labels = ['Chaos' if c < 0 else ('QP' if c == 0 else f'p={int(c)}')
               for c in unique_codes]

idx_seeds = np.array([code_to_idx[c] for c in period_code])
idx_traj  = np.broadcast_to(idx_seeds, phi_arr.shape)   # (n_keep, n_seeds), read-only


# ============================================================================
#%% Plot 2 — Poincaré section coloured by orbital period
# ============================================================================

fig, ax = _new_fig_ax(
    title=f'Poincaré Section — orbital period  (κ={kappa}, order={order}){_zoom_tag}'
)
sc = _scatter(ax,
              phi_arr[in_zoom], theta_arr[in_zoom],
              c=idx_traj[in_zoom], cmap=cmap_disc, norm=norm_disc, s=0.5, alpha=0.6)
_add_cbar(fig, sc, ax, label='Orbit type',
          discrete_ticks=np.arange(n_codes), discrete_labels=tick_labels)
plt.tight_layout()
plt.show()


# ============================================================================
#%% Plot 3 — Seed-origin map coloured by orbital period
# ============================================================================

fig, ax = _new_fig_ax(
    title=f'Initial Conditions — orbital period  (κ={kappa}, order={order}){_zoom_tag}'
)
sc = _scatter(ax, seeds_phi0, seeds_theta0,
              c=idx_seeds, cmap=cmap_disc, norm=norm_disc,
              s=20, edgecolors='k', linewidths=0.4, zorder=3)
_add_cbar(fig, sc, ax, label='Orbit type',
          discrete_ticks=np.arange(n_codes), discrete_labels=tick_labels)
plt.tight_layout()
plt.show()


# ============================================================================
#%% Maximal Lyapunov exponent per seed
# ============================================================================

v_ly = np.array([cf.sph_to_cart(th, ph)
                 for th, ph in zip(seeds_theta0, seeds_phi0)])

rng = np.random.default_rng(0)
dv  = rng.standard_normal(v_ly.shape)
dv -= np.sum(dv * v_ly, axis=1, keepdims=True) * v_ly   # project onto tangent plane
dv /= np.linalg.norm(dv, axis=1, keepdims=True)

lyap_sum = np.zeros(n_seeds)
eps_ly   = 1e-7

for i in trange(n_iter, desc='Lyapunov'):
    v_plus  = cf.kicked_top_step(v_ly + eps_ly * dv, kappa, alpha, order)
    v_minus = cf.kicked_top_step(v_ly - eps_ly * dv, kappa, alpha, order)
    v_ly    = cf.kicked_top_step(v_ly,               kappa, alpha, order)

    dv_new  = (v_plus - v_minus) / (2 * eps_ly)
    dv_new -= np.sum(dv_new * v_ly, axis=1, keepdims=True) * v_ly  # re-project
    norms   = np.linalg.norm(dv_new, axis=1)
    if i >= n_discard:
        lyap_sum += np.log(norms + 1e-300)
    dv = dv_new / np.maximum(norms[:, np.newaxis], 1e-300)

lyap_exp = lyap_sum / (n_iter - n_discard)
print(f"\nLyapunov  min={lyap_exp.min():.4f}  max={lyap_exp.max():.4f}  "
      f"median={np.median(lyap_exp):.4f}")


# ============================================================================
#%% Plot 4 — Seed-origin map coloured by Lyapunov exponent
# ============================================================================

fig, ax = _new_fig_ax(
    title=f'Maximal Lyapunov Exponent  (κ={kappa}, order={order}){_zoom_tag}'
)
sc = _scatter(ax, seeds_phi0, seeds_theta0,
              c=lyap_exp, cmap='inferno', s=20,
              vmin=0, vmax=np.percentile(lyap_exp, 99))
_add_cbar(fig, sc, ax, label=r'$\lambda_{\max}$')
plt.tight_layout()
plt.show()


# ============================================================================
#%% Plot 5 — 2D histogram of trajectory points in zoom window
# ============================================================================

n_bins = 100

fig, ax = plt.subplots(figsize=(8, 6))
_, _, _, img = ax.hist2d(
    phi_arr[in_zoom], theta_arr[in_zoom],
    bins=n_bins,
    range=[list(_phi_range), list(_theta_range)],
    cmap='inferno',
)
ax.set_xlabel(r'$\phi$ (azimuthal)', fontsize=12)
ax.set_ylabel(r'$\theta$ (polar)',   fontsize=12)
ax.set_title(f'Poincaré Section — 2D Histogram  (κ={kappa}, order={order}){_zoom_tag}',
             fontsize=13)
ax.grid(True, alpha=0.3, linestyle=':')
plt.colorbar(img, ax=ax, label='Point density')
plt.tight_layout()
plt.show()