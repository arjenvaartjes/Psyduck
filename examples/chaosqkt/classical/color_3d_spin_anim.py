"""3-D spinning-sphere Poincaré section animation with parameter sweep.

Combines color_plot_3d_spin.py (rotating sphere display) and
color_plot_anim.py (κ / α sweep) with two fully independent timescales:

  spin_period   — frames for one complete sphere rotation
  chaos_period  — frames for one complete parameter sweep

By default the sphere rotates faster than the chaos evolves, so each
Poincaré section is visible from several angles before the parameter steps.

COLOR_MODE = 'seed'    → continuous nipy_spectral gradient by seed position
COLOR_MODE = 'period'  → discrete palette from orbital-period classification
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers '3d' projection
import io
from PIL import Image
from IPython.display import Image as IPyImage, display
from tqdm import tqdm

import examples.chaosqkt.ClassicalSimFunc as cf

# ============================================================================
# CONFIGURATION
# ============================================================================

COLOR_MODE = 'period'     # 'seed'  or  'period'

# --- parameter sweep --------------------------------------------------------
sweep_param = 'kappa'                       # 'kappa'  or  'alpha'
kappa_sweep = np.linspace(0.1, 4.0, 500)[::-1]    # κ values when sweep_param='kappa'
alpha_sweep = np.linspace(0.1, np.pi, 20)  # α values when sweep_param='alpha'
fixed_kappa = 3.0                           # used when sweep_param='alpha'
fixed_alpha = np.pi / 2                     # used when sweep_param='kappa'
order = 2

# --- seed grid / trajectory -------------------------------------------------
n_seeds_phi   = 20
n_seeds_theta = 20
n_iter        = 500
n_discard     = 100

# --- period-classification (only used when COLOR_MODE='period') -------------
max_p          = 15
recurrence_tol = 0.05
aspect_thresh  = 0.15

# --- animation timing — two independent clocks ------------------------------
#   spin_period   frames for one complete sphere rotation
#   chaos_period  frames for one complete parameter sweep
#
# The sphere rotates  chaos_period / spin_period  full laps while the chaos
# makes one sweep.  Default: 5 laps per sweep → sphere 5× faster than chaos.
#
# chaos_period must be an integer multiple of spin_period for the GIF to
# loop seamlessly (sphere returns to its start orientation after every sweep).
spin_period  = 500    # frames per rotation  (10 fps → 36 s per lap)
chaos_period = 500   # frames per sweep     (10 fps → 36 s per sweep)
# → chaos_period / spin_period = 5 full sphere rotations per parameter sweep

fps      = 10
dpi      = 120
filename = 'kicked_top_3d_sweep.gif'

# ============================================================================
# 1. Sweep values
# ============================================================================

sweep_vals = kappa_sweep if sweep_param == 'kappa' else alpha_sweep
n_sweep    = len(sweep_vals)
n_frames   = chaos_period   # total animation length = one full sweep

# ============================================================================
# 2. Seed grid
# ============================================================================

phis   = np.linspace(-np.pi, np.pi, n_seeds_phi, endpoint=False)
thetas = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_seeds_theta)

seeds_cart, seeds_theta0, seeds_phi0 = [], [], []
for th in thetas:
    for ph in phis:
        seeds_cart.append(cf.sph_to_cart(th, ph))
        seeds_theta0.append(th)
        seeds_phi0.append(ph)

seeds_cart   = np.stack(seeds_cart, axis=0)   # (n_seeds, 3)
seeds_theta0 = np.asarray(seeds_theta0)
seeds_phi0   = np.asarray(seeds_phi0)
n_seeds      = seeds_cart.shape[0]
c_seed       = (seeds_theta0 / np.pi) + ((seeds_phi0 + np.pi) / (2 * np.pi))

# ============================================================================
# 3. Precompute trajectory and colours for each sweep value
# ============================================================================

def _run_map(kappa_val, alpha_val):
    """Iterate kicked-top map; return (phi_arr, theta_arr) of shape (n_keep, n_seeds)."""
    v      = seeds_cart.copy()
    n_keep = n_iter - n_discard
    phi_arr   = np.empty((n_keep, n_seeds))
    theta_arr = np.empty((n_keep, n_seeds))
    for i in range(n_iter):
        v = cf.kicked_top_step(v, kappa_val, alpha_val, order)
        if i >= n_discard:
            phi_arr[i - n_discard], theta_arr[i - n_discard] = cf.spherical_angles(v)
    return phi_arr, theta_arr


def _build_colors(phi_arr, theta_arr):
    """Return (plot_colors, cmap_use, norm_use, vmin_use, vmax_use)."""
    n_keep = phi_arr.shape[0]

    if COLOR_MODE == 'seed':
        return (
            np.broadcast_to(c_seed, phi_arr.shape).ravel().copy(),
            'nipy_spectral', None,
            float(c_seed.min()), float(c_seed.max()),
        )

    # --- 'period' mode --------------------------------------------------
    x_traj = np.sin(theta_arr) * np.cos(phi_arr)
    y_traj = np.sin(theta_arr) * np.sin(phi_arr)
    z_traj = np.cos(theta_arr)
    cart_traj = np.stack([x_traj, y_traj, z_traj], axis=-1)  # (n_keep, n_seeds, 3)
    traj = cart_traj[n_keep // 2:]

    period_arr = np.zeros(n_seeds, dtype=int)
    for p in range(1, max_p + 1):
        dots = np.clip(np.einsum('ijk,ijk->ij', traj[:-p], traj[p:]), -1.0, 1.0)
        period_arr[(period_arr == 0) & (np.mean(np.arccos(dots), axis=0) < recurrence_tol)] = p

    aspect = np.zeros(n_seeds)
    for s in range(n_seeds):
        eigs      = np.sort(np.linalg.eigvalsh(np.cov(traj[:, s, :].T)))[::-1]
        aspect[s] = eigs[1] / (eigs[0] + 1e-12)

    period_code = period_arr.astype(float)
    period_code[(period_arr == 0) & (aspect > aspect_thresh)] = -1.0

    unique_codes = sorted(np.unique(period_code))
    code_to_idx  = {c: i for i, c in enumerate(unique_codes)}
    n_codes      = len(unique_codes)

    color_list = []
    for c in unique_codes:
        if c < 0:    color_list.append('#444444')
        elif c == 0: color_list.append('#aaaaaa')
        else:        color_list.append(plt.cm.tab20.colors[int(c - 1) % 20])

    cmap_use = mcolors.ListedColormap(color_list)
    norm_use = mcolors.BoundaryNorm(np.arange(-0.5, n_codes), cmap_use.N)
    idx_seeds   = np.array([code_to_idx[c] for c in period_code])
    plot_colors = np.broadcast_to(idx_seeds, phi_arr.shape).ravel().astype(float)

    return plot_colors, cmap_use, norm_use, None, None


_r = 1.001   # fractionally outside unit sphere to stay in front of surface

precomputed = []
for val in tqdm(sweep_vals, desc='Precomputing trajectories'):
    kappa_val = val        if sweep_param == 'kappa' else fixed_kappa
    alpha_val = fixed_alpha if sweep_param == 'kappa' else val

    phi_arr, theta_arr = _run_map(kappa_val, alpha_val)

    phi_flat   = phi_arr.ravel()
    theta_flat = theta_arr.ravel()
    x_pts = _r * np.sin(theta_flat) * np.cos(phi_flat)
    y_pts = _r * np.sin(theta_flat) * np.sin(phi_flat)
    z_pts = _r * np.cos(theta_flat)

    plot_colors, cmap_use, norm_use, vmin_use, vmax_use = _build_colors(phi_arr, theta_arr)

    precomputed.append(dict(
        kappa=kappa_val, alpha=alpha_val,
        x_pts=x_pts, y_pts=y_pts, z_pts=z_pts,
        plot_colors=plot_colors,
        cmap_use=cmap_use, norm_use=norm_use,
        vmin_use=vmin_use, vmax_use=vmax_use,
    ))

# ============================================================================
# 4. Sphere reference surface
# ============================================================================

_u = np.linspace(0, 2 * np.pi, 60)
_v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(_u), np.sin(_v))
ys = np.outer(np.sin(_u), np.sin(_v))
zs = np.outer(np.ones_like(_u), np.cos(_v))

# ============================================================================
# 5. Build animation frames
# ============================================================================

# Sphere camera — spin_period controls how many frames per full rotation.
# The overall camera path runs chaos_period / spin_period complete laps so
# the sphere always returns to its start orientation when the GIF loops.
_t        = np.linspace(0, 2 * np.pi * (chaos_period / spin_period),
                        n_frames, endpoint=False)
azim_vals = np.degrees(_t)       # continuous horizontal laps
elev_vals = 55.0 * np.sin(_t)    # ±55° elevation, one cycle per spin_period
roll_vals = 20.0 * np.sin(2 * _t) # ±20° roll, two cycles per spin_period

# Chaos — sweep_val index for each frame, independent of the spin clock.
chaos_indices = (np.linspace(0, n_sweep, n_frames, endpoint=False)).astype(int)

frames = []
for k in tqdm(range(n_frames), desc='Rendering frames'):
    azim = azim_vals[k]
    elev = elev_vals[k]
    roll = roll_vals[k]
    data = precomputed[chaos_indices[k]]

    x_pts       = data['x_pts']
    y_pts       = data['y_pts']
    z_pts       = data['z_pts']
    plot_colors = data['plot_colors']
    cmap_use    = data['cmap_use']
    norm_use    = data['norm_use']
    vmin_use    = data['vmin_use']
    vmax_use    = data['vmax_use']

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection='3d')

    ax.plot_surface(xs, ys, zs, color='white', alpha=1.0,
                    linewidth=0, edgecolors='none',
                    antialiased=False, shade=False, zorder=0)

    # Hemisphere visibility filter — suppress back-side bleed-through.
    _er  = np.radians(float(elev))
    _ar  = np.radians(float(azim))
    _vd  = np.array([np.cos(_er) * np.cos(_ar),
                     np.cos(_er) * np.sin(_ar),
                     np.sin(_er)])
    _vis = (x_pts * _vd[0] + y_pts * _vd[1] + z_pts * _vd[2]) > 0

    ax.scatter(x_pts[_vis], y_pts[_vis], z_pts[_vis],
               c=plot_colors[_vis], cmap=cmap_use, norm=norm_use,
               vmin=vmin_use, vmax=vmax_use,
               s=0.4, alpha=0.6, depthshade=False, zorder=5)

    ax.view_init(elev=float(elev), azim=float(azim), roll=float(roll))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    if sweep_param == 'kappa':
        label = fr'$\kappa$={data["kappa"]:.2f},  $\alpha$={data["alpha"]:.2f}'
    else:
        label = fr'$\kappa$={data["kappa"]:.2f},  $\alpha$={data["alpha"]:.2f}'

    ax.set_title(
        f'Poincaré section — {COLOR_MODE} colour\n{label},  order={order}',
        fontsize=11, pad=-10,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)

# ============================================================================
# 6. Save GIF
# ============================================================================

frames[0].save(
    filename,
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=int(1000 / fps),
)
print(f'\n✓ Saved  {filename}  ({n_frames} frames, {fps} fps, '
      f'{chaos_period / spin_period:.0f} sphere rotations per sweep)')
display(IPyImage(filename=filename))