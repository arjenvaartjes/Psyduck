"""3-D spinning-sphere Poincaré section animation.

Computes the kicked-top trajectory (same as color_plot.py), then renders
each trajectory point on the surface of a 3-D unit sphere and exports a
rotating GIF.

COLOR_MODE = 'period'  → discrete palette from orbital-period classification
COLOR_MODE = 'seed'    → continuous nipy_spectral diagonal-gradient by seed

Frame-building approach follows color_plot_anim.py (io.BytesIO → PIL → GIF).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers '3d' projection
import io
from PIL import Image
from IPython.display import Image as IPyImage, display
from tqdm import trange, tqdm

import examples.chaosqkt.ClassicalSimFunc as cf

# ============================================================================
# CONFIGURATION
# ============================================================================

COLOR_MODE = 'period'   # 'period'  or  'seed'

n_seeds_phi   = 20
n_seeds_theta = 20
n_iter        = 500
n_discard     = 100

kappa = 2.2
alpha = np.pi / 2
order = 2

# Period-classification parameters (only used in 'period' mode)
max_p          = 20
recurrence_tol = 0.05
aspect_thresh  = 0.15

# Animation
# The camera path uses three simultaneous rotations parameterised by t ∈ [0, 2π]:
#   azim  sweeps 0 → 360° (one full horizontal lap, seamless loop)
#   elev  = A_e · sin(t)     — one elevation cycle, ±55° (shows both poles)
#   roll  = A_r · sin(2·t)   — two roll cycles, ±20°
# Integer frequency multiples guarantee all three angles return to their
# starting values at t = 2π, so the GIF loops without a visible seam.
n_frames = 120    # 120 frames at 10 fps = 12 s per loop
fps      = 10
dpi      = 120
filename = 'poincare_3d_spin.gif'

# ============================================================================
# 1. Seed grid
# ============================================================================

phis   = np.linspace(-np.pi, np.pi, n_seeds_phi, endpoint=False)
thetas = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_seeds_theta)

seeds_cart, seeds_theta0, seeds_phi0 = [], [], []
for th in thetas:
    for ph in phis:
        seeds_cart.append(cf.sph_to_cart(th, ph))
        seeds_theta0.append(th)
        seeds_phi0.append(ph)

v            = np.stack(seeds_cart, axis=0)   # (n_seeds, 3)
seeds_theta0 = np.asarray(seeds_theta0)
seeds_phi0   = np.asarray(seeds_phi0)
n_seeds      = v.shape[0]

# ============================================================================
# 2. Run trajectory (vectorised over all seeds)
# ============================================================================

n_keep    = n_iter - n_discard
phi_arr   = np.empty((n_keep, n_seeds))
theta_arr = np.empty((n_keep, n_seeds))

for i in trange(n_iter, desc='Trajectory'):
    v = cf.kicked_top_step(v, kappa, alpha, order)
    if i >= n_discard:
        phi_arr[i - n_discard], theta_arr[i - n_discard] = cf.spherical_angles(v)

# ============================================================================
# 3. Build colour arrays
# ============================================================================

c_seed    = (seeds_theta0 / np.pi) + ((seeds_phi0 + np.pi) / (2 * np.pi))
color_arr = np.broadcast_to(c_seed, phi_arr.shape)   # (n_keep, n_seeds), read-only

if COLOR_MODE == 'period':
    # Reconstruct Cartesian trajectory from stored angles
    x_traj = np.sin(theta_arr) * np.cos(phi_arr)
    y_traj = np.sin(theta_arr) * np.sin(phi_arr)
    z_traj = np.cos(theta_arr)
    cart_traj = np.stack([x_traj, y_traj, z_traj], axis=-1)   # (n_keep, n_seeds, 3)

    traj = cart_traj[n_keep // 2:]   # second half — drops residual transient
    n_t  = traj.shape[0]

    # Recurrence test: period-p orbit → mean geodesic distance ≈ 0
    period_arr = np.zeros(n_seeds, dtype=int)
    for p in range(1, max_p + 1):
        v1   = traj[:-p]
        v2   = traj[p:]
        dots = np.clip(np.einsum('ijk,ijk->ij', v1, v2), -1.0, 1.0)
        mean_dist = np.mean(np.arccos(dots), axis=0)
        period_arr[(period_arr == 0) & (mean_dist < recurrence_tol)] = p

    # PCA eigenvalue ratio: λ₂/λ₁ ≈ 0 → quasi-periodic, ≈ 1 → chaotic
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

    cmap_use    = mcolors.ListedColormap(color_list)
    norm_use    = mcolors.BoundaryNorm(np.arange(-0.5, n_codes), cmap_use.N)
    tick_labels = ['Chaos' if c < 0 else ('QP' if c == 0 else f'p={int(c)}')
                   for c in unique_codes]

    idx_seeds   = np.array([code_to_idx[c] for c in period_code])
    idx_traj    = np.broadcast_to(idx_seeds, phi_arr.shape)
    plot_colors = idx_traj.ravel().astype(float)   # copy so it's writable
    vmin_use    = None
    vmax_use    = None

else:  # 'seed'
    cmap_use    = 'nipy_spectral'
    norm_use    = None
    tick_labels = None
    plot_colors = color_arr.ravel().copy()
    vmin_use    = float(c_seed.min())
    vmax_use    = float(c_seed.max())

# ============================================================================
# 4. Convert trajectory to 3-D Cartesian for scatter
# ============================================================================

phi_flat   = phi_arr.ravel()
theta_flat = theta_arr.ravel()

# r = 1.001 places scatter points fractionally outside the sphere surface so
# they always have a smaller camera-Z than the corresponding sphere patch,
# keeping them reliably in front regardless of the depth-sort order.
_r    = 1.001
x_pts = _r * np.sin(theta_flat) * np.cos(phi_flat)
y_pts = _r * np.sin(theta_flat) * np.sin(phi_flat)
z_pts = _r * np.cos(theta_flat)

# ============================================================================
# 5. Sphere reference surface (drawn behind the scatter)
# ============================================================================

_u = np.linspace(0, 2 * np.pi, 60)
_v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(_u), np.sin(_v))
ys = np.outer(np.sin(_u), np.sin(_v))
zs = np.outer(np.ones_like(_u), np.cos(_v))

# ============================================================================
# 6. Build animation frames  (follows color_plot_anim.py pattern)
# ============================================================================

frames  = []
_t      = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
azim_vals = np.degrees(_t)           # 0 → 360°, one full lap
elev_vals = 55.0 * np.sin(_t)        # ±55°, one cycle → shows both poles
roll_vals = 20.0 * np.sin(2 * _t)    # ±20°, two cycles

for k, (azim, elev, roll) in enumerate(
        tqdm(zip(azim_vals, elev_vals, roll_vals), total=n_frames, desc='Rendering frames')):
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # Solid white sphere base.
    # edgecolors='none' suppresses the mesh-line flash that appears when polygon
    # edges become view-aligned at certain rotation angles.
    ax.plot_surface(xs, ys, zs, color='white', alpha=1.0,
                    linewidth=0, edgecolors='none',
                    antialiased=False, shade=False, zorder=0)

    # Per-frame hemisphere filter: only draw scatter points on the front-facing
    # hemisphere.  The view direction in world space is derived from elev/azim;
    # points with a positive dot-product face the camera.  This eliminates
    # back-hemisphere points that would otherwise bleed through the sphere
    # due to matplotlib's imperfect 3-D depth sorting.
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
    ax.set_title(
        f'Poincaré section — {COLOR_MODE} colour\n'
        f'κ={kappa},  α={alpha:.2f},  order={order}',
        fontsize=11, pad=-10,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)

frames[0].save(
    filename,
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=int(1000 / fps),
)
print(f'\n✓ Saved  {filename}  ({n_frames} frames, {fps} fps)')
display(IPyImage(filename=filename))