# Animate the Hammer-projected Poincaré section across a sweep of κ (or α).
# Each frame re-runs the classical kicked-top map for one parameter value.

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from IPython.display import Image as IPyImage, display

import examples.chaosqkt.ClassicalSimFunc as cf

# --- sweep configuration -------------------------------------------------
sweep_param = 'kappa'                       # 'kappa' or 'alpha'
kappa_sweep = np.linspace(0.1, 4.0, 50)     # κ values if sweep_param='kappa'
alpha_sweep = np.linspace(0.1, np.pi, 30)   # α values if sweep_param='alpha'
fixed_kappa = 3.0                           # used when sweep_param='alpha'
fixed_alpha = np.pi / 2                     # used when sweep_param='kappa'
order = 2              # Nonlinearity order: p=2 (quadratic)

# Lighter than the main run so the animation builds in seconds, not minutes
n_seeds_phi= 12
n_seeds_theta=12
n_iter_anim    = 600
n_discard_anim = 50
cmap           = 'nipy_spectral'

filename = 'kicked_top_sweep.gif'
fps      = 5
dpi      = 100

sweep_vals = kappa_sweep if sweep_param == 'kappa' else alpha_sweep
# -------------------------------------------------------------------------

# Reusable seed grid (cheap, depends only on n_seeds_phi/theta)
phis_s   = np.linspace(-np.pi, np.pi, n_seeds_phi, endpoint=False)
thetas_s = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_seeds_theta)

seeds_cart_a, seeds_th0_a, seeds_ph0_a = [], [], []
for th in thetas_s:
    for ph in phis_s:
        seeds_cart_a.append(cf.sph_to_cart(th, ph))
        seeds_th0_a.append(th)
        seeds_ph0_a.append(ph)
seeds_cart_a = np.stack(seeds_cart_a, axis=0)
seeds_th0_a  = np.asarray(seeds_th0_a)
seeds_ph0_a  = np.asarray(seeds_ph0_a)
n_seeds_a    = seeds_cart_a.shape[0]
c_seed_a     = (seeds_th0_a / np.pi) + ((seeds_ph0_a + np.pi) / (2 * np.pi))


def run_map(kappa_val, alpha_val):
    """Iterate the classical kicked-top map and return (phi, theta) point clouds."""
    v = seeds_cart_a.copy()
    n_keep_a = n_iter_anim - n_discard_anim
    phi_a   = np.empty((n_keep_a, n_seeds_a))
    theta_a = np.empty((n_keep_a, n_seeds_a))
    for i in range(n_iter_anim):
        v = np.array([cf.kicked_top_step(vi, kappa_val, alpha_val, order) for vi in v])
        if i >= n_discard_anim:
            ang = np.array([cf.spherical_angles(vi) for vi in v])
            phi_a[i - n_discard_anim]   = ang[:, 0]
            theta_a[i - n_discard_anim] = ang[:, 1]
    return phi_a, theta_a


frames = []
for k, val in enumerate(sweep_vals):
    if sweep_param == 'kappa':
        kappa_val, alpha_val = val, fixed_alpha
        label = fr'$\kappa$ = {val:.2f},  $\alpha$ = {alpha_val:.2f}'
    else:
        kappa_val, alpha_val = fixed_kappa, val
        label = fr'$\kappa$ = {kappa_val:.2f},  $\alpha$ = {val:.2f}'

    phi_a, theta_a = run_map(kappa_val, alpha_val)

    lon = phi_a.ravel()
    lat = np.pi / 2 - theta_a.ravel()
    col = np.broadcast_to(c_seed_a, phi_a.shape).ravel()

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111, projection='hammer')
    ax.scatter(lon, lat, c=col, cmap=cmap, s=0.4, alpha=0.5,
               vmin=c_seed_a.min(), vmax=c_seed_a.max())
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_title(f'Poincaré section (Hammer)  —  {label}',
                 fontsize=12, pad=15)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)
    print(f'  frame {k+1:>3}/{len(sweep_vals)}  ({label})')

frames[0].save(
    filename,
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=int(1000 / fps),
)
print(f'\n✓ Saved {filename}  ({len(frames)} frames, {fps} fps)')

display(IPyImage(filename=filename))
