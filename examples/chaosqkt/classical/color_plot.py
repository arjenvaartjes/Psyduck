import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import examples.ChaosQKT.ClassicalSimFunc as cf
#%%
n_seeds_phi= 20  # 100
n_seeds_theta= 20  # 100
n_iter= 500
n_discard= 100

# Set the parameters for the system here
kappa = 2.2
alpha = np.pi / 2
order = 2


phis = np.linspace(-np.pi, np.pi, n_seeds_phi, endpoint=False)  # x
thetas = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_seeds_theta)  # y


seeds_cart, seeds_theta0, seeds_phi0 = [], [], []
for th in thetas:
    for ph in phis:
        seeds_cart.append(cf.sph_to_cart(th, ph))
        seeds_theta0.append(th)
        seeds_phi0.append(ph)
v = np.stack(seeds_cart, axis=0)
seeds_theta0 = np.asarray(seeds_theta0)
seeds_phi0   = np.asarray(seeds_phi0)
n_seeds = v.shape[0]

# Per-seed color value: diagonal gradient on the seed grid (in [0, 2])
c_seed = (seeds_theta0 / np.pi) + ((seeds_phi0 + np.pi) / (2 * np.pi))

# Pre-allocate: rows = iteration after discard, cols = seed
n_keep = n_iter - n_discard
phi_arr   = np.empty((n_keep, n_seeds))
theta_arr = np.empty((n_keep, n_seeds))

for i in trange(n_iter):
    v = np.array([cf.kicked_top_step(vi, kappa, alpha, order) for vi in v])
    if i >= n_discard:
        ang = np.array([cf.spherical_angles(vi) for vi in v])  # shape (n_seeds, 2): (phi, theta)
        phi_arr[i - n_discard]   = ang[:, 0]
        theta_arr[i - n_discard] = ang[:, 1]

# Broadcast the per-seed color over all iterations
color_arr = np.broadcast_to(c_seed, phi_arr.shape)

#%%

# Plot: each point colored by its seed's starting position

# cmap = 'viridis'
cmap = 'nipy_spectral'

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(phi_arr.ravel(), theta_arr.ravel(),
                c=color_arr.ravel(), cmap=cmap, s=0.5, alpha=0.5)

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(0, np.pi)
ax.set_xlabel(r'$\phi$ (azimuthal angle)', fontsize=12)
ax.set_ylabel(r'$\theta$ (polar angle)', fontsize=12)
ax.set_title(f'Poincaré Section colored by starting (θ, φ)  (κ={kappa})', fontsize=13)

ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax.set_yticks([0, np.pi/2, np.pi])
ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
ax.grid(True, alpha=0.3, linestyle=':')

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Seed diagonal coordinate', fontsize=10)

plt.tight_layout()
plt.show()

#%%


# cmap = 'viridis'
# cmap = 'nipy_spectral'
cmap = "jet"

# Convert spherical → Hammer coords
lon = phi_arr.ravel()                     # already in [-π, π]
lat = np.pi / 2 - theta_arr.ravel()       # θ ∈ [0, π]  → lat ∈ [π/2, -π/2]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='hammer')

sc = ax.scatter(lon, lat, c=color_arr.ravel(), cmap=cmap, s=0.5, alpha=0.5)

ax.set_title(f'Poincaré Section (Hammer)  (κ={kappa}, p={order})', fontsize=13, pad=20)

# Hammer axes label in radians by default; switch longitude labels to degrees-free π notation if you like
# ax.set_xticks(np.deg2rad([-150, -120, -90, -60, -30, 0, 30,
# 60, 90, 120, 150]))
ax.set_xticks([])
ax.grid(True, alpha=0.3, linestyle=':')

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                  fraction=0.05, pad=0.08, shrink=0.7)
cbar.set_label('Seed diagonal coordinate', fontsize=10)

plt.tight_layout()
plt.show()


#%%
# ======================================================================
# Orbital period classification
# ======================================================================
import matplotlib.colors as mcolors

max_p = 20            # highest period to test
recurrence_tol = 0.05  # mean geodesic distance (rad) to call an orbit period-p
aspect_thresh = 0.15   # PCA eigenvalue ratio below this → quasi-periodic (1D curve)

# Reconstruct Cartesian vectors from stored angles
x_traj = np.sin(theta_arr) * np.cos(phi_arr)
y_traj = np.sin(theta_arr) * np.sin(phi_arr)
z_traj = np.cos(theta_arr)
cart_traj = np.stack([x_traj, y_traj, z_traj], axis=-1)  # (n_keep, n_seeds, 3)

# Drop first half to remove any residual transient
traj = cart_traj[n_keep // 2:]   # (n_t, n_seeds, 3)
n_t = traj.shape[0]

# --- Recurrence: for period-p, v[i] ≈ v[i+p] so mean geodesic ≈ 0 ---
period_arr = np.zeros(n_seeds, dtype=int)   # 0 = unclassified
for p in range(1, max_p + 1):
    v1 = traj[:-p]    # (n_t-p, n_seeds, 3)
    v2 = traj[p:]
    dots = np.clip(np.einsum('ijk,ijk->ij', v1, v2), -1.0, 1.0)
    mean_dist = np.mean(np.arccos(dots), axis=0)  # (n_seeds,)
    period_arr[(period_arr == 0) & (mean_dist < recurrence_tol)] = p

# --- PCA: separate quasi-periodic (1D) from chaotic (2D) ---
# Eigenvalue ratio λ₂/λ₁: near 0 for a curve, near 1 for a filled region
aspect = np.zeros(n_seeds)
for s in range(n_seeds):
    cov = np.cov(traj[:, s, :].T)             # (3, 3)
    eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]   # descending
    aspect[s] = eigs[1] / (eigs[0] + 1e-12)

# period_code: -1 = chaotic, 0 = quasi-periodic KAM, p>0 = true period-p
period_code = period_arr.astype(float)
period_code[(period_arr == 0) & (aspect > aspect_thresh)] = -1.0

print(f"Classification  (aspect_thresh={aspect_thresh}, recurrence_tol={recurrence_tol} rad)")
for c in sorted(np.unique(period_code)):
    mask = period_code == c
    label = "Chaotic" if c < 0 else ("Quasi-periodic" if c == 0 else f"Period {int(c)}")
    print(f"  {label:15s}: {mask.sum():3d} seeds  "
          f"aspect={aspect[mask].mean():.3f}±{aspect[mask].std():.3f}")

# --- Discrete colormap: dark-grey for chaos, light-grey for QP, tab20 for periods ---
# Remap period_code → consecutive integers so every colorbar band has equal width
unique_codes = sorted(np.unique(period_code))
n_codes = len(unique_codes)
code_to_idx = {c: i for i, c in enumerate(unique_codes)}

color_list = []
for c in unique_codes:
    if c < 0:
        color_list.append('#444444')
    elif c == 0:
        color_list.append('#aaaaaa')
    else:
        color_list.append(plt.cm.tab20.colors[int(c - 1) % 20])

cmap_disc = mcolors.ListedColormap(color_list)
norm_disc = mcolors.BoundaryNorm(np.arange(-0.5, n_codes), cmap_disc.N)
tick_labels = ['!' if c < 0 else ('QP' if c == 0 else f'{int(c)}')
               for c in unique_codes]

# Remap seed codes and trajectory codes to consecutive indices
idx_seeds = np.array([code_to_idx[c] for c in period_code])
idx_traj = np.broadcast_to(idx_seeds, phi_arr.shape)

# ======================================================================
# Plot 1: Poincaré section — every trajectory point colored by period
# ======================================================================

#%%
lon = phi_arr.ravel()
lat = np.pi / 2 - theta_arr.ravel()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='hammer')
sc = ax.scatter(lon, lat, c=idx_traj.ravel(), cmap=cmap_disc, norm=norm_disc,
                s=0.5, alpha=0.6)
ax.set_title(f'Poincaré Section: Orbital Period  (κ={kappa}, order={order})', fontsize=13, pad=20)
ax.set_xticks([])
ax.grid(True, alpha=0.3, linestyle=':')
cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                    fraction=0.05, pad=0.08, shrink=0.7, ticks=np.arange(n_codes))
cbar.set_ticklabels(tick_labels)
cbar.set_label('Orbit type', fontsize=10)
plt.tight_layout()
plt.show()

#%%
# ======================================================================
# Plot 2: Seed-origin map — starting (θ₀, φ₀) colored by period
# ======================================================================

lon0 = seeds_phi0
lat0 = np.pi / 2 - seeds_theta0

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='hammer')
sc = ax.scatter(lon0, lat0, c=idx_seeds, cmap=cmap_disc, norm=norm_disc,
                s=20, edgecolors='k', linewidths=0.4, zorder=3)
ax.set_title(f'Initial Condition Map: Orbital Period  (κ={kappa}, order={order})', fontsize=13, pad=20)
ax.set_xticks([])
ax.grid(True, alpha=0.3, linestyle=':')
cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                    fraction=0.05, pad=0.08, shrink=0.7, ticks=np.arange(n_codes))
cbar.set_ticklabels(tick_labels)
cbar.set_label('Orbit type', fontsize=10)
plt.tight_layout()
plt.show()


#%%
# ======================================================================
# Maximal Lyapunov exponent per seed
# ======================================================================

# Reconstruct initial states (v has been overwritten by the trajectory loop above)
v_ly = np.array([cf.sph_to_cart(th, ph) for th, ph in zip(seeds_theta0, seeds_phi0)])

# Random unit tangent vector perpendicular to each starting point
rng = np.random.default_rng(0)
dv = rng.standard_normal(v_ly.shape)
dv -= np.sum(dv * v_ly, axis=1, keepdims=True) * v_ly   # project onto tangent plane
dv /= np.linalg.norm(dv, axis=1, keepdims=True)

lyap_sum = np.zeros(n_seeds)
eps_ly = 1e-7

for i in trange(n_iter, desc='Lyapunov'):
    # Tangent map via central finite differences: J(v)·dv ≈ [f(v+ε·dv) - f(v-ε·dv)] / 2ε
    v_plus  = cf.kicked_top_step(v_ly + eps_ly * dv, kappa, alpha, order)
    v_minus = cf.kicked_top_step(v_ly - eps_ly * dv, kappa, alpha, order)
    v_ly    = cf.kicked_top_step(v_ly, kappa, alpha, order)

    dv_new  = (v_plus - v_minus) / (2 * eps_ly)
    dv_new -= np.sum(dv_new * v_ly, axis=1, keepdims=True) * v_ly  # re-project onto tangent plane

    norms = np.linalg.norm(dv_new, axis=1)
    if i >= n_discard:
        lyap_sum += np.log(norms + 1e-300)
    dv = dv_new / np.maximum(norms[:, np.newaxis], 1e-300)

lyap_exp = lyap_sum / (n_iter - n_discard)

print(f"Lyapunov exponents — min={lyap_exp.min():.4f}  max={lyap_exp.max():.4f}  "
      f"median={np.median(lyap_exp):.4f}")

# Seed-origin map colored by Lyapunov exponent
lon0 = seeds_phi0
lat0 = np.pi / 2 - seeds_theta0

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='hammer')
sc = ax.scatter(lon0, lat0, c=lyap_exp, cmap='inferno', s=20,
                vmin=0, vmax=np.percentile(lyap_exp, 99))
ax.set_title(f'Maximal Lyapunov Exponent  (κ={kappa}, order={order})', fontsize=13, pad=20)
ax.set_xticks([])
ax.grid(True, alpha=0.3, linestyle=':')
cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                    fraction=0.05, pad=0.08, shrink=0.7)
cbar.set_label(r'$\lambda_{\max}$', fontsize=10)
plt.tight_layout()
plt.show()


#%%

# Take the generated data and bin into a histogram

n_bins = 100

fig, ax = plt.subplots(figsize=(8, 6))

h, xedges, yedges, img = ax.hist2d(
    phi_arr.ravel(), theta_arr.ravel(),
    bins=n_bins,
    range=[[-np.pi, np.pi], [0, np.pi]],
    cmap='inferno',
)

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(0, np.pi)
ax.set_xlabel(r'$\phi$ (azimuthal angle)', fontsize=12)
ax.set_ylabel(r'$\theta$ (polar angle)', fontsize=12)
ax.set_title(f'Poincaré Section: 2D Histogram  (κ={kappa}, p={order})', fontsize=13)

ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax.set_yticks([0, np.pi/2, np.pi])
ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
ax.grid(True, alpha=0.3, linestyle=':')

cbar = plt.colorbar(img, ax=ax)
cbar.set_label('Point density', fontsize=10)

plt.tight_layout()
plt.show()