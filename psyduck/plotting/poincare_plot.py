"""Poincaré-section plotting utilities for the classical kicked top.

The classical kicked top is iterated stroboscopically — every kick produces
one point in (phi, theta) per seed.  This module groups the repetitive
simulation/plotting code into a small, composable API:

* :func:`simulate_kicked_top`  — run the map on a meshgrid of seeds and
  return per-iterate (phi, theta) arrays.
* :func:`diagonal_seed_colors` — per-seed diagonal-gradient colour values
  used by every coloured plot in Section 4.
* :func:`poincare_plot_rectangular` / :func:`poincare_plot_hammer` /
  :func:`poincare_plot_3d` — single-frame plotters following the same
  style as :mod:`psyduck.plotting.wigner_plot`.
* :func:`make_poincare_gif` — assemble a sequence of frames (parameter
  sweep, camera sweep, …) into an animated GIF.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_kicked_top(kappa, alpha, n_seeds_phi, n_seeds_theta,
                        n_iter, n_discard=0, order=2,
                        theta_range=(0, np.pi),
                        phi_range=(-np.pi, np.pi)):
    """Iterate the classical kicked top on a meshgrid of seeds.

    Parameters
    ----------
    kappa, alpha : float
        Kick strength and rotation angle (see :func:`psyduck.classical_dynamics.kicked_top_step`).
    n_seeds_phi, n_seeds_theta : int
        Seed-grid resolution along the azimuthal / polar angles.
    n_iter : int
        Number of kicks to apply.  ``n_iter = 0`` returns the seeds untouched.
    n_discard : int, optional
        Number of leading iterates to drop as transients (default 0).
    order : int, optional
        Nonlinearity order of the twist (default 2).
    theta_range, phi_range : tuple of float, optional
        Seed-grid extents.  Defaults span the full sphere.

    Returns
    -------
    phi_arr, theta_arr : ndarray, shape (n_iter + 1 - n_discard, n_seeds)
        Stroboscopic azimuthal / polar angles for every retained iterate
        and seed.  ``n_seeds = n_seeds_phi * n_seeds_theta``.
    seeds_theta0, seeds_phi0 : ndarray, shape (n_seeds,)
        Initial spherical coordinates of the seeds, in meshgrid-flattened
        order.  Useful for colouring trajectories by their starting point.
    """
    from psyduck.classical_dynamics import kicked_top_step  # deferred to avoid circular import

    phis_seed = np.linspace(phi_range[0], phi_range[1],
                            n_seeds_phi, endpoint=False)
    thetas_seed = np.linspace(theta_range[0], theta_range[1],
                              n_seeds_theta, endpoint=False)

    seeds_theta0, seeds_phi0 = (a.ravel() for a in
                                np.meshgrid(thetas_seed, phis_seed,
                                            indexing='ij'))
    v = np.stack([np.sin(seeds_theta0) * np.cos(seeds_phi0),
                  np.sin(seeds_theta0) * np.sin(seeds_phi0),
                  np.cos(seeds_theta0)], axis=-1)
    n_seeds = v.shape[0]

    n_keep = n_iter + 1 - n_discard
    phi_arr   = np.empty((n_keep, n_seeds))
    theta_arr = np.empty((n_keep, n_seeds))
    for i in range(n_iter + 1):
        if i >= n_discard:
            theta_arr[i - n_discard] = np.arccos(np.clip(v[..., 2], -1.0, 1.0))
            phi_arr[i - n_discard]   = np.arctan2(v[..., 1], v[..., 0])
        if i < n_iter:
            v = kicked_top_step(v, kappa, alpha, order)

    return phi_arr, theta_arr, seeds_theta0, seeds_phi0


def diagonal_seed_colors(seeds_theta0, seeds_phi0,
                         theta_range=(0, np.pi),
                         phi_range=(-np.pi, np.pi)):
    """Per-seed diagonal-gradient colour values used for the coloured Poincaré plots.

    Values lie in ``[0, 2]`` for the default ranges: the polar coordinate is
    rescaled to ``[0, 1]`` and the azimuthal coordinate to ``[0, 1]``, then
    they are added.  Broadcasting against ``phi_arr`` reproduces the
    "colour by seed" convention used throughout Section 4.
    """
    t_lo, t_hi = theta_range
    p_lo, p_hi = phi_range
    return ((seeds_theta0 - t_lo) / (t_hi - t_lo)
            + (seeds_phi0   - p_lo) / (p_hi - p_lo))


# ---------------------------------------------------------------------------
# Single-frame plotters
# ---------------------------------------------------------------------------

def _broadcast_color(c, target_shape):
    """Broadcast a per-seed colour array to the full (n_keep, n_seeds) shape."""
    if c is None or isinstance(c, str):
        return c
    c = np.asarray(c)
    flat_len = int(np.prod(target_shape))
    if c.size == flat_len:
        return c.ravel()
    return np.broadcast_to(c, target_shape).ravel()


def _apply_rect_axes(ax, xlim=(-np.pi, np.pi), ylim=(0, np.pi)):
    """Standard tick / grid / label styling for a rectangular Poincaré axes."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\phi$ (azimuthal angle)', fontsize=12)
    ax.set_ylabel(r'$\theta$ (polar angle)',   fontsize=12)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_yticks([0, np.pi / 2, np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.grid(True, alpha=0.3, linestyle=':')


def poincare_plot_rectangular(phi_arr, theta_arr, *,
                              c=None, cmap=None, s=0.5, alpha=0.5,
                              vmin=None, vmax=None, title=None,
                              figsize=(8, 6), fig=None, ax=None,
                              xlim=(-np.pi, np.pi), ylim=(0, np.pi),
                              colorbar=False, cbar_label=None, **kwargs):
    """Plot a Poincaré section on rectangular (phi, theta) axes.

    Parameters
    ----------
    phi_arr, theta_arr : ndarray
        Azimuthal / polar coordinates of the stroboscopic samples.  Any
        shape is accepted — both arrays are flattened.
    c : str, scalar, or ndarray, optional
        Marker colour or per-point colour values.  A per-seed array of
        length ``n_seeds`` is auto-broadcast to ``phi_arr.shape``.
    cmap : str or Colormap, optional
        Used when ``c`` is numeric.
    s, alpha : float, optional
        Marker size and transparency.
    vmin, vmax : float, optional
        Colour-scale limits.
    title : str, optional
        Axes title.
    figsize : tuple, optional
        Figure size when creating a new figure.
    fig, ax : optional
        Reuse an existing figure / axes.
    xlim, ylim : tuple, optional
        Axis limits (defaults: full sphere).
    colorbar : bool, optional
        If True and ``c`` is numeric, draw a colorbar.
    cbar_label : str, optional
        Colorbar label.
    **kwargs : forwarded to :func:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    fig, ax, sc : matplotlib objects.
    """
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = fig.add_subplot(111)
    phi_arr = np.asarray(phi_arr)
    theta_arr = np.asarray(theta_arr)
    c_eff = _broadcast_color(c, phi_arr.shape)
    sc = ax.scatter(phi_arr.ravel(), theta_arr.ravel(),
                    c=c_eff, cmap=cmap, s=s, alpha=alpha,
                    vmin=vmin, vmax=vmax, **kwargs)
    _apply_rect_axes(ax, xlim=xlim, ylim=ylim)
    if title is not None:
        ax.set_title(title, fontsize=13)
    if colorbar and not isinstance(c_eff, (str, type(None))):
        cbar = plt.colorbar(sc, ax=ax)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=10)
    return fig, ax, sc


def poincare_plot_hammer(phi_arr, theta_arr, *,
                         c=None, cmap=None, s=0.5, alpha=0.5,
                         vmin=None, vmax=None, title=None,
                         figsize=(8, 5), fig=None, ax=None,
                         colorbar=False, cbar_label=None, **kwargs):
    """Plot a Poincaré section on a Hammer equal-area projection.

    The polar angle ``theta in [0, pi]`` is converted to latitude
    ``pi/2 - theta`` so the north pole (theta = 0) sits at the top of the
    map, matching :func:`spherical_plot_hammer`'s convention.

    Parameters are the same as :func:`poincare_plot_rectangular`.
    """
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize,
                                   subplot_kw={'projection': 'hammer'})
        else:
            ax = fig.add_subplot(111, projection='hammer')
    phi_arr = np.asarray(phi_arr)
    theta_arr = np.asarray(theta_arr)
    latitude = np.pi / 2 - theta_arr.ravel()
    c_eff = _broadcast_color(c, phi_arr.shape)
    sc = ax.scatter(phi_arr.ravel(), latitude,
                    c=c_eff, cmap=cmap, s=s, alpha=alpha,
                    vmin=vmin, vmax=vmax, **kwargs)
    if title is not None:
        ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3, linestyle=':')
    if colorbar and not isinstance(c_eff, (str, type(None))):
        cbar = plt.colorbar(sc, ax=ax)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=10)
    return fig, ax, sc


def poincare_plot_3d(phi_arr, theta_arr, *,
                     c=None, cmap=None, s=0.4, alpha=0.6,
                     vmin=None, vmax=None, title=None,
                     azim=30.0, elev=25.0, hide_back=True,
                     sphere_color='white',
                     figsize=(6, 6), fig=None, ax=None, **kwargs):
    """Plot a Poincaré section on a unit sphere with a fixed camera angle.

    Points are lifted slightly outside the sphere (r = 1.001) so they read
    cleanly on top of the white reference sphere.  When ``hide_back`` is
    True only points on the camera-facing hemisphere are drawn — this is
    what produces the "spinning sphere" look in :func:`make_poincare_gif`
    when the camera angles are swept frame by frame.

    Additional parameters:

    azim, elev : float, optional
        Camera angles in degrees.
    hide_back : bool, optional
        Hide points on the far hemisphere relative to the camera.
    sphere_color : str, optional
        Reference-sphere fill colour.

    Returns
    -------
    fig, ax : matplotlib objects (no ``sc`` because scatter is hemispherewise).
    """
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    phi_arr = np.asarray(phi_arr)
    theta_arr = np.asarray(theta_arr)
    _r = 1.001
    phi_flat   = phi_arr.ravel()
    theta_flat = theta_arr.ravel()
    x_pts = _r * np.sin(theta_flat) * np.cos(phi_flat)
    y_pts = _r * np.sin(theta_flat) * np.sin(phi_flat)
    z_pts = _r * np.cos(theta_flat)
    c_eff = _broadcast_color(c, phi_arr.shape)

    # Reference sphere mesh
    _u = np.linspace(0, 2 * np.pi, 40)
    _v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(_u), np.sin(_v))
    ys = np.outer(np.sin(_u), np.sin(_v))
    zs = np.outer(np.ones_like(_u), np.cos(_v))
    ax.plot_surface(xs, ys, zs, color=sphere_color, alpha=1.0,
                    linewidth=0, edgecolors='none',
                    antialiased=False, shade=False, zorder=0)

    if hide_back:
        vd = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                       np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                       np.sin(np.radians(elev))])
        vis = (x_pts * vd[0] + y_pts * vd[1] + z_pts * vd[2]) > 0
        x_pts, y_pts, z_pts = x_pts[vis], y_pts[vis], z_pts[vis]
        if isinstance(c_eff, np.ndarray):
            c_eff = c_eff[vis]

    ax.scatter(x_pts, y_pts, z_pts, c=c_eff, cmap=cmap, s=s, alpha=alpha,
               vmin=vmin, vmax=vmax, depthshade=False, zorder=5, **kwargs)
    ax.view_init(elev=float(elev), azim=float(azim))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=11, pad=-10)
    return fig, ax


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

_PROJECTIONS = {
    'rectangular': poincare_plot_rectangular,
    'hammer':      poincare_plot_hammer,
    '3d':          poincare_plot_3d,
}


def make_poincare_gif(filename, frames, projection='rectangular',
                      fps=2, dpi=100, verbose=False, **plot_kwargs):
    """Assemble a sequence of Poincaré-map frames into an animated GIF.

    Parameters
    ----------
    filename : str
        Output GIF path.
    frames : list of dict
        Per-frame keyword arguments.  Each dict must include ``phi_arr`` and
        ``theta_arr``; any other key (e.g. ``c``, ``title``, ``azim``,
        ``elev``) overrides the matching ``plot_kwargs`` default for that
        frame only.
    projection : {'rectangular', 'hammer', '3d'}
        Which single-frame plotter to use.
    fps : float, optional
        Frames per second.
    dpi : int, optional
        Figure DPI for each rendered frame.
    verbose : bool, optional
        Print progress per frame.
    **plot_kwargs : forwarded to the chosen plot function for every frame
        (e.g. ``cmap='nipy_spectral'``, ``vmin=0``, ``vmax=2``).

    Returns
    -------
    filename : str
    """
    from PIL import Image

    if projection not in _PROJECTIONS:
        raise ValueError(
            f"projection must be one of {tuple(_PROJECTIONS)}, got {projection!r}"
        )
    plot_fn = _PROJECTIONS[projection]

    images = []
    for k, frame in enumerate(frames):
        kw = dict(plot_kwargs)
        kw.update(frame)
        phi = kw.pop('phi_arr')
        theta = kw.pop('theta_arr')
        result = plot_fn(phi, theta, **kw)
        fig = result[0]
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        images.append(Image.open(buf).copy())
        plt.close(fig)
        if verbose:
            print(f'  frame {k + 1}/{len(frames)}')

    images[0].save(filename, save_all=True, append_images=images[1:],
                   loop=0, duration=int(1000 / fps))
    return filename
