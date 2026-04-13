import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from qutip import *
from qutip.wigner import spin_q_function
from qutip import Qobj, spin_coherent
import cmath, math


# ---------------------------------------------------------------------------
# Generic spherical-surface helpers (no QuTiP dependency)
# ---------------------------------------------------------------------------

def spherical_plot_3d(data, theta_mesh=None, phi_mesh=None, cmap='bwr', vmin=None, vmax=None,
                      fig=None, ax=None, **kwargs):
    """Plot scalar data on a unit sphere as a coloured surface.

    :param data: 2-D array of values, shape (n_theta, n_phi) — theta-major
                 (rows = polar angle, cols = azimuthal angle).
    :param theta_mesh: Polar-angle mesh in [0, π], same shape as data.
                       Defaults to linspace(0, π, n_theta) derived from data.shape.
    :param phi_mesh: Azimuthal-angle mesh in [−π, π], same shape as data.
                     Defaults to linspace(−π, π, n_phi) derived from data.shape.
    :param cmap: Matplotlib colormap name or object.
    :param vmin: Colour-scale minimum (default: data.min()).
    :param vmax: Colour-scale maximum (default: data.max()).
    :param fig: Existing Figure; created if None.
    :param ax: Existing 3-D Axes; created if None (added to fig).
    :param kwargs: Forwarded to ax.plot_surface.
    :return: (fig, ax)
    """
    if theta_mesh is None or phi_mesh is None:
        n_theta, n_phi = data.shape
        _theta, _phi = np.meshgrid(np.linspace(0, np.pi, n_theta),
                                   np.linspace(0, 2*np.pi, n_phi),
                                   indexing='ij')
        if theta_mesh is None:
            theta_mesh = _theta
        if phi_mesh is None:
            phi_mesh = _phi
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    # Shifted convention: theta measured from the equator, matching QuTiP's spin_wigner
    # orientation (z = -cos(theta), so m = -I is at the bottom).
    x = np.cos(phi_mesh) * np.cos(theta_mesh - np.pi / 2)
    y = np.sin(phi_mesh) * np.cos(theta_mesh - np.pi / 2)
    z = np.sin(theta_mesh - np.pi / 2)

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pcm = ax.plot_surface(x, y, z, facecolors=cmap_obj(norm(data)),
                    rstride=1, cstride=1, shade=False, **kwargs)
    return fig, ax, pcm


def spherical_plot_hammer(data, theta_mesh=None, phi_mesh=None, cmap='bwr', vmin=None, vmax=None,
                          fig=None, ax=None, **kwargs):
    """Plot scalar data on a Hammer equal-area projection.

    Phi must be in [−π, π].  Theta is shifted internally to [−π/2, π/2].

    :param data: 2-D array of values, shape (n_theta, n_phi).
    :param theta_mesh: Polar-angle mesh in [0, π].
                       Defaults to linspace(0, π, n_theta) derived from data.shape.
    :param phi_mesh: Azimuthal-angle mesh in [−π, π].
                     Defaults to linspace(−π, π, n_phi) derived from data.shape.
    :param cmap: Matplotlib colormap name or object.
    :param vmin: Colour-scale minimum (default: data.min()).
    :param vmax: Colour-scale maximum (default: data.max()).
    :param fig: Existing Figure; created if None.
    :param ax: Existing Axes with hammer projection; created if None.
    :param kwargs: Forwarded to ax.pcolormesh.
    :return: (fig, ax)
    """
    if theta_mesh is None or phi_mesh is None:
        n_theta, n_phi = data.shape
        _theta, _phi = np.meshgrid(np.linspace(0, np.pi, n_theta),
                                   np.linspace(-np.pi, np.pi, n_phi),
                                   indexing='ij')
        if theta_mesh is None:
            theta_mesh = _theta
        if phi_mesh is None:
            phi_mesh = _phi
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'hammer'})
        else:
            ax = fig.add_subplot(111, projection='hammer')
    pcm = ax.pcolormesh(phi_mesh, theta_mesh - np.pi / 2, data,
                  cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    # Polar-angle (theta) tick labels along the central meridian
    ax.set_yticks([np.pi / 2, np.pi / 4, 0, -np.pi / 4, -np.pi / 2])
    ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°'], fontsize=7)

    # Azimuthal-angle (phi) tick labels along the equator
    ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
    ax.set_xticklabels(['-y', '+x', '+y'], fontsize=7)

    ax.grid(False)
    return fig, ax, pcm


def spherical_plot_3d_with_projections(data, theta_mesh=None, phi_mesh=None, cmap='RdBu',
                                       vmin=None, vmax=None, r=1,
                                       fig=None, ax=None, figsize=(8, 6), **kwargs):
    """Plot scalar data on a sphere with three flat side projections.

    The main sphere is drawn with :func:`spherical_plot_3d`.  Three additional
    flat panels show projections onto the x=−1.5, y=+1.5, and z=−1.5 planes.

    :param data: 2-D array of values, shape (n_phi, n_theta) — phi-major,
                 as returned by QuTiP's spin_wigner / spin_q_function.
    :param theta_mesh: Polar-angle mesh in [0, π], same shape as data.
    :param phi_mesh: Azimuthal-angle mesh in [0, 2π], same shape as data.
    :param cmap: Matplotlib colormap name or object.
    :param vmin: Colour-scale minimum (default: data.min()).
    :param vmax: Colour-scale maximum (default: data.max()).
    :param r: Sphere radius (default 1).
    :param fig: Existing Figure; created if None.
    :param ax: Existing 3-D Axes; created if None.
    :param figsize: Figure size when creating a new figure.
    :param kwargs: Forwarded to all ax.plot_surface calls.
    :return: (fig, ax)
    """
    try:
        cmap_obj = plt.get_cmap(cmap)
    except Exception:
        cmap_obj = cmap

    n_theta, n_phi = data.shape

    if theta_mesh is None or phi_mesh is None:
        _theta, _phi = np.meshgrid(np.linspace(0, np.pi, n_theta),
                                   np.linspace(0, 2*np.pi, n_phi),
                                   indexing='ij')
        if theta_mesh is None:
            theta_mesh = _theta
        if phi_mesh is None:
            phi_mesh = _phi

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        fig = plt.figure(dpi=300, figsize=figsize, edgecolor='None')
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    light_source = mpl.colors.LightSource(azdeg=-45, altdeg=10, hsv_min_val=1, hsv_max_val=1,
                                           hsv_min_sat=1, hsv_max_sat=1)
    _, _, pcm = spherical_plot_3d(data, theta_mesh, phi_mesh, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                      fig=fig, ax=ax, lightsource=light_source, **kwargs)

    # -- y projection: second half of phi values (y ≥ 0 hemisphere, 0 ≤ φ ≤ π) --
    theta_mesh_y = theta_mesh[:, :n_phi // 2]
    phi_mesh_y = phi_mesh[:, :n_phi // 2]
    data_y = data[:, :n_phi // 2]
    x = r * np.cos(phi_mesh_y) * np.cos(theta_mesh_y - np.pi / 2)
    y = r * np.sin(phi_mesh_y) * np.cos(theta_mesh_y - np.pi / 2)
    z = r * np.sin(theta_mesh_y - np.pi / 2)
    ax.plot_surface(x, np.zeros_like(x) + 1.5, z, facecolors=cmap_obj(norm(data_y)),
                    shade=False, rstride=1, cstride=1, **kwargs)

    # -- z projection: first half of theta values (upper hemisphere) --
    theta_mesh_z = theta_mesh[:n_theta // 2, :]
    phi_mesh_z = phi_mesh[:n_theta // 2, :]
    data_z = data[:n_theta // 2, :]
    x = r * np.cos(phi_mesh_z) * np.cos(theta_mesh_z - np.pi / 2)
    y = r * np.sin(phi_mesh_z) * np.cos(theta_mesh_z - np.pi / 2)
    z = r * np.sin(theta_mesh_z - np.pi / 2)
    ax.plot_surface(x, y, np.zeros_like(x) - 1.5, facecolors=cmap_obj(norm(data_z)),
                    shade=False, rstride=1, cstride=1, **kwargs)

    # -- x projection: middle half of phi (−π/2 ≤ φ ≤ π/2, x ≥ 0 hemisphere) --
    theta_mesh_x = theta_mesh[:, n_phi // 4:3 * n_phi // 4]
    phi_mesh_x = phi_mesh[:, n_phi // 4:3 * n_phi // 4]
    data_x = data[:, n_phi // 4:3 * n_phi // 4]
    x = r * np.cos(phi_mesh_x) * np.cos(theta_mesh_x - np.pi / 2)
    y = r * np.sin(phi_mesh_x) * np.cos(theta_mesh_x - np.pi / 2)
    z = r * np.sin(theta_mesh_x - np.pi / 2)
    ax.plot_surface(np.zeros_like(y) - 1.5, y, z, facecolors=cmap_obj(norm(data_x)),
                    shade=False, rstride=1, cstride=1, **kwargs)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    ax.plot([-1.5, -1.5], [-1.6, 1.6], [-1.5, -1.5], color='grey', lw=0.4)
    ax.plot([-1.6, 1.6], [1.5, 1.5], [-1.5, -1.5], color='grey', lw=0.4)
    ax.plot([-1.5, -1.5], [1.5, 1.5], [-1.6, 1.6], color='grey', lw=0.4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(0)
    ax.xaxis.pane.set_facecolor((0.85, 0.85, 0.85, 1))
    ax.yaxis.pane.set_facecolor((0.8, 0.8, 0.8, 1))
    ax.zaxis.pane.set_facecolor((0.825, 0.825, 0.825, 1))
    ax.view_init(22, -60)

    return fig, ax, pcm


def spherical_plot_polar(data, theta_mesh=None, phi_mesh=None, cmap='bwr', vmin=None, vmax=None,
                         fig=None, ax=None, **kwargs):
    """Plot scalar data on a polar (azimuthal equidistant) projection.

    Phi maps to the angular axis; theta maps to the radial axis (θ=0 at centre).
    Phi should be in [0, 2π].

    :param data: 2-D array of values, shape (n_theta, n_phi).
    :param theta_mesh: Polar-angle mesh in [0, π].
                       Defaults to linspace(0, π, n_theta) derived from data.shape.
    :param phi_mesh: Azimuthal-angle mesh in [0, 2π].
                     Defaults to linspace(0, 2π, n_phi) derived from data.shape.
    :param cmap: Matplotlib colormap name or object.
    :param vmin: Colour-scale minimum (default: data.min()).
    :param vmax: Colour-scale maximum (default: data.max()).
    :param fig: Existing Figure; created if None.
    :param ax: Existing Axes with polar projection; created if None.
    :param kwargs: Forwarded to ax.pcolormesh.
    :return: (fig, ax)
    """
    if theta_mesh is None or phi_mesh is None:
        n_theta, n_phi = data.shape
        _theta, _phi = np.meshgrid(np.linspace(0, np.pi, n_theta),
                                   np.linspace(0, 2 * np.pi, n_phi),
                                   indexing='ij')
        if theta_mesh is None:
            theta_mesh = _theta
        if phi_mesh is None:
            phi_mesh = _phi
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        else:
            ax = fig.add_subplot(111, projection='polar')
    ax.pcolormesh(phi_mesh, theta_mesh, data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig, ax


# ---------------------------------------------------------------------------
# Wigner / Husimi wrappers
# ---------------------------------------------------------------------------

def wigner_plot(rho, dpi=300, prob_function='wigner'):
    nTheta, nPhi = (101, 201)
    theta = np.linspace(0, np.pi, num=nTheta, endpoint=True)
    phi = np.linspace(-np.pi, np.pi, num=nPhi, endpoint=True)
    if prob_function == 'husimi':
        W, theta_mesh, phi_mesh = spin_q_function(rho, theta, phi)
    else:
        W, theta_mesh, phi_mesh = spin_wigner(rho, theta, phi)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    spherical_plot_3d(W, theta_mesh, phi_mesh, fig=fig, ax=ax)
    ax.set_axis_off()
    s = 0.7
    ax.set_xlim([-s, s])
    ax.set_ylim([-s, s])
    ax.set_zlim([-s, s])

    ax2 = fig.add_subplot(1, 2, 2, projection='hammer')
    spherical_plot_hammer(W, theta_mesh, phi_mesh, fig=fig, ax=ax2)


def projection_plot_spin_wigner(psi, n_theta=200, n_phi=200, r=1, cmap='RdBu', figsize=(8, 6),
                                ax=None, fig=None, vmin=None, vmax=None, prob_function='wigner',
                                **kwargs):
    """Plot spin Wigner/Husimi function on a sphere with three side projections.

    :param psi: QuTiP Qobj state vector or density matrix.
    :param n_theta: Number of polar-angle grid points.
    :param n_phi: Number of azimuthal-angle grid points.
    :param r: Sphere radius (default 1).
    :param cmap: Matplotlib colormap.
    :param figsize: Figure size when creating a new figure.
    :param ax: Existing 3-D Axes; created if None.
    :param fig: Existing Figure; created if None.
    :param vmin/vmax: Colour-scale limits (default: data min/max).
    :param prob_function: 'wigner' or 'husimi'.
    :param kwargs: Forwarded to plot_surface calls.
    :return: (fig, ax, data, theta_mesh, phi_mesh)
    """
    try:
        cmap = plt.get_cmap(cmap)
    except Exception:
        cmap = cmap

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)

    if prob_function == 'husimi':
        wigner, theta_mesh, phi_mesh = spin_q_function(psi, theta, phi)
    elif prob_function == 'wigner':
        wigner, theta_mesh, phi_mesh = spin_wigner(psi, theta, phi)

    fig, ax = spherical_plot_3d_with_projections(wigner, theta_mesh, phi_mesh,
                                                  cmap=cmap, vmin=vmin, vmax=vmax,
                                                  r=r, fig=fig, ax=ax, figsize=figsize,
                                                  **kwargs)
    return fig, ax, wigner, theta_mesh, phi_mesh


def wigner_plot_3d(rho, n_theta=101, n_phi=201, cmap='bwr', prob_function='wigner',
                   vmin=None, vmax=None, fig=None, ax=None, **kwargs):
    """Plot spin Wigner function as a coloured surface on a sphere."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    if prob_function == 'husimi':
        W, theta_mesh, phi_mesh = spin_q_function(rho, theta, phi)
    elif prob_function == 'wigner':
        W, theta_mesh, phi_mesh = spin_wigner(rho, theta, phi)

    if ax is None:
        fig = plt.figure() if fig is None else fig
        ax = fig.add_subplot(111, projection='3d')
    fig, ax = spherical_plot_3d(W, theta_mesh, phi_mesh, cmap=cmap, vmin=vmin, vmax=vmax,
                                fig=fig, ax=ax, **kwargs)
    s = 1.1
    ax.set_xlim([-s, s])
    ax.set_ylim([-s, s])
    ax.set_zlim([-s, s])
    ax.set_axis_off()
    return fig, ax


def wigner_plot_hammer(rho, n_theta=101, n_phi=201, cmap='bwr', prob_function='wigner',
                       vmin=None, vmax=None, fig=None, ax=None, **kwargs):
    """Plot spin Wigner function on a Hammer equal-area projection."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    if prob_function == 'husimi':
        W, theta_mesh, phi_mesh = spin_q_function(rho, theta, phi)
    elif prob_function == 'wigner':
        W, theta_mesh, phi_mesh = spin_wigner(rho, theta, phi)

    return spherical_plot_hammer(W, theta_mesh, phi_mesh, cmap=cmap, vmin=vmin, vmax=vmax,
                                 fig=fig, ax=ax, **kwargs)


def wigner_plot_polar(rho, n_theta=101, n_phi=201, cmap='bwr', prob_function='wigner',
                      vmin=None, vmax=None, fig=None, ax=None, **kwargs):
    """Plot spin Wigner function on a polar (azimuthal equidistant) projection.

    Azimuthal angle phi maps to the angular axis; polar angle theta maps to the
    radial axis (theta=0 at centre, theta=pi at the outer edge).
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    if prob_function == 'husimi':
        W, theta_mesh, phi_mesh = spin_q_function(rho, theta, phi)
    elif prob_function == 'wigner':
        W, theta_mesh, phi_mesh = spin_wigner(rho, theta, phi)

    return spherical_plot_polar(W, theta_mesh, phi_mesh, cmap=cmap, vmin=vmin, vmax=vmax,
                                fig=fig, ax=ax, **kwargs)


def make_wigner_gif(states, filename='wigner.gif', projection='hammer', fps=10, dpi=100, **kwargs):
    """Create a GIF of the spin Wigner function over a sequence of states.

    :param states: list of qt.Qobj kets, or (N, d) complex numpy array where each
                   row is a flattened state vector (e.g. from nucleus.state.full().flatten())
    :param filename: output filename (default 'wigner.gif')
    :param projection: '3d', 'hammer', or 'polar' (default '3d')
    :param fps: frames per second
    :param dpi: figure resolution
    :param kwargs: passed to the underlying plot function (n_theta, n_phi, cmap, etc.)
    :return: filename
    """
    import io
    from PIL import Image

    if isinstance(states, np.ndarray):
        qstates = [Qobj(states[i].reshape(-1, 1)) for i in range(states.shape[0])]
    else:
        qstates = list(states)

    _plot_funcs = {
        '3d':     wigner_plot_3d,
        'hammer': wigner_plot_hammer,
        'polar':  wigner_plot_polar,
    }
    if projection not in _plot_funcs:
        raise ValueError(f"projection must be '3d', 'hammer', or 'polar', got {projection!r}")
    plot_fn = _plot_funcs[projection]

    frames = []
    for state in qstates:
        fig, ax = plot_fn(state, **kwargs)
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
    return filename
