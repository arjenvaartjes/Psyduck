import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import qutip as qt
from qutip.wigner import spin_wigner, spin_q_function

def tomo_plot_3d(
    obj,
    kind="auto",             # "auto", "wigner", or "husimi"
    n_theta=70,
    n_phi=120,
    r=1.0,
    cmap="bwr",
    ax=None,
    add_projections=True,
    shift_deg=90,            # rotate the colormap around phi to center features
    lights_az=-45, lights_alt=10
):
    """
    Plot a spin quasi-probability (Wigner or Husimi/Q) on the Bloch sphere with three projections.

    Parameters
    ----------
    obj : qutip.Qobj
        State ket or density matrix.
    kind : {"auto","wigner","husimi"}
        - "auto": Wigner if obj.isket, Husimi if obj.isoper
        - "wigner": force spin Wigner
        - "husimi": force spin Husimi/Q
    n_theta, n_phi : int
        Angular grid resolution for theta (0..pi) and phi (-pi..pi).
    r : float
        Sphere radius.
    cmap : str or Colormap
        Matplotlib colormap.
    ax : matplotlib 3D axis or None
        If None, a new figure/axis is created and returned.
    add_projections : bool
        If True, draws orthogonal projections on x/y/z planes.
    shift_deg : float
        Amount of azimuthal roll (in degrees) applied to the data for nicer framing.
    lights_az, lights_alt : float
        Light source azimuth/altitude for subtle shading.

    Returns
    -------
    fig, ax : (matplotlib.figure.Figure, mpl_toolkits.mplot3d.Axes3D)
    """
    # --- choose representation ---
    if kind == "auto":
        if obj.isket:
            kind = "wigner"
        elif obj.isoper:
            kind = "husimi"
        else:
            # fallback: try wigner
            kind = "wigner"

    try:
        cmap = plt.get_cmap(cmap)
    except Exception:
        # if user passed a Colormap already
        pass

    # --- grids ---
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)

    if kind.lower() in ("husimi", "q", "qfunc", "q-function"):
        F, theta_mesh, phi_mesh = spin_q_function(obj, theta, phi)
        # Husimi >= 0 ⇒ use 0..max normalization
        vmin, vmax = 0.0, np.max(F) if np.max(F) > 0 else 1.0
    else:
        F, theta_mesh, phi_mesh = spin_wigner(obj, theta, phi)
        # symmetric normalization about zero
        m = np.max(np.abs(F))
        vmin, vmax = -m, m if m > 0 else (-1.0, 1.0)

    # roll along phi for a nicer "front" view (default 90°)
    shift_amount = int(np.rint((-shift_deg / 360.0) * n_phi))
    F = np.roll(F, shift_amount, axis=0)

    # mesh for plotting
    PHI, THETA = np.meshgrid(phi, theta)  # shapes (n_theta, n_phi), but we'll use theta_mesh/phi_mesh for correctness

    # sphere parameterization (match your original orientation)
    X = r * np.cos(phi_mesh) * np.cos(theta_mesh - np.pi / 2)
    Y = r * np.sin(phi_mesh) * np.cos(theta_mesh - np.pi / 2)
    Z = r * np.sin(theta_mesh - np.pi / 2)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    light_source = mpl.colors.LightSource(azdeg=lights_az, altdeg=lights_alt,
                                          hsv_min_val=1, hsv_max_val=1, hsv_min_sat=1, hsv_max_sat=1)

    # axis setup
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure

    # main sphere
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(F)), rstride=1, cstride=1,
                    shade=False, lightsource=light_source)

    if add_projections:
        # --- Y-projection (plane y = +1.5) : use phi second half
        F_y = F[n_phi // 2:, :]
        tm_y = theta_mesh[n_phi // 2:, :]
        pm_y = phi_mesh[n_phi // 2:, :]
        Xy = r * np.cos(pm_y) * np.cos(tm_y - np.pi / 2)
        Yy = np.zeros_like(Xy) + 1.5
        Zy = r * np.sin(tm_y - np.pi / 2)
        ax.plot_surface(Xy, Yy, Zy, facecolors=cmap(norm(F_y)), shade=False, rstride=1, cstride=1)

        # --- Z-projection (plane z = -1.5) : use theta first half
        F_z = F[:, :n_theta // 2]
        tm_z = theta_mesh[:, :n_theta // 2]
        pm_z = phi_mesh[:, :n_theta // 2]
        Xz = r * np.cos(pm_z) * np.cos(tm_z - np.pi / 2)
        Yz = r * np.sin(pm_z) * np.cos(tm_z - np.pi / 2)
        Zz = np.zeros_like(Xz) - 1.5
        ax.plot_surface(Xz, Yz, Zz, facecolors=cmap(norm(F_z)), shade=False, rstride=1, cstride=1)

        # --- X-projection (plane x = -1.5) : phi in [-pi/2, pi/2]
        sl = slice(n_phi // 4, 3 * n_phi // 4)
        F_x = F[sl, :]
        tm_x = theta_mesh[sl, :]
        pm_x = phi_mesh[sl, :]
        Xx = np.zeros_like(pm_x) - 1.5
        Yx = r * np.sin(pm_x) * np.cos(tm_x - np.pi / 2)
        Zx = r * np.sin(tm_x - np.pi / 2)
        ax.plot_surface(Xx, Yx, Zx, facecolors=cmap(norm(F_x)), shade=False, rstride=1, cstride=1)

        # light axis lines (same as your original)
        ax.plot([-1.5, -1.5], [-1.6,  1.6], [-1.5, -1.5], color=(0.75, 0.75, 0.75, 1), lw=0.5)
        ax.plot([-1.6,  1.6], [ 1.5,  1.5], [-1.5, -1.5], color=(0.75, 0.75, 0.75, 1), lw=0.5)
        ax.plot([-1.5, -1.5], [ 1.5,  1.5], [-1.6,  1.6], color=(0.75, 0.75, 0.75, 1), lw=0.5)

    # bounds & cosmetics
    lim = 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(0)

    ax.grid(False)
    ax.xaxis.set_pane_color((0.85, 0.85, 0.85, 1))
    ax.yaxis.set_pane_color((0.80, 0.80, 0.80, 1))
    ax.zaxis.set_pane_color((0.825, 0.825, 0.825, 1))

    # helpful title
    title_kind = "Husimi Q" if kind.lower().startswith("husimi") else "Wigner"
    ax.set_title(f"Spin {title_kind} function", pad=12)

    if created_fig:
        plt.tight_layout()

    return fig, ax

# --- convenience alias that mirrors your original naming (if you want it) ---
def plot_3d_spin_wigner_or_husimi(obj, **kwargs):
    """Wrapper that calls plot_spin_quasiprob with the same signature by name."""
    return plot_spin_quasiprob(obj, **kwargs)