import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from psyduck.hamiltonians import get_quadrupole_stark_shift, get_quadrupole_splittings


def _fq_grid(V_ab, I, B0, gamma, Q, n_theta, n_phi):
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    fq1, fq2 = get_quadrupole_splittings(V_ab, I, B0, gamma, Q, thetas, phis)
    return thetas, phis, fq1, fq2


def _surface_3d(ax, r, TH, PH, thetas, phis, cmap_name, sym=False):
    X = np.abs(r) * np.sin(TH) * np.cos(PH)
    Y = np.abs(r) * np.sin(TH) * np.sin(PH)
    Z = np.abs(r) * np.cos(TH)

    cmap = plt.get_cmap(cmap_name)
    if sym:
        vmax = np.nanmax(np.abs(r))
        norm = Normalize(vmin=-vmax, vmax=vmax)
    else:
        norm = Normalize(vmin=np.nanmin(r), vmax=np.nanmax(r))
    facecolors = cmap(norm(r))
    facecolors[..., 3] *= 0.3
    ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)

    for phi_val in [0.0, np.pi]:
        j = int(np.argmin(np.abs(phis - phi_val)))
        r_slice = r[:, j]
        for mask, color in [(r_slice >= 0, 'tab:blue'), (r_slice < 0, 'tab:red')]:
            if mask.any():
                th = thetas[mask]
                rv = np.abs(r_slice[mask])
                ax.plot(rv * np.sin(th) * np.cos(phi_val),
                        rv * np.sin(th) * np.sin(phi_val),
                        rv * np.cos(th), color=color, lw=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor('white')
    return cm.ScalarMappable(norm=norm, cmap=cmap)


def plot_quadrupole_stark_shift(dfq1, dfq2, thetas, phis):
    """Visualize quadrupole Stark shift sensitivity d(fq)/dE over all field orientations.

    Takes pre-computed sensitivity grids (from get_quadrupole_stark_shift) and
    renders side and top 3D views for fq1 and fq2. The colormap is symmetric
    about zero so white regions immediately identify sweet spots.

    :param dfq1: d(fq1)/dE grid, shape (len(thetas), len(phis)), units Hz/(V/m).
    :param dfq2: d(fq2)/dE grid, shape (len(thetas), len(phis)), units Hz/(V/m).
    :param thetas: 1-D array of polar angles (rad).
    :param phis: 1-D array of azimuthal angles (rad).
    """
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')

    with plt.style.context('default'):
        fig = plt.figure(figsize=(12, 6), dpi=150)
        fig.subplots_adjust(top=0.88, bottom=0.05, left=0.02, right=0.98,
                            hspace=0.4, wspace=-0.6)

        configs = [
            (dfq1 * 1e-3, 'RdBu', r'$\partial f_{\rm q1}/\partial E$ (kHz·m/V)', 1),
            (dfq2,        'RdBu', r'$\partial f_{\rm q2}/\partial E$ (Hz·m/V)',   3),
        ]

        pending_colorbars = []
        for r, cmap_name, label, base_idx in configs:
            ax_side = fig.add_subplot(2, 2, base_idx, projection='3d')
            ax_top = fig.add_subplot(2, 2, base_idx + 1, projection='3d')

            sm = _surface_3d(ax_side, r, TH, PH, thetas, phis, cmap_name, sym=True)
            _surface_3d(ax_top, r, TH, PH, thetas, phis, cmap_name, sym=True)

            ax_side.view_init(elev=30, azim=-70)
            ax_top.view_init(elev=90, azim=-90)
            plt.setp(ax_top.get_zticklabels(), visible=False)
            ax_top.set_zlabel('')

            pending_colorbars.append((ax_side, ax_top, sm, label))

        for ax_l, ax_r, sm, label in pending_colorbars:
            pos_l = ax_l.get_position()
            pos_r = ax_r.get_position()
            span_x = pos_r.x1 - pos_l.x0
            cb_w = span_x * 0.45
            cb_x = pos_l.x0 + (span_x - cb_w) / 2
            cb_y = max(pos_l.y1, pos_r.y1) + 0.01
            cax = fig.add_axes([cb_x, cb_y, cb_w, 0.015])
            cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
            cb.set_label(label, labelpad=4)
            cb.ax.xaxis.set_label_position('top')
            cb.ax.xaxis.set_ticks_position('top')

        fig.patch.set_facecolor('white')

    plt.show()


def plot_quadrupole_tensor(V_ab, I, B0, gamma, Q, n_theta=50, n_phi=100):
    """Visualize the EFG tensor as 3D color plots of fq1 and fq2.

    Plots the first- and second-order quadrupole splittings (fq1, fq2) as a
    function of magnetic field orientation, using side and top 3D views.

    :param V_ab: EFG tensor in SI units (V/m²), shape (3, 3).
    :param I: Nuclear spin quantum number.
    :param B0: Static magnetic field (T).
    :param gamma: Nuclear gyromagnetic ratio (Hz/T).
    :param Q: Nuclear quadrupole moment (C·m²).
    :param n_theta: Number of polar angle grid points (default 50).
    :param n_phi: Number of azimuthal angle grid points (default 100).
    :return: matplotlib Figure.
    """
    thetas, phis, fq1, fq2 = _fq_grid(V_ab, I, B0, gamma, Q, n_theta, n_phi)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')

    with plt.style.context('default'):
        fig = plt.figure(figsize=(12, 6), dpi=150)
        fig.subplots_adjust(top=0.88, bottom=0.05, left=0.02, right=0.98,
                            hspace=0.4, wspace=-0.6)

        configs = [
            (fq1 * 1e-3, 'RdBu', r'$f_{\rm q1}$ (kHz)', 1),
            (fq2, 'PuOr', r'$f_{\rm q2}$ (Hz)', 3),
        ]

        pending_colorbars = []
        for r, cmap_name, label, base_idx in configs:
            ax_side = fig.add_subplot(2, 2, base_idx, projection='3d')
            ax_top = fig.add_subplot(2, 2, base_idx + 1, projection='3d')

            sm = _surface_3d(ax_side, r, TH, PH, thetas, phis, cmap_name)
            _surface_3d(ax_top, r, TH, PH, thetas, phis, cmap_name)

            ax_side.view_init(elev=30, azim=-70)
            ax_top.view_init(elev=90, azim=-90)
            plt.setp(ax_top.get_zticklabels(), visible=False)
            ax_top.set_zlabel('')

            pending_colorbars.append((ax_side, ax_top, sm, label))

        # Place colorbars using actual axis positions
        for ax_l, ax_r, sm, label in pending_colorbars:
            pos_l = ax_l.get_position()
            pos_r = ax_r.get_position()
            span_x = pos_r.x1 - pos_l.x0
            cb_w = span_x * 0.45
            cb_x = pos_l.x0 + (span_x - cb_w) / 2
            cb_y = max(pos_l.y1, pos_r.y1) + 0.01
            cax = fig.add_axes([cb_x, cb_y, cb_w, 0.015])
            cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
            cb.set_label(label, labelpad=4)
            cb.ax.xaxis.set_label_position('top')
            cb.ax.xaxis.set_ticks_position('top')

        fig.patch.set_facecolor('white')

    plt.show()
