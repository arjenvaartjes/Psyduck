import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import qutip as qt
from qutip.wigner import spin_wigner

from .wigner_plot import wigner_plot_hammer


def _populations(state: qt.Qobj) -> np.ndarray:
    """Return diagonal populations for a ket or density matrix."""
    if state.isket:
        return np.abs(state.full().flatten()) ** 2
    return np.real(np.diag(state.full()))


def plot_wigner_and_populations(
    state: qt.Qobj,
    title: str = None,
    cmap=None,
    fig: plt.Figure = None,
) -> tuple[plt.Figure, list]:
    """Plot the Wigner function alongside z- and x-basis population bars.

    Layout:
      - Top-left (6/7 width):  Wigner function (hammer projection, Iz-frame rotated)
      - Top-right (1/7 width): horizontal bar chart of z-basis populations
      - Bottom-left (6/7 width): vertical bar chart of x-basis populations

    :param state: Quantum state (ket or density matrix)
    :param title: Optional title for the Wigner panel
    :param cmap: Matplotlib colormap for the Wigner plot (default: coolwarm)
    :param fig: Optional existing figure to draw into
    :return: (fig, axes) tuple
    """
    if cmap is None:
        cmap = plt.get_cmap('coolwarm')

    d = state.shape[0]
    I = (d - 1) / 2

    # Rotate by 90° around Iz to align the Wigner plot with convention
    U_frame = (-1j * np.pi / 2 * qt.jmat(I, 'z')).expm()
    if state.isket:
        rotated = U_frame * state
    else:
        rotated = U_frame * state * U_frame.dag()

    if fig is None:
        fig = plt.figure(figsize=(5, 3))
    mpl.rcParams['axes.linewidth'] = 1.5
    gs = mpl.gridspec.GridSpec(4, 7, figure=fig)

    xx, yy = 6, 3

    # Wigner (hammer projection) — symmetric color scale around zero
    n_theta, n_phi = 101, 201
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    W, _, _ = spin_wigner(rotated, theta, phi)
    w_max = np.abs(W).max()

    ax_wigner = fig.add_subplot(gs[0:yy, 0:xx], projection='hammer')
    wigner_plot_hammer(rotated, fig=fig, ax=ax_wigner, cmap=cmap,
                       vmin=-w_max, vmax=w_max)
    if title is not None:
        ax_wigner.set_title(title, fontsize=14)
    ax_wigner.scatter(
        [0, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, np.pi, -np.pi],
        [0, 0, 0, np.pi/2, -np.pi/2, 0, 0],
        color='white', s=30, marker='.', zorder=5,
    )
    ax_wigner.set_xticks([])
    ax_wigner.set_yticks([])

    # z-basis populations (horizontal bars)
    ax_z = fig.add_subplot(gs[0:yy, xx:])
    pops_z = _populations(state)
    ax_z.barh(np.arange(d), pops_z, color='tab:blue')
    ax_z.set_yticks([])
    ax_z.set_xticks([])

    # x-basis populations (vertical bars)
    U_x = (-1j * np.pi / 2 * qt.jmat(I, 'y')).expm()
    if state.isket:
        state_x = U_x * state
    else:
        state_x = U_x * state * U_x.dag()
    pops_x = _populations(state_x)

    ax_x = fig.add_subplot(gs[yy:, 0:xx])
    ax_x.bar(np.arange(d), pops_x, color='tab:blue')
    ax_x.set_yticks([])
    ax_x.set_xticks([])

    fig.tight_layout()
    return fig, [ax_wigner, ax_z, ax_x]
