import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms


def plot_transition_matrix(transition_matrix, electron_states=1, ax=None, title=None) -> plt.Axes:
    """Plot a transition matrix as a colour-mapped grid with percentage annotations.

    The matrix rows/columns are labelled by nuclear spin projection (m_I) and,
    optionally, electron spin state.  Absolute values are plotted on a log scale;
    each cell is annotated with its value as a percentage of the matrix maximum.

    Parameters
    ----------
    transition_matrix : array-like, shape (n, n)
        Square matrix whose absolute values are displayed.
    electron_states : int, optional
        Number of electron charge/spin states in the basis.  Default is 3
        (spin-down, spin-up, ionized donor).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure and axes are created if not provided.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    Z = np.abs(transition_matrix).copy()
    dim_N = Z.shape[0] // electron_states
    n = Z.shape[0]

    Z[Z == 0] = 1e-7
    Z_percent = Z / np.max(Z) * 100

    pcm = ax.pcolormesh(np.arange(n), np.arange(n), Z, norm=mcolors.LogNorm(), cmap="viridis")

    nucleus_labels = [f"{2*i - (dim_N - 1)}/2" for i in range(dim_N)]
    electron_labels = [r"$\downarrow$", r"$\uparrow$", r"$0$"]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    if electron_states > 1:
        ax.set_xticklabels([nucleus_labels[k % dim_N] + electron_labels[k // dim_N] for k in range(n)])
        ax.set_yticklabels([nucleus_labels[k % dim_N] + electron_labels[k // dim_N] for k in range(n)])
    else:
        ax.set_xticklabels([nucleus_labels[k % dim_N] for k in range(n)])
        ax.set_yticklabels([nucleus_labels[k % dim_N] for k in range(n)])

    for i in range(n):
        for j in range(n):
            color = 'black' if i == j else 'white'
            ax.text(j, i, f"{Z_percent[i, j]:.2f}%",
                    ha='center', va='center', fontsize=8, color=color)

    ax.set_title(title)
    plt.colorbar(pcm, ax=ax, label='Abs')
    return ax

def plot_transition_matrix_simplified(eigenstate_matrix, electron_states=3, ax=None) -> plt.Axes:
    """Plot the decomposition of eigenstates in the nuclear-spin ⊗ electron basis. No labels

    Parameters
    ----------
    eigenstate_matrix : array-like, shape (n, n)
        Matrix of eigenvectors as columns (as returned by ``numpy.linalg.eigh``
        or QuTiP).
    electron_states : int, optional
        Number of electron charge/spin states in the basis.  Default is 3
        (spin-down, spin-up, ionized donor).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure and axes are created if not provided.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.get_figure().subplots_adjust(left=0.18, bottom=0.18)

    Z = np.abs(eigenstate_matrix).copy()
    n = Z.shape[0]
    dim_N = n // electron_states

    Z[Z == 0] = 1e-7

    pcm = ax.pcolormesh(np.arange(n), np.arange(n), Z,
                        norm=mcolors.LogNorm(), cmap="viridis")

    electron_labels = [r"$\downarrow$", r"$\uparrow$", r"$0$"]

    ax.set_xticks([])
    ax.set_yticks([])

    # Block boundary separators
    for i in range(1, electron_states):
        ax.axvline(x=i * dim_N - 0.5, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.axhline(y=i * dim_N - 0.5, color='white', linewidth=1.5, linestyle='--', alpha=0.7)

    # Bracket labels for each electron state block
    bracket_offset = 0.04   # axes fraction offset from axis edge
    label_offset = 0.10     # axes fraction position of text

    x_trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    y_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    for e in range(min(electron_states, len(electron_labels))):
        center = e * dim_N + (dim_N - 1) / 2
        start = e * dim_N - 0.5
        end = (e + 1) * dim_N - 0.5

        # X-axis: bracket below the plot
        ax.annotate('',
                    xy=(end, -bracket_offset), xycoords=x_trans,
                    xytext=(start, -bracket_offset), textcoords=x_trans,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle='|-|', color='w', lw=1.0, mutation_scale=5))
        ax.text(center, -label_offset, electron_labels[e],
                transform=x_trans, ha='center', va='top', fontsize=13, clip_on=False)

        # Y-axis: bracket to the left of the plot
        ax.annotate('',
                    xy=(-bracket_offset, end), xycoords=y_trans,
                    xytext=(-bracket_offset, start), textcoords=y_trans,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle='|-|', color='w', lw=1.0, mutation_scale=5))
        ax.text(-label_offset, center, electron_labels[e],
                transform=y_trans, ha='right', va='center', fontsize=13, clip_on=False)

    ax.set_title('Nuclear Eigenstates of the Hamiltonian')
    plt.colorbar(pcm, ax=ax, label='|amplitude|')
    return ax
