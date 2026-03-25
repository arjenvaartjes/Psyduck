"""Plotting utilities for SpinSeries objects."""

import numpy as np
import matplotlib.pyplot as plt


def plot_populations(series, coord_label=None, levels=None):
    """Plot Zeeman level populations over a SpinSeries coordinate axis.

    Two-panel figure: left panel is a pcolor heatmap, right panel shows
    individual level traces with a legend.

    :param series: SpinSeries object
    :param coord_label: Label for both x-axes. Defaults to 'Time' if coords are set, else 'Index'.
    :param levels: List of level indices to include. Defaults to all levels.
    :return: (fig, axes) — axes is a length-2 array
    """
    pop = series.populations()                          # (N, dim)
    coords = series.coords if series.coords is not None else np.arange(len(series))
    xlabel = coord_label or ('Time' if series.coords is not None else 'Index')

    dim = series.dim
    state_labels = [f'$|{int(dim - 1 - 2*i)}/2\\rangle$' for i in range(dim)]

    if levels is None:
        levels = list(range(dim))

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), gridspec_kw={'wspace': 0.3})

    pcm = axes[0].pcolor(coords, np.arange(dim), pop[:, levels].T)
    plt.colorbar(pcm, ax=axes[0])
    axes[0].set_ylabel('State index')
    axes[0].set_xlabel(xlabel)

    for idx in levels:
        axes[1].plot(coords, pop[:, idx], label=state_labels[idx])
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Population')

    return fig, axes
