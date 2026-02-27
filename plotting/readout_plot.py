import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_transition_matrix(transition_matrix, ax, title=None):

    Z = np.abs(transition_matrix)
    X, Y = np.arange(8), np.arange(8)
    # Plot with pcolormesh using LogNorm
    Z[Z==0]=1e-7

    # Convert to percentage
    Z_percent = Z * 100  
    pcm = ax.pcolormesh(X, Y, Z, norm=mcolors.LogNorm(vmin=1e-7, vmax=1e-3), cmap="viridis")

    nucleus_labels = ["-7/2", "-5/2", "-3/2", "-1/2", "1/2", "3/2", "5/2", "7/2"]
    electron_labels = [r"$\downarrow$", r"$\uparrow$", r"$^0$"]

    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels([nucleus_labels[k%8] + electron_labels[(k//8)] for k in range(8)])
    ax.set_yticklabels([nucleus_labels[k%8] + electron_labels[(k//8)] for k in range(8)])

    # Annotate with percentages
    for i in range(8):
        for j in range(8):
            color = 'black' if i == j else 'white'  # Diagonal elements in black
            ax.text(j, i, f"{Z_percent[i, j]:.2f}%", 
                    ha='center', va='center', fontsize=8, color=color)

    ax.set_title(title)
    plt.colorbar(pcm, ax=ax, label='Abs')

def plot_eigenstate_decomposition(eigenstate_matrix, ax):
    
    Z = np.abs(eigenstate_matrix)**2
    X, Y = np.arange(24), np.arange(24)
    # Plot with pcolormesh using LogNorm
    Z[Z==0]=1e-7

    # Convert to percentage
    Z_percent = Z * 100  
    pcm = plt.pcolormesh(X, Y, Z, norm=mcolors.LogNorm(vmin=1e-7, vmax=1e-3), cmap="viridis")

    nucleus_labels = ["-7/2", "-5/2", "-3/2", "-1/2", "1/2", "3/2", "5/2", "7/2"]
    electron_labels = [r"$\downarrow$", r"$\uparrow$", r"$^0$"]

    ax.set_xticks(np.arange(24))
    ax.set_yticks(np.arange(24))
    ax.set_xticklabels([nucleus_labels[k%8] + electron_labels[(k//8)] for k in range(24)])
    ax.set_yticklabels([nucleus_labels[k%8] + electron_labels[(k//8)] for k in range(24)])

    # Annotate with percentages
    for i in range(24):
        for j in range(24):
            color = 'black' if i == j else 'white'  # Diagonal elements in black
            ax.text(j, i, f"{Z_percent[i, j]:.2f}%", 
                    ha='center', va='center', fontsize=8, color=color)

    ax.set_title('Nuclear Eigenstates of the Hamiltonian')

    plt.colorbar(label='Abs**2')