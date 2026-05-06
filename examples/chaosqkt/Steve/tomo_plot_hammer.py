import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.wigner import spin_q_function

def tomo_plot_hammer(rho, tomo, dpi=300):

    # Define grid sizes
    nTheta, nPhi = (101, 201)
    theta = np.linspace(0, np.pi, num=nTheta, endpoint=True)
    phi = np.linspace(-np.pi, np.pi, num=nPhi, endpoint=True)

    # Compute your tomography data
    if tomo == 'w':
        husimi0, theta_mesh, phi_mesh = spin_wigner(rho, theta, phi)
    if tomo == 'h':
        husimi0, theta_mesh, phi_mesh = spin_q_function(rho, theta, phi)

    # Create a figure and add a subplot with the Hammer projection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='hammer')  # Define 'ax' here

    # Optionally, remove the axis
    ax.set_axis_off()

    # # Set plot scale limits (for 2D plots, only x and y limits are used)
    # s = 0.7  # plot_scale
    # ax.set_xlim([-s, s])
    # ax.set_ylim([-s, s])

    # Plot your data using pcolormesh
    pc = ax.pcolormesh(phi_mesh, theta_mesh - np.pi/2, husimi0, cmap='bwr')
    ax.set_xticklabels([])

    # Remove grid lines if desired
    ax.grid(False)

    plt.show()

    return ax