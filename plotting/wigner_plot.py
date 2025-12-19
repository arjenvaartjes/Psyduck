import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from qutip import *
from qutip.wigner import spin_q_function
from qutip import Qobj, spin_coherent
import cmath, math

def wigner_plot(rho,dpi=300):
    nTheta, nPhi = (101, 201)
    theta = np.linspace(0, np.pi, num=nTheta, endpoint=True)
    phi = np.linspace(-np.pi, np.pi, num=nPhi, endpoint=True)
    husimi0, theta_mesh, phi_mesh = spin_wigner(rho, theta, phi)

    r = 1
    x = r*np.cos(phi_mesh)*np.cos(theta_mesh-np.pi/2)
    y = r*np.sin(phi_mesh)*np.cos(theta_mesh-np.pi/2)
    z = r*np.sin(theta_mesh-np.pi/2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    cmap = mpl.cm.bwr
    norm = mpl.colors.Normalize(vmin=husimi0.min(), vmax=husimi0.max())
    ax.plot_surface(x, y, z, facecolors=cmap(norm(husimi0)), rstride=1, cstride=1, shade=False)

    # ax.plot_surface(x,y,z, rstride=1, cstride=1)

    ax.set_axis_off()
    s = 0.7 # plot_scale
    ax.set_xlim([-s,s])
    ax.set_ylim([-s,s])
    ax.set_zlim([-s,s])
    ax = fig.add_subplot(1, 2, 2, projection='hammer')
    ax.pcolormesh(phi_mesh, theta_mesh - np.pi/2, husimi0, cmap='bwr')
    ax.set_xticklabels([])

    ax.grid(False)
    
def projection_plot_spin_wigner(psi, n_theta=200, n_phi=200, r=1, cmap='RdBu', figsize=(8,6), ax=None, fig=None, vmin=None, vmax=None, prob_function='wigner', **kwargs):
    """Plots spin wigner function as a surface plot on a sphere, with three projections on the side panels
    Inputs: 
    psi: Qutip Qubj (state vector)
    n_theta: number of grid points for polar angles theta
    n_phi: number of grid points for azimuthal angles phi
    r: radius of the sphere
    cmap: matplotlib colormap
    
    Outputs:
    fig: matplotlib figure
    ax: matplotlib axis
    """
    
    try:
        cmap = plt.get_cmap(cmap)
    except:
        cmap = cmap

    theta = np.linspace(0,np.pi,n_theta)
    phi = np.linspace(0, 2*np.pi,n_phi)
    if prob_function == 'wigner':
        wigner, theta_mesh, phi_mesh = spin_wigner(psi, theta, phi)
    else:
        wigner, theta_mesh, phi_mesh = spin_q_function(psi, theta, phi)

    if vmin == None:
        vmin = wigner.min()
    if vmax == None:
        vmax = wigner.max()

    phi,theta = np.meshgrid(phi,theta)

    x = r*np.cos(phi_mesh)*np.cos(theta_mesh-np.pi/2)
    y = r*np.sin(phi_mesh)*np.cos(theta_mesh-np.pi/2)
    z = r*np.sin(theta_mesh-np.pi/2)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    if ax==None:
        fig = plt.figure(dpi=300, figsize=figsize, edgecolor='None')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    light_source = mpl.colors.LightSource(azdeg=-45, altdeg=10, hsv_min_val=1, hsv_max_val=1, hsv_min_sat=1, hsv_max_sat=1)
    ax.plot_surface(x, y, z, facecolors=cmap(norm(wigner)), rstride=1, cstride=1, shade=False, lightsource=light_source, **kwargs)

    # -- for y projection, take second half of phi values (first index) --
    theta_mesh_y = theta_mesh[:n_phi//2, :]
    phi_mesh_y = phi_mesh[:n_phi//2, :]
    wigner_y = wigner[:n_phi//2, :]
    
#     theta_mesh_y = theta_mesh[n_phi//4:3*n_phi//4, :]
#     phi_mesh_y = phi_mesh[n_phi//4:3*n_phi//4, :]
#     wigner_y = wigner[n_phi//4:3*n_phi//4, :]

    x = r*np.cos(phi_mesh_y)*np.cos(theta_mesh_y-np.pi/2)
    y = r*np.sin(phi_mesh_y)*np.cos(theta_mesh_y-np.pi/2)
    z = r*np.sin(theta_mesh_y-np.pi/2)

    ax.plot_surface(x, np.zeros_like(x)+1.5, z, facecolors = cmap(norm(wigner_y)), shade=False, rstride=1, cstride=1, **kwargs)

    # -- for z projection, take second half of theta values (second index) --
    theta_mesh_z = theta_mesh[:, :n_theta//2]
    phi_mesh_z = phi_mesh[:, :n_theta//2]
    wigner_z = wigner[:, :n_theta//2]

    x = r*np.cos(phi_mesh_z)*np.cos(theta_mesh_z-np.pi/2)
    y = r*np.sin(phi_mesh_z)*np.cos(theta_mesh_z-np.pi/2)
    z = r*np.sin(theta_mesh_z-np.pi/2)

    ax.plot_surface(x, y, np.zeros_like(x)-1.5, facecolors = cmap(norm(wigner_z)), shade=False, rstride=1, cstride=1, **kwargs)

    # -- for x projection, take phi values from -pi/2 to pi/2: 1/4 to 3/4 --

    theta_mesh_x = theta_mesh[n_phi//4:3*n_phi//4, :]
    phi_mesh_x = phi_mesh[n_phi//4:3*n_phi//4, :]
    wigner_x = wigner[n_phi//4:3*n_phi//4, :]

    x = r*np.cos(phi_mesh_x)*np.cos(theta_mesh_x-np.pi/2)
    y = r*np.sin(phi_mesh_x)*np.cos(theta_mesh_x-np.pi/2)
    z = r*np.sin(theta_mesh_x-np.pi/2)
    ax.plot_surface(np.zeros_like(y)-1.5, y, z, facecolors = cmap(norm(wigner_x)), shade=False, rstride=1, cstride=1, **kwargs)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5,1.5)
    
    # # -- axis lines --
    ax.plot([-1.5, -1.5], [-1.6, 1.6], [-1.5, -1.5], color='grey', lw=0.4)
    ax.plot([-1.6, 1.6], [1.5, 1.5], [-1.5, -1.5], color='grey', lw=0.4)
    ax.plot([-1.5, -1.5], [1.5, 1.5], [-1.6, 1.6], color='grey', lw=0.4)

    # make axes invisible
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(0)
        
#     plot_box()
#     ax.set_xlabel('x')

#     ax.grid(False)
    ax.w_xaxis.set_pane_color((0.85, 0.85, 0.85, 1))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1))
    ax.w_zaxis.set_pane_color((0.825, 0.825, 0.825, 1))
    ax.view_init(22, -60)

    return fig, ax, wigner, theta_mesh, phi_mesh