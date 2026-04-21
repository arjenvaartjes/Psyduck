"""Spin class for quantum spin simulations using QuTiP."""

from psyduck import SpinInterface
import qutip as qt
import numpy as np
from typing import Union, List
from numpy import ndarray
from psyduck.operations import parity_operator, global_rotation, subspace_rotation, shift_operator
from psyduck.plotting import wigner_plot_3d, projection_plot_spin_wigner, wigner_plot_hammer, wigner_plot_polar
from psyduck.spin_series import SpinSeries


class Spin(SpinInterface):
    """Represents a single quantum spin system.

    Attributes:
        I: Spin quantum number
        state: Current quantum state (Qobj)
    """

    def __init__(self, I: float = 7 / 2, state: qt.Qobj = None):
        """Initialize a Spin object.

        :param I: Spin quantum number (e.g., 1/2, 1, 3/2, 7/2)
        :param state: Initial quantum state. If None, initializes to ground state |7/2>.
        """
        self.I = I
        self.dim = int(2 * I + 1)  # Hilbert space dimension

        if state is None:
            self.state = qt.basis(self.dim, 0)  # Ground state |I, -I>
        else:
            self.state = state

    def expectation(self, operator: qt.Qobj) -> complex | float:
        """Calculate expectation value of an operator.

        :param operator: Quantum operator (Qobj)
        :return: Expectation value <operator>
        """
        return qt.expect(operator, self.state)

    def apply_operator(self, U: qt.Qobj) -> None:
        """Apply a unitary operator to the current state.

        :param U: Unitary operator (Qobj)
        """
        self.state = U * self.state

    def state_labels(self):
        return [f'|{self.dim - 1 - 2 * i}/2>' for i in range(0, self.dim)]

    def copy(self) -> 'Spin':
        """Create a deep copy of this Spin object.

        :return: New Spin object with copied state
        """
        return Spin(self.I, self.state.copy())

    def __repr__(self) -> str:
        return f"Spin(I={self.I}, dim={self.dim})"

    # ============================================================================
    # Useful methods for a single spin system
    # ============================================================================

    def get_spin_operators(self) -> tuple:
        """Get the spin operators Ix, Iy, Iz as Qobj.

        :return: Tuple of (Ix, Iy, Iz)
        """
        Ix = qt.jmat(self.I, 'x')
        Iy = qt.jmat(self.I, 'y')
        Iz = qt.jmat(self.I, 'z')
        return Ix, Iy, Iz

    def evolve(self, H: qt.Qobj, times: Union[float, List[float]],
               c_ops: List[qt.Qobj] = None) -> SpinSeries:
        """Evolve the spin under a Hamiltonian.

        Uses QuTiP's sesolve (closed) or mesolve (open system).

        :param H: Hamiltonian (Qobj)
        :param times: Evolution time(s) - float for single time point, or list of times
        :param c_ops: Collapse operators for open system evolution (default None for closed system)
        :return: SpinSeries over the requested times
        """
        if isinstance(times, (int, float)):
            times = [0, times]
        else:
            times = list(times)

        if times[0] != 0:
            times = [0] + sorted(times)
        times = sorted(set(times))

        if c_ops is None or len(c_ops) == 0:
            result = qt.sesolve(H, self.state, times)
        else:
            result = qt.mesolve(H, self.state, times, c_ops)

        self.state = result.states[-1]

        return SpinSeries.from_result(result, self.I, coords=times)

    def plot_wigner(self, projection: str = '3d', **kwargs):
        """Plot the Wigner function of the current state.

        :param projection: '3d', 'hammer', or 'polar' (default '3d')
        :param kwargs: passed to the underlying plot function
        :return: (fig, ax)
        """

        _dispatch = {
            '3d': wigner_plot_3d,
            '3d_projection': projection_plot_spin_wigner,
            'hammer': wigner_plot_hammer,
            'polar': wigner_plot_polar,
        }
        if projection not in _dispatch:
            raise ValueError(f"projection must be '3d', 'hammer', or 'polar', got {projection!r}")
        return _dispatch[projection](self.state, **kwargs)
    
    def displace(self, theta: float, phi: float) -> None:
        """
        Apply displacement operator to the current state.
        
        Implements: D(theta, phi) = exp(theta/2 * (e^{i*phi} * J_- - e^{-i*phi} * J_+))
        
        Parameters
        ----------
        theta : float
            Displacement angle (colatitude).
        phi : float
            Azimuthal angle.
        """
        Ip = qt.jmat(self.I, '+')
        Im = qt.jmat(self.I, '-')
        D = (theta / 2 * (np.exp(1j * phi) * Im - np.exp(-1j * phi) * Ip)).expm()
        self.state = D * self.state
    # ============================================================================
    # State initialization methods
    # ============================================================================

    def make_eigenstate(self, eigenvalue):
        self.state = qt.basis(self.dim, int(-eigenvalue + self.I))

    def make_zcat_state(self, phi: float) -> None:
        """Prepare a cat state of the form (|I, -I> + e^(i*phi) |I, I>)/sqrt(2).

        :param phi: Relative phase angle in radians
        """
        d = int(2 * self.I + 1)
        psi_cat = (qt.basis(d, 0) + np.exp(1j * phi) * qt.basis(d, d - 1)).unit()
        self.state = psi_cat

    def make_xcat_state(self, phi: float) -> None:
        """Prepare a cat state of the form (|I, -I> + e^(i*phi) |I, I>)/sqrt(2) rotated to x-axis.

        :param phi: Relative phase angle in radians
        """
        d = int(2 * self.I + 1)
        psi_cat_z = (qt.basis(d, 0) + np.exp(1j * phi) * qt.basis(d, d - 1)).unit()
        # Rotate to x-axis using a pi/2 rotation around y-axis
        R_y = global_rotation(self.I, np.pi / 2, 'y')
        self.state = R_y * psi_cat_z


    def make_displaced_coherent_state(self, theta: float, phi: float) -> None:
        """
        Create a spin-coherent state with displacement.
        
        Initializes the state to |I, -I> (ground state) and applies a
        displacement operator to create a spin-coherent state.
        
        Parameters
        ----------
        theta : float
            Colatitude angle.
        phi : float
            Azimuthal angle.
        """
        # Initialize to ground state |I, -I>
        self.state = qt.basis(self.dim, 0)
        # Apply displacement
        self.displace(theta, phi)