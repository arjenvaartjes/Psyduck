"""Spin class for quantum spin simulations using QuTiP."""

import qutip as qt
import numpy as np
from typing import Union, List
from numpy import ndarray
from .operations import parity_operator, global_rotation


class Spin:
    """Represents a quantum spin system.
    
    Attributes:
        I: Spin quantum number
        state: Current quantum state (Qobj)
    """
    
    def __init__(self, I: float = 7/2, state: qt.Qobj = None):
        """Initialize a Spin object.
        
        :param I: Spin quantum number (e.g., 1/2, 1, 3/2, 7/2)
        :param state: Initial quantum state. If None, initializes to ground state.
        """
        self.I = I
        self.dim = int(2 * I + 1)  # Hilbert space dimension
        
        if state is None:
            self.state = qt.basis(self.dim, 0)  # Ground state |I, -I>
        else:
            self.state = state
        self.dm = self.state * self.state.dag()  # Density matrix
    
    def expectation(self, operator: qt.Qobj) -> float:
        """Calculate expectation value of an operator.
        
        :param operator: Quantum operator (Qobj)
        :return: Expectation value <operator>
        """
        return qt.expect(operator, self.state)
    
    def Ix(self) -> float:
        """Calculate expectation value <Ix>.
        
        :return: Expectation value of Ix
        """
        Ix = qt.jmat(self.I, 'x')
        return self.expectation(Ix)
    
    def Iy(self) -> float:
        """Calculate expectation value <Iy>.
        
        :return: Expectation value of Iy
        """
        Iy = qt.jmat(self.I, 'y')
        return self.expectation(Iy)
    
    def Iz(self) -> float:
        """Calculate expectationvalue <Iz>.
        
        :return: Expectation value of Iz
        """
        Iz = qt.jmat(self.I, 'z')
        return self.expectation(Iz)
    
    def get_spin_operators(self) -> tuple:
        """Get the spin operators Ix, Iy, Iz as Qobj.
        
        :return: Tuple of (Ix, Iy, Iz)
        """
        Ix = qt.jmat(self.I, 'x')
        Iy = qt.jmat(self.I, 'y')
        Iz = qt.jmat(self.I, 'z')
        return Ix, Iy, Iz

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

    def parity(self) -> float:
        """Calculate the parity of the current state.
        
        Parity is defined as +1 for even m states and -1 for odd m states.
        
        :return: Parity expectation value
        """
        parity_op = parity_operator(self.I)
        return self.expectation(parity_op)
    
    def evolve(self, H: qt.Qobj, times: Union[float, List[float]], 
               c_ops: List[qt.Qobj] = None) -> qt.Result:
        """Evolve the spin under a Hamiltonian.
        
        Uses QuTiP's Schrödinger equation solver (sesolve).
        
        :param H: Hamiltonian (Qobj)
        :param times: Evolution time(s) - float for single time point, or list of times
        :param c_ops: Collapse operators for open system evolution (default None for closed system)
        :return: Result object with final state(s) and expectation values
        """
        # Convert single time to list
        if isinstance(times, (int, float)):
            times = [0, times]
        else:
            times = list(times)
        
        # Ensure times is sorted and starts from 0
        if times[0] != 0:
            times = [0] + sorted(times)
        times = sorted(set(times))  # Remove duplicates and sort
        
        # Run solver
        if c_ops is None or len(c_ops) == 0:
            # Closed system: sesolve
            result = qt.sesolve(H, self.state, times)
        else:
            # Open system: use mesolve with collapse operators
            result = qt.mesolve(H, self.state, times, c_ops)
        
        # Update internal state to final state
        self.state = result.states[-1]
        
        return result
    
    def apply_operator(self, U: qt.Qobj) -> None:
        """Apply a unitary operator to the current state.
        
        :param U: Unitary operator (Qobj)
        """
        self.state = U * self.state
    
    def copy(self) -> 'Spin':
        """Create a deep copy of this Spin object.
        
        :return: New Spin object with copied state
        """
        return Spin(self.I, self.state.copy())
    
    def __repr__(self) -> str:
        return f"Spin(I={self.I}, dim={self.dim})"
