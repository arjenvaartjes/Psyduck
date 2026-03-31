"""Spin class for quantum spin simulations using QuTiP."""
from abc import ABC, abstractmethod
import qutip as qt
from typing import Union, List
from numpy import ndarray
from psyduck.operations import parity_operator, global_rotation, subspace_rotation, shift_operator


class SpinInterface(ABC):
    """Represents a generic quantum spin system.

    Spin objects should inherit this class and implement the required methods
    
    Attributes:
        I: Spin quantum number
        state: Current quantum state (Qobj)
    """

    @abstractmethod
    def expectation(self, operator: qt.Qobj) -> complex | float:
        """Calculate expectation value of an operator.
        
        :param operator: Quantum operator (Qobj)
        :return: Expectation value <operator>
        """
        ...

    @abstractmethod
    def apply_operator(self, U: qt.Qobj) -> None:
        """Apply a unitary operator to the current state.

        :param U: Unitary operator (Qobj)
        """
        ...

    @abstractmethod
    def copy(self):
        """Create a deep copy of this Spin object.

        :return: New Spin object with copied state
        """
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def state_labels(self) -> list[str]:
        """Create a set of state labels associated with this Spin object

        :return: Set of state labels"""
        ...


    # Common methods based on the abstract methods defined above
    # These will then work on any abstract Spin object!

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

    def parity(self) -> float:
        """Calculate the parity of the current state.
        
        Parity is defined as +1 for even m states and -1 for odd m states.
        
        :return: Parity expectation value
        """
        parity_op = parity_operator(self.I)
        return self.expectation(parity_op)
    
    def fidelity(self, target_state: qt.Qobj) -> float:
        """Calculate fidelity with respect to a target state.

        :param target_state: Target quantum state (Qobj)
        :return: Fidelity value between 0 and 1
        """
        return qt.fidelity(self.state, target_state)
    
    def global_rotate(self, angle: float, axis: Union[str, ndarray]) -> None:
        """Apply a global rotation to the spin state.

        :param angle: Rotation angle in radians
        :param axis: Rotation axis - 'x', 'y', 'z', or 3-element array
        """
        U = global_rotation(self.I, angle, axis)
        self.apply_operator(U)

    def subspace_rotate(self, angle: float, axis: Union[str, ndarray], levels: Union[tuple, list, ndarray]) -> None:
        """Apply a rotation in a two-level subspace (Givens rotation).

        :param angle: Rotation angle in radians
        :param axis: Rotation axis - 'x', 'y', 'z', or 3-element array
        :param levels: Tuple (m1, m2) of magnetic quantum numbers defining the subspace
        """
        U = subspace_rotation(self.I, angle, axis, levels)
        self.apply_operator(U)

    def shift(self):
        U = shift_operator(self.I)
        self.apply_operator(U)
    
