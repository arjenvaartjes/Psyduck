from psyduck import SpinInterface
import qutip as qt
import numpy as np
from typing import Union, List
from numpy import ndarray
from psyduck.operations import parity_operator, global_rotation, subspace_rotation, shift_operator
from psyduck.plotting import wigner_plot_3d, projection_plot_spin_wigner, wigner_plot_hammer, wigner_plot_polar
from psyduck.spin_series import SpinSeries


class SpinComposite(SpinInterface):
    """Represents multiple quantum spin system.

    Attributes:
        I: Spin quantum numbers
        state: Current quantum state (Qobj)
    """

    def __init__(self, I: list[float] = None, state: qt.Qobj = None):
        """Initialize a Spin object.

        :param I: Spin quantum number (e.g., 1/2, 1, 3/2, 7/2)
        :param state: Initial quantum state. If None, initializes to ground state |7/2>.
        """
        self.I = I if I is not None else [1/2, 7/2]

        if state is None:
            self.state = qt.tensor(qt.basis(2, 0), qt.basis(8, 0))  # Ground state |I, -I>
        else:
            self.state = state
        self.dm = self.state * self.state.dag()  # Density matrix

    def expectation(self, operator: qt.Qobj, state=0) -> complex | float:
        """Calculate expectation value of an operator.

        :param operator: Quantum operator (Qobj)
        :return: Expectation value <operator>
        """
        return qt.expect(operator, self.state)
