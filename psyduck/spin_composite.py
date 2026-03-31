from psyduck import SpinInterface
import qutip as qt
import numpy as np
from typing import Union, List
from numpy import ndarray
from psyduck.operations import parity_operator, global_rotation, subspace_rotation, shift_operator
from psyduck.plotting import wigner_plot_3d, projection_plot_spin_wigner, wigner_plot_hammer, wigner_plot_polar
from psyduck.spin_series import SpinSeries


class SpinComposite(SpinInterface):
    """Composite quantum spin system built from multiple Spin sub-systems.

    Attributes:
        spins: List of constituent Spin objects
        I:     Convenience list of spin quantum numbers (derived from spins)
        state: Current composite quantum state (tensor-product Qobj)
    """

    def __init__(self, spins: list = None, state: qt.Qobj = None):
        """Initialize a composite spin system.

        :param spins: List of Spin objects. Defaults to [Spin(1/2), Spin(7/2)].
        :param state: Initial composite state. If None, uses the tensor product
                      of each spin's current state.
        """
        from psyduck.spin import Spin as _Spin
        if spins is None:
            spins = [_Spin(1/2), _Spin(7/2)]
        self.spins = spins
        self.I = [s.I for s in spins]   # convenience attribute

        if state is None:
            self.state = qt.tensor(*[s.state for s in spins])
        else:
            self.state = state
        self.dm = self.state * self.state.dag()

    def expectation(self, operator: qt.Qobj, state=0) -> complex | float:
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
        self.dm = U @ self.dm @ U.dag()

    def copy(self) -> 'SpinComposite':
        return SpinComposite(self.spins, self.state.copy())

    def __repr__(self) -> str:
        return f"SpinComposite(spins={self.spins})"

    def state_labels(self) -> list[str]:
        return [s.state_labels() for s in self.spins]

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _dims(self) -> list[int]:
        return [s.dim for s in self.spins]

    def _spin_values(self) -> list[float]:
        return [s.I for s in self.spins]

    def _embed(self, op: qt.Qobj, k: int) -> qt.Qobj:
        """Embed single-spin operator `op` for spin index k into the full space."""
        parts = [qt.qeye(d) for d in self._dims()]
        parts[k] = op
        return qt.tensor(*parts)

    # -----------------------------------------------------------------------
    # Overrides of SpinInterface methods that assume self.I is a scalar
    # -----------------------------------------------------------------------

    def Ix(self, spin_idx: int = None) -> float:
        if spin_idx is None:
            op = sum(self._embed(qt.jmat(s.I, 'x'), k) for k, s in enumerate(self.spins))
        else:
            op = self._embed(qt.jmat(self.spins[spin_idx].I, 'x'), spin_idx)
        return self.expectation(op)

    def Iy(self, spin_idx: int = None) -> float:
        if spin_idx is None:
            op = sum(self._embed(qt.jmat(s.I, 'y'), k) for k, s in enumerate(self.spins))
        else:
            op = self._embed(qt.jmat(self.spins[spin_idx].I, 'y'), spin_idx)
        return self.expectation(op)

    def Iz(self, spin_idx: int = None) -> float:
        if spin_idx is None:
            op = sum(self._embed(qt.jmat(s.I, 'z'), k) for k, s in enumerate(self.spins))
        else:
            op = self._embed(qt.jmat(self.spins[spin_idx].I, 'z'), spin_idx)
        return self.expectation(op)

    def parity(self) -> float:
        op = qt.tensor(*[parity_operator(s.I) for s in self.spins])
        return self.expectation(op)

    def global_rotate(self, angle: float, axis: Union[str, ndarray],
                      spin_idx: int = None) -> None:
        if spin_idx is None:
            U = qt.tensor(*[global_rotation(s.I, angle, axis) for s in self.spins])
        else:
            parts = [qt.qeye(s.dim) for s in self.spins]
            parts[spin_idx] = global_rotation(self.spins[spin_idx].I, angle, axis)
            U = qt.tensor(*parts)
        self.apply_operator(U)

    def subspace_rotate(self, angle: float, axis: Union[str, ndarray],
                        levels: Union[tuple, list, ndarray], spin_idx: int = 0) -> None:
        parts = [qt.qeye(s.dim) for s in self.spins]
        parts[spin_idx] = subspace_rotation(self.spins[spin_idx].I, angle, axis, levels)
        self.apply_operator(qt.tensor(*parts))

    def shift(self, spin_idx: int = 0) -> None:
        parts = [qt.qeye(s.dim) for s in self.spins]
        parts[spin_idx] = shift_operator(self.spins[spin_idx].I)
        self.apply_operator(qt.tensor(*parts))
