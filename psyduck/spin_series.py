"""SpinSeries: a sequence of spin states over a coordinate axis."""

import numpy as np
import qutip as qt
from psyduck.operations import parity_operator


class SpinSeries:
    """An ordered sequence of spin states, optionally labeled by a coordinate array.

    The coordinate axis can be time, angle, field, or any swept parameter.

    Attributes:
        I: Spin quantum number
        states: List of QuTiP Qobj states
        coords: 1-D numpy array labeling each state (e.g. times), or None
        result: Original qt.Result object if constructed from one, else None
    """

    def __init__(self, states: list, I: float, coords=None, result=None):
        self.I = I
        self.dim = int(2 * I + 1)
        self.states = list(states)
        self.coords = np.asarray(coords) if coords is not None else None
        self.result = result

    @classmethod
    def from_result(cls, result: qt.Result, I: float, coords=None) -> "SpinSeries":
        return cls(result.states, I, coords=coords, result=result)

    def expectation(self, operator: qt.Qobj) -> np.ndarray:
        return np.array([qt.expect(operator, s) for s in self.states])

    def Ix(self) -> np.ndarray:
        return self.expectation(qt.jmat(self.I, 'x'))

    def Iy(self) -> np.ndarray:
        return self.expectation(qt.jmat(self.I, 'y'))

    def Iz(self) -> np.ndarray:
        return self.expectation(qt.jmat(self.I, 'z'))

    def parity(self) -> np.ndarray:
        return self.expectation(parity_operator(self.I))

    def populations(self) -> np.ndarray:
        """Return state populations as an array of shape (N, dim).

        For kets: |<m|psi>|^2. For density matrices: diagonal elements.
        """
        out = np.zeros((len(self.states), self.dim))
        for i, s in enumerate(self.states):
            if s.type == 'ket':
                out[i] = np.abs(s.full().flatten()) ** 2
            else:
                out[i] = np.real(s.diag())
        return out

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx) -> "Spin":
        from psyduck.spin import Spin
        return Spin(self.I, self.states[idx])

    def __repr__(self) -> str:
        coord_info = f", coords={self.coords[0]:.3g}..{self.coords[-1]:.3g}" if self.coords is not None else ""
        return f"SpinSeries(I={self.I}, N={len(self.states)}{coord_info})"
