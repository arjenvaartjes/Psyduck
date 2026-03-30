"""SpinSeries: a sequence of spin states over a coordinate axis."""

from psyduck import SpinInterface
import numpy as np
import qutip as qt


class SpinSeries(SpinInterface):
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
    
    def fidelity(self, target_state: qt.Qobj) -> np.ndarray:
        return np.array([qt.fidelity(s, target_state) for s in self.states])

    def apply_operator(self, U: qt.Qobj):
        self.states = [U * state for state in self.states]

    def copy(self):
        return SpinSeries(self.states.copy(), self.I, coords=self.coords.copy(), result=self.result)

    def state_labels(self) -> list[str]:
        return [f'|{self.dim - 1 - 2 * i}/2>' for i in range(0, self.dim)]

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

    def plot_populations(self, coord_label=None, levels=None):
        """Plot Zeeman level populations over the coordinate axis.

        :param coord_label: x-axis label.
        :param levels: List of level indices to include. Defaults to all.
        :return: (fig, axes)
        """
        from psyduck.plotting.spin_series_plot import plot_populations
        return plot_populations(self, coord_label=coord_label, levels=levels)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx) -> "Spin":
        from psyduck.spin import Spin
        return Spin(self.I, self.states[idx])

    def __repr__(self) -> str:
        coord_info = f", coords={self.coords[0]:.3g}..{self.coords[-1]:.3g}" if self.coords is not None else ""
        return f"SpinSeries(I={self.I}, N={len(self.states)}{coord_info})"
