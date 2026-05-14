"""CatQubit — high-spin cat-qubit encoded in an I=7/2 (or general half-integer) nuclear spin."""
from __future__ import annotations

import numpy as np
import qutip as qt

from psyduck import SpinInterface


def _dm_to_ket(dm: qt.Qobj) -> qt.Qobj:
    """Extract the dominant eigenvector from a (near-)pure density matrix."""
    evals, evecs = dm.eigenstates()
    return evecs[-1]  # eigenvector with largest eigenvalue


class CatQubit(SpinInterface):
    """Cat-qubit encoded in a high-spin nucleus.

    The logical qubit is encoded in the extreme eigenstates of the chosen spin
    axis ('Ix', 'Iy', or 'Iz').  The code space is spanned by the two states
    with maximum ±projection.

    Inherits Ix/Iy/Iz expectation values, parity, fidelity, global_rotate,
    subspace_rotate, and shift from SpinInterface.  State and dm are always
    kept in sync via the SpinInterface property pair.

    Attributes:
        I:      Spin quantum number (default 7/2).
        axis:   Encoding axis — 'Ix', 'Iy', or 'Iz'.
        dim:    Hilbert-space dimension = 2I+1.
        spin_op: Dict of cat-qubit-specific operators (logical Paulis, etc.).
    """

    def __init__(self, I: float = 7 / 2, axis: str = 'Ix'):
        assert (I * 2) % 2 == 1, "Spin must be half-integer"
        self.I = I
        self.dim = int(2 * I + 1)
        self.axis = axis

        self._build_spin_ops()

        self.Hamiltonian_params = {
            'fz': 7.7e6,
            'fq': 25e3,
            'fq2nd': 0,
            'B1amp': 1.0,
            'NER1amp': 1.0,
            'NER2amp': 1.0,
        }

        # State — ket, not density matrix.  dm is kept in sync automatically.
        self.state = qt.basis(self.dim, 0)
        self.state_list: list[qt.Qobj] = []

    # -----------------------------------------------------------------------
    # SpinInterface abstract methods
    # -----------------------------------------------------------------------

    def expectation(self, operator: qt.Qobj) -> complex | float:
        return qt.expect(operator, self.state)

    def apply_operator(self, U: qt.Qobj) -> "CatQubit":
        self.state = U * self.state
        return self

    def copy(self) -> CatQubit:
        new = CatQubit(self.I, self.axis)
        new.state = self.state.copy()
        new.spin_op = {k: v.copy() for k, v in self.spin_op.items()}
        return new

    def __repr__(self) -> str:
        return f"CatQubit(I={self.I}, axis={self.axis!r})"

    def state_labels(self) -> list[str]:
        return [f'|{self.dim - 1 - 2*i}/2⟩' for i in range(self.dim)]

    # -----------------------------------------------------------------------
    # Operator dictionary
    # -----------------------------------------------------------------------

    def _build_spin_ops(self) -> None:
        d, I = self.dim, self.I
        Ip = qt.jmat(I, '+')
        Im = qt.jmat(I, '-')
        self.spin_op: dict[str, qt.Qobj] = {
            'Ix':    qt.jmat(I, 'x'),
            'Iy':    qt.jmat(I, 'y'),
            'Iz':    qt.jmat(I, 'z'),
            'Ix^2':  qt.jmat(I, 'x') ** 2,
            'Iy^2':  qt.jmat(I, 'y') ** 2,
            'Iz^2':  qt.jmat(I, 'z') ** 2,
            'Ip':    Ip,
            'Im':    Im,
            'I2AT':  Ip ** 2 + Im ** 2,
            # Parity: +1 for even-indexed, -1 for odd-indexed basis states
            'parity': qt.Qobj(np.diag([1 if i % 2 == 0 else -1 for i in range(d)])),
            # Logical Paulis in the Iz basis (before encoding)
            'Lx_z':  qt.Qobj(np.flip(np.eye(d), 1)),
            'Ly_z':  qt.Qobj(np.flip(
                         np.eye(d, dtype=complex)
                         * np.array([*[-1j] * (d // 2), *[1j] * (d // 2)]), 1)),
            'Lz_z':  qt.Qobj(np.diag(
                         np.array([*[1] * (d // 2), *[-1] * (d // 2)], dtype=complex))),
        }

    def save_state(self) -> None:
        self.state_list.append(self.state.copy())

    def clear_state_list(self) -> None:
        self.state_list = []

    # -----------------------------------------------------------------------
    # Initialisation / encoding
    # -----------------------------------------------------------------------

    def initialize(self, mz: int = 0) -> None:
        """Reset to basis state |mz⟩."""
        self.state = qt.basis(self.dim, mz)

    def encode(self, sign: int = 1) -> None:
        """Rotate from the Iz basis into the encoding axis basis.

        Also builds the logical Pauli operators Lx/Ly/Lz in the encoded frame.
        """
        if self.axis == 'Iz':
            # No rotation needed; logical operators are the Iz-basis ones
            self.spin_op['Lx'] = self.spin_op['Lx_z']
            self.spin_op['Ly'] = self.spin_op['Ly_z']
            self.spin_op['Lz'] = self.spin_op['Lz_z']
            return

        orthogonal = 'Iy' if self.axis == 'Ix' else 'Ix'
        U = (sign * 1j * (np.pi / 2) * self.spin_op[orthogonal]).expm()
        self.apply_operator(U)

        # Rotate logical operators into the encoded frame
        Udag = U.dag()
        for key in ('Lx_z', 'Ly_z', 'Lz_z'):
            out_key = key[:-2]  # strip '_z'
            self.spin_op[out_key] = Udag * self.spin_op[key] * U

    def decode(self, sign: int = 1) -> None:
        """Inverse of encode."""
        self.encode(-sign)

    # -----------------------------------------------------------------------
    # Logical gates
    # -----------------------------------------------------------------------

    def LRx_gate(self, angle: float) -> None:
        """Logical Rx(angle) gate."""
        U = (-1j * angle / 2 * self.spin_op['parity']).expm()
        self.apply_operator(U)

    def LRz_gate(self, angle: float, bias_preserving: bool = True) -> None:
        """Logical Rz(angle) gate."""
        if bias_preserving:
            self.decode()
            U = (-1j * angle / 2 * self.spin_op['Lz_z']).expm()
            self.apply_operator(U)
            self.encode()
        else:
            U = (-1j * angle / 7 * self.spin_op[self.axis]).expm()
            self.apply_operator(U)

    # -----------------------------------------------------------------------
    # Noise / errors
    # -----------------------------------------------------------------------

    def phase_error(self, Iz: float = 0, Iz_sq: float = 0) -> None:
        U = (-1j * (Iz * self.spin_op['Iz'] + Iz_sq * self.spin_op['Iz^2'] / 2)).expm()
        self.apply_operator(U)

    def random_phase_error(self, Iz_range: float = 0, Iz_sq_range: float = 0) -> None:
        Iz = np.random.uniform(-Iz_range, Iz_range)
        Iz_sq = np.random.uniform(-Iz_sq_range, Iz_sq_range)
        self.phase_error(Iz, Iz_sq)

    # -----------------------------------------------------------------------
    # Error correction / detection — internally use self.dm (always in sync)
    # -----------------------------------------------------------------------

    def discard_error_space(self) -> None:
        """Project onto the code space {|0⟩, |d-1⟩} and renormalise."""
        dm_arr = self.dm.full().copy()
        dm_arr[1:-1, :] = 0
        dm_arr[:, 1:-1] = 0
        trace_val = np.real(np.trace(dm_arr))
        if trace_val > 0:
            dm_arr /= trace_val
        self.state = _dm_to_ket(qt.Qobj(dm_arr))

    def correct_errors(self) -> None:
        """One round of syndrome extraction: collapse each error subspace onto
        the code space while preserving coherences."""
        self.decode()
        dm_arr = self.dm.full().copy()
        new_dm = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.dim // 2):
            j = self.dim - 1 - i
            sub = np.zeros((self.dim, self.dim), dtype=complex)
            sub[0, 0]   = dm_arr[i, i]
            sub[-1, -1] = dm_arr[j, j]
            sub[0, -1]  = dm_arr[i, j]
            sub[-1, 0]  = dm_arr[j, i]
            new_dm += sub
        self.state = _dm_to_ket(qt.Qobj(new_dm))
        self.encode()

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def state_populations(self, basis: str = 'z') -> np.ndarray:
        """Population in each Fock state, measured in the given basis."""
        state = self.state.copy()
        if basis == 'x':
            state = (-1j * np.pi / 2 * self.spin_op['Iy']).expm() * state
        elif basis == 'y':
            state = (-1j * np.pi / 2 * self.spin_op['Ix']).expm() * state
        elif basis != 'z':
            raise ValueError("basis must be 'x', 'y', or 'z'.")
        return np.abs(state.full().flatten()) ** 2

    def logic_exp_values(self, Iz_basis: bool = False) -> list[float]:
        """Expectation values of the three logical Paulis [⟨Lx⟩, ⟨Ly⟩, ⟨Lz⟩]."""
        keys = ('Lx_z', 'Ly_z', 'Lz_z') if Iz_basis else ('Lx', 'Ly', 'Lz')
        return [qt.expect(self.spin_op[k], self.state) for k in keys]

    # parity(), fidelity(), Ix(), Iy(), Iz(), global_rotate(), subspace_rotate()
    # are all inherited from SpinInterface.

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot_state(self, title: str = None) -> tuple:
        """Wigner function + z- and x-basis population bars for the current state."""
        from psyduck.plotting import plot_wigner_and_populations
        return plot_wigner_and_populations(self.state, title=title)

    def plot_logic_exp_values(self, Iz_basis: bool = False, ax=None):
        """Bar chart of logical Pauli expectation values plus purity."""
        import matplotlib.pyplot as plt
        exp_values = self.logic_exp_values(Iz_basis)
        if ax is None:
            plt.figure(figsize=(5.5, 3))
            ax = plt.gca()
        ax.bar(['X', 'Y', 'Z'], exp_values)
        ax.bar(['purity', 'logic purity'],
               [self.dm.purity(), float(np.sqrt(np.sum(np.array(exp_values, dtype=float) ** 2)))])
        ax.axhline(0, linewidth=1)
        ax.set_title('Logic Expectation Values')
        plt.show()
        return ax

    def plot_wigner(self, projection: str = 'hammer', **kwargs):
        """Plot the Wigner function of the current state.

        :param projection: 'hammer' (default), '3d', or 'polar'
        """
        from psyduck.plotting import wigner_plot_hammer, wigner_plot_3d, wigner_plot_polar
        dispatch = {
            'hammer': wigner_plot_hammer,
            '3d':     wigner_plot_3d,
            'polar':  wigner_plot_polar,
        }
        if projection not in dispatch:
            raise ValueError(f"projection must be one of {list(dispatch)}, got {projection!r}")
        # Rotate 90° around Iz to align the plot with convention
        U_frame = (-1j * np.pi / 2 * self.spin_op['Iz']).expm()
        rotated = U_frame * self.state
        return dispatch[projection](rotated, **kwargs)
