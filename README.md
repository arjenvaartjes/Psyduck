# Psyduck
Simulation framework for high-spin qudit systems, with a focus on the antimony (Sb) nucleus (I=7/2) in semiconductor devices. Built on [QuTiP](https://qutip.org/).

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/5076a302-52f7-4500-9ddb-ab6e8c10c74c" />

## Quick start

```bash
uv sync          # install dependencies
```

```python
from psyduck import Spin
from psyduck import hamiltonians
import numpy as np

nucleus = Spin(I=7/2)                            # Sb nucleus in |m=+7/2⟩
nucleus.global_rotate(angle=np.pi/2, axis='x')  # π/2 pulse about x

H = hamiltonians.zeeman_hamiltonian(7/2, B0=1.0, gamma=5.55e6)
times = np.linspace(0, 1e-3, 500)
traj = nucleus.evolve(H, times)                  # returns a SpinSeries

traj.populations()   # (N, 8) — Zeeman level populations
traj.Iz()            # (N,)   — ⟨Iz⟩ at each time step
traj.parity()        # (N,)   — parity expectation value
traj[250]            # Spin object at index 250
```

## Library structure (`psyduck/`)

### Core classes

| Class | Module | Description |
|-------|--------|-------------|
| `Spin` | `spin.py` | Primary state container. Wraps a QuTiP density matrix or ket. State prep, evolution, rotations, diagnostics, and Wigner plots. |
| `SpinSeries` | `spin_series.py` | Ordered sequence of `Spin` states over an axis (time, angle, field, …). Returned by `Spin.evolve()`. |
| `SpinComposite` | `spin_composite.py` | Tensor-product composite of multiple sub-systems (e.g. nucleus + electron). |
| `CatQubit` | `cat_qubit.py` | Cat-qubit encoded in the extreme eigenstates of a high-spin nucleus. |

### Physics modules

| Module | Key contents |
|--------|-------------|
| `hamiltonians.py` | `zeeman_hamiltonian`, `quadrupole_hamiltonian`, `quadrupole_hamiltonian_from_Vab`, `hyperfine_hamiltonian`, `ner1_hamiltonian`, `ner2_hamiltonian`, `drive_hamiltonian` |
| `operations.py` | `get_spin_operators`, `get_transition_operators`, `global_rotation`, `subspace_rotation`, `snap`, `shift_operator`, `parity_operator`, `permutation_operator`, `euler_rotation` |
| `tensors.py` | `get_Q_tensor`, `Vab_to_Qab`, `Qab_to_Vab`, `get_R_tensor`, `get_S_tensor`, `voigt_to_tensor`, `tensor_to_voigt` |
| `noise.py` | `get_collapse_operators` — magnetic (Iz) and electric (Iz²) Lindblad operators with stretched-exponential profiles |
| `evolve.py` | `free_decay` — open-system evolution over multiple initial states with T₂\* fitting |
| `fit_toolbox.py` | `Fit` base class + subclasses: `ExponentialFit`, `SineFit`, `T1Fit`, `RabiFrequencyFit`, `DoubleExponentialFit`, `BayesianUpdateFit`, … |

### Plotting (`psyduck/plotting/`)

| Function | Description |
|----------|-------------|
| `wigner_plot` / `wigner_plot_3d` / `wigner_plot_hammer` / `wigner_plot_polar` | Wigner function visualisations in various projections |
| `make_wigner_gif` | Animated Wigner function over a SpinSeries |
| `plot_quadrupole_tensor(V_ab, I, B0, gamma, Q)` | 3D colour plots of fq1 and fq2 as a function of field orientation |
| `plot_transition_matrix` | Eigenstate decomposition and transition matrices |
| `plot_populations` | Zeeman level populations across a SpinSeries |
| `plot_wigner_and_populations` | Combined Wigner + population plot (cat qubit states) |

### Data flow

```
Spin  ──►  hamiltonians / operations  ──►  Spin.evolve()  ──►  SpinSeries  ──►  plotting
            (build H, collapse ops)         (sesolve /              │
                                             mesolve)          fit_toolbox
```

Pass `c_ops` from `noise.get_collapse_operators()` to `Spin.evolve()` to switch from unitary (`sesolve`) to open-system (`mesolve`) evolution.

## Examples / tutorials

All notebooks live under `examples/`. Key starting points:

| Notebook | Topic |
|----------|-------|
| `global rotations/global_rabi.ipynb` | Global (non-selective) Rabi oscillations across the full I=7/2 manifold |
| `pulse sequences/PulseSequences.ipynb` | Building and simulating selective pulse sequences |
| `noise model/sb_dephasing_model_fit.ipynb` | Fitting T₂\* and noise exponents from Ramsey data |
| `dynamical decoupling/continuous_DD.ipynb` | Dynamical decoupling sequences (XY-8, CPMG) |
| `quadrupole modeling/extracting_quadrupole_tensor_few_angles.ipynb` | Fitting the EFG tensor from NMR/NER spectroscopy |
| `cat qubit/CatQubitStatesAndGates.ipynb` | Cat-qubit states and logical gates |
| `hamiltonian/TimeDependentHamiltonians.ipynb` | Time-dependent drives and the GRF Hamiltonian |
| `Hyperfine/NuclearRamseyElectronMeasurement.ipynb` | Electron-mediated nuclear spin readout |

## Dependencies

- [QuTiP](https://qutip.org/) ≥ 5.2
- NumPy, SciPy, Matplotlib
- [lmfit](https://lmfit.github.io/lmfit-py/) — curve fitting
- [xarray](https://xarray.pydata.org/) — labelled arrays
- [pandas](https://pandas.pydata.org/)
