# Psyduck
Simulation framework for high-spin qudit systems, with a focus on the antimony (Sb) nucleus (I=7/2) in semiconductor devices.

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/5076a302-52f7-4500-9ddb-ab6e8c10c74c" />

## Repository structure

### `src/` — Core simulation library
The canonical QuTiP-based library shared by all notebooks.

| File | Contents |
|------|----------|
| `spin.py` | `Spin` class — wraps a QuTiP state with helper methods for expectation values and state manipulation |
| `operations.py` | Spin operators (`get_spin_operators`), Hamiltonians (Zeeman, quadrupole), pulse/rotation operators (`pulse_operator`, `rotation_operator`, `global_rotation`, `parity_operator`) |
| `noise.py` | Lindblad collapse operators for magnetic (Iz) and electric/quadrupole (Iz²) dephasing noise |
| `fit_toolbox.py` | Curve-fitting classes built on lmfit/xarray: `Fit`, `ExponentialFit`, `SineFit`, `RabiFrequencyFit`, `T1Fit`, and more |
| `pulse_sequences.py` | Placeholder for composite pulse sequences |

### `noise model/` — Dephasing and coherence modelling
Notebooks studying how magnetic and electric noise limit coherence times.

| File | Contents |
|------|----------|
| `sb_dephasing_model_fit.ipynb` | Fits experimental Sb Ramsey data to a Lindblad master equation with stretched-exponential decay; extracts T₂\* and noise exponents for magnetic and electric channels |
| `dephasing_vs_dimension.ipynb` | Compares Z cat state dephasing rates as a function of spin dimension |
| `dephasing_vs_cat_angle.ipynb` | Explores how the cat-state superposition angle affects the free-decay dephasing time |
| `dynamical_decoupling_pulse_construction.ipynb` | Constructs dynamical decoupling sequences (XY-8, CPMG, etc.) decomposed into physical pulses on the ionised Sb nucleus |

### `beating/` — Ramsey beating simulations
| File | Contents |
|------|----------|
| `ramsey_beating.ipynb` | Simulates a spin-1/2 in a periodically modulated magnetic field; studies beating patterns in Ramsey experiments from a bichromatic drive |

### `dynamical decoupling/` — Spin-locking and echo sequences
| File | Contents |
|------|----------|
| `rotary_echo.ipynb` | Simulates spin-locking and rotary echo sequences for I=7/2; investigates decoupling from transverse noise |

### `global rotations/` — Global rotation and Rabi simulations
| File | Contents |
|------|----------|
| `global_rabi.ipynb` | Simulates global (non-selective) rotations of all Zeeman levels simultaneously; plots population dynamics across the full I=7/2 manifold |

### `ionization shock/` — Ionisation event modelling
Notebooks modelling what happens to the nuclear spin state when the electron is captured or emitted (ionisation shock).

| File | Contents |
|------|----------|
| `ionization_shock.py` | Helper module defining the ionisation-shock Hamiltonian and state-projection utilities |
| `ionization_shock_direct_projection.ipynb` | Builds the Hamiltonian model and directly projects the nuclear state across the ionisation event |
| `errorspace_readout_simulation.ipynb` | Simulates random readout time traces in the presence of ionisation errors |
| `lindblad_modeling_Sb.ipynb` | Lindblad master-equation model of the Sb nucleus after an ionisation shock |

### `quadrupole modeling/` — Quadrupole Hamiltonian fitting
| File | Contents |
|------|----------|
| `first and second order quadrupole.ipynb` | Fits first- and second-order quadrupole parameters to spectroscopic data; stores fitted Hamiltonians as `.npy` files |
| `H_quad.npy`, `H_quad_fit.npy`, `H_quad_laura.npy` | Saved quadrupole Hamiltonian matrices |
| `analyzed_data_24.nc` | Experimental spectroscopy dataset (NetCDF) |

### `plotting/` — Visualisation utilities
| File | Contents |
|------|----------|
| `wigner_plot.py` | `wigner_plot()` and `projection_plot_spin_wigner()` — Wigner function visualisations for arbitrary spin states |
| `readout_plot.py` | Readout histogram and time-trace plotting helpers |

### Root-level files
| File | Contents |
|------|----------|
| `fit_toolbox.py` | Source copy of the fitting framework (also installed as `src/fit_toolbox.py`) |
| `example.py` | Minimal usage example showing how to create a `Spin` object and compute expectation values |

## Quick start

```python
from src.spin import Spin
from src.operations import pulse_operator

nucleus = Spin(I=7/2)          # initialise Sb nucleus in ground state
nucleus.rotate(angle=np.pi/2, axis='x')   # apply π/2 pulse about x
print(nucleus.Iz())            # expectation value of Iz
```

## Dependencies
- [QuTiP](https://qutip.org/) — quantum toolbox in Python
- NumPy, SciPy, Matplotlib
- [lmfit](https://lmfit.github.io/lmfit-py/) — curve fitting
- [xarray](https://xarray.pydata.org/) — labelled data arrays
- [QCoDeS](https://microsoft.github.io/Qcodes/) — experimental data loading (some notebooks)
