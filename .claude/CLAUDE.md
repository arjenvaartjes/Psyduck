# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Psyduck is a quantum simulation framework for high-spin qudit systems, primarily targeting the antimony (Sb) nucleus (I=7/2) in semiconductor devices. Built on top of QuTiP (Quantum Toolbox in Python).

## Development Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run a script
uv run python example.py

# Quick Python snippet
python -c "from psyduck import Spin; s = Spin(7/2); print(s.Iz())"
```

There is no formal test suite. Validation is typically done interactively in Jupyter notebooks under `examples/`.

## Architecture

### Core Data Flow

`Spin` (state container) → `operations` (build Hamiltonians/unitaries) → `Spin.evolve()` or `Spin.apply_operator()` → `plotting` (visualize)

### Key Classes and Modules

**`psyduck/spin.py` — `Spin` class**
The primary user-facing object. Wraps a QuTiP `Qobj` density matrix or ket. All simulation workflows start here.
- State preparation: `make_zcat_state()`, `make_xcat_state()`
- Evolution: `evolve(H, times, c_ops)` delegates to QuTiP `sesolve`/`mesolve`
- Rotations: `global_rotate()` for full Hilbert space; `subspace_rotate()` for two-level (Givens) rotations on selected levels
- Diagnostics: `expectation()`, `Ix/Iy/Iz()`, `parity()`
- Visualization: `plot_wigner()` wraps the plotting module

**`psyduck/operations.py` — Hamiltonian and operator constructors**
Stateless functions that return QuTiP `Qobj` operators. Key Hamiltonians:
- `zeeman_hamiltonian`, `quadrupole_hamiltonian`, `hyperfine_hamiltonian`, `rf_hamiltonian`

Key unitaries: `global_rotation`, `subspace_rotation`, `snap`, `shift_operator`, `parity_operator`, `permutation_operator`

**`psyduck/noise.py` — Lindblad collapse operators**
`get_collapse_operators()` builds dephasing operators for magnetic noise (∝ Iz) and electric noise (∝ Iz²), supporting stretched-exponential decay via a configurable exponent.

**`psyduck/evolve.py` — Higher-level simulation helpers**
`free_decay()` runs open-system evolution for multiple initial states and returns fitted T2* times and decay exponents.

**`psyduck/fit_toolbox.py` — Curve fitting**
`lmfit`-based framework. Base `Fit` class with subclasses: `ExponentialFit`, `SineFit`, `RabiFrequencyFit`, `T1Fit`. Supports auto-initialization, parameter fixing, and evaluation via `fit(sweep_vals)`.

**`psyduck/plotting/`**
- `wigner_plot.py`: Wigner function visualizations in 3D, Hammer projection, polar projection, and GIF animation
- `readout_plot.py`: Transition matrix and eigenstate decomposition plots

### Open System vs Closed System

Pass `c_ops` (from `noise.get_collapse_operators()`) to `Spin.evolve()` to switch from `sesolve` (closed) to `mesolve` (open/Lindblad). Without `c_ops`, evolution is unitary.

### Quadrupole Interaction

The quadrupole Hamiltonian takes a 3×3 voltage gradient tensor `V_ab` and a quadrupole moment `Q`. This is the dominant interaction distinguishing high-spin qudits from simple qubits and is central to the physics of the Sb nucleus.