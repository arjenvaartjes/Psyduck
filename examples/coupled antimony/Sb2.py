"""Simulation tools for one electron coupled to two 123Sb nuclear spins.

This module follows the style of ``James_elecSb.py`` but removes the charge
degree of freedom and promotes the single antimony donor to two spin-7/2
antimony nuclei coupled to a common electron spin.

Hilbert-space convention
------------------------
Tensor-product ordering is fixed everywhere as

    electron spin (2D) x Sb_i nucleus (8D) x Sb_x nucleus (8D)

so the total Hilbert space has dimension 2 * 8 * 8 = 128.  The electron
operators are named ``Sx``, ``Sy`` and ``Sz``.  The first antimony nucleus uses
``Iix``, ``Iiy`` and ``Iiz``.  The second antimony nucleus uses ``Ixi``,
``Iyi`` and ``Izi``.  These names are intentionally explicit because notebook
work often benefits from seeing exactly which subsystem an operator acts on.

Units
-----
All Hamiltonian frequencies are expressed in GHz.  Because QuTiP evolves with
angular frequencies, helper functions that perform time evolution multiply
Hamiltonians by ``2*pi`` when the time axis is supplied in ns.  Direct gate
constructors such as ``nuclear_rotation`` return dimensionless unitaries.

Physics model
-------------
The static Hamiltonian is the neutral-donor antimony Hamiltonian from
Fernandez de Fuentes et al. generalized to two nuclei:

    H = B0 * gamma_e * Sz
        - B0 * gamma_n_i * Iiz
        - B0 * gamma_n_x * Izi
        + A_i * S.I_i
        + A_x * S.I_x
        + H_Q_i + H_Q_x
        + H_nuclear_nuclear

The papers model a single 123Sb donor with I=7/2 and S=1/2.  A real two-donor
device may have position-dependent hyperfine constants, quadrupole tensors,
and an indirect nuclear-nuclear interaction.  Those are exposed as parameters
instead of hard-coded assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import Qobj, basis, expect, jmat, qeye, tensor


###############################################################################
# PHYSICAL PARAMETERS
# All energies / frequencies are in GHz unless stated otherwise.
###############################################################################

# External magnetic field used in the original example.  The Nature papers work
# near 1 T; 1.38 T is kept here for continuity with ``James_elecSb.py``.
B0 = 1.38  # [T]

# Electron gyromagnetic ratio.  Paper [1] quotes gamma_e ~= 27.97 GHz/T for the
# donor-bound electron.  The older example used 28.02 GHz/T; either can be
# supplied to the Hamiltonian functions.
gamma_e = 27.97  # [GHz/T]

# 123Sb nuclear gyromagnetic ratio.  Paper [1] quotes gamma_n = 5.55 MHz/T.
gamma_n = 5.553e-3  # [GHz/T]

# Bulk 123Sb hyperfine coupling quoted in paper [1].  Gate fields, donor depth,
# and strain can renormalize this value, so A_i and A_x are explicit arguments
# in the Hamiltonian builders.
A0 = 101.52e-3  # [GHz]

# A small axial quadrupole scale for demonstration.  ``James_elecSb.py`` used
# -44.1 kHz.  In the paper the quadrupole interaction is generally a tensor
# sum_ab Q_ab I_a I_b; this module supports both the simple axial form and an
# explicit 3x3 tensor.
f_Q = -44.1e-6  # [GHz]


###############################################################################
# BASIS AND OPERATORS
###############################################################################

S_ELECTRON = 1 / 2
I_SB = 7 / 2

DIM_ELECTRON = 2
DIM_NUCLEAR = int(2 * I_SB + 1)
DIM_FULL = DIM_ELECTRON * DIM_NUCLEAR * DIM_NUCLEAR

# Basis order follows QuTiP's angular momentum convention:
# index 0 -> m = +I, index 1 -> m = +I-1, ..., last index -> m = -I.
mI_values = np.array([I_SB - k for k in range(DIM_NUCLEAR)], dtype=float)

Id_e = qeye(DIM_ELECTRON)
Id_i = qeye(DIM_NUCLEAR)
Id_x = qeye(DIM_NUCLEAR)
Id_full = tensor(Id_e, Id_i, Id_x)

# Electron spin-1/2 operators in the full 128D Hilbert space.
Sx = tensor(jmat(S_ELECTRON, "x"), Id_i, Id_x)
Sy = tensor(jmat(S_ELECTRON, "y"), Id_i, Id_x)
Sz = tensor(jmat(S_ELECTRON, "z"), Id_i, Id_x)

# First 123Sb nuclear spin operators.
Iix = tensor(Id_e, jmat(I_SB, "x"), Id_x)
Iiy = tensor(Id_e, jmat(I_SB, "y"), Id_x)
Iiz = tensor(Id_e, jmat(I_SB, "z"), Id_x)

# Second 123Sb nuclear spin operators.  The names mirror the user's requested
# convention: Ixi, Iyi, Izi.
Ixi = tensor(Id_e, Id_i, jmat(I_SB, "x"))
Iyi = tensor(Id_e, Id_i, jmat(I_SB, "y"))
Izi = tensor(Id_e, Id_i, jmat(I_SB, "z"))


@dataclass(frozen=True)
class Sb2Parameters:
    """Container for the static Hamiltonian parameters.

    Parameters are deliberately simple scalars so notebooks can sweep them
    without repeatedly editing function signatures.  The optional coupling
    terms default to zero because their values are device-geometry dependent.

    Attributes
    ----------
    B:
        Static magnetic field along the lab z axis in tesla.
    gamma_electron:
        Electron gyromagnetic ratio in GHz/T.
    gamma_n_i, gamma_n_x:
        Nuclear gyromagnetic ratios for the first and second 123Sb nuclei.
        They are normally identical but kept separate for isotope or fitting
        studies.
    A_i, A_x:
        Isotropic Fermi-contact hyperfine couplings between the electron and
        each antimony nucleus.
    fq_i, fq_x:
        Axial quadrupole coefficients used by ``H_quadrupole_axial``.
    J_iso:
        Optional isotropic direct or effective nuclear-nuclear coupling,
        ``J_iso * I_i.I_x``.
    J_zz:
        Optional secular nuclear-nuclear coupling, ``J_zz * Iiz * Izi``.
    """

    B: float = B0
    gamma_electron: float = gamma_e
    gamma_n_i: float = gamma_n
    gamma_n_x: float = gamma_n
    A_i: float = A0
    A_x: float = A0
    fq_i: float = f_Q
    fq_x: float = f_Q
    J_iso: float = 0.0
    J_zz: float = 0.0


###############################################################################
# SMALL HELPERS
###############################################################################


def _normalise_site(site: str) -> Literal["i", "x"]:
    """Map common site labels onto the two internal labels ``i`` and ``x``."""
    aliases_i = {"i", "1", "first", "left", "inner", "donor_i", "sb_i"}
    aliases_x = {"x", "2", "second", "right", "outer", "donor_x", "sb_x"}
    site_key = str(site).lower()
    if site_key in aliases_i:
        return "i"
    if site_key in aliases_x:
        return "x"
    raise ValueError("site must identify the first nucleus ('i') or second nucleus ('x')")


def _normalise_axis(axis: str | Sequence[float]) -> tuple[float, float, float]:
    """Return a unit vector for axis labels or arbitrary 3-vectors."""
    if isinstance(axis, str):
        axis_key = axis.lower()
        if axis_key == "x":
            return 1.0, 0.0, 0.0
        if axis_key == "y":
            return 0.0, 1.0, 0.0
        if axis_key == "z":
            return 0.0, 0.0, 1.0
        raise ValueError("axis must be 'x', 'y', 'z', or a 3-element vector")

    vec = np.asarray(axis, dtype=float)
    if vec.shape != (3,):
        raise ValueError("axis vectors must have exactly three components")
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("axis vector cannot be zero")
    vec = vec / norm
    return float(vec[0]), float(vec[1]), float(vec[2])


def electron_operator(axis: str | Sequence[float]) -> Qobj:
    """Return ``S_axis`` embedded in the full Hilbert space.

    This is useful when constructing custom electron drives.  For example,
    ``electron_operator('x')`` is the same object as the module-level ``Sx``.
    """
    nx, ny, nz = _normalise_axis(axis)
    return nx * Sx + ny * Sy + nz * Sz


def nuclear_operator(site: str, axis: str | Sequence[float]) -> Qobj:
    """Return a full-space nuclear spin operator for one antimony atom.

    Parameters
    ----------
    site:
        Which antimony nucleus to address.  Accepted labels include ``'i'`` or
        ``'1'`` for the first nucleus and ``'x'`` or ``'2'`` for the second.
    axis:
        ``'x'``, ``'y'``, ``'z'`` or a length-3 vector.
    """
    nx, ny, nz = _normalise_axis(axis)
    if _normalise_site(site) == "i":
        return nx * Iix + ny * Iiy + nz * Iiz
    return nx * Ixi + ny * Iyi + nz * Izi


def nuclear_spin_vector(site: str) -> tuple[Qobj, Qobj, Qobj]:
    """Return ``(Ix, Iy, Iz)`` for one nucleus, already embedded in 128D."""
    if _normalise_site(site) == "i":
        return Iix, Iiy, Iiz
    return Ixi, Iyi, Izi


def m_to_index(m: float) -> int:
    """Convert an Sb nuclear spin projection ``m`` to its basis index.

    ``m=+7/2`` maps to index 0 and ``m=-7/2`` maps to index 7.  A clear error is
    raised if ``m`` is not one of the allowed half-integer values.
    """
    matches = np.where(np.isclose(mI_values, float(m)))[0]
    if len(matches) != 1:
        allowed = ", ".join(f"{m:g}" for m in mI_values)
        raise ValueError(f"m must be one of [{allowed}]")
    return int(matches[0])


def electron_index(state: str | int) -> int:
    """Convert electron labels to basis indices.

    QuTiP's spin basis uses index 0 for ``m_s=+1/2`` and index 1 for
    ``m_s=-1/2``.  Accepted string labels are ``'up'``/``'u'`` and
    ``'down'``/``'d'``.
    """
    if isinstance(state, int):
        if state not in (0, 1):
            raise ValueError("electron index must be 0 or 1")
        return state

    key = state.lower()
    if key in {"up", "u", "+", "+1/2", "ms=+1/2"}:
        return 0
    if key in {"down", "d", "-", "-1/2", "ms=-1/2"}:
        return 1
    raise ValueError("electron state must be 'up', 'down', 0, or 1")


def product_state(electron: str | int = "down", m_i: float = -I_SB, m_x: float = -I_SB) -> Qobj:
    """Create a product ket ``|electron, m_i, m_x>`` in the 128D basis."""
    return tensor(
        basis(DIM_ELECTRON, electron_index(electron)),
        basis(DIM_NUCLEAR, m_to_index(m_i)),
        basis(DIM_NUCLEAR, m_to_index(m_x)),
    )


def product_projector(electron: str | int = "down", m_i: float = -I_SB, m_x: float = -I_SB) -> Qobj:
    """Projector onto a product basis state."""
    psi = product_state(electron=electron, m_i=m_i, m_x=m_x)
    return psi * psi.dag()


def density_matrix(psi: Qobj) -> Qobj:
    """Return ``|psi><psi|`` if ``psi`` is a ket, otherwise return ``psi``."""
    if psi.isket:
        return psi * psi.dag()
    return psi


###############################################################################
# HAMILTONIAN TERMS
###############################################################################


def S_dot_I(site: str) -> Qobj:
    """Scalar product between the electron spin and one antimony nucleus."""
    Ix_site, Iy_site, Iz_site = nuclear_spin_vector(site)
    return Sx * Ix_site + Sy * Iy_site + Sz * Iz_site


def I_dot_I() -> Qobj:
    """Scalar product between the two antimony nuclear spins."""
    return Iix * Ixi + Iiy * Iyi + Iiz * Izi


def H_electron_zeeman(B: float = B0, ge: float = gamma_e) -> Qobj:
    """Electron Zeeman Hamiltonian.

    ``H_eZ = gamma_e * B * Sz``

    The sign convention matches paper [1], where the neutral donor Hamiltonian
    contains ``B0 * gamma_e * Sz``.  With QuTiP's basis ordering the electron
    ``|up>`` state has ``m_s=+1/2`` and a higher Zeeman energy for positive
    ``gamma_e`` and positive ``B``.
    """
    return ge * B * Sz


def H_nuclear_zeeman(B: float = B0, gn_i: float = gamma_n, gn_x: float = gamma_n) -> Qobj:
    """Nuclear Zeeman Hamiltonian for both 123Sb nuclei.

    ``H_nZ = -B * (gamma_n_i * Iiz + gamma_n_x * Izi)``

    For 123Sb, ``gamma_n`` is positive, so the ``m_I=+7/2`` state is lower in
    energy than ``m_I=-7/2`` in a positive magnetic field.
    """
    return -B * (gn_i * Iiz + gn_x * Izi)


def H_hyperfine(A_i: float = A0, A_x: float = A0) -> Qobj:
    """Isotropic Fermi-contact hyperfine coupling to both nuclei.

    ``H_hf = A_i * S.I_i + A_x * S.I_x``

    If the electron is localized closer to one donor, set the other ``A`` to a
    smaller value.  If one nucleus is ionized or effectively uncoupled, use
    ``A=0`` for that site.
    """
    return A_i * S_dot_I("i") + A_x * S_dot_I("x")


def H_quadrupole_axial(site: str, fq: float = f_Q, subtract_identity: bool = False) -> Qobj:
    """Axial quadrupole Hamiltonian for one antimony nucleus.

    The compact model used in the original example was ``fq * Iz^2``.  The
    physically complete expression contains an arbitrary tensor
    ``sum_ab Q_ab I_a I_b``; use ``H_quadrupole_tensor`` for that case.

    Parameters
    ----------
    site:
        Which nucleus receives the quadrupole interaction.
    fq:
        Axial quadrupole coefficient in GHz.
    subtract_identity:
        If true, use ``fq * (Iz^2 - I(I+1)/3)``.  This removes the trace-like
        identity shift and leaves transition frequencies unchanged.
    """
    Iz_site = nuclear_operator(site, "z")
    Hq = fq * Iz_site * Iz_site
    if subtract_identity:
        Hq -= fq * I_SB * (I_SB + 1) / 3 * Id_full
    return Hq


def H_quadrupole_tensor(site: str, Qab: np.ndarray) -> Qobj:
    """General quadrupole Hamiltonian ``sum_ab Qab[a,b] I_a I_b``.

    ``Qab`` must be a 3x3 tensor in GHz.  It can be fitted from EFG simulations,
    Stark-shift measurements, or the tensor notation in paper [1].  The matrix
    is symmetrized numerically so that small floating-point asymmetries do not
    create an artificial anti-Hermitian component.
    """
    Qab = np.asarray(Qab, dtype=float)
    if Qab.shape != (3, 3):
        raise ValueError("Qab must have shape (3, 3)")

    Qab = 0.5 * (Qab + Qab.T)
    I_ops = nuclear_spin_vector(site)
    Hq = 0 * Id_full
    for a in range(3):
        for b in range(3):
            Hq += Qab[a, b] * I_ops[a] * I_ops[b]
    return Hq


def H_quadrupole(fq_i: float = f_Q, fq_x: float = f_Q, subtract_identity: bool = False) -> Qobj:
    """Axial quadrupole Hamiltonian for both antimony nuclei."""
    return (
        H_quadrupole_axial("i", fq=fq_i, subtract_identity=subtract_identity)
        + H_quadrupole_axial("x", fq=fq_x, subtract_identity=subtract_identity)
    )


def H_nuclear_nuclear(J_iso: float = 0.0, J_zz: float = 0.0) -> Qobj:
    """Optional effective coupling between the two antimony nuclei.

    ``J_iso`` adds an isotropic ``I_i.I_x`` interaction.  ``J_zz`` adds the
    secular coupling ``Iiz * Izi``.  Both default to zero because the mechanism
    and scale depend on the device geometry and electron wavefunction.
    """
    return J_iso * I_dot_I() + J_zz * Iiz * Izi


def H_total(
    B: float = B0,
    A_i: float = A0,
    A_x: float = A0,
    fq_i: float = f_Q,
    fq_x: float = f_Q,
    gamma_electron: float = gamma_e,
    gamma_n_i: float = gamma_n,
    gamma_n_x: float = gamma_n,
    J_iso: float = 0.0,
    J_zz: float = 0.0,
    Qab_i: np.ndarray | None = None,
    Qab_x: np.ndarray | None = None,
) -> Qobj:
    """Full static Hamiltonian for the electron plus two 123Sb nuclei.

    Parameters are in GHz or GHz/T.  The default quadrupole model is axial:
    ``fq_i * Iiz^2 + fq_x * Izi^2``.  Supplying ``Qab_i`` or ``Qab_x`` replaces
    the corresponding axial term with a full tensor expression.
    """
    H = H_electron_zeeman(B=B, ge=gamma_electron)
    H += H_nuclear_zeeman(B=B, gn_i=gamma_n_i, gn_x=gamma_n_x)
    H += H_hyperfine(A_i=A_i, A_x=A_x)

    if Qab_i is None:
        H += H_quadrupole_axial("i", fq=fq_i)
    else:
        H += H_quadrupole_tensor("i", Qab_i)

    if Qab_x is None:
        H += H_quadrupole_axial("x", fq=fq_x)
    else:
        H += H_quadrupole_tensor("x", Qab_x)

    H += H_nuclear_nuclear(J_iso=J_iso, J_zz=J_zz)
    return H


def H_from_params(params: Sb2Parameters) -> Qobj:
    """Build ``H_total`` from an ``Sb2Parameters`` dataclass instance."""
    return H_total(
        B=params.B,
        A_i=params.A_i,
        A_x=params.A_x,
        fq_i=params.fq_i,
        fq_x=params.fq_x,
        gamma_electron=params.gamma_electron,
        gamma_n_i=params.gamma_n_i,
        gamma_n_x=params.gamma_n_x,
        J_iso=params.J_iso,
        J_zz=params.J_zz,
    )


###############################################################################
# DRIVE HAMILTONIANS
###############################################################################


def H_esr_drive(B1: float, phase: float = 0.0, ge: float = gamma_e) -> Qobj:
    """Electron magnetic-resonance drive amplitude.

    ``H_drive = gamma_e * B1 * (Sx cos phase + Sy sin phase)``

    ``B1`` is the transverse oscillating magnetic-field amplitude in tesla.
    Use this as the amplitude operator in a QuTiP time-dependent Hamiltonian or
    with a rotating-wave approximation in a custom frame.
    """
    return ge * B1 * (np.cos(phase) * Sx + np.sin(phase) * Sy)


def H_nmr_drive(site: str, B1: float, phase: float = 0.0, gn: float = gamma_n) -> Qobj:
    """Nuclear magnetic-resonance drive amplitude for one antimony nucleus.

    ``H_drive = -gamma_n * B1 * (Ix cos phase + Iy sin phase)``

    The sign matches the nuclear Zeeman term.  In the generalized rotating
    frame of paper [2], the phase of each tone can be shifted independently;
    for exact diagonal phase control use the SNAP helpers below.
    """
    return -gn * B1 * (
        np.cos(phase) * nuclear_operator(site, "x")
        + np.sin(phase) * nuclear_operator(site, "y")
    )


def H_global_nmr_drive(B1_i: float, B1_x: float | None = None, phase: float = 0.0) -> Qobj:
    """Drive both antimony nuclei with the same RF phase.

    If ``B1_x`` is omitted, both nuclei receive the same RF amplitude.  This is
    useful for modelling non-local RF antennas.  Site-selective addressing can
    still arise spectrally if the nuclei have different ``A`` or quadrupole
    shifts.
    """
    if B1_x is None:
        B1_x = B1_i
    return H_nmr_drive("i", B1_i, phase=phase) + H_nmr_drive("x", B1_x, phase=phase)


###############################################################################
# UNITARIES AND GATES
###############################################################################


def electron_rotation(angle: float, axis: str | Sequence[float] = "x") -> Qobj:
    """Electron spin rotation ``exp(-1j * angle * S_axis)``."""
    return (-1j * angle * electron_operator(axis)).expm()


def nuclear_rotation(site: str, angle: float, axis: str | Sequence[float] = "x") -> Qobj:
    """Single-nucleus SU(2) rotation ``exp(-1j * angle * I_axis)``."""
    return (-1j * angle * nuclear_operator(site, axis)).expm()


def global_nuclear_rotation(angle: float, axis: str | Sequence[float] = "x") -> Qobj:
    """Apply the same SU(2) rotation to both 123Sb nuclei.

    This is a model of a truly global RF pulse acting on both nuclei:

        U = exp[-i angle (I_i_axis + I_x_axis)]

    Because the two spin operators act on different tensor factors, this is
    equivalent to ``nuclear_rotation('i', angle, axis) *
    nuclear_rotation('x', angle, axis)``.
    """
    return (-1j * angle * (nuclear_operator("i", axis) + nuclear_operator("x", axis))).expm()


def covariant_su2_rotation(site: str, theta: float, phi: float = 0.0, paper_sign: bool = False) -> Qobj:
    """Covariant SU(2) rotation used for spin-7/2 control.

    Paper [2] writes the generalized rotating-frame operation as

        R_Theta(phi) = exp(i * Theta * (Ix cos(phi) + Iy sin(phi))).

    The default here follows the usual physics convention ``exp(-i theta I)``.
    Set ``paper_sign=True`` to use the paper's sign convention exactly.
    """
    generator = np.cos(phi) * nuclear_operator(site, "x") + np.sin(phi) * nuclear_operator(site, "y")
    sign = 1j if paper_sign else -1j
    return (sign * theta * generator).expm()


def global_covariant_su2_rotation(theta: float, phi: float = 0.0, paper_sign: bool = False) -> Qobj:
    """Covariant SU(2) rotation applied to both nuclei at once."""
    generator = (
        np.cos(phi) * (Iix + Ixi)
        + np.sin(phi) * (Iiy + Iyi)
    )
    sign = 1j if paper_sign else -1j
    return (sign * theta * generator).expm()


def naked_nuclear_snap(phases: Sequence[float]) -> Qobj:
    """SNAP gate on an isolated 8D spin-7/2 nuclear Hilbert space.

    ``phases[k]`` is applied to the basis state with
    ``m = +7/2 - k``.  A global phase is physically irrelevant but retained so
    phase-programming experiments can be represented exactly.
    """
    phases = np.asarray(phases, dtype=float)
    if phases.shape != (DIM_NUCLEAR,):
        raise ValueError(f"phases must have length {DIM_NUCLEAR}")
    return Qobj(np.diag(np.exp(1j * phases)), dims=[[DIM_NUCLEAR], [DIM_NUCLEAR]])


def snap_gate(site: str, phases: Sequence[float]) -> Qobj:
    """Selective number-dependent arbitrary phase gate on one nucleus.

    This is the finite-dimensional analogue of the SNAP gate discussed in
    paper [2].  In the generalized rotating frame it can be virtual: the phases
    are implemented by changing software-defined oscillator phases instead of
    applying a physical pulse.
    """
    U_n = naked_nuclear_snap(phases)
    if _normalise_site(site) == "i":
        return tensor(Id_e, U_n, Id_x)
    return tensor(Id_e, Id_i, U_n)


def two_nucleus_snap(phases_i: Sequence[float], phases_x: Sequence[float]) -> Qobj:
    """Apply independent SNAP phase vectors to both nuclei."""
    return tensor(Id_e, naked_nuclear_snap(phases_i), naked_nuclear_snap(phases_x))


def one_axis_twist(site: str, chi: float, sign: int = -1) -> Qobj:
    """One-axis twisting unitary generated by ``Iz^2``.

    ``U = exp(sign * 1j * chi * Iz^2)``

    This can also be viewed as a virtual SNAP because ``Iz^2`` is diagonal in
    the ``|m_I>`` basis.  Paper [2] uses virtual-SNAP phases to realize the
    phase pattern required for cat-state creation.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    phases = sign * chi * (mI_values ** 2)
    return snap_gate(site, phases)


def givens_rotation(site: str, m_a: float, m_b: float, theta: float, phase: float = 0.0) -> Qobj:
    """Two-level Givens rotation within one nuclear spin ladder.

    The rotation acts only on the two selected ``|m_a>`` and ``|m_b>`` levels
    of one nucleus and leaves all other nuclear levels untouched.  This is a
    convenient idealized model of the level-selective pulses used to assemble
    arbitrary SU(8) operations in paper [2].
    """
    ia = m_to_index(m_a)
    ib = m_to_index(m_b)
    if ia == ib:
        raise ValueError("m_a and m_b must be different states")

    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    U = np.eye(DIM_NUCLEAR, dtype=np.complex128)
    U[ia, ia] = c
    U[ib, ib] = c
    U[ia, ib] = -np.exp(-1j * phase) * s
    U[ib, ia] = np.exp(1j * phase) * s
    U_n = Qobj(U, dims=[[DIM_NUCLEAR], [DIM_NUCLEAR]])

    if _normalise_site(site) == "i":
        return tensor(Id_e, U_n, Id_x)
    return tensor(Id_e, Id_i, U_n)


###############################################################################
# STATE PREPARATION AND ANALYSIS
###############################################################################


def naked_nuclear_cat_state(m_abs: float = I_SB, phase: float = 0.0) -> Qobj:
    """Return ``(|+m_abs> + exp(i phase) |-m_abs>) / sqrt(2)``.

    The returned ket acts only in the 8D nuclear Hilbert space.  Use
    ``embed_nuclear_state`` or ``two_nucleus_product`` to place it in the full
    electron-plus-two-nuclei Hilbert space.
    """
    ket_plus = basis(DIM_NUCLEAR, m_to_index(abs(m_abs)))
    ket_minus = basis(DIM_NUCLEAR, m_to_index(-abs(m_abs)))
    return (ket_plus + np.exp(1j * phase) * ket_minus).unit()


def naked_spin_coherent_state(theta: float, phi: float, reference_m: float = -I_SB) -> Qobj:
    """Create an ideal spin coherent state in the 8D nuclear Hilbert space.

    The default reference is ``|-7/2>``.  A coherent state in direction
    ``(theta, phi)`` is produced by rotating this reference state with the
    ordinary spin-7/2 SU(2) representation.
    """
    psi0 = basis(DIM_NUCLEAR, m_to_index(reference_m))
    Ix = jmat(I_SB, "x")
    Iy = jmat(I_SB, "y")
    Iz = jmat(I_SB, "z")
    U = (-1j * phi * Iz).expm() * (-1j * theta * Iy).expm()
    return (U * psi0).unit()


def embed_nuclear_state(
    site: str,
    psi_nuclear: Qobj,
    electron: str | int = "down",
    other_m: float = -I_SB,
) -> Qobj:
    """Embed an 8D nuclear ket into the full 128D Hilbert space.

    The selected site receives ``psi_nuclear``.  The other nucleus is placed in
    the product state ``|other_m>`` and the electron in ``|electron>``.
    """
    if not psi_nuclear.isket or psi_nuclear.shape != (DIM_NUCLEAR, 1):
        raise ValueError("psi_nuclear must be an 8D ket")

    e_ket = basis(DIM_ELECTRON, electron_index(electron))
    other_ket = basis(DIM_NUCLEAR, m_to_index(other_m))
    if _normalise_site(site) == "i":
        return tensor(e_ket, psi_nuclear, other_ket)
    return tensor(e_ket, other_ket, psi_nuclear)


def two_nucleus_product(
    psi_i: Qobj,
    psi_x: Qobj,
    electron: str | int = "down",
) -> Qobj:
    """Create ``|electron> x |psi_i> x |psi_x>``."""
    if not psi_i.isket or psi_i.shape != (DIM_NUCLEAR, 1):
        raise ValueError("psi_i must be an 8D ket")
    if not psi_x.isket or psi_x.shape != (DIM_NUCLEAR, 1):
        raise ValueError("psi_x must be an 8D ket")
    return tensor(basis(DIM_ELECTRON, electron_index(electron)), psi_i, psi_x)


def nuclear_populations(state: Qobj, site: str) -> np.ndarray:
    """Return the marginal ``|m_I>`` populations for one nucleus.

    Works for kets and density matrices.  The result is ordered as
    ``mI_values``: ``+7/2, +5/2, ..., -7/2``.
    """
    rho = density_matrix(state)
    pops = np.zeros(DIM_NUCLEAR, dtype=float)
    site_key = _normalise_site(site)
    for k, m in enumerate(mI_values):
        Pn = basis(DIM_NUCLEAR, k) * basis(DIM_NUCLEAR, k).dag()
        P = tensor(Id_e, Pn, Id_x) if site_key == "i" else tensor(Id_e, Id_i, Pn)
        pops[k] = float(np.real(expect(P, rho)))
    return pops


def parity_operator(site: str) -> Qobj:
    """Parity operator used for spin-cat interference readout.

    The diagonal entries are ``(-1)^k`` in the basis ordered from
    ``m=+7/2`` down to ``m=-7/2``.  This matches the alternating parity used in
    the cat-state paper for spin-7/2 tomography.
    """
    diag = np.array([(-1) ** k for k in range(DIM_NUCLEAR)], dtype=float)
    Pn = Qobj(np.diag(diag), dims=[[DIM_NUCLEAR], [DIM_NUCLEAR]])
    if _normalise_site(site) == "i":
        return tensor(Id_e, Pn, Id_x)
    return tensor(Id_e, Id_i, Pn)


def parity_expectation(state: Qobj, site: str) -> float:
    """Expectation value of the nuclear parity operator for one site."""
    return float(np.real(expect(parity_operator(site), density_matrix(state))))


def iz_expectation(state: Qobj, site: str) -> float:
    """Expectation value of ``Iz`` for one nucleus."""
    return float(np.real(expect(nuclear_operator(site, "z"), density_matrix(state))))


###############################################################################
# SPECTROSCOPY HELPERS
###############################################################################


def eigenenergies(H: Qobj, subtract_ground: bool = True) -> np.ndarray:
    """Return sorted eigenenergies of ``H`` in GHz."""
    energies = np.asarray(H.eigenenergies(), dtype=float)
    if subtract_ground:
        energies = energies - energies[0]
    return energies


def transition_frequency_product_basis(
    H: Qobj,
    site: str,
    m_from: float,
    m_to: float,
    electron: str | int = "down",
    other_m: float = -I_SB,
) -> float:
    """Approximate a transition frequency using product-basis energies.

    This is most useful in the high-field regime where the eigenstates are
    close to ``|electron, m_i, m_x>`` product states.  It returns
    ``abs(E_to - E_from)`` in GHz.
    """
    site_key = _normalise_site(site)
    if site_key == "i":
        psi_from = product_state(electron=electron, m_i=m_from, m_x=other_m)
        psi_to = product_state(electron=electron, m_i=m_to, m_x=other_m)
    else:
        psi_from = product_state(electron=electron, m_i=other_m, m_x=m_from)
        psi_to = product_state(electron=electron, m_i=other_m, m_x=m_to)

    E_from = float(np.real(expect(H, psi_from)))
    E_to = float(np.real(expect(H, psi_to)))
    return abs(E_to - E_from)


def adjacent_nmr_frequencies(
    H: Qobj,
    site: str,
    electron: str | int = "down",
    other_m: float = -I_SB,
) -> list[tuple[float, float, float]]:
    """Return product-basis adjacent NMR frequencies for one nucleus.

    Each tuple is ``(m_high, m_low, frequency_GHz)`` for transitions
    ``m_high <-> m_low`` in descending basis order.
    """
    rows = []
    for k in range(DIM_NUCLEAR - 1):
        m_high = float(mI_values[k])
        m_low = float(mI_values[k + 1])
        freq = transition_frequency_product_basis(
            H,
            site=site,
            m_from=m_high,
            m_to=m_low,
            electron=electron,
            other_m=other_m,
        )
        rows.append((m_high, m_low, freq))
    return rows


def print_adjacent_nmr_frequencies(
    H: Qobj,
    site: str,
    electron: str | int = "down",
    other_m: float = -I_SB,
    unit: Literal["GHz", "MHz", "kHz"] = "MHz",
) -> None:
    """Pretty-print adjacent NMR transition frequencies for notebook use."""
    scale = {"GHz": 1.0, "MHz": 1e3, "kHz": 1e6}[unit]
    print(f"Adjacent NMR transitions for Sb_{_normalise_site(site)} with electron={electron}:")
    for m_high, m_low, freq in adjacent_nmr_frequencies(H, site, electron, other_m):
        print(f"  {m_high:+g} <-> {m_low:+g}: {freq * scale:.6f} {unit}")


###############################################################################
# TIME EVOLUTION AND PLOTTING
###############################################################################


def unitary_from_hamiltonian(H: Qobj, duration_ns: float) -> Qobj:
    """Return ``exp(-i 2*pi H duration_ns)`` for ``H`` in GHz."""
    return (-1j * 2 * np.pi * H * duration_ns).expm()


def evolve_state(
    H: Qobj | list,
    psi0: Qobj,
    times_ns: Sequence[float],
    e_ops: Iterable[Qobj] | None = None,
    c_ops: Iterable[Qobj] | None = None,
    **solver_kwargs,
) -> qt.solver.Result:
    """Evolve a state using ns as the time unit and GHz as the Hamiltonian unit.

    QuTiP's Schrodinger equation expects angular frequency units.  This helper
    multiplies every Hamiltonian operator by ``2*pi`` so a static Hamiltonian in
    GHz evolves correctly over a time array in ns.

    ``H`` can be a single ``Qobj`` or a QuTiP time-dependent Hamiltonian list.
    Coefficient functions in a list still receive time in ns.
    """
    times_ns = np.asarray(times_ns, dtype=float)
    e_ops = [] if e_ops is None else list(e_ops)
    c_ops = [] if c_ops is None else list(c_ops)

    if isinstance(H, Qobj):
        H_qutip = 2 * np.pi * H
    else:
        H_qutip = []
        for term in H:
            if isinstance(term, Qobj):
                H_qutip.append(2 * np.pi * term)
            else:
                op, coeff = term
                H_qutip.append([2 * np.pi * op, coeff])

    if c_ops:
        return qt.mesolve(H_qutip, psi0, times_ns, c_ops=c_ops, e_ops=e_ops, **solver_kwargs)
    return qt.sesolve(H_qutip, psi0, times_ns, e_ops=e_ops, **solver_kwargs)


def plot_nuclear_populations(
    states: Sequence[Qobj],
    site: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot nuclear ``|m_I>`` populations for one or more states."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    states = list(states)
    pop_matrix = np.vstack([nuclear_populations(state, site) for state in states])
    im = ax.imshow(pop_matrix.T, aspect="auto", origin="upper", interpolation="nearest")
    ax.set_yticks(range(DIM_NUCLEAR))
    ax.set_yticklabels([f"{m:g}" for m in mI_values])
    ax.set_xlabel("state index")
    ax.set_ylabel(f"Sb_{_normalise_site(site)} m_I")
    ax.set_title(title or f"Sb_{_normalise_site(site)} nuclear populations")
    plt.colorbar(im, ax=ax, label="population")
    return ax


def plot_energy_spectrum(
    H_values: Sequence[Qobj],
    x_values: Sequence[float] | None = None,
    ax: plt.Axes | None = None,
    subtract_ground: bool = True,
    unit: Literal["GHz", "MHz"] = "GHz",
) -> plt.Axes:
    """Plot eigenenergies for a list of Hamiltonians."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if x_values is None:
        x_values = np.arange(len(H_values))
    x_values = np.asarray(x_values)

    scale = 1.0 if unit == "GHz" else 1e3
    spectra = np.vstack([eigenenergies(H, subtract_ground=subtract_ground) for H in H_values]) * scale
    for level in range(spectra.shape[1]):
        ax.plot(x_values, spectra[:, level], color="black", lw=0.8, alpha=0.45)

    ax.set_xlabel("sweep parameter")
    ax.set_ylabel(f"energy ({unit})")
    return ax

