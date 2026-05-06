# Simulate the Sb-e system
#
# System: 123-Sb donor in silicon with a single bound electron.
# The electron can occupy the donor site or a nearby interface (e.g.
# Si/SiO2 gate interface), creating a charge qubit degree of freedom.
# This is the "flip-flop qubit" architecture (cf. Tosi et al., Nat. Commun.
# 8, 450, 2017) adapted for 123-Sb instead of 31-P.
#
# Hilbert space decomposition (total dimension 32):
#   H_total = H_charge (2D) ⊗ H_electron_spin (2D) ⊗ H_nuclear_spin (8D)
#
# Subspaces:
#   - Charge (2D):   |d⟩ = electron at donor,  |i⟩ = electron at interface
#   - Electron (2D): spin-1/2, |↑⟩ and |↓⟩
#   - Nuclear (8D):  123-Sb has I = 7/2, states |m_I⟩ for m_I = +7/2 ... -7/2
#
# The charge degree of freedom is crucial because:
#   - At donor: full hyperfine coupling A ≈ 93 MHz → shifts NMR frequencies
#   - At interface: hyperfine coupling ≈ 0 → different NMR frequencies
#   - The electron g-factor also differs between donor and interface sites
#
# References:
#   Tosi et al., Nat. Commun. 8, 450 (2017)        — flip-flop qubit (31-P)
#   Savytskyy et al., Science 380, 1201 (2023)      — flip-flop qubit operation
#   Franke et al., Sci. Adv. 1, e1500022 (2015)     — 123-Sb donor in Si

import qutip as qt
from qutip import tensor, sigmax, sigmay, sigmaz, qeye, jmat, Qobj
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# PHYSICAL PARAMETERS
# All energies / frequencies in GHz unless otherwise noted.
###############################################################################

# --- External magnetic field ---
B0 = 1.38  # [T] Static magnetic field along z-axis

# --- Electron spin properties ---
# ASSUMPTION: γ_e = 28.02 GHz/T is used as the baseline electron gyromagnetic
# ratio (γ_e = g_e × μ_B / h). This corresponds to g_e ≈ 2.0014, consistent
# with a donor-bound electron in bulk silicon.
gamma_e = 28.02  # [GHz/T] Electron gyromagnetic ratio

# ASSUMPTION: The electron g-factor differs slightly between the donor site
# and the Si/SiO2 interface due to interface spin-orbit coupling effects.
# delta_g is the RELATIVE g-factor shift at the interface:
#   g_donor     = γ_e         (bulk silicon value, taken as reference)
#   g_interface = γ_e × (1 + δg)
# The sign and magnitude of δg depend on the interface details. A value of
# ~0.1% is typical for Si/SiO2 interfaces.
delta_g = 0.001  # [dimensionless] Relative g-factor shift at interface

# --- Nuclear spin properties (123-Sb) ---
# ASSUMPTION: We model the 123-Sb isotope with nuclear spin I = 7/2
# (natural abundance 42.8%). The other stable isotope 121-Sb has I = 5/2
# and would need a 6-dimensional nuclear Hilbert space instead.
# ASSUMPTION: The nuclear gyromagnetic ratio γ_n for 123-Sb is positive.
# Value: γ_n = 5.5532 MHz/T from nuclear data tables.
gamma_n = 5.553e-3  # [GHz/T] Nuclear gyromagnetic ratio for 123-Sb (positive)

# --- Hyperfine coupling ---
# The isotropic Fermi contact hyperfine constant A is proportional to the
# electron spin density at the nucleus: A ∝ |ψ_e(r_nuc)|².
# ASSUMPTION: The hyperfine interaction is purely isotropic (Fermi contact).
# The anisotropic (dipolar) component is negligible for substitutional donors
# in silicon due to the cubic Td symmetry of the donor site.
# ASSUMPTION: A0 = 93 MHz is used here. The bulk value for 123-Sb in Si is
# ~101.52 MHz (Franke et al., 2015). The reduced value may reflect strain
# or electric-field modification of the electron wavefunction.
A0 = 93e-3  # [GHz] Isotropic hyperfine coupling constant (≈ 93 MHz)

# --- Nuclear electric quadrupole ---
# For I > 1/2 (here I = 7/2), the nucleus has a non-zero electric quadrupole
# moment Q that interacts with the electric field gradient (EFG) at the
# donor site.
# ASSUMPTION: Axial symmetry (asymmetry parameter η = 0). The EFG at a
# substitutional donor in bulk silicon has Td symmetry, giving η = 0.
# Strain or nearby interfaces can break this symmetry.
# ASSUMPTION: The EFG principal axis is along the magnetic field (z-axis).
# Misalignment would introduce additional I_x² - I_y² terms.
# ASSUMPTION: The quadrupole coupling is independent of the charge state.
# The EFG is dominated by the crystal lattice, not the bound electron.
# In reality the electron presence at the donor slightly modifies the local
# EFG, but this effect is typically very small and is neglected.
# Value: f_Q ≈ -44.1 kHz, from MATLAB reference simulations of Sb in Si.
f_Q = -44.1e-6  # [GHz] Quadrupole coupling parameter (≈ -44.1 kHz)

# --- Charge qubit parameters ---
# ASSUMPTION: The electron can occupy only two spatial states: the donor
# orbital |d⟩ and the interface orbital |i⟩. Higher-lying orbital states
# at both sites are assumed to be energetically well separated and are
# excluded. See Tosi et al. Supplementary Note 1.
# ASSUMPTION: The tunnel coupling V_t is constant (not gate-voltage
# dependent). In practice, V_t depends exponentially on the donor-interface
# distance and can be tuned via vertical electric fields.
tunnel_coupling = 5.0  # [GHz] Tunnel coupling between donor and interface

# The electric dipole moment p converts gate voltage (detuning ε) to
# energy: E = p × ε. The product p × ε must have units of GHz.
# ASSUMPTION: The dipole moment p is a constant that captures the lever arm
# between gate voltage and charge qubit detuning.
dipole_moment = 5.0  # [GHz per unit detuning] Effective electric dipole moment

# Detuning offset: the detuning value at which the charge qubit is at the
# symmetry point (equal energy for |d⟩ and |i⟩ in the absence of tunneling).
epsilon_0 = 0.0  # [detuning units] Zero-detuning offset


###############################################################################
# OPERATOR DEFINITIONS IN THE FULL 32-DIMENSIONAL HILBERT SPACE
###############################################################################
# Tensor product ordering throughout: charge ⊗ electron_spin ⊗ nuclear_spin
#
# SIGN CONVENTION for charge qubit σ_z eigenstates:
#   σ_z |i⟩ = +1 |i⟩   (electron at interface ↔ σ_z = +1)
#   σ_z |d⟩ = -1 |d⟩   (electron at donor     ↔ σ_z = -1)
#
# Projectors:
#   P_donor     = (I - σ_z) / 2 = |d⟩⟨d|
#   P_interface = (I + σ_z) / 2 = |i⟩⟨i|
#
# At large negative detuning → σ_z ≈ -1 → ground state is |d⟩ (donor).
# At large positive detuning → σ_z ≈ +1 → ground state is |i⟩ (interface).

# --- Identity operators for each subspace ---
Id_c = qeye(2)   # Charge qubit identity (2×2)
Id_e = qeye(2)   # Electron spin identity (2×2)
Id_n = qeye(8)   # Nuclear spin identity (8×8, since 2I+1 = 8 for I = 7/2)
Id_full = tensor(Id_c, Id_e, Id_n)  # Full 32×32 identity

# --- Charge qubit Pauli operators (in the full 32D Hilbert space) ---
sigma_x_c = tensor(sigmax(), Id_e, Id_n)
sigma_y_c = tensor(sigmay(), Id_e, Id_n)
sigma_z_c = tensor(sigmaz(), Id_e, Id_n)

# --- Charge state projectors ---
# P_donor: projects onto |d⟩, the charge state with the electron at the donor
P_donor = (Id_full - sigma_z_c) / 2
# P_interface: projects onto |i⟩, the charge state with the electron at the interface
P_interface = (Id_full + sigma_z_c) / 2

# --- Electron spin-1/2 operators (in the full 32D Hilbert space) ---
# jmat(1/2, 'x') returns the 2×2 spin-1/2 matrix S_x = (1/2) σ_x, etc.
# Eigenvalues of S_z: +1/2 (spin up ↑) and -1/2 (spin down ↓)
Sx = tensor(Id_c, jmat(1/2, 'x'), Id_n)
Sy = tensor(Id_c, jmat(1/2, 'y'), Id_n)
Sz = tensor(Id_c, jmat(1/2, 'z'), Id_n)

# --- Nuclear spin-7/2 operators (in the full 32D Hilbert space) ---
# jmat(7/2, 'x') returns the 8×8 spin-7/2 matrix I_x, etc.
# Eigenvalues of I_z: +7/2, +5/2, +3/2, +1/2, -1/2, -3/2, -5/2, -7/2
# NOTE: No sign flip on nuclear operators. The sign of γ_n in the Zeeman
# Hamiltonian determines the magnetic moment direction. For 123-Sb, γ_n > 0
# (positive nuclear magnetic moment → m_I = +7/2 is lowest Zeeman energy).
Ix = tensor(Id_c, Id_e, jmat(7/2, 'x'))
Iy = tensor(Id_c, Id_e, jmat(7/2, 'y'))
Iz = tensor(Id_c, Id_e, jmat(7/2, 'z'))


###############################################################################
# HAMILTONIAN CONSTRUCTION
###############################################################################
# The total Hamiltonian is (all terms in GHz):
#
#   H = H_charge + H_eZ + H_nZ + H_hf + H_Q
#
# Term          | Physics                    | Typical scale at B=1.38 T
# --------------|----------------------------|---------------------------
# H_charge      | Tunnel coupling + detuning | ~1 GHz (tunable)
# H_eZ          | Electron Zeeman            | ~38.7 GHz (dominant)
# H_nZ          | Nuclear Zeeman (123-Sb)    | ~7.7 MHz
# H_hf          | Hyperfine S·I × P_donor    | ~93 MHz (charge-dependent!)
# H_Q           | Nuclear quadrupole I_z²    | ~44 kHz (smallest term)
#
# The hierarchy γ_e B ≫ A ≫ γ_n B ≫ f_Q means that to first order the
# eigenstates are tensor products of charge, electron spin, and nuclear
# spin states. The hyperfine S·I term couples electron and nuclear spins
# and creates the characteristic multi-line NMR/ESR spectra.


def H_charge_qubit(detuning, V_t=tunnel_coupling, p=dipole_moment, eps0=epsilon_0):
    """
    Charge qubit Hamiltonian: controls the electron's spatial position.

    H_charge = (V_t / 2) σ_x  -  [p (ε - ε_0) / 2] σ_z

    - σ_x term: tunnel coupling that hybridizes |d⟩ and |i⟩
    - σ_z term: detuning (energy difference between donor and interface)

    Limiting cases:
      ε → -∞:  ground state → |d⟩ (electron at donor)
      ε → +∞:  ground state → |i⟩ (electron at interface)
      ε = ε_0: hybridized, charge splitting = V_t

    ASSUMPTION: Two-level approximation is valid — higher orbital states at
    both the donor and interface are energetically well separated.
    """
    return (V_t / 2) * sigma_x_c - (p * (detuning - eps0) / 2) * sigma_z_c


def H_electron_zeeman(B=B0, ge=gamma_e, dg=delta_g):
    """
    Electron Zeeman Hamiltonian with a charge-state-dependent g-factor.

    H_eZ = γ_e B [1 + δg × P_interface] S_z

    The electron Larmor frequency depends on where the electron sits:
      At donor (σ_z = -1):     f_e = γ_e × B           ≈ 38.67 GHz
      At interface (σ_z = +1): f_e = γ_e × (1+δg) × B  ≈ 38.71 GHz

    This difference (≈ 40 MHz for δg = 0.001) is what enables electric-dipole
    spin resonance (EDSR) in the flip-flop qubit: modulating the charge
    state effectively modulates the ESR frequency.

    ASSUMPTION: The g-factor shift δg occurs at the INTERFACE only. The
    donor-site g-factor is the bulk silicon value (reference).
    ASSUMPTION: Only the z-component of the spin couples to B0 (the field
    defines the quantization axis). g-tensor anisotropy is neglected.
    """
    # g-factor operator: I at donor, (I + δg) at interface
    g_operator = Id_full + dg * P_interface
    return ge * B * g_operator * Sz


def H_nuclear_zeeman(B=B0, gn=gamma_n):
    """
    Nuclear Zeeman Hamiltonian.

    H_nZ = -γ_n × B × I_z

    For 123-Sb, γ_n > 0, so:
      E(m_I) = -γ_n B m_I
      m_I = +7/2 has the LOWEST energy (spin aligned with field)

    Level spacing: Δf = γ_n × B ≈ 5.553e-3 × 1.38 ≈ 7.66 MHz

    The 8 nuclear Zeeman levels are equally spaced (in the absence of
    quadrupole and hyperfine interactions). The quadrupole shifts these
    levels unequally (see H_quadrupole), and the hyperfine further
    shifts them when the electron is at the donor.

    ASSUMPTION: The nuclear Zeeman does NOT depend on the charge state.
    The nucleus is always at the donor site regardless of where the
    electron is located.
    ASSUMPTION: Chemical shift and Knight shift corrections are neglected.
    """
    return -gn * B * Iz


def H_hyperfine(A=A0):
    """
    Isotropic Fermi contact hyperfine interaction (charge-state dependent).

    H_hf = A × (S · I) × P_donor
         = A × (S_x I_x + S_y I_y + S_z I_z) × (1 - σ_z^charge) / 2

    Physical picture:
      - The Fermi contact hyperfine coupling A is proportional to the electron
        wavefunction density at the nucleus: A ∝ |ψ_e(r_nuc)|².
      - At the donor: full overlap → A = A0
      - At the interface: negligible overlap → A ≈ 0

    The P_donor projector implements this charge-state dependence:
      At donor (σ_z = -1):     P_donor = 1 → H_hf = A × S·I
      At interface (σ_z = +1): P_donor = 0 → H_hf = 0

    This is THE key mechanism for charge-dependent NMR frequencies:
      NMR frequency ≈ γ_n B + A m_s    (when electron is at donor)
      NMR frequency ≈ γ_n B            (when electron is at interface)
    The shift A m_s ≈ ±46.5 MHz (for m_s = ±1/2, A = 93 MHz) is easily
    resolvable in experiment.

    For the flip-flop qubit, the S·I = S_x I_x + S_y I_y part (the "flip-flop"
    terms) couples |↑,m_I⟩ ↔ |↓,m_I+1⟩ states. Modulating A electrically
    (by moving the electron between donor and interface) drives these
    transitions — this is the EDSR mechanism for the flip-flop qubit.

    ASSUMPTION: Purely isotropic (Fermi contact) hyperfine interaction.
    The anisotropic dipolar contribution is negligible for substitutional
    donors in Si (cubic Td symmetry).
    ASSUMPTION: A = 0 exactly when the electron is at the interface.
    A tiny residual coupling from the exponential tail of the interface
    wavefunction is neglected.
    """
    # Scalar product S · I
    S_dot_I = Sx * Ix + Sy * Iy + Sz * Iz
    # P_donor acts on the charge subspace; S and I act on the spin subspaces.
    # Since they act on different subspaces of the tensor product, all three
    # commute, so the ordering in the product does not matter.
    return A * S_dot_I * P_donor


def H_quadrupole(fq=f_Q):
    """
    Nuclear electric quadrupole interaction.

    H_Q = f_Q × I_z²

    For I = 7/2, the 123-Sb nucleus has a non-zero electric quadrupole
    moment Q. Its interaction with the electric field gradient (EFG) at
    the donor site lifts the equal spacing of the nuclear Zeeman levels.

    Full axially symmetric quadrupole Hamiltonian:
      H_Q = P × [3 I_z² - I(I+1)]
    The constant term -P × I(I+1) is proportional to the identity and shifts
    all levels uniformly — it doesn't affect transition frequencies. The
    factor 3 and other prefactors are absorbed into f_Q.

    Effect on NMR transitions (Δm_I = ±1):
      Without quadrupole: all 7 transitions have equal frequency spacing
      With quadrupole: transition m_I ↔ m_I-1 shifts by f_Q × (2m_I - 1),
      producing 7 unequally-spaced NMR lines. This "fingerprint" pattern
      is characteristic of the quadrupole coupling and can be used to
      identify the donor species and local strain environment.

    ASSUMPTION: Axial symmetry (η = 0). Strain or interfaces can break this.
    ASSUMPTION: EFG principal axis is along the magnetic field (z-axis).
    ASSUMPTION: Quadrupole coupling is charge-state independent.
    """
    return fq * Iz * Iz


def H_total(detuning, B=B0, A=A0, fq=f_Q) -> Qobj:
    """
    Full Hamiltonian of the 123-Sb donor + electron spin + charge qubit system.

    H = H_charge + H_eZ + H_nZ + H_hf + H_Q

    This is a 32×32 Hermitian matrix in units of GHz.

    Parameters
    ----------
    detuning : float
        Charge qubit detuning (gate voltage parameter).
        Negative values favor the electron at the donor.
        Positive values favor the electron at the interface.
    B : float
        External magnetic field [T]. Default: 1.38 T.
    A : float
        Hyperfine coupling constant [GHz]. Default: 0.093 GHz (93 MHz).
    fq : float
        Quadrupole coupling parameter [GHz]. Default: -44.1e-6 GHz (-44.1 kHz).

    Returns
    -------
    H : qutip.Qobj
        32×32 Hermitian operator in units of GHz.

    Energy scale hierarchy at B = 1.38 T:
        Electron Zeeman:  γ_e B  ≈ 38.7   GHz     (dominant)
        Tunnel coupling:  V_t    ≈  1      GHz     (tunable)
        Hyperfine:        A      ≈  0.093  GHz     (93 MHz)
        Nuclear Zeeman:   γ_n B  ≈  0.0077 GHz     (7.7 MHz)
        Quadrupole:       f_Q    ≈ -4.4e-5 GHz     (44 kHz, smallest)
    """
    H = (H_charge_qubit(detuning)
         + H_electron_zeeman(B)
         + H_nuclear_zeeman(B)
         + H_hyperfine(A)
         + H_quadrupole(fq))
    return H