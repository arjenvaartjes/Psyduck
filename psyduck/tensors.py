import numpy as np

from psyduck.operations import euler_rotation


# Voigt notation order: [xx, yy, zz, yz, xz, xy] (indices 0–5)


def get_Q_tensor(f_q: float, eta: float = 0.0,
                 theta: float = 0.0, phi: float = 0.0, psi: float = 0.0) -> np.ndarray:
    """Quadrupole coupling tensor Q_ab in the lab frame (Hz).

    Constructs the traceless EFG tensor in its principal axis frame (PAF)
    from the quadrupole splitting frequency and asymmetry parameter, then
    rotates it into the lab frame via ZYZ Euler angles.

    In the PAF: V = diag([-(1-eta)/2, -(1+eta)/2, 1]) * V_zz (normalised).
    Q_ab = (f_q / 3) * R @ V_PAF @ R^T

    :param f_q: Quadrupole splitting frequency (Hz).
    :param eta: Asymmetry parameter, 0 (axial) to 1 (fully asymmetric).
    :param theta: Polar tilt of PAF z-axis from B0 (rad, ZYZ Euler angle).
    :param phi: Azimuthal angle of PAF z-axis (rad, ZYZ Euler angle).
    :param psi: Twist around PAF z-axis (rad, ZYZ Euler angle).
    :return: 3×3 quadrupole coupling tensor in Hz.
    """
    V_PAF = np.diag([-(1 - eta) / 2, -(1 + eta) / 2, 1.0])
    R = euler_rotation(phi, theta, psi)
    V_lab = R @ V_PAF @ R.T
    return (f_q / 3.0) * V_lab


def get_S_tensor(S11: float = 2e22, S44: float = 5.9e22) -> np.ndarray:
    """6×6 piezospectroscopic tensor (Si, (110) crystal orientation).

    Maps strain Voigt vector [xx, yy, zz, yz, xz, xy] to EFG Voigt vector
    [xx, yy, zz, yz, xz, xy] (units: V/m per unit strain).

    Parameters
    ----------
    S11 : float
        Piezospectroscopic constant S11 (V/m). Default: 2e22.
    S44 : float
        Piezospectroscopic constant S44 (V/m). Default: 5.9e22.
    """
    return np.array([
        [S11/4 + S44, -S11/2,       S11/4 - S44, 0,      0,          0     ],
        [-S11/2,       S11,         -S11/2,       0,      0,          0     ],
        [S11/4 - S44, -S11/2,       S11/4 + S44, 0,      0,          0     ],
        [0,            0,            0,           2*S44,  0,          0     ],
        [0,            0,            0,           0,      3*S11/2,    0     ],
        [0,            0,            0,           0,      0,          2*S44 ],
    ])


def get_R_tensor(R14: float = 1.7e12) -> np.ndarray:
    """6×3 piezoelectric tensor (Si, (110) crystal orientation).

    Maps electric field vector [Ex, Ey, Ez] to EFG Voigt vector
    [xx, yy, zz, yz, xz, xy] (units: V/m per V/m = dimensionless coupling).

    Parameters
    ----------
    R14 : float
        Piezoelectric constant R14 (V/m²). Default: 1.7e12.
    """
    return np.array([
        [ 0,    -R14,  0   ],
        [ 0,     0,    0   ],
        [ 0,     R14,  0   ],
        [ 0,     0,    R14 ],
        [ 0,     0,    0   ],
        [-R14,   0,    0   ],
    ])


def voigt_to_tensor(vec: np.ndarray) -> np.ndarray:
    """Convert Voigt vector(s) to symmetric 3×3 tensor(s).

    Voigt order: [xx, yy, zz, yz, xz, xy].

    Parameters
    ----------
    vec : array_like, shape (6,) or (N, 6)

    Returns
    -------
    np.ndarray, shape (3, 3) or (N, 3, 3)
    """
    vec = np.asarray(vec)
    batched = vec.ndim == 2
    if not batched:
        vec = vec[np.newaxis]

    N = len(vec)
    T = np.zeros((N, 3, 3))
    T[:, 0, 0] = vec[:, 0]  # xx
    T[:, 1, 1] = vec[:, 1]  # yy
    T[:, 2, 2] = vec[:, 2]  # zz
    T[:, 1, 2] = T[:, 2, 1] = vec[:, 3]  # yz
    T[:, 0, 2] = T[:, 2, 0] = vec[:, 4]  # xz
    T[:, 0, 1] = T[:, 1, 0] = vec[:, 5]  # xy

    return T if batched else T[0]


def Vab_to_Qab(V_ab: np.ndarray, I: float, Q: float,
               e: float = 1.6e-19, h: float = 6.626e-34) -> np.ndarray:
    """Convert EFG tensor V_ab to quadrupole coupling tensor Q_ab (Hz).

    Q_ab = e * Q * V_ab / (2I(2I-1) * h)

    Parameters
    ----------
    V_ab : array_like, shape (3, 3) or (N, 3, 3)
        Electric field gradient tensor in SI units (V/m²).
    I : float
        Nuclear spin quantum number.
    Q : float
        Nuclear quadrupole moment (C·m²).
    e : float
        Elementary charge (default 1.6e-19 C).
    h : float
        Planck constant (default 6.626e-34 J·s).

    Returns
    -------
    np.ndarray, shape (3, 3) or (N, 3, 3), units Hz
    """
    scale = e * Q / (2 * I * (2 * I - 1) * h)
    return np.asarray(V_ab) * scale


def Qab_to_Vab(Q_ab: np.ndarray, I: float, Q: float,
               e: float = 1.6e-19, h: float = 6.626e-34) -> np.ndarray:
    """Convert quadrupole coupling tensor Q_ab (Hz) to EFG tensor V_ab (V/m²).

    V_ab = Q_ab * 2I(2I-1) * h / (e * Q)

    Parameters
    ----------
    Q_ab : array_like, shape (3, 3) or (N, 3, 3)
        Quadrupole coupling tensor in Hz.
    I : float
        Nuclear spin quantum number.
    Q : float
        Nuclear quadrupole moment (C·m²).
    e : float
        Elementary charge (default 1.6e-19 C).
    h : float
        Planck constant (default 6.626e-34 J·s).

    Returns
    -------
    np.ndarray, shape (3, 3) or (N, 3, 3), units V/m²
    """
    scale = e * Q / (2 * I * (2 * I - 1) * h)
    return np.asarray(Q_ab) / scale


def tensor_to_voigt(tensor: np.ndarray) -> np.ndarray:
    """Convert symmetric 3×3 tensor(s) to Voigt vector(s).

    Voigt order: [xx, yy, zz, yz, xz, xy].

    Parameters
    ----------
    tensor : array_like, shape (3, 3) or (N, 3, 3)

    Returns
    -------
    np.ndarray, shape (6,) or (N, 6)
    """
    tensor = np.asarray(tensor)
    batched = tensor.ndim == 3
    if not batched:
        tensor = tensor[np.newaxis]

    vec = np.stack([
        tensor[:, 0, 0],  # xx
        tensor[:, 1, 1],  # yy
        tensor[:, 2, 2],  # zz
        tensor[:, 1, 2],  # yz
        tensor[:, 0, 2],  # xz
        tensor[:, 0, 1],  # xy
    ], axis=1)

    return vec if batched else vec[0]
