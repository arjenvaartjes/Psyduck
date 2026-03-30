import numpy as np


# Voigt notation order: [xx, yy, zz, yz, xz, xy] (indices 0–5)


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
