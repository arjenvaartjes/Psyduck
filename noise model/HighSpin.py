import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

"Translated from Thaddeus Ladd's Matlab by AV"
class HighSpin:
    """
    HighSpin simulates the behavior of a high-spin quantum system.

    This class provides methods for constructing and manipulating spin operators, as well as performing various quantum measurements and visualizations.

    Attributes:
    I (float): Spin quantum number.
    id (numpy array): Identity matrix for the spin system.
    mm (numpy array): Array of magnetic quantum numbers.
    x, y, z (numpy array): Spin operator matrices for the x, y, and z directions.
    p, m (numpy array): Raising and lowering operators.
    T0, T1, T2 (numpy array): Tensor operators.
    Qx, Qy, Qphi (numpy array): Grid points for Q function calculations.
    Qmatrix (list): Matrices for Q function calculations.
    Wmatrix (list): Matrices for Wigner function calculations.
    Parity (numpy array): Parity measurement matrix.
    IzExp (numpy array): Expectation value matrix for the z component of spin.

    Methods:
    __init__(Iin): Initializes the high-spin system with a given spin quantum number.
    rho(in_vector): Converts a state vector to a density matrix.
    paritymeas(in_vector, direction): Measures the parity in the specified direction.
    ramseymeas(in_vector): Measures the Ramsey signal.
    polarQ(in_vector): Calculates the polar Q function.
    equatorialQ(in_vector): Calculates the equatorial Q function.
    plotPolarQ(in_vector): Plots the polar Q function.
    plotEquatorialQ(in_vector, *args): Plots the equatorial Q function.
    polarW(in_vector): Calculates the polar Wigner function.
    equatorialW(in_vector): Calculates the equatorial Wigner function.
    plotPolarW(in_vector, *args): Plots the polar Wigner function.
    """
    def __init__(self, Iin):
        """
        Initialize the HighSpin system with the given spin quantum number.

        Parameters:
        Iin (float): Spin quantum number.
        """
        self.I = Iin
        d = int(2 * self.I + 1)
        filename = f'highspin{2*Iin+1}.mat'
        try:
            data = scipy.io.loadmat(filename)
            self.__dict__.update(data['obj'].item().__dict__)
        except:
            self.id = np.eye(d)
            self.mm = np.arange(-Iin, Iin+1)
            self.x = np.diag(np.sqrt(Iin * (Iin + 1) - self.mm[0:int(2*Iin)] * self.mm[1:int(2*Iin+1)]), 1)
            self.y = (self.x - self.x.conj().T) / 2.0j
            self.x = (self.x + self.x.conj().T) / 2.0
            self.z = np.diag(-self.mm)
            self.p = self.x + 1j * self.y
            self.m = self.x - 1j * self.y
            self.T0 = 2 * self.z**2 - self.x**2 - self.y**2
            self.T1 = self.z @ self.x + self.x @ self.z
            self.T2 = self.p**2 + self.m**2

            self.Qx = np.linspace(-1.2, 1.2, 63)
            self.Qy = np.linspace(-1.2, 1.2, 65)
            self.Qphi = np.linspace(0, 2 * np.pi, 99)

            # self.Qmatrix = [np.zeros((len(self.Qx) * len(self.Qy), d**2)), 
            #                 np.zeros((len(self.Qphi), d**2))]
            # self.Wmatrix = [np.zeros((len(self.Qx) * len(self.Qy), d**2)), 
            #                 np.zeros((len(self.Qphi), d**2))]
            
            # rhokq = np.zeros((d, 2 * d + 1, d**2), dtype=complex)
            # print('Building rhokq')
            # for k in range(d):
            #     for q in range(-k, k + 1):
            #         for j in range(d**2):
            #             a, b = divmod(j, d)
            #             m = a - self.I
            #             mp = b - self.I
            #             rhokq[k, q + k, j] = (-1)**(self.I - m - q) * clebschgordan(self.I, m, self.I, mp, k, q)

            # print('Building Qmatrix')
            # for j in range(len(self.Qx)):
            #     for k in range(len(self.Qy)):
            #         theta = np.pi / 2 - np.sqrt(self.Qx[j]**2 + self.Qy[k]**2)
            #         phi = np.arctan2(self.Qx[j], -self.Qy[k])
            #         index = j * len(self.Qy) + k
            #         Ry = expm(-1j * self.y * theta)
            #         ket = expm(-1j * self.z * phi) @ Ry[:, int(2 * self.I)]
            #         self.Qmatrix[0][index, :] = ket @ ket.conj().T.flatten()

            Ry = expm(-1j * self.y * np.pi / 2)
            # for j in range(len(self.Qphi)):
            #     ket = expm(-1j * self.z * self.Qphi[j]) @ Ry[:, 0]
            #     self.Qmatrix[1][j, :] = ket @ ket.conj().T.flatten()

            Nphi = 300
            ParityOp = np.diag((-1)**(self.I - self.mm))
            self.Mphi = np.linspace(0, 2 * np.pi, Nphi)

            self.Parity = np.zeros((Nphi, d**2), dtype=complex)
            self.IzExp = np.zeros((Nphi, d**2), dtype=complex)
            for j in range(Nphi):
                R = expm(1j * np.pi / 2 * (self.x * np.cos(self.Mphi[j]) + self.y * np.sin(self.Mphi[j])))
                RR = np.kron(R.conj(), R)
                self.Parity[j, :] = (ParityOp.flatten() @ RR)
                self.IzExp[j, :] = (self.z.flatten() @ RR)

#             scipy.io.savemat(filename, {'obj': self})

    def rho(self, in_vector):
        """
        Convert a state vector to a density matrix.

        Parameters:
        in_vector (numpy array): State vector or density matrix.

        Returns:
        numpy array: Density matrix.
        
        Raises:
        ValueError: If input vector size is incorrect.
        """
        if len(in_vector) == 2 * self.I + 1:
            return np.outer(in_vector.conj(), in_vector).flatten()
        elif len(in_vector) == (2 * self.I + 1)**2:
            return in_vector.flatten()
        else:
            raise ValueError('Wrong size input for highspin')

    def paritymeas(self, in_vector, direction):
        """
        Measure the parity in the specified direction.

        Parameters:
        in_vector (numpy array): State vector or density matrix.
        direction (str): Measurement direction ('x', 'y', or 'z').

        Returns:
        numpy array: Parity measurement results.
        
        Raises:
        ValueError: If direction is invalid.
        """
        if direction == 'x':
            Ry = expm(-1j * self.y * np.pi / 2)
            self.zaxis = np.kron(Ry.T, Ry.conj().T) @ in_vector
        elif direction == 'y':
            Rx = expm(-1j * self.x * np.pi / 2)
            self.zaxis = np.kron(Rx.T, Rx.conj().T) @ in_vector
        elif direction == 'z':
            self.zaxis = in_vector
        else:
            raise ValueError('highspin.paritymeas needs direction')
        return (self.Parity @ self.zaxis).real

    def ramseymeas(self, in_vector):
        """
        Measure the Ramsey signal.

        Parameters:
        in_vector (numpy array): State vector or density matrix.

        Returns:
        numpy array: Ramsey measurement results.
        """
        return (self.IzExp @ in_vector).real

    def polarQ(self, in_vector):
        return self.Qmatrix[0] @ self.rho(in_vector).reshape(len(self.Qx), len(self.Qy))

    def equatorialQ(self, in_vector):
        return self.Qmatrix[1] @ self.rho(in_vector)

    def plotPolarQ(self, in_vector):
        plt.imshow(self.polarQ(in_vector), extent=[self.Qx[0], self.Qx[-1], self.Qy[0], self.Qy[-1]], origin='lower')
        plt.show()

    def plotEquatorialQ(self, in_vector, *args):
        plt.polar(self.Qphi, self.equatorialQ(in_vector), *args)
        plt.show()

    def polarW(self, in_vector):
        return self.Wmatrix[0] @ self.rho(in_vector).reshape(len(self.Qx), len(self.Qy))

    def equatorialW(self, in_vector):
        return self.Wmatrix[1] @ self.rho(in_vector)

    def plotPolarW(self, in_vector, *args):
        plt.imshow(self.polarW(in_vector), extent=[self.Qx[0], self.Qx[-1], self.Qy[0], self.Qy[-1]], origin='lower')
        plt.show()