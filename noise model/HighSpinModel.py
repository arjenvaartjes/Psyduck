import numpy as np
from scipy.linalg import expm

"""Freely translated from Thaddeus Ladd's Matlab script

~Arjen Vaartjes"""
class HighSpinModel:
    """
    HighSpinModel simulates the dynamics of a high-spin system under the influence of magnetic and electric noise.

    This class provides methods for modeling Ramsey experiments and other pulse sequences, accounting for 
    different noise sources characterized by T2 and T1 times and their corresponding exponents.

    Attributes:
    T2Mstar (float): Characteristic T2* time for magnetic noise.
    alphaM (float): Exponent for magnetic noise decay.
    T2Estar (float): Characteristic T2* time for electric noise.
    alphaE (float): Exponent for electric noise decay.
    T2M (float): T2 time for magnetic noise.
    T2E (float): T2 time for electric noise.
    T1 (float): T1 relaxation time.
    rot (numpy array): Rotation matrix for the Hamiltonian.
    Msop (numpy array): Matrix representing the second-order magnetic noise operator.
    Esop (numpy array): Matrix representing the second-order electric noise operator.

    Methods:
    __init__(I, params=None): Initializes the model with given spin I and optional parameters.
    F(ket, decays): Calculates the fidelity of a state undergoing decays.
    Ramsey(t, omega): Simulates the Ramsey experiment for a given time and frequency.
    Pulse_with_loss(t, omega, Hsop): Simulates a pulse sequence with loss for a given time, frequency, and Hamiltonian operator.
    Hahn(I, rho, t): Simulates a Hahn echo sequence for a given spin operator, initial state, and time array.
    """
    def __init__(self, I, params=None):
        """
        Initialize the HighSpinModel with the given spin operator and optional parameters.

        Parameters:
        I (object): Spin operator object containing necessary matrices.
        params (dict, optional): Dictionary of optional parameters to override default values.
                                  Keys include 'T2Mstar', 'T2Estar', 'alphaM', 'alphaE', 'T2M', 'T2E', 'T1'.
        """
        self.T2Mstar = 80  # based on Fig. (a)
        self.alphaM = 2
        self.T2Estar = 2000  # > T2M
        self.alphaE = 1
        self.T2M = 1000
        self.T2E = 1  # = T2Estar
        self.T1 = 10000
        
        self.rot = -1j * np.diag(np.kron(I.z, I.id) - np.kron(I.id, I.z))  #only take diagonal of the liouvillian (saves time)
        self.Msop = np.diag(np.kron(I.z**2, I.id) + np.kron(I.id, I.z**2) - 2 * np.kron(I.z, I.z)) / 2
        self.Esop = np.diag(np.kron(I.z**4, I.id) + np.kron(I.id, I.z**4) - 2 * np.kron(I.z**2, I.z**2)) / 2
        
        if params:
            self.T2Mstar = params.get('T2Mstar', self.T2Mstar)
            self.T2Estar = params.get('T2Estar', self.T2Estar)
            self.alphaM = params.get('alphaM', self.alphaM)
            self.alphaE = params.get('alphaE', self.alphaE)
            self.T2M = params.get('T2M', self.T2M)
            self.T2E = params.get('T2E', self.T2E)
            self.T1 = params.get('T1', self.T1)

    @staticmethod
    def F(ket, decays):
        """
        Calculate the fidelity of a state undergoing decays.

        Parameters:
        ket (numpy array): State vector (ket).
        decays (numpy array): Decay matrix.

        Returns:
        float: Calculated fidelity.
        """
        rho = np.reshape(np.outer(ket, np.conj(ket)), (-1, 1))
        out = np.dot(np.conj(rho.T), np.multiply(rho, decays))
        return out

    def Ramsey(self, t, omega):
        """
        Simulate the Ramsey experiment for a given time and frequency.

        Parameters:
        t (numpy array): Time array.
        omega (float): Frequency of the oscillations.

        Returns:
        numpy array: Simulated Ramsey experiment results.
        """
        out = np.exp(
            self.rot[:, np.newaxis] * t * omega
            - self.Msop[:, np.newaxis] * (t / self.T2Mstar) ** self.alphaM
            - self.Esop[:, np.newaxis] * (t / self.T2Estar) ** self.alphaE
        )
        return out
    
    def Pulse_with_loss(self, t, omega, Hsop):
        """
        Simulate a pulse sequence with loss for a given time, frequency, and Hamiltonian operator.

        Parameters:
        t (numpy array): Time array.
        omega (float): Frequency of the oscillations.
        Hsop (numpy array): Hamiltonian operator matrix.

        Returns:
        numpy array: Simulated pulse sequence results.
        
        Under construction
        """
        out = np.exp(
            Hsop[:, np.newaxis] * t * omega
            - self.Msop[:, np.newaxis] * (t / self.T2Mstar) ** self.alphaM
            - self.Esop[:, np.newaxis] * (t / self.T2Estar) ** self.alphaE
        )
        return out

    def Hahn(self, I, rho, t):
        """
        Simulate a Hahn echo sequence for a given spin operator, initial state, and time array.

        Parameters:
        I (object): Spin operator object containing necessary matrices.
        rho (numpy array): Initial density matrix.
        t (numpy array): Time array.

        Returns:
        numpy array: Simulated Hahn echo sequence results.
        """
        t = np.array(t)
        
        if self.alphaE == 1:
            T2alphaE = 1
        else:
            T2alphaE = 1 + self.alphaE
        
        if self.alphaM == 1:
            T2alphaM = 1
        else:
            T2alphaM = 1 + self.alphaM
        
        out = (np.dot(self.Msop, rho)) * (t / self.T2M) ** T2alphaM + \
              (np.dot(self.Esop, rho)) * (t / self.T2E) ** T2alphaE
        return out