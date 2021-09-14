# -- Import some built-in modules
import os, re
from typing import Any, Union, Optional, Callable
from functools import lru_cache

# -- Import some third party modules
import numpy as np
from scipy.optimize import fsolve

def resample(self, data: np.ndarray) -> np.ndarray:
    """ Obtain a bootstrap resample of the dataset provided. The resample 
    is a dataset of equivalent size composed by random rows of the first
    dataset.
    
    --- Parameters
    data: np.array
        Dataset to be resampled. The new dataset will be composed by
        random rows extracted from this dataset.

    --- Returns
    np.ndarray
        Numpy array containing the resampled dataset
    """

    # -- Generate a numpy random engine
    rng = np.random.default_rng()

    # -- Return some random indices from the data
    return data[rng.integers(low = 0, high = data.shape[0], size = data.shape[0])]

# -- Main class definition {{{
class Hadron:
    """ Class to manage hadronic correlation functions """

    def __init__(self, init: Union[str, np.ndarray], Nk: int, name: str = ''):

        # -- Select the initialiser depending on the type
        if isinstance(init, str):
            # Assert the path exists and then load the data
            assert os.path.exists(init), f'{init = } does not exist in system'

            # Load the data from the file
            self.__data = np.loadtxt(init)
        else:
            # Assert the data has only one dimension
            assert init.ndim == 1, f'{init.ndim = }, but ndim must be one'

            # Save the data from the numpy array
            self.__data = init

        # Reshape the data to the correct dimensions
        self.__data = np.reshape(self.__data, (self.__data.shape[0] // Nk, Nk))

        # -- Save the number of time points in the Hadron
        self.__info = {'Nk': Nk, 'name': name, 'Nc': self.__data.shape[0]}

    # -- Magic methods of the class {{{
    def __str__(self) -> str:
        return f'<Hadron: {self.name} Nk = {self.Nk}>'

    def __repr__(self) -> str:
        return f'<Hadron: {self.name} Nk = {self.Nk}>'

    def __add__(self, other):
        """ The sum of two hadrons is another hadron with concatenated data. """

        # Assert the time points are the same for both hadrons
        assert self.Nk == other.Nk, \
            f'{self.Nk = } != {other.Nk = }. Both non-integrated directions ' + \
            'must have the same dimension'

        # Concatenate the data contiguously
        data = np.concatenate((self.data.flatten(), other.data.flatten()))

        return Hadron(data, self.Nk, self.name + other.name)
    # -- }}}

    # -- Some important statistics methods of the class {{{
    @lru_cache(maxsize = 1)
    def central_value(self, folded: bool = False, normalised: bool = False) -> np.ndarray:
        r""" Central value of the correlation function, that is, the sample mean at
        each different nk \in [0, Nk]. The result can be folded and normalised.

        --- Parameters
        folded: bool
            Fold the correlation function by its middle point. Can enhance the stability
            in the fits by adding information. The fold is done by compute the mean value
            of C(nk) and C(Nk - nk) for all samples.
        normalised: bool
            Normalise the correlation function by its middle point. Ensures that the
            middle point Nk // 2 value is 1.

        --- Returns
        np.ndarray
            Array of Nk or Nk // 2 + 1 values corresponding to the central values of each
            nk.
        """
        return np.mean(self.fold_and_normalise(folded, normalised), axis = 0)

    @lru_cache(maxsize = 1)
    def standard_error(self, folded: bool = False, normalised: bool = False, bootstrap: Optional[int] = None) -> np.ndarray:
        r""" Standard errors of the central values of the correaltion function at
        each different nk \in [0, Nk]. The result can be folded and normalised.
        The standard error can be computed using bootstrap or the standard formula
        of the sample variance of the sample mean.

        --- Parameters
        folded: bool
            Fold the correlation function by its middle point. Can enhance the stability
            in the fits by adding information. The fold is done by compute the mean value
            of C(nk) and C(Nk - nk) for all samples.
        normalised: bool
            Normalise the correlation function by its middle point. Ensures that the
            middle point Nk // 2 value is 1.
        bootstrap: int
            Use the given bootstrap iterations to compute the standard errors.

        --- Returns
        np.ndarray
            Array of Nk or Nk // 2 + 1 values corresponding to the standard errors of each
            nk.
        """

        # -- Generate the correct version of the data to be usedc
        data = self.fold_and_normalise(folded, normalised)

        # -- If bootstrap is not passed, then use the standard error formula
        if not bootstrap:
            return np.std(data, axis = 0, ddof = 1) / np.sqrt(self.Nc)
        else:

            # -- Allocate a buffer for the bootstrap estimates of the mean
            bbuffer = np.empty([bootstrap, data.shape[1]])

            # -- Iterate to calculate the bootstrap estimates
            for nb in range(bootstrap):
                bbufer[nb,:] = np.mean(resample(data), axis = 0)

            return np.std(bbuffer, axis = 0, ddof = 1)

    @lru_cache(maxsize = 1)
    def correlation_matrix(self, folded: bool = False, normalised: bool = False) -> np.ndarray:
        """ Correlation matrix of the central values of the correlation function data.
        The correlation matrix is a matrix of [Nk, Nk] value (Nk//2 if folded is true).

        --- Parameters
        folded: bool
            Fold the correlation function by its middle point. Can enhance the stability
            in the fits by adding information. The fold is done by compute the mean value
            of C(nk) and C(Nk - nk) for all samples.
        normalised: bool
            Normalise the correlation function by its middle point. Ensures that the
            middle point Nk // 2 value is 1.

        --- Returns
        np.ndarray
            2-dimensional array of Nk or Nk // 2 values corresponding to the correlation
            matrix.
        """
        return np.corrcoef(self.fold_and_normalise(folded, normalised).T)

    @lru_cache(maxsize = 1)
    def covariance_matrix(self, folded: bool = False, normalised: bool = False) -> np.ndarray:
        """ Covariance matrix of the central value of the correlation function data.
        The covariance matrix is a matrix of [Nk, Nk] value (Nk//2 if folded is true).

        The diagonal elements are the variance of the central value data, thus, they are
        equivalent to the square of the result of .standard_error()

        --- Parameters
        folded: bool
            Fold the correlation function by its middle point. Can enhance the stability
            in the fits by adding information. The fold is done by compute the mean value
            of C(nk) and C(Nk - nk) for all samples.
        normalised: bool
            Normalise the correlation function by its middle point. Ensures that the
            middle point Nk // 2 value is 1.

        --- Returns
        np.ndarray
            2-dimensional array of Nk or Nk // 2 values corresponding to the covariance
            matrix.
        """
        return np.cov(self.fold_and_normalise(folded, normalised).T) / self.Nc

    # -- Utility methods of the class {{{
    def set_info(self, **kwargs) -> dict:
        """ Set some information into the information dictionary of the object

        --- Parameters
        **kwargs: dict
            Key and value to be stored in the information dictionary.

        --- Returns
        dict:
            Dictionary of stored information in the class
        """

        # -- Add new keys and values to the dictionary
        for key, value in kwargs.items():
            self.__info[key] = value

        return self.__info

    def del_info(self, key: Any) -> None:
        """ Delete an element in the information dictionary of the object

        --- Parameters
        key: Any
            Key to be removed from the dictionary.

        --- Returns
        None
        """
        # Do not delete these keys
        do_not_del = ['name', 'Nk', 'Nc']

        # Assert the deleted information is not name, Nk or Nc
        assert key not in do_not_del, f'{key = } should not be in {do_not_del}'

        # Pop the key in the dictionary of information
        self.__info.pop(key, None)

    @lru_cache(maxsize = 2)
    def nk(self, folded: bool = False) -> np.ndarray:
        """ Vector of possible variables (times) in the correlation function.

        --- Parameters
        folded: bool
            Fold the correlation function by its middle point. Can enhance the stability
            in the fits by adding information. The fold is done by compute the mean value
            of C(nk) and C(Nk - nk) for all samples.

        --- Returns
        np.array
            Vector of possibles times available in the correlation function
        """
        return np.arange(self.Nk // 2 + 1) if folded else np.arange(self.Nk)

    @lru_cache(maxsize = 2)
    def fold_and_normalise(self, folded: bool, normalised: bool) -> np.ndarray:
        """ Fold and normalise the correlation function. 

        Fold computes the middle point between C(nk) and C(Nk - nk) for all samples.
        It is only useful if the correlation functions exhibits cosh-like behaviour.
        Could enhance the stability of the fits.

        Normalise means that all values are divided by the value of the correlation
        function at the middle of the lattice (Nk // 2). The normalisation is carried
        out using the central value at the middle of the lattice. This implies that
        the central value standard error at Nk // 2 does not have standard error of
        0, which is undesirable.
        
        --- Parameters
        folded: bool
            Fold the correlation function by its middle point. Can enhance the stability
            in the fits by adding information. The fold is done by compute the mean value
            of C(nk) and C(Nk - nk) for all samples.
        normalised: bool
            Normalise the correlation function by its middle point. Ensures that the
            middle point Nk // 2 value is 1.

        --- Returns
        np.ndarray
            Folded and normalised version of the correlation function dataset.
        """

        # First, fold the data if needed
        data = self.__data if not folded else self.__fold()

        # Then normalise the data
        data = data if not normalised else self.__normalise(data)

        return data

    def __fold(self) -> np.ndarray:
        """ Fold the correlation function using the middle point of the correlation
        function. For more information: read .fold_and_normalise()
        """

        # Buffer to save the folded data
        folded = np.empty((self.Nc, self.Nk // 2 + 1))

        # Generate the folded correlator
        for nt in range(self.Nk // 2 + 1):
            if nt == 0:
                folded[:,nt] = self.__data[:,nt]
            else:
                folded[:,nt] = 0.5 * (self.__data[:,nt] + self.__data[:,self.Nk - nt])

        return folded

    def __normalise(self, data: np.ndarray) -> np.ndarray:
        """ Normalise the correlation function using the middle point of the correlation
        function. For more information: read .fold_and_normalise()
        """
        return data / np.mean(data, axis = 0)[self.Nk // 2]

    # -- }}}

    # -- Property methods of the class {{{
    @property
    def data(self) -> np.ndarray: 
        """ Dataset defining the hadron """
        return self.__data

    @property
    def Nk(self) -> int: 
        """ Number of points in the non-integrated direction. For thermal correlations
        functions, it corresponds to Nt
        """
        return self.__info['Nk']

    @property
    def Nc(self) -> int:
        """ Number of configurations/ensembles contained in the hadron's dataset """
        return self.__data.shape[0]

    @property
    def name(self) -> str:
        """ Name identifier of the hadron. """
        return self.__info['name']

    @name.setter
    def name(self, name: str) -> None:
        """ Setter the name identifier of the hadron """
        self.__info['name'] = name

    @property
    def info(self) -> dict:
        """ Dictionary of information of the hadron """
        return self.__info
    # -- }}}
# -- }}}

# -- Create an ansatz type
Ansatz = Callable[[float, int, Hadron], np.ndarray]

# -- Create some ansatz functions {{{
def cosh_ansatz(M: float, t: int, hadron: Hadron) -> np.ndarray:
    """ Cosh ansatz of an antiperiodic correlation function """
    return np.cosh(M * (t - 0.5 * hadron.Nk))

def exp_ansatz(M: float, t: int, hadron: Hadron) -> np.ndarray:
    """ Exponential ansatz of a correlation function """
    return np.exp(- M * t)
# -- }}}

# -- Define the effective mass method of the class {{{
def __effective_mass(self, t0: int, tf: int, ansatz: Ansatz = cosh_ansatz, folded: bool = False) -> np.ndarray:
    """ Compute the effective mass of the Hadron by solving the following
    equation,

        ansatz(M, t, self) / ansatz(M, t+1, self) = corr[t] / corr[t+1],

    where M is the mass of the state, t is the non-integrated variable (nk)
    and self is a pointer to the same object, which can be used to alter the
    ansatz internally: for example, by using Nk.

    --- Parameters
    t0: int
        Initial non-integrated variable (nk) at which the equation above will
        be solved.
    tf: int
        Final non-integrated variable (nk) at which the equation above will
        be solved.
    ansatz: Ansatz
        Ansatz (function with signature [M: float, t: int, hadron: Hadron]) used
        to solve the effective mass equation.
    folded: bool
        Flag used to decide whether the correlation function should be folded
        or not.
    """

    # The initial time must be positive definite
    assert t0 >= 0, f'{t0 = } must be positive definite'

    # Assert some conditions on tf depending on folded
    if folded:
        assert t0 < tf <= self.Nk // 2 + 1, f'{tf = } must hold {t0 = } < {tf = } <= {self.Nk // 2 + 1}'
    else:
        assert t0 < tf, f'{tf = } must be smaller than {t0 = }'

    # Obtain the central value using folded as argument
    corr = self.central_value(folded = folded)

    # Function used to solve the equation
    to_solve = lambda M, t, hadron: ansatz(M, t, hadron) / ansatz(M, t+1, hadron) - corr[t] / corr[t+1]

    # List containing all effective masses
    eff_masses = []

    # Iterate to compute the effective masses
    for t in range(t0, tf):
        eff_masses.append(fsolve(to_solve, 0.1, args = (t, self))[0])

    return np.array(eff_masses)

# -- Set the correct effective mass method in the Hadron class
Hadron.effective_mass = __effective_mass
# -- }}}

if __name__ == '__main__':
    pass
