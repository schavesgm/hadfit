# -- Import some built-in modules
import operator
from functools import lru_cache
from typing import Callable, Union, Iterable, Optional

# -- Import some third party modules
import numpy as np
import sympy as sy
import lmfit as lm

# -- Import some user defined shared definitions
from .shared_defs import set_parameters
from .shared_defs import show_jacobian
from .shared_defs import eval_jacobian
from .shared_defs import guess

class CompositeModel(lm.CompositeModel):
    """ Custom implementation of the Composite model class """

    def __init__(self, left, right, op, **kwargs):
        super().__init__(left, right, op, **kwargs)

    def fit(
        self, data: np.ndarray, inv_cov: Optional[np.ndarray] = None, 
        params: Optional[lm.Parameters] = None, weights: Optional[Iterable] = None, 
        method: Optional[str] = 'leastsq', iter_cb: Optional[Callable] = None, 
        scale_covar: Optional[bool] = True, verbose: Optional[bool] = False, 
        fit_kws: Optional[dict] = None, nan_policy: Optional[str] = None, 
        calc_covar: Optional[bool] = True, max_nfev: Optional[int] = None, **kwargs
        ):
        """
            Overwritten fit method for this instance of the lmfit.Model class.
            It contains a new parameter called inv_cov; inv_cov is the inverse
            of the covariance data of the error term in the predictors. To 
            understand inv_cov, we can start with a simple linear model with
            one regressor,

                                    Yi = A * Xi + ui.

            If we assume E(ui | Xi) = 0, then the variance if E(ui^2) = \sigma_i.
            Assuming we have N different correlated Yi with heterogeneous variances
            (\sigma_i != \sigma_j), then inv_cov is our estimate of the covariance
            of the error term, that is, the inverse of E(u * u^T) = \Sigma * R * \Sigma;
            \Sigma is diag(\sigma_1, \sigma_2, ..., \sigma_N) and R is the matrix
            of correlation coefficients.

            Long story short, if \Sigma is not known, then pass the inverse of 
            the covariance matrix of the measured Y.

            When the inverse of the covariance matrix estimation is passed, problem is
            solved as a maximum likelihood estimate of N correlated normally distributed
            random variables. As a result, the function to minimise is

            L(\theta, y) \sim  exp(-0.5 * (y - f(x; \theta))^T \Sigma^{-1} (y - f(x; \theta)).
        """

        # If we pass the inverse covariance, then use it
        if inv_cov is not None:
            # Assert the inverse covariance has the same size of data
            assert inv_cov.shape == (data.size, data.size), \
                f'{inv_cov.shape = } dimensions must be equal to {data.size = }'

            # Create the callback inner function
            def callback_inner(params, iter, resid, *args, **kws):
                ''' Inner callback function to modify resid before minimisation '''
                resid[:] = resid @ inv_cov * resid

            def reduce_fcn(resid):
                return resid.sum()
        else:
            # Create the callback inner function
            def callback_inner(params, iter, resid, *args, **kws):
                ''' Inner callback function to modify resid before minimisation '''
                return

            def reduce_fcn(resid):
                return

        # Use the correct reduce function in case no fit_kws are passed
        fit_kws = {'reduce_fcn': reduce_fcn} if fit_kws is None else fit_kws

        # Fit the data using the callback function and the parent fit
        results = super(Model, self).fit(
            data, params = params, weights = weights, method = method, 
            iter_cb = callback_inner, scale_covar = scale_covar, 
            verbose = verbose, fit_kws = fit_kws, 
            nan_policy = nan_policy, calc_covar = calc_covar, max_nfev = max_nfev, 
            **kwargs
        )

        # Compute the errors using the jacobian of the model
        if inv_cov is not None:

            # Evaluate the jacobian at the best parameters
            jac = self.eval_jacobian(results.params, **kwargs)

            # If no singular matrix is obtained, use the jacobian for errors
            try:
                # Calculate the covariance matrix using the errors
                results.covar = np.linalg.inv(jac @ inv_cov @ jac.T)

                # Calculate the standard errors of the parameters
                for p, param in enumerate(results.params.values()):
                    param.stderr = np.sqrt(np.abs(results.covar[p,p]))

            except np.linalg.LinAlgError:
                pass

        return results

    # -- Private methods of the class {{{
    @lru_cache(maxsize = 1)
    def __gen_function(self, prefix: str) -> np.ufunc:
        """ Generate a callable function from the model. """
        return sy.lambdify(self.symb_regressors + self.symb_parameters, self.expr, 'numpy')

    @lru_cache(maxsize = 1)
    def __gen_jacobian(self, prefix: str) -> list[np.ufunc]:
        """ Generate the jacobian of the model with respect to the parameters. """
        # Compute the Jacobian of the expression
        jacobian = sy.Matrix([self.expr]).jacobian(self.symb_parameters)

        # Generate the lambdify version of the jacobian
        return [sy.lambdify(self.symbols, j, "numpy") for j in jacobian]
    # -- }}}

    # -- Attribute methods of the class {{{
    @property
    def symb_regressors(self) -> tuple[sy.Symbol]:
        """ Symbolic regressors/independent variables of the model. """
        if self.left.symb_regressors == self.right.symb_regressors:
            return self.left.symb_regressors
        else:
            return self.left.symb_regressors + self.right.symb_regressors

    @property
    def symb_parameters(self) -> tuple[sy.Symbol]:
        """ Symbolic parameters in the model. It takes into account the prefix of the
        model. 
        """
        # Get the parameters for the model in the left and the right
        return self.left.symb_parameters + self.right.symb_parameters

    @property
    def symbols(self) -> tuple[sy.Symbol]:
        """ Tuple containing all symbols in the model """
        return self.symb_regressors + self.symb_parameters

    @property
    def params(self) -> lm.Parameters:
        """ Parameters on which the model depends """
        return self.make_params()

    @property
    def expr(self) -> sy.core.Basic:
        """ Mathematical definition of the model """
        return self.left.expr + self.right.expr

    @property
    def function(self) -> np.ufunc:
        """ Return the correct function of the model using prefix. The function
        can be evaluated using arguments or kwargs with the names of the parameters
        taking into account the prefix of the model. 

        Example: Model('A * x + B * y + C', 'x, y', 'A, B, C')
            If we define the model with expression A * x + B * y + C and we set its
            prefix to model.prefix = 'je', then we can evaluate its function using
            either:

            1. model.function(x_vals, y_vals, A_val, B_val, C_val)
            2. model.function(x = x_vals, y = y_vals, jeA = A_val, jeB = B_val, jeC = C_val)

            and any other combination using *args and **kwargs.
        """
        return self.__gen_function(self.prefix)

    @property
    def jacobian(self) -> list[np.ufunc]:
        """ Return the jacobian of the model with respect to the parameters. The
        result is a list of functions that contains the derivatives of the function
        with respect to the parameters. """
        return self.__gen_jacobian(self.prefix)
    # -- }}}

    # -- Magic methods of the class {{{
    def __str__(self) -> str:
        return f'<CompositeModel: {self.expr}; x = {self.symb_regressors}, p = {self.symb_parameters}>'

    def __repr__(self) -> str:
        return f'<CompositeModel: {self.expr}; x = {self.symb_regressors}, p = {self.symb_parameters}>'

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """ Evaluate the model using args or kwargs as parameters. Information about how 
        to evaluate the model can be obtained in the documention of model.function().
        """

        # List of parameter strings
        str_param = [str(p) for p in self.symb_parameters]

        # List of keys in dictionary
        key_list = list(kwargs.keys())

        # If keyword arguments are passed, try cleaning them
        if kwargs and self.prefix != '':

            # Iterate through all keys in the dictionary
            for key in key_list:
                if f'{self.prefix}{key}' in str_param:
                    kwargs[f'{self.prefix}{key}'] = kwargs.pop(key)
        return self.function(*args, **kwargs)

    def __add__(self, other):
        return CompositeModel(self, other, operator.add)

    def __sub__(self, other):
        return CompositeModel(self, other, operator.sub)

    def __mul__(self, other):
        return CompositeModel(self, other, operator.mul)

    def __truediv__(self, other):
        return CompositeModel(self, other, operator.truediv)
    # -- }}}

# -- Set some shared methods in the class {{{
CompositeModel.set_parameters  = set_parameters
CompositeModel.show_jacobian   = show_jacobian
CompositeModel.eval_jacobian   = eval_jacobian
CompositeModel.guess           = guess
# -- }}}

if __name__ == '__main__':
    pass
