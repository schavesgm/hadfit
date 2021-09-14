# -- Shared defintions for Model and CompositeModel

# -- Import some built-in modules
from typing    import Optional
from typing    import Union
from typing    import Iterable
from typing    import Callable

# -- Import some third party modules
import sympy as sy
import numpy as np
import lmfit as lm

def set_parameters(self, params: lm.Parameters = None, **kwargs) -> None:
    """ Set the parameters in the model to given values and properties
    using a Parameters object or dictionary of properties.
    """
    
    # If params is passed, then use it to set the parameters
    if params is not None:

        # Iterate for all names in the parameters
        for name in params:
            if name in str(self.symb_parameters):
                param = params[name]
                self.set_param_hint(name, value = param.value, min = param.min, max = param.max)
    # If not, set the hints using the dictionary
    else:
        for p_name, p_hint in kwargs.items():
            self.set_param_hint(p_name, **p_hint)

def show_jacobian(self) -> sy.Matrix:
    """ Show the Jacobian of the model as a mathematical expression """
    return sy.Matrix([self.expr]).jacobian(self.symb_parameters)

def eval_jacobian(self, params: Optional[lm.Parameters] = None, *args, **kwargs) -> np.ndarray:
    """ Evalulate the jacobian of the model at the correct parameters. """
    
    # If the parameters are provided, then get their values
    if params: kwargs = kwargs | {p.name : p.value for p in params.values()}

    # If the parameters are provided, then evaluate using their values
    jeval =  [jac(*args, **kwargs) for jac in self.jacobian]

    # Take into account the case in which one value is constant
    max_size = max([j.size for j in jeval if not (isinstance(j, int) or isinstance(j, float))])

    # Change the constant values multiplying them by the np.ones(max_size)
    return np.array([j * np.ones(max_size) for j in jeval])

def guess(self, data: np.ndarray, init_vals: Union[lm.Parameters, float] = 1.0, **kwargs) -> lm.Parameters:
    """ Guess the parameters of the model given some data. """

    # Check if regressors where passed in **kwargs
    assert all([str(reg) in kwargs.keys() for reg in self.symb_regressors]), \
        f'Regressors/independent variables not passed as kwargs {kwargs.keys()}'

    # Generate the dictionary of parameters to use as guesses
    param_guess = self.make_params()

    # Set the parameter dictionary initial value to unity
    if isinstance(init_vals, float) or isinstance(init_vals, int):
        for param in param_guess.values(): param.set(value = init_vals)
    else:
        param_guess = init_vals

    return self.fit(data, params = param_guess, **kwargs).params

if __name__ == '__main__':
    pass
