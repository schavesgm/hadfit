# -- Import built-in libraries
import copy

# -- Import third-party modules
import numpy as np
import lmfit as lm

# -- Import user-defined libraries
from .Model import Model
from .Model import CompositeModel

def bootstrap_fit(model: Model, params: lm.Parameters, data: np.ndarray, num_boot: int = 500, **kwargs) -> lm.Parameters:
    """ Fit a model to some data using bootstrap to estimate the confidence intervals

    --- Parameters
    model: Model
        Model used in the regression.
    params: lm.Parameters
        Parameters to be updated in the fit
    data: np.ndarray
        Data measured used to regressed the model
    num_boot: int
        Number of bootstrap iterations used in the calculation of errors
    **kwargs:
        Optional keyword arguments. The regressor data, for example, x, should be passed this way.

    --- Returns
    lm.Parameters
        Updated parameters with bootstrap estimated standard errors.
    """

    # Assert the data is a 2d array
    assert len(data.shape) == 2, 'data must be a 2-dimensional array of Nc observations of Nk variables'

    # Set the parameter hints in the model to prepare the fit
    model.set_parameters(params)

    # Inverse of the covariance matrix
    inv_cov = np.linalg.inv(np.cov(data.T) / data.shape[0])

    # Fit the data to the unresampled dataset to obtain the sample estimate
    sample_results = model.fit(np.mean(data, axis=0), inv_cov=inv_cov, **kwargs)

    # Get the values of the parameters, that is, the sample estimates
    params_sample = sample_results.params.valuesdict().values()

    # Container that will hold the distribution of delta_star
    delta_star = np.empty([num_boot, len(model.symb_parameters)])

    # Iterate several times
    for nb in range(num_boot):

        # Resample the data in this iteration
        res_data = data[np.random.randint(0, data.shape[0], data.shape[0])]

        # Obtain the covariance matrix of the data
        inv_cov = np.linalg.inv(np.cov(res_data.T) / data.shape[0])

        # Obtain the data to be fitted: the central value
        data_fit = np.mean(res_data, axis = 0)

        # Obtain the fitted parameters in the current resample
        params_star = model.fit(data_fit, inv_cov = inv_cov, **kwargs).params.valuesdict().values()

        # Save the delta_star estimate params_star - params_sample in this iteration
        for ip, (p_star, p_samp) in enumerate(zip(params_star, params_sample)):
            delta_star[nb, ip] = p_star - p_samp

    # Compute the correct standard errors -- 
    for ip, param in enumerate(model.symb_parameters):

        # Compute the 80% Confidence intervals
        Q10, Q90 = np.quantile(emp_bootstrap[:, ip], [0.1, 0.9])

        # Compute the standard error using the CI, assume symmetry
        stderr = 0.5 * (abs(Q90) + abs(Q10))

        # Set the standard error
        results.params[str(param)].stderr = stderr

    return results

def generate_multistate(model: Model, num_states: int) -> CompositeModel:
    r""" Generate a composite model composed by the sum of multiple models.
    For example, if the model is A * exp(-M * x), the sum of Ns models would
    be

        m(x, \theta) = \sum_{s = 0}^{Ns - 1} A_s * exp(-M_s * x).

    --- Parameters
    model: Model
        Template model used to generate the combined multistate model.
    num_states: int
        Number of states -- copied of the model -- included in the combined model.

    --- Returns
    CompositeModel
        Composite model containing the sum of the Ns template models. Each model
        has its own prefix, ns.
    """

    # Composite model result of the addition of several base models
    comp_model = None

    # Iterate for each number of states
    for ns in range(num_states):

        # Generate a new copy of the model
        copied_model = copy.copy(model)

        # Change the prefix of the base_model to be gns
        copied_model.prefix = f's{ns}'

        # Add the base_model to the composite model
        if ns == 0:
            comp_model = copied_model
        else:
            comp_model = comp_model + copied_model

    return comp_model

def generate_cosh_model(n_tau: int, num_states: int) -> CompositeModel:
    """ Generate a composite model composed by multiple cosh functions. """

    # Generate a base model to be added to the composite model
    base_model = Model(f'A * cosh(M * (t - 0.5 * {n_tau}))', 't', 'A M')

    return generate_multistate(base_model, num_states)

def generate_exp_model(num_states: int) -> CompositeModel:
    """ Generate a composite model composed by multiple exp functions. """

    # Generate a base model to be added to the composite model
    base_model = Model(f'A * exp(- M * t)', 't', 'A M')

    return generate_multistate(base_model, num_states)

if __name__ == '__main__':
    pass
