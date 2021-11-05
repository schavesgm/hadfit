# -- Import third-party modules
import numpy as np

# -- Import user-defined modules
from hadfit import Channel
from hadfit import Flavour
from hadfit import Hadron

# Mapping to almost physical fastsum states
init_mass_fastsum = {
    Channel.PSEUDOSCALAR: {
        Flavour.UU: 236.0,   Flavour.US: 500.0,
        Flavour.SS: 650.0,   Flavour.UC: 1800.0,
        Flavour.SC: 1900.0,  Flavour.CC: 3000.0,
    },
    Channel.VECTOR: {
        Flavour.UU: 750.0,   Flavour.US: 900.0,
        Flavour.SS: 1000.0,  Flavour.UC: 1900.0,
        Flavour.SC: 2000.0,  Flavour.CC: 3000.0,
    },
    Channel.AXIAL_PLUS: {
        Flavour.UU: 1200.0, Flavour.US: 1300.0,
        Flavour.SS: 1500.0, Flavour.UC: 2300.0,
        Flavour.SC: 2500.0, Flavour.CC: 3400.0,
    },
    Channel.SCALAR: {
        Flavour.UU: 600.0,  Flavour.US: 800.0,
        Flavour.SS: 1100.0, Flavour.UC: 2200.0,
        Flavour.SC: 2400.0, Flavour.CC: 3400.0,
    }
}

def clean_parenthesis(obj: object):
    """ Clean the parenthesis in an object after casting it to a string """
    return str(obj).replace('(', '').replace(')', '')

def select_initial_mass(hadron: Hadron, inv_ak: float) -> float:
    """ Select the initial mass depending on the inverse lattice spacing in the 
    non-integrated direction. For thermal correlation functions, we use the inverse 
    of the lattice spacing in the time direction. The initial mass is set to around 
    200MeV in case the hadron does not belong to the Fastsum famility. In case it is,
    we set the inital mass close to the physical state.

    --- Parameters:
    hadron: Hadron
        Hadron used in the calculation. It is a FASTSUM meson if it contains 
        ['is_fastsum'] in the information dictionary.
    inv_ak: float
        Inverse of the lattice spacing in the non-integrated direction. It should be 
        in MeV.

    --- Returns:
    float:
        Initial value of the mass for the given hadron.
    """
    # Check whether the hadron is a fastsum hadron
    if 'is_fastsum' not in hadron.info: 
        return 200.0 / inv_ak
    else:
        channel, flavour = hadron.info['channel'], hadron.info['flavour']
        return init_mass_fastsum[channel][flavour] / inv_ak

def compute_best_estimate(relevant_info: list, mc_iters: int):
    """ Compute the best estimate of the ground mass using the
    AICc criterion as a weight. The final value for the mass is computed
    using a MonteCarlo resampling with mc_iters iterations.

    --- Parameters
    relevant_info: list
        List containing all relevant information to be used in the computation
        of the best estimate.
    mc_iters: int
        Number of MonteCarlo calculations used to compute the empirical
        distribution
    
    --- Returns
    dict[str, float]
        Best estimate of the mass, its error, AICc and reduced chisquared.
    """

    # Dictionary that will contain the best model selection
    best_estimate = {'M0': 0.0, 'dM0': 0.0, 'AICc': 0.0, 'rchi': 0.0}

    # Number of relevant models included
    num_rel = len(relevant_info)

    # Calculate the best AICc value from the data
    min_AICc = np.array([info['AICc'] for info in relevant_info]).min()

    # Container of all masses, errors and weights
    masses, errors, weights = np.empty(num_rel), np.empty(num_rel), np.empty(num_rel)

    # Iterate through all information contained in the dictionary
    for i, info in enumerate(relevant_info):

        # Calculate the weight
        weights[i] = np.exp(0.5 * (min_AICc - info['AICc']))

        # Save the masses and the errors in the containers
        masses[i], errors[i] = info['M0'], info['dM0']

        # Calculate the weighted sum of each item
        for key, value in info.items():
            best_estimate[key] += value * weights[i]

    # Calculate the weighted average by normalising the data
    for key, value in best_estimate.items():
        best_estimate[key] = best_estimate[key] / np.sum(weights)

    # Sample of masses and errors using weighted MonteCarlo
    M_dist = np.empty(mc_iters)

    # Generate the MonteCarlo sample
    for nc in range(mc_iters):

        # Resample some indices from the mass
        idx = np.random.randint(0, masses.size, masses.size)

        # Generate a new resample of the mass vector
        res_M = masses[idx] + np.random.normal(scale=errors[idx], size=errors.size)

        # Obtain the weighted average of the resampled vector
        M_dist[nc] = np.average(res_M, weights=weights[idx])

    # Compute the weighted average and its standard error
    best_estimate['M0']  = np.average(masses, weights=weights)
    best_estimate['dM0'] = np.std(M_dist, ddof=1)

    return best_estimate

if __name__ == '__main__':
    pass
