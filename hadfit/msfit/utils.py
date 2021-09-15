# -- Import third-party modules
import numpy as np

def clean_parenthesis(obj: object):
    """ Clean the parenthesis in an object after casting it to a string """
    return str(obj).replace('(', '').replace(')', '')

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
    # Dictionary that will contain the best result
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
        res_M = masses[idx] + np.random.normal(scale = errors[idx], size = errors.size)

        # Obtain the weighted average of the resampled vector
        M_dist[nc] = np.average(res_M, weights = weights[idx])

    # Compute the weighted average and its standard error
    best_estimate['M0']  = np.average(masses, weights = weights)
    best_estimate['dM0'] = np.std(M_dist, ddof = 1)

    return best_estimate

if __name__ == '__main__':
    pass
