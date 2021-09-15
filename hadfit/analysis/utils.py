# -- Import third-party modules
import numpy as np

# -- Import user-defined modules
from hadfit import Hadron

def outliers_score(data: np.ndarray) -> np.ndarray:
    """ Compute the Z scores to try detecting outliers in the dataset.

    As the regression is pretty unstable due to its multistate and
    correlated nature, the presence of outliers is common. The computation
    of the final mass independent of the fit window cannot be done using
    outliers, as they will sharply impact the distribution of the data.
    Thus, one should try removing the outliers. 

    In order to remove the outliers, one should use a robust statistic.
    The mean is not valid and we should use the median. We define the
    O-score as,

                    O_i = abs(abs(X_i - median(X)) / MAD),

    where MAD is the median of the numerator. An O value close to zero
    implies that the datapoint X_i is close to the median. An O value 
    far from zero implies that the datapoint X_i might be an outlier.
    If O_i > 3.5, then X_i is possibly an outlier.

    As with any tests, it needs some assumptions. One of them is that
    the distribution of X_i should be normal.

    --- Parameters
    data: np.ndarray
        Unidimensional dataset composing the sample of X

    --- Returns
    np.ndarray
        Ordered, O-scores for all the X in the dataset.
    """

    # Calculate the absolute deviation from the median
    abs_dev_from_median = np.abs(data - np.median(data))

    # Compute the MAD
    MAD = np.median(abs_dev_from_median)

    # Compute the O-score
    return np.abs(abs_dev_from_median / MAD) if MAD else 0.

def median_distribution(vals: np.ndarray, errs: np.ndarray, num_boot: int = 2000) -> np.ndarray:
    """ Empirical distribution of the median of the data whose central values
    are contained in vals and whose errors are contained in errs. The distribution
    is calculated using bootstrap resampling, using a resampled dataset res_data
    computed taking into account the errors. res_data is computed as

    1. First, we select N different random indices of the original sample with
    repetition. Note that N is the size of the dataset.
    2. Compute res_data assuming the distribution of vals[i] is normal.

                res_data = vals[ridx] + N(0, errs[ridx])
    3. Compute the median of res_data and append it to the distribution array.

    --- Parameters
    vals: np.ndarray
        Point estimate of the parameter/statistic whose median error must be
        extracted.
    errs: np.ndarray
        Estimation of the errors attached to vals.
    num_boot: int
        Number of bootstrap iterations. The larger, the more precise the distribution.

    --- Returns
    np.ndarray:
        Array containing num_boot elements with random realistaions of the median.
    """

    # Container for the median distribution
    median_dist = np.empty(num_boot)

    # Compute num_boot different estimates
    for nb in range(median_dist.size):

        # Get some indices to avoid overpopulation of outliers
        ridx = np.random.randint(0, vals.size, size = vals.size)
        
        # Resample using the errors
        res_vals = vals[ridx] + np.random.normal(scale = errs[ridx], size = vals.size)

        # Calculate the median
        median_dist[nb] = np.median(res_vals)

    return median_dist

def select_init_windows(hadron: Hadron):
    """ Select the initial and final windows depending on the Hadron's Nk. """
    if (hadron.Nk >= 40):
        return 2, 6
    elif (30 < hadron.Nk < 40):
        return 2, 4
    else:
        return 2, 3

if __name__ == '__main__':
    pass
