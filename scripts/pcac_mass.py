from typing import NamedTuple

from hadfit import hadron_from_fastsum
from hadfit import Hadron

import numpy as np
import matplotlib.pyplot as plt

# Namedtuple containing the PCAC mass estimates
PCACMass = NamedTuple('PCACMass', [('pe', np.ndarray), ('se', np.ndarray)])

def compute_pcac(AP: Hadron, PP: Hadron, num_boot: int = 1000) -> PCACMass:
    """ Compute the PCAC mass of two hadrons. """

    assert AP.Nk == PP.Nk, f'{AP.Nk=} is not equal to {PP.Nk=}'

    # Get a reference to each correlation function data
    AP_corr, PP_corr = AP.data, PP.data

    # Get a reference to each central value
    AP_cent, PP_cent = np.abs(AP.central_value()), np.abs(PP.central_value())

    # Estimate the uncertainty in the estimate using bootstrap
    PCAC_un = np.empty([num_boot, AP.Nk - 1])

    # Compute some estimates of the PCAC mass
    for nb in range(num_boot):

        # Resample each of the correlation functions
        AP_res = AP_corr[np.random.randint(0, AP.Nc, AP.Nc), :]
        PP_res = PP_corr[np.random.randint(0, PP.Nc, PP.Nc), :]

        # Compute the central value of each resample
        AP_res, PP_res = np.mean(AP_res, axis=0), np.mean(PP_res, axis=0)

        # Compute the PCAC mass
        PCAC_un[nb, :] = (0.5 * np.abs(AP_res[1:] - AP_res[:-1])) / PP_res[:-1]

    # Compute the point statistic of the PCAC mass
    PCAC_pe = (0.5 * np.abs(AP_cent[1:] - AP_cent[:-1])) / PP_cent[:-1]

    return PCACMass(pe=PCAC_pe, se=np.std(PCAC_un, axis=0, ddof=1))

if __name__ == '__main__':
    
    # Load the needed hadrons
    AP = hadron_from_fastsum('./data/128x32_ss/', 'uu', 'AP')
    PP = hadron_from_fastsum('./data/128x32_ss/', 'uu', 'PP')

    # Compute the corresponding PCAC mass
    pcac = compute_pcac(AP, PP, num_boot=1500)
    
    plt.errorbar(np.arange(pcac.pe.size), pcac.pe, yerr=pcac.se)
    plt.xlim(10, 100)
    plt.show()
