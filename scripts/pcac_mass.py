from typing import NamedTuple
from pathlib import Path

from hadfit import hadron_from_fastsum
from hadfit import Hadron

import numpy as np
import matplotlib.pyplot as plt

# Get the styles
plt.style.use(['science', 'ieee', 'monospace'])

# Namedtuple containing the PCAC mass estimates
PCACMass = NamedTuple('PCACMass', [('pe', np.ndarray), ('se', np.ndarray)])

def compute_pcac(AP: Hadron, PP: Hadron, num_boot: int = 1000) -> PCACMass:
    """ Compute the PCAC mass of two hadrons. """

    assert AP.Nk == PP.Nk, f'{AP.Nk=} is not equal to {PP.Nk=}'

    # Get a reference to each correlation function data
    AP_corr, PP_corr = AP.data, PP.data

    # Get a reference to each central value
    AP_cent, PP_cent = np.abs(AP.central_value()), PP.central_value()

    # Estimate the uncertainty in the estimate using bootstrap
    PCAC_un = np.empty([num_boot, AP.Nk - 1])

    # Compute some estimates of the PCAC mass
    for nb in range(num_boot):

        # Resample each of the correlation functions
        AP_res = np.abs(AP_corr[np.random.randint(0, AP.Nc, AP.Nc), :])
        PP_res = PP_corr[np.random.randint(0, PP.Nc, PP.Nc), :]

        # Compute the central value of each resample
        AP_res, PP_res = np.mean(AP_res, axis=0), np.mean(PP_res, axis=0)

        # Compute the PCAC mass
        PCAC_un[nb, :] = (0.5 * np.abs(AP_res[1:] - AP_res[:-1])) / PP_res[:-1]

    # Compute the point statistic of the PCAC mass
    PCAC_pe = (0.5 * np.abs(AP_cent[1:] - AP_cent[:-1])) / PP_cent[:-1]

    return PCACMass(pe=PCAC_pe, se=np.std(PCAC_un, axis=0, ddof=1))

def sample_median(pe: np.ndarray, se: np.ndarray) -> np.ndarray:
    """ Sample the median from the collection of points and their errors. """
    idx = np.random.randint(0, pe.size, pe.size)
    return pe[idx] + np.random.normal(0, scale=se[idx])

if __name__ == '__main__':

    # Sources used in the calculation
    sources = 'll'

    # Initial and final times included in the estimation
    tau_0, tau_f = 30, 98
    
    # Load the needed hadrons
    AP = hadron_from_fastsum(f'./data/128x32_{sources}/', 'uu', 'AP')
    PP = hadron_from_fastsum(f'./data/128x32_{sources}/', 'uu', 'PP')

    # Compute the PCAC mass from both correlation functions
    pcac = compute_pcac(AP, PP, num_boot=1500)

    # Exclude the first and last points in the data
    pcac_est, pcac_err, taus = pcac.pe[tau_0:tau_f], pcac.se[tau_0:tau_f], np.arange(tau_0, tau_f)

    # Compute the median of the data and its error
    med_est = np.median(pcac_est)
    med_err = np.std(np.stack([sample_median(pcac_est, pcac_err) for _ in range(5000)]), ddof=1)

    # Plot the data in a figure
    fig  = plt.figure(figsize=(6, 4))
    axis = fig.add_subplot()

    # Set some properties on the axis
    axis.set(xlabel=r'$\tau$', ylabel=r'$m_q^{PCAC}(\tau)$')
    axis.grid('#fefefe', alpha=0.3)

    # Plot the data in the axis
    axis.errorbar(taus, pcac_est, yerr=pcac_err, color='navy')
    axis.fill_between(taus, med_est - med_err, med_est + med_err, alpha=0.3, color='crimson')

    # Set some title in the figure
    fig.suptitle(r'$M_q^{PCAC} = ' rf'{med_est * 5997:.4f} \pm {med_err * 5997:.4f}$ [MeV]')

    # Save the data in the corresponding folder
    path = Path('./figures/pcac')
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f'pcac_mass_{sources}.pdf')
