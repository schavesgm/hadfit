# -- Import some built-in modules
import os
import argparse

from typing import NamedTuple
from pathlib import Path

# -- Import some third-party modules
import numpy as np
import matplotlib.pyplot as plt

# -- Import some user-defined modules
from hadfit import hadron_from_fastsum
from hadfit import FastsumRetriever
from hadfit import Hadron
from hadfit import Flavour

# Add custom styles to the plots
plt.style.use(['science', 'ieee', 'monospace'])

# Namedtuple used to hold the ratio statistic and its uncertainty
Ratio = NamedTuple('Ratio', [('pe', np.ndarray), ('se', np.ndarray)])

def compute_ratio(hadron_1: Hadron, hadron_2: Hadron, Nb: int = 5000) -> Ratio:
    """ Compute the ratio between two hadrons, defined as:

        R(t) = (cv1(t) * cv2(Nt // 2)) / (cv2(t) * cv1(Nt // 2)),
    being cv1, cv2 the central values of each of the hadrons. The standard
    errors are computed using bootstrap.

    Parameters:
    hadron_1: Hadron
        First hadron to be used in the numerator of the ratio statistic.
    hadron_2: Hadron
        Second hadron to be used in the denominator of the ratio statistic.
    Nb: int
        Number of bootstrap iterations used to estimate the standard errors.

    Returns:
    Ratio = NamedTuple[np.ndarray, np.ndarray]:
        Named tuple containing two numpy arrays, the first one corresponds to the 
    point statistic and the second one to the standard error.
    """

    # Both hadrons should have the same number of points
    assert hadron_1.Nk == hadron_2.Nk, \
        f'Both hadrons should have the same points: {hadron_1.Nk} != {hadron_2.Nk}'

    # Get the middle point of the lattice
    Nk = hadron_1.Nk

    # Get the central values for each of the hadrons
    cv1, cv2 = hadron_1.central_value(), hadron_2.central_value()

    # Estimate its uncertainty using bootstrap -> Buffer
    ratio_un = np.empty([Nb, cv1.size])
    
    # Bootstrap iterations
    for nb in range(ratio_un.shape[0]):

        # Generate some resampled indices
        r1 = np.random.randint(0, hadron_1.Nc, hadron_1.Nc)
        r2 = np.random.randint(0, hadron_2.Nc, hadron_2.Nc)

        # Generate some random data from each of the hadrons
        data_h1, data_h2 = hadron_1.data[r1, :], hadron_2.data[r2, :]

        # Compute the central value for each of the resamples
        rcv1, rcv2 = np.mean(data_h1, axis=0), np.mean(data_h2, axis=0)

        # Save the ratio statistic in this iteration
        ratio_un[nb, :] = (rcv1 * rcv2[Nk // 2]) / (rcv2 * rcv1[Nk // 2])

    # Compute the point estimate of the ratio
    ratio_pe = (cv1 * cv2[Nk // 2]) / (cv2 * cv1[Nk // 2])

    # Compute the standard error of the statistic
    ratio_se = np.std(ratio_un, axis=0, ddof=1)

    return Ratio(pe=ratio_pe, se=ratio_se)

if __name__ == '__main__':

    # Generate the argument parser object
    parser = argparse.ArgumentParser('Generate the ration between vector and axial')

    # Add some data to the parser
    parser.add_argument('-s', '--sources', type=str, choices=['ll', 'ss'], help='Sources used in the calculation')
    parser.add_argument('-p', '--path', type=str, default='./data', help='Path where the data is stored')
    parser.add_argument('-i', '--include_128', action='store_true', help='Compute the ratio of Nt=128')
    parser.add_argument('-t', '--use_tau_tilde', action='store_true', help='Use tau_tilde = t - Nt / 2 in the plot')

    # Parse the command line arguments
    args = parser.parse_args()

    # Assert the path exists
    assert os.path.exists(args.path), f'{args.path=} does not exist in system'

    # Get some variables from the command line parser
    sources, path, include_128, use_tau_tilde = args.sources, args.path, args.include_128, args.use_tau_tilde

    # Color palette used in the calculation
    COLORS = [
        '#212f45', '#577590', '#34a0a4',  # 64 - 56 - 48
        '#76c893', '#495057', '#f9c74f',  # 40 - 36 - 32
        '#f9844a', '#f3722c', '#f94144',  # 28 - 24 - 20
    ]

    # Filter used in the files
    ffunc = lambda p : p.name.endswith(f'_{sources}') and ('128x32' not in p.name if not include_128 else True)

    # Prepend the colour for Nt=128 if included
    if include_128: COLORS.insert(0, '#00043a')

    # Get all needed files in the data directory sorted from low to high temperature
    files = sorted(filter(ffunc, Path(args.path).iterdir()), key=lambda p: int(p.name.split('x')[0]))[::-1]

    # x-label and y-label strings
    if not use_tau_tilde:
        xlabel = r'$\tau / N_\tau$'
        ylabel = r'$R(\tau) = \frac{V(\tau) \cdot A(N_\tau/2)}{A(\tau) \cdot V(N_\tau/2)}$'
    else:
        xlabel = r'$\tilde{\tau} = \tau - \frac{N_\tau}{2}$'
        ylabel = r'$R(\tilde{\tau}) = \frac{V(\tilde{\tau}) \cdot A(N_\tau/2)}{A(\tilde{\tau}) \cdot V(N_\tau/2)}$'

    # Folder where the figures will be stored
    if not os.path.exists('./figures/'): os.makedirs('./figures')

    # Iterate for each flavour available
    for flavour in Flavour:

        print(f'Currently processing flavour={flavour}')

        # Figure where the data will be plotted
        fig = plt.figure(figsize=(6, 5))

        # Add an axis to the figure
        axis = fig.add_subplot(1, 1, 1)

        # Add some properties to the axis
        axis.set(xlabel=xlabel, ylabel=ylabel), axis.grid('#fefefe', alpha=0.5)

        # Starting minimum and maximum values
        min_val, max_val = 10000000, -10000000

        # Iterate for each temperature
        for f, file_temp in enumerate(files):

            # Load the vector and axial plus data
            vec = hadron_from_fastsum(file_temp.__str__(), flavour, 'gi')
            axp = hadron_from_fastsum(file_temp.__str__(), flavour, 'gig5')

            # Compute the ratio statistic
            ratio = compute_ratio(vec, axp, Nb=500)

            # Generate the x-axis data
            t_data = np.arange(vec.Nk) / vec.Nk if not use_tau_tilde else np.arange(vec.Nk) - vec.Nk // 2

            # Plot the data in the axis
            axis.errorbar(
                t_data, ratio.pe, ratio.se, color=COLORS[f], 
                label=fr'$N_\tau={vec.Nk}$', alpha=1.0 if vec.Nk < 128 else 0.3
            )

            # Compute the 0.4 and 0.6 point in the dataset
            point_40, point_60 = int(0.4 * vec.Nk), int(0.6 * vec.Nk)

            # Update the minimum and maximum value of the region
            if vec.Nk < 128:
                min_val = min(min_val, min(ratio.pe[point_40:point_60]))
                max_val = max(max_val, max(ratio.pe[point_40:point_60]))

        # Load the lowest temperature data
        vec_data = FastsumRetriever('./output/', flavour, 'gi', sources, nk_or_prop='p1.0')
        axv_data = FastsumRetriever('./output/', flavour, 'gig5', sources, nk_or_prop='p0.7')

        # Get the lowest temperature median estimates
        M0_vec, M0_axv = vec_data.get_median_estimates()[128], axv_data.get_median_estimates()[128]

        # Assume normality to obtain the standard errors in the measurements
        se_vec = max(abs(M0_vec[0] - M0_vec[1]), abs(M0_vec[1] - M0_vec[2])) / 1.96
        se_axv = max(abs(M0_axv[0] - M0_axv[1]), abs(M0_axv[1] - M0_axv[2])) / 1.96

        # Get the difference of squared masses
        delta_M_pe = (M0_vec[1] ** 2 - M0_axv[1] ** 2)

        # Function to generate a new random estimate of delta_M
        gen_dM = lambda: (M0_vec[1] + np.random.normal(0.0, scale=se_vec)) ** 2 - \
                         (M0_axv[1] + np.random.normal(0.0, scale=se_axv)) ** 2

        # Estimate the uncertainties by propagating using MonteCarlo
        delta_M_se = np.std([gen_dM() for _ in range(5000)], ddof=1)

        # Plot the curve using the fitted data
        t    = np.linspace(0, 1, 500) if not use_tau_tilde else np.arange(128) - 128 // 2
        t_sq = 5997 * (t - 0.5) ** 2 if not use_tau_tilde else t ** 2

        # Generate the central value and its standard error
        central = 1 + 0.5 * delta_M_pe * t_sq
        std_err = np.std(np.vstack([1 + 0.5 * gen_dM() * t_sq for _ in range(5000)]), ddof=1, axis=0)

        # Add the data to the axis
        axis.plot(t, central, color='#7b2cbf', label='Analytic')
        axis.fill_between(t, central - 1.96 * std_err, central + 1.96 * std_err, color='#7b2cbf', alpha=0.3)

        # Restrict the limits in the x axis and y axis
        axis.set_ylim(0.97 * min_val, 1.02 * max_val)
        if not use_tau_tilde: 
            axis.set_xlim(0.4, 0.6)
        else:
            axis.set_xlim(-20, 20)

        # Get the legend
        handles, labels = axis.get_legend_handles_labels()

        leg1 = axis.legend(
            handles[:6], labels[:6], loc='upper center', 
            frameon=False, ncol=6, bbox_to_anchor=(0.5, 1.15)
        )

        leg2 = axis.legend(
            handles[6:], labels[6:], loc='upper center', 
            frameon=False, ncol=5, bbox_to_anchor=(0.5, 1.10)
        )

        # Add the legend and adjust the plot
        axis.add_artist(leg1)
        fig.subplots_adjust(top=0.85)

        # Add a title to the figure
        fig.suptitle(f'Flavour={flavour.name} - Sources={sources}')

        # Save the figure in the corresponding path
        fig.savefig(f'./figures/ratioAV_{flavour}_{sources}_i{int(include_128)}_t{int(use_tau_tilde)}.pdf')
