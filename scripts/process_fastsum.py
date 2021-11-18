# -- Load some builtin modules
import sys
import argparse

# -- Load user defined modules
from hadfit import Model
from hadfit import MultiStateFit
from hadfit import hadron_from_fastsum
from hadfit import tidy_fastsum
from hadfit import save_fastsum
from hadfit import select_init_windows

if __name__ == '__main__':

    # Generate an argparse object
    parser = argparse.ArgumentParser('Process FASTSUM correlation functions')

    # Add some content to the parser
    parser.add_argument('--path', type=str, help='Path where the hadron data for given Nt and sources is stored')
    parser.add_argument('--channel', '-c', type=str, choices=['g5', 'gi', 'gig5', '1'], help='Channel to be processed.')
    parser.add_argument('--flavour', '-f', type=str, choices=['uu', 'us', 'uc', 'ss', 'sc', 'cc'], help='Flavour to be processed.')
    parser.add_argument('--prop', '-p', type=float, help='Proportion of the data (halved) to be used.')
    parser.add_argument('--Ns_max', type=int, default=4, help='Maximum number of states to be used in the analysis.')
    parser.add_argument('--normalise', type=bool, default=True, help='Normalise the data by its middle point (Recommended).')
    parser.add_argument('--num_boot', type=int, default=1000, help='Number of MC iterations used to compute median stderrs.')

    # Retrieve the command line arguments
    args = parser.parse_args()

    # Assert prop is a proportion
    assert 0 < args.prop <= 1.0, f'{prop=} must a proportion: (0, 1.0]'

    # Load the hadron from the data
    hadron = hadron_from_fastsum(args.path, args.flavour, args.channel)

    # Generate the ansatz to be used in the fit
    ansatz = Model(f'A * cosh(M * (t - {hadron.Nk // 2}))', 't', 'A, M')

    # Generate a MultiState object to fit the hadron
    msfit = MultiStateFit(hadron, ansatz, Ns_max=args.Ns_max, normalise=args.normalise, prop=args.prop)

    # Compute the dictionary of estimates of ground masses
    mass_est = msfit.estimate_ground_mass(*select_init_windows(hadron), False)

    # Clean the estimate of the ground mass
    best_mass = msfit.analyse_ground_mass(mass_est, args.num_boot)

    # Tidy the data and compute some nice values
    results = tidy_fastsum(hadron, best_mass)

    # Save all the fastsum data where it should
    save_fastsum(hadron, results, prop=args.prop)
