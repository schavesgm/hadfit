# -- Load some builtin modules
import argparse
from typing import Union

# -- Load user defined modules
from hadfit import Model
from hadfit import MultiStateFit
from hadfit import hadron_from_fastsum
from hadfit import tidy_fastsum
from hadfit import save_fastsum
from hadfit import select_init_windows

if __name__ == '__main__':

    # -- Collection of help messages used in the arguments
    help_messages = {
        'path'       : 'Path where the mesonic data for a given Nt and sources is stored',
        'channel'    : 'Channel to be processed',
        'flavour'    : 'Flavour combination to be processed',
        'normalise'  : 'Normalise the data around the middle point of the lattice: C(t) -> C(t) / C(Nt/2).',
        't_max'      : 'Maximum time used in the regression. If set, then fold is set to false by default,',
        'proportion' : 'Proportion of Nt/2 to be used in the regression. Ignored if t_max is present',
        'Ns_max'     : 'Maximum number of states to be used in the analysis',
        'num_boot'   : 'Number of MC iterations used to estimate the median standard errors',
    }

    # Generate an argparse object
    parser = argparse.ArgumentParser('Process FASTSUM mesonic correlation functions')

    # Add some content to the parser
    parser.add_argument('--path', type=str, help=help_messages['path'])
    parser.add_argument('--channel', '-c', type=str, choices=['g5', 'gi', 'gig5', '1'], help=help_messages['channel'])
    parser.add_argument('--flavour', '-f', type=str, choices=['uu', 'us', 'uc', 'ss', 'sc', 'cc'], help=help_messages['flavour'])
    parser.add_argument('--normalise', action='store_true', help=help_messages['normalise'])
    parser.add_argument('--t_max', '-t', type=int, required=False, help=help_messages['t_max'])
    parser.add_argument('--proportion', '-p', type=float, default=1.0, help=help_messages['proportion'])
    parser.add_argument('--Ns_max', type=int, default=4, help=help_messages['Ns_max'])
    parser.add_argument('--num_boot', type=int, default=1000, help=help_messages['num_boot'])

    # Retrieve the command line arguments
    args = parser.parse_args()

    # Load the hadron from the data
    hadron = hadron_from_fastsum(args.path, args.flavour, args.channel)

    # Generate the ansatz to be used in the fit
    ansatz = Model(f'A * cosh(M * (t - {hadron.Nk // 2}))', 't', 'A, M')

    # If t_max is passed, fold the data if the value is smaller than hadron.Nk // 2
    if args.t_max:
        assert 0 < args.t_max < hadron.Nk, f'{args.t_max=} should be in (0, {hadron.Nk})'
        t_max, fold = args.t_max, args.t_max < hadron.Nk // 2 + 1
    else:
        t_max, fold = int(args.proportion * (hadron.Nk // 2 + 1)), True

    # Generate a MultiState object to fit the hadron
    msfit = MultiStateFit(hadron, ansatz, Ns_max=args.Ns_max, nk_max=t_max, normalise=args.normalise, fold=fold)

    # Compute the dictionary of estimates of ground masses
    mass_est = msfit.estimate_ground_mass(*select_init_windows(hadron), False)

    # Clean the estimate of the ground mass
    best_mass = msfit.analyse_ground_mass(mass_est, args.num_boot)

    # Tidy the data and compute some nice values
    results = tidy_fastsum(hadron, best_mass)

    # Save all the fastsum data in the right path
    save_fastsum(hadron, results, nk_max_or_prop=f'p{args.proportion}' if not args.t_max else str(t_max))
