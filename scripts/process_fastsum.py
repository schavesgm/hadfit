# -- Load some builtin modules
import sys

# -- Load user defined modules
from hadfit import Model
from hadfit import MultiStateFit
from hadfit import hadron_from_fastsum
from hadfit import tidy_fastsum
from hadfit import save_fastsum
from hadfit import select_init_windows

if __name__ == '__main__':

    # Retrieve the command line arguments
    if len(sys.argv[1:]) == 0:
        help_str = \
        '\n\tprocess_fastsum.py path_to_data channel flavour\n\n' + \
        ' - path_to_data: str\n' + \
        '   Path where the hadron data for the given Nt is stored.\n' + \
        ' - channel: str\n' + \
        '   Channel to be processed as string.\n' + \
        ' - flavour: str\n' + \
        '   Flavour to be processed as string.\n\n' + \
        ' Example: process_fastsum.py ./data/64x32_ll gig5 us\n' + \
        '   Process the 64x32_ll configurations of the us mesons in the\n' + \
        '   axial plus channel.'
        print(help_str)
        sys.exit()

    # Path to data, channel and flavour to use
    path, channel, flavour = sys.argv[1:]

    # Load the hadron from the data
    hadron = hadron_from_fastsum(path, flavour, channel)

    # Generate the ansatz to be used in the fit
    ansatz = Model(f'A * cosh(M * (t - {hadron.Nk // 2}))', 't', 'A, M')

    # Generate a MultiState object to fit the hadron
    msfit = MultiStateFit(hadron, ansatz, Ns_max = 4, fold = True, normalise = True)

    # Compute the dictionary of estimates of ground masses
    mass_est = msfit.estimate_ground_mass(*select_init_windows(hadron), False)

    # Clean the estimate of the ground mass
    best_mass = msfit.analyse_ground_mass(mass_est, 5000)

    # Tidy the data and compute some nice values
    results = tidy_fastsum(hadron, best_mass)

    # Save all the fastsum data where it should
    save_fastsum(hadron, results)
