# -- Import some builtin modules
import os, json, re
from typing import Union

# -- Import some third party modules
import numpy as np
import matplotlib.pyplot as plt

# -- Import some user-defined modules
from hadfit import Hadron
from hadfit import Flavour
from hadfit import Channel
from .utils import median_distribution

class FastsumRetriever:
    """ Class used to retrieve all Fastsum output results in a tidy and easy to manage way. """

    def __init__(self, path: str, flavour: Union[Flavour, str], channel: Union[Channel, str], sources: str):

        # Save the flavour and channel depending on type passed
        self.flavour = flavour if isinstance(flavour, Flavour) else Flavour.from_str(flavour)
        self.channel = channel if isinstance(channel, Channel) else Channel.from_str(channel)

        # Save some parameters in the class
        self.sources, self.path = sources, path

        # Assert the sources are correct
        assert sources in ['ll', 'ss'], f'{sources = } must be in [ll, ss]'

        # Assert the path where the data is stored exists
        assert os.path.exists(self.full_path)

        # Dictionary that will index the results
        self.__dataset = self.load_data()

    # -- Important methods of the class {{{
    def get_median_estimates(self) -> dict:
        """ For each N_t in the dataset, retrieve a list containing the 
        lower limit of the 95% confidence interval, the central value and
        the upper limit of the 95% confidence interval.

                    {nt: (M0l, M0c, M0u) for each nt in N_t}

        --- Returns
        dict:
            Dictionary containing results in the format expressed above.
        """
        return {key: (val['M0l'], val['M0c'], val['M0u']) for key, val in self.__dataset.items()}

    def get_fit_window_estimates(self) -> dict:
        """ For each N_t in the dataset, retrieve a tuple containing the 
        mass estimate for each estimate.

                    {nt: (fw, M0fw, dM0fw) for each nt in N_t}

        --- Returns
        dict:
            Dictionary containing the results in the format expressed above.
            The tuples contains numpy arrays as elements.
        """
        return {
            key: (np.array(val['fw']), np.array(val['M0fw']), np.array(val['dM0fw'])) \
            for key, val in self.__dataset.items()
        }

    def load_data(self) -> dict:
        """ Load the results file into a dictionary. The results are retrieved
        from the self.folders data. 

        --- Returns
            Dictionary containing N_t as keys and the results as values
        """

        # Dictionary that will contain the data
        information = {}

        # Iterate for each folder to load the data
        for nt, folder in zip(self.n_tau, self.folders):

            # Load the data from the folder into the dictionary
            with open(os.path.join(folder, self.results_file_name), 'r') as file:
                information[nt] = json.loads(file.read())

        return information
    # -- }}}

    # -- Private methods of the class {{{
    def __list_folders(self) -> list:
        """ List all temperature folders """
        return sorted(
            filter(lambda f: re.match(r'\d+$', f), os.listdir(self.full_path)),
            reverse = True, key = int
        )
    # -- }}}

    # -- Property methods of the class {{{
    @property
    def dataset(self) -> dict:
        return self.__dataset

    @property
    def full_path(self) -> str:
        return os.path.join(self.path, str(self.channel), str(self.flavour), self.sources)

    @property
    def n_tau(self) -> list:
        return [int(nt) for nt in self.__list_folders()]

    @property
    def folders(self) -> list:
        return [os.path.join(self.full_path, nt) for nt in self.__list_folders()]

    @property
    def results_file_name(self) -> str:
        """ Name of the results file name, can be changed if needed. """
        return "results.json"

    @results_file_name.setter
    def results_file_name(self, new_name: str) -> None:
        """ Change the name of the results file name if needed. """
        self.results_file_name = new_name
    # -- }}}

def tidy_fastsum(hadron: Hadron, best_est: dict) -> dict:
    """ Extract the results from the dictionary of best results.
    The hadron should be a Fastsum hadron as it should contain
    some important keys in its information dictionary.

    --- Parameters
    hadron: Hadron
        Fastsum hadron used to extract the results stored in best_est
        using MultiFit
    best_est: dict
        Dictionary of ground masses extracted using MultiFit.

    --- Returns
    dict
        Dictionary containing the following keys:
        - fig:   matplotlib.Figure object containing a plot.
        - M0c:   Central median value -- Estimate of M0(FW).
        - M0l:   Lower limit of the 95% CI of M0c.
        - M0u:   Upper limit of the 95% CI of M0l.
        - fw:    Fit windows used when extracting best_est.
        - M0fw:  Ground mass extracted at each fit window.
        - dM0fw: Error of the ground mass at each fit window
    """

    # Extract some information from the best estimate
    f_wind  = np.array(list(best_est.keys()))
    M0_vals = np.array([m['M0']  for m in best_est.values()])
    M0_errs = np.array([m['dM0'] for m in best_est.values()])

    # Compute the distribution of medians using bootstrap
    median_dist = median_distribution(M0_vals, M0_errs, 5000)

    # Compute the median of the initial (unresampled) dataset
    median_M0 = np.median(M0_vals)

    # Compute the percentiles of the median distribution
    Q05_median, Q95_median = np.percentile(median_dist, [5, 95])

    # Generate a figure to plot the data into
    fig = plt.figure(figsize = (18, 12))

    # Add two axes to the figure, one for the histogram, another for values
    a_hist, a_vals = [fig.add_subplot(1, 2, a) for a in (1, 2)]

    # Add some information to each of the axes
    a_hist.set_xlabel(r'$\tilde{M}_0$')
    a_hist.set_ylabel(r'$\rho(\tilde{M}_0)$')
    a_hist.grid('#4a4e69', alpha = 0.4)

    a_vals.set_xlabel(r'$FW$')
    a_vals.set_ylabel(r'$M_0(FW)$')
    a_vals.grid('#4a4e69', alpha = 0.4)

    # Append the histrogram plot to the corresponding axes
    a_hist.hist(median_dist, bins = 100, color = '#1E352F', density = True)
    a_hist.axvline(median_M0, color = '#828C51')
    a_hist.axvspan(Q05_median, Q95_median, color = '#828C51', alpha = 0.4)

    # Append the mass and the value to the second axes
    a_vals.errorbar(f_wind, M0_vals, M0_errs, color = '#1E352F')
    a_vals.plot(f_wind, median_M0 * np.ones(M0_vals.size), color = '#828C51')
    a_vals.fill_between(f_wind, Q05_median, Q95_median, color = '#828C51', alpha = 0.4)

    # Construct the hadron identifier label
    had_label = str(hadron.info['channel'].to_latex()) + '\quad\quad;\quad\quad' + \
                str(hadron.info['flavour']) + '\quad\quad;\quad\quad' + \
                str(hadron.info['sources'])

    # Construct the masses and convert them to MeV
    masses_MeV = (Q05_median, median_M0, Q95_median)
    masses_MeV = tuple(round(q * 5997, 3) for q in masses_MeV)

    # Set the title of the figure
    fig.suptitle(f'${had_label}$\n${masses_MeV}$ [MeV]')

    return {
        'fig': fig, 
        'M0c': median_M0, 'M0l': Q05_median, 'M0u': Q95_median, 
        'fw':  f_wind.tolist(), 'M0fw': M0_vals.tolist(), 'dM0fw': M0_errs.tolist() 
    }

def save_fastsum(hadron: Hadron, results: dict, output_path: str = './output', show_plot: bool = False):
    """ Save the Fastsum results into a folder. The folder will be
    named using the locator:
            
            {output_path}/{channel}/{flavour}/{sources}/{Nt}.

    Inside the folder, one can find different files. One for the mass at different
    fit windows, one for the final estimate of the mass independent of the fit
    window and one plot that shows the distribution of medians and the final result.

    --- Parameters
    hadron: Hadron
        Hadron used in the fit. The hadron must be a "Fastsum" type hadron with
        some key information inside its information dictionary.
    results: dict
        Dictionary of results obtained by the function "tidy_fastsum"
    output_path: str
        Path where the data will be stored.
    """

    # Obtain some important information in the meson
    channel, flavour = hadron.info['channel'], hadron.info['flavour']
    sources, Nk      = hadron.info['sources'], hadron.Nk

    # Generate the full path to the output, channel, flavour, source, Nt
    full_path = os.path.join(str(channel), str(flavour), str(sources), str(Nk))

    # Prepend the output path to the full path
    full_path = os.path.join(output_path, full_path)

    # Generate the full_path
    if not os.path.exists(full_path): os.makedirs(full_path)

    # First, save the plotted figure in its specific place
    results.pop('fig').savefig(os.path.join(full_path, 'plot.pdf'))

    # Save the contents of the dictionary as JSON
    with open(os.path.join(full_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent = 4)

    if show_plot: plt.show()

if __name__ == '__main__':
    pass
