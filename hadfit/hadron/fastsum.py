# -- Load built-in modules
import os, re
from typing import Callable, Union

# -- Import some third party modules
import numpy as np

# -- Load user-defined classes
from .hadron import Hadron
from .enums  import Flavour
from .enums  import Channel

# Function to filter Fastsum data using Flavour and Channel
def fastsum_filter(flavour: Flavour, channel: Channel) -> Callable[[str], bool]:
    """ Function used to obtain the relevant files from a set of different
    Fastsum correlation functions files. The filter uses the flavour and channel
    as filters. It should be generisable to any naming conventions.

    This is a second-order function, which implies that is is wrapper around
    another function that modifies its behaviour.

    --- Parameters:
    flavour: Flavour
        Flavour combinations to be searched.
    channel: Channel
        Channel to be searched

    --- Returns
    Callable[[str], bool]
        Actual function used to filter the data.
    """
    # Generate the channel helper list
    helper = [f'g{c}' for c in channel.value]
    
    def actual_filter(file_name: str) -> bool:
        ''' Actual filter function '''
        return str(flavour) in file_name and any([str(c) in file_name for c in helper])

    return actual_filter

def hadron_from_fastsum(path: str, flavour: Union[str, Flavour], channel: Union[str, Channel]) -> Hadron:
    """ Function that generates a Hadron using Fastsum correlation function. 
    It matches the Fastsum naming convention.
    
    --- Parameters
    path: str
        Path containin the correlation function files. Should be on the form
        x/{Nt}x32_{source}, where x is any path, Nt is the number of time points
        and source is the sources used in the computation.
    flavour: Union[str, Flavour]
        Flavour object or name defininig the Flavour combination to be extracted.
    channel: Union[str, Channel]
        Channel object or name defining the Channel combination to be extract

    --- Returns
    Hadron
        Hadron object containing the Fastsum correlation function of the flavour and channel.

    --- Example: hadron_from_fastsum('./data/64x32_ll/', 'uu', 'g5')
        Retrieves the pseudoscalar 'uu' meson of Nt = 64 and local-local sources from the
        directory ./data/64x32_ll.
    """

    # Assert the path exists
    assert os.path.exists(path), f'{path = } does not exist in system'

    # Save the path where the data is stored
    path = path if path[-1] != '/' else path[:-1]

    # Generate the flavour and channel objects used in the file
    flavour = flavour if isinstance(flavour, Flavour) else Flavour.from_str(flavour)
    channel = channel if isinstance(channel, Channel) else Channel.from_str(channel)

    # Get all relevant files in the path
    files = list(filter(fastsum_filter(flavour, channel), os.listdir(path)))

    # Get all relevant information from the path to define the Meson
    info = re.match(r'(\d+)x(\d+)_(\w+)', os.path.basename(path))

    # Get the number of points in the non-integrated direction
    Nk = int(info.group(1))

    # Iterate for each of the files to generate a Hadron object
    for it, fastsum_file in enumerate([os.path.join(path, f) for f in files]):

        # Load the data from the file
        data = np.loadtxt(fastsum_file, skiprows = 1)[:,1]

        # Generate a hadron with the current data
        if it == 0:
            hadron = Hadron(data, Nk)
        else:
            hadron = hadron + Hadron(data, Nk)

    # Set some information in the hadron dictionary
    hadron.set_info(flavour = flavour, channel = channel)
    hadron.set_info(Ns = int(info.group(2)), sources = info.group(3))

    # Set the correct name of the hadron
    hadron.name = f'{channel}_{flavour}_{info.group(3)}'

    return hadron

if __name__ == '__main__':
    pass
