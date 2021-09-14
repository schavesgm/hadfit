# -- Import built-in modules
import re

# -- Import third-party modules
import pytest

# -- Import user-defined modules
from hadfit import Flavour
from hadfit import Channel

# -- Tests for the Flavour enumeration {{{
class TestFlavour:

    def test_all_str(self):
        """ All Flavour enums should contain strings. """
        assert all([isinstance(f.value, str) for f in Flavour])

    def test_str_repr(self):
        """ The string representation must be equal to the value. """
        assert all([str(f) == f.value for f in Flavour])

    def test_from_str(self):
        """ Calling Flavour() and using Flavour.from_str() must be equivalent. """
        assert all([Flavour(f.value) == Flavour.from_str(f.value) for f in Flavour])
# -- }}}

# -- Tests for the Channel enumeration {{{
class TestChannel:

    def test_all_lists(self):
        """ All Channel enums possibilities should be a list of ints. """
        for channel in Channel:
            assert isinstance(channel.value, list)
            assert all([isinstance(i, int) for i in channel.value])

    def test_channel_values(self):
        """ The maximum possible channel value is 255. """
        for channel in Channel:
            assert all([0 <= i < 256 for i in channel.value])

    def test_latex_output(self):
        """ Every channel should have its latex output. """
        assert all([c.to_latex() is not None for c in Channel])

    def test_output_regex(self):
        """ The string representation of a Channel should be in a proper way. """
        assert all([re.match(r'\w+_\d[pm]{2}$', str(c)) for c in Channel])
# -- }}}
