# -- Load built-int modules
from enum import Enum

# -- String definition of the possible Channel names
str_g5   = '(g5)_(0mp)_(pseudoscalar)_(pp)'
str_gi   = '(gi)_(1mm)_(vector)_(vv)'
str_gig5 = '(gig5)_(1pp)_(axial_plus)_(aa)'
str_gigj = '(gigj)_(1pm)_(axial_minus)'
str_1    = '(1)_(0pp)_(scalar)_(ss)'
str_axps = '(axps)_(ap)'
str_a0a0 = '(aa)_(a0a0)_(g0g5)'

class Channel(Enum):
    """ Enumeration containing all relevant Fastsum channels """

    # All possible channels contained in the enumeration
    PSEUDOSCALAR = [85]
    VECTOR       = [34, 51, 68]
    AXIAL_PLUS   = [119, 136, 153]
    SCALAR       = [0]
    AXIAL_MINUS  = [221, 238, 255]
    AXIAL_PSEUD  = [101]
    A0_A0        = [102]

    @classmethod
    def from_str(cls, name: str):
        """ Create a Channel object using a string identifier """

        # Make the string lowercase and format it
        lower = '(' + name.lower() + ')'

        if   lower in str_g5:
            return cls.PSEUDOSCALAR
        elif lower in str_gi:
            return cls.VECTOR
        elif lower in str_gig5: 
            return cls.AXIAL_PLUS
        elif lower in str_gigj: 
            return cls.AXIAL_MINUS
        elif lower in str_1:    
            return cls.SCALAR
        elif lower in str_axps:    
            return cls.AXIAL_PSEUD
        elif lower in str_a0a0:    
            return cls.A0_A0
        else: 
            raise ValueError(
                f'{name = } is not a valid Channel. \n' + \
                f'\tList: {[c.name for c in Channel]}'
            )

    def to_latex(self):
        """ Latex representation of the channel """
        if   self == Channel.PSEUDOSCALAR:
            return r'\gamma_5\,(0^{-+})'
        elif self == Channel.VECTOR:
            return r'\gamma_i\,(1^{--})'
        elif self == Channel.AXIAL_PLUS:
            return r'\gamma_i\gamma_5\,(1^{++})'
        elif self == Channel.AXIAL_MINUS:
            return r'\gamma_i\gamma_j\,(1^{+-})'
        elif self == Channel.SCALAR:
            return r'1\,(0^{++})'
        elif self == Channel.AXIAL_PSEUD:
            return r'\gamma_0\gamma_5-\gamma_5'
        elif self == Channel.A0_A0:
            return r'\gamma_0\gamma_5-\gamma_0\gamma_5'
        else:
            return

    def __str__(self):
        """ String representation of the channel """

        if   self == Channel.PSEUDOSCALAR:
            return 'g5_0mp'
        elif self == Channel.VECTOR:
            return 'gi_1mm'
        elif self == Channel.AXIAL_PLUS:
            return 'gig5_1pp'
        elif self == Channel.AXIAL_MINUS:
            return 'gigj_1pm'
        elif self == Channel.SCALAR:
            return '1_0pp'
        elif self == Channel.AXIAL_PSEUD:
            return 'g0g5-g5'
        elif self == Channel.A0_A0:
            return 'g0g5-g0g5'
        else:
            'None'

    def __repr__(self) -> str:
        """ Representation of the Channel object. """
        return f'<Channel: {self.name} - {self.value}>'

class Flavour(Enum):
    """ Enumeration containing all the relevant Fastum flavour combinations 

    Right now, it only contains mesonic combinations, it should be straightforward
    to extend it to baryons.
    """

    # All possible flavour combinations in the enumeration
    UU = 'uu'; SS = 'ss'
    US = 'us'; SC = 'sc'
    UC = 'uc'; CC = 'cc'

    @classmethod
    def from_str(cls, name: str):
        """ String generator of the enumeration. Use Flavour(name) instead. """
        return cls(name)

    def __str__(self) -> str:
        """ String representation of a Flavour object """
        return self.value

    def __repr__(self) -> str:
        """ Repr of a Flavour object """
        return f'<Flavour: {self.value}>'
    
if __name__ == '__main__':
    pass
