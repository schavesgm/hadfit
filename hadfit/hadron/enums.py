# -- Load built-int modules
from enum import Enum

# -- String definition of the possible Channel names
str_g5   = '(g5)_(0mp)_(pseudoscalar)'
str_gi   = '(gi)_(1mm)_(vector)'
str_gig5 = '(gig5)_(1pp)_(axial_plus)'
str_gigj = '(gigj)_(1pm)_(axial_minus)'
str_1    = '(1)_(0pp)_(scalar)'

class Channel(Enum):
    """ Enumeration containing all relevant Fastsum channels """

    # All possible channels contained in the enumeration
    PSEUDOSCALAR: list = [85]
    VECTOR:       list = [34, 51, 68]
    AXIAL_PLUS:   list = [119, 136, 153]
    SCALAR:       list = [0]
    AXIAL_MINUS:  list = [221, 238, 255]

    @classmethod
    def from_str(cls, name: str):
        """ Create a Channel object using a string identifier """

        # Make the string lowercase and format it
        lower = '(' + name.lower() + ')'

        if   lower in str_g5:   return cls.PSEUDOSCALAR
        elif lower in str_gi:   return cls.VECTOR
        elif lower in str_gig5: return cls.AXIAL_PLUS
        elif lower in str_gigj: return cls.AXIAL_MINUS
        elif lower in str_1:    return cls.SCALAR
        else: 
            raise ValueError(
                f'{name = } is not a valid Channel. \n' + \
                f'\tList: {[c.name for c in Channel]}'
            )

    def to_latex(self):
        """ Latex representation of the channel """
        if   self == Channel.PSEUDOSCALAR: return r'\gamma_5\,(0^{-+})'
        elif self == Channel.VECTOR:       return r'\gamma_i\,(1^{--})'
        elif self == Channel.AXIAL_PLUS:   return r'\gamma_i\gamma_5\,(1^{++})'
        elif self == Channel.AXIAL_MINUS:  return r'\gamma_i\gamma_j\,(1^{+-})'
        elif self == Channel.SCALAR:       return r'1\,(0^{++})'
        else:                              return

    def __str__(self) -> str:
        """ String representation of the channel """
        if   self == Channel.PSEUDOSCALAR: return 'g5_0mp'
        elif self == Channel.VECTOR:       return 'gi_1mm'
        elif self == Channel.AXIAL_PLUS:   return 'gig5_1pp'
        elif self == Channel.AXIAL_MINUS:  return 'gigj_1pm'
        elif self == Channel.SCALAR:       return '1_0pp'

    def __repr__(self) -> str:
        """ Representation of the Channel object. """
        return f'<Channel: {self.name} - {self.value}>'

class Flavour(Enum):
    """ Enumeration containing all the relevant Fastum flavour combinations 

    Right now, it only contains mesonic combinations, it should be straightforward
    to extend it to baryons.
    """

    # All possible names in the enumeration
    UU: str = 'uu'; SS: str = 'ss'
    US: str = 'us'; SC: str = 'sc'
    UC: str = 'uc'; CC: str = 'cc'

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
