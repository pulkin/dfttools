from dfttools.types import Basis, Cell
from dfttools.presentation import svgwrite_unit_cell

from numericalunits import angstrom as a

si_basis = Basis((3.9*a/2, 3.9*a/2, 3.9*a/2, .5,.5,.5), kind = 'triclinic')
si_cell = Cell(si_basis, (.5,.5,.5), 'Si')
svgwrite_unit_cell(si_cell, 'output.svg', size = (440,360), show_cell = True)
