from dfttools.types import Basis, UnitCell
from dfttools.presentation import svgwrite_unit_cell

from numericalunits import angstrom as a

graphene_basis = Basis(
    (2.46*a, 2.46*a, 6.7*a, 0,0,.5),
    kind = 'triclinic'
)

# Unit cell
graphene_cell = UnitCell(graphene_basis, (
    (1./3,1./3,.5),
    (2./3,2./3,.5),
), ('C','C'))

# A top layer
l1 = graphene_cell.supercell(
    (9,1,0),
    (-1,10,0),
    (0,0,1)
)

# A bottom layer
l2 = graphene_cell.supercell(
    (6,5,0),
    (-5,11,0),
    (0,0,1)
)

# Make the basis fit
l2.vectors[:2] = l1.vectors[:2]

# Draw
svgwrite_unit_cell(l1.stack(l2, vector='z').repeated(2,2,1), 'output.svg', size = (440,360), camera = (0,0,-1), camera_top = (0,1,0), show_atoms = False)
