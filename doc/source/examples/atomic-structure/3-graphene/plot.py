from dfttools.types import Basis, Cell
from dfttools.presentation import svgwrite_unit_cell

from numericalunits import angstrom as a

graphene_basis = Basis(
    (2.46*a, 2.46*a, 6.7*a, 0,0,.5),
    kind = 'triclinic'
)

# Unit cell
graphene_cell = Cell(graphene_basis, (
    (1./3,1./3,.5),
    (2./3,2./3,.5),
), ('C','C'))

# Moire matching vectors
moire = [1, 26, 6, 23]

# A top layer
l1 = graphene_cell.supercell(
    (moire[0],moire[1],0),
    (-moire[1],moire[0]+moire[1],0),
    (0,0,1)
)

# A bottom layer
l2 = graphene_cell.supercell(
    (moire[2],moire[3],0),
    (-moire[3],moire[2]+moire[3],0),
    (0,0,1)
)

# Make the basis fit
l2.vectors[:2] = l1.vectors[:2]

# Draw
svgwrite_unit_cell(l1.stack(l2, vector='z'), 'output.svg', size = (440,360), camera = (0,0,-1), camera_top = (0,1,0), show_atoms = False)
