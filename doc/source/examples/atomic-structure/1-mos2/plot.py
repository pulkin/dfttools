from dfttools.types import Basis, UnitCell
from dfttools.presentation import svgwrite_unit_cell

from numericalunits import angstrom as a

mos2_basis = Basis(
    (3.19*a, 3.19*a, 20*a, 0,0,.5),
    kind = 'triclinic'
)
d = 1.57722483162840/20

# Unit cell with 3 atoms
mos2_cell = UnitCell(mos2_basis, (
    (1./3,1./3,.5),
    (2./3,2./3,0.5+d),
    (2./3,2./3,0.5-d),
), ('Mo','S','S'))

# Rectangular supercell with 6 atoms
mos2_rectangular = mos2_cell.supercell(
    (1,0,0),
    (-1,2,0),
    (0,0,1)
)

# Rectangular sheet with a defect
mos2_defect = mos2_rectangular.normalized()
mos2_defect.discard((mos2_defect.values == "S") * (mos2_defect.coordinates[:,1] < .5) * (mos2_defect.coordinates[:,2] < .5))

# Prepare a sheet
mos2_sheet = UnitCell.stack(*((mos2_rectangular,)*3 + (mos2_defect,) + (mos2_rectangular,)*3), vector = 'y')

# Draw
svgwrite_unit_cell(mos2_sheet.repeated(10,1,1), 'output.svg', size = (440,360), camera = (1,1,0.3), camera_top = (0,0,1))
