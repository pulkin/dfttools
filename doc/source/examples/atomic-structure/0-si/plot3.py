cubic_cell = si_cell.supercell(
    (1,-1,1),
    (1,1,-1),
    (-1,1,1),
)
svgwrite_unit_cell(cubic_cell, 'output3.svg', size = (440,360), show_cell = True, camera = (1,1,1))
