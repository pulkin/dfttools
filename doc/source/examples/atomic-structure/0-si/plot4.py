slab_cell = cubic_cell.repeated(5,5,3).isolated(0,0,10*a)
svgwrite_unit_cell(slab_cell, 'output4.svg', size = (440,360), camera = (1,1,1))
