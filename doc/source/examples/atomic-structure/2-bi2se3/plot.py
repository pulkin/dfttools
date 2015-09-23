from dfttools.presentation import svgwrite_unit_cell
from dfttools.simple import parse

# Parse
with open("plot.py.data", "r") as f:
    cell = parse(f, "unit-cell")
    
# Draw
svgwrite_unit_cell(cell, 'output.svg', size = (440,360), camera = (1,0,0))
