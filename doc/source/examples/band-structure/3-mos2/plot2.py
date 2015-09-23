from dfttools.simple import parse
from dfttools import presentation

import numpy
from matplotlib import pyplot

with open("plot.py.data",'r') as f:
    
    # Retrieve the last band structure from the file
    bands = parse(f, "band-structure")[-1]
    
    # Convert to a grid
    grid = bands.as_grid()
    
    # Interpolate
    kp_path = numpy.linspace(0,1)[:,numpy.newaxis] * ((1./3,2./3,0),)
    bands = grid.interpolate_to_cell(kp_path)
    
    # Plot
    presentation.matplotlib_bands(bands, pyplot.gca())
    pyplot.show()
