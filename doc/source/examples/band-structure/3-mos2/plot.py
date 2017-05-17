from dfttools.simple import parse
from dfttools import presentation

from matplotlib import pyplot

with open("plot.py.data",'r') as f:
    
    # Retrieve the last band structure from the file
    bands = parse(f, "band-structure")
    
    # Convert to a grid
    grid = bands.as_grid()
    
    # Plot both
    presentation.matplotlib_bands_density(bands, pyplot.gca(), 200, energy_range = (-2, 2), label = "bands")
    presentation.matplotlib_bands_density(grid,  pyplot.gca(), 200, energy_range = (-2, 2), label = "grid")
    pyplot.legend()
    pyplot.show()
