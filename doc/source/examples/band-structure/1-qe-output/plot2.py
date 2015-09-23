from dfttools.simple import parse
from dfttools import presentation

from matplotlib import pyplot

with open("plot.py.data",'r') as f:

    # Read bands data
    bands = parse(f, "band-structure")[0]

    # Prepare axes
    ax_left  = pyplot.subplot2grid((1,3), (0, 0), colspan=2)
    ax_right = pyplot.subplot2grid((1,3), (0, 2))
    
    # Plot bands
    presentation.matplotlib_bands(bands,ax_left)
    presentation.matplotlib_bands_density(bands, ax_right, 100, orientation = 'portrait')
    ax_right.set_ylabel('')
    pyplot.show()

