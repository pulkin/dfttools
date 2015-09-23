from dfttools.parsers.openmx import bands
from dfttools import presentation

from matplotlib import pyplot

with open("plot.py.data",'r') as f:

    # Read bands data
    b = bands(f.read()).bands()

    # Plot bands
    presentation.matplotlib_bands(b,pyplot.gca())
    pyplot.show()

