from dfttools.simple import parse
from dfttools import presentation

from matplotlib import pyplot

with open("plot.py.data",'r') as f:

    # Read bands data
    bands = parse(f, "band-structure")

    # Plot bands
    presentation.matplotlib_bands(bands,pyplot.gca())
    pyplot.show()

