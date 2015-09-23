from dfttools.types import Basis, Grid
from dfttools import presentation

from matplotlib import pyplot
from numericalunits import eV
import numpy

# A reciprocal basis
basis = Basis((1,1,1,0,0,-0.5), kind = 'triclinic', meta = {"Fermi": 0})

# Grid shape
shape = (50,50,1)

# A dummy grid with correct grid coordinates and empty grid values
grid = Grid(
    basis,
    tuple(numpy.linspace(0,1,x, endpoint = False)+.5/x for x in shape),
    numpy.zeros(shape+(2,), dtype = numpy.float64),
)

# Calculate graphene band
k = grid.cartesian()*numpy.pi/3.**.5*2
e = (1+4*numpy.cos(k[...,1])**2 + 4*numpy.cos(k[...,1])*numpy.cos(k[...,0]*3.**.5))**.5*eV

# Set the band values
grid.values[...,0] = -e
grid.values[...,1] = e

presentation.matplotlib_bands_density(grid, pyplot.gca(), 200, energy_range = (-1, 1))
pyplot.show()
