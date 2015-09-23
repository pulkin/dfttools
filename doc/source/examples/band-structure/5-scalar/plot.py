from dfttools.types import Basis, Grid
from dfttools import presentation

from numericalunits import angstrom
from matplotlib import pyplot
import numpy

grid = Grid(
    Basis((1*angstrom,1*angstrom,1*angstrom,0,0,-0.5), kind = 'triclinic'),
    (
        numpy.linspace(0,1,30,endpoint = False),
        numpy.linspace(0,1,30,endpoint = False),
        numpy.linspace(0,1,30,endpoint = False),
    ),
    numpy.zeros((30,30,30)),
)
grid.values = numpy.prod(numpy.sin(grid.explicit_coordinates()*2*numpy.pi), axis = -1)

presentation.matplotlib_scalar(grid, pyplot.gca(), (0.1,0.1,0.1), 'z', show_cell = True)
pyplot.show()
