from dfttools.types import Basis, UnitCell
from dfttools import presentation

from matplotlib import pyplot
from numericalunits import eV
import numpy

# A reciprocal basis
basis = Basis((1,1,1,0,0,-0.5), kind = 'triclinic', meta = {"Fermi": 0})

# G-K path
kp = numpy.linspace(0,1,100)[:,numpy.newaxis] * numpy.array(((1./3,2./3,0),))

# A dummy grid UnitCell with correct kp-path
bands = UnitCell(
    basis,
    kp,
    numpy.zeros((100,2), dtype = numpy.float64),
)

# Calculate graphene band
k = bands.cartesian()*numpy.pi/3.**.5*2
e = (1+4*numpy.cos(k[...,1])**2 + 4*numpy.cos(k[...,1])*numpy.cos(k[...,0]*3.**.5))**.5*eV

# Set the band values
bands.values[...,0] = -e
bands.values[...,1] = e

# Assign some weights
weights = bands.values.copy()
weights -= weights.min()
weights /= weights.max()

# Prepare axes
ax_left  = pyplot.subplot2grid((1,3), (0, 0), colspan=2)
ax_right = pyplot.subplot2grid((1,3), (0, 2))
    
# Plot bands
p = presentation.matplotlib_bands(bands,ax_left,weights = weights)
presentation.matplotlib_bands_density(bands, ax_right, 100, orientation = 'portrait')
presentation.matplotlib_bands_density(bands, ax_right, 100, orientation = 'portrait', weights = weights, use_fill = True, color = "#AAAAFF")

ax_right.set_ylabel('')
pyplot.colorbar(p)
pyplot.show()
