===============
Getting started
===============

DFT Tools package is written in python. To be able to use it you have to
download and install it locally.

----------
Installing
----------

The easiest way to install DFT Tools is to use `pip <https://pypi.python.org>`_::

    $ pip install dfttools

For a local user it can be done with a ``--user`` option::

    $ pip install dfttools --user
    
You may also download the package and use the bundled setup.py::

    $ python setup.py install
    $ python setup.py install --user
    
The package explicitly requires `numpy <https://scipy.org/>`_ and
`numericalunits <https://pypi.python.org/pypi/numericalunits/>`_ which
will be automatically installed if not yet present in your system. Also,
it is recommended to install `matplotlib <https://matplotlib.org/>`_ and
`svgwrite <https://pypi.python.org/pypi/svgwrite/>`_ for data visualisation
and `scipy <https://scipy.org/>`_ to be able to use some other functions.
All packages are available through pip::

    $ pip install matplotlib
    $ pip install svgwrite
    $ pip install scipy

-----
Using
-----

Once installed you may start using it by importing the package in your
python script::

    import dfttools
    
or just using one of the pre-set scripts::

    $ dft-plot-bands my_dft_output_file
    
