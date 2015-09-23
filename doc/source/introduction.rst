*************************
Introduction to DFT Tools
*************************

A number of codes implementing DFT and it's flavors is available in the
web, see `Wikipedia <https://en.wikipedia.org/wiki/List_of_quantum_chemistry_and_solid-state_physics_software>`_
for example. The developers of these codes are usually scientists who
never aimed to develop a user-friendly application mainly because they
are not get paid for that. Thus, to be able to use such codes one has to
master several tools, among which is data post-processing and presentation.

An average DFT code produces a set of text and binary data during the run.
Typically, the data cannot be plotted directly and one needs a program
to collect this data and present it. Here is an example of a Quantum
Espresso band structure:

.. literalinclude:: examples/band-structure/1-qe-output/plot.py.data
   :lines: 144-154

With DFT Tools it can be plotted as easy as is following script:

.. plot:: examples/band-structure/1-qe-output/plot.py
      :include-source:

Not only the band structure can be plotted, but atomic structure, data
on the grid, etc., see :doc:`examples <examples/index>`.
