.. _quickstart:

Quickstart
==========

This guide will get you started with SPARKX quickly. 
SPARKX is designed to help users of `SMASH <https://smash-transport.github.io/>`_ 
and `JETSCAPE/X-SCAPE <https://jetscape.org/>`_ codes analyze output simulation data with ease.

Key Components of SPARKX
------------------------

This is a general overview of the main components of SPARKX.
SPARKX is designed to be modular, allowing users to load, filter, and analyze data
from different sources. The following diagram illustrates the main components:

.. mermaid::

   %%{ init : { "flowchart": { "wrap": true } } }%%
   flowchart LR
       A[Loading / Filtering Data</br>Oscar / Jetscape</br>Custom Input] -->|Particle Objects List| B[Analyze Data]

The following classes will be useful for loading, filtering, and analyzing your SMASH or JETSCAPE/X-SCAPE data.

Data Loading and Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~

These classes are used to load and filter data from different sources:

* `Jetscape <classes/Jetscape/index.html>`_: Reads hadron or parton output from JETSCAPE/X-SCAPE and allows particle filtering.
* `Oscar <classes/Oscar/index.html>`_: Reads various Oscar data formats and provides particle filtering methods.

Event Analysis
~~~~~~~~~~~~~~

For more advanced analysis with individual particle objects, 
SPARKX offers a range of classes for calculating centrality, flow, and more:

* `BulkObservables <classes/BulkObservables/index.html>`_: Calculates bulk observables like :math:`\frac{dN}{dy}` and midrapdity yields.
* `CentralityClasses <classes/CentralityClasses/index.html>`_: Calculates centrality classes for a given set of events.
* `EventCharacteristics <classes/EventCharacteristics/index.html>`_: Calculates event characteristics like eccentricities and energy densities, which can be used as input for hydrodynamical simulations.
* `JetAnalysis <classes/JetAnalysis/index.html>`_: Finds jets in the events using `FastJet <https://github.com/scikit-hep/fastjet>`_.
* `MultiParticlePtCorrelations <classes/MultiParticlePtCorrelations/index.html>`_: Calculates multi-particle transverse momentum correlations.

Flow Calculations
~~~~~~~~~~~~~~~~~

SPARKX provides several classes for calculating anisotropic flow:

* `EventPlaneFlow <classes/flow/EventPlaneFlow/index.html>`_: Calculates event plane flow.
* `LeeYangZeroFlow <classes/flow/LeeYangZeroFlow/index.html>`_: Calculates the Lee-Yang zero flow.
* `PCAFlow <classes/flow/PCAFlow/index.html>`_: Calculates flow using principal component analysis.
* `QCumulantFlow <classes/flow/QCumulantFlow/index.html>`_: Calculates flow using Q-cumulants.
* `ReactionPlaneFlow <classes/flow/ReactionPlaneFlow/index.html>`_: Calculates flow using the reaction plane method.
* `ScalarProductFlow <classes/flow/ScalarProductFlow/index.html>`_: Calculates flow using the scalar product method.

Additional Tools
~~~~~~~~~~~~~~~~

* `Histogram <classes/Histogram/index.html>`_: Creates histograms.
* `Jackknife <classes/Jackknife/index.html>`_: Calculates delete-d jackknife errors.
* `Lattice3D <classes/Lattice3D/index.html>`_: Handles 3D lattices.
* `Particle <classes/Particle/index.html>`_: Handles individual particle (hadron or parton) objects.

For detailed information on these classes and their methods, see the `classes documentation <classes/index.html>`_.

Installation
------------
For information on how to install the package, please refer to the `installation documentation <install.html>`_.

Basic Usage Example
-------------------

Once installed, here's a simple example to load data and perform some basic filtering and analysis:

.. code-block:: python

    from sparkx import Oscar
    from sparkx import Histogram

    # Load data using the Oscar class and apply some filters, return Particle objects
    all_filters={'charged_particles': True, 'pT_cut': (0.15, 3.5), 'pseudorapidity_cut': 0.5}
    data = Oscar("./particle_lists.oscar", filters=all_filters).particle_objects_list()

    # data contains now the lists of Particle objects for each event that passed the filters
    # assume that we want to create a charged hadron transverse momentum histogram
    # extract the transverse momentum of each particle for all events
    pT = [particle.pT_abs() for event in data for particle in event]

    # create a histogram of the transverse momentum with 10 bins between 0.15 and 3.5 GeV
    hist = Histogram(bin_boundaries=(0.15, 3.5, 10))
    hist.add_value(pT) # add the data to the histogram
    hist.statistical_error() # calculate the statistical errors assuming Poisson statistics
    hist.scale_histogram(1./(len(data)*hist.bin_width())) # normalize the histogram to the number of events and divide by the bin width

    # define the names of the columns in the output file
    column_labels = [{'bin_center': 'pT [GeV]',
                    'bin_low': 'pT_low [GeV]',
                    'bin_high': 'pT_high [GeV]',
                    'distribution': '1/N_ev * dN/dpT [1/GeV]',
                    'stat_err+': 'stat_err+',
                    'stat_err-': 'stat_err-',
                    'sys_err+': 'sys_err+',
                    'sys_err-': 'sys_err-'}]
    # write the histogram to a file
    hist.write_to_file('pT_histogram.dat', hist_labels=column_labels)

You can find more examples in the documentation for the individual classes.

Troubleshooting
---------------

If you encounter issues:

* Ensure all dependencies are installed with the correct versions.
* Check the `GitHub repository <https://github.com/smash-transport/sparkx/issues>`_ to report bugs or view open issues.
