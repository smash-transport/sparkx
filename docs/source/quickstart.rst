.. _quickstart:

Quickstart
==========

This is a short guide to get you started with the SPARKX package.
The package is designed to assist users of the `SMASH <https://smash-transport.github.io/>`_ 
and `JETSCAPE/X-SCAPE <https://jetscape.org/>`_ codes to analyze the output of the simulation data.

For loading and filtering of the SMASH or JETSCAPE output data, we have 
implemented the following classes:

* `Oscar <classes/Oscar/index.html>`_: Reads in various Oscar formats and provides methods to filter the particles.
* `Jetscape <classes/Jetscape/index.html>`_: Reads in the hadron or parton output of JETSCAPE/X-SCAPE and provides methods to filter the particles.

For the further analysis of the events with individual particle objects, we have implemented the following classes:

* `CentralityClasses <classes/CentralityClasses/index.html>`_: Provides methods to calculate the centrality classes from a given set of the events.
* `EventCharacteristics <classes/EventCharacteristics/index.html>`_: Provides methods to calculate the event characteristics like eccentricities and it can compute energy densities as input for hydrodynamical simulations.
* `EventPlaneFlow <classes/flow/EventPlaneFlow/index.html>`_: Provides methods to calculate the event plane flow.
* `LeeYangZeroFlow <classes/flow/LeeYangZeroFlow/index.html>`_: Provides methods to calculate the Lee-Yang zero flow.
* `PCAFlow <classes/flow/PCAFlow/index.html>`_: Provides methods to calculate the flow using principal component analysis.
* `QCumulantFlow <classes/flow/QCumulantFlow/index.html>`_: Provides methods to calculate the flow using Q-cumulants.
* `ReactionPlaneFlow <classes/flow/ReactionPlaneFlow/index.html>`_: Provides methods to calculate the flow using the reaction plane method.
* `ScalarProductFlow <classes/flow/ScalarProductFlow/index.html>`_: Provides methods to calculate the flow using the scalar product method.
* `Histogram <classes/Histogram/index.html>`_: Provides methods to create histograms.
* `Jackknife <classes/Jackknife/index.html>`_: Provides methods to calculate delete-d jackknife errors.
* `JetAnalysis <classes/JetAnalysis/index.html>`_: Provides methods to find jets in the events.
* `Lattice3D <classes/Lattice3D/index.html>`_: Provides methods to handle 3D lattices.
* `MultiParticlePtCorrelations <classes/MultiParticlePtCorrelations/index.html>`_: Provides methods to calculate multi-particle transverse momentum correlations.
* `Particle <classes/Particle/index.html>`_: Provides methods to handle individual particle (hadron or parton) objects.

For more information on the classes and their methods, please refer to the `classes documentation <classes/index.html>`_.

