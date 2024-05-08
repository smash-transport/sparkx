# Changelog

All notable changes to this project will be documented in this file.

The versioning of the codebase is inspired by [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a version number `X.Y.Z`, where

* `X` is incremented for major changes (e.g. large backwards incompatible updates),
* `Y` is incremented for minor changes (e.g. external pull-request that adds one feature) and
* `Z` is incremented for the indication of a bug fix or other very small changes that are not backwards incompatible.

The main categories for changes in this file are:

* `Input / Output` for all, in particular breaking, changes, fixes and additions to the in- and output files;
* `Added` for new features;
* `Changed` for changes in existing functionality;
* `Fixed` for any bug fixes;
* `Removed` for now removed features.

A `Deprecated` section could be added if needed for soon-to-be removed features.

## v1.2.1-Newton
Date: 2024-05-08

### Fixed
* Jetscape: Fix for Jetscape charged particle filtering

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v1.2.0...v1.2.1)

## v1.2.0-Newton
Date: 2024-04-21

### Added

* QCumulantFlow: Differential flow for two and four particle cumulants
* Tests: Tests for Histogram, Utilities, CentralityClasses, Oscar, Jetscape, JetAnalysis, Flow, Lattice3D, Filter, EventCharacteristics
* JetAnalysis: Add switch to write only charged associated particles
* JetAnalysis: Add hole subtraction for particles with negative status
* Histogram: Add possibility to add values with weights and possibility to create probability densities
* Histogram: Choose which columns to be printed in the output file
* EventCharacteristics: Add option for different radial weight in eccentricities

### Fixed

* Oscar/Jetscape: Fix bug in `remove_particle_species` filter function if only one PDG id is given (issue #182)
* Oscar/Jetscape: Fix bug in `pseudorapidity_cut` filter function if only one cut value is given (issue #188)
* Oscar/Jetscape: Fix bug in keyword argument parsing
* ReactionPlaneFlow: Fix bug in particle weight
* Oscar: Fix various bugs in printout
* EventCharacteristics: Fix sign of the eccentricity
* Jetscape: Fix bug in keyword argument filtering, where filters were not applied to the last event

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v1.1.1...v1.2.0)

## v1.1.1-Newton
Date: 2023-12-20

### Added

* Tests: Add tests for Particle class

### Fixed

* Particle: Fix bug in tests for `nan` values
* Particle: Casts to `int` and `short` types added, which fixes the crash of `read_jet_data` in JetAnalysis (issue #174)

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v1.1.0...v1.1.1)

## v1.1.0-Newton
Date: 2023-12-07

### Added

* Import with `from sparkx import *` is now possible
* EventCharacteristics: Possibility to smear (Gaussian or covariant) energy, baryon, charge and strangeness densities in Milne and Minkowski coordinates with writing to a file for subsequent hydro evolution
* Histogram: Method to set the systematic error
* Lattice3D: Add covariant smearing of densities for particles
* Oscar: Add option to apply filters while read-in to improve RAM usage
* Oscar: Add spacetime cut
* Particle: Add strangeness, spin and spin_degeneracy functions 
* Tests: Add automatic tests for the Particle class
* GenerateFlow: Add functionality to generate flow with k-particle correlations
* LeeYangZeroFlow: Add beta version of integrated and differential flow analysis with the Lee-Yang zero method
* QCumulantFlow: Add beta version of integrated flow analysis with the Q-Cumulant method
* PCAFlow: Add flow analysis with PCA method
* CentralityClasses: Class to determine the centrality classes from a set of events

### Fixed

* Histogram: Correct error handling when reweighting or averaging over histograms
* Oscar/Jetscape: Fix bug when reading in a single event from input file
* Jetscape: Fix asymmetric pseudorapidity cut

### Changed

* Histogram: Can write multiple histograms to file now
* Oscar/Jetscape: Handling of PDG id's which are not present in the `particle` package is moved to the Particle class
* Oscar/Jetscape: Improved writing to file
* Particle: Particle construction is now done within the constructor by providing a format and an array of values
* Particle: Internal structure is now a numpy float array
* Particle: Functions using `PDGID` from the `particle` package handle now the case if the PDG ID is unknown to the package
* Particle: Returns `nan` if the quantity is not known or can not be computed

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v1.0.2...v1.1.0)

## v1.0.2-Newton
Date: 2023-06-26

### Fixed

* Bug fix in Histogram class for values outside the bin range

### Changed

* Histogram printout to the terminal

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v1.0.1...v1.0.2)

## v1.0.1-Newton
Date: 2023-06-19

### Added

* Read in of X-SCAPE hadron output

### Fixed

* JetAnalysis class can be imported now
* Jet analysis can be performed with different jet finding algorithms without throwing errors

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v1.0.0...v1.0.1)

## v1.0.0-Newton
Date: 2023-06-14

**[First public version of SPARKX](https://github.com/smash-transport/sparkx/releases/tag/v1.0.0)**

### Added

* SMASH / Jetscape particle data read in and processing
* Anisotropic flow analysis (reaction plane, event plane, scalar product methods)
* Jet analysis (wrapper for fastjet)
* Initial state characterization via eccentricities
* Histogram class
