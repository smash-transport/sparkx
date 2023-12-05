# Changelog

All notable changes to this project will be documented in this file.

The versioning of the codebase is inpired by [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a version number `X.Y.Z`, where

* `X` is incremented for major changes (e.g. large backwards incompatible updates),
* `Y` is incremented for minor changes (e.g. external pull-request that adds one feature) and
* `Z` is incremented for the indication of a bug fix or other very small changes that are not backwards incompatible.

The main categories for changes in this file are:

* `Input / Output` for all, in particular breaking, changes, fixes and additions to the in- and output files;
* `Added` for new features;
* `Changed` for changes in existing functionality;
* `Fixed` for any bug fixes;
* `Removed` for now removed features.

A `Depracted` section could be added if needed for soon-to-be removed features.

## Unreleased

### Added

* Import with `from sparkx import *` is now possible
* EventCharacteristics: Possibility to smear (Gaussian or covariant) energy, baryon, charge and strangeness densities in Milne and Minkowski coordinates with writing to a file for subsequent hydro evolution
* Histogram: Method to set the systematic error
* Lattice3D: Add covariant smearing of densities for particles
* Oscar: Add spacetime cut
* Particle: Add strangeness, spin and spin_degeneracy functions 
* GenerateFlow: Add functionality to generate flow with k-particle correlations
* LeeYangZeroFlow: Add beta version of integrated and differential flow analysis with the Lee-Yang zero method
* QCumulantFlow: Add beta version of integrated flow analysis with the Q-Cumulant method

### Fixed

* Histogram: Correct error handling when reweighting or averaging over histograms
* Oscar/Jetscape: Fix bug when reading in a single event from input file
* Jetscape: Fix asymmetric pseudorapidity cut

### Changed

* Histogram: Can write multiple histograms to file now
* Oscar/Jetscape: Handling of PDG id's which are not present in the `particle` package is moved to the Particle class
* Oscar/Jetscape: Improved writing to file
* Particle: Particle construction is now done within the constructor by providing a format and an array of values
* Particle: Functions using `PDGID` from the `particle` package handle now the case if the PDG ID is unknown to the package
* Particle: Returns `None` if the quantity is not known or can not be computed

# Removed


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
