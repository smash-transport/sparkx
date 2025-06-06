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

## Unpublished
Date: 2025-06-XX

### Fixed
* Oscar: Replace hardcoded comment line parsing with `_extract_integer_after_keyword` to support updated output format with ensemble info.

### Added
* Tests: Unit tests for the new comment parsing logic, including a test file in the updated Oscar format.

### Changed
* Particle: The rapidity is now calculated with an `arcsinh` function to avoid numerical issues. If this is not possible, it falls back to the old method using the `log` function.
* Histogram: The class does not throw a warning anymore if a value added to a histogram is outside the bin range. Instead, it simply ignores the value.
* Tests: Test BulkObservables rapidity distribution with massive particles in a more realistic way.

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v2.0.2...v2.1.0)

## v2.0.2-Chatelet
Date: 2025-03-12

### Fixed
* QCumulantFlow: Fix bug in differential identified hadron flow

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v2.0.1...v2.0.2)

## v2.0.1-Chatelet
Date: 2025-02-26

### Fixed
* Jackknife: Fix bug in the calculation of the jackknife error
* BulkObservables: Fix bug in the calculation of the integrated yields
* Documentation: Fix quickstart example

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v2.0.0...v2.0.1)

## v2.0.0-Chatelet
Date: 2024-12-18

### Added
* Particle: Add member functions mT, is_quark, is_lepton, has_up, has_down, has_strange, has_charm, has_bottom, has_top
* Filter: Add keep_up, keep_down, keep_charm, keep_bottom, keep_top, keep_hadrons, keep_leptons, keep_quarks, keep_mesons, keep_baryons, remove_photons filters
* Oscar: Add transverse mass cut to methods
* Jetscape: Add transverse mass cut to methods
* Added ParticleObjectStorer to store generic particle lists
* SPARKX checks now for static typing consistency
* Added support for the SMASH 3.2 feature of custom output format
* Add option to add two Oscar/Jetscape/ParticleObjectStorer instances while preserving the event order
* BulkObservables: Add a class for calculating spectra and integrated yields
* Oscar: Add function to extract the impact parameters
* Jetscape: Test if input file is complete by checking for 'sigmaGen' string in last line

### Changed
* Particle: Rename several methods for a more intuitive naming scheme. Renamed methods are:

| Old Method Name                     | New Method Name             |
|-------------------------------------|-----------------------------|
| momentum_rapidity_Y()               | rapidity()                  |
| spatial_rapidity()                  | spacetime_rapidity()        |
| spatial_rapidity_cut()              | spacetime_rapidity_cut()    |
| pt_abs()                            | pT_abs()                    |
| pt_cut()                            | pT_cut()                    |
| compute_mass_from_energy_momentum() | mass_from_energy_momentum() |
| compute_charge_from_pdg()           | charge_from_pdg()           |
| is_strange()                        | has_strange()               |

* Filter: Perform general clean up to reduce code duplications
* Filter: Rename strange_particles filter to keep_strange
* Changed class architecture to separate loader and storer classes
* Changed code formatter to `black`
* SPARKX requires now python versions >= 3.9

### Fixed
* Oscar/Jetscape: Improve writing speed for large file outputs in writer functions
* Histogram: Bugfix in the scaling function of Histogram
* Oscar: Bugfix in comment line identification 

[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v1.3.0...v2.0.0)


## v1.3.0-Newton
Date: 2024-07-25

### Added
* ReactionPlaneFlow: Returns also the angle, not only the flow value
* Histogram: Add possibility to add and remove bins
* Jetscape: Possibility to read in parton output files with optional `particletype` parameter in constructor
* Particle: Include changes for parton read in from Jetscape class, multiply quark charges by 3 to make them integers
* MultiParticlePtCorrelations: Class to compute higher order mean transverse momentum correlations and the corresponding cumulants
* Jackknife: Implement class to compute uncertainty estimates with a delete-d Jackknife
* Sparkx: Add `autopep8` code formatter and require the correct format for merges to `main` and `sparkx_devel`

### Fixed
* QCumulantFlow and LeeYangZeroFlow: Fix bug in the calculation of the differential flow


[Link to diff from previous version](https://github.com/smash-transport/sparkx/compare/v1.2.1...v1.3.0)

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
