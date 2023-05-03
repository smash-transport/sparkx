# particle-analysis

This repository contains multiple python files with useful classes to analyze the particle output from [SMASH](https://smash-transport.github.io/) written in OSCAR2013/OSCAR2013Extended format or from the hadron output of [JETSCAPE/X-SCAPE](https://jetscape.org/).

## Coding Conventions
All modules in this package follow distinct formatting structure which is summarized in the following:
### Nomenclature
- **Class members:** Names of class members end with an underscore, e.g. `.mass_`  
- **Setter methods:** Names of methods that assign values to members start with 'set', e.g. `.set_mass()`
- **Getter methods:** Names of methods to get members have the same name as the member **without** an underscore at the end, e.g. `.mass()`

### Returning Values
- If not otherwise explicitely stated, all methods that return multiple values return them as **numpy array**

# Classes & Usage

## Oscar Class
