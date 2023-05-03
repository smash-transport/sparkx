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


# Oscar Class
## Members
- __particle_list__:         contains a nested list with all output quantities per line for all events
- __oscar_type__:            ...
- __num_output_per_event__:  ...
- __num_event__:             ...

## Methods

- __particle_list()__: returnes nested list containing all output quantities per line for all events
- __particle_objects_list()__: ...

## Usage

### __particle_list()__
particle_list() returns a nested python list containing all quantities from the Oscar2013/Oscar2013Extended output as numerical values with the following shape:
 
&nbsp;&nbsp;&nbsp;&nbsp; *Single Event:    [output_line][particle_quantity]*\
&nbsp;&nbsp;&nbsp;&nbsp; *Multiple Events: [event][output_line][particle_quantity]*

```
import Oscar

PATH_OSCAR_FILE = [oscar_path]
data = Oscar(PATH_OSCAR_FILE)

data_as_nested_list = data.particle_list()
```

