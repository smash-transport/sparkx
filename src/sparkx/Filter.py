#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
import numpy as np
from sparkx.Particle import Particle
import warnings

def charged_particles(particle_list):
    """
    Keep only charged particles in particle_list.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [elem for elem in particle_list[i]
                                    if (elem.charge != 0 and 
                                        not np.isnan(elem.charge))]
    return particle_list

def uncharged_particles(particle_list):
    """
    Keep only uncharged particles in particle_list.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [elem for elem in particle_list[i]
                                    if (elem.charge == 0 and 
                                        not np.isnan(elem.charge))]
    return particle_list

def strange_particles(particle_list):
    """
    Keep only strange particles in particle_list.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [elem for elem in particle_list[i]
                                    if elem.is_strange() and 
                                    not np.isnan(elem.is_strange())]
    return particle_list

def particle_species(particle_list, pdg_list):
    """
    Keep only particle species given by their PDG ID in every event.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events
    
    pdg_list : int
        To keep a single particle species only, pass a single PDG ID

    pdg_list : tuple/list/array
        To keep multiple particle species, pass a tuple or list or array
        of PDG IDs

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(pdg_list, (str, int, list, np.integer, np.ndarray, tuple, float)):
        raise TypeError('Input value for pgd codes has not one of the ' +\
                        'following types: str, int, float, np.integer, list ' +\
                        'of str, list of int, list of int np.ndarray, tuple')

    if isinstance(pdg_list, float):
        if np.isnan(pdg_list):
            raise ValueError('Input value for PDG code is NaN')

    elif isinstance(pdg_list, (list, np.ndarray, tuple)):
        if np.isnan(pdg_list).any():
            raise ValueError('Input value for PDG codes contains NaN values')

    if isinstance(pdg_list, (int, float, str, np.integer)):
        pdg_list = int(pdg_list)
        
        for i in range(0, len(particle_list)):
            particle_list[i] = [elem for elem in particle_list[i]
                                        if (int(elem.pdg) == pdg_list 
                                            and not np.isnan(elem.pdg))]

    elif isinstance(pdg_list, (list, np.ndarray, tuple)):
        pdg_list = np.asarray(pdg_list, dtype=np.int64)

        for i in range(0, len(particle_list)):
            particle_list[i] = [elem for elem in particle_list[i]
                                        if (int(elem.pdg) in pdg_list 
                                            and not np.isnan(elem.pdg))]

    else:
        raise TypeError('Input value for pgd codes has not one of the ' +\
                        'following types: str, int, float, np.integer, list ' +\
                        'of str, list of int, list of float, np.ndarray, tuple')
    return particle_list

def remove_particle_species(particle_list, pdg_list):
    """
    Remove particle species from particle_list by their PDG ID in every
    event.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    pdg_list : int
        To remove a single particle species only, pass a single PDG ID

    pdg_list : tuple/list/array
        To remove multiple particle species, pass a tuple or list or array
        of PDG IDs

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(pdg_list, (str, int, float, list, np.integer, np.ndarray, tuple)):
        raise TypeError('Input value for pgd codes has not one of the ' +\
                        'following types: str, int, float, np.integer, list ' +\
                        'of str, list of int, list of float, np.ndarray, tuple')

    if isinstance(pdg_list, float):
        if np.isnan(pdg_list):
            raise ValueError('Input value for PDG code is NaN')

    elif isinstance(pdg_list, (list, np.ndarray, tuple)):
        if np.isnan(pdg_list).any():
            raise ValueError('Input value for PDG codes contains NaN values')

    if isinstance(pdg_list, (int, float, str, np.integer)):
        pdg_list = int(pdg_list)

        for i in range(0, len(particle_list)):
            particle_list[i] = [elem for elem in particle_list[i]
                                        if (int(elem.pdg) != pdg_list 
                                            and not np.isnan(elem.pdg))]

    elif isinstance(pdg_list, (list, np.ndarray, tuple)):
        pdg_list = np.asarray(pdg_list, dtype=np.int64)
        print(pdg_list)

        for i in range(0, len(particle_list)):
            particle_list[i] = [elem for elem in particle_list[i]
                                        if (int(elem.pdg) not in pdg_list 
                                            and not np.isnan(elem.pdg))]

    else:
        raise TypeError('Input value for pgd codes has not one of the ' +\
                        'following types: str, int, float, np.integer, list ' +\
                        'of str, list of int, list of float, np.ndarray, tuple')
    return particle_list

def participants(particle_list):
    """
    Keep only participants in particle_list.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [elem for elem in particle_list[i] 
                                    if (elem.ncoll != 0 and not np.isnan(elem.ncoll))]

    return particle_list


def spectators(particle_list):
    """
    Keep only spectators in particle_list.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [elem for elem in particle_list[i]
                                    if (elem.ncoll == 0 and not np.isnan(elem.ncoll))]

    return particle_list

def lower_event_energy_cut(particle_list,minimum_event_energy):
    """
    Filters out events with total energy lower than a threshold.
    Events with smaller energies are removed and not kept as empty events.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    minimum_event_energy : int or float
        The minimum event energy threshold. Should be a positive integer or float.

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event

    Raises
    ------
    TypeError
        If the minimum_event_energy parameter is not an integer or float.
    ValueError
        If the minimum_event_energy parameter is less than or equal to 0.
    ValueError
        If the minimum_event_energy parameter is NaN.
    """
    if not isinstance(minimum_event_energy, (int, float)):
        raise TypeError('Input value for lower event energy cut has not ' +\
                        'one of the following types: int, float')
    if np.isnan(minimum_event_energy):
        raise ValueError('Input value should not be NaN')
    if minimum_event_energy <= 0.:
        raise ValueError('The lower event energy cut value should be positive')

    updated_particle_list = []
    for event_particles in particle_list:
        total_energy = sum(particle.E for particle in event_particles if 
                           not np.isnan(particle.E))
        if total_energy >= minimum_event_energy:
            updated_particle_list.append(event_particles)
    particle_list = updated_particle_list

    if len(particle_list) == 0:
        particle_list = [[]]

    return particle_list

def spacetime_cut(particle_list, dim, cut_value_tuple):
    """
    Apply spacetime cut to all events by passing an acceptance range by
    ::code`cut_value_tuple`. All particles outside this range will
    be removed.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    dim : string
        String naming the dimension on which to apply the cut.
        Options: 't','x','y','z'

    cut_value_tuple : tuple
        Tuple with the upper and lower limits of the coordinate space
        acceptance range :code:`(cut_min, cut_max)`. If one of the limits 
        is not required, set it to :code:`None`, i.e.
        :code:`(None, cut_max)` or :code:`(cut_min, None)`.

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(cut_value_tuple, tuple) or len(cut_value_tuple) != 2:
        raise TypeError('Input value must be a tuple of length two')
    elif any(val is not None and not isinstance(val, (int, float)) for val in cut_value_tuple):
        raise ValueError('Non-numeric value found in cut_value_tuple')
    elif cut_value_tuple[0] is None and cut_value_tuple[1] is None:
        raise ValueError('At least one cut limit must be a number')
    elif dim == "t" and cut_value_tuple[0]<0:
        raise ValueError('Time boundary must be positive or zero.')
    if dim not in ("x","y","z","t"):
        raise ValueError('Only "t, x, y and z are possible dimensions.')

    if cut_value_tuple[0] is None:
        if(dim != "t"):
            lower_cut = float('-inf')
        else:
            lower_cut = 0.0
    else:
        lower_cut = cut_value_tuple[0]
    if cut_value_tuple[1] is None:
        upper_cut = float('inf')
    else:
        upper_cut = cut_value_tuple[1]

    if upper_cut < lower_cut:
        raise ValueError('The upper cut is smaller than the lower cut!')

    updated_particle_list = []
    for i in range(0, len(particle_list)):
        if (dim == "t"):
            particle_list_tmp = [elem for elem in particle_list[i] if
                                    (lower_cut <= elem.t <= upper_cut and not np.isnan(elem.t))]
        elif (dim == "x"):
            particle_list_tmp = [elem for elem in particle_list[i] if
                                    (lower_cut <= elem.x <= upper_cut and not np.isnan(elem.x))]
        elif (dim == "y"):
            particle_list_tmp = [elem for elem in particle_list[i] if
                                    (lower_cut <= elem.y <= upper_cut and not np.isnan(elem.y))]
        else:
            particle_list_tmp = [elem for elem in particle_list[i] if
                                    (lower_cut <= elem.z <= upper_cut and not np.isnan(elem.z))]
        updated_particle_list.append(particle_list_tmp)
    particle_list = updated_particle_list
    return particle_list

def pt_cut(particle_list, cut_value_tuple):
    """
    Apply p_t cut to all events by passing an acceptance range by
    ::code`cut_value_tuple`. All particles outside this range will
    be removed.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value_tuple : tuple
        Tuple with the upper and lower limits of the pT acceptance
        range :code:`(cut_min, cut_max)`. If one of the limits is not
        required, set it to :code:`None`, i.e. :code:`(None, cut_max)`
        or :code:`(cut_min, None)`.

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(cut_value_tuple, tuple) or len(cut_value_tuple) != 2:
        raise TypeError('Input value must be a tuple containing either '+\
                        'positive numbers or None of length two')
    elif any(val is not None and not isinstance(val, (int, float)) for val in cut_value_tuple):
        raise ValueError('Non-numeric value found in cut_value_tuple')
    elif (cut_value_tuple[0] is not None and cut_value_tuple[0]<0) or \
            (cut_value_tuple[1] is not None and cut_value_tuple[1]<0):
                raise ValueError('The cut limits must be positive or None')
    elif cut_value_tuple[0] is None and cut_value_tuple[1] is None:
        raise ValueError('At least one cut limit must be a number')

    if cut_value_tuple[0] is None:
        lower_cut = 0.0
    else:
        lower_cut = cut_value_tuple[0]
    if cut_value_tuple[1] is None:
        upper_cut = float('inf')
    else:
        upper_cut = cut_value_tuple[1]

    if upper_cut < lower_cut:
        raise ValueError('The upper cut is smaller than the lower cut!')

    updated_particle_list = []
    for i in range(0, len(particle_list)):
        particle_list_tmp = [elem for elem in particle_list[i] if
                                    (lower_cut <= elem.pt_abs() <= upper_cut 
                                     and not np.isnan(elem.pt_abs()))]
        updated_particle_list.append(particle_list_tmp)
    particle_list = updated_particle_list
    return particle_list

def rapidity_cut(particle_list, cut_value):
    """
    Apply rapidity cut to all events and remove all particles with rapidity
    not complying with cut_value.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value : float
        If a single value is passed, the cut is applied symmetrically
        around 0.
        For example, if cut_value = 1, only particles with rapidity in
        [-1.0, 1.0] are kept.

    cut_value : tuple
        To specify an asymmetric acceptance range for the rapidity
        of particles, pass a tuple (cut_min, cut_max)

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if isinstance(cut_value, tuple):
        if len(cut_value) != 2:
            raise TypeError('If input value is a tuple, then it must contain '+\
                            'two numbers')
        elif any(not isinstance(val, (int, float)) for val in cut_value):
            raise ValueError('Non-numeric value found in cut_value')

        if cut_value[0] > cut_value[1]:
            warn_msg = 'Lower limit {} is greater that upper limit {}. Switched order is assumed in the following.'.format(cut_value[0], cut_value[1])
            warnings.warn(warn_msg)

    elif not isinstance(cut_value, (int, float)):
        raise TypeError('Input value must be a number or a tuple ' +\
                        'with the cut limits (cut_min, cut_max)')

    if isinstance(cut_value, (int, float)):
        # cut symmetrically around 0
        limit = np.abs(cut_value)

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [elem for elem in particle_list[i] if
                                        (-limit<=elem.momentum_rapidity_Y()<=limit 
                                        and not np.isnan(elem.momentum_rapidity_Y()))]
            updated_particle_list.append(particle_list_tmp)
    elif isinstance(cut_value, tuple):
        lim_max = max(cut_value[0], cut_value[1])
        lim_min = min(cut_value[0], cut_value[1])

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [elem for elem in particle_list[i] if
                                        (lim_min<=elem.momentum_rapidity_Y()<=lim_max 
                                        and not np.isnan(elem.momentum_rapidity_Y()))]
            updated_particle_list.append(particle_list_tmp)

    particle_list = updated_particle_list
    return particle_list

def pseudorapidity_cut(particle_list, cut_value):
    """
    Apply pseudo-rapidity cut to all events and remove all particles with
    pseudo-rapidity not complying with cut_value.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value : float
        If a single value is passed, the cut is applied symmetrically
        around 0.
        For example, if cut_value = 1, only particles with pseudo-rapidity
        in [-1.0, 1.0] are kept.

    cut_value : tuple
        To specify an asymmetric acceptance range for the pseudo-rapidity
        of particles, pass a tuple (cut_min, cut_max)

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if isinstance(cut_value, tuple):
        if len(cut_value) != 2:
            raise TypeError('If input value is a tuple, then it must contain '+\
                            'two numbers')
        elif any(not isinstance(val, (int, float)) for val in cut_value):
            raise ValueError('Non-numeric value found in cut_value')

        if cut_value[0] > cut_value[1]:
            warn_msg = 'Lower limit {} is greater that upper limit {}. Switched order is assumed in the following.'.format(cut_value[0], cut_value[1])
            warnings.warn(warn_msg)

    elif not isinstance(cut_value, (int, float)):
        raise TypeError('Input value must be a number or a tuple ' +\
                        'with the cut limits (cut_min, cut_max)')

    if isinstance(cut_value, (int, float)):
        # cut symmetrically around 0
        limit = np.abs(cut_value)

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [elem for elem in particle_list[i] if
                                        (-limit<=elem.pseudorapidity()<=limit
                                        and not np.isnan(elem.pseudorapidity()))]
            updated_particle_list.append(particle_list_tmp)
    elif isinstance(cut_value, tuple):
        lim_max = max(cut_value[0], cut_value[1])
        lim_min = min(cut_value[0], cut_value[1])

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [elem for elem in particle_list[i] if
                                        (lim_min<=elem.pseudorapidity()<=lim_max
                                        and not np.isnan(elem.pseudorapidity()))]
            updated_particle_list.append(particle_list_tmp)

    particle_list = updated_particle_list
    return particle_list

def spatial_rapidity_cut(particle_list, cut_value):
    """
    Apply spatial rapidity (space-time rapidity) cut to all events and
    remove all particles with spatial rapidity not complying with cut_value.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value : float
        If a single value is passed, the cut is applied symmetrically
        around 0.
        For example, if cut_value = 1, only particles with spatial rapidity
        in [-1.0, 1.0] are kept.

    cut_value : tuple
        To specify an asymmetric acceptance range for the spatial rapidity
        of particles, pass a tuple (cut_min, cut_max)

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if isinstance(cut_value, tuple):
        if len(cut_value) != 2:
            raise TypeError('If input value is a tuple, then it must contain '+\
                            'two numbers')
        elif any(not isinstance(val, (int, float)) for val in cut_value):
            raise ValueError('Non-numeric value found in cut_value')

        if cut_value[0] > cut_value[1]:
            warn_msg = 'Lower limit {} is greater that upper limit {}. Switched order is assumed in the following.'.format(cut_value[0], cut_value[1])
            warnings.warn(warn_msg)

    elif not isinstance(cut_value, (int, float)):
        raise TypeError('Input value must be a number or a tuple ' +\
                        'with the cut limits (cut_min, cut_max)')

    if isinstance(cut_value, (int, float)):
        # cut symmetrically around 0
        limit = np.abs(cut_value)

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [elem for elem in particle_list[i] if
                                        (-limit<=elem.spatial_rapidity()<=limit
                                        and not np.isnan(elem.spatial_rapidity()))]
            updated_particle_list.append(particle_list_tmp)
    elif isinstance(cut_value, tuple):
        lim_max = max(cut_value[0], cut_value[1])
        lim_min = min(cut_value[0], cut_value[1])

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [elem for elem in particle_list[i] if
                                        (lim_min<=elem.spatial_rapidity()<=lim_max
                                        and not np.isnan(elem.spatial_rapidity()))]
            updated_particle_list.append(particle_list_tmp)

    particle_list = updated_particle_list
    return particle_list

def multiplicity_cut(particle_list, min_multiplicity):
    """
    Apply multiplicity cut. Remove all events with a multiplicity lower
    than min_multiplicity.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    min_multiplicity : int
        Lower bound for multiplicity. If the multiplicity of an event is
        lower than min_multiplicity, this event is discarded.

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(min_multiplicity, int):
        raise TypeError('Input value for multiplicity cut must be an int')
    if min_multiplicity < 0:
        raise ValueError('Minimum multiplicity must >= 0')

    idx_keep_event = []
    for idx, event_particles in enumerate(particle_list):
        multiplicity = len(event_particles)
        if multiplicity >= min_multiplicity:
            idx_keep_event.append(idx)

    particle_list = [particle_list[idx] for idx in idx_keep_event]

    return particle_list

def particle_status(particle_list, status_list):
    """
    Keep only particles with a given particle status.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    status_list : int
        To keep a particles with a single status only, pass a single status

    status_list : tuple/list/array
        To keep hadrons with different hadron status, pass a tuple or list
        or array

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(status_list, (int, list, tuple, np.ndarray)):
        raise TypeError('Input for status codes must be int, or  '+\
                        'list/tuple/array of int values')
    if isinstance(status_list, (list, tuple, np.ndarray)):
        if any(not isinstance(val, int) for val in status_list):
            raise TypeError('status_list contains non int value')

    if isinstance(status_list, int):
        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [elem for elem in particle_list[i]
                                        if (elem.status == status_list 
                                            and not np.isnan(elem.status))]
            updated_particle_list.append(particle_list_tmp)

    elif isinstance(status_list, (list, np.ndarray, tuple)):
        status_list = np.asarray(status_list, dtype=np.int64)

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [elem for elem in particle_list[i]
                                        if (elem.status in status_list 
                                            and not np.isnan(elem.status))]
        updated_particle_list.append(particle_list_tmp)

    particle_list = updated_particle_list

    return particle_list
