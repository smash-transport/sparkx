import numpy as np
from sparkx.Particle import Particle

def charged_particles(particle_list):
    """
    Keep only charged particles in particle_list

    Parameters
    ----------
    particle_list:
        List with lists containing particles for the events.

    Returns
    -------
    list of lists
        Filtered list of lists containing particles for each event.
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [elem for elem in particle_list[i]
                                    if (elem.charge != 0 and elem.charge != np.nan)]
    return particle_list

def uncharged_particles(particle_list):
    """
    Keep only uncharged particles in particle_list

    Parameters
    ----------
    particle_list:
        List with lists containing particles for the events.

    Returns
    -------
    list of lists
        Filtered list of lists containing particles for each event.
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [elem for elem in particle_list[i]
                                    if (elem.charge == 0 and elem.charge != np.nan)]
    return particle_list

def strange_particles(particle_list):
    """
    Keep only strange particles in particle_list

    Parameters
    ----------
    particle_list:
        List with lists containing particles for the events.

    Returns
    -------
    list of lists
        Filtered list of lists containing particles for each event.
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [elem for elem in particle_list[i]
                                    if elem.is_strange()]
    return particle_list