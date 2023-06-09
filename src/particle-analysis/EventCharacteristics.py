import numpy as np
from ParticleClass import Particle

class EventCharacteristics:
    """
    This class computes event-by-event characteristics, e.g., eccentricities
    or certain densities.

    Parameters
    ----------
    particle_data: list, numpy.ndarray
        List or array containing particle objects for one event.

    Attributes
    ----------
    particle_data_: list, numpy.ndarray
        List or array containing particle objects for one event.

    Methods
    -------
    eccentricity_from_particles:
        Computes the spatial eccentricity from particles.
    """
    def __init__(self,particle_data):
        # check that the input is a list/numpy.ndarray containing Particle objects
        if not isinstance(particle_data, (list, np.ndarray)):
            raise TypeError('The input is not a list nor a numpy.ndarray.')
        for particle in particle_data:
            if not isinstance(particle, Particle):
                raise TypeError('At least one element in the input is not a ' +\
                                'Particle type.')
        self.particle_data_ = particle_data

    
    def eccentricity_from_particles(self,harmonic_n):
        eps = 0. + 0.j
        # @Niklas: implement your computation here (maybe the complex eps?)
        return eps

        