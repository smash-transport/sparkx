import numpy as np
from ParticleClass import Particle
from Lattice3D import Lattice3D

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
        real_eps=0
        imag_eps=0
        norm=0
        for particle in particle_data:
            x = particle.x
            y = particle.y
            rn = x**harmonic_n+y**harmonic_n
            t = np.arctan2(y,x)
            real_eps += rn*np.cos(harmonic_n*t)*particle.E
            imag_eps += rn*np.sin(harmonic_n*t)*particle.E
            norm += rn*particle.E
        return np.mean(real_eps)/np.mean(norm)+np.mean(imag_eps)/np.mean(norm)*1j
    
    def eccentricity_from_energy(self,harmonic_n,lattice):
        real_eps=0
        imag_eps=0
        norm=0
        for i, j, k in np.ndindex(lattice.grid_.shape):
            x, y, z = lattice.get_coordinates(i, j, k)
            rn = x**harmonic_n+y**harmonic_n
            t = np.arctan2(y,x)
            energy_density = lattice.get_value_by_index(i, j, k)
            real_eps += rn*np.cos(harmonic_n*t)*energy_density
            imag_eps += rn*np.sin(harmonic_n*t)*energy_density
            norm += rn*energy_density
        return np.mean(real_eps)/np.mean(norm)+np.mean(imag_eps)/np.mean(norm)*1j

#particle = Particle()
#particle.t_=1.0
#part1=Particle()
#part1.x_=0.0
#part1.y_=1.0
#part1.z_=0.0
#part1.E_=1
#part2=Particle()
#part2.x_=0.0
#part2.y_=1.0
#part2.z_=0.0
#part2.E_=1.0
#particle_data=[part1,part2]
#event=EventCharacteristics(particle_data)
#print(event.eccentricity_from_particles(2))
#latt=Lattice3D(-2, 2, -2, 2, -2, 2, 10, 10, 10)
#latt.add_particle_data(particle_data, 0.1, "energy density")
#print(event.eccentricity_from_energy(2,latt))
