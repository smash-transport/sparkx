import numpy as np
from Particle import Particle
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

    
    def eccentricity_from_particles(self,harmonic_n, weight_quantity = "energy"):
        real_eps=0
        imag_eps=0
        norm=0
        if harmonic_n < 1:
            raise ValueError("Eccentricity is only defined for positive expansion orders.")
        for particle in self.particle_data:
            if weight_quantity == "energy":
                weight = particle.E
            elif weight_quantity == "number":
                weight = 1
            elif weight_quantity == "charge":
                weight = particle.charge
            elif weight_quantity == "baryon number":
                weight = particle.baryon_number
            else:
                raise ValueError("Unknown weight for eccentricity")
            x = particle.x
            y = particle.y
            #Exception for dipole asymmetry
            if harmonic_n == 1:
                rn = (x**2+y**2)**(3/2.0)
            else:
                rn = (x**2+y**2)**(harmonic_n/2.0)
            t = np.arctan2(y,x)
            real_eps += rn*np.cos(harmonic_n*t)*weight
            imag_eps += rn*np.sin(harmonic_n*t)*weight
            norm += rn*weight
        return real_eps/norm + (imag_eps/norm)*1j
    
    def eccentricity_from_lattice(self,harmonic_n,lattice):
        real_eps=0
        imag_eps=0
        norm=0
        if harmonic_n < 1:
            raise ValueError("Eccentricity is only defined for positive expansion orders.")
        for i, j, k in np.ndindex(lattice.grid_.shape):
            x, y, z = lattice.get_coordinates(i, j, k)
            #Exception for dipole asymmetry
            if harmonic_n == 1:
                rn = (x**2+y**2)**(3/2.0)
            else:
                rn = (x**2+y**2)**(harmonic_n/2.0)
            t = np.arctan2(y,x)
            lattice_density = lattice.get_value_by_index(i, j, k)
            real_eps += rn*np.cos(harmonic_n*t)*lattice_density
            imag_eps += rn*np.sin(harmonic_n*t)*lattice_density
            norm += rn*lattice_density
        return real_eps/norm + (imag_eps/norm)*1j

#particle = Particle()
#particle.t_=1.0
#part1=Particle()
#part1.x_=1.0
#part1.y_=0.0
#part1.z_=0.0
#part1.E_=1
#part2=Particle()
#part2.x_=0.0
#part2.y_=1.0
#part2.z_=0.0
#part2.E_=2.0
#particle_data=[part1,part2]
#event=EventCharacteristics(particle_data)
#print(event.eccentricity_from_particles(2))
#latt=Lattice3D(-2, 2, -2, 2, -2, 2, 50, 50, 50)
#latt.add_particle_data(particle_data, 0.1, "energy density")
#print(event.eccentricity_from_lattice(2,latt))
#print(event.eccentricity_from_particles(2,"number"))
#latt.add_particle_data(particle_data, 0.1, "number")
#print(event.eccentricity_from_lattice(2,latt))
