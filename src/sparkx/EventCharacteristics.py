import numpy as np
from Particle import Particle
from Lattice3D import Lattice3D

class EventCharacteristics:
    """
    This class computes event-by-event characteristics, e.g., eccentricities
    or certain densities.

    Parameters
    ----------
    event_data: list, numpy.ndarray or Lattice3D
        List or array containing particle objects for one event, or a lattice containing the relevant densities.

    Attributes
    ----------
    event_data_: list, numpy.ndarray or Lattice3D
        List or array containing particle objects for one event, or a lattice containing the relevant densities.
    has_lattice_: bool
        Contains information if characteristics are derived from a lattice or particles

    Methods
    -------
    set_event_data:
        Overwrites the event data.
    eccentricity:
        Computes the spatial eccentricity.
    eccentricity_from_particles:
        Computes the spatial eccentricity from particles.
    eccentricity_from_lattice:
        Computes the spatial eccentricity from a 3D lattice.
    """
    def __init__(self, event_data):
        self.set_event_data(event_data)

    def set_event_data (self, event_data):
        # check if the input is a Lattice3D object
        if isinstance(event_data, Lattice3D):
            self.event_data_ = event_data
            self.has_lattice_ = True
        else:
            # check that the input is a list/numpy.ndarray containing Particle objects
            if not isinstance(event_data, (list, np.ndarray)):
                raise TypeError('The input is not a list nor a numpy.ndarray.')
            for particle in event_data:
                if not isinstance(particle, Particle):
                    raise TypeError('At least one element in the input is not a ' +
                                    'Particle type.')
            self.event_data_ = event_data
            self.has_lattice_ = False
            
    def eccentricity_from_particles(self,harmonic_n, weight_quantity = "energy"):
        real_eps=0
        imag_eps=0
        norm=0
        if harmonic_n < 1:
            raise ValueError("Eccentricity is only defined for positive expansion orders.")
        for particle in self.event_data_:
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
    
    def eccentricity_from_lattice(self,harmonic_n):
        real_eps=0
        imag_eps=0
        norm=0
        if harmonic_n < 1:
            raise ValueError("Eccentricity is only defined for positive expansion orders.")
        for i, j, k in np.ndindex(self.event_data_.grid_.shape):
            x, y, z = self.event_data_.get_coordinates(i, j, k)
            #Exception for dipole asymmetry
            if harmonic_n == 1:
                rn = (x**2+y**2)**(3/2.0)
            else:
                rn = (x**2+y**2)**(harmonic_n/2.0)
            t = np.arctan2(y,x)
            lattice_density = self.event_data_.get_value_by_index(i, j, k)
            real_eps += rn*np.cos(harmonic_n*t)*lattice_density
            imag_eps += rn*np.sin(harmonic_n*t)*lattice_density
            norm += rn*lattice_density
        return real_eps/norm + (imag_eps/norm)*1j
    
    def eccentricity(self,harmonic_n,weight_quantity = "energy"):
        if self.has_lattice_:
            return self.eccentricity_from_lattice(harmonic_n)
        else:
            return self.eccentricity_from_particles(harmonic_n, weight_quantity)

""" 
particle = Particle()
particle.t_=1.0
part1=Particle()
part1.x_=1.0
part1.y_=0.0
part1.z_=0.0
part1.E_=1
part2=Particle()
part2.x_=0.0
part2.y_=1.0
part2.z_=0.0
part2.E_=2.0
particle_data=[part1,part2]
event=EventCharacteristics(particle_data)
print("Particle eps2")
print(event.eccentricity(2))
latt=Lattice3D(-2, 2, -2, 2, -2, 2, 50, 50, 50)
latt2=Lattice3D(-2, 2, -2, 2, -2, 2, 50, 50, 50,3,3,3)
latt3=Lattice3D(-2, 2, -2, 2, -2, 2, 50, 50, 50,1,1,1)

latt.add_particle_data(particle_data, 0.2, "energy density")
latt2.add_particle_data(particle_data, 0.2, "energy density")
latt3.add_particle_data(particle_data, 0.2, "energy density")
event2=EventCharacteristics(latt)
event3=EventCharacteristics(latt2)
event4=EventCharacteristics(latt3)
print("Lattice eps, no cutoff")
print(event2.eccentricity(2))


#event2.set_event_data(latt2)
print("Lattice eps, big cutoff")
print(event3.eccentricity(2, "energy"))

#event3.set_event_data(latt3)
print("Lattice eps, small cutoff")
print(event4.eccentricity(2, "energy")) 
"""
