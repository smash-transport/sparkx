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
from sparkx.Lattice3D import Lattice3D
import warnings

class EventCharacteristics:
    """
    This class computes event-by-event characteristics, e.g., eccentricities
    or certain densities.

    Parameters
    ----------
    event_data: list, numpy.ndarray or Lattice3D
        List or array containing particle objects for one event, or a lattice 
        containing the relevant densities.

    Attributes
    ----------
    event_data_: list, numpy.ndarray or Lattice3D
        List or array containing particle objects for one event, or a lattice 
        containing the relevant densities.
    has_lattice_: bool
        Contains information if characteristics are derived from a lattice or 
        particles

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
    generate_eBQS_densities_Milne_from_OSCAR_IC:
        Generates energy, baryon, charge, and strangeness densities in Milne
        coordinates.
    generate_eBQS_densities_Minkowski_from_OSCAR_IC:    
        Generates energy, baryon, charge, and strangeness densities in Minkowski
        coordinates.

    Examples
    --------

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Oscar import Oscar
        >>> from sparkx.EventCharacteristics import EventCharacteristics
        >>>
        >>> OSCAR_FILE_PATH = [Oscar_directory]/particle_lists.oscar
        >>>
        >>> # Oscar object containing all events
        >>> oscar = Oscar(OSCAR_FILE_PATH).particle_objects_list()
        >>>
        >>> # compute epsilon2 for the first event
        >>> event_characterization = EventCharacteristics(oscar[0])
        >>> eps2 = event_characterization.eccentricity(2, weight_quantity = "number")

    """
    def __init__(self, event_data):
        self.set_event_data(event_data)

    def set_event_data(self, event_data):
        """
        Overwrites the event data.

        Parameters
        ----------
        event_data : list, numpy.ndarray, or Lattice3D
            List or array containing particle objects for one event, or a
            lattice containing the relevant densities.

        Raises
        ------
        TypeError
            If the input is not a list or numpy.ndarray when deriving
            characteristics from particles.
            If at least one element in the input is not of type Particle.
        """
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

    def eccentricity_from_particles(self,harmonic_n, harmonic_m = None, weight_quantity = "energy"):
        """
        Computes the spatial eccentricity from particles.

        Parameters
        ----------
        harmonic_n : int
            The harmonic order for the eccentricity calculation.
        harmonic_m : int, optional
            The power of the radial weight.
        weight_quantity : str, optional
            The quantity used for particle weighting.
            Valid options are "energy", "number", "charge", "baryon", "strangeness".
            Default is "energy".

        Returns
        -------
        complex
            The complex-valued eccentricity.

        Raises
        ------
        ValueError
            If the harmonic order is less than 1.
            If the weight quantity is unknown.
        """
        real_eps = 0.
        imag_eps = 0.
        norm = 0.
        if harmonic_n < 1:
            raise ValueError("Eccentricity is only defined for positive expansion orders.")
        if harmonic_m != None and harmonic_m < 1:
            raise ValueError("harmonic_m must be positive")
        for particle in self.event_data_:
            if weight_quantity == "energy":
                weight = particle.E
            elif weight_quantity == "number":
                weight = 1.
            elif weight_quantity == "charge":
                weight = particle.charge
            elif weight_quantity == "baryon":
                weight = particle.baryon_number
            elif weight_quantity == "strangeness":
                weight = particle.strangeness
            else:
                raise ValueError("Unknown weight for eccentricity")
            x = particle.x
            y = particle.y
            #Exception for dipole asymmetry
            if harmonic_n == 1 and harmonic_m == None:
                rn = (x**2 + y**2)**(3/2.)
            elif harmonic_n != 1 and harmonic_m == None:
                rn = (x**2 + y**2)**(harmonic_n/2.)
            else:
                rn = (x**2 + y**2)**(harmonic_m/2.)

            phi = np.arctan2(y,x)
            real_eps += rn*np.cos(harmonic_n*phi)*weight
            imag_eps += rn*np.sin(harmonic_n*phi)*weight
            norm += rn*weight

        return -(real_eps/norm + (imag_eps/norm)*1j)

    def eccentricity_from_lattice(self,harmonic_n,harmonic_m = None):
        """
        Computes the spatial eccentricity from a 3D lattice. Takes all z-values
        into account.

        Parameters
        ----------
        harmonic_n : int
            The harmonic order for the eccentricity calculation.
        harmonic_m : int, optional
            The power of the radial weight.

        Returns
        -------
        complex
            The complex-valued eccentricity.

        Raises
        ------
        ValueError
            If the harmonic order is less than 1.
        """
        real_eps = 0.
        imag_eps = 0.
        norm = 0.
        if harmonic_n < 1:
            raise ValueError("Eccentricity is only defined for positive expansion orders.")
        if harmonic_m != None and harmonic_m < 1:
            raise ValueError("harmonic_m must be positive")
        for i, j, k in np.ndindex(self.event_data_.grid_.shape):
            x, y, z = self.event_data_.get_coordinates(i, j, k)
            #Exception for dipole asymmetry
            if harmonic_n == 1 and harmonic_m == None:
                rn = (x**2 + y**2)**(3/2.)
            elif harmonic_n != 1 and harmonic_m == None:
                rn = (x**2 + y**2)**(harmonic_n/2.)
            else:
                rn = (x**2 + y**2)**(harmonic_m/2.)

            phi = np.arctan2(y,x)
            lattice_density = self.event_data_.get_value_by_index(i, j, k)
            real_eps += rn*np.cos(harmonic_n*phi)*lattice_density
            imag_eps += rn*np.sin(harmonic_n*phi)*lattice_density
            norm += rn*lattice_density

        return -(real_eps/norm + (imag_eps/norm)*1j)

    def eccentricity(self,harmonic_n,harmonic_m = None,weight_quantity = "energy"):
        """
        Computes the spatial eccentricity.

        .. math::

            \\varepsilon_{m,n}e^{\\mathrm{i}n\\Phi_{m,n}} = -\\frac{\\lbrace{r^{m}e^{\\mathrm{i}n\phi}\\rbrace}}{\\lbrace{r^{m}\\rbrace}}

        For `harmonic_n=1`, :math:`n=3` is used. If `harmonic_m` is provided,
        then the given value is used as radial exponent. 

        Parameters
        ----------
        harmonic_n : int
            The harmonic order for the eccentricity calculation.
        harmonic_m : int, optional
            The power of the radial weight.
        weight_quantity : str, optional
            The quantity used for particle weighting.
            Valid options are "energy", "number", "charge", "baryon", "strangeness".
            Default is "energy".

        Returns
        -------
        complex
            The complex-valued eccentricity.

        Raises
        ------
        ValueError
            If the harmonic order is less than 1.
        """
        if self.has_lattice_:
            return self.eccentricity_from_lattice(harmonic_n=harmonic_n,harmonic_m=harmonic_m)
        else:
            return self.eccentricity_from_particles(harmonic_n=harmonic_n, harmonic_m=harmonic_m, weight_quantity=weight_quantity)

    def generate_eBQS_densities_Milne_from_OSCAR_IC(self,x_min,x_max,y_min,y_max,z_min,z_max,Nx,Ny,Nz,n_sigma_x,n_sigma_y,n_sigma_z,sigma_smear,eta_range,output_filename,IC_info=None):
        """
        Generates energy, baryon, charge, and strangeness densities in Milne 
        coordinates from OSCAR initial conditions.

        The total energy in GeV can be obtained by integrating the energy 
        density with :math:`\\tau \\mathrm{d}x\\mathrm{d}y\\mathrm{d}\\eta`.

        Parameters
        ----------
        x_min, x_max, y_min, y_max, z_min, z_max : float
            Minimum and maximum coordinates in the x, y, and z directions.

        Nx, Ny, Nz : int
            Number of grid points in the x, y, and z directions.

        n_sigma_x, n_sigma_y, n_sigma_z : float
            Width of the smearing in the x, y, and z directions in units of 
            sigma_smear.

        sigma_smear : float
            Smearing parameter for particle data.

        eta_range : list, tuple
            A list containing the minimum and maximum values of spacetime 
            rapidity (eta) and the number of grid points.

        output_filename : str
            The name of the output file where the densities will be saved.

        IC_info : str
            A string containing info about the initial condition, e.g., 
            collision energy or centrality.

        Raises
        ------
        TypeError
            If the given IC_info is not a string and if the class is initialized
            with a lattice.

        Returns
        -------
        None
        """
        if not all(isinstance(val, (float, int)) for val in [x_min, x_max, y_min, y_max, z_min, z_max, sigma_smear]):
            raise TypeError("Coordinates and sigma_smear must be float or int")
        if not all((isinstance(val, int) and val > 0) for val in [Nx, Ny, Nz]):
            raise TypeError("Nx, Ny, Nz must be positive integers")
        if not all((isinstance(val, (float,int)) and val > 0) for val in [n_sigma_x, n_sigma_y, n_sigma_z]):
            raise TypeError("n_sigma_x, n_sigma_y, n_sigma_z must be positive float or int")
        if not isinstance(eta_range, (list, tuple)):
            raise TypeError("eta_range must be a list or tuple")
        if len(eta_range) != 3:
            raise ValueError("eta_range must contain min, max, and number of grid points")
        if not all(isinstance(val, (float, int)) for val in eta_range):
            raise TypeError("Values in eta_range must be float or int")
        if not isinstance(output_filename, str):
            raise TypeError("output_filename must be a string")
        if (IC_info is not None) and not isinstance(IC_info,str):
            raise TypeError("The given IC_info is not a string")

        if self.has_lattice_:
            raise TypeError("The smearing function only works with EventCharacteristics derived from particles.")

        energy_density = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, Nx, Ny, Nz, n_sigma_x, n_sigma_y, n_sigma_z)
        baryon_density = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, Nx, Ny, Nz, n_sigma_x, n_sigma_y, n_sigma_z)
        charge_density = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, Nx, Ny, Nz, n_sigma_x, n_sigma_y, n_sigma_z)
        strangeness_density = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, Nx, Ny, Nz, n_sigma_x, n_sigma_y, n_sigma_z)
        
        # smear the particles on the 3D lattice
        energy_density.add_particle_data(self.event_data_, sigma_smear, "energy_density")
        baryon_density.add_particle_data(self.event_data_, sigma_smear, "baryon_density")
        charge_density.add_particle_data(self.event_data_, sigma_smear, "charge_density")
        strangeness_density.add_particle_data(self.event_data_, sigma_smear, "strangeness_density")
        # get the proper time of one of the particles from the iso-tau surface
        tau = self.event_data_[0].proper_time()
        if np.isnan(tau):
            raise ValueError("The proper time is not defined for the given particles.")
        # take the x and y coordinates from the lattice and use the set eta range
        x = energy_density.x_values_
        y = energy_density.y_values_
        eta = np.linspace(eta_range[0], eta_range[1], eta_range[2])
        if(1.05*tau*(np.sinh(eta[int(eta_range[2]/2.)])-np.sinh(eta[int(eta_range[2]/2.-1)])) < (z_max-z_min)/Nz):
            warnings.warn("Warning: The grid for z is not fine enough for the requested eta-grid.")
        
        # generate the header for the output file
        file_header = "# smeared density from SPARKX in Milne coordinates\n# "
        if IC_info is not None:
            file_header += IC_info
        file_header += "\n# grid info: n_x n_y n_eta x_min x_max y_min y_max eta_min eta_max\n# "
        file_header += "%d %d %d %g %g %g %g %g %g\n"%(Nx,Ny,eta_range[2],x_min,x_max,y_min,y_max,eta_range[0],eta_range[1])
        file_header += "# tau [fm], x [fm], y [fm], eta, energy_density [GeV/fm^3], baryon_density [1/fm^3], charge density [1/fm^3], strangeness_density [1/fm^3]\n"

        # print the 3D lattice in Milne coordinates to a file
        # Open the output file for writing
        with open(output_filename, 'w') as output_file:
            output_file.write(file_header)
            for x_val in x:
                for y_val in y:
                    for eta_val in eta:
                        z_val = tau * np.sinh(eta_val)
                        milne_conversion = tau*np.cosh(eta_val) # dz = tau * cosh(eta) * deta
                        value_energy_density = energy_density.interpolate_value(x_val,y_val,z_val) * milne_conversion
                        value_baryon_density = baryon_density.interpolate_value(x_val,y_val,z_val) * milne_conversion
                        value_charge_density = charge_density.interpolate_value(x_val,y_val,z_val) * milne_conversion
                        value_strangeness_density = strangeness_density.interpolate_value(x_val,y_val,z_val) * milne_conversion

                        if value_energy_density == None:
                            value_energy_density = 0.
                            value_baryon_density = 0.
                            value_charge_density = 0.
                            value_strangeness_density = 0.

                        output_file.write(f"{tau:g} {x_val:g} {y_val:g} {eta_val:g} {value_energy_density:g} {value_baryon_density:g} {value_charge_density:g} {value_strangeness_density:g}\n")

    def generate_eBQS_densities_Minkowski_from_OSCAR_IC(self,x_min,x_max,y_min,y_max,z_min,z_max,Nx,Ny,Nz,n_sigma_x,n_sigma_y,n_sigma_z,sigma_smear,output_filename,IC_info=None):
        """
        Generates energy, baryon, charge, and strangeness densities in 
        Minkowski coordinates from OSCAR initial conditions.

        The total energy in GeV can be obtained by integrating the energy 
        density with :math:`\\mathrm{d}x\\mathrm{d}y\\mathrm{d}z`.

        Parameters
        ----------
        x_min, x_max, y_min, y_max, z_min, z_max : float
            Minimum and maximum coordinates in the x, y, and z directions.

        Nx, Ny, Nz : int
            Number of grid points in the x, y, and z directions.

        n_sigma_x, n_sigma_y, n_sigma_z : float
            Width of the smearing in the x, y, and z directions in units of 
            sigma_smear.

        sigma_smear : float
            Smearing parameter for particle data.

        output_filename : str
            The name of the output file where the densities will be saved.

        IC_info : str
            A string containing info about the initial condition, e.g., 
            collision energy or centrality.

        Raises
        ------
        TypeError
            If the given IC_info is not a string and if the class is initialized
            with a lattice.

        Returns
        -------
        None
        """
        if not all(isinstance(val, (float, int)) for val in [x_min, x_max, y_min, y_max, z_min, z_max, sigma_smear]):
            raise TypeError("Coordinates and sigma_smear must be float or int")
        if not all((isinstance(val, int) and val > 0) for val in [Nx, Ny, Nz]):
            raise TypeError("Nx, Ny, Nz must be positive integers")
        if not all((isinstance(val, (float,int)) and val > 0) for val in [n_sigma_x, n_sigma_y, n_sigma_z]):
            raise TypeError("n_sigma_x, n_sigma_y, n_sigma_z must be positive float or int")
        if (IC_info is not None) and not isinstance(IC_info,str):
            warnings.warn("The given IC_info is not a string")
        if not isinstance(output_filename, str):
            raise TypeError("output_filename must be a string")

        if self.has_lattice_:
            raise TypeError("The smearing function only works with EventCharacteristics derived from particles.")

        energy_density = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, Nx, Ny, Nz, n_sigma_x, n_sigma_y, n_sigma_z)
        baryon_density = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, Nx, Ny, Nz, n_sigma_x, n_sigma_y, n_sigma_z)
        charge_density = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, Nx, Ny, Nz, n_sigma_x, n_sigma_y, n_sigma_z)
        strangeness_density = Lattice3D(x_min, x_max, y_min, y_max, z_min, z_max, Nx, Ny, Nz, n_sigma_x, n_sigma_y, n_sigma_z)

        # smear the particles on the 3D lattice
        energy_density.add_particle_data(self.event_data_, sigma_smear,"energy_density")
        baryon_density.add_particle_data(self.event_data_, sigma_smear,"baryon_density")
        charge_density.add_particle_data(self.event_data_, sigma_smear,"charge_density")
        strangeness_density.add_particle_data(self.event_data_, sigma_smear,"strangeness_density")

        # get the proper time of one of the particles from the iso-tau surface
        tau = self.event_data_[0].proper_time()
        # take the x, y and z coordinates from the lattice
        x = energy_density.x_values_
        y = energy_density.y_values_
        z = energy_density.z_values_

        # generate the header for the output file
        file_header = "# smeared density from SPARKX in Milne coordinates\n# "
        if IC_info is not None:
            file_header += IC_info
        file_header += "\n# grid info: n_x n_y n_eta x_min x_max y_min y_max eta_min eta_max\n# "
        file_header += "%d %d %d %g %g %g %g %g %g\n"%(Nx,Ny,Nz,x_min,x_max,y_min,y_max,z_min,z_max)
        file_header += "# tau [fm], x [fm], y [fm], z [fm], energy_density [GeV/fm^3], baryon_density [1/fm^3], charge density [1/fm^3], strangeness_density [1/fm^3]\n"


        # print the 3D lattice in Minkowski coordinates to a file
        # Open the output file for writing
        with open(output_filename, 'w') as output_file:
            output_file.write(file_header)
            for x_val in x:
                for y_val in y:
                    for z_val in z:
                        value_energy_density = energy_density.interpolate_value(x_val,y_val,z_val)
                        value_baryon_density = baryon_density.interpolate_value(x_val,y_val,z_val)
                        value_charge_density = charge_density.interpolate_value(x_val,y_val,z_val)
                        value_strangeness_density = strangeness_density.interpolate_value(x_val,y_val,z_val)

                        if value_energy_density == None:
                            value_energy_density = 0.
                            value_baryon_density = 0.
                            value_charge_density = 0.
                            value_strangeness_density = 0.

                        output_file.write(f"{tau:g} {x_val:g} {y_val:g} {z_val:g} {value_energy_density:g} {value_baryon_density:g} {value_charge_density:g} {value_strangeness_density:g}\n")    