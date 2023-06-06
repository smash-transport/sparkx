import numpy as np
import random as rd

class GenerateFlow:
    """
    Generate particle data with anisotropic flow for testing.

    This class generates particle lists in JETSCAPE or OSCAR output format 
    to test the correct implementation of flow analysis routines.

    Attributes
    ----------
    n_:
        Type of flow harmonics.
    vn_:
        Value of the flow harmonics.
    phi_:
        List containing the azimuths of the particles.
    px_:
        Particle momenta in x-direction.
    py_:
        Particle momenta in y-direction.
    pz_:
        Particle momenta in z-direction.

    Methods
    -------
    generate_dummy_JETSCAPE_file:
        Generate dummy flow data in JETSCAPE format.
    generate_dummy_OSCAR_file:
        Generate dummy flow data in OSCAR format.

    Examples
    --------
    To use the class the GenerateFlow object has to be created with the desired
    anisotropic flow harmonics and then a dummy data file can be created:
    
    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> flow_object = GenerateFlow(v2=0.06, v3=0.02, v4=0.03)
        >>> number_events = 100
        >>> event_multiplicity = 10000
        >>> random_seed = 42
        >>> flow_object.generate_dummy_JETSCAPE_file(path_to_output,number_events,event_multiplicity,random_seed)
        
    """

    def __init__(self, *vn, **vn_kwargs):
        """
        Initialize the class instance.

        Parameters
        ----------
            *vn: Variable-length arguments for harmonic components (vn).
            **vn_kwargs: Keyword arguments for harmonic components (vn) provided as a dictionary.

        Returns
        -------
            None

        Raises
        ------
            TypeError: If the input is not in the expected format.

        Notes
        -----
            - This method initializes the class instance with harmonic 
            components (vn) and sets up other instance variables.
        """
        if not vn and not vn_kwargs:
            self.n_ = self.vn_ = None
        else:
            try:
                vn_dictionary = {int(kw.lstrip('v')): val for kw, val in vn_kwargs.items()}
            except ValueError:
                raise TypeError("Input must have the form of a dictionary with 'vN' "
                                "where N is an integer.")
            vn_dictionary.update((k, v) for k, v in enumerate(vn, start=2) if v is not None and v != 0.)
            kwargs = dict(dtype=float, count=len(vn_dictionary))
            self.n_ = np.fromiter(vn_dictionary.keys(), **kwargs)
            self.vn_ = np.fromiter(vn_dictionary.values(), **kwargs)

        self.phi_ = []
        self.px_ = []
        self.py_ = []
        self.pz_ = []

    def __distribution_function(self,phi):
        """
        Calculate the distribution function value for a given angle.

        Parameters
        ----------
            phi: float
                The azimuthal angle.

        Returns
        -------
            float 
                The value of the distribution function at the given angle.

        Notes
        -----
            - This method calculates the value of the distribution function 
            based on the harmonic components (vn) and angles (phi).
        """
        f = 1. / (2. * np.pi)
        f_harmonic = 1.0

        for term in range(len(self.n_)):
            f_harmonic += 2. * self.vn_[term] * np.cos(self.n_[term] * phi)

        return f * f_harmonic

    def sample_angles(self, multiplicity):
        """
        Sample angles for a given multiplicity according to a given distribution.

        Parameters
        ----------
            multiplicity: int
                The number of angles to sample.

        Returns
        -------
            None
        """
        f_max = (1. + 2. * self.vn_.sum()) / (2. * np.pi)
        phi = []

        while len(phi) < multiplicity:
            random_phi = rd.uniform(0., 2. * np.pi)
            random_dist_val = rd.uniform(0., f_max)
        
            if random_dist_val <= self.__distribution_function(random_phi):
                phi.append(random_phi)

        self.phi_ = phi

    def __thermal_distribution(self,temperature,mass):
        """
        Calculate the momentum magnitude from a thermal distribution.

        Parameters
        ----------
            temperature: float
                The temperature of the system.
            mass: float
                The mass of the particles.

        Returns
        -------
            momentum_radial: float
                The magnitude of the momentum.
        """
        momentum_radial = 0
        energy = 0.
        if temperature > 0.6*mass:
            while True:
                rand_values = [rd.uniform(0., 1.) for _ in range(3)]
                if all(rand_values):
                    rand_a, rand_b, rand_c = rand_values
                    momentum_radial = temperature * (rand_a + rand_b + rand_c)
                    energy = np.sqrt(momentum_radial ** 2. + mass ** 2.)
                    if rd.uniform(0., 1.) < np.exp((momentum_radial - energy) / temperature):
                        break
        else:
            while True:
                r0 = rd.uniform(0., 1.)
                I1 = mass ** 2.
                I2 = 2. * mass * temperature
                I3 = 2. * temperature ** 2.
                Itot = I1 + I2 + I3
                K = 0.0
                if r0 < I1 / Itot:
                    r1 = rd.uniform(0., 1.)
                    if r1 != 0.:
                        K = -temperature * np.log(r1)
                elif r0 < (I1 + I2) / Itot:
                    r1, r2 = rd.uniform(0., 1.), rd.uniform(0., 1.)
                    if r1 != 0. and r2 != 0.:
                        K = -temperature * np.log(r1 * r2)
                else:
                    r1, r2, r3 = rd.uniform(0., 1.), rd.uniform(0., 1.), rd.uniform(0., 1.)
                    if r1 != 0. and r2 != 0. and r3 != 0.:
                        K = -temperature * np.log(r1 * r2 * r3)

                energy = K + mass
                momentum_radial = np.sqrt((energy + mass) * (energy - mass))
                if rd.uniform(0., 1.) < momentum_radial / energy:
                    break

        return momentum_radial
    
    def sample_momenta(self, multiplicity, temperature, mass):
        """
        Sample momenta for a given multiplicity, temperature, and mass from 
        a thermal distribution function.

        Parameters
        ----------
            multiplicity: int
                The number of particles to sample.
            temperature: float
                The temperature of the system.
            mass: float
                The mass of the particles.

        Returns
        -------
            None
        """
        p_abs = [self.__thermal_distribution(temperature, mass) for _ in range(multiplicity)]

        # compute the directions
        azimuths = [self.phi_[p] for p in range(len(p_abs))]
        costheta_values = [rd.uniform(-1., 1.) for _ in range(len(p_abs))]
        polar_values = np.arccos(costheta_values)

        # convert to cartesian
        px = [p_abs[p] * np.sin(polar_values[p]) * np.cos(azimuths[p]) for p in range(len(p_abs))]
        py = [p_abs[p] * np.sin(polar_values[p]) * np.sin(azimuths[p]) for p in range(len(p_abs))]
        pz = [p_abs[p] * np.cos(polar_values[p]) for p in range(len(p_abs))]

        self.px_ = px
        self.py_ = py
        self.pz_ = pz

    def generate_dummy_JETSCAPE_file(self,output_path,number_events,multiplicity,seed):
        """
        Generate a dummy JETSCAPE file with random particle momenta.

        Parameters
        ----------
            output_path: str
                The output file path.
            number_events: int
                The number of events to generate.
            multiplicity: int
                The number of particles per event.
            seed: int
                The random seed for reproducibility.

        Returns
        -------
            None
        """
        rd.seed(seed)
        temperature = 0.140
        mass = 0.138
        pdg = 211
        status = 27

        with open(output_path, "w") as output:
            output.write("#	JETSCAPE_FINAL_STATE	v2	|	N	pid	status	E	Px	Py	Pz\n")

            for event in range(number_events):
                self.sample_angles(multiplicity)
                self.sample_momenta(multiplicity, temperature, mass)

                output.write(f"# Event {event + 1} weight 1 EPangle 0 N_hadrons {multiplicity}\n")
                for particle in range(multiplicity):
                    energy = np.sqrt(
                        self.px_[particle] ** 2.
                        + self.py_[particle] ** 2.
                        + self.pz_[particle] ** 2.
                        + mass ** 2.
                    )
                    output.write("%d %d %d %g %g %g %g\n"
                                 %(particle,pdg,status,energy,self.px_[particle],
                                   self.py_[particle],self.pz_[particle]))

            output.write("#	sigmaGen	0.0	sigmaErr	0.0")
            
    def generate_dummy_OSCAR_file(self,output_path,number_events,multiplicity,seed):
            """
            Generate a dummy OSCAR2013 file with random particle momenta.

            Parameters
            ----------
                output_path: str
                    The output file path.
                number_events: int
                    The number of events to generate.
                multiplicity: int
                    The number of particles per event.
                seed: int
                    The random seed for reproducibility.

            Returns
            -------
                None
            """
            rd.seed(seed)
            temperature = 0.140
            mass = 0.138
            pdg = 211
            status = 27

            with open(output_path, "w") as output:
                output.write("#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge\n")
                output.write("# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e\n")
                output.write("# SMASH-2.2\n")

                for event in range(number_events):
                    self.sample_angles(multiplicity)
                    self.sample_momenta(multiplicity, temperature, mass)

                    output.write(f"# event {event} out {multiplicity}\n")
                    for particle in range(multiplicity):
                        energy = np.sqrt(
                            self.px_[particle] ** 2.
                            + self.py_[particle] ** 2.
                            + self.pz_[particle] ** 2.
                            + mass ** 2.
                        )
                        output.write("%g %g %g %g %g %g %g %g  %g %d %d %d\n"
                                    %(1,1,1,1,mass, energy,self.px_[particle],
                                    self.py_[particle],self.pz_[particle],pdg, particle, 1))

                    output.write(f"# event {event} end 0 impact  -1.000 scattering_projectile_target no")
