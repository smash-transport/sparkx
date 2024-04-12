#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
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
        Generate a dummy JETSCAPE file with random particle momenta resulting in
        the same flow for all transverse momenta.
    generate_dummy_JETSCAPE_file_realistic_pt_shape:
        Generate a dummy JETSCAPE file with particles having flow with a more
        realistic transverse momentum distribution.
    generate_dummy_JETSCAPE_file_multi_particle_correlations:
        Generate a dummy JETSCAPE file with random particle momenta resulting in
        the same flow for all transverse momenta. With this function multi-particle
        correlations can be introduced.
    generate_dummy_JETSCAPE_file_realistic_pt_shape_multi_particle_correlations:
        Generate a dummy JETSCAPE file with particles having flow with a more
        realistic transverse momentum distribution. With this function multi-particle
        correlations can be introduced.
    generate_dummy_OSCAR_file:
        Generate dummy flow data in OSCAR format.
    generate_dummy_OSCAR_file_realistic_pt_shape:
        Generate a dummy OSCAR2013 file with particles having flow with a more
        realistic transverse momentum distribution.
    generate_dummy_OSCAR_file_multi_particle_correlations:
        Generate a dummy OSCAR2013 file with random particle momenta resulting in
        the same flow for all transverse momenta. With this function multi-particle
        correlations can be introduced.
    generate_dummy_OSCAR_file_realistic_pt_shape_multi_particle_correlations:
        Generate a dummy OSCAR2013 file with particles having flow with a more
        realistic transverse momentum distribution. With this function multi-particle
        correlations can be introduced.

    Examples
    --------
    To use the class the GenerateFlow object has to be created with the desired
    anisotropic flow harmonics and then a dummy data file can be created:

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.flow.GenerateFlow import GenerateFlow
        >>>
        >>> flow_object = GenerateFlow(v2=0.06, v3=0.02, v4=0.03)
        >>> number_events = 100
        >>> event_multiplicity = 10000
        >>> random_seed = 42
        >>> flow_object.generate_dummy_JETSCAPE_file(path_to_output,number_events,event_multiplicity,random_seed)

    Notes
    -----
    If you use the :py:meth:`generate_dummy_JETSCAPE_file_realistic_pt_shape` or
    :py:meth:`generate_dummy_OSCAR_file_realistic_pt_shape` keep in mind, that the
    flow values given during construction are used for the saturation value of the
    flow at large transverse momentum. They do **not** reflect the value of the
    integrated flow.

    The implemented method for the more realistic transverse momentum profile is
    taken from Nicolas Borghini implemented in this
    `event generator <https://www.physik.uni-bielefeld.de/~borghini/Software/flow_analysis_codes/generator.cc>`__.
    """

    def __init__(self, *vn, **vn_kwargs):
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

    def __sample_angles(self, multiplicity):
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

    def __sample_momenta_thermal(self, multiplicity, temperature, mass):
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

    def __artificial_pT_distribution(self, pT, pTmin, pT0, pT1, T0):
        """
        Calculate the artificial pT distribution.

        Parameters
        ----------
        pT : float
            Transverse momentum.
        pTmin : float
            Minimum transverse momentum.
        pT0 : float
            Lower threshold for the flat region.
        pT1 : float
            Cutoff transverse momentum.
        T0 : float
            Inverse slope parameter.

        Returns
        -------
        float
            The value of the artificial transverse momentum distribution.

        Notes
        -----
        This function gives the transverse-momentum distribution, following a simple functional shape:

        - dN/dpT is flat for 0 < pT < pT0.
        - Decreases exponentially (with inverse slope parameter T0) for pT0 < pT < pT1.
        - Decreases with an inverse power-law for pT > pT1.

        Momenta (pT, pT0...) are given in GeV.
        """
        value = 0.
        if pT < pTmin:
            value = 0.
        elif pT <= pT0:
            value = 1.
        elif pT <= pT1:
            value = np.exp(-(pT-pT0)/T0)
        else:
            value = np.exp(-(pT1-pT0)/T0) * (pT1/pT)**7
        return value

    def __artificial_flow_pT_shape(self, pT, pT0_bis, pT_sat, vn_sat):
        """
        Calculate the artificial flow pT shape.

        Parameters
        ----------
        pT : float
            Transverse momentum.
        pT0 : float
            Lower threshold for the quadratic rise.
        pT_sat : float
            Saturation point for the quadratic rise.
        vn_sat : float
            Saturation value.

        Returns
        -------
        float
            The value of the artificial flow pT shape.

        Notes
        -----
        This function mimics the shape measured at RHIC:

        - Quadratic rise for 0 < pT < pT0.
        - Linear rise for pT0 < pT < pT_sat - pT0.
        - Quadratic rise again for pT_sat - pT0 < pT < pT_sat.
        - Constant value vn_sat for pT > pT_sat.
        """
        value = 0.
        vn_pT0_bis = 0.5 * vn_sat * pT0_bis / (pT_sat - pT0_bis)

        if pT < pT0_bis:
            value = (pT / pT0_bis)**2. * vn_pT0_bis
        elif pT < (pT_sat - pT0_bis):
            value = (vn_sat - 2.*vn_pT0_bis) * (pT - pT0_bis) / (pT_sat - 2.*pT0_bis) + vn_pT0_bis
        elif pT < pT_sat:
            value = vn_sat - ((pT - pT_sat) / pT0_bis)**2. * vn_pT0_bis
        else:
            value = vn_sat

        return value

    def __distribution_function_pT_differential(self, phi, vn_pt_list):
        """
        Calculates the pT-differential distribution function for a given
        azimuthal angle.

        Parameters
        ----------
        phi : float
            The azimuthal angle at which to calculate the distribution function.
        vn_pt_list : list
            A list of flow harmonics vn for different transverse momenta pT.

        Returns
        -------
        float
            The calculated pT-differential distribution function value.
        """
        f_harmonic = 1.0
        f_norm = 1.0
        for term in range(len(self.n_)):
            f_harmonic += 2. * vn_pt_list[term] * np.cos(self.n_[term] * phi)
            f_norm += 2. * np.abs(vn_pt_list[term])

        return f_harmonic / f_norm

    def __create_k_particle_correlations(self, multiplicity, k_particle_correlation, correlation_fraction):
        """
        Generate momentum components with k-particle correlations.

        Parameters
        ----------
        multiplicity : int
            The desired multiplicity of the event.
        k_particle_correlation : int
            The number of particles to be correlated.
        correlation_fraction : float
            The fraction of particles to be correlated.

        Returns
        -------
        None
            Updates the internal arrays (px_, py_, pz_) with the 
            generated correlated momenta.
        """
        px = []
        py = []
        pz = []
        idx = 0
        while len(px) <= multiplicity:
            if rd.random() <= correlation_fraction:
                for k in range(k_particle_correlation):
                    px.append(self.px_[idx])
                    py.append(self.py_[idx])
                    pz.append(self.pz_[idx])
                idx += 1
            else:
                px.append(self.px_[idx])
                py.append(self.py_[idx])
                pz.append(self.pz_[idx])
                idx += 1

        self.px_ = px
        self.py_ = py
        self.pz_ = pz

    def __generate_flow_realistic_pt_distribution(self, multiplicity, reaction_plane_angle):
        pTmax = 4.5
        pTmin = 0.1
        pT0 = 0.5
        pT1 = 3.0
        T0 = 0.6
        pT0_bis = 0.1

        pT_sat = 1.5

        for particle in range(multiplicity):
            pT_chosen = 0.
            need_momentum = True
            while need_momentum:
                pT = rd.random()*pTmax
                if rd.random() < self.__artificial_pT_distribution(pT, pTmin, pT0, pT1, T0):
                    pT_chosen = pT
                    need_momentum = False

            vn_pt_list = []
            for harmonic in range(len(self.n_)):
                vn_pt_list.append(self.__artificial_flow_pT_shape(pT,pT0_bis,pT_sat,self.vn_[harmonic]))

            phi_chosen = 0.
            need_angle = True
            while need_angle:
                phi = 2.*np.pi*rd.random()
                if rd.random() < self.__distribution_function_pT_differential(phi,vn_pt_list):
                    phi_chosen = phi
                    need_angle = False

            if reaction_plane_angle != 0.:
                phi_chosen += reaction_plane_angle
                if phi_chosen > 2.*np.pi:
                    phi_chosen -= 2.*np.pi

            # convert to cartesian
            self.px_.append(pT_chosen * np.cos(phi_chosen))
            self.py_.append(pT_chosen * np.sin(phi_chosen))
            self.pz_.append(rd.uniform(-1,1)*pTmax)

    def generate_dummy_JETSCAPE_file(self,output_path,number_events,multiplicity,seed):
        """
        Generate a dummy JETSCAPE file with random particle momenta resulting in
        the same flow for all transverse momenta.

        For simplicity we generate :math:`\\pi^+` particles with a mass of
        :math:`m_{\\pi^+}=0.138` GeV from a thermal distribution with a
        temperature of :math:`T=0.140` GeV.

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
        if not isinstance(output_path,str):
            raise TypeError("'output_path' is not a str")
        if not isinstance(number_events,int):
            raise TypeError("'number_events' is not int")
        if not isinstance(multiplicity,int):
            raise TypeError("'multiplicity' is not int")
        if not isinstance(seed,int):
            raise TypeError("'seed' is not int")
        if number_events < 1 or multiplicity < 1:
            raise ValueError("'number_events' and/or 'multiplicity' must be larger than 0")
        
        rd.seed(seed)
        temperature = 0.140
        mass = 0.138
        pdg = 211
        status = 27

        with open(output_path, "w") as output:
            output.write("#	JETSCAPE_FINAL_STATE	v2	|	N	pid	status	E	Px	Py	Pz\n")

            for event in range(number_events):
                self.__sample_angles(multiplicity)
                self.__sample_momenta_thermal(multiplicity, temperature, mass)

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
                self.px_.clear()
                self.py_.clear()
                self.pz_.clear()

            output.write("#	sigmaGen	0.0	sigmaErr	0.0")

    def generate_dummy_JETSCAPE_file_realistic_pt_shape(self,output_path,number_events,multiplicity,seed,random_reaction_plane=True):
        """
        Generate a dummy JETSCAPE file with particles having flow with a more
        realistic transverse momentum distribution.

        For more details on the chosen parameters have a look at the source code.

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
            random_reaction_plane: bool
                Switch for random reaction plane angle. Default is `True`.
                Should be switched off for testing ReactionPlaneFlow.

        Returns
        -------
            None
        """
        if not isinstance(output_path,str):
            raise TypeError("'output_path' is not a str")
        if not isinstance(number_events,int):
            raise TypeError("'number_events' is not int")
        if not isinstance(multiplicity,int):
            raise TypeError("'multiplicity' is not int")
        if not isinstance(seed,int):
            raise TypeError("'seed' is not int")
        if number_events < 1 or multiplicity < 1:
            raise ValueError("'number_events' and/or 'multiplicity' must be larger than 0")
        
        rd.seed(seed)
        mass = 0.138
        pdg = 211
        status = 27

        with open(output_path, "w") as output:
            output.write("#	JETSCAPE_FINAL_STATE	v2	|	N	pid	status	E	Px	Py	Pz\n")

            for event in range(number_events):
                if random_reaction_plane:
                    reaction_plane_angle = 2.*np.pi*rd.random()
                else:
                    reaction_plane_angle = 0.
                self.__generate_flow_realistic_pt_distribution(multiplicity,reaction_plane_angle)

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
                self.px_.clear()
                self.py_.clear()
                self.pz_.clear()

            output.write("#	sigmaGen	0.0	sigmaErr	0.0")

    def generate_dummy_JETSCAPE_file_multi_particle_correlations(self,output_path,number_events,multiplicity,seed,k_particle_correlation,correlation_fraction):
        """
        Generate a dummy JETSCAPE file with random particle momenta resulting in
        the same flow for all transverse momenta. A fraction of k-particle
        correlations can be introduced.

        For simplicity we generate :math:`\\pi^+` particles with a mass of
        :math:`m_{\\pi^+}=0.138` GeV from a thermal distribution with a
        temperature of :math:`T=0.140` GeV.

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
            k_particle_correlation: int
                The order of k-particle correlations.
            correlation_fraction:
                The fraction of correlated particles.

        Returns
        -------
            None
        """
        if not isinstance(output_path,str):
            raise TypeError("'output_path' is not a str")
        if not isinstance(number_events,int):
            raise TypeError("'number_events' is not int")
        if not isinstance(multiplicity,int):
            raise TypeError("'multiplicity' is not int")
        if not isinstance(seed,int):
            raise TypeError("'seed' is not int")
        if number_events < 1 or multiplicity < 1:
            raise ValueError("'number_events' and/or 'multiplicity' must be larger than 0")
        if not isinstance(k_particle_correlation,int):
            raise TypeError("'k_particle_correlation' is not int")
        if not isinstance(correlation_fraction,float):
            raise TypeError("'correlation_fraction' is not float")
        if k_particle_correlation < 2:
            raise ValueError("'k_particle_correlation' must be at least 2")
        if correlation_fraction < 0. or correlation_fraction > 1.:
            raise ValueError("'correlation_fraction' must be between 0 and 1")
    
        rd.seed(seed)
        temperature = 0.140
        mass = 0.138
        pdg = 211
        status = 27

        with open(output_path, "w") as output:
            output.write("#	JETSCAPE_FINAL_STATE	v2	|	N	pid	status	E	Px	Py	Pz\n")

            for event in range(number_events):
                self.__sample_angles(multiplicity)
                self.__sample_momenta_thermal(multiplicity, temperature, mass)
                self.__create_k_particle_correlations(multiplicity,k_particle_correlation,correlation_fraction)

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
                self.px_.clear()
                self.py_.clear()
                self.pz_.clear()

            output.write("#	sigmaGen	0.0	sigmaErr	0.0")

    def generate_dummy_JETSCAPE_file_realistic_pt_shape_multi_particle_correlations(self,output_path,number_events,multiplicity,seed,k_particle_correlation,correlation_fraction,random_reaction_plane=True):
        """
        Generate a dummy JETSCAPE file with particles having flow with a more
        realistic transverse momentum distribution. A fraction of k-particle
        correlations can be introduced.

        For more details on the chosen parameters have a look at the source code.

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
            k_particle_correlation: int
                The order of k-particle correlations.
            correlation_fraction:
                The fraction of correlated particles.
            random_reaction_plane: bool
                Switch for random reaction plane angle. Default is `True`.
                Should be switched off for testing ReactionPlaneFlow.

        Returns
        -------
            None
        """
        if not isinstance(output_path,str):
            raise TypeError("'output_path' is not a str")
        if not isinstance(number_events,int):
            raise TypeError("'number_events' is not int")
        if not isinstance(multiplicity,int):
            raise TypeError("'multiplicity' is not int")
        if not isinstance(seed,int):
            raise TypeError("'seed' is not int")
        if number_events < 1 or multiplicity < 1:
            raise ValueError("'number_events' and/or 'multiplicity' must be larger than 0")
        if not isinstance(k_particle_correlation,int):
            raise TypeError("'k_particle_correlation' is not int")
        if not isinstance(correlation_fraction,float):
            raise TypeError("'correlation_fraction' is not float")
        if k_particle_correlation < 2:
            raise ValueError("'k_particle_correlation' must be at least 2")
        if correlation_fraction < 0. or correlation_fraction > 1.:
            raise ValueError("'correlation_fraction' must be between 0 and 1")
        
        rd.seed(seed)
        mass = 0.138
        pdg = 211
        status = 27

        with open(output_path, "w") as output:
            output.write("#	JETSCAPE_FINAL_STATE	v2	|	N	pid	status	E	Px	Py	Pz\n")

            for event in range(number_events):
                if random_reaction_plane:
                    reaction_plane_angle = 2.*np.pi*rd.random()
                else:
                    reaction_plane_angle = 0.
                self.__generate_flow_realistic_pt_distribution(multiplicity,reaction_plane_angle)
                self.__create_k_particle_correlations(multiplicity,k_particle_correlation,correlation_fraction)

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
                self.px_.clear()
                self.py_.clear()
                self.pz_.clear()

            output.write("#	sigmaGen	0.0	sigmaErr	0.0")

    def generate_dummy_OSCAR_file(self,output_path,number_events,multiplicity,seed):
        """
        Generate a dummy OSCAR2013 file with random particle momenta
        resulting in the same flow for all transverse momenta.

        For simplicity we generate :math:`\\pi^+` particles with a mass of
        :math:`m_{\\pi^+}=0.138` GeV from a thermal distribution with a
        temperature of :math:`T=0.140` GeV.

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
        if not isinstance(output_path,str):
            raise TypeError("'output_path' is not a str")
        if not isinstance(number_events,int):
            raise TypeError("'number_events' is not int")
        if not isinstance(multiplicity,int):
            raise TypeError("'multiplicity' is not int")
        if not isinstance(seed,int):
            raise TypeError("'seed' is not int")
        if number_events < 1 or multiplicity < 1:
            raise ValueError("'number_events' and/or 'multiplicity' must be larger than 0")
        
        rd.seed(seed)
        temperature = 0.140
        mass = 0.138
        pdg = 211

        with open(output_path, "w") as output:
            output.write("#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge\n")
            output.write("# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e\n")
            output.write("# SMASH-2.2\n")

            for event in range(number_events):
                self.__sample_angles(multiplicity)
                self.__sample_momenta_thermal(multiplicity, temperature, mass)

                output.write(f"# event {event} out {multiplicity}\n")
                for particle in range(multiplicity):
                    energy = np.sqrt(
                        self.px_[particle] ** 2.
                        + self.py_[particle] ** 2.
                        + self.pz_[particle] ** 2.
                        + mass ** 2.
                    )
                    output.write("%g %g %g %g %g %g %g %g %g %d %d %d\n"
                                %(1,1,1,1,mass, energy,self.px_[particle],
                                self.py_[particle],self.pz_[particle],pdg, particle, 1))

                self.px_.clear()
                self.py_.clear()
                self.pz_.clear()

                output.write(f"# event {event} end 0 impact  -1.000 scattering_projectile_target no\n")

    def generate_dummy_OSCAR_file_realistic_pt_shape(self,output_path,number_events,multiplicity,seed,random_reaction_plane=True):
        """
        Generate a dummy OSCAR2013 file with particles having flow with a more
        realistic transverse momentum distribution.

        For more details on the chosen parameters have a look at the source code.

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
            random_reaction_plane: bool
                Switch for random reaction plane angle. Default is `True`.
                Should be switched off for testing ReactionPlaneFlow.

        Returns
        -------
            None
        """
        if not isinstance(output_path,str):
            raise TypeError("'output_path' is not a str")
        if not isinstance(number_events,int):
            raise TypeError("'number_events' is not int")
        if not isinstance(multiplicity,int):
            raise TypeError("'multiplicity' is not int")
        if not isinstance(seed,int):
            raise TypeError("'seed' is not int")
        if number_events < 1 or multiplicity < 1:
            raise ValueError("'number_events' and/or 'multiplicity' must be larger than 0")
        
        rd.seed(seed)
        mass = 0.138
        pdg = 211

        with open(output_path, "w") as output:
            output.write("#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge\n")
            output.write("# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e\n")
            output.write("# SMASH-2.2\n")

            for event in range(number_events):
                if random_reaction_plane:
                    reaction_plane_angle = 2.*np.pi*rd.random()
                else:
                    reaction_plane_angle = 0.
                self.__generate_flow_realistic_pt_distribution(multiplicity,reaction_plane_angle)

                output.write(f"# event {event} out {multiplicity}\n")
                for particle in range(multiplicity):
                    energy = np.sqrt(
                        self.px_[particle] ** 2.
                        + self.py_[particle] ** 2.
                        + self.pz_[particle] ** 2.
                        + mass ** 2.
                    )
                    output.write("%g %g %g %g %g %g %g %g %g %d %d %d\n"
                                %(1,1,1,1,mass, energy,self.px_[particle],
                                self.py_[particle],self.pz_[particle],pdg, particle, 1))

                self.px_.clear()
                self.py_.clear()
                self.pz_.clear()

                output.write(f"# event {event} end 0 impact  -1.000 scattering_projectile_target no\n")

    def generate_dummy_OSCAR_file_multi_particle_correlations(self,output_path,number_events,multiplicity,seed,k_particle_correlation,correlation_fraction):
        """
        Generate a dummy OSCAR2013 file with random particle momenta
        resulting in the same flow for all transverse momenta. A fraction of 
        k-particle correlations can be introduced.

        For simplicity we generate :math:`\\pi^+` particles with a mass of
        :math:`m_{\\pi^+}=0.138` GeV from a thermal distribution with a
        temperature of :math:`T=0.140` GeV.

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
            k_particle_correlation: int
                The order of k-particle correlations.
            correlation_fraction:
                The fraction of correlated particles.

        Returns
        -------
            None
        """
        if not isinstance(output_path,str):
            raise TypeError("'output_path' is not a str")
        if not isinstance(number_events,int):
            raise TypeError("'number_events' is not int")
        if not isinstance(multiplicity,int):
            raise TypeError("'multiplicity' is not int")
        if not isinstance(seed,int):
            raise TypeError("'seed' is not int")
        if number_events < 1 or multiplicity < 1:
            raise ValueError("'number_events' and/or 'multiplicity' must be larger than 0")
        if not isinstance(k_particle_correlation,int):
            raise TypeError("'k_particle_correlation' is not int")
        if not isinstance(correlation_fraction,float):
            raise TypeError("'correlation_fraction' is not float")
        if k_particle_correlation < 2:
            raise ValueError("'k_particle_correlation' must be at least 2")
        if correlation_fraction < 0. or correlation_fraction > 1.:
            raise ValueError("'correlation_fraction' must be between 0 and 1")
        
        rd.seed(seed)
        temperature = 0.140
        mass = 0.138
        pdg = 211

        with open(output_path, "w") as output:
            output.write("#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge\n")
            output.write("# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e\n")
            output.write("# SMASH-2.2\n")

            for event in range(number_events):
                self.__sample_angles(multiplicity)
                self.__sample_momenta_thermal(multiplicity, temperature, mass)
                self.__create_k_particle_correlations(multiplicity,k_particle_correlation,correlation_fraction)

                output.write(f"# event {event} out {multiplicity}\n")
                for particle in range(multiplicity):
                    energy = np.sqrt(
                        self.px_[particle] ** 2.
                        + self.py_[particle] ** 2.
                        + self.pz_[particle] ** 2.
                        + mass ** 2.
                    )
                    output.write("%g %g %g %g %g %g %g %g %g %d %d %d\n"
                                %(1,1,1,1,mass, energy,self.px_[particle],
                                self.py_[particle],self.pz_[particle],pdg, particle, 1))

                self.px_.clear()
                self.py_.clear()
                self.pz_.clear()

                output.write(f"# event {event} end 0 impact  -1.000 scattering_projectile_target no\n")

    def generate_dummy_OSCAR_file_realistic_pt_shape_multi_particle_correlations(self,output_path,number_events,multiplicity,seed,k_particle_correlation,correlation_fraction,random_reaction_plane=True):
        """
        Generate a dummy OSCAR2013 file with particles having flow with a more
        realistic transverse momentum distribution. A fraction of k-particle 
        correlations can be introduced.

        For more details on the chosen parameters have a look at the source code.

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
            k_particle_correlation: int
                The order of k-particle correlations.
            correlation_fraction:
                The fraction of correlated particles.
            random_reaction_plane: bool
                Switch for random reaction plane angle. Default is `True`.
                Should be switched off for testing ReactionPlaneFlow.

        Returns
        -------
            None
        """
        if not isinstance(output_path,str):
            raise TypeError("'output_path' is not a str")
        if not isinstance(number_events,int):
            raise TypeError("'number_events' is not int")
        if not isinstance(multiplicity,int):
            raise TypeError("'multiplicity' is not int")
        if not isinstance(seed,int):
            raise TypeError("'seed' is not int")
        if number_events < 1 or multiplicity < 1:
            raise ValueError("'number_events' and/or 'multiplicity' must be larger than 0")
        if not isinstance(k_particle_correlation,int):
            raise TypeError("'k_particle_correlation' is not int")
        if not isinstance(correlation_fraction,float):
            raise TypeError("'correlation_fraction' is not float")
        if k_particle_correlation < 2:
            raise ValueError("'k_particle_correlation' must be at least 2")
        if correlation_fraction < 0. or correlation_fraction > 1.:
            raise ValueError("'correlation_fraction' must be between 0 and 1")
        
        rd.seed(seed)
        mass = 0.138
        pdg = 211

        with open(output_path, "w") as output:
            output.write("#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge\n")
            output.write("# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e\n")
            output.write("# SMASH-2.2\n")

            for event in range(number_events):
                if random_reaction_plane:
                    reaction_plane_angle = 2.*np.pi*rd.random()
                else:
                    reaction_plane_angle = 0.
                self.__generate_flow_realistic_pt_distribution(multiplicity,reaction_plane_angle)
                self.__create_k_particle_correlations(multiplicity,k_particle_correlation,correlation_fraction)

                output.write(f"# event {event} out {multiplicity}\n")
                for particle in range(multiplicity):
                    energy = np.sqrt(
                        self.px_[particle] ** 2.
                        + self.py_[particle] ** 2.
                        + self.pz_[particle] ** 2.
                        + mass ** 2.
                    )
                    output.write("%g %g %g %g %g %g %g %g %g %d %d %d\n"
                                %(1,1,1,1,mass, energy,self.px_[particle],
                                self.py_[particle],self.pz_[particle],pdg, particle, 1))

                self.px_.clear()
                self.py_.clear()
                self.pz_.clear()

                output.write(f"# event {event} end 0 impact  -1.000 scattering_projectile_target no\n")