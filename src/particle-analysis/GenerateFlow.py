import numpy as np
import random as rd

class GenerateFlow:

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
        f = (1./(2.*np.pi))
        f_harmonic = 1.
        for term in range(len(self.n_)):
            f_harmonic += 2. * self.vn_[term] * np.cos(self.n_[term]*phi)
        return f * f_harmonic

    def sample_angles(self, multiplicity):
        f_max = (1. + 2.*self.vn_.sum()) / (2.*np.pi)
        phi = []
        number_particles = 0
        while number_particles < multiplicity:
            random_phi = rd.uniform(0.,2.*np.pi)
            random_dist_val = rd.uniform(0.,f_max)
            if random_dist_val <= self.__distribution_function(random_phi):
                phi.append(random_phi)
                number_particles += 1
        self.phi_ = phi

    def __thermal_distribution(self,temperature,mass):
        momentum_radial = 0
        energy = 0.
        if temperature > 0.6*mass:
            run = True
            while run:
                rand_a = rd.uniform(0.,1.)
                rand_b = rd.uniform(0.,1.)
                rand_c = rd.uniform(0.,1.)
                if rand_a != 0. and rand_b != 0. and rand_c != 0.:
                    momentum_radial = temperature * (rand_a + rand_b + rand_c)
                    energy = np.sqrt(momentum_radial**2. + mass**2.)
                    if rd.uniform(0.,1.) < np.exp((momentum_radial - energy)/temperature):
                        run = False
        else:
            run = True
            while run:
                r0 = rd.uniform(0.,1.)
                I1 = mass**2.
                I2 = 2.*mass*temperature
                I3 = 2.*temperature**2.
                Itot = I1 + I2 + I3
                K = 0.
                if r0 < I1 / Itot:
                    r1 = rd.uniform(0.,1.)
                    if r1 != 0.:
                        K = -temperature * np.log(r1)
                elif r0 < (I1 + I2) / Itot:
                    r1 = rd.uniform(0.,1.)
                    r2 = rd.uniform(0.,1.)
                    if r1 != 0. and r2 != 0.:
                        K = -temperature * np.log(r1 * r2)
                else:
                    r1 = rd.uniform(0.,1.)
                    r2 = rd.uniform(0.,1.)
                    r3 = rd.uniform(0.,1.)
                    if r1 != 0. and r2 != 0. and r3 != 0.:
                        K = -temperature * np.log(r1 * r2 * r3)
                energy = K + mass
                momentum_radial = np.sqrt((energy + mass)*(energy - mass))
                if rd.uniform(0.,1.) < momentum_radial / energy:
                    run = False
        return momentum_radial

    def sample_momenta(self,multiplicity,temperature,mass):
        p_abs = []
        for p in range(multiplicity):
            p_abs.append(self.__thermal_distribution(temperature,mass))

        # compute the directions
        px = []
        py = []
        pz = []
        for p in range(len(p_abs)):
            azimuth = self.phi_[p]
            # random pz component (without destroying the generated flow)
            costheta = rd.uniform(-1.,1.)
            polar = np.arccos(costheta)

            # convert to cartesian
            px.append(p_abs[p] * np.sin(polar) * np.cos(azimuth))
            py.append(p_abs[p] * np.sin(polar) * np.sin(azimuth))
            pz.append(p_abs[p] * np.cos(polar))
        self.px_ = px
        self.py_ = py
        self.pz_ = pz

    def generate_dummy_JETSCAPE_file(self,output_path,number_events,multiplicity,seed):
        rd.seed(seed)
        temperature = 0.140
        mass = 0.138
        pdg = 211
        status = 27

        output = open(output_path, "w")
        output.write("#	JETSCAPE_FINAL_STATE	v2	|	N	pid	status	E	Px	Py	Pz\n")

        for event in range(number_events):
            self.sample_angles(multiplicity)
            self.sample_momenta(multiplicity,temperature,mass)

            output.write(f"#	Event	{event+1}	weight	1	EPangle	0	N_hadrons	{multiplicity}\n")
            for particle in range(multiplicity):
                energy = np.sqrt(self.px_[particle]**2. 
                                 + self.py_[particle]**2. 
                                 + self.pz_[particle]**2. 
                                 + mass**2.)
                output.write("%d %d %d %g %g %g %g\n"%(particle,pdg,status,energy,self.px_[particle],self.py_[particle],self.pz_[particle])) 
        output.write(f"#	sigmaGen	0.0	sigmaErr	0.0")
        output.close()


#flow = GenerateFlow(v2=0.06, v3=0.0, v4=0.0)
#flow.generate_dummy_JETSCAPE_file("/home/hendrik/Git/particle-analysis/src/particle-analysis/flow_test_data.dat",1000,10000,42)
