from . import FlowInterface
import numpy as np

class ReactionPlaneFlow(FlowInterface.FlowInterface):
    def __init__(self,n=2):
        if not isinstance(n, int):
            raise TypeError('n has to be int')
        elif n <= 0:
            raise ValueError('n-th harmonic with value n<=0 can not be computed')
        else:
            self.n_ = n

    def integrated_flow(self, particle_data):
        flow_event_average = 0. + 0.j
        number_particles = 0.
        for event in range(len(particle_data)):
            flow_event = 0. + 0.j
            for particle in range(len(particle_data[event])):
                weight = 1. if particle_data[event][particle].weight is None else particle_data[event][particle].weight
                pt = particle_data[event][particle].pt_abs()
                phi = particle_data[event][particle].phi()
                flow_event += weight*(pt**self.n_ * np.exp(1j*self.n_*phi) / pt**self.n_)
                number_particles += weight
            if number_particles != 0.:
                flow_event_average += flow_event
            else:
                flow_event_average = 0. + 0.j
        flow_event_average /= number_particles
        return flow_event_average

    def differential_flow(self,particle_data,bins,flow_as_function_of):
        if not isinstance(bins, (list,np.ndarray)):
            raise TypeError('bins has to be list or np.ndarray')
        if not isinstance(flow_as_function_of, str):
            raise TypeError('flow_as_function_of is not a string')
        if flow_as_function_of not in ["pt","rapidity","pseudorapidity"]:
            raise ValueError("flow_as_function_of must be either 'pt', 'rapidity', 'pseudorapidity'")

        particles_bin = []
        for bin in range(len(bins)-1):
            events_bin = []
            for event in range(len(particle_data)):
                particles_event = []
                for particle in particle_data[event]:
                    val = 0.
                    if flow_as_function_of == "pt":
                        val = particle.pt_abs()
                    elif flow_as_function_of == "rapidity":
                        val = particle.momentum_rapidity_Y()
                    elif flow_as_function_of == "pseudorapidity":
                        val = particle.pseudorapidity()
                        print(val)
                    if val >= bins[bin] and val < bins[bin+1]:
                        particles_event.append(particle)
                events_bin.extend([particles_event])
            particles_bin.extend([events_bin])

        return self.__differential_flow_calculation(particles_bin)

    def __differential_flow_calculation(self, binned_particle_data):
        flow_differential = [0.+0.j for i in range(len(binned_particle_data))]
        for bin in range(len(binned_particle_data)):
            number_particles = 0.
            flow_event_average = 0. + 0.j
            for event in range(len(binned_particle_data[bin])):
                flow_event = 0. + 0.j
                for particle in range(len(binned_particle_data[bin][event])):
                    weight = 1. if binned_particle_data[bin][event][particle].weight is None else binned_particle_data[bin][event][particle].weight
                    pt = binned_particle_data[bin][event][particle].pt_abs()
                    phi = binned_particle_data[bin][event][particle].phi()
                    flow_event += weight*(pt**self.n_ * np.exp(1j*self.n_*phi) / pt**self.n_)
                    number_particles += weight
                flow_event_average += flow_event
            if number_particles != 0.:
                flow_event_average /= number_particles
            else:
                flow_event_average = 0. + 0.j
            flow_differential[bin] = flow_event_average
        return flow_differential

# from ..Jetscape import Jetscape
# oscar1 = Jetscape("/home/niklas/Downloads/new_testdata_no_rot.dat")
# liste1 = oscar1.particle_objects_list()

# test1 = ReactionPlaneFlow()
# print(test1.differential_flow(liste1, [0.1,0.2,0.3,0.5,1,2,3,4.5], "pt"))