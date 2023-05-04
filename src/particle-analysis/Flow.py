import numpy as np
import warnings

class Flow:
    '''
    flow_type:
        "ReactionPlane", "EventPlane", "ScalarProduct", "QCumulants", "LeeYangZeros"
    '''
    def __init__(self,flow_type=None,n=None):
        if flow_type == None:
            warnings.warn('No flow type selected, set QCumulants as default')
            self.flow_type = "QCumulants"
        elif not isinstance(flow_type, str):
            raise TypeError('flow_type has to be a string')
        elif flow_type not in ["ReactionPlane", "EventPlane", "ScalarProduct", "QCumulants", "LeeYangZeros"]:
            raise ValueError("Invalid flow_type given, choose one of the following flow_types: 'EventPlane', 'ScalarProduct', 'QCumulants', 'LeeYangZeros'")
        else:
            self.flow_type = flow_type

        if n == None:
            warnings.warn("No 'n' for the flow harmonic given, default set to 2")
            self.n = 2
        elif not isinstance(n, int):
            raise TypeError('n has to be int')
        elif n <= 0:
            raise ValueError('n-th harmonic with value n<=0 can not be computed')
        else:
            self.n = n

        self.integrated_flow_ = 0.+0.j
        self.differential_flow_ = []

    def __integrated_flow_reaction_plane(self,particle_data):
        flow_event_average = 0. + 0.j
        reduce_nevents = 0. # triggered, if there is no particle in the event
        for event in range(len(particle_data)):
            flow_event = 0. + 0.j
            for particle in range(len(particle_data[event])):
                pt = particle_data[event][particle].pt_abs()
                phi = particle_data[event][particle].phi()
                flow_event += pt**self.n * np.exp(1j*self.n*phi) / pt**self.n
            if len(particle_data[event]) != 0:
                flow_event /= len(particle_data[event])
            else:
                flow_event = 0. + 0.j
            flow_event_average += flow_event
        flow_event_average /= len(particle_data)-reduce_nevents
        self.integrated_flow_ = flow_event_average

    def __integrated_flow_event_plane(self):
        return 0
    
    def __integrated_flow_scalar_product(self):
        return 0
    
    def __integrated_flow_Q_cumulants(self):
        return 0
    
    def __integrated_flow_LeeYang_zeros(self):
        return 0

    def integrated_flow(self,particle_data):
        """
        Compute the integrated anisotropic flow.

        Return
        ------
        integrated_flow_: complex
        """
        if self.flow_type == "ReactionPlane":
            self.__integrated_flow_reaction_plane(particle_data)
        return self.integrated_flow_

    def __differential_flow_reaction_plane(self,binned_particle_data):
        flow_differential = [0.+0.j for i in range(len(binned_particle_data))]
        reduce_nevents = 0. # triggered, if there is no particle in the event
        for bin in range(len(binned_particle_data)):
            flow_event_average = 0. + 0.j
            for event in range(len(binned_particle_data[bin])):
                flow_event = 0. + 0.j
                for particle in range(len(binned_particle_data[bin][event])):
                    pt = binned_particle_data[bin][event][particle].pt_abs()
                    phi = binned_particle_data[bin][event][particle].phi()
                    flow_event += pt**self.n * np.exp(1j*self.n*phi) / pt**self.n
                if len(binned_particle_data[bin][event]) != 0:
                    flow_event /= len(binned_particle_data[bin][event])
                else:
                    flow_event = 0. + 0.j
                flow_event_average += flow_event
            flow_event_average /= len(binned_particle_data[bin])-reduce_nevents
            flow_differential[bin] = flow_event_average
        self.differential_flow_ = flow_differential

    def __differential_flow_event_plane(self):
        return 0
    
    def __differential_flow_scalar_product(self):
        return 0
    
    def __differential_flow_Q_cumulants(self):
        return 0
    
    def __differential_flow_LeeYang_zeros(self):
        return 0

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
                    if flow_as_function_of == "rapidity":
                        val = particle.momentum_rapidity_Y()
                    if flow_as_function_of == "pseudorapidity":
                        val = particle.pseudorapidity()
                    if val >= bins[bin] and val < bins[bin+1]:
                        particles_event.append(particle)
                events_bin.extend([particles_event])
            particles_bin.extend([events_bin])

        if self.flow_type == "ReactionPlane":
            self.__differential_flow_reaction_plane(particles_bin)
        return self.differential_flow_



from OscarClass import Oscar
data = Oscar("/home/hendrik/Git/particle-analysis/src/particle-analysis/particle_lists.oscar").charged_particles().pseudorapidity_cut(0.5).particle_objects_list()

a = Flow("ReactionPlane",2)
print(a.integrated_flow(data))

b = Flow("ReactionPlane",2)
print(b.differential_flow(data,[0.,0.5,1.,2.,3.],"pt"))

