import numpy as np
import warnings

class Flow:
    '''
    flow_type:
        "ReactionPlane", "EventPlane", "ScalarProduct", "QCumulants", "LeeYangZeros", "PCA"
    n: int
        Integer for the n-th flow harmonic
    weight: str
        String defining the weights in the calculation of the Q vectors.
        Options are "pt", "pt2", "ptn", "rapidity".
        There is no effect for the ReactionPlane method.
    '''
    def __init__(self,flow_type=None,n=None,weight=None):
        if flow_type == None:
            warnings.warn('No flow type selected, set QCumulants as default')
            self.flow_type = "QCumulants"
        elif not isinstance(flow_type, str):
            raise TypeError('flow_type has to be a string')
        elif flow_type not in ["ReactionPlane", "EventPlane", "ScalarProduct", "QCumulants", "LeeYangZeros", "PCA"]:
            raise ValueError("Invalid flow_type given, choose one of the following flow_types: 'EventPlane', 'ScalarProduct', 'QCumulants', 'LeeYangZeros', 'PCA'")
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

        if self.flow_type not in ["ReactionPlane"]:
            if weight == None:
                warnings.warn("No weight for the flow harmonic was set. Set pt**2 weight as default. Make sure this is wanted behavior!")
                self.weight = "pt2"
            elif not isinstance(weight, str):
                raise TypeError('weight has to be a string')
            elif weight not in ["pt", "pt2", "ptn", "rapidity", "pseudorapidity"]:
                raise ValueError("Invalid weight given, choose one of the following: 'pt', 'pt2', 'ptn', 'rapidity', 'pseudorapidity'")
            else:
                self.weight = weight

        self.integrated_flow_ = 0.+0.j
        self.integrated_flow_err_ = 0.+0.j
        self.differential_flow_ = []
        self.differential_flow_err_ = []

    def __compute_particle_weights(self, particle_data):
        event_weights = []
        for event in range(len(particle_data)):
            particle_weights = []
            for particle in particle_data[event]:
                weight = 0.
                if self.weight == "pt":
                    weight = particle.pt_abs()
                if self.weight == "pt2":
                    weight = particle.pt_abs()**2.
                if self.weight == "ptn":
                    weight = particle.pt_abs()**self.n
                if self.weight == "rapidity":
                    weight = particle.momentum_rapidity_Y()
                if self.weight == "pseudorapidity":
                    weight = particle.pseudorapidity()
                particle_weights.append(weight)
            event_weights.append(particle_weights)
        return event_weights
    
    def __compute_flow_vectors(self, particle_data, event_weights):
        # Q vector whole event
        Q_vector = []
        relevant_weights = []
        relevant_particles = []
        for event in range(len(particle_data)):
            Q_vector_val = 0. + 0.j
            relevant_weights_event = []
            relevant_particles_event = []
            for particle in range(len(particle_data[event])):
                if particle_data[event][particle].pt_abs() > 0.3:
                    Q_vector_val += event_weights[event][particle] * np.exp(1.0j * float(self.n) * particle_data[event][particle].phi())
                    relevant_weights_event.append(event_weights[event][particle])
                    relevant_particles_event.append(particle_data[event][particle])
            Q_vector.append(Q_vector_val)
            relevant_weights.extend([relevant_weights_event])
            relevant_particles.extend([relevant_particles_event])

        # Q vector sub-event A
        Q_vector_A = []
        relevant_weights_A = []
        for event in range(len(particle_data)):
            Q_vector_A_val = 0. + 0.j
            relevant_weights_A_event = []
            for particle in range(len(particle_data[event])):
                if particle_data[event][particle].pt_abs() > 0.3 and particle_data[event][particle].pseudorapidity() > +0.1:
                    Q_vector_A_val += event_weights[event][particle] * np.exp(1.0j * float(self.n) * particle_data[event][particle].phi())
                    relevant_weights_A_event.append(event_weights[event][particle])
            Q_vector_A.append(Q_vector_A_val)
            relevant_weights_A.extend([relevant_weights_A_event])

        # Q vector sub-event B
        Q_vector_B = []
        relevant_weights_B = []
        for event in range(len(particle_data)):
            Q_vector_B_val = 0. + 0.j
            relevant_weights_B_event = []
            for particle in range(len(particle_data[event])):
                if particle_data[event][particle].pt_abs() > 0.3 and particle_data[event][particle].pseudorapidity() < -0.1:
                    Q_vector_B_val += event_weights[event][particle] * np.exp(1.0j * float(self.n) * particle_data[event][particle].phi())
                    relevant_weights_B_event.append(event_weights[event][particle])
            Q_vector_B.append(Q_vector_B_val)
            relevant_weights_B.extend([relevant_weights_B_event])

        #sum of weights
        sum_weights = []
        for event in range(len(relevant_weights)):
            weight_val = 0.
            for weight in range(len(relevant_weights[event])):
                weight_val += relevant_weights[event][weight]**2.
            sum_weights.append(weight_val)

        sum_weights_A = []
        for event in range(len(relevant_weights_A)):
            weight_val = 0.
            for weight in range(len(relevant_weights_A[event])):
                weight_val += relevant_weights_A[event][weight]**2.
            sum_weights_A.append(weight_val)

        sum_weights_B = []
        for event in range(len(relevant_weights_B)):
            weight_val = 0.
            for weight in range(len(relevant_weights_B[event])):
                weight_val += relevant_weights_B[event][weight]**2.
            sum_weights_B.append(weight_val)

        for event in range(len(particle_data)):
            # avoid division by 0, if there is no weight, there is also no particle and no flow for an event
            if sum_weights_A[event] == 0.0:
                Q_vector_A[event] = 0.0
            else:
                Q_vector_A[event] /= np.sqrt(sum_weights_A[event])

            if sum_weights_B[event] == 0.0:
                Q_vector_B[event] = 0.0
            else:
                Q_vector_B[event] /= np.sqrt(sum_weights_B[event])

        # compute event plane angles of sub-events
        Psi_A = []
        Psi_B = []
        for event in range(len(particle_data)):
            Psi_A.append((1./float(self.n)) * np.arctan2(Q_vector_A[event].imag,Q_vector_A[event].real))
            Psi_B.append((1./float(self.n)) * np.arctan2(Q_vector_B[event].imag,Q_vector_B[event].real))

        return Q_vector, Q_vector_A, Q_vector_B, Psi_A, Psi_B, sum_weights, sum_weights_A, sum_weights_B, relevant_weights, relevant_particles

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

    def __integrated_flow_event_plane(self,particle_data):
        event_weights = self.__compute_particle_weights(particle_data)

        Q_vector, Q_vector_A, Q_vector_B, Psi_A, Psi_B, sum_weights, sum_weights_A, sum_weights_B, relevant_weights, relevant_particles = self.__compute_flow_vectors(particle_data,event_weights)

        u_vectors = [] # [event][particle]
        for event in range(len(relevant_particles)):
            u_vector_event = []
            for particle in range(len(relevant_particles[event])):
                u_vector_event.append(np.exp(1.0j*float(self.n)*relevant_particles[event][particle].phi()))
            u_vectors.extend([u_vector_event])

        sum_weights_u = [] # [event]
        for event in range(len(relevant_weights)):
            weight_val_u = 0.
            for particle in range(len(relevant_weights[event])):
                weight_val_u += relevant_weights[event][particle]**2.
            sum_weights_u.append(weight_val_u)

        # calculate resolution factor
        event_counter = 0
        total_radicant = 0.0
        for i in range(len(Psi_A)):
            event_counter += 1
            total_radicant += abs(np.cos(float(self.n)*(Psi_A[i] - Psi_B[i])))
        resolution = np.sqrt(total_radicant / event_counter)
        if resolution < 0.5: #correction in case of low resolution
            resolution *= np.sqrt(2.0)

        flow_values = []
        for event in range(len(relevant_particles)):
            flow_values_event = []
            for particle in range(len(relevant_particles[event])):
                weight_particle = np.abs(relevant_weights[event][particle])
                Q_vector_particle = Q_vector[event]
                Q_vector_particle -= weight_particle*u_vectors[event][particle] # avoid autocorrelation
                Q_vector_particle /= np.sqrt(sum_weights[event] - weight_particle**2.)

                u_vector = u_vectors[event][particle]
                u_vector *= weight_particle
                u_vector /= np.sqrt(sum_weights_u)

                Psi_n = (1./float(self.n)) * np.arctan2(Q_vector_particle.imag, Q_vector_particle.real)
                numerator_of_particle = np.cos(float(self.n)* (relevant_particles[event][particle].phi() - Psi_n))

                flow_of_particle = numerator_of_particle / resolution
                flow_values_event.append(flow_of_particle)
            flow_values.extend([flow_values_event])

        # compute the integrated flow
        number_of_particles = 0
        flowvalue = 0.0
        flowvalue_squared = 0.0
        for event in range(len(relevant_particles)):
            for particle in range(len(relevant_particles[event])):
                number_of_particles += 1
                flowvalue += flow_values[event][particle]
                flowvalue_squared += flow_values[event][particle]**2.
        print("flowvalue = ",flowvalue)
        print("flowvalue_squared = ",flowvalue_squared)
        
        vn_integrated = 0.0
        sigma = 0.0
        if number_of_particles == 0:
            vn_integrated = 0.0
            sigma = 0.0
        else:
            vn_integrated = flowvalue / number_of_particles
            vn_squared = flowvalue_squared / number_of_particles
            std_deviation = np.sqrt(vn_integrated**2. - vn_squared)
            sigma = std_deviation / np.sqrt(number_of_particles)

        self.integrated_flow_ = vn_integrated
        self.integrated_flow_err_ = sigma
    
    def __integrated_flow_scalar_product(self):
        return 0
    
    def __integrated_flow_Q_cumulants(self):
        return 0
    
    def __integrated_flow_LeeYang_zeros(self):
        return 0
    
    def __integrated_flow_PCA(self):
        return 0

    def integrated_flow(self,particle_data):
        """
        Compute the integrated anisotropic flow.

        Return
        ------
        integrated_flow_: complex / real
            Value of the integrated flow
        """
        if self.flow_type == "ReactionPlane":
            self.__integrated_flow_reaction_plane(particle_data)
        elif self.flow_type == "EventPlane":
            self.__integrated_flow_event_plane(particle_data)
        return self.integrated_flow_, self.integrated_flow_err_

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


'''
from OscarClass import Oscar
data = Oscar("/home/hendrik/Git/particle-analysis/src/particle-analysis/particle_lists_nevents100_snn200_b3.oscar",events=(0,10)).pseudorapidity_cut(0.5).particle_objects_list()

a = Flow("ReactionPlane",2)
print(a.integrated_flow(data))

#b = Flow("ReactionPlane",2)
#print(b.differential_flow(data,[0.,0.5,1.0,1.5,2.,2.5,3.],"pt"))

#c = Flow("ReactionPlane",3)
#print(c.integrated_flow(data))

#d = Flow("ReactionPlane",3)
#print(d.differential_flow(data,[0.,0.5,1.0,1.5,2.,2.5,3.],"pt"))

#e = Flow("ReactionPlane",4)
#print(e.integrated_flow(data))

#f = Flow("ReactionPlane",4)
#print(f.differential_flow(data,[0.,0.5,1.0,1.5,2.,2.5,3.],"pt"))

g = Flow("EventPlane",n=2,weight="pt2")
print(g.integrated_flow(data))

#h = Flow("EventPlane",n=3,weight="pt")
#print(h.integrated_flow(data))

#i = Flow("EventPlane",n=4,weight="pt")
#print(i.integrated_flow(data))

'''