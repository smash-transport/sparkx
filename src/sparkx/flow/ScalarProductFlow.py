from FlowInterface import FlowInterface
import numpy as np

class ScalarProductFlow(FlowInterface):
    def __init__(self,n=2,weight="pt2",pseudorapidity_gap=0.):

        if not isinstance(n, int):
            raise TypeError('n has to be int')
        elif n <= 0:
            raise ValueError('n-th harmonic with value n<=0 can not be computed')
        else:
            self.n_ = n

        if not isinstance(weight, str):
            raise TypeError('weight has to be a string')
        elif weight not in ["pt", "pt2", "ptn", "rapidity", "pseudorapidity"]:
            raise ValueError("Invalid weight given, choose one of the following: 'pt', 'pt2', 'ptn', 'rapidity', 'pseudorapidity'")
        else:
            self.weight_ = weight

        if not isinstance(pseudorapidity_gap, (int, float)):
            raise TypeError('n has to be int')
        elif pseudorapidity_gap < 0:
            raise ValueError('pseudorapidity value with gap < 0 can not be computed')
        else:
            self.pseudorapidity_gap_ = pseudorapidity_gap

    def __compute_particle_weights(self, particle_data):
        event_weights = []
        for event in range(len(particle_data)):
            particle_weights = []
            for particle in particle_data[event]:
                weight = 0.
                if self.weight_ == "pt":
                    weight = particle.pt_abs()
                elif self.weight_ == "pt2":
                    weight = particle.pt_abs()**2.
                elif self.weight_ == "ptn":
                    weight = particle.pt_abs()**self.n_
                elif self.weight_ == "rapidity":
                    weight = particle.momentum_rapidity_Y()
                elif self.weight_ == "pseudorapidity":
                    weight = particle.pseudorapidity()
                particle_weights.append(weight)
            event_weights.append(particle_weights)
        return event_weights

    def __compute_flow_vectors(self, particle_data, weights):
        # Q vector whole event
        Q_vector = []
        for event in range(len(particle_data)):
            Q_vector_val = 0. + 0.j
            for particle in range(len(particle_data[event])):
                Q_vector_val += weights[event][particle] * np.exp(1.0j * float(self.n_) * particle_data[event][particle].phi())
            Q_vector.append(Q_vector_val)

        return Q_vector

    def __sum_weights(self, weights):
        sum_weights = []
        for event in weights:
            weight_val = np.sum(np.square(event))
            sum_weights.append(weight_val)

        return sum_weights

    def __compute_event_angles_sub_events(self, particle_data, weights):
        # Q vector sub-event A
        Q_vector_A = []
        relevant_weights_A = []
        for event in range(len(particle_data)):
            Q_vector_A_val = 0. + 0.j
            relevant_weights_A_event = []
            for particle in range(len(particle_data[event])):
                if particle_data[event][particle].pseudorapidity() >= +self.pseudorapidity_gap_:
                    Q_vector_A_val += weights[event][particle] * np.exp(1.0j * float(self.n_) * particle_data[event][particle].phi())
                    relevant_weights_A_event.append(weights[event][particle])
            Q_vector_A.append(Q_vector_A_val)
            relevant_weights_A.extend([relevant_weights_A_event])

        # Q vector sub-event B
        Q_vector_B = []
        relevant_weights_B = []
        for event in range(len(particle_data)):
            Q_vector_B_val = 0. + 0.j
            relevant_weights_B_event = []
            for particle in range(len(particle_data[event])):
                if particle_data[event][particle].pseudorapidity() < -self.pseudorapidity_gap_:
                    Q_vector_B_val += weights[event][particle] * np.exp(1.0j * float(self.n_) * particle_data[event][particle].phi())
                    relevant_weights_B_event.append(weights[event][particle])
            Q_vector_B.append(Q_vector_B_val)
            relevant_weights_B.extend([relevant_weights_B_event])

        sum_weights_A = self.__sum_weights(relevant_weights_A)
        sum_weights_B = self.__sum_weights(relevant_weights_B)

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

        return Q_vector_A, Q_vector_B

    def __compute_u_vectors(self, particle_data):
        u_vectors = [] # [event][particle]
        for event in particle_data:
            u_vector_event = []
            for particle in event:
                u_vector_event.append(np.exp(1.0j*float(self.n_)*particle.phi()))
            u_vectors.extend([u_vector_event])

        return u_vectors

    def __compute_event_plane_resolution(self, Q_vector_A, Q_vector_B):
        # implements Eq.15 from arXiv:0809.2949
        QnSquared = np.asarray([(np.conjugate(Q_vector_A[event]) * Q_vector_B[event]).real for event in range(len(Q_vector_A))])
        QnSquaredSum = np.sum(QnSquared)
        return 2. * np.sqrt(QnSquaredSum / len(Q_vector_A))

    def __compute_flow_particles(self, particle_data, weights, Q_vector, u_vectors, sum_weights_u, resolution, self_corr):
        flow_values = []
        for event in range(len(particle_data)):
            flow_values_event = []
            for particle in range(len(particle_data[event])):
                weight_particle = np.abs(weights[event][particle])
                Q_vector_particle = Q_vector[event]
                if (self_corr):
                    Q_vector_particle -= weight_particle*u_vectors[event][particle] # avoid autocorrelation
                #Q_vector_particle /= np.sqrt(sum_weights_u[event] - weight_particle**2.)

                u_vector = u_vectors[event][particle]
                u_vector *= weight_particle
                u_vector /= np.sqrt(sum_weights_u[event])
                u_vector /= abs(u_vector) # correlate with unit vector of POI

                vn_obs = (np.conjugate(u_vector) * Q_vector_particle).real
                flow_of_particle = vn_obs / resolution
                flow_values_event.append(flow_of_particle)
            flow_values.extend([flow_values_event])

        return flow_values

    def __calculate_reference(self, particle_data_event_plane):
        event_weights_event_plane = self.__compute_particle_weights(particle_data_event_plane)
        Q_vector_A, Q_vector_B = self.__compute_event_angles_sub_events(particle_data_event_plane,event_weights_event_plane)
        resolution = self.__compute_event_plane_resolution(Q_vector_A,Q_vector_B)
        Q_vector = self.__compute_flow_vectors(particle_data_event_plane,event_weights_event_plane)

        return resolution, Q_vector

    def __calculate_particle_flow(self, particle_data, resolution, Q_vector, self_corr):
        event_weights = self.__compute_particle_weights(particle_data)
        u_vectors = self.__compute_u_vectors(particle_data)
        sum_weights_u = self.__sum_weights(event_weights)

        return self.__compute_flow_particles(particle_data,event_weights,Q_vector,u_vectors,sum_weights_u,resolution, self_corr)

    def __calculate_flow_event_average(self, particle_data, flow_particle_list):
        # compute the integrated flow
        number_of_particles = 0
        flowvalue = 0.0
        flowvalue_squared = 0.0
        for event in range(len(flow_particle_list)):
            for particle in range(len(flow_particle_list[event])):
                weight = 1. if particle_data[event][particle].weight is None else particle_data[event][particle].weight
                number_of_particles += weight
                flowvalue += flow_particle_list[event][particle]*weight
                flowvalue_squared += flow_particle_list[event][particle]**2.*weight**2.

        vn_integrated = 0.0
        sigma = 0.0
        if number_of_particles == 0:
            vn_integrated = 0.0
            sigma = 0.0
        else:
            vn_integrated = flowvalue / number_of_particles
            vn_squared = flowvalue_squared / number_of_particles**2.
            std_deviation = np.sqrt(vn_integrated**2. - vn_squared)
            sigma = std_deviation / np.sqrt(number_of_particles)

        return vn_integrated, sigma

    def integrated_flow(self,particle_data,particle_data_event_plane, self_corr=True):
        resolution, Q_vector = self.__calculate_reference(particle_data_event_plane)
        return self.__calculate_flow_event_average(particle_data, self.__calculate_particle_flow(particle_data, resolution, Q_vector, self_corr))


    def differential_flow(self, particle_data, bins, flow_as_function_of, particle_data_event_plane, self_corr=True):

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
                    if val >= bins[bin] and val < bins[bin+1]:
                        particles_event.append(particle)
                events_bin.extend([particles_event])
            particles_bin.extend([events_bin])

        resolution, Q_vector = self.__calculate_reference(particle_data_event_plane)

        flow_bin = []
        for bin in range(len(bins)-1):
            flow_bin.append(self.__calculate_flow_event_average(particle_data, self.__calculate_particle_flow(particles_bin[bin],resolution,Q_vector,self_corr)))

        return flow_bin
import sys
sys.path.append("/home/niklas/Desktop/sparkx/src/sparkx")
from Jetscape import Jetscape
oscar = Jetscape("/home/niklas/Downloads/LYZ_testdata.dat")
liste = oscar.particle_objects_list()

test = ScalarProductFlow()
print(test.integrated_flow(liste, liste, True))
print(test.differential_flow(liste, [0.,0.1,0.2,0.3,0.5,1,1.5], "pt", liste, True))
