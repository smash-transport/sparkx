#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
from sparkx.flow import FlowInterface
import numpy as np
from scipy import special
import scipy.optimize as optimize
import warnings

class EventPlaneFlow(FlowInterface.FlowInterface):
    """
    This class implements a event plane flow analysis algorithm
    `Poskanzer, Arthur M., and Sergei A. Voloshin. "Methods for analyzing anisotropic flow in relativistic nuclear collisions." Physical Review C 58.3 (1998): 1671. <https://link.aps.org/pdf/10.1103/PhysRevC.58.1671?casa_token=rinfpV-zLrYAAAAA:Mv0gWZZBE8I9zpBPLog8myr7uKzLHO1TelYz0BN7boUORUWnP3Fyybuc0OKNKK5YH5ceNy5qVNO8rws>`__.

    For this method, the flow is calculated by calculating the nth harmonic of 
    the angle the event plane :math:`Q`, relative to the event plane angle 
    determined from the flow itself.
    Since finite multiplicity limits the estimation of the angle of the reaction 
    plane, the flow is corrected for the event plane resolution for each harmonic.

    In summary, this class calculates the following:

    .. math::

        v_n = \\frac{v_{n}^{\\mathrm{obs}}}{R_n}

    with

    .. math::

        v_{n}^{\\mathrm{obs}} = \\langle \\cos(n (\\phi_i -\\Psi_n)) \\rangle


    where we average over all particles of all events. :math:`R_n` is the 
    resolution, which we calculate in an iterative method according to the 
    mentioned reference. This is done by using two sub-events, which we assume 
    to be of same size and which are determined by splitting the event into 
    subsets of positive and negative pseudorapidity.


    Parameters
    ----------
    n : int, optional
        The value of the harmonic. Default is 2.
    weight : str, optional
        The weight used for calculating the flow. Default is "pt2".
    pseudorapidity_gap : float, optional
        The pseudorapidity gap used for dividing the particles into sub-events.
        Default is 0.0.

    Methods
    -------
    integrated_flow:
        Computes the integrated flow.
    differential_flow:
        Computes the differential flow.

    Examples
    --------

    A demonstration how to calculate flow according to the event plane of a 
    separate particle list.
    The same particle list can also be used to determine the event plane and 
    the flow.

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Jetscape import Jetscape
        >>> from sparkx.flow.EventPlaneFlow import EventPlaneFlow
        >>>
        >>> JETSCAPE_FILE_PATH_FLOW = [Jetscape_directory]/particle_lists_flow.dat
        >>> JETSCAPE_FILE_PATH_EVENT_PLANE = [Jetscape_directory]/particle_lists_ep.dat
        >>>
        >>> # Jetscape object containing the particles on which we want to calculate flow
        >>> jetscape_flow = Jetscape(JETSCAPE_FILE_PATH_FLOW).particle_objects_list()
        >>>
        >>> # Jetscape object containing the particles which determine the event plane
        >>> jetscape_event = Jetscape(JETSCAPE_FILE_EVENT_PLANE).particle_objects_list()
        >>>
        >>> # Create flow objects for v2, weighted with pT**2 and v3 weighted with pT**2
        >>> flow2 = EventPlaneFlow(n=2, weight="pt2",pseudorapidity_gap=0.1)
        >>> flow3 = EventPlaneFlow(n=3, weight="pt2",pseudorapidity_gap=0.1)
        >>>
        >>> # Calculate the integrated flow with error
        >>> v2, v2_error = flow2.integrated_flow(jetscape_flow,jetscape_event)
        >>> v3, v3_error = flow3.integrated_flow(jetscape_flow,jetscape_event)


    """
    def __init__(self,n=2,weight="pt2",pseudorapidity_gap=0.):
        """
        Initialize the ScalarProductFlow object.

        Parameters
        ----------
        n : int, optional
            The value of the harmonic. Default is 2.
        weight : str, optional
            The weight used for calculating the flow. Default is "pt2".
        pseudorapidity_gap : float, optional
            The pseudorapidity gap used for dividing the particles into sub-events.
            Default is 0.0.
        """

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

        # compute event plane angles of sub-events
        Psi_A = [(1./float(self.n_)) * np.arctan2(Q_vector_A[event].imag,Q_vector_A[event].real) for event in range(len(particle_data))]
        Psi_B = [(1./float(self.n_)) * np.arctan2(Q_vector_B[event].imag,Q_vector_B[event].real) for event in range(len(particle_data))]

        return Psi_A, Psi_B

    def __compute_u_vectors(self, particle_data):
        u_vectors = [] # [event][particle]
        for event in particle_data:
            u_vector_event = []
            for particle in event:
                u_vector_event.append(np.exp(1.0j*float(self.n_)*particle.phi()))
            u_vectors.extend([u_vector_event])

        return u_vectors

    def __compute_event_plane_resolution(self, Psi_A, Psi_B):
        RnSquared = np.asarray([np.cos(self.n_ * (Psi_A[event] - Psi_B[event])) for event in range(len(Psi_A))])
        RnSquaredSum = np.sum(RnSquared)
        Rn = np.sqrt(RnSquaredSum / len(Psi_A))

        # calculate x for given Rn, then update R with x value:
        # x_new = sqrt(2) * x
        # alternative: if Rn < 0.5: R = Rn * sqrt(2), we don't do that
        # implements: arXiv:0809.2949
        def resolution(x):
            R = (np.sqrt(np.pi) / 2.) * x * np.exp(-0.5 * x * x) * (special.i0(0.5 * x * x) + special.i1(0.5 * x * x))
            return R

        def f1(x,Rn):
            return resolution(x)-Rn

        def f1_wrapper(x):
            return f1(x,Rn)

        try:
            xi = optimize.root_scalar(f1_wrapper, bracket=[0, 20], method='brentq').root
        except:
            warnings.warn("Could not find solution of resolution equation. Use approximation instead.")
            return Rn

        xi_new = np.sqrt(2) * xi
        return resolution(xi_new)

    def __compute_flow_particles(self, particle_data, weights, Q_vector, u_vectors, resolution, self_corr):
        flow_values = []
        for event in range(len(particle_data)):
            flow_values_event = []
            for particle in range(len(particle_data[event])):
                weight_particle = np.abs(weights[event][particle])
                Q_vector_particle = Q_vector[event]
                if (self_corr):
                    Q_vector_particle -= weight_particle*u_vectors[event][particle] # avoid autocorrelation

                Psi_n = (1./float(self.n_)) * np.arctan2(Q_vector_particle.imag, Q_vector_particle.real)
                vn_obs = np.cos(float(self.n_)* (particle_data[event][particle].phi() - Psi_n))

                flow_of_particle = vn_obs / resolution
                flow_values_event.append(flow_of_particle)
            flow_values.extend([flow_values_event])

        return flow_values

    def __calculate_reference(self, particle_data_event_plane):
        event_weights_event_plane = self.__compute_particle_weights(particle_data_event_plane)
        Psi_A, Psi_B = self.__compute_event_angles_sub_events(particle_data_event_plane,event_weights_event_plane)
        resolution = self.__compute_event_plane_resolution(Psi_A,Psi_B)
        Q_vector = self.__compute_flow_vectors(particle_data_event_plane,event_weights_event_plane)

        return resolution, Q_vector

    def __calculate_particle_flow(self, particle_data, resolution, Q_vector, self_corr):
        event_weights = self.__compute_particle_weights(particle_data)
        u_vectors = self.__compute_u_vectors(particle_data)
        sum_weights_u = self.__sum_weights(event_weights)

        return self.__compute_flow_particles(particle_data,event_weights,Q_vector,u_vectors,resolution, self_corr)

    def __calculate_flow_event_average(self, particle_data, flow_particle_list):
        # compute the integrated flow
        number_of_particles = 0
        flowvalue = 0.0
        flowvalue_squared = 0.0
        for event in range(len(flow_particle_list)):
            for particle in range(len(flow_particle_list[event])):
                weight = 1. if np.isnan(particle_data[event][particle].weight) else particle_data[event][particle].weight
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
        """
        Compute the integrated flow.

        Parameters
        ----------
        particle_data : list
            List of particle data of which the flow is calculated.
        particle_data_event_plane : list
            List of particle data for the event plane calculation.
        self_corr : bool, optional
            Whether to consider self-correlation in the flow calculation.
            Default is True.

        Returns
        -------
        tuple
            A tuple containing the integrated flow value and the corresponding 
            uncertainty.
        """
        if not isinstance(self_corr, bool):
            raise TypeError('self_corr has to be bool')
        resolution, Q_vector = self.__calculate_reference(particle_data_event_plane)
        return self.__calculate_flow_event_average(particle_data, self.__calculate_particle_flow(particle_data, resolution, Q_vector, self_corr))

    def differential_flow(self, particle_data, bins, flow_as_function_of, particle_data_event_plane, self_corr=True):
        """
        Compute the differential flow.

        Parameters
        ----------
        particle_data : list
            List of particle data of which the flow is calculated.
        bins : list or np.ndarray
            Bins used for the differential flow calculation.
        flow_as_function_of : str
            Variable on which the flow is calculated 
            ("pt", "rapidity", or "pseudorapidity").
        particle_data_event_plane : list
            List of particle data for the event plane calculation.
        self_corr : bool, optional
            Whether to consider self-correlation in the flow calculation.
            Default is True.

        Returns
        -------
        list
            A list of tuples containing the flow values and uncertainties for 
            each bin.
        """
        if not isinstance(self_corr, bool):
            raise TypeError('self_corr has to be bool')
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
