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
import random as rd
import warnings

rd.seed(42)

class LeeYangZeroFlow(FlowInterface.FlowInterface):
    """
    Compute integrated and differential anisotropic flow using the Lee-Yang 
    zero method from

    - [1] `Phys. Lett. B 580 (2004) 157  [nucl-th/0307018] <https://inspirehep.net/literature/622649>`__
    - [2] `Nucl. Phys. A 727 (2003) 373  [nucl-th/0310016] <https://inspirehep.net/literature/629783>`__
    - [3] `J. Phys. G: Nucl. Part. Phys. 30 (2004) S1213  [nucl-th/0402053] <https://inspirehep.net/literature/644572>`__

    For the computation of the anisotropic flow it is important to have an 
    estimate on how large the anisotropic flow will be. To set up the correct
    range of radial values along which the generating function is calculated.

    For a practical guide of the implementation we refer to Ref. [3], where all
    relevant equations are given.

    Notes
    -----
    If a result contains `np.nan` or `np.inf`, the corresponding value is set to `None`.
    
    A few remarks from Ref. [2] about the applicability of the method and the resolution parameter :math:`\\chi`:
        - :math:`\\chi > 1`:  The statistical error on the flow is not significantly larger than with the standard method. At the same time systematic errors due to nonflow effects are much smaller. The present method should be used, and statistics will not be a problem.
        - :math:`0.5 < \\chi < 1`: The method is applicable, but the weights should be optimized to increase :math:`\\chi`. This is not possible with the current implementation of the flow analysis method.
        - :math:`\\chi < 0.5`: Too large statistical errors, the present method should not be used. Using more events will not help much. Use the cumulant methods instead, which are still applicable if the number of events is large enough.

    Parameters
    ----------
    vmin : float
        Minimum flow value.
    vmax : float
        Maximum flow value.
    vstep : float
        Step size for the flow values.
    n : int, optional
        The harmonic order. Default is 2.

    Methods
    -------
    integrated_flow:
        Computes the integrated flow.

    differential_flow:
        Computes the differential flow.

    Examples
    --------
    A demonstration of how to use the LeeYangZerosFlow class to calculate flow.

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.flow.LeeYangZerosFlow import LeeYangZerosFlow
        >>>
        >>> JETSCAPE_FILE_PATH = [Jetscape_directory]/particle_lists_flow.dat
        >>> # Jetscape object containing the particles on which we want to calculate flow
        >>> particle_data = Jetscape(JETSCAPE_FILE_PATH).particle_objects_list()

        >>> # Create a LeeYangZerosFlow object
        >>> flow_instance = LeeYangZerosFlow(vmin=0.01, vmax=0.10, vstep=0.001, n=2)
        >>>
        >>> # Calculate the integrated flow with error and resolution parameter
        >>> result = flow_instance.integrated_flow(particle_data)

    """
    def __init__(self,vmin,vmax,vstep,n=2):

        self.j01_ = 2.4048256
        self.J1rootJ0_ = 0.5191147 # J1(j01)

        if vmin > vmax:
            raise ValueError("'vmin' is larger than 'vmax'")
        if (vmax-vmin) < vstep:
            raise ValueError("'vstep' is larger than the difference between minimum and maximum flow")
        
        if not isinstance(n, int):
            raise TypeError('n has to be int')
        elif n <= 0:
            raise ValueError('n-th harmonic with value n<=0 can not be computed')
        else:
            self.n_ = n

        # define the r_space_ such that one achieves the wanted precision vstep
        number_interpolation_points_r = np.ceil((vmax-vmin)/vstep)
        self.r_space_ = np.array([self.j01_ / (vmax - vstep * r_i) for r_i in range(int(number_interpolation_points_r))])
        self.theta_space_ = np.linspace(0.,np.pi/self.n_,5) # equally spaced between 0 and pi/n

        self.computed_integrated_flow_ = False
        self.chi_ = None
        self.Vntheta_ = None
        self.r0theta_ = None
        self.rand_reaction_planes_ = None
        self.denominator_event_avg_diff_ = None
        self.numerator_event_avg_diff_ = None


    def __g_theta(self, n, r, theta, weight_j, phi_j):
        """
        Calculate the generating function g^{\\theta}(\\mathrm{i}r) defined by 
        Eq. (3) in Ref. [3].

        Parameters
        ----------
        n : int
            The harmonic order.
        r : float
            The radius.
        theta : float
            Reaction plane angle.
        weight_j : list or np.ndarray
            List of weights corresponding to each term in the product.
        phi_j : list or np.ndarray
            List of phase angles corresponding to each term in the product.

        Returns
        -------
        complex
            The computed value of the generating function.

        Raises
        ------
        ValueError
            If input formats for weight_j or phi_j are incorrect, or if the 
            lengths of weight_j and phi_j differ.
        """
        if not isinstance(weight_j, (list, np.ndarray)) or not isinstance(phi_j, (list, np.ndarray)):
            raise ValueError("Not the correct input format for g_theta")
        if len(weight_j) != len(phi_j):
            raise ValueError("weight_j and phi_j do not have the same length")

        g_theta = 1.0 + 0.0j
        for j in range(len(weight_j)):
            g_theta *= (1.0 + 1.0j * r * weight_j[j] * np.cos(n * (phi_j[j] - theta)))
        return g_theta

    def __Q_x(self, n, weight_j, phi_j):
        """
        Calculate the quantity Q_x defined by Eq. (4) in Ref. [3].

        Parameters
        ----------
        n : int
            The harmonic order.
        weight_j : list or np.ndarray
            List of weights corresponding to each term in the summation.
        phi_j : list or np.ndarray
            List of azimuthal angles corresponding to each term in the summation.

        Returns
        -------
        float
            The computed value of the quantity Q_x.

        Raises
        ------
        ValueError
            If input formats for weight_j or phi_j are incorrect, or if the 
            lengths of weight_j and phi_j differ.
        """
        if not isinstance(weight_j,(list,np.ndarray)) or not isinstance(phi_j,(list,np.ndarray)):
            raise ValueError('Not the correct input format for g_theta')
        if len(weight_j) != len(phi_j):
            raise ValueError('weight_j and phi_j do not have the same length')
        Q_x = 0.
        for j in range(len(weight_j)):
            Q_x += weight_j[j] * np.cos(n * phi_j[j])
        return Q_x

    def __Q_y(self,n,weight_j,phi_j):
        """
        Calculate the quantity Q_y defined by Eq. (4) in Ref. [3].

        Parameters
        ----------
        n : int
            The harmonic order.
        weight_j : list or np.ndarray
            List of weights corresponding to each term in the summation.
        phi_j : list or np.ndarray
            List of azimuthal angles corresponding to each term in the summation.

        Returns
        -------
        float
            The computed value of the quantity Q_y.

        Raises
        ------
        ValueError
            If input formats for weight_j or phi_j are incorrect, or if the 
            lengths of weight_j and phi_j differ.
        """
        if not isinstance(weight_j,(list,np.ndarray)) or not isinstance(phi_j,(list,np.ndarray)):
            raise ValueError('Not the correct input format for g_theta')
        if len(weight_j) != len(phi_j):
            raise ValueError('weight_j and phi_j do not have the same length')
        Q_y = 0.
        for j in range(len(weight_j)):
            Q_y += weight_j[j] * np.sin(n * phi_j[j])
        return Q_y

    def __sigma(self,QxSqPQySq,Qx,Qy,VnInfty):
        """
        Calculate the value of :math:`\\sigma` based on Eq. (7) in Ref. [3].

        Parameters
        ----------
        QxSqPQySq : float
            The value of :math:`\\langle Q_x^2 + Q_y^2\\rangle`.
        Qx : float
            The value of :math:`Q_x`.
        Qy : float
            The value of :math:`Q_y`..
        VnInfty : float
            The value of :math:`V_n\{\\infty\}`.

        Returns
        -------
        float
            The computed value of sigma based on Eq. (7) in Ref. [3].
        """
        # Eq. 7 in Ref. [3]
        return np.sqrt(QxSqPQySq - Qx**2. - Qy**2. - VnInfty**2.)

    def __chi(self,VnInfty,sigma):
        """
        Calculate the resolution parameter :math:`\\chi` based on the 
        given parameters.

        Parameters
        ----------
        VnInfty : float
            The value of :math:`V_n\{\\infty\}`.
        sigma : float
            The value of :math:`\\sigma`.

        Returns
        -------
        float
            The computed value of chi.
        """
        return VnInfty / sigma

    def __relative_Vn_fluctuation(self,NEvents,chi):
        """
        Calculate the relative flow fluctuation based on the given parameters.
        This is based on Eq. (8) in Ref. [3].

        Parameters
        ----------
        NEvents : int
            The number of events.
        chi : float
            The value of :math:`\\chi`.

        Returns
        -------
        float
            The computed relative flow fluctuation based on Eq. (8) in Ref. [3].
        """
        return (1./(2.*NEvents*self.j01_**2.*self.J1rootJ0_**2.)) * (np.exp(self.j01_**2. / (2.*chi*chi)) + np.exp(-self.j01_**2. / (2.*chi**2.)) * (-0.2375362))

    def __sample_random_reaction_planes(self, events):
        """
        Sample random reaction planes for a specified number of events.

        Parameters
        ----------
        events : int
            The number of events for which random reaction planes are sampled.

        Returns
        -------
        None
        """
        self.rand_reaction_planes_ = [rd.uniform(0., 2.*np.pi) for _ in range(events)]

    def integrated_flow(self, particle_data):
        """
        Computes the integrated flow.

        Parameters
        ----------
        particle_data : list
            List of particle data for multiple events.

        Returns
        -------
        list
            A list containing the following values:

            - vn_inf (float): Integrated flow magnitude.
            - vn_inf_error (float): Standard error on the integrated flow magnitude.
            - chi_value (float): Resolution parameter :math:`\\chi`.

        If `vn_inf` is `np.nan` or `np.inf`, the method returns `[None, None, None]`.
        """
        number_events = len(particle_data)
        mean_multiplicity = 0

        AvgQx = 0.
        AvgQy = 0.
        AvgQxSqPQySq = 0.

        G = np.zeros((len(self.theta_space_),len(self.r_space_)),dtype=np.complex_)
        self.__sample_random_reaction_planes(len(particle_data))

        for event in range(number_events):
            event_multiplicity = len(particle_data[event])
            mean_multiplicity +=event_multiplicity

            phi_j = []
            weight_j = []
            for particle in particle_data[event]:
                phi_j.append(particle.phi())
                weight_j.append(1./event_multiplicity)
            
            # randomize the event plane
            phi_j = [phi+self.rand_reaction_planes_[event] for phi in phi_j]

            g = np.zeros((len(self.theta_space_),len(self.r_space_)),dtype=np.complex_)
            for theta in range(len(self.theta_space_)):
                for r in range(len(self.r_space_)):
                    g[theta][r] = self.__g_theta(self.n_,self.r_space_[r],self.theta_space_[theta],weight_j,phi_j)
                    G[theta][r] += g[theta][r] / number_events

            Qx = self.__Q_x(self.n_,weight_j,phi_j) # event flow vector
            Qy = self.__Q_y(self.n_,weight_j,phi_j) # event flow vector
            AvgQx += Qx # average event flow vector
            AvgQy += Qy # average event flow vector
            AvgQxSqPQySq += (Qx**2. + Qy**2.)

        mean_multiplicity /= number_events
        AvgQx /= number_events
        AvgQy /= number_events
        AvgQxSqPQySq /= number_events

        # compute the minimum for each theta value
        min_r0_theta = []
        for theta, G_theta_values in enumerate(abs(G)):
            r = 0
            while r < len(G_theta_values) - 1 and G_theta_values[r] > G_theta_values[r + 1]:
                r += 1
            min_r0_theta.append(self.r_space_[r])
            if r == len(self.r_space_)-1:
                warnings.warn("The minimum r0 is at the upper boundary of \
                              the selected r range. Please choose a smaller \
                              'vmin'.")
            elif r == 0:
                warnings.warn("The minimum r0 is at the lower boundary of \
                              the selected r range. Please choose a larger \
                              'vmax'.")

        vn_inf_theta = []
        for theta in range(len(self.theta_space_)):
            vn_inf_theta.append(self.j01_ / min_r0_theta[theta])
        
        vn_inf = np.mean(vn_inf_theta)
        sigma_value = self.__sigma(AvgQxSqPQySq,AvgQx,AvgQy,vn_inf)
        chi_value = self.__chi(vn_inf,sigma_value)
        relative_Vn_fluctuation = self.__relative_Vn_fluctuation(number_events,chi_value)

        # set the members for the differential flow computation
        self.computed_integrated_flow_ = True
        self.chi_ = chi_value
        self.Vntheta_ = vn_inf_theta
        self.r0theta_ = min_r0_theta
        
        if np.isnan(vn_inf) or np.isinf(vn_inf):
            return [None, None, None]
        else:
            # factor 1/2 because the statistical error on Vninf is a factor 2 smaller
            return [vn_inf,0.5*np.sqrt(vn_inf*relative_Vn_fluctuation),chi_value]

    def __compute_reference_differential_flow(self,particle_data):
        """
        This function computes the event average in the denominator of Eq. (9) in Ref. [3].
        """
        number_events = len(particle_data)
        
        denominator_event_avg = 0.
        for event in range(number_events):
            event_multiplicity = len(particle_data[event])

            phi_j = []
            weight_j = []
            for particle in particle_data[event]:
                phi_j.append(particle.phi())
                weight_j.append(1./event_multiplicity)
            
            # randomize the event plane
            phi_j = [phi+self.rand_reaction_planes_[event] for phi in phi_j]

            g = np.zeros(len(self.theta_space_),dtype=np.complex_)
            for theta in range(len(self.theta_space_)):
                g[theta] = self.__g_theta(self.n_,self.r0theta_[theta],self.theta_space_[theta],weight_j,phi_j)

                sum_j = 0.
                for p in range(event_multiplicity):
                    numerator = weight_j[p]*np.cos(self.n_*(phi_j[p]-self.theta_space_[theta]))
                    denominator = 1.0 + 1.0j * self.r0theta_[theta] * weight_j[p] * np.cos(self.n_ * (phi_j[p] - self.theta_space_[theta]))
                    sum_j += numerator / denominator

                g[theta] *= sum_j

            denominator_event_avg += g
        denominator_event_avg /= number_events
        self.denominator_event_avg_diff_ = denominator_event_avg

    def __vn_differential_uncertainty(self,number_particles_tot):
        """
        Calculates the differential flow uncertainty.

        Parameters
        ----------
        number_particles_tot : int or float
            Total number of particles involved in the measurement.

        Returns
        -------
        float
            The calculated differential flow uncertainty.
        """
        err_sq = (1./(4.*number_particles_tot**self.J1rootJ0_**2.)) * (np.exp(self.j01_**2. / (2.*self.chi_**2.)) - np.exp(-self.j01_**2. / (2.*self.chi_**2.)) * (-0.2375362))
        return np.sqrt(err_sq)

    def __compute_differential_flow_bin(self,particle_data_bin):
        """
        Computes the differential flow in the given phase space bin with Eq. (9) 
        in Ref. [3].

        Parameters
        ----------
        particle_data_bin : list of lists
            List containing particle data for each event in a specific bin.

        Returns
        -------
        list
            A list containing the computed values of differential flow 
            (vn_differential) and its uncertainty (vn_differential_uncertainty).
        """
        number_events = len(particle_data_bin)
        
        number_particles_tot = 0.
        numerator_particle_avg = 0.
        for event in range(number_events):
            event_multiplicity = len(particle_data_bin[event])
            number_particles_tot += event_multiplicity

            phi_j = []
            weight_j = []
            for particle in particle_data_bin[event]:
                phi_j.append(particle.phi())
                weight_j.append(1./event_multiplicity)
            
            # randomize the event plane
            phi_j = [phi+self.rand_reaction_planes_[event] for phi in phi_j]

            g = np.zeros(len(self.theta_space_),dtype=np.complex_)
            for theta in range(len(self.theta_space_)):
                g[theta] = self.__g_theta(self.n_,self.r0theta_[theta],self.theta_space_[theta],weight_j,phi_j)

                sum_j = 0.
                for p in range(event_multiplicity):
                    numerator = np.cos(self.n_*(phi_j[p]-self.theta_space_[theta]))
                    denominator = 1.0 + 1.0j * self.r0theta_[theta] * weight_j[p] * np.cos(self.n_ * (phi_j[p] - self.theta_space_[theta]))
                    sum_j += numerator / denominator
                
                g[theta] *= sum_j

            numerator_particle_avg += g

        numerator_particle_avg /= number_particles_tot
        self.numerator_event_avg_diff_ = numerator_particle_avg

        vn_theta = []
        for theta in range(len(self.theta_space_)):
            vn_theta.append(self.Vntheta_[theta] * np.real(self.numerator_event_avg_diff_[theta]/self.denominator_event_avg_diff_[theta]))
        
        vn_differential = np.mean(vn_theta)
        # factor 1/2 because the statistical error on Vninf is a factor 2 smaller
        vn_differential_uncertainty = 0.5*self.__vn_differential_uncertainty(number_particles_tot)

        if np.isnan(vn_differential) or np.isinf(vn_differential_uncertainty):
            return [None, None]
        else:
            return [vn_differential,vn_differential_uncertainty]

    def differential_flow(self,particle_data,bins,flow_as_function_of, poi_pdg=None):
        """
        Compute the differential flow.

        Parameters
        ----------
        particle_data : list
            List of particle data for multiple events.
        bins : list or np.ndarray
            Bins used for the differential flow calculation.
        flow_as_function_of : str
            Variable on which the flow is calculated 
            ("pt", "rapidity", or "pseudorapidity").
        poi_pdg : list
            List of PDG id for identified particle differential flow.

        Returns
        -------
        list
            A list containing the differential flow for each bin. 
            Each element in the list corresponds to a bin and contains:

            - vn_inf (float): Differential flow magnitude for the bin.
            - vn_inf_error (float): Error on the differential flow magnitude for the bin.
            
        If a bin has no events, the corresponding element in the result list is 
        set to `None`.

        Notes
        -----
        This method will call the `integrated_flow` method if it was not called
        before and computes the integrated flow for the given `particle_data`.
        Make sure that the same `particle_data` is used. Otherwise this could
        lead to wrong results.
        """
        if not isinstance(bins, (list,np.ndarray)):
            raise TypeError('bins has to be list or np.ndarray')
        if not isinstance(flow_as_function_of, str):
            raise TypeError('flow_as_function_of is not a string')
        if poi_pdg is not None:
            if not isinstance(poi_pdg, (list,np.ndarray)):
                raise TypeError('poi_pdg has to be list or np.ndarray')
            for pdg in poi_pdg:
                if not isinstance(pdg, int):
                    raise TypeError('poi_pdg elements must be integers')
        if flow_as_function_of not in ["pt","rapidity","pseudorapidity"]:
            raise ValueError("flow_as_function_of must be either 'pt', 'rapidity', 'pseudorapidity'")
        
        # differential flow needs the integrated flow as reference
        if self.computed_integrated_flow_ == False:
            integrated_flow = self.integrated_flow(particle_data)

        particle_data_bin = []
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
                    if val >= bins[bin] and val < bins[bin+1] and particle.pdg in poi_pdg:
                        particles_event.append(particle)
                if len(particles_event) > 0:
                    events_bin.extend([particles_event])
            particle_data_bin.extend([events_bin])

        self.__compute_reference_differential_flow(particle_data)

        flow_bins = []
        for bin in range(len(particle_data_bin)):
            if len(particle_data_bin[bin]) > 0:
                flow_bins.append(self.__compute_differential_flow_bin(particle_data_bin[bin]))
            else:
                flow_bins.append(None)
        return flow_bins