#===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================

import numpy as np


class MultiParticlePtCorrelations:
    """
    Compute multi-particle transverse momentum correlations and cumulants up 
    to the 8th order. This class is based on the following paper:

    - [1] `Eur. Phys. J. A 60 (2024) 2, 38 [2312.00492 [nucl-th]] <https://inspirehep.net/literature/2729183>`__

    For the computation of transverse momentum correlations and cumulants, the 
    implementation closely follows the equations and methods described in Ref. [1].


    Parameters
    ----------
    max_order : int
        Maximum order of correlations and cumulants to compute (must be between 1 and 8).

    Methods
    -------
    compute_mean_pt_correlations:
        Computes the mean transverse momentum correlations for each order k
        across all events.
    
    compute_mean_pt_cumulants:
        Computes the mean transverse momentum cumulants for each order k
        from the correlations.

    Examples
    --------
    A demonstration of how to use the MultiParticlePtCorrelations class to 
    calculate transverse momentum correlations and cumulants.

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx import *
        >>>
        >>> # Maximum order for correlations and cumulants
        >>> max_order = 8
        >>> # Create a MultiParticlePtCorrelations object
        >>> corr_obj = MultiParticlePtCorrelations(max_order=max_order)
        >>>
        >>> # List of events, where each event is a list of particle objects
        >>> particle_list = Jetscape("./particles.dat").particle_object_list()
        >>>
        >>> # Compute mean transverse momentum correlations
        >>> mean_pt_correlations = corr_obj.compute_mean_pt_correlations(particle_list_all_events)
        >>> print(mean_pt_correlations)
        >>>
        >>> # Compute mean transverse momentum cumulants
        >>> mean_pt_cumulants = corr_obj.compute_mean_pt_cumulants(particle_list_all_events)
        >>> print(mean_pt_cumulants)
    """
    def __init__(self, max_order):

        self.max_order = max_order
        # Check if max_order is an integer
        if not isinstance(self.max_order, int):
            raise ValueError("max_order must be an integer")
        # Check that max_order is greater than 0 and less than 9
        if self.max_order < 1 or self.max_order > 8:
            raise ValueError("max_order must be greater than 0 and less than 9")
        
        self.mean_pt_correlations = None
        self.kappa = None

    def P_W_k(self, particle_list_event):
        """
        This implements Eq. 7 in [1].

        Parameters
        ----------
        particle_list_event : list
            List of particle objects in a single event.

        Returns
        -------
        tuple of np.ndarray
            Pk : ndarray
                Transverse momentum for each order.
            Wk : ndarray
                Weights for each order.
        """
        Pk = np.zeros(self.max_order)
        Wk = np.zeros(self.max_order)
        for particle in particle_list_event:
            for k in range(1,self.max_order+1):
                Pk[k] += (particle.weight * particle.pt_abs())**k
                Wk[k] += particle.weight**k
        return (Pk, Wk)
    
    def transverse_momentum_correlations_event(self, particle_list_event):
        """
        Compute the transverse momentum correlations for a single event.
        
        Computes the numerators and denominators of Eqs. A1-A7 in Ref. [1] separately.

        Parameters
        ----------
        particle_list_event : list
            List of particle objects in a single event.

        Returns
        -------
        tuple of np.ndarray
            N : ndarray
                Numerators for each order.
            D : ndarray
                Denominators for each order.
        """
        Pk, Wk = self.P_W_k(particle_list_event, self.max_order)
        
        N = np.zeros(self.max_order)
        D = np.zeros(self.max_order)
        for order in range(self.max_order):
            if order == 0: # k = 1
                N[order] = Pk[order]
                D[order] = Wk[order]
            elif order == 1: # k = 2
                N[order] = Pk[0]**2. - Pk[1]
                D[order] = Wk[0]**2. - Wk[1]
            elif order == 2: # k = 3
                N[order] = Pk[0]**3. - 3.*Pk[1]*Pk[0] + 2.*Pk[2]
                D[order] = Wk[0]**3. - 3.*Wk[1]*Wk[0] + 2.*Wk[2]
            elif order == 3: # k = 4
                N[order] = (Pk[0]**4. - 6.*Pk[1]**2.*Pk[0] + 3.*Pk[1]**2. 
                            + 8.*Pk[2]*Pk[0] - 6.*Pk[3])
                D[order] = (Wk[0]**4. - 6.*Wk[1]**2.*Wk[0] + 3.*Wk[1]**2. 
                            + 8.*Wk[2]*Wk[0] - 6.*Wk[3])
            elif order == 4: # k = 5
                N[order] = (Pk[0]**5. - 10.*Pk[1]*Pk[0]**3. 
                            + 15.*Pk[1]**2.*Pk[0] + 20.*Pk[2]*Pk[0]**2.
                            - 20.*Pk[2]*Pk[1] - 30.*Pk[3]*Pk[0] + 24.*Pk[4])
                D[order] = (Wk[0]**5. - 10.*Wk[1]*Wk[0]**3.
                            + 15.*Wk[1]**2.*Wk[0] + 20.*Wk[2]*Wk[0]**2.
                            - 20.*Wk[2]*Wk[1] - 30.*Wk[3]*Wk[0] + 24.*Wk[4])
            elif order == 5: # k = 6
                N[order] = (Pk[0]**6. - 15.*Pk[1]*Pk[0]**4.
                            + 45.*Pk[0]**2.*Pk[1]**2. - 15.*Pk[1]**3.
                            - 40.*Pk[2]*Pk[0]**3. - 120.*Pk[2]*Pk[1]*Pk[0]
                            + 40.*Pk[2]**2. - 90.*Pk[3]*Pk[0]**2. 
                            + 90.*Pk[3]*Pk[1] + 144.*Pk[4]*Pk[0] - 120.*Pk[5])
                D[order] = (Wk[0]**6. - 15.*Wk[1]*Wk[0]**4.
                            + 45.*Wk[0]**2.*Wk[1]**2. - 15.*Wk[1]**3.
                            - 40.*Wk[2]*Wk[0]**3. - 120.*Wk[2]*Wk[1]*Wk[0]
                            + 40.*Wk[2]**2. - 90.*Wk[3]*Wk[0]**2. 
                            + 90.*Wk[3]*Wk[1] + 144.*Wk[4]*Wk[0] - 120.*Wk[5])
            elif order == 6: # k = 7
                N[order] = (Pk[0]**7. - 21.*Pk[1]*Pk[0]**5.
                            + 105.*Pk[0]**3.*Pk[1]**2. - 105.*Pk[1]**3.*Pk[0]
                            + 70.*Pk[2]*Pk[0]**4. - 420.*Pk[2]*Pk[1]*Pk[0]**2.
                            + 210.*Pk[2]*Pk[1]**2. + 280.*Pk[2]**2.*Pk[0]
                            - 210.*Pk[3]*Pk[0]**3. - 630.*Pk[3]*Pk[1]*Pk[0]
                            - 420.*Pk[3]*Pk[2] + 504.*Pk[4]*Pk[0]**2.
                            - 504.*Pk[4]*Pk[1] - 840.*Pk[5]*Pk[0] + 720.*Pk[6])
                D[order] = (Wk[0]**7. - 21.*Wk[1]*Wk[0]**5.
                            + 105.*Wk[0]**3.*Wk[1]**2. - 105.*Wk[1]**3.*Wk[0]
                            + 70.*Wk[2]*Wk[0]**4. - 420.*Wk[2]*Wk[1]*Wk[0]**2.
                            + 210.*Wk[2]*Wk[1]**2. + 280.*Wk[2]**2.*Wk[0]
                            - 210.*Wk[3]*Wk[0]**3. - 630.*Wk[3]*Wk[1]*Wk[0]
                            - 420.*Wk[3]*Wk[2] + 504.*Wk[4]*Wk[0]**2.
                            - 504.*Wk[4]*Wk[1] - 840.*Wk[5]*Wk[0] + 720.*Wk[6])
            elif order == 7: # k = 8
                N[order] = (Pk[0]**8. - 28.*Pk[1]*Pk[0]**6. 
                            - 210.*Pk[1]**2.*Pk[0]**4. 
                            - 420.*Pk[1]**3.*Pk[0]**2. + 105.*Pk[1]**4.
                            + 112.*Pk[2]*Pk[0]**5. + 1120.*Pk[2]*Pk[1]*Pk[0]**3.
                            + 1680.*Pk[2]*Pk[1]**2.*Pk[0] 
                            + 1120.*Pk[2]**2.*Pk[0]**2. + 1120.*Pk[2]**2.*Pk[1]
                            - 420.*Pk[3]*Pk[0]**4. + 2520.*Pk[3]*Pk[1]*Pk[0]**2.
                            - 1260.*Pk[3]*Pk[1]**2. - 3360.*Pk[3]*Pk[2]*Pk[0]
                            + 1260.*Pk[4]**2. + 1344.*Pk[4]*Pk[0]**3.
                            - 4032.*Pk[4]*Pk[1]*Pk[0] + 2688.*Pk[4]*Pk[2]
                            - 3360.*Pk[5]*Pk[0]**2. + 3360.*Pk[5]*Pk[1]
                            + 5760.*Pk[6]*Pk[0] - 5040.*Pk[7])
                D[order] = (Wk[0]**8. - 28.*Wk[1]*Wk[0]**6.
                            - 210.*Wk[1]**2.*Wk[0]**4.
                            - 420.*Wk[1]**3.*Wk[0]**2. + 105.*Wk[1]**4.
                            + 112.*Wk[2]*Wk[0]**5. + 1120.*Wk[2]*Wk[1]*Wk[0]**3.
                            + 1680.*Wk[2]*Wk[1]**2.*Wk[0]
                            + 1120.*Wk[2]**2.*Wk[0]**2. + 1120.*Wk[2]**2.*Wk[1]
                            - 420.*Wk[3]*Wk[0]**4. + 2520.*Wk[3]*Wk[1]*Wk[0]**2.
                            - 1260.*Wk[3]*Wk[1]**2. - 3360.*Wk[3]*Wk[2]*Wk[0]
                            + 1260.*Wk[4]**2. + 1344.*Wk[4]*Wk[0]**3.
                            - 4032.*Wk[4]*Wk[1]*Wk[0] + 2688.*Wk[4]*Wk[2]
                            - 3360.*Wk[5]*Wk[0]**2. + 3360.*Wk[5]*Wk[1]
                            + 5760.*Wk[6]*Wk[0] - 5040.*Wk[7])

        return N, D

    def compute_mean_pt_correlations(self, particle_list_all_events):
        """
        Computes the mean transverse momentum correlations for each order k
        in all events.

        Parameters
        ----------
        particle_list_all_events : list
            List of events, where each event is a list of particle objects.

        Returns
        -------
        np.ndarray
            Mean transverse momentum correlations for each order.
            This computes the mean transverse momentum correlations for each 
            order k from all events in the list.
        """
        sum_numerator = np.zeros(self.max_order)
        sum_denominator = np.zeros(self.max_order)

        for event in particle_list_all_events:
            N, D = self.transverse_momentum_correlations_event(event)
            sum_numerator += N
            sum_denominator += D

        self.mean_pt_correlations = sum_numerator / sum_denominator
        
        return self.mean_pt_correlations

    def compute_mean_pt_cumulants(self, particle_list_all_events):
        """
        Computes the mean transverse momentum cumulants for each order k
        from Eqs. B9-B16 in Ref. [1].

        Parameters
        ----------
        particle_list_all_events : list
            List of events, where each event is a list of particle objects.

        Returns
        -------
        np.ndarray
            Mean transverse momentum cumulants for each order.
        """
        kappa = np.zeros(self.max_order)

        if self.mean_pt_correlations is None:
            self.compute_mean_pt_correlations(particle_list_all_events)

        C = self.mean_pt_correlations
        for order in range(self.max_order):
            if order == 0: # k = 1
                kappa[0] = C[0]
            elif order == 1: # k = 2
                kappa[1] = C[1] - C[0]**2
            elif order == 2: # k = 3
                kappa[2] = C[2] - 3.*C[1]*C[0] + 2.*C[0]**3
            elif order == 3: # k = 4
                kappa[3] = (C[3] - 4.*C[2]*C[0] - 3.*C[1]**2 + 12.*C[1]*C[0]**2 
                            - 6.*C[0]**4)
            elif order == 4: # k = 5
                kappa[4] = (C[4] - 5.*C[3]*C[0] - 10.*C[2]*C[1] 
                            + 30.*C[1]**2*C[0] + 20.*C[2]*C[0]**2
                            - 60.*C[1]*C[0]**3 + 24.*C[0]**5)
            elif order == 5: # k = 6
                kappa[5] = (C[5] - 6.*C[4]*C[0] - 15.*C[3]*C[1] 
                            - 10.*C[2]**2 + 30.*C[1]**3 + 30.*C[3]*C[0]**2
                            + 120.*C[2]*C[1]*C[0] - 270.*C[1]**2*C[0]**2
                            - 120.*C[2]*C[0]**3 + 360.*C[1]*C[0]**4 
                            - 120.*C[0]**6)
            elif order == 6: # k = 7
                kappa[6] = (C[6] - 7.*C[5]*C[0] - 21.*C[4]*C[1] 
                            + 42.*C[4]*C[0]**2 - 35.*C[3]*C[2]
                            + 210.*C[3]*C[1]*C[0] - 210.*C[3]*C[0]**3
                            + 140.*C[2]**2*C[0] + 210.*C[2]*C[1]**2
                            - 1260.*C[2]*C[1]*C[0]**2 + 840.*C[2]*C[0]**4
                            - 630.*C[1]**3*C[0] + 2520.*C[1]**2*C[0]**3
                            - 2520.*C[1]*C[0]**5 + 720.*C[0]**7)
            elif order == 7: # k = 8
                kappa[7] = (C[7] - 8.*C[6]*C[0] - 28.*C[5]*C[1] 
                            + 56.*C[5]*C[0]**2 - 56.*C[4]*C[2]
                            + 336.*C[4]*C[1]*C[0] - 336.*C[4]*C[0]**3
                            - 35.*C[3]**2 + 560.*C[3]*C[2]*C[0]
                            + 420.*C[1]**2*C[3] - 2520.*C[3]*C[1]*C[0]**2
                            + 1680.*C[3]*C[0]**4 + 560.*C[2]**2*C[1]
                            - 1680.*C[2]**2*C[0]**2 - 5040.*C[2]*C[1]**2*C[0]
                            + 13440.*C[2]*C[1]*C[0]**3 - 6720.*C[2]*C[0]**5
                            - 630.*C[1]**4 + 10080.*C[1]**3*C[0]**2
                            - 25200.*C[1]**2*C[0]**4 + 20160.*C[1]*C[0]**6
                            - 5040.*C[0]**8)
        self.kappa = kappa
        return self.kappa