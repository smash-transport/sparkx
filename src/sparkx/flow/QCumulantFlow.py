from sparkx.flow import FlowInterface
import numpy as np
import random as rd
import warnings

rd.seed(42)

class QCumulantFlow(FlowInterface.FlowInterface):
    """
    This class implements the Q-cumulant method for anisotropic flow analysis.

    References:

    - [1] `PhD Thesis A. Bilandzic <https://inspirehep.net/literature/1186272>`__
    - [2] `Phys.Rev.C 83 (2011) 044913 <https://inspirehep.net/literature/871528>`__
    
    Parameters
    ----------
    n : int, optional
        The order of the harmonic flow (default is 2).
    k : int, optional
        The order of the cumulant (2, 4, or 6) (default is 2).
    imaginary : str, optional
        Specifies the treatment of imaginary roots. Options are 'zero', 
        'negative', or 'nan' (default is 'zero').

    Attributes
    ----------
    n_ : int
        The order of the harmonic flow.
    k_ : int
        The order of the cumulant.
    imaginary_ : str
        Specifies the treatment of imaginary roots.
    cumulant_factor_ : dict
        Dictionary mapping cumulant order to corresponding flow.
    rand_reaction_planes_ : list
        List to store randomly sampled reaction planes.

    Methods
    -------
    integrated_flow:
        Computes the integrated flow.

    differential_flow:
        Computes the differential flow.

    Examples
    --------
    A demonstration of how to use the QCumulantFlow class to calculate flow.

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> flow_instance = QCumulantFlow(n=2, k=2, imaginary='zero')
        >>> result = flow_instance.integrated_flow(particle_data)
    """
    def __init__(self,n=2,k=2,imaginary='zero'):
        if not isinstance(n, int):
            raise TypeError('n has to be int')
        elif n <= 0:
            raise ValueError('n-th harmonic with value n<=0 can not be computed')
        else:
            self.n_ = n

        if not isinstance(k, int):
            raise TypeError('k has to be int')
        elif k not in [2,4,6]:
            raise ValueError(f"{k} particle cumulant is not implemented, choose from [2,4]")
        else:
            self.k_ = k

        if not isinstance(imaginary, str):
            raise TypeError("Chosen 'imaginary' is not implemented")
        elif imaginary not in ['zero','negative','nan']:
            raise ValueError(f"Chosen 'imaginary' = {imaginary} is not an option")
        else:
            self.imaginary_ = imaginary

        self.cumulant_factor_ = {
            2: 1,
            4: -1,
            6: 1./4.
        }

        self.rand_reaction_planes_ = []

    def __Qn(self, phi, n):
        """
        Compute the Q_n vector for each event based on azimuthal angles.

        Parameters
        ----------
        phi : list
            List of azimuthal angles for each particle in an event.
        n : int
            Order of the flow vector (Q_n).

        Returns
        -------
        np.ndarray
            Array of complex numbers representing the Q_n vector for each event.

        """
        Q_vector = []
        for event in range(len(phi)):
            phi_event = np.array([i for i in phi[event]])
            Q_vector_val = np.sum(np.exp(1.0j * float(n) * phi_event))
            Q_vector.append(Q_vector_val)
        
        return np.array(Q_vector)

    def __calculate_corr(self,phi,k):
        """
        Calculate cumulant and its error for a given order (k).

        Parameters
        ----------
        phi : list
            List of azimuthal angles for each particle in an event.
        k : int
            Order of the cumulant to be calculated (2, 4, or 6).

        Returns
        -------
        tuple
            A tuple containing the computed cumulant, its error, 
            and the event-by-event cumulant.
        """
        mult = np.array([float(len(i)) for i in phi])
        sum_mult = np.sum(mult)
        sum_mult_squared = np.inner(mult,mult)

        Qn = self.__Qn(phi,self.n_)
        Qn_sq_sum = np.vdot(Qn,Qn).real # |Q_n|^2

        if k == 2:
            # this implements Eq. (16) from Ref. [1]
            corr = (Qn_sq_sum - sum_mult) / (sum_mult_squared - sum_mult)
            
            W2 = mult*(mult-1)
            sum_W2 = np.sum(W2)
            sum_W2_sq = np.inner(W2,W2)

            # ebe difference from mean: <2>_i - <<2>>
            ebe_2p_corr = ((np.real(Qn*Qn.conj())-mult) / W2)
            difference = ebe_2p_corr - corr
            # weighted variance
            variance = np.sum(W2*np.square(difference)) / sum_W2
            # unbiased variance^2
            variance_sq = variance / (1. - sum_W2_sq/(sum_W2**2.))
            # error of <<2>>, Eq. (C18) Ref. [1]
            corr_err = np.sqrt(sum_W2_sq*variance_sq)/sum_W2

            return corr, corr_err, ebe_2p_corr
        
        if k == 4:
            # this implements Eq. (18) from Ref. [1]
            Q2n = self.__Qn(phi,2.*self.n_)
            Q2n_sq_sum = np.vdot(Q2n,Q2n).real
            Qn_sq = np.square(Qn.real) + np.square(Qn.imag)
            Qn_to4_sum = np.inner(Qn_sq,Qn_sq)

            corr = (Qn_to4_sum 
                    + Q2n_sq_sum 
                    - 2.*np.inner(Q2n, np.square(Qn.conj())).real
                    - 2*np.sum(2*(mult-2)*Qn_sq - mult*(mult-3))
                    ) / (np.sum(mult*(mult-1)*(mult-2)*(mult-3)))

            # corr_err computation here:
            W4 = mult*(mult-1)*(mult-2)*(mult-3)
            sum_W4 = np.sum(W4)
            sum_W4_sq = np.inner(W4,W4) 

            # ebe difference from mean: <4>_i - <<4>>
            ebe_4p_corr = (np.real(Qn*Qn*Qn.conj()*Qn.conj()) + np.real(Q2n*Q2n.conj()) 
                           - 2.* np.real(Q2n*Qn.conj()*Qn.conj())
                           - 2.*(2.*(mult-2)*np.real(Q2n*Q2n.conj()) - mult*(mult-3))) / W4
            difference = ebe_4p_corr - corr
            # weighted variance
            variance = np.sum(W4*np.square(difference)) / sum_W4
            # unbiased variance^2
            variance_sq = variance / (1. - sum_W4_sq/(sum_W4**2.))
            # error of <<4>>, Eq. (C18) Ref. [1]
            corr_err = np.sqrt(sum_W4_sq*variance_sq)/sum_W4   

            return corr, corr_err, ebe_4p_corr
        
        if k == 6:
            # this implements Eq. (A10) from Ref. [1]
            Q2n = self.__Qn(phi,2.*self.n_)
            Q2n_sq_sum = np.vdot(Q2n,Q2n).real
            Qn_sq = np.square(Qn.real) + np.square(Qn.imag)
            Qn_to4_sum = np.inner(Qn_sq,Qn_sq)
            Qn_to6 = np.power(Qn_sq, 3)
            Qn_to6_sum = np.sum(Qn_to6)
            Q3n = self.__Qn(phi,3.*self.n_)
            Q3n_sq_sum = np.vdot(Q3n,Q3n).real
            ReQ2nQnConjSq = np.inner(Q2n, np.square(Qn.conj())).real
            ReQ3nQ2nConjQnConj = np.vdot(Q3n,np.multiply(Q2n.conj(),Qn.conj())).real
            QnConj_cub = np.power(Qn.conj(), 3)
            ReQ3nQnConjCub = np.inner(Q3n,QnConj_cub).real
            ReQ2nQnQnConjCub = np.inner(np.multiply(Q2n,Qn),QnConj_cub).real
            norm1 = mult*(mult-1)*(mult-2)*(mult-3)*(mult-4)*(mult-5)
            norm1_sum = np.sum(norm1)
            norm2 = mult*(mult-1)*(mult-2)*(mult-3)*(mult-5)
            norm2_sum = np.sum(norm2)
            norm3 = mult*(mult-1)*(mult-3)*(mult-4)
            norm3_sum = np.sum(norm3)
            norm4 = (mult-1)*(mult-2)*(mult-3)
            norm4_sum = np.sum(norm4)
            
            corr1 = (Qn_to6_sum
                + 9.*Q2n_sq_sum*Qn_sq_sum
                - 6.*ReQ2nQnQnConjCub
                )/norm1_sum
            corr2 = 4.* (ReQ3nQnConjCub
                - 3.*ReQ3nQ2nConjQnConj
                )/norm1_sum
            corr3 = 2. * (9.*np.sum((mult-4)*ReQ2nQnConjSq)
                + 2.*Q3n_sq_sum
                )/norm1_sum
            corr4 = -9. * (Qn_to4_sum
                + Q2n_sq_sum
                )/norm2_sum
            corr5 = 18. * Qn_sq_sum / norm3_sum
            corr6 = -6. / norm4_sum
            corr = corr1 + corr2 + corr3 + corr4 + corr5 + corr6
            
            # corr_err computation here:
            W6 = mult*(mult-1)*(mult-2)*(mult-3)*(mult-4)*(mult-5)
            sum_W6 = np.sum(W6)
            sum_W6_sq = np.inner(W6,W6) 

            # ebe difference from mean: <6>_i - <<6>>
            ebe_6p_corr1 = (np.real(Qn*Qn*Qn*Qn.conj()*Qn.conj()*Qn.conj()) 
                            + 9.*np.real(Q2n*Q2n.conj()) * np.real(Qn*Qn.conj()) 
                            - 6.* np.real(Q2n*Qn*QnConj_cub))/norm1
            ebe_6p_corr2 = (4.*(np.real(Q3n*QnConj_cub)
                            - 3.*np.real(Q3n*Q2n.conj()*Qn.conj())))/norm1
            ebe_6p_corr3 = (2.*(9.*(mult-4)*np.real(Q2n*Qn.conj()*Qn.conj())
                            + 2.*np.real(Q3n*Q3n.conj())))/norm1
            ebe_6p_corr4 = (-9.*np.real(Qn*Qn*Qn.conj()*Qn.conj()) + np.real(Q2n*Q2n.conj()))/norm2
            ebe_6p_corr5 = (18.*np.real(Qn*Qn.conj()))/norm3
            ebe_6p_corr6 = -6./norm4
            ebe_6p_corr = (ebe_6p_corr1 + ebe_6p_corr2 + ebe_6p_corr3 
                           + ebe_6p_corr4 + ebe_6p_corr5 + ebe_6p_corr6)
            difference = ebe_6p_corr - corr
            # weighted variance
            variance = np.sum(W6*np.square(difference)) / sum_W6
            # unbiased variance^2
            variance_sq = variance / (1. - sum_W6_sq/(sum_W6**2.))
            # error of <<6>>, Eq. (C18) Ref. [1]
            corr_err = np.sqrt(sum_W6_sq*variance_sq)/sum_W6

            return corr, corr_err, ebe_6p_corr

    def __cov(self,wx,wy,x,y):
        """
        Compute the covariance between two sets of variables based on event weights.

        Parameters
        ----------
        wx : ndarray
            Event weights for the first set of variables.
        wy : ndarray
            Event weights for the second set of variables.
        x : ndarray
            Values of the first set of variables.
        y : ndarray
            Values of the second set of variables.

        Returns
        -------
        float
            The computed covariance.

        Notes
        -----
        This function calculates the covariance between two sets of variables 
        using event weights. It follows the formula provided in Eq. (C12) from Ref. [1].
        """
        sum_wx_wy_x_y = np.sum(wx*wy*x*y)
        sum_wx_wy = np.sum(wx*wy)
        sum_wx_x = np.sum(wx*x)
        sum_wx = np.sum(wx)
        sum_wy_y = np.sum(wy*y)
        sum_wy = np.sum(wy)

        cov = (((sum_wx_wy_x_y/sum_wx_wy) 
               - (sum_wx_x/sum_wx)*(sum_wy_y/sum_wy)) 
               / (1. - (sum_wx_wy/(sum_wx*sum_wy))))
        
        return cov

    def __cov_term(self,k1,k2,phi,ebe_corr1,ebe_corr2):
        """
        Compute the covariance term for the last term in Eq. (C29) from Ref. [1].

        Parameters
        ----------
        k1 : int
            The order of the first cumulant.
        k2 : int
            The order of the second cumulant.
        phi : list of lists
            List of particle data, where each sublist represents the azimuthal angles (phi) for each event.
        ebe_corr1 : ndarray
            Event-by-event correlation data for the first cumulant.
        ebe_corr2 : ndarray
            Event-by-event correlation data for the second cumulant.

        Returns
        -------
        float
            The computed covariance term.

        Notes
        -----
        This function implements the fraction and covariance term in the last 
        term of Eq. (C29) from Ref. [1]. The implementation is designed to be 
        more general and can be used for all higher order cumulants.
        """
        mult = np.array([float(len(i)) for i in phi])
        W1 = 1.
        for i in range(k1):
            W1 *= (mult - i)

        W2 = 1.
        for i in range(k2):
            W2 *= (mult - i)

        W1_sum = np.sum(W1)
        W2_sum = np.sum(W2)
        W1W2_sum = np.inner(W1,W2)
        cov_term = (W1W2_sum/(W1_sum*W2_sum))*self.__cov(W1,W2,ebe_corr1,ebe_corr2)

        return cov_term

    def __flow_from_cumulant(self,cnk):
        """
        Compute the flow magnitude from a cumulant value.

        Parameters
        ----------
        cnk : float
            Cumulant value corresponding to the order of the flow.

        Returns
        -------
        float
            The computed flow magnitude.

        Notes
        -----
        This function calculates the flow magnitude from a cumulant value using 
        the specified cumulant order (k).
        It considers the sign of the cumulant value and the chosen behavior 
        for imaginary roots.
        """
        vnk_to_k = self.cumulant_factor_[self.k_] * cnk
        
        if vnk_to_k >= 0.:
            vnk = vnk_to_k**(1/self.k_)
        elif self.imaginary_ == 'negative':
            vnk = -1.*(-vnk_to_k)**(1/self.k_)
        elif self.imaginary_ == 'zero':
            vnk = 0.
        else:
            vnk = float('nan')
        return vnk

    def __cumulant_flow(self,phi):
        """
        Compute the flow magnitude and its uncertainty from cumulant values.

        Parameters
        ----------
        phi : list of lists
            List of particle data, where each sublist represents the azimuthal 
            angles (phi) for each event.

        Returns
        -------
        tuple
            A tuple containing the computed flow magnitude and its uncertainty.

        Notes
        -----
        This function calculates the flow magnitude and its uncertainty based on 
        cumulant values. It supports cumulant of different orders, following the 
        equations from Ref. [1].
        """
        if self.k_ == 2:
            n2_corr, n2_corr_err, ebe_2p_corr = self.__calculate_corr(phi,k=2)

            # returns <v_n{2}> and s_{<v_n{2}>}, Eqs. (C22),(C24) Ref. [1]
            avg_vn2 = self.__flow_from_cumulant(n2_corr)
            avg_v2_err = (1./(2.*avg_vn2))*n2_corr_err
            
            return avg_vn2, avg_v2_err
        
        elif self.k_ == 4:
            n2_corr, n2_corr_err, ebe_2p_corr = self.__calculate_corr(phi,k=2)
            n4_corr, n4_corr_err, ebe_4p_corr = self.__calculate_corr(phi,k=4)

            QC4 = n4_corr - 2.*n2_corr**2.
            avg_vn4 = self.__flow_from_cumulant(QC4)

            # compute Eq. (C28) Ref. [1]
            avg_vn4_err_sq = (1./(2.*n2_corr**2.-n4_corr)**(3./2)) * (
                            n2_corr**2. * n2_corr_err**2.
                            + (1./16.)*n4_corr_err**2.
                            - (1./2.)*n2_corr*self.__cov_term(2,4,phi,ebe_2p_corr,ebe_4p_corr)
                            )
            
            # returns <v_n{4}> and s_{<v_n{4}>}, Eqs. (C27),(C28) Ref. [1]
            return avg_vn4, np.sqrt(avg_vn4_err_sq)
        
        elif self.k_ == 6:
            n2_corr, n2_corr_err, ebe_2p_corr = self.__calculate_corr(phi,k=2)
            n4_corr, n4_corr_err, ebe_4p_corr = self.__calculate_corr(phi,k=4)
            n6_corr, n6_corr_err, ebe_6p_corr = self.__calculate_corr(phi,k=6)

            QC6 = n6_corr - 9.*n2_corr*n4_corr + 12.*n2_corr**3.
            avg_vn6 = self.__flow_from_cumulant(QC6)

            # compute Eq. (C32) Ref. [1]
            avg_vn6_err_sq = ((1./(2.*2.**(2./3.))) * (1./(QC6)**(5./3.)) 
                              * ((9./4.)*(4.*n2_corr**2.-n4_corr)**2.*n2_corr_err**2.
                            + (9./2.)*n2_corr**2.*n4_corr_err**2.
                            + (1./18.)*n6_corr_err**2.
                            - 9.*n2_corr*(4.*n2_corr**2.-n4_corr)*self.__cov_term(2,4,phi,ebe_2p_corr,ebe_4p_corr)
                            + (4.*n2_corr**2.-n4_corr)*self.__cov_term(2,6,phi,ebe_2p_corr,ebe_6p_corr)
                            - n2_corr*self.__cov_term(4,6,phi,ebe_4p_corr,ebe_6p_corr))
                            )
            print(avg_vn6_err_sq,np.sqrt(-avg_vn6_err_sq))
            # returns <v_n{6}> and s_{<v_n{6}>}, Eq. (C33) Ref. [1]
            return avg_vn6, np.sqrt(avg_vn6_err_sq)

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
        angles = []
        for i in range(events):
            angles.append(rd.random()*2.*np.pi)
        self.rand_reaction_planes_ = angles

    def integrated_flow(self,particle_data):
        """
        Compute the integrated flow.

        Parameters
        ----------
        particle_data : list
            List of particle data, where each sublist represents an event 
            with particles.

        Returns
        -------
        tuple
            A tuple containing the computed flow magnitude and its uncertainty.
        """
        number_events = len(particle_data)
        self.__sample_random_reaction_planes(number_events)

        phi = []
        for event in range(number_events):
            event_phi = []
            for particle in particle_data[event]:
                event_phi.append(particle.phi()+self.rand_reaction_planes_[event])
            phi.extend([event_phi])

        vnk, vnk_err = self.__cumulant_flow(phi)
        
        return vnk, vnk_err


    def differential_flow(self, particle_data, bins, flow_as_function_of):
        """
        Compute the differential flow.

        Parameters
        ----------
        particle_data : list
            List of particle data.
        bins : list or np.ndarray
            Bins used for the differential flow calculation.
        flow_as_function_of : str
            Variable on which the flow is calculated ("pt", "rapidity", or "pseudorapidity").

        Returns
        -------
        list of tuples
            A list of tuples containing a flow values and their corresponding uncertainty.
        """
        warnings.warn("This feature is not yet available.")
        return None