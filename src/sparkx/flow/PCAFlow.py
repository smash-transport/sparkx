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
import warnings
import random as rd

rd.seed(42)

class PCAFlow(FlowInterface.FlowInterface):
    """
    This class implements the flow analysis with Principal Component 
    Analysis (PCA).

    PCAFlow is designed for analyzing anisotropic flow in high-energy nuclear 
    collisions using two-particle correlations.
    It uses Principal Component Analysis (PCA) to extract flow information from 
    particle data.

    This class implements the method proposed in:

    - [1] `Phys.Rev.Lett. 114 (2015) 15, 152301 [nucl-th/1410.7739] <https://inspirehep.net/literature/1324816>`__
    
    The implementation is done in a similar way as in:
    
    - [2] `Bachelor thesis D.J.W. Verweij (2016) <https://studenttheses.uu.nl/handle/20.500.12932/26817>`__

    Parameters
    ----------
    n : int, optional
        The flow harmonic to compute. Must be a positive integer. Default is 2.
    alpha : int, optional
        The order in sub-leading flow up to which the flow is computed. 
        Must be an integer greater than or equal to 1. Default is 2.
    number_subcalc : int, optional
        The number of sub-calculations to estimate the error of the flow. 
        Must be an integer greater than or equal to 2. Default is 4.

    Methods
    -------
    differential_flow:
        Computes the differential flow.
    Pearson_correlation:
        Returns the Pearson coefficient and its uncertainty if the flow is 
        already computed.

    Attributes
    ----------
    n_ : int
        The flow harmonic to compute.
    alpha_ : int
        The order in sub-leading flow up to which the flow is computed.
    number_subcalc_ : int
        The number of sub-calculations to estimate the error of the flow.
    number_events_ : int or None
        The total number of events.
    subcalc_counter_ : int
        A counter to keep track of the current sub-calculation.
    normalization_ : None or numpy.ndarray
        Normalization factors for each bin.
    bin_multiplicity_total_ : None or numpy.ndarray
        Total multiplicity in each bin across all events.
    sigma_multiplicity_total_ : list of numpy.ndarray
        List of arrays containing the multiplicity in each bin for each 
        sub-calculation.
    number_events_subcalc_ : None or numpy.ndarray
        Number of events used in each sub-calculation.
    QnRe_total_ : None or numpy.ndarray
        Real part of the flow vector for each bin across all events.
    QnIm_total_ : None or numpy.ndarray
        Imaginary part of the flow vector for each bin across all events.
    SigmaQnReSub_total_ : None or numpy.ndarray
        Real part of the flow vector for each bin and sub-calculation.
    SigmaQnImSub_total_ : None or numpy.ndarray
        Imaginary part of the flow vector for each bin and sub-calculation.
    VnDelta_total_ : None or numpy.ndarray
        Covariance matrix for flow vectors across all events.
    SigmaVnDelta_total_ : None or numpy.ndarray
        Covariance matrix for flow vectors for each sub-calculation.
    FlowSubCalc_ : None or numpy.ndarray
        An array containing the flow magnitude for sub-calculations.
    Flow_ : None or numpy.ndarray
        An array containing the flow magnitude.
    FlowUncertainty_ : None or numpy.ndarray
        An array containing the uncertainty of the flow.
    Pearson_r_ : None or numpy.ndarray
        An array containing the Pearson correlation between two bins.
    Pearson_r_uncertainty_ : None or numpy.ndarray
        An array containing the Pearson correlation uncertainty between two bins.

    Raises
    ------
    TypeError
        If n, alpha, or number_subcalc is not an integer.
    ValueError
        If n is not a positive integer, alpha is less than 1, 
        or number_subcalc is less than 2.
    """

    def __init__(self,n=2,alpha=2,number_subcalc=4):
        # flow harmonic to compute
        if not isinstance(n, int):
            raise TypeError('n has to be int')
        elif n <= 0:
            raise ValueError('n-th harmonic with value n<=0 can not be computed')
        else:
            self.n_ = n

        # order in sub-leading flow up to which the flow is computed
        if not isinstance(alpha, int):
            raise TypeError('alpha has to be int')
        elif alpha < 1:
            raise ValueError('alpha has to be >= 1')
        else:
            self.alpha_ = alpha

        # number of sub-calculations to estimate the error of the flow
        if not isinstance(number_subcalc, int):
            raise TypeError('number_subcalc has to be int')
        elif number_subcalc < 2:
            raise ValueError('number_subcalc has to be >= 2')
        else:
            self.number_subcalc_ = number_subcalc

        self.number_events_ = None
        self.subcalc_counter_ = 0

        self.normalization_ = None
        self.bin_multiplicity_total_ = None
        self.sigma_multiplicity_total_ = []
        self.number_events_subcalc_ = None
        self.QnRe_total_ = None
        self.QnIm_total_ = None
        self.SigmaQnReSub_total_ = None
        self.SigmaQnImSub_total_ = None
        self.VnDelta_total_ = None
        self.SigmaVnDelta_total_ = None
        
        self.FlowSubCalc_ = None
        self.Flow_ = None
        self.FlowUncertainty_ = None

        self.Pearson_r_ = None
        self.Pearson_r_uncertainty_ = None


    def integrated_flow(self, particle_data):
        warnings.warn("Integrated flow is not supported for PCAFlow")
        return None
    
    def __compute_normalization(self,bins):
        self.normalization_ = np.zeros((len(bins)-1))
        bin_widths = np.diff(bins)
        for bin in range(len(bins)-1):
            self.normalization_[bin] = 1. / (2.*np.pi*bin_widths[bin])**2.

    def __update_event(self,event_data,bins,flow_as_function_of,event_number):
        """
        Update the anisotropic flow calculations based on particle data from 
        a single event.

        Parameters
        ----------
        event_data : list
            List of particle data for a single event.
        bins : list or np.ndarray
            Bins used for the flow calculation.
        flow_as_function_of : str
            Variable on which the flow is calculated ("pt", "rapidity" 
            or "pseudorapidity").
        event_number : int
            Index of the current event.

        Returns
        -------
        None
        """
        number_bins = len(bins)-1
        bin_multiplicity_event = np.zeros(number_bins)
        QnRe = np.zeros(number_bins)
        QnIm = np.zeros(number_bins)
        SigmaQnReSubTot = np.zeros((number_bins,self.number_subcalc_))
        SigmaQnImSubTot = np.zeros((number_bins,self.number_subcalc_))
        VnDelta_event = np.zeros((number_bins,number_bins))
        SigmaVnDelta_event = np.zeros((number_bins,number_bins,self.number_subcalc_))
        if event_number == 0:
            self.bin_multiplicity_total_ = np.zeros(number_bins)
            for i in range(self.number_subcalc_):
                self.sigma_multiplicity_total_.append(np.zeros(number_bins))
            self.number_events_subcalc_ = np.zeros(self.number_subcalc_)
            self.QnRe_total_ = np.zeros(number_bins)
            self.QnIm_total_ = np.zeros(number_bins)
            self.SigmaQnReSub_total_ = np.zeros((number_bins,self.number_subcalc_))
            self.SigmaQnImSub_total_ = np.zeros((number_bins,self.number_subcalc_))
            self.VnDelta_total_ = np.zeros((number_bins,number_bins))
            self.SigmaVnDelta_total_ = np.zeros((number_bins,number_bins,self.number_subcalc_))

        # update the sub-calculation counter if needed
        number_events_subcalc = self.number_events_//self.number_subcalc_
        if (event_number%number_events_subcalc == 0) and (event_number != 0) and (self.subcalc_counter_ < self.number_subcalc_-1):
            self.subcalc_counter_ += 1

        random_reaction_plane = rd.random()*2.*np.pi

        # loop over all event particles and compute the flow vectors in the bins
        for particle in event_data:
            if flow_as_function_of == "pt":
                val = particle.pt_abs()
            elif flow_as_function_of == "rapidity":
                val = particle.momentum_rapidity_Y()
            elif flow_as_function_of == "pseudorapidity":
                val = particle.pseudorapidity()

            bin_index = np.digitize(val, bins, right=False)-1
            # handle the case in which the particle is not in the binned phase space
            if (bin_index < 0) or (bin_index > number_bins-1):
                continue

            phi = particle.phi()+random_reaction_plane

            bin_multiplicity_event[bin_index] += 1.
            QnRe[bin_index] += np.cos(self.n_*phi)
            QnIm[bin_index] += np.sin(self.n_*phi)

            SigmaQnReSubTot[bin_index][self.subcalc_counter_] += np.cos(self.n_*phi)
            SigmaQnImSubTot[bin_index][self.subcalc_counter_] += np.sin(self.n_*phi)

        # compute the covariance matrix
        for a in range(number_bins):
            for b in range(number_bins):
                if a == b: # correct with multiplicity term
                    VnDelta_event[a][b] += (QnRe[a]*QnRe[b] + QnIm[a]*QnIm[b] - bin_multiplicity_event[a])/self.normalization_[a]
                    SigmaVnDelta_event[a][b][self.subcalc_counter_] += (QnRe[a]*QnRe[b] + QnIm[a]*QnIm[b] - bin_multiplicity_event[a])/self.normalization_[a]
                else:
                    VnDelta_event[a][b] += (QnRe[a]*QnRe[b] + QnIm[a]*QnIm[b])/self.normalization_[a]
                    SigmaVnDelta_event[a][b][self.subcalc_counter_] += (QnRe[a]*QnRe[b] + QnIm[a]*QnIm[b])/self.normalization_[a]

        # update the class members
        self.bin_multiplicity_total_ += bin_multiplicity_event
        self.sigma_multiplicity_total_[self.subcalc_counter_] += bin_multiplicity_event
        self.number_events_subcalc_[self.subcalc_counter_] += 1
        self.QnRe_total_ += QnRe
        self.QnIm_total_ += QnIm
        self.SigmaQnReSub_total_ += SigmaQnReSubTot
        self.SigmaQnImSub_total_ += SigmaQnImSubTot
        self.VnDelta_total_ += VnDelta_event
        self.SigmaVnDelta_total_ += SigmaVnDelta_event

    def __compute_flow_PCA(self,bins):
        """
        Perform Principal Component Analysis (PCA) to compute the anisotropic 
        flow.

        Parameters
        ----------
        bins : list or np.ndarray
            Bins used for the flow calculation.

        Returns
        -------
        None
        """
        number_bins = len(bins)-1

        for a in range(number_bins):
            for b in range(number_bins):
                self.VnDelta_total_[a][b] /= self.number_events_
                self.VnDelta_total_[a][b] -= (self.QnRe_total_[a]*self.QnRe_total_[b] + self.QnIm_total_[a]*self.QnIm_total_[b]) / self.number_events_**2.
        
                for sub in range(self.number_subcalc_):
                    self.SigmaVnDelta_total_[a][b][sub] /= self.number_events_subcalc_[sub]
                    self.SigmaVnDelta_total_[a][b][sub] -= (self.SigmaQnReSub_total_[a][sub]*self.SigmaQnReSub_total_[b][sub] + self.SigmaQnImSub_total_[a][sub]*self.SigmaQnImSub_total_[b][sub]) / (self.number_events_subcalc_[sub]**2.*self.normalization_[a])
        
        # perform sub calculations for error estimation
        self.FlowSubCalc_ = np.zeros((self.number_subcalc_,number_bins,self.alpha_))
        for sub in range(self.number_subcalc_):
            VnDelta_local = np.zeros((number_bins,number_bins))
            for a in range(number_bins):
                for b in range(number_bins):
                    VnDelta_local[a][b] = self.SigmaVnDelta_total_[a][b][sub]
            
            eigenvalues, eigenvectors = np.linalg.eig(VnDelta_local)
            sort_indices = np.argsort(eigenvalues)[::-1]

            # Apply sorting to eigenvalues and eigenvectors
            sorted_eigenvalues = eigenvalues[sort_indices]
            sorted_eigenvectors = eigenvectors[:, sort_indices]
            
            for bin in range(number_bins):
                for alpha in range(self.alpha_):
                    eval = sorted_eigenvalues[alpha]
                    evec = sorted_eigenvectors[alpha]
                    if (eval >= 0.) and (self.sigma_multiplicity_total_[sub][bin] > 0.):
                        # compute the flow here for the subcalc
                        self.FlowSubCalc_[sub][bin][alpha] = np.sqrt(eval) * evec[bin] * self.number_events_subcalc_[sub] / self.sigma_multiplicity_total_[sub][bin]
                    else:
                        self.FlowSubCalc_[sub][bin][alpha] = None

        # compute the anisotropic flow
        VnDelta_local = np.zeros((number_bins,number_bins))
        for a in range(number_bins):
            for b in range(number_bins):
                VnDelta_local[a][b] = self.VnDelta_total_[a][b]

        eigenvalues, eigenvectors = np.linalg.eig(VnDelta_local)
        sort_indices = np.argsort(eigenvalues)[::-1]

        # Apply sorting to eigenvalues and eigenvectors
        sorted_eigenvalues = eigenvalues[sort_indices]
        sorted_eigenvectors = eigenvectors[:, sort_indices]

        self.Flow_ = np.zeros((number_bins,self.alpha_))
        for bin in range(number_bins):
            for alpha in range(self.alpha_):
                eval = sorted_eigenvalues[alpha]
                evec = sorted_eigenvectors[alpha]
                if (eval >= 0.) and (self.bin_multiplicity_total_[bin] > 0.):
                    self.Flow_[bin][alpha] = np.sqrt(eval) * evec[bin] * self.number_events_ / self.bin_multiplicity_total_[bin]
                else:
                    self.Flow_[bin][alpha] = None

    def __compute_uncertainty(self,bins):
        """
        Compute the uncertainty of the anisotropic flow for each bin and 
        order alpha.

        Parameters
        ----------
        bins : list or np.ndarray
            Bins used for the flow calculation.

        Returns
        -------
        None
        """
        number_bins = len(bins)-1
        self.FlowUncertainty_ = np.zeros((number_bins,self.alpha_))
        
        # sum the squared deviation from the mean
        for sub in range(self.number_subcalc_):
            for bin in range(number_bins):
                for alpha in range(self.alpha_):
                    self.FlowUncertainty_[bin][alpha] += (self.FlowSubCalc_[sub][bin][alpha]-self.Flow_[bin][alpha])**2. / (self.number_subcalc_-1)

        # take the sqrt to obtain the standard deviation
        for bin in range(number_bins):
            for alpha in range(self.alpha_):
                self.FlowUncertainty_[bin][alpha] = np.sqrt(self.FlowUncertainty_[bin][alpha])

        # Conservative ansatz: If the error cannot be computed 
        # (e.g., negative eigenvalue), set the flow magnitude to NaN
        for bin in range(number_bins):
            for alpha in range(self.alpha_):
                if np.isnan(self.FlowUncertainty_[bin][alpha]):
                    self.Flow_[bin][alpha] = np.nan

    def __compute_factorization_breaking(self,bins):
        """
        Compute the factorization breaking using the Pearson correlation 
        coefficient Eq. (12), Ref. [1].

        Parameters
        ----------
        bins : list or np.ndarray
            Bins used for the flow calculation.

        Returns
        -------
        None
        """
        # compute the Pearson coefficient Eq. (12), Ref. [1]
        number_bins = len(bins)-1
        r = np.zeros((number_bins,number_bins))
        for a in range(number_bins):
            for b in range(number_bins):
                r[a][b] = self.VnDelta_total_[a][b] / np.sqrt(self.VnDelta_total_[a][a]*self.VnDelta_total_[b][b])

        self.Pearson_r_ = r

        # do this for each sub calculation
        r_sub = np.zeros((number_bins,number_bins,self.number_subcalc_))
        for sub in range(self.number_subcalc_):
            for a in range(number_bins):
                for b in range(number_bins):
                    r_sub[a][b][sub] = self.SigmaVnDelta_total_[a][b][sub] / np.sqrt(self.SigmaVnDelta_total_[a][a][sub]*self.SigmaVnDelta_total_[b][b][sub])

        r_err = np.zeros((number_bins,number_bins))
        # sum the squared deviation from the mean
        for sub in range(self.number_subcalc_):
            for a in range(number_bins):
                for b in range(number_bins):
                    r_err[a][b] += (r_sub[a][b][sub]-r[a][b])**2. / (self.number_subcalc_-1)

        # take the sqrt to obtain the standard deviation
        for a in range(number_bins):
            for b in range(number_bins):
                r_err[a][b] = np.sqrt(r_err[a][b])

        self.Pearson_r_uncertainty_ = r_err

    def differential_flow(self, particle_data, bins, flow_as_function_of):
        """
        Compute the differential flow.

        Parameters
        ----------
        particle_data : list
            List of particle data for multiple events.
        bins : list or np.ndarray
            Bins used for the differential flow calculation.
        flow_as_function_of : str
            Variable on which the flow is calculated ("pt", "rapidity" 
            or "pseudorapidity").

        Returns
        -------
        list of numpy.ndarrays
            A list containing the differential flow and its uncertainty. 
            Each array in the list corresponds:

            - anisotropic flow of order alpha (float): Differential flow magnitude for the bin.
            - anisotropic flow error (float): Error on the differential flow magnitude for the bin.
        
        Notes
        -----
        - The flow or the uncertainty can be accessed by: `function_return[bin][alpha]`
        - If a bin has no events or the uncertainty could not be computed, the corresponding element in the result list is set to `np.nan`.
        """
        if not isinstance(bins, (list,np.ndarray)):
            raise TypeError('bins has to be list or np.ndarray')
        if not isinstance(flow_as_function_of, str):
            raise TypeError('flow_as_function_of is not a string')
        if flow_as_function_of not in ["pt","rapidity","pseudorapidity"]:
            raise ValueError("flow_as_function_of must be either 'pt', 'rapidity', 'pseudorapidity'")

        self.__compute_normalization(bins)
        self.number_events_ = len(particle_data)

        for event in range(self.number_events_):
            self.__update_event(particle_data[event],bins,flow_as_function_of,event)

        self.__compute_flow_PCA(bins)
        self.__compute_uncertainty(bins)
        self.__compute_factorization_breaking(bins)

        return [self.Flow_, self.FlowUncertainty_]
    
    def Pearson_correlation(self):
        """
        Retrieve the Pearson correlation coefficient and its uncertainty
        (Eq. (12), Ref. [1]).

        Returns
        -------
        list
            A list containing the Pearson correlation coefficient and its uncertainty.

            - Pearson_r (numpy.ndarray): The Pearson correlation coefficient for factorization breaking between each pair of bins.
            - Pearson_r_uncertainty (numpy.ndarray): The uncertainty of the Pearson correlation coefficient for factorization breaking between each pair of bins.

        Notes
        -----
        The values in the list can be accessed by name[bin1][bin2].
        """
        return [self.Pearson_r_, self.Pearson_r_uncertainty_]