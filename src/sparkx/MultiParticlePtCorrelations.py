# ===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import numpy as np
from sparkx.Jackknife import Jackknife
from typing import List, Tuple, Union, Any
from sparkx.Particle import Particle


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

    Attributes
    ----------
    mean_pt_correlation : np.ndarray, optional
        Mean transverse momentum correlations for each order up to `max_order`.
    mean_pt_correlation_error : np.ndarray, optional
        Error estimates (if computed) for mean transverse momentum correlations.
    kappa : np.ndarray, optional
        Mean transverse momentum cumulants for each order up to `max_order`.
    kappa_error : np.ndarray, optional
        Error estimates (if computed) for mean transverse momentum cumulants.
    N_events : list
        List to store numerators (N) of correlations for each event.
    D_events : list
        List to store denominators (D) of correlations for each event.

    Methods
    -------
    mean_pt_correlations:
        Computes the mean transverse momentum correlations for each order k
        across all events.

    mean_pt_cumulants:
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
        >>> mean_pt_correlations = corr_obj.mean_pt_correlations(particle_list_all_events)
        >>> print(mean_pt_correlations)
        >>>
        >>> # Compute mean transverse momentum cumulants
        >>> mean_pt_cumulants = corr_obj.mean_pt_cumulants(particle_list_all_events)
        >>> print(mean_pt_cumulants)
    """

    def __init__(self, max_order: int) -> None:
        self.max_order = max_order
        # Check if max_order is an integer
        if not isinstance(self.max_order, int):
            raise TypeError("max_order must be an integer")
        # Check that max_order is greater than 0 and less than 9
        if self.max_order < 1 or self.max_order > 8:
            raise ValueError("max_order must be greater than 0 and less than 9")

        self.mean_pt_correlation: Union[np.ndarray, None] = None
        self.mean_pt_correlation_error: Union[np.ndarray, None] = None
        self.kappa: Union[np.ndarray, None] = None
        self.kappa_error: Union[np.ndarray, None] = None
        self.N_events: Any = None
        self.D_events: Any = None

    def _P_W_k(
        self, particle_list_event: List[Particle]
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            for k in range(self.max_order):
                # if particle.weight is np.nan, then set it to 1
                if np.isnan(particle.weight):
                    particle.weight = 1.0
                Pk[k] += (particle.weight * particle.pT_abs()) ** (k + 1)
                Wk[k] += particle.weight ** (k + 1)
        return (Pk, Wk)

    def _transverse_momentum_correlations_event_num_denom(
        self, particle_list_event: List[Particle]
    ) -> None:
        """
        Compute the transverse momentum correlations for a single event.

        Computes the numerators and denominators of Eqs. A1-A7 in Ref. [1]
        separately.

        Parameters
        ----------
        particle_list_event : list
            List of particle objects in a single event.

        Returns
        -------
        None
        """
        Pk, Wk = self._P_W_k(particle_list_event)

        N = np.zeros(self.max_order)
        D = np.zeros(self.max_order)
        for order in range(self.max_order):
            if order == 0:  # k = 1
                N[order] = Pk[order]
                D[order] = Wk[order]
            elif order == 1:  # k = 2
                N[order] = Pk[0] ** 2.0 - Pk[1]
                D[order] = Wk[0] ** 2.0 - Wk[1]
            elif order == 2:  # k = 3
                N[order] = Pk[0] ** 3.0 - 3.0 * Pk[1] * Pk[0] + 2.0 * Pk[2]
                D[order] = Wk[0] ** 3.0 - 3.0 * Wk[1] * Wk[0] + 2.0 * Wk[2]
            elif order == 3:  # k = 4
                N[order] = (
                    Pk[0] ** 4.0
                    - 6.0 * Pk[1] ** 2.0 * Pk[0]
                    + 3.0 * Pk[1] ** 2.0
                    + 8.0 * Pk[2] * Pk[0]
                    - 6.0 * Pk[3]
                )
                D[order] = (
                    Wk[0] ** 4.0
                    - 6.0 * Wk[1] ** 2.0 * Wk[0]
                    + 3.0 * Wk[1] ** 2.0
                    + 8.0 * Wk[2] * Wk[0]
                    - 6.0 * Wk[3]
                )
            elif order == 4:  # k = 5
                N[order] = (
                    Pk[0] ** 5.0
                    - 10.0 * Pk[1] * Pk[0] ** 3.0
                    + 15.0 * Pk[1] ** 2.0 * Pk[0]
                    + 20.0 * Pk[2] * Pk[0] ** 2.0
                    - 20.0 * Pk[2] * Pk[1]
                    - 30.0 * Pk[3] * Pk[0]
                    + 24.0 * Pk[4]
                )
                D[order] = (
                    Wk[0] ** 5.0
                    - 10.0 * Wk[1] * Wk[0] ** 3.0
                    + 15.0 * Wk[1] ** 2.0 * Wk[0]
                    + 20.0 * Wk[2] * Wk[0] ** 2.0
                    - 20.0 * Wk[2] * Wk[1]
                    - 30.0 * Wk[3] * Wk[0]
                    + 24.0 * Wk[4]
                )
            elif order == 5:  # k = 6
                N[order] = (
                    Pk[0] ** 6.0
                    - 15.0 * Pk[1] * Pk[0] ** 4.0
                    + 45.0 * Pk[0] ** 2.0 * Pk[1] ** 2.0
                    - 15.0 * Pk[1] ** 3.0
                    - 40.0 * Pk[2] * Pk[0] ** 3.0
                    - 120.0 * Pk[2] * Pk[1] * Pk[0]
                    + 40.0 * Pk[2] ** 2.0
                    - 90.0 * Pk[3] * Pk[0] ** 2.0
                    + 90.0 * Pk[3] * Pk[1]
                    + 144.0 * Pk[4] * Pk[0]
                    - 120.0 * Pk[5]
                )
                D[order] = (
                    Wk[0] ** 6.0
                    - 15.0 * Wk[1] * Wk[0] ** 4.0
                    + 45.0 * Wk[0] ** 2.0 * Wk[1] ** 2.0
                    - 15.0 * Wk[1] ** 3.0
                    - 40.0 * Wk[2] * Wk[0] ** 3.0
                    - 120.0 * Wk[2] * Wk[1] * Wk[0]
                    + 40.0 * Wk[2] ** 2.0
                    - 90.0 * Wk[3] * Wk[0] ** 2.0
                    + 90.0 * Wk[3] * Wk[1]
                    + 144.0 * Wk[4] * Wk[0]
                    - 120.0 * Wk[5]
                )
            elif order == 6:  # k = 7
                N[order] = (
                    Pk[0] ** 7.0
                    - 21.0 * Pk[1] * Pk[0] ** 5.0
                    + 105.0 * Pk[0] ** 3.0 * Pk[1] ** 2.0
                    - 105.0 * Pk[1] ** 3.0 * Pk[0]
                    + 70.0 * Pk[2] * Pk[0] ** 4.0
                    - 420.0 * Pk[2] * Pk[1] * Pk[0] ** 2.0
                    + 210.0 * Pk[2] * Pk[1] ** 2.0
                    + 280.0 * Pk[2] ** 2.0 * Pk[0]
                    - 210.0 * Pk[3] * Pk[0] ** 3.0
                    - 630.0 * Pk[3] * Pk[1] * Pk[0]
                    - 420.0 * Pk[3] * Pk[2]
                    + 504.0 * Pk[4] * Pk[0] ** 2.0
                    - 504.0 * Pk[4] * Pk[1]
                    - 840.0 * Pk[5] * Pk[0]
                    + 720.0 * Pk[6]
                )
                D[order] = (
                    Wk[0] ** 7.0
                    - 21.0 * Wk[1] * Wk[0] ** 5.0
                    + 105.0 * Wk[0] ** 3.0 * Wk[1] ** 2.0
                    - 105.0 * Wk[1] ** 3.0 * Wk[0]
                    + 70.0 * Wk[2] * Wk[0] ** 4.0
                    - 420.0 * Wk[2] * Wk[1] * Wk[0] ** 2.0
                    + 210.0 * Wk[2] * Wk[1] ** 2.0
                    + 280.0 * Wk[2] ** 2.0 * Wk[0]
                    - 210.0 * Wk[3] * Wk[0] ** 3.0
                    - 630.0 * Wk[3] * Wk[1] * Wk[0]
                    - 420.0 * Wk[3] * Wk[2]
                    + 504.0 * Wk[4] * Wk[0] ** 2.0
                    - 504.0 * Wk[4] * Wk[1]
                    - 840.0 * Wk[5] * Wk[0]
                    + 720.0 * Wk[6]
                )
            elif order == 7:  # k = 8
                N[order] = (
                    Pk[0] ** 8.0
                    - 28.0 * Pk[1] * Pk[0] ** 6.0
                    - 210.0 * Pk[1] ** 2.0 * Pk[0] ** 4.0
                    - 420.0 * Pk[1] ** 3.0 * Pk[0] ** 2.0
                    + 105.0 * Pk[1] ** 4.0
                    + 112.0 * Pk[2] * Pk[0] ** 5.0
                    + 1120.0 * Pk[2] * Pk[1] * Pk[0] ** 3.0
                    + 1680.0 * Pk[2] * Pk[1] ** 2.0 * Pk[0]
                    + 1120.0 * Pk[2] ** 2.0 * Pk[0] ** 2.0
                    + 1120.0 * Pk[2] ** 2.0 * Pk[1]
                    - 420.0 * Pk[3] * Pk[0] ** 4.0
                    + 2520.0 * Pk[3] * Pk[1] * Pk[0] ** 2.0
                    - 1260.0 * Pk[3] * Pk[1] ** 2.0
                    - 3360.0 * Pk[3] * Pk[2] * Pk[0]
                    + 1260.0 * Pk[4] ** 2.0
                    + 1344.0 * Pk[4] * Pk[0] ** 3.0
                    - 4032.0 * Pk[4] * Pk[1] * Pk[0]
                    + 2688.0 * Pk[4] * Pk[2]
                    - 3360.0 * Pk[5] * Pk[0] ** 2.0
                    + 3360.0 * Pk[5] * Pk[1]
                    + 5760.0 * Pk[6] * Pk[0]
                    - 5040.0 * Pk[7]
                )
                D[order] = (
                    Wk[0] ** 8.0
                    - 28.0 * Wk[1] * Wk[0] ** 6.0
                    - 210.0 * Wk[1] ** 2.0 * Wk[0] ** 4.0
                    - 420.0 * Wk[1] ** 3.0 * Wk[0] ** 2.0
                    + 105.0 * Wk[1] ** 4.0
                    + 112.0 * Wk[2] * Wk[0] ** 5.0
                    + 1120.0 * Wk[2] * Wk[1] * Wk[0] ** 3.0
                    + 1680.0 * Wk[2] * Wk[1] ** 2.0 * Wk[0]
                    + 1120.0 * Wk[2] ** 2.0 * Wk[0] ** 2.0
                    + 1120.0 * Wk[2] ** 2.0 * Wk[1]
                    - 420.0 * Wk[3] * Wk[0] ** 4.0
                    + 2520.0 * Wk[3] * Wk[1] * Wk[0] ** 2.0
                    - 1260.0 * Wk[3] * Wk[1] ** 2.0
                    - 3360.0 * Wk[3] * Wk[2] * Wk[0]
                    + 1260.0 * Wk[4] ** 2.0
                    + 1344.0 * Wk[4] * Wk[0] ** 3.0
                    - 4032.0 * Wk[4] * Wk[1] * Wk[0]
                    + 2688.0 * Wk[4] * Wk[2]
                    - 3360.0 * Wk[5] * Wk[0] ** 2.0
                    + 3360.0 * Wk[5] * Wk[1]
                    + 5760.0 * Wk[6] * Wk[0]
                    - 5040.0 * Wk[7]
                )

        self.N_events.append(N)
        self.D_events.append(D)

    def _compute_numerator_denominator_all_events(
        self, particle_list_all_events: List[List[Particle]]
    ) -> None:
        """
        Compute numerators and denominators of correlations for all events.

        Parameters
        ----------
        particle_list_all_events : list
            List of events, where each event is a list of particle objects.
        """
        for event in particle_list_all_events:
            self._transverse_momentum_correlations_event_num_denom(event)

    def _compute_mean_pt_correlations(
        self, numerator_denominator_array: np.ndarray
    ) -> float:
        """
        Compute mean transverse momentum correlations from the array containing
        the numerator and denominator.

        Parameters
        ----------
        numerator_denominator_array : np.ndarray
            Array of shape (num_events, 2) containing numerators and denominators.

        Returns
        -------
        float
            Mean transverse momentum correlations.
        """
        sum_numerator = 0.0
        sum_denominator = 0.0
        for i in range(numerator_denominator_array.shape[0]):
            sum_numerator += numerator_denominator_array[i, 0]
            sum_denominator += numerator_denominator_array[i, 1]

        return sum_numerator / sum_denominator

    def mean_pt_correlations(
        self,
        particle_list_all_events: List[List[Particle]],
        compute_error: bool = True,
        delete_fraction: float = 0.4,
        number_samples: int = 100,
        seed: int = 42,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Computes mean transverse momentum correlations for each order up to
        max_order using Eq. [14] in Ref. [1].
        The weight is chosen to be :math:`W^{\\prime}_{m} = D\\langle m\\rangle_{p_{\\mathrm{T}}}`.


        Parameters
        ----------
        particle_list_all_events : list
            List of events, where each event is a list of particle objects.
        compute_error : bool, optional
            Whether to compute error estimates (default is True).
        delete_fraction : float, optional
            Fraction of data to delete for jackknife method (default is 0.4).
        number_samples : int, optional
            Number of jackknife samples (default is 100).
        seed : int, optional
            Random seed for reproducibility (default is 42).

        Returns
        -------
        np.ndarray or tuple
            Mean transverse momentum correlations for each order.
            If compute_error is True, returns a tuple (mean_pt_correlation, mean_pt_correlation_error).

        Raises
        ------
        TypeError
            If delete_fraction is not a float.
            If number_samples is not an integer.
            If seed is not an integer.
            If compute_error is not a boolean.
        ValueError
            If delete_fraction is not between 0 and 1.
            If number_samples is not greater than 0.
        """
        if not isinstance(delete_fraction, float):
            raise TypeError("delete_fraction must be a float")
        if not 0.0 < delete_fraction < 1.0:
            raise ValueError("delete_fraction must be between 0 and 1")
        if not isinstance(number_samples, int):
            raise TypeError("number_samples must be an integer")
        if not number_samples > 0:
            raise ValueError("number_samples must be greater than 0")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(compute_error, bool):
            raise TypeError("compute_error must be a boolean")

        self.N_events = []
        self.D_events = []
        self._compute_numerator_denominator_all_events(particle_list_all_events)
        self.N_events = np.array(self.N_events)
        self.D_events = np.array(self.D_events)

        mean_pt_correlations_store = np.zeros(self.max_order)
        mean_pt_correlations_error_store = np.zeros(self.max_order)
        for order in range(self.max_order):
            # combine the numerator and denominator for each order in an array
            numerator_denominator_array = np.array(
                [self.N_events[:, order], self.D_events[:, order]]
            ).T
            mean_pt_correlations_store[order] = self._compute_mean_pt_correlations(
                numerator_denominator_array
            )

            if compute_error:
                jackknife = Jackknife(delete_fraction, number_samples, seed)
                mean_pt_correlations_error_store[
                    order
                ] = jackknife.compute_jackknife_estimates(
                    numerator_denominator_array,
                    function=self._compute_mean_pt_correlations,
                )

        self.mean_pt_correlation = mean_pt_correlations_store
        if compute_error:
            self.mean_pt_correlation_error = mean_pt_correlations_error_store
            return (mean_pt_correlations_store, mean_pt_correlations_error_store)
        else:
            return mean_pt_correlations_store

    def _kappa_cumulant(self, C: np.ndarray, k: int) -> float:
        """
        Compute cumulant kappa from correlations C up to order k
        (Eqns. B9-B16 in [1]).

        Parameters
        ----------
        C : np.ndarray
            Array of correlations up to order k.
        k : int
            Order of cumulant to compute.

        Returns
        -------
        float
            Cumulant kappa for the given order k.
        """
        if k == 1:
            kappa = C[0]
        elif k == 2:
            kappa = C[1] - C[0] ** 2
        elif k == 3:
            kappa = C[2] - 3.0 * C[1] * C[0] + 2.0 * C[0] ** 3
        elif k == 4:
            kappa = (
                C[3]
                - 4.0 * C[2] * C[0]
                - 3.0 * C[1] ** 2
                + 12.0 * C[1] * C[0] ** 2
                - 6.0 * C[0] ** 4
            )
        elif k == 5:
            kappa = (
                C[4]
                - 5.0 * C[3] * C[0]
                - 10.0 * C[2] * C[1]
                + 30.0 * C[1] ** 2 * C[0]
                + 20.0 * C[2] * C[0] ** 2
                - 60.0 * C[1] * C[0] ** 3
                + 24.0 * C[0] ** 5
            )
        elif k == 6:
            kappa = (
                C[5]
                - 6.0 * C[4] * C[0]
                - 15.0 * C[3] * C[1]
                - 10.0 * C[2] ** 2
                + 30.0 * C[1] ** 3
                + 30.0 * C[3] * C[0] ** 2
                + 120.0 * C[2] * C[1] * C[0]
                - 270.0 * C[1] ** 2 * C[0] ** 2
                - 120.0 * C[2] * C[0] ** 3
                + 360.0 * C[1] * C[0] ** 4
                - 120.0 * C[0] ** 6
            )
        elif k == 7:
            kappa = (
                C[6]
                - 7.0 * C[5] * C[0]
                - 21.0 * C[4] * C[1]
                + 42.0 * C[4] * C[0] ** 2
                - 35.0 * C[3] * C[2]
                + 210.0 * C[3] * C[1] * C[0]
                - 210.0 * C[3] * C[0] ** 3
                + 140.0 * C[2] ** 2 * C[0]
                + 210.0 * C[2] * C[1] ** 2
                - 1260.0 * C[2] * C[1] * C[0] ** 2
                + 840.0 * C[2] * C[0] ** 4
                - 630.0 * C[1] ** 3 * C[0]
                + 2520.0 * C[1] ** 2 * C[0] ** 3
                - 2520.0 * C[1] * C[0] ** 5
                + 720.0 * C[0] ** 7
            )
        elif k == 8:
            kappa = (
                C[7]
                - 8.0 * C[6] * C[0]
                - 28.0 * C[5] * C[1]
                + 56.0 * C[5] * C[0] ** 2
                - 56.0 * C[4] * C[2]
                + 336.0 * C[4] * C[1] * C[0]
                - 336.0 * C[4] * C[0] ** 3
                - 35.0 * C[3] ** 2
                + 560.0 * C[3] * C[2] * C[0]
                + 420.0 * C[1] ** 2 * C[3]
                - 2520.0 * C[3] * C[1] * C[0] ** 2
                + 1680.0 * C[3] * C[0] ** 4
                + 560.0 * C[2] ** 2 * C[1]
                - 1680.0 * C[2] ** 2 * C[0] ** 2
                - 5040.0 * C[2] * C[1] ** 2 * C[0]
                + 13440.0 * C[2] * C[1] * C[0] ** 3
                - 6720.0 * C[2] * C[0] ** 5
                - 630.0 * C[1] ** 4
                + 10080.0 * C[1] ** 3 * C[0] ** 2
                - 25200.0 * C[1] ** 2 * C[0] ** 4
                + 20160.0 * C[1] * C[0] ** 6
                - 5040.0 * C[0] ** 8
            )
        else:
            raise ValueError("Invalid order k for cumulant calculation")
        return kappa

    def _compute_mean_pt_cumulants(self, data: np.ndarray, k: int) -> float:
        """
        Compute mean transverse momentum cumulants from data at order k.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (num_events, 2*(k+1)) containing numerators and
            denominators up to order k.
        k : int
            Order of cumulant to compute.

        Returns
        -------
        float
            Mean transverse momentum cumulant for order k.
        """
        kappa_store = 0.0
        C = np.zeros(k + 1)
        for order in range(k + 1):
            sum_numerator_tmp = 0.0
            sum_denominator_tmp = 0.0
            for i in range(data.shape[0]):
                sum_numerator_tmp += data[i, 2 * order]
                sum_denominator_tmp += data[i, 2 * order + 1]
            C_order = sum_numerator_tmp / sum_denominator_tmp
            C[order] = C_order
        kappa_store = self._kappa_cumulant(C, k + 1)
        return kappa_store

    def mean_pt_cumulants(
        self,
        particle_list_all_events: List[List[Particle]],
        compute_error: bool = True,
        delete_fraction: float = 0.4,
        number_samples: int = 100,
        seed: int = 42,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Computes the mean transverse momentum cumulants for each order k
        from Eqs. B9-B16 in Ref. [1].

        Parameters
        ----------
        particle_list_all_events : list
            List of events, where each event is a list of particle objects.
        compute_error : bool, optional
            Whether to compute error estimates (default is True).
        delete_fraction : float, optional
            Fraction of data to delete for jackknife method (default is 0.4).
        number_samples : int, optional
            Number of jackknife samples (default is 100).
        seed : int, optional
            Random seed for reproducibility (default is 42).

        Returns
        -------
        np.ndarray or tuple
            Mean transverse momentum cumulants for each order.
            If compute_error is True, returns a tuple (kappa, kappa_error).

        Raises
        ------
        TypeError
            If delete_fraction is not a float.
            If number_samples is not an integer.
            If seed is not an integer.
            If compute_error is not a boolean.
        ValueError
            If delete_fraction is not between 0 and 1.
            If number_samples is not greater than 0.
        """
        if not isinstance(delete_fraction, float):
            raise TypeError("delete_fraction must be a float")
        if not 0.0 < delete_fraction < 1.0:
            raise ValueError("delete_fraction must be between 0 and 1")
        if not isinstance(number_samples, int):
            raise TypeError("number_samples must be an integer")
        if not number_samples > 0:
            raise ValueError("number_samples must be greater than 0")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(compute_error, bool):
            raise TypeError("compute_error must be a boolean")

        self.N_events = []
        self.D_events = []
        self._compute_numerator_denominator_all_events(particle_list_all_events)
        self.N_events = np.array(self.N_events)
        self.D_events = np.array(self.D_events)

        kappa_store = np.zeros(self.max_order)
        kappa_store_error = np.zeros(self.max_order)
        for order in range(self.max_order):
            dataN = self.N_events[:, : 2 * (order + 1)]
            dataD = self.D_events[:, : 2 * (order + 1)]
            # combine the numerator and denominator for each order in data
            # always alternate the columns of the numerator and denominator
            combined_data = np.empty(
                (dataN.shape[0], dataN.shape[1] + dataD.shape[1]), dtype=dataN.dtype
            )
            combined_data[:, 0::2] = dataN
            combined_data[:, 1::2] = dataD
            kappa_store[order] = self._compute_mean_pt_cumulants(combined_data, order)

            if compute_error:
                jackknife = Jackknife(delete_fraction, number_samples, seed)
                kappa_store_error[order] = jackknife.compute_jackknife_estimates(
                    combined_data, function=self._compute_mean_pt_cumulants, k=order
                )

        self.kappa = kappa_store
        if compute_error:
            self.kappa_error = kappa_store_error
            return (kappa_store, kappa_store_error)
        else:
            return kappa_store
