# ===================================================
#
#    Copyright (c) 2024-2025
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import os
import random as rd
import numpy as np
from multiprocessing import Pool
from typing import Callable, Optional, Any


class Jackknife:
    """
    Delete-d Jackknife class for computation of confidence intervals for many different
    statistics. The Jackknife method is similar to a bootstrap method, but
    instead of replacing the data with random samples, it deletes a fraction
    of the data and computes the statistic on the reduced data.

    Literature on the method can be found in:

    - `Avery McIntosh "The Jackknife Estimation Method" arXiv:1606.00497 <https://arxiv.org/abs/1606.00497>`__.
    - J. Shao and D. Tu "The Jackknife and Bootstrap" Springer Series in Statistics, 0172-739, 1995.

    The code supports parallel processing for the computation of the Jackknife
    samples.

    If the function applied to the data needs multi-dimensional data, the data
    array is truncated along axis 0.

    .. note::
        It is the user's responsibility to ensure that the function applied to
        the data is well-defined and that the data array is in the correct
        input format for the specific function. This code always dilutes the
        data along axis 0.
        This class only checks that the function is callable and that the return
        value is a single number (int or float).

    Parameters
    ----------
    delete_fraction : float
        The fraction of the data to be randomly deleted.
    number_samples : int
        The number of Jackknife samples to be computed.
    seed : int, optional
        The random seed to be used for the Jackknife computation. The default
        is 42.

    Attributes
    ----------
    delete_fraction : float
        The fraction of the data to be randomly deleted.
    number_samples : int
        The number of Jackknife samples to be computed.
    seed : int
        The random seed to be used for the Jackknife computation.

    Methods
    -------
    compute_jackknife_estimates:
        Compute the Jackknife estimates

    Raises
    ------
    ValueError
        If :code:`delete_fraction` is less than 0 or greater than or equal to 1.
        If :code:`number_samples` is less than 1.
    TypeError
        If :code:`delete_fraction` is not a float.
        If :code:`number_samples` is not an integer.
        If :code:`seed` is not an integer.

    Examples
    --------

    A demonstration how to the Jackknife class:

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Jackknife import Jackknife
        >>>
        >>> data_Gaussian = np.random.normal(0, 1, 100)
        >>> jackknife = Jackknife(delete_fraction=0.4, number_samples=100)
        >>> jackknife.compute_jackknife_estimates(data_Gaussian, function=np.mean)
    """

    def __init__(
        self, delete_fraction: float, number_samples: int, seed: int = 42
    ) -> None:

        if not isinstance(delete_fraction, float):
            raise TypeError("delete_fraction must be a float.")
        if not isinstance(number_samples, int):
            raise TypeError("number_samples must be an integer.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        if not isinstance(delete_fraction, float):
            raise TypeError("delete_fraction must be a float.")
        if not isinstance(number_samples, int):
            raise TypeError("number_samples must be an integer.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if delete_fraction < 0.0 or delete_fraction >= 1.0:
            raise ValueError("delete_fraction must be between 0 and 1.")
        if number_samples < 1:
            raise ValueError("number_samples must be greater than 0.")

        self.delete_fraction = delete_fraction
        self.number_samples = number_samples

        self.seed = seed
        self._init_random()

    def _init_random(self) -> None:
        """
        Initialize random seed.
        """
        rd.seed(self.seed)

    def _randomly_delete_data(self, data: np.ndarray) -> np.ndarray:
        """
        Randomly delete a fraction of the data along axis 0 and return the
        remaining data.

        Parameters
        ----------
        data : np.ndarray
            The data to be randomly deleted.

        Returns
        -------
        np.ndarray
            The remaining data after randomly deleting a fraction of it.
        """
        data = data.copy()
        delete_indices = rd.sample(
            range(len(data)), int(self.delete_fraction * len(data))
        )
        data = np.delete(data, delete_indices, axis=0)
        return data

    def _apply_function_to_reduced_data(
        self,
        reduced_data: np.ndarray,
        function: Callable[..., Any],
        *args: tuple,
        **kwargs: Any,
    ) -> Any:
        """
        Apply a function to the reduced data.

        Parameters
        ----------
        reduced_data : np.ndarray
            The reduced dataset.
        function : function
            The function to be applied to the reduced data. The function has to
            accept a single argument, which is the reduced data array.
        *args : tuple
            Additional arguments to be passed to the function.
        **kwargs : dict
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        any
            The result of applying the function to the reduced data.
        """
        return function(reduced_data, *args, **kwargs)

    def _compute_one_jackknife_sample(
        self,
        data: np.ndarray,
        function: Callable[..., Any],
        *args: tuple,
        **kwargs: Any,
    ) -> Any:
        """
        Compute one Jackknife sample.

        Parameters
        ----------
        data : np.ndarray
            The data to be used to compute the Jackknife sample.
        function : function
            The function to be applied to the reduced data. The function has to
            accept a single argument, which is the reduced data array.
        *args : tuple
            Additional arguments to be passed to the function.
        **kwargs : dict
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        any
            The result of applying the function to the reduced data.
        """
        reduced_data = self._randomly_delete_data(data)
        return self._apply_function_to_reduced_data(
            reduced_data, function, *args, **kwargs
        )

    @staticmethod
    def _helper_unpack(
        instance: "Jackknife",
        index: int,
        data: np.ndarray,
        function: Callable[..., Any],
        args: tuple,
        kwargs: Any,
    ) -> Any:
        """
        Helper function to unpack the arguments for parallel processing.

        Parameters
        ----------
        instance : Jackknife
            The Jackknife instance.
        index : int
            The index of the Jackknife sample.
        data : np.ndarray
            The data to be used to compute the Jackknife sample.
        function : function
            The function to be applied to the reduced data. The function has to
            accept a single argument, which is the reduced data array.
        args : tuple
            Additional arguments to be passed to the function.
        kwargs : dict
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        any
            The result of applying _compute_one_jackknife_sample.
        """
        rd.seed(instance.seed + index)
        return instance._compute_one_jackknife_sample(
            data, function, *args, **kwargs
        )

    def _init_random_subprocess(self, seed: int) -> None:
        """
        Initialize random seed for subprocesses.

        Parameters
        ----------
        seed : int
            The random seed to be used for subprocesses.

        Returns
        -------
        None
        """
        rd.seed(seed)

    def _compute_jackknife_samples(
        self,
        data: np.ndarray,
        function: Callable[..., Any],
        num_cores: Optional[int] = None,
        *args: tuple,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute all Jackknife samples.

        Parameters
        ----------
        data : np.ndarray
            The data to be used to compute the Jackknife samples.
        function : function
            The function to be applied to the reduced data. The function has to
            accept a single argument, which is the reduced data array.
        num_cores : int, optional
            The number of cores to be used for parallel processing. The default
            is None, which means that all available cores will be used.
        *args : tuple
            Additional arguments to be passed to the function.
        **kwargs : dict
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        np.ndarray
            The results of applying the function to the reduced data.

        Raises
        ------
        ValueError
            If num_cores is less than 1.
        """
        if isinstance(num_cores, int) and num_cores < 1:
            raise ValueError("num_cores must be greater than 0.")
        if num_cores is None:
            num_cores = os.cpu_count()

        with Pool(
            num_cores,
            initializer=self._init_random_subprocess,
            initargs=(self.seed,),
        ) as pool:
            results = pool.starmap(
                self._helper_unpack,
                [
                    (self, index, data, function, args, kwargs)
                    for index in range(self.number_samples)
                ],
            )
        return np.array(results)

    def compute_jackknife_estimates(
        self,
        data: np.ndarray,
        function: Callable[..., Any] = np.mean,
        num_cores: Optional[int] = None,
        *args: tuple,
        **kwargs: Any,
    ) -> float:
        """
        Compute the Jackknife uncertainty estimates for a function applied to
        a data array. The default function is :code:`np.mean`, but it can be
        changed to any other function that accepts a numpy array as input.
        Multiple other arguments can be passed to the function as args and
        kwargs.

        Parameters
        ----------
        data : np.ndarray
            The data to be used to compute the Jackknife samples. The data is
            truncated along axis 0.
        function : function, optional
            The function to be applied to the reduced data. The function can
            accept additional arguments and keyword arguments. The default is
            np.mean. It has to return a single number (int or float).
        num_cores : int, optional
            The number of cores to be used for parallel processing. The default
            is None, which means that all available cores will be used.
        *args : tuple
            Additional arguments to be passed to the function.
        **kwargs : dict
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        float
            The Jackknife estimate for the standard deviation of the function
            applied to the data.

        Raises
        ------
        ValueError
            If :code:`delete_n_points` is less than 1 (:code:`delete_fraction` too small).
        TypeError
            If data is not a numpy array or if function is not callable.
        TypeError
            If the function does not return a single number (float or int).
        """
        delete_n_points = int(self.delete_fraction * len(data))
        if delete_n_points < 1:
            raise ValueError("The delete_fraction is too small.")
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array.")
        if not callable(function):
            raise TypeError("function must be a callable object.")

        # Check if the function returns a single number
        test_result = function(
            data[: max(1, len(data) // 100)], *args, **kwargs
        )
        if not isinstance(test_result, (int, float)):
            raise TypeError("function must return a single number.")

        jackknife_samples = self._compute_jackknife_samples(
            data, function, num_cores, *args, **kwargs
        )

        # mean of jackknife samples
        mean_samples = np.mean(jackknife_samples)
        # variance of the jackknife samples
        variance_samples = 0.0
        for i in range(len(jackknife_samples)):
            variance_samples += (jackknife_samples[i] - mean_samples) ** 2.0

        if delete_n_points == 1:
            variance_samples *= (len(jackknife_samples) - 1) / len(
                jackknife_samples
            )
        else:
            variance_samples *= (len(data) - delete_n_points) / (
                delete_n_points * len(jackknife_samples)
            )

        return np.sqrt(variance_samples)
