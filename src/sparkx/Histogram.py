# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import numpy as np
import csv
import warnings
from typing import Optional, Union, List, Tuple, Dict


class Histogram:
    """
    Defines a histogram object.

    The histograms can be initialized either with a tuple
    :code:`(hist_min,hist_max,num_bins)` or a list/numpy.ndarray containing the bin
    boundaries, which allows for different bin widths.
    Multiple histograms can be added and averaged.

    .. note::
        It may be important to keep in mind that each bin contains it left edge
        but not the right edge. If a value is added to a histogram the bin
        assignment therefore follows:

            .. raw:: html

                 <p><code style="color:red; background-color:transparent;
                 font-size: 0.9em;">bins[i-1] &lt;= value &lt; bins[i]</code></p>

    Parameters
    ----------
    bin_boundaries : tuple
        A tuple with three values :code:`(hist_min, hist_max, num_bins)`. This
        creates an evenly sized histogram in the range :code:`[hist_min, hist_max]`
        divided into :code:`num_bins` bins.
    bin_boundaries : list/numpy.ndarray
        A list or numpy array that contains all bin edges. This allows for non-
        uniform bin sizes.

    Attributes
    ----------
    number_of_bins_: int
        Number of histogram bins.
    bin_edges_: numpy.ndarray
        Array containing the edges of the bins.
    number_of_histograms_: int
        Number of created histograms.
    histograms_: numpy.ndarray
        Array containing the histograms (might be scaled).
    histograms_raw_count_: numpy.ndarray
        Array containing the raw counts of the histograms.
    error_: numpy.ndarray
        Array containing the histogram error.
    scaling_: numpy.ndarray
        Array containing scaling factors for each bin.

    Methods
    -------
    histogram:
        Get the created histogram(s)
    histogram_raw_counts:
        Get the raw bin counts of the histogram(s).
    number_of_histograms:
        Get the number of current histograms.
    bin_centers: numpy.ndarray
        Get the bin centers.
    bin_width: numpy.ndarray
        Get the bin widths.
    bin_bounds_left: numpy.ndarray
        Extract the lower bounds of the individual bins.
    bin_bounds_right: numpy.ndarray
        Extract the upper bounds of the individual bins.
    bin_boundaries: numpy.ndarray
        Get the bin boundaries.
    add_value:
        Add one or multiple values to the latest histogram.
    add_histogram:
        Add a new histogram.
    make_density:
        Create probability density from last histogram.
    average:
        Averages over all histograms.
    average_weighted:
        Performs a weighted average over all histograms.
    standard_error:
        Get the standard deviation for each bin.
    statistical_error:
        Statistical error of all histogram bins in all histograms.
    set_systematic_error:
        Sets the systematic histogram error by hand.
    scale_histogram:
        Multiply latest histogram with a factor.
    set_error:
        Set the error for the histogram by hand.
    print_histogram:
        Print the histogram(s) to the terminal.
    write_to_file:
        Write one histogram to a csv file.

    Examples
    --------
    **Creating histograms**

    To create multiple histograms with the same number and size of bins we can
    initialize the Histogram object with a tuple (otherwise initialize with a
    list containing all bin edges). To fill these histograms store them in a
    variable, the methods ``add_value(value)``, ``add_histogram()`` and
    ``histogram()`` can be used.

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Histogram import Histogram
        >>>
        >>> # Initialize a histogram in the range [0,10] with 10 bins
        >>> histObj = Histogram((0, 10, 10))
        >>>
        >>> # Add the values [1, 1.2, 3, 5, 6.5, 9, 9] to the histogram
        >>> histObj.add_value([1, 1.2, 3, 5, 6.5, 9, 9])
        >>> print(histObj.histogram())
        [0. 2. 0. 1. 0. 1. 1. 0. 0. 2.]
        >>>
        >>> # Add a second histogram and add the values [2, 7, 7.2, 9]
        >>> histObj.add_histogram()
        >>> histObj.add_value([2, 7, 7.2, 9])
        >>> print(histObj.histogram())
        [[0. 2. 0. 1. 0. 1. 1. 0. 0. 2.]
         [0. 0. 1. 0. 0. 0. 0. 2. 0. 1.]]
        >>>
        >>> # Store the histograms in hist as numpy.ndarray
        >>> hist = histObj.histogram()
    """

    def __init__(
        self,
        bin_boundaries: Union[
            Tuple[float, float, int], List[float], np.ndarray
        ],
    ) -> None:
        self.number_of_bins_: Optional[int] = None
        self.bin_edges_: Optional[np.ndarray] = None
        self.number_of_histograms_: int = 1
        self.histograms_: Optional[np.ndarray] = None
        self.histograms_raw_count_: Optional[np.ndarray] = None
        self.error_: Optional[np.ndarray] = None
        self.scaling_: Optional[np.ndarray] = None
        self.systematic_error_: Optional[np.ndarray] = None

        if isinstance(bin_boundaries, tuple) and len(bin_boundaries) == 3:
            hist_min = bin_boundaries[0]
            hist_max = bin_boundaries[1]
            num_bins = bin_boundaries[2]

            if hist_min > hist_max or hist_min == hist_max:
                raise ValueError("hist_min must be smaller than hist_max")

            elif not isinstance(num_bins, int) or num_bins <= 0:
                raise ValueError("Number of bins must be a positive integer")

            self.number_of_bins_ = num_bins
            self.bin_edges_ = np.linspace(hist_min, hist_max, num=num_bins + 1)
            self.histograms_ = np.asarray([np.zeros(num_bins)])
            self.histograms_raw_count_ = np.asarray([np.zeros(num_bins)])
            self.scaling_ = np.asarray([np.ones(num_bins)])
            self.error_ = np.asarray([np.zeros(num_bins)])
            self.systematic_error_ = np.asarray([np.zeros(num_bins)])

        elif isinstance(bin_boundaries, (list, np.ndarray)):
            self.number_of_bins_ = len(bin_boundaries) - 1
            self.bin_edges_ = np.asarray(bin_boundaries)
            self.histograms_ = np.asarray([np.zeros(self.number_of_bins_)])
            self.histograms_raw_count_ = np.asarray(
                [np.zeros(self.number_of_bins_)]
            )
            self.scaling_ = np.asarray([np.ones(self.number_of_bins_)])
            self.error_ = np.asarray([np.zeros(self.number_of_bins_)])
            self.systematic_error_ = np.asarray(
                [np.zeros(self.number_of_bins_)]
            )

        else:
            raise TypeError(
                "Input must be a tuple (hist_min, hist_max, num_bins) "
                + "or a list/numpy.ndarray containing the bin edges!"
            )

    def histogram(self) -> np.ndarray:
        """
        Get the current histogram(s).

        Returns
        -------
        `histograms_`: numpy.ndarray
            Array containing the histogram(s).
        """
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'histogram' function."
            )

        return self.histograms_

    def histogram_raw_counts(self) -> np.ndarray:
        """
        Get the raw bin counts of the histogram(s), even after the original
        histograms are scaled or averaged. If weights are used, then they
        affect the raw counts histogram.

        Returns
        -------
        `histograms_raw_count_`: numpy.ndarray
            Array containing the raw counts of the histogram(s)
        """
        if self.histograms_raw_count_ is None:
            raise TypeError(
                "'histograms_raw_count_' is None. It must be initialized before calling the 'histogram_raw_counts' function."
            )

        return self.histograms_raw_count_

    def number_of_histograms(self) -> int:
        """
        Get the number of current histograms.

        Returns
        -------
        `number_of_histograms_`: int
            Number of histograms.
        """
        return self.number_of_histograms_

    def bin_centers(self) -> np.ndarray:
        """
        Get the bin centers.

        Returns
        -------
        numpy.ndarray
            Array containing the bin centers.
        """
        if self.bin_edges_ is None:
            raise TypeError(
                "'bin_edges_' is None. It must be initialized before calling the 'bin_centers' function."
            )

        return (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2.0

    def bin_width(self) -> np.ndarray:
        """
        Get the bin widths.

        Returns
        -------
        numpy.ndarray
            Array containing the bin widths.
        """
        if self.bin_edges_ is None:
            raise TypeError(
                "'bin_edges_' is None. It must be initialized before calling the 'bin_width' function."
            )

        return self.bin_edges_[1:] - self.bin_edges_[:-1]

    def bin_bounds_left(self) -> np.ndarray:
        """
        Extract the lower bounds of the individual bins.

        Returns
        -------
        numpy.ndarray
            Array containing the lower bin boundaries.
        """
        if self.bin_edges_ is None:
            raise TypeError(
                "'bin_edges_' is None. It must be initialized before calling the 'bin_bounds_left' function."
            )

        return self.bin_edges_[:-1]

    def bin_bounds_right(self) -> np.ndarray:
        """
        Extract the upper bounds of the individual bins.

        Returns
        -------
        numpy.ndarray
            Array containing the upper bin boundaries.
        """
        if self.bin_edges_ is None:
            raise TypeError(
                "'bin_edges_' is None. It must be initialized before calling the 'bin_bounds_right' function."
            )

        return self.bin_edges_[1:]

    def bin_boundaries(self) -> np.ndarray:
        """
        Get the bin boundaries.

        Returns
        -------
        numpy.ndarray
            Array containing the bin boundaries.
        """
        if self.bin_edges_ is None:
            raise TypeError(
                "'bin_edges_' is None. It must be initialized before calling the 'bin_boundaries' function."
            )

        return self.bin_edges_

    def remove_bin(self, index: int) -> "Histogram":
        """
        Remove a bin from all histograms, starting from the 0th bin.

        The numbering of of all following bins is reduced by one.

        Raises
        ------
        TypeError
            If `index` is not an integer or `bin_edge` is not a float.

        ValueError
            If `index` is out of range.
        """
        if self.number_of_bins_ is None:
            raise TypeError(
                "'number_of_bins_' is None. It must be initialized before calling the'remove_bin' function."
            )
        if self.bin_edges_ is None:
            raise TypeError(
                "'bin_edges_' is None. It must be initialized before calling the 'remove_bin' function."
            )
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'remove_bin' function."
            )
        if self.error_ is None:
            raise TypeError(
                "'error_' is None. It must be initialized before calling the 'remove_bin' function."
            )
        if self.histograms_raw_count_ is None:
            raise TypeError(
                "'histograms_raw_count_' is None. It must be initialized before calling the 'remove_bin' function."
            )
        if self.systematic_error_ is None:
            raise TypeError(
                "'systematic_error_' is None. It must be initialized before calling the 'remove_bin' function."
            )

        if isinstance(index, (int)):
            if np.isnan(index):
                raise ValueError("Bin number in remove_bin is NaN.")

            if index < 0 or index >= len(self.bin_edges_):
                raise ValueError("Bin number in remove_bin is out of range.")
        else:
            raise TypeError("Bin number in remove_bin must be an integer.")

        self.number_of_bins_ -= 1
        self.bin_edges_ = np.delete(self.bin_edges_, index)

        self.histograms_ = np.asarray(
            [np.delete(hist, index) for hist in self.histograms_]
        )
        self.error_ = np.asarray([np.delete(err, index) for err in self.error_])
        self.histograms_raw_count_ = np.asarray(
            [
                np.delete(raw_count, index)
                for raw_count in self.histograms_raw_count_
            ]
        )
        self.systematic_error_ = np.asarray(
            [np.delete(sys_err, index) for sys_err in self.systematic_error_]
        )

        return self

    def add_bin(self, index: int, bin_edge: float) -> "Histogram":
        """
        Add a bin to all histograms at the specified index.
        Attention: If values were added to bins before inserting a new bin, the information about its value is lost.
        That means that if a value of 5.5 was added to a bin from 5 to 6, and afterwards a new bin from 5.4 to 6. is added, the value will still remain in the old bin, which goes now from 5 to 5.4.

        Parameters
        ----------
        index : int
            The position where the new bin should be inserted.

        bin_edge : float
            The lower edge value of the new bin.

        Raises
        ------
        TypeError
            If `index` is not an integer or `bin_edge` is not a float.

        ValueError
            If `index` is out of range or the bin edges are not monotonically increasing.
        """
        if not isinstance(index, int):
            raise TypeError("Index in add_bin must be an integer.")
        if not isinstance(bin_edge, (int, float)):
            raise TypeError("Bin edge in add_bin must be a float or int.")
        if self.bin_edges_ is None:
            raise TypeError(
                "'bin_edges_' is None. It must be initialized before calling the 'add_bin' function."
            )
        if self.number_of_bins_ is None:
            raise TypeError(
                "'number_of_bins_' is None. It must be initialized before calling the 'add_bin' function."
            )
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'add_bin' function."
            )
        if self.error_ is None:
            raise TypeError(
                "'error_' is None. It must be initialized before calling the 'add_bin' function."
            )
        if self.histograms_raw_count_ is None:
            raise TypeError(
                "'histograms_raw_count_' is None. It must be initialized before calling the 'add_bin' function."
            )
        if self.systematic_error_ is None:
            raise TypeError(
                "'systematic_error_' is None. It must be initialized before calling the 'add_bin' function."
            )

        if index < 0 or index >= len(self.bin_edges_):
            raise ValueError("Index in add_bin is out of range.")
        if index > 0 and bin_edge <= self.bin_edges_[index - 1]:
            raise ValueError("Bin edges must be monotonically increasing.")
        if index < len(self.bin_edges_) and bin_edge >= self.bin_edges_[index]:
            raise ValueError("Bin edges must be monotonically increasing.")

        self.number_of_bins_ += 1
        self.bin_edges_ = np.insert(self.bin_edges_, index, bin_edge)

        self.histograms_ = np.asarray(
            [np.insert(hist, index, 0) for hist in self.histograms_]
        )
        self.error_ = np.asarray(
            [np.insert(err, index, 0) for err in self.error_]
        )
        self.histograms_raw_count_ = np.asarray(
            [
                np.insert(raw_count, index, 0).tolist()
                for raw_count in self.histograms_raw_count_
            ]
        )
        self.systematic_error_ = np.asarray(
            [
                np.insert(sys_err, index, 0).tolist()
                for sys_err in self.systematic_error_
            ]
        )

        return self

    def add_value(
        self,
        value: Union[float, List[float], np.ndarray],
        weight: Optional[Union[float, List[float], np.ndarray]] = None,
    ) -> None:
        """
        Add value(s) to the latest histogram.

        Different cases, if there is just one number added or a whole
        list/array of numbers.

        Parameters
        ----------
        value: int, float, np.number, list, numpy.ndarray
            Value(s) which are supposed to be added to the histogram instance.
        weight: int, float, np.number, list, numpy.ndarray, optional
            Weight(s) associated with the value(s). If provided, it should have
            the same length as the value parameter.

        Raises
        ------
        TypeError
            if the input is not a number or numpy.ndarray or list
        ValueError
            if an input :code:`value` is :code:`np.nan`
        ValueError
            if the input :code:`weight` has not the same dimension as :code:`value`
        ValueError
            if a :code:`weight` value is :code:`np.nan`
        """
        if self.bin_edges_ is None:
            raise TypeError(
                "'bin_edges_' is None. It must be initialized before calling the 'add_value' function."
            )
        if self.number_of_bins_ is None:
            raise TypeError(
                "'number_of_bins_' is None. It must be initialized before calling the 'add_value' function."
            )
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'add_value' function."
            )
        if self.histograms_raw_count_ is None:
            raise TypeError(
                "'histograms_raw_count_' is None. It must be initialized before calling the 'add_value' function."
            )

        # Check if weight has the same length as value
        if weight is not None:
            if isinstance(weight, (int, float, np.number)):
                if not isinstance(value, (int, float, np.number)):
                    raise ValueError(
                        "Value must be numeric when weight is scalar."
                    )
                if np.isnan(weight):
                    raise ValueError(
                        "Value cannot be NaN when weight is scalar."
                    )
            elif len(weight) != np.atleast_1d(value).shape[0]:
                raise ValueError("Weight must have the same length as value.")
            else:
                if np.isnan(value).any():
                    raise ValueError(
                        "Value cannot contain NaN when weight is not scalar."
                    )

        # Case 1.1: value is a single number
        if isinstance(value, (int, float, np.number)):
            if np.isnan(value):
                raise ValueError("Input value in add_value is NaN.")

            bin_index = np.digitize(value, self.bin_edges_)
            if bin_index == 0 or bin_index > self.number_of_bins_:
                pass
            else:
                if weight is not None:
                    self.histograms_[-1, bin_index - 1] += weight
                    self.histograms_raw_count_[-1, bin_index - 1] += weight
                else:
                    self.histograms_[-1, bin_index - 1] += 1
                    self.histograms_raw_count_[-1, bin_index - 1] += 1

        # Case 1.2: value is a list of numbers
        elif isinstance(value, (list, np.ndarray)):
            if np.isnan(value).any():
                raise ValueError(
                    "At least one input value in add_value is NaN."
                )
            if weight is not None:
                if isinstance(weight, (int, float, np.number)):
                    for element in value:
                        self.add_value(element, weight=weight)
                else:
                    for element, w in zip(value, weight):
                        self.add_value(element, weight=w)
            else:
                for element in value:
                    self.add_value(element)

        # Case 1.3: value has an invalid input type
        else:
            err_msg = (
                "Invalid input type! Input value must have one of the "
                + "following types: (int, float, np.number, list, np.ndarray)"
            )
            raise TypeError(err_msg)

    def make_density(self) -> None:
        """
        Make a probability density from the last histogram.
        The result represents the probability density function in a given bin.
        It is normalized such that the integral over the whole histogram
        yields 1. This behavior is similar to the one of numpy histograms.

        Raises
        ------
        ValueError
            if there is no histogram available
        ValueError
            if the integral over the histogram is zero
        """
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'make_density' function."
            )

        if self.number_of_histograms_ == 0:
            raise ValueError("No histograms available to compute density.")

        last_histogram = self.histograms_[-1]

        bin_widths = self.bin_width()
        density = last_histogram / bin_widths
        integral = np.sum(density * bin_widths)
        if integral == 0:
            raise ValueError("Integral over the histogram is zero.")
        scale_factor = 1.0 / integral

        self.statistical_error()
        self.scale_histogram(scale_factor)

    def add_histogram(self) -> "Histogram":
        """
        Add a new histogram to the Histogram class instance.

        If new values are added to the histogram afterwards, these are added
        to the last histogram.
        """
        if self.number_of_bins_ is None:
            raise TypeError(
                "'number_of_bins_' is None. It must be initialized before calling the 'add_histogram' function."
            )
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'add_histogram' function."
            )
        if self.histograms_raw_count_ is None:
            raise TypeError(
                "'histograms_raw_count_' is None. It must be initialized before calling the 'add_histogram' function."
            )
        if self.scaling_ is None:
            raise TypeError(
                "'scaling_' is None. It must be initialized before calling the 'add_histogram' function."
            )
        if self.error_ is None:
            raise TypeError(
                "'error_' is None. It must be initialized before calling the 'add_histogram' function."
            )
        if self.systematic_error_ is None:
            raise TypeError(
                "'systematic_error_' is None. It must be initialized before calling the 'add_histogram' function."
            )

        empty_histogram = np.zeros(self.number_of_bins_)
        self.histograms_ = np.vstack((self.histograms_, empty_histogram))
        self.histograms_raw_count_ = np.vstack(
            (self.histograms_raw_count_, empty_histogram)
        )
        self.scaling_ = np.vstack(
            (self.scaling_, np.ones(self.number_of_bins_))
        )
        self.error_ = np.vstack((self.error_, np.zeros(self.number_of_bins_)))
        self.systematic_error_ = np.vstack(
            (self.systematic_error_, np.zeros(self.number_of_bins_))
        )
        self.number_of_histograms_ += 1

        return self

    def average(self) -> "Histogram":
        """
        Average over all histograms.

        When this function is called the previously generated histograms are
        averaged with the unit weights and they are overwritten by the
        averaged histogram.
        The standard error of the averaged histograms is computed.
        ``histogram_raw_counts`` is summed over all histograms.
        ``scaling_`` is set to the value of the first histogram.

        Returns
        -------
        Histogram
            Returns a Histogram object.

        """
        self.average_weighted(np.ones(self.number_of_histograms_))
        return self

    def average_weighted(self, weights: np.ndarray) -> "Histogram":
        """
        Weighted average over all histograms.

        When this function is called the previously generated histograms are
        averaged with the given weights and they are overwritten by the
        averaged histogram.
        The weighted standard error of the histograms is computed.
        ``histogram_raw_counts`` is summed over all histograms.
        ``scaling_`` is set to the value of the first histogram.

        Parameters
        ----------
        weights: numpy.ndarray
            Array containing a weight for each histogram.

        Returns
        -------
        Histogram
            Returns a Histogram object.

        """
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'average_weighted' function."
            )
        if self.systematic_error_ is None:
            raise TypeError(
                "'systematic_error_' is None. It must be initialized before calling the 'average_weighted' function."
            )
        if self.histograms_raw_count_ is None:
            raise TypeError(
                "'histograms_raw_count_' is None. It must be initialized before calling the 'average_weighted' function."
            )
        if self.scaling_ is None:
            raise TypeError(
                "'scaling_' is None. It must be initialized before calling the 'average_weighted' function."
            )

        average = np.average(self.histograms_, axis=0, weights=weights)
        variance = np.average(
            (self.histograms_ - average) ** 2.0, axis=0, weights=weights
        )
        # Ensure the result is a 2D array
        if average.ndim == 1:
            average = average.reshape(1, -1)

        self.histograms_ = average
        self.error_ = np.sqrt(variance).reshape(1, -1)
        self.systematic_error_ = np.sqrt(
            np.average(self.systematic_error_**2.0, axis=0, weights=weights)
        )
        self.histogram_raw_count_ = np.sum(self.histograms_raw_count_, axis=0)
        self.scaling_ = np.asarray(self.scaling_[0])

        if self.scaling_.ndim == 1:
            self.scaling_ = self.scaling_.reshape(1, -1)

        self.number_of_histograms_ = 1

        return self

    def average_weighted_by_error(self) -> "Histogram":
        """
        Weighted average over all histograms, where each entry is weighted by its associated error.

        When this function is called the previously generated histograms are
        averaged with the weights determined by the inverse of the error associated with each entry.
        The histograms are then overwritten by the averaged histogram.
        The weighted standard error of the histograms is computed.
        ``histogram_raw_counts`` is summed over all histograms.
        ``scaling_`` is set to the value of the first histogram.

        Returns
        -------
        Histogram
            Returns a Histogram object.

        Raises
        ------
        TypeError
            if the error is zero for any entry.
        """
        if self.error_ is None:
            raise TypeError(
                "'error_' is None. It must be initialized before calling the 'average_weighted' function."
            )
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'average_weighted' function."
            )
        if self.systematic_error_ is None:
            raise TypeError(
                "'systematic_error_' is None. It must be initialized before calling the 'average_weighted' function."
            )
        if self.histograms_raw_count_ is None:
            raise TypeError(
                "'histograms_raw_count_' is None. It must be initialized before calling the 'average_weighted' function."
            )
        if self.scaling_ is None:
            raise TypeError(
                "'scaling_' is None. It must be initialized before calling the 'average_weighted' function."
            )

        if np.any(self.error_ == 0):
            raise TypeError(
                "Error cannot be zero for any entry when averaging by error."
            )

        weights = 1 / self.error_**2
        average = np.average(self.histograms_, axis=0, weights=weights)

        # Ensure the result is a 2D array
        if average.ndim == 1:
            average = average.reshape(1, -1)

        self.histograms_ = average
        self.error_ = np.sqrt(
            1.0 / np.sum(1.0 / np.square(self.error_), axis=0)
        )
        self.systematic_error_ = np.sqrt(
            np.average(self.systematic_error_**2.0, axis=0, weights=weights)
        )
        self.histogram_raw_count_ = np.sum(self.histograms_raw_count_, axis=0)
        self.scaling_ = np.asarray(self.scaling_[0])

        if self.scaling_.ndim == 1:
            self.scaling_ = self.scaling_.reshape(1, -1)

        self.number_of_histograms_ = 1

        return self

    def standard_error(self) -> np.ndarray:
        """
        Get the standard deviation over all histogram counts for each bin.

        Returns
        -------
        numpy.ndarray
            Array containing the standard deviation for each bin.
        """
        if self.error_ is None:
            raise TypeError(
                "'error_' is None. It must be initialized before calling the 'standard_error' function."
            )

        return self.error_

    def statistical_error(self) -> np.ndarray:
        """
        Compute the statistical error of all histogram bins for all histograms.
        This assumes Poisson distributed counts in each bin and independent draws.

        Returns
        -------
        numpy.ndarray
            2D Array containing the statistical error (standard deviation) for
            each bin and histogram.
        """
        if self.error_ is None:
            raise TypeError(
                "'error_' is None. It must be initialized before calling the 'statistical_error' function."
            )

        counter_histogram = 0
        for histogram in self.histogram():
            self.error_[counter_histogram] = np.sqrt(histogram)
            counter_histogram += 1
        return self.error_

    def scale_histogram(
        self, value: Union[int, float, np.number, List[float], np.ndarray]
    ) -> None:
        """
        Scale the latest histogram by a factor.

        Multiplies the latest histogram by a number or a list/numpy array with a
        scaling factor for each bin.

        The standard deviation of the histogram(s) is also rescaled by the same factor.

        Parameters
        ----------
        value: int, float, np.number, list, numpy.ndarray
            Scaling factor for the histogram.
        """
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'scale_histogram' function."
            )
        if self.scaling_ is None:
            raise TypeError(
                "'scaling_' is None. It must be initialized before calling the 'scale_histogram' function."
            )
        if self.error_ is None:
            raise TypeError(
                "'error_' is None. It must be initialized before calling the 'scale_histogram' function."
            )

        if isinstance(value, (int, float, np.number)) and value < 0:
            raise ValueError(
                "The scaling factor of the histogram cannot be negative"
            )
        elif (
            isinstance(value, (list, np.ndarray))
            and sum(1 for number in value if number < 0) > 0
        ):
            raise ValueError(
                "The scaling factor of the histogram cannot be negative"
            )
        elif (
            isinstance(value, (list, np.ndarray))
            and len(value) != self.number_of_bins_
        ):
            raise ValueError(
                "The length of list/array not compatible with number_of_bins_ of the histogram"
            )

        if isinstance(value, (int, float, np.number)):
            self.histograms_[-1] *= value
            self.scaling_[-1] *= value
            self.error_[-1] *= value

        elif isinstance(value, (list, np.ndarray)):
            if np.asarray(value).shape != self.histograms_[-1].shape:
                raise ValueError(
                    "The shape of the scaling factor array is not compatible with the histogram shape"
                )

            self.histograms_[-1] *= np.asarray(value)
            self.scaling_[-1] *= np.asarray(value)
            self.error_[-1] *= np.asarray(value)

    def set_error(self, own_error: Union[List[float], np.ndarray]) -> None:
        """
        Sets the histogram error by hand. This is helpful for weighted
        histograms where the weight has also an uncertainty.

        This function has to be called after averaging, otherwise the error will
        be overwritten by the standard error.

        Parameters
        ----------
        value: list, numpy.ndarray
            Values for the uncertainties of the individual bins.
        """
        if len(own_error) != self.number_of_bins_ or not isinstance(
            own_error, (list, np.ndarray)
        ):
            error_message = (
                "The input error has a different length than the"
                + " number of histogram bins or it is not a list/numpy.ndarray"
            )
            raise ValueError(error_message)
        if self.error_ is None:
            raise TypeError(
                "'error_' is None. It must be initialized before calling the 'set_error' function."
            )

        self.error_[-1] = own_error

    def set_systematic_error(
        self, own_error: Union[List[float], np.ndarray]
    ) -> None:
        """
        Sets the systematic histogram error of the last created histogram by hand.

        Parameters
        ----------
        value: list, numpy.ndarray
            Values for the systematic uncertainties of the individual bins.
        """
        if len(own_error) != self.number_of_bins_ or not isinstance(
            own_error, (list, np.ndarray)
        ):
            error_message = (
                "The input error has a different length than the"
                + " number of histogram bins or it is not a list/numpy.ndarray"
            )
            raise ValueError(error_message)
        if self.systematic_error_ is None:
            raise TypeError(
                "'systematic_error_' is None. It must be initialized before calling the 'set_systematic_error' function."
            )

        self.systematic_error_[-1] = own_error

    def print_histogram(self) -> None:
        """Print the histograms to the terminal."""
        if self.number_of_histograms_ is None:
            raise TypeError(
                "'number_of_histograms_' is None. It must be initialized before calling the 'print_histogram' function."
            )
        if self.number_of_bins_ is None:
            raise TypeError(
                "'number_of_bins_' is None. It must be initialized before calling the 'print_histogram' function."
            )
        if self.bin_edges_ is None:
            raise TypeError(
                "'bin_edges_' is None. It must be initialized before calling the 'print_histogram' function."
            )
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'print_histogram' function."
            )

        print("bin_low,bin_high,bin_value")
        for hist in range(self.number_of_histograms_):
            print(f"{hist}. histogram:")
            for bin in range(self.number_of_bins_):
                print(
                    f"[{self.bin_edges_[bin]},{self.bin_edges_[bin+1]}):\
                          {self.histograms_[hist][bin]}"
                )
            print("")

    def write_to_file(
        self,
        filename: str,
        hist_labels: List[Dict[str, str]],
        comment: str = "",
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Write multiple histograms to a CSV file along with their headers.

        Parameters
        ----------
        filename : str
            Name of the output CSV file.

        hist_labels : list of dicts
            List containing dictionaries with header labels for each histogram.
            Provide a list of dictionaries, where each dictionary contains the
            header labels for the corresponding histogram. If the list contains
            only one dictionary, then this is used for all histograms.

            The dictionaries should have the following keys:
                - 'bin_center': Label for the bin center column.
                - 'bin_low': Label for the lower boundary of the bins.
                - 'bin_high': Label for the upper boundary of the bins.
                - 'distribution': Label for the histogram/distribution.
                - 'stat_err+': Label for the statistical error (positive).
                - 'stat_err-': Label for the statistical error (negative).
                - 'sys_err+': Label for the systematic error (positive).
                - 'sys_err-': Label for the systematic error (negative).

            Other keys are possible if non-default columns are used.

        comment : str, optional
            Additional comment to be included at the beginning of the file.
            It is possible to give a multi-line comment where each line should
            start with a '#'.

        columns : list of str, optional
            List of columns to include in the output. If None, all columns are included which are:
                - 'bin_center': Bin center.
                - 'bin_low': Lower bin boundary.
                - 'bin_high': Upper bin boundary.
                - 'distribution': Histogram value.
                - 'stat_err+': Statistical error (positive).
                - 'stat_err-': Statistical error (negative).
                - 'sys_err+': Systematic error (positive).
                - 'sys_err-': Systematic error (negative).

        Raises
        ------
        TypeError
            If `hist_labels` is not a list of dictionaries, or if `columns` is
            not a list of strings or contains keys which are not present in 'hist_labels'.

        ValueError
            If the number of histograms is greater than 1, and the number of
            provided headers in `hist_labels` does not match the number of
            histograms. An exception is the case, where the number of histograms
            is greater than 1 and the number of provided dictionaries is 1.
            Then the same dictionary is used for all histograms.
        """
        if not isinstance(hist_labels, list) or not all(
            isinstance(hist_label, dict) for hist_label in hist_labels
        ):
            raise TypeError("hist_labels must be a list of dictionaries")

        if columns is not None and (
            not isinstance(columns, list)
            or not all(isinstance(col, str) for col in columns)
        ):
            raise TypeError("columns must be a list of strings")
        if self.number_of_bins_ is None:
            raise TypeError(
                "'number_of_bins_' is None. It must be initialized before calling the 'write_to_file' function."
            )
        if self.histograms_ is None:
            raise TypeError(
                "'histograms_' is None. It must be initialized before calling the 'write_to_file' function."
            )
        if self.error_ is None:
            raise TypeError(
                "'error_' is None. It must be initialized before calling the 'write_to_file' function."
            )
        if self.systematic_error_ is None:
            raise TypeError(
                "'systematic_error_' is None. It must be initialized before calling the 'write_to_file' function."
            )

        if columns is not None and not all(
            col in hist_labels[0].keys() for col in columns
        ):
            raise TypeError(
                "columns must contain only keys present in hist_labels"
            )

        if self.number_of_histograms_ > 1 and len(hist_labels) == 1:
            error_message = (
                "Print multiple histograms to file, only one header"
                + " provided. Use the header for all histograms."
            )
            warnings.warn(error_message)
        elif self.number_of_histograms_ > 1 and (
            len(hist_labels) > 1
            and len(hist_labels) < self.number_of_histograms_
        ):
            raise ValueError(
                "Print multiple histograms to file, more than one,"
                + " but less than number of histograms headers provided."
            )

        if columns is None:
            columns = [
                "bin_center",
                "bin_low",
                "bin_high",
                "distribution",
                "stat_err+",
                "stat_err-",
                "sys_err+",
                "sys_err-",
            ]

        with open(filename, "w") as f:
            writer = csv.writer(f)
            if comment != "":
                f.write(comment)
                f.write("\n")

            for idx in range(self.number_of_histograms_):
                header = [hist_labels[idx][col] for col in columns]
                writer.writerow(header)
                for i in range(self.number_of_bins_):
                    data = [
                        self.bin_centers()[i],
                        self.bin_bounds_left()[i],
                        self.bin_bounds_right()[i],
                        self.histograms_[idx][i],
                        self.error_[idx][i],
                        self.error_[idx][i],
                        self.systematic_error_[idx][i],
                        self.systematic_error_[idx][i],
                    ]
                    data = [data[columns.index(col)] for col in columns]
                    writer.writerow(data)
                f.write("\n")
