import numpy as np
import csv
import warnings


class Histogram:
    """
    Defines a histogram object.

    The histograms can be initialized either with a tuple
    (hist_min,hist_max,num_bins) or a list/numpy.ndarray containing the bin
    boundaries, which allows for different bin widths.
    Multiple histograms can be added and averaged over.

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
        Get the number of created histograms.
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
    """
    def __init__(self, bin_boundaries):
        self.number_of_bins_ = None
        self.bin_edges_ = None
        self.number_of_histograms_ = 1
        self.histograms_ = None
        self.histograms_raw_count_ = None
        self.error_ = None
        self.scaling_ = None

        if isinstance(bin_boundaries, tuple) and len(bin_boundaries) == 3:
            hist_min = bin_boundaries[0]
            hist_max = bin_boundaries[1]
            num_bins = bin_boundaries[2]

            if hist_min > hist_max or hist_min == hist_max:
                raise ValueError('hist_min must be smaller than hist_max')

            elif not isinstance(num_bins,int) or num_bins <= 0:
                raise ValueError('Number of bins must be a positive integer')

            self.number_of_bins_ = num_bins
            self.bin_edges_ = np.linspace(hist_min, hist_max, num=num_bins+1)
            self.histograms_ = np.zeros(num_bins)
            self.histograms_raw_count_ = np.zeros(num_bins)
            self.scaling_ = np.ones(num_bins)
            self.error_ = np.zeros(num_bins)

        elif isinstance(bin_boundaries, (list, np.ndarray)):

            self.number_of_bins_ = len(bin_boundaries)-1
            self.bin_edges_ = np.asarray(bin_boundaries)
            self.histograms_ = np.zeros(self.number_of_bins_)
            self.histograms_raw_count_ = np.zeros(self.number_of_bins_)
            self.scaling_ = np.ones(self.number_of_bins_)
            self.error_ = np.zeros(self.number_of_bins_)

        else:
            raise TypeError('Input must be a tuple (hist_min, hist_max, num_bins) '+\
                            'or a list/numpy.ndarray containing the bin edges!')

    def histogram(self):
        """
        Get the created histogram(s).

        Returns
        -------
        histograms_: numpy.ndarray
            Array containing the histogram(s).
        """
        return self.histograms_

    def histogram_raw_counts(self):
        """
        Get the raw bin counts of the histogram(s), even after the original
        histograms are scaled or averaged.

        Returns
        -------
        histograms_raw_count_: numpy.ndarray
            Array containing the raw counts of the histogram(s)
        """
        return self.histograms_raw_count_

    def number_of_histograms(self):
        """
        Get the number of created histograms.

        Returns
        -------
        number_of_histograms_: int
            Number of histograms.
        """
        return self.number_of_histograms_

    def bin_centers(self):
        """
        Get the bin centers.

        Returns
        -------
        numpy.ndarray
            Array containing the bin centers.
        """
        return (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2.0

    def bin_width(self):
        """
        Get the bin widths.

        Returns
        -------
        numpy.ndarray
            Array containing the bin widths.
        """
        return self.bin_edges_[1:] - self.bin_edges_[:-1]

    def bin_bounds_left(self):
        """
        Extract the lower bounds of the individual bins.

        Returns
        -------
        numpy.ndarray
            Array containing the lower bin boundaries.
        """
        return self.bin_edges_[:-1]

    def bin_bounds_right(self):
        """
        Extract the upper bounds of the individual bins.

        Returns
        -------
        numpy.ndarray
            Array containing the upper bin boundaries.
        """
        return self.bin_edges_[1:]

    def bin_boundaries(self):
        """
        Get the bin boundaries.

        Returns
        -------
        numpy.ndarray
            Array containing the bin boundaries.
        """
        return np.asarray(self.bin_edges_)

    def add_value(self, value):
        """
        Add value(s) to the histogram.

        Different cases, if there is just one number added or a whole list/
        array of numbers.

        Parameters
        ----------
        value: int, float, np.number, list, numpy.ndarray
            Value(s) which ar supposed to be added to the histogram instance.
        """
        # Case 1.1: value is a single number
        if isinstance(value, (int, float, np.number)):

            counter_warnings = 0
            if (value < self.bin_edges_[0] or value > self.bin_edges_[-1]) and counter_warnings == 0:
                warn_msg = 'One or more values lie outside the histogram '+\
                          'range ['+str(self.bin_edges_[0])+','+str(self.bin_edges_[-1])+\
                          ']. Exceeding values are ignored. Increase histogram range!'
                warnings.warn(warn_msg)

            else:
                for bin_index in range(self.number_of_bins_):
                    # Case 2.1: histogram contains only 1 instance
                    if self.number_of_histograms_ == 1:
                        if bin_index == 0 and value == self.bin_edges_[0]:
                            self.histograms_[0] += 1
                            self.histograms_raw_count_[0] += 1
                        elif value > self.bin_edges_[bin_index] and value <= self.bin_edges_[bin_index+1]:
                            self.histograms_[bin_index] += 1
                            self.histograms_raw_count_[bin_index] += 1

                    # Case 2.2: If histogram contains multiple instances,
                    #           always add values to the latest histogram
                    else:
                        if bin_index == 0 and value == self.bin_edges_[0]:
                            self.histograms_[-1,0] += 1
                            self.histograms_raw_count_[-1,0] += 1
                        elif value > self.bin_edges_[bin_index] and value <= self.bin_edges_[bin_index+1]:
                            self.histograms_[-1,bin_index] += 1
                            self.histograms_raw_count_[-1,bin_index] += 1

        # Case 1.2: value is a list of numbers
        elif isinstance(value, (list, np.ndarray)):
            for element in value:
                self.add_value(element)

        # Case 1.3: value has an invalid input type
        else:
            err_msg = 'Invalid input type! Input value must have one of the '+\
                      'following types: (int, float, np.number, list, np.ndarray)'
            raise TypeError(err_msg)


    def add_histogram(self):
        """
        Add a new histogram to the Histogram class instance.

        If new values are added to the histogram afterwards, these are added
        to the last histogram.
        """
        empty_histogram = np.zeros(self.number_of_bins_)
        self.histograms_ = np.vstack((self.histograms_, empty_histogram))
        self.histograms_raw_count_ = np.vstack((self.histograms_raw_count_, empty_histogram))
        self.scaling_ = np.vstack((self.scaling_, np.ones(self.number_of_bins_)))
        self.error_ = np.vstack((self.error_, np.zeros(self.number_of_bins_)))

        self.number_of_histograms_ += 1


    def average(self):
        """
        Average the created histograms.

        When this function is called the previously generated histograms are
        averaged with the same weigths and they are overwritten by the
        averaged histogram.
        The standard error of the histograms is computed.

        Returns
        -------
        Histogram
            Returns a Histogram object.
        """
        
        if self.histograms_.ndim == 1:
            raise TypeError('Cannot average an array of dim = 1')
        else:
            self.error_ = np.sqrt(np.sum(self.histograms_, axis=0))/self.number_of_histograms_
            self.histograms_ = np.mean(self.histograms_, axis=0)
            self.number_of_histograms_ = 1

            return self


    def average_weighted(self,weights):
        """
        Weighted average over the created histograms.

        When this function is called the previously generated histograms are
        averaged with the given weigths and they are overwritten by the
        averaged histogram.
        The standard error of the histograms is computed.

        Returns
        -------
        Histogram
            Returns a Histogram object.
        """
        #TODO: correct error as in average()
        
        if self.histograms_.ndim == 1:
            raise TypeError('Cannot average an array of dim = 1')
        else:
            self.error_ = np.std(self.histograms_, axis=0)/np.sqrt(self.number_of_histograms_)
            self.histograms_ = np.average(self.histograms_, axis=0, weights=weights)
            self.number_of_histograms_ = 1

            return self


    def standard_error(self):
        """
        Get the standard deviation of the created histograms.

        Returns
        -------
        numpy.ndarray
            Array containing the standard deviation for each bin.
        """
        return self.error_


    def statistical_error(self):
        """
        Compute the statistical error of all histogram bins for all histograms.
        
        Returns
        -------
        numpy.ndarray
            2D Array containing the statistical error for each bin and histogram.
        """
        counter_histogram = 0
        for histogram in self.histogram():
            self.error_[counter_histogram] = np.sqrt(histogram)
            counter_histogram += 1

        return self.error_


    def scale_histogram(self,value):
        """
        Scale the histogram by a factor.

        Multiplies the histogram by a number or a list/numpy array with a
        scaling factor for each bin.

        Parameters
        ----------
        value: int, float, np.number, list, numpy.ndarray
            Scaling factor for the histogram.
        """
        if self.histograms_.ndim == 1:
            if isinstance(value, (int, float, np.number)):
                self.histograms_ *= value
                self.scaling_ *= value

            elif isinstance(value, (list, np.ndarray)):
                self.histograms_ *= np.asarray(value)
                self.scaling_ *= np.asarray(value)
        else:
            if isinstance(value, (int, float, np.number)):
                self.histograms_[-1] *= value
                self.scaling_[-1] *= value

            elif isinstance(value, (list, np.ndarray)):
                self.histograms_[-1] *= np.asarray(value)
                self.scaling_[-1] *= np.asarray(value)
        


    def set_error(self,own_error):
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
        if len(own_error) != self.number_of_bins_ and\
              not isinstance(own_error, (list,np.ndarray)):
            error_message = "The input error has a different length than the"\
                 + " number of histogram bins or it is not a list/numpy.ndarray"
            raise ValueError(error_message)

        self.error_ = own_error


    def print_histogram(self):
        """Print the histograms to the terminal."""
        print("bin_low,bin_high,bin_value,bin_error")
        for hist in range(self.number_of_histograms_):
            print(f"{hist}. histogram:")
            for bin in range(self.number_of_bins_):
                if self.number_of_histograms_ == 1:
                    print(f'{self.bin_edges_[bin]},{self.bin_edges_[bin+1]},\
                          {self.histograms_[bin]}')
                else:
                    print(f'{self.bin_edges_[bin]},{self.bin_edges_[bin+1]},\
                          {self.histograms_[hist][bin]}')


    def write_to_file(self,filename,label_bin_center,label_bin_low,\
                      label_bin_high,label_distribution,label_error,comment=''):
        """
        Write one histogram to a csv file.

        Parameters
        ----------
        filename: string
            Name for the output file
        label_bin_center: string
            Label for the bin center column.
        label_bin_low: string
            Label for the lower boundary of the bins.
        label_bin_high: string
            Label for the upper boundary of the bins.
        label_distribution: string
            Label for the histogram / distribution.
        label_error: string
            Label for the statistical error.
        comment: string
            Additional comment at the beginning of the file. It is possible
            to give a multi line comment, where each line should start with
            a '#'.
        """
        if self.number_of_histograms_ > 1:
            raise ValueError("At the moment only a single histogram can be"+\
                            " written to a file")

        f = open(filename, 'w')
        writer = csv.writer(f)
        if comment != '':
            f.write(comment)
            f.write('\n')
        header = [label_bin_center,label_bin_low,label_bin_high,\
                  label_distribution,label_error]
        writer.writerow(header)
        bin_centers = self.bin_centers()
        bin_low = self.bin_bounds_left()
        bin_high = self.bin_bounds_right()
        distribution = self.histograms_
        error = self.error_
        for i in range(self.number_of_bins_):
            data = [bin_centers[i],bin_low[i],bin_high[i],distribution[i],\
                    error[i]]
            writer.writerow(data)

            