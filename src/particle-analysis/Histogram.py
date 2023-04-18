import numpy as np
import csv


class Histogram:

    def __init__(self, bin_boundaries):
        """
        Defines a histogram object.

        The histograms can be initialized either with a tuple
        (hist_min,hist_max,num_bins) or a list/numpy.ndarray containing the bin
        boundaries, which allows for different bin widths.
        Multiple histograms can be added and averaged over.
        """
    
        self.number_of_bins = None
        self.bin_edges = None
        self.number_of_histograms = 1
        self.histograms = None
        self.histograms_raw_count = None
        self.error = None
        self.scaling = None
        
        if isinstance(bin_boundaries, tuple) and len(bin_boundaries) == 3:
            hist_min = bin_boundaries[0]
            hist_max = bin_boundaries[1]
            num_bins = bin_boundaries[2]
            
            if hist_min > hist_max or hist_min == hist_max:
                raise ValueError('hist_min must be smaller than hist_max')
                
            elif not isinstance(num_bins,int) or num_bins <= 0:
                raise ValueError('Number of bins must be a positive integer')
            
            self.number_of_bins = num_bins
            self.bin_edges = np.linspace(hist_min, hist_max, num=num_bins+1)
            self.histograms = np.zeros(num_bins)
            self.histograms_raw_count = np.zeros(num_bins)
            self.scaling = np.ones(num_bins)
            
        elif isinstance(bin_boundaries, (list, np.ndarray)):
            
            self.number_of_bins = len(bin_boundaries)-1
            self.bin_edges = np.asarray(bin_boundaries)
            self.histograms = np.zeros(self.number_of_bins)
            self.histograms_raw_count = np.zeros(self.number_of_bins)
            self.scaling = np.ones(self.number_of_bins)
            
        else:
            raise TypeError('Input must be a tuple (hist_min, hist_max, num_bins) '+\
                            'or a list/numpy.ndarray containing the bin edges!')
        
        
    def histogram(self):
        """
        Get the created histogram(s).
        
        Returns
        -------
        numpy.ndarray
            Array containing the histogram(s).
        """
        return self.histograms
    

    def histogram_raw_counts(self):
        """
        Get the raw bin counts of the histogram(s), even after the oroginal
        histograms are scaled or averaged.

        Returns
        -------
        numpy.ndarray
            Array containing the raw counts of the histogram(s)
        """
        return self.histograms_raw_count
    
    
    def number_of_histograms(self):
        """
        Get the number of created histograms.

        Returns
        -------
        int
            Number of histograms.
        """
        return self.number_of_histograms
    
    
    def bin_centers(self):
        """
        Compute the bin centers.
        
        Returns
        -------
        numpy.ndarray
            Array containing the bin centers.
        """
        
        
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
    

    def bin_width(self):
        """
        Compute the bin widths.

        Returns
        -------
        numpy.ndarray
            Array containing the bin widths.
        """
        return self.bin_edges[1:] - self.bin_edges[:-1]
    

    def bin_bounds_left(self):
        """
        Extract the lower bounds of the individual bins.
        
        Returns
        -------
        numpy.ndarray
            Array containing the lower bin boundaries.
        """
        return self.bin_edges[:-1]
    

    def bin_bounds_right(self):
        """
        Extract the upper bounds of the individual bins.
        
        Returns
        -------
        numpy.ndarray
            Array containing the upper bin boundaries.
        """
        return self.bin_edges[1:]
    

    def bin_boundaries(self):
        """
        Get the bin boundaries.

        Returns
        -------
        numpy.ndarray
            Array containing the bin boundaries.
        """
        return np.asarray(self.bin_edges)
    
    
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
            
            # Throw warning and add warning counter. Cut off after first warning
            if value < self.bin_edges[0] or value > self.bin_edges[-1]:
                err_msg = 'Value '+str(value)+' lies outside the histogram '+\
                          'range ['+str(self.bin_edges[0])+','+str(self.bin_edges[-1])+\
                          ']. Increase histogram range!'
                raise ValueError(err_msg)
                #print(err_msg)
                
            else:
                for bin_index in range(self.number_of_bins):
                    # Case 2.1: histogram contains only 1 instance
                    if self.number_of_histograms == 1:
                        if bin_index == 0 and value == self.bin_edges[0]:
                            self.histograms[0] += 1
                            self.histograms_raw_count[0] += 1
                        elif value > self.bin_edges[bin_index] and value <= self.bin_edges[bin_index+1]:
                            self.histograms[bin_index] += 1
                            self.histograms_raw_count[bin_index] += 1
                            
                    # Case 2.2: If histogram contains multiple instances, 
                    #           always add values to the latest histogram
                    else:
                        if bin_index == 0 and value == self.bin_edges[0]:
                            self.histograms[-1,0] += 1
                            self.histograms_raw_count[-1,0] += 1
                        elif value > self.bin_edges[bin_index] and value <= self.bin_edges[bin_index+1]:
                            self.histograms[-1,bin_index] += 1
                            self.histograms_raw_count[-1,bin_index] += 1
            
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
        empty_histogram = np.zeros(self.number_of_bins)
        self.histograms = np.vstack((self.histograms, empty_histogram))
        
        self.number_of_histograms += 1


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
        if self.histograms.ndim == 1:
            raise TypeError('Cannot average an array of dim = 1')
        else:
            self.error = np.std(self.histograms, axis=0)/np.sqrt(self.number_of_histograms)
            self.histograms = np.mean(self.histograms, axis=0)
            self.number_of_histograms = 1
            
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
        if self.histograms.ndim == 1:
            raise TypeError('Cannot average an array of dim = 1')
        else:
            self.error = np.std(self.histograms, axis=0)/np.sqrt(self.number_of_histograms)
            self.histograms = np.average(self.histograms, axis=0, weights=weights)
            self.number_of_histograms = 1

            return self
        

    def standard_error(self):
        """
        Get the standard deviation of the created histograms.

        Returns
        -------
        numpy.ndarray
            Array containing the standard deviation for each bin.
        """
        return self.error
    

    def statistical_error(self):
        """
        Compute the statistical error of a single histogram bin.

        Only usable if the there is one histogram. Calling this function will 
        overwrite all other errors.

        Returns
        -------
        numpy.ndarray
            Array containing the statistical error for each bin.
        """
        if self.number_of_histograms == 1:
            for bin in range(self.number_of_bins):
                if self.histograms_raw_count[bin] > 0:
                    self.error[bin] = self.scaling[bin]\
                        * np.sqrt(self.histograms_raw_count[bin])
                else:
                    self.error[bin] = 0.
            
            return self.error
    
    
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
        if isinstance(value, (int, float, np.number)):
            self.histograms *= value
            self.scaling *= value

        elif isinstance(value, (list, np.ndarray)):
            self.histograms *= np.asarray(value)
            self.scaling *= np.asarray(value)


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
        if len(own_error) != self.number_of_bins and\
              not isinstance(own_error, (list,np.ndarray)):
            error_message = "The input error has a different length than the"\
                 + " number of histogram bins or it is not a list/numpy.ndarray"
            raise ValueError(error_message)

        self.error = own_error


    def print_histogram(self):
        """Print the histograms to the terminal."""
        print("bin_low,bin_high,bin_value,bin_error")
        for hist in range(self.number_of_histograms):
            print(f"{hist}. histogram:")
            for bin in range(self.number_of_bins):
                if self.number_of_histograms == 1:
                    print(f'{self.bin_edges[bin]},{self.bin_edges[bin+1]},\
                          {self.histograms[bin]}')
                else:
                    print(f'{self.bin_edges[bin]},{self.bin_edges[bin+1]},\
                          {self.histograms[hist][bin]}')


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
        if self.number_of_histograms > 1:
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
        distribution = self.histograms
        error = self.error
        for i in range(self.number_of_bins):
            data = [bin_centers[i],bin_low[i],bin_high[i],distribution[i],\
                    error[i]]
            writer.writerow(data)