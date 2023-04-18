import numpy as np


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
        self.error = None
        
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
            
        elif isinstance(bin_boundaries, (list, np.ndarray)):
            
            self.number_of_bins = len(bin_boundaries)-1
            self.bin_edges = np.asarray(bin_boundaries)
            self.histograms = np.zeros(self.number_of_bins)
            
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
                
            else:
                for bin_index in range(self.number_of_bins):
                    # Case 2.1: histogram contains only 1 instance
                    if self.number_of_histograms == 1:
                        if bin_index == 0 and value == self.bin_edges[0]:
                            self.histograms[0] += 1
                        elif value > self.bin_edges[bin_index] and value <= self.bin_edges[bin_index+1]:
                            self.histograms[bin_index] += 1
                            
                    # Case 2.2: If histogram contains multiple instances, 
                    #           always add values to the latest histogram
                    else:
                        if bin_index == 0 and value == self.bin_edges[0]:
                            self.histograms[-1,0] += 1
                        elif value > self.bin_edges[bin_index] and value <= self.bin_edges[bin_index+1]:
                            self.histograms[-1,bin_index] += 1
            
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
        average histogram.
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

        elif isinstance(value, (list, np.ndarray)):
            self.histograms *= np.asarray(value)
    
        

        
#test = Histogram([-1, -0.5, -0.2, 0, 0.5, 0.9, 1])
test = Histogram((-1,1, 4))
test.add_value([-1, 1, 0, 0.4, 0.1, -0.9, -0.95])
test.add_histogram()
print(test.histogram())
test.average()
print('average: ', test.histogram())
del test