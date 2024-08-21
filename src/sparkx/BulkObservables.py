#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================

from sparkx.Histogram import Histogram

# This class ensures a list to be read-only
class ReadOnlyList:
    def __init__(self, nested_list):
        self._nested_list = nested_list  # Store reference to the original list

    def __getitem__(self, index):
        return self._nested_list[index]  # Allow read access to list items

    def __len__(self):
        return len(self._nested_list)  # Allow getting the length of the list

    def __iter__(self):
        return iter(self._nested_list)  # Allow iteration over the list

    def __repr__(self):
        return repr(self._nested_list)  # Allow printing the list

    # Prevent modification by raising exceptions
    def __setitem__(self, index, value):
        raise TypeError("This list is read-only.")

    def append(self, value):
        raise TypeError("This list is read-only.")

    def extend(self, values):
        raise TypeError("This list is read-only.")

    def insert(self, index, value):
        raise TypeError("This list is read-only.")

    def remove(self, value):
        raise TypeError("This list is read-only.")

    def pop(self, index=-1):
        raise TypeError("This list is read-only.")

    def clear(self):
        raise TypeError("This list is read-only.")


class BulkObservables:
    def __init__(self, particle_objects_list, *args):
        
        # Wrapping in ReadOnlyList prevents particles from being modified
        self.particle_objects = ReadOnlyList(particle_objects_list)  
        self.observables_list = args

    # PRIVATE CLASS METHODS
    """
    This is the base method for all differential yields. It is called by all 
    corresponding methods for each specific differential yield
    """
    def _differential_yield(self, quantity, bins=(-4, 4, 21)):
        hist = Histogram(bins)
        num_events = len(self.particle_objects)
        inverse_bin_width = 1/hist.bin_width()
        
        # Fill histograms
        for event in range(num_events):
            for particle in self.particle_objects[event]:
                """
                Call the corresponding method in Particle given by 
                'quantity' as string
                """
                particle_method = getattr(particle, quantity)
                if callable(particle_method):
                    hist.add_value(particle_method())
                else:
                    raise AttributeError(f"'{quantity}' is not a callable method of Particle")
            if num_events > 1:
                hist.add_histogram()
                
        hist.average()
        hist.scale_histogram(inverse_bin_width)
        return hist

    # PUBLIC CLASS METHODS
    def dNdy(self, bins=None):
        """
        Calculate the event averaged yield :math:`\\frac{dN}{dy}`

        Args:
        - bins: Optional tuple (start, stop, num) for histogram binning. If 
          not given, the default of differential_yield() will be used

        Returns:
        - 1D histogram containing the event averaged particle counts per 
          rapidity bin.
        """
        if bins is None:
            return self._differential_yield("rapidity")
        else:
            return self._differential_yield("rapidity", bins)

    def dNdpT(self, bins=None):
        """
        Calculate the event averaged yield :math:`\\frac{dN}{dp_T}`

        Args:
        - bins: Optional tuple (start, stop, num) for histogram binning. If 
          not given, the default of differential_yield() will be used

        Returns:
        - 1D histogram containing the event averaged particle counts per 
          transverse momentum bin.
        """
        if bins is None:
            return self._differential_yield("pT_abs")
        else:
            return self._differential_yield("pT_abs", bins)
        
    def dNdEta(self, bins=None):
        """
        Calculate the event averaged yield :math:`\\frac{dN}{d\eta}`

        Args:
        - bins: Optional tuple (start, stop, num) for histogram binning. If 
          not given, the default of differential_yield() will be used

        Returns:
        - 1D histogram containing the event averaged particle counts per 
          pseudo-rapidity bin.
        """
        if bins is None:
            return self._differential_yield("pseudorapidity")
        else:
            return self._differential_yield("pseudorapidity", bins)

    def dNdmT(self, bins=None):
        """
        Calculate the event averaged yield :math:`\\frac{dN}{dm_T}`

        Args:
        - bins: Optional tuple (start, stop, num) for histogram binning. If
          not given, the default of differential_yield() will be used

        Returns:
        - 1D histogram containing the event averaged particle counts per
          transverse mass bin.
        """
        if bins is None:
            return self._differential_yield("mT")
        else:
            return self._differential_yield("mT", bins)
