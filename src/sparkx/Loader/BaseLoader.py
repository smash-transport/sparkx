#===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================

from abc import ABC, abstractmethod

class BaseLoader(ABC):
    
    @abstractmethod
    def __init__(self, path):
        pass

    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError("load method is not implemented")
    
    def _check_that_tuple_contains_integers_only(self, events_tuple):
        """
        Check if all elements inside the event tuple are integers.

        Parameters
        ----------
        events_tuple : tuple
            Tuple containing event boundary events for read in.

        Raises
        ------
        TypeError
            If one or more elements inside the event tuple are not integers.
        """
        if not all(isinstance(event, int) for event in events_tuple):
            raise TypeError("All elements inside the event tuple must be integers.")
        
    def _skip_lines(self, fname):
        """
        Once a file is opened with :code:`open()`, this method skips the
        initial header and comment lines such that the first line called with
        :code:`fname.readline()` is the first particle in the first event.

        Parameters
        ----------
        fname : variable name
            Name of the variable for the file opened with the :code:`open()`
            command.

        """
        num_skip = self._get_num_skip_lines()
        for i in range(0, num_skip):
            fname.readline()