# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import numpy as np
import warnings
from typing import Union, List


class CentralityClasses:
    """
    Class for defining centrality classes based on event multiplicity.

    .. note::
        It is the user's responsibility to ensure that the amount of events used
        to determine the centrality classes is sufficient to provide reliable
        results. The recommended minimum number of events is at least a few
        hundred.

    Parameters
    ----------
    events_multiplicity : list or numpy.ndarray
        List or array containing the multiplicity values for each event.

    centrality_bins : list or numpy.ndarray
        List or array defining the boundaries of centrality classes as percentages.

    Raises
    ------
    TypeError
        If :code:`events_multiplicity` or :code:`centrality_bins` is not a list
        or :code:`numpy.ndarray`.

    Attributes
    ----------
    events_multiplicity_ : list or numpy.ndarray
        Stores the input multiplicity values for each event.
    centrality_bins_ : list or numpy.ndarray
        Stores the input boundaries of centrality classes.
    dNchdetaMin_ : list
        Minimum values of multiplicity for each centrality class.
    dNchdetaMax_ : list
        Maximum values of multiplicity for each centrality class.
    dNchdetaAvg_ : list
        Average values of multiplicity for each centrality class.
    dNchdetaAvgErr_ : list
        Average errors of multiplicity for each centrality class.

    Methods
    -------
    get_centrality_class:
        Return the index of the centrality bin for a given multiplicity value.
    output_centrality_classes:
        Write centrality class information to a file.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx import *
        >>> # ============================================================
        >>> # 1. Load and filter input particles from simulation output
        >>> # ============================================================

        >>> # You can obtain a list of Particle objects using either the
        >>> # Oscar or Jetscape classes. Alternatively, you may use a 
        >>> # custom user-defined list of Particle objects.
        >>> # 
        >>> # In this example, we use the Oscar class and filter the output
        >>> # to include only charged particles.
        >>>
        >>> OSCAR_FILE_PATH = [Oscar_directory]/particle_lists.oscar
        >>> oscar = Oscar(OSCAR_FILE_PATH, filters={'charged_particles': True})
        >>> number_particles_event = oscar.num_output_per_event()
        >>> # Keep multiplicities from events with non-zero multiplicity only
        >>> multiplicities = number_particles_event[number_particles_event[:, 1] != 0, 1]

        
        >>> # =====================================
        >>> # 2. Calculate centrality classes
        >>> # =====================================
        >>> # Define the centrality class boundaries (in percent)
        >>> centrality_bins = [0, 10, 30, 50, 70, 90, 100]
        >>> centrality_obj = CentralityClasses(events_multiplicity=multiplicities,
        ...                                   centrality_bins=centrality_bins)
        >>> # Get the centrality class for a specific event multiplicity value
        >>> # (e.g. 1490)
        >>> # The function returns the index of the centrality bin in `centrality_bins`.
        >>> centrality_obj.get_centrality_class(1490)
        0
        >>> # This event has index 0, which corresponds to the most central bin (0-10%).
        >>> centrality_obj.output_centrality_classes('centrality_output.txt')
    """

    def __init__(
        self,
        events_multiplicity: Union[List[float], np.ndarray],
        centrality_bins: Union[List[float], np.ndarray],
    ) -> None:
        if not isinstance(events_multiplicity, (list, np.ndarray)):
            raise TypeError(
                "'events_multiplicity' is not list or numpy.ndarray"
            )
        if not isinstance(centrality_bins, (list, np.ndarray)):
            raise TypeError("'centrality_bins' is not list or numpy.ndarray")
        # Check if centrality_bins is sorted
        if not all(
            centrality_bins[i] <= centrality_bins[i + 1]
            for i in range(len(centrality_bins) - 1)
        ):
            warnings.warn(
                "'centrality_bins' is not sorted. Sorting automatically."
            )
            centrality_bins.sort()

        # Check for uniqueness of values
        # Remove duplicates from the list
        unique_bins = []
        seen = set()
        multiple_same_entries = False
        for item in centrality_bins:
            if item not in seen:
                unique_bins.append(item)
                seen.add(item)
            else:
                multiple_same_entries = True

        if multiple_same_entries:
            warnings.warn(
                "'centrality_bins' contains duplicate values. They are removed automatically."
            )

        # Check for negative values and values greater than 100
        if any(value < 0.0 or value > 100.0 for value in centrality_bins):
            raise ValueError(
                "'centrality_bins' contains values less than 0 or greater than 100."
            )

        self.events_multiplicity_ = events_multiplicity
        self.centrality_bins_ = unique_bins

        self.dNchdetaMin_: List[float] = []
        self.dNchdetaMax_: List[float] = []
        self.dNchdetaAvg_: List[float] = []
        self.dNchdetaAvgErr_: List[float] = []

        self.__create_centrality_classes()

    def __create_centrality_classes(self) -> None:
        """
        Create centrality classes based on event multiplicity.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of events is less than 4.
            If the multiplicity in 'events_multiplicity' is negative.

        Notes
        -----
        This function creates four sub-samples for error determination based on the
        event multiplicity. The numbers are distributed evenly to the four sub-samples,
        which are then sorted in descending order. The averages and errors are
        calculated for each sub-sample. Finally, the events are sorted by multiplicity,
        and boundaries of centrality classes are determined.
        """
        number_events = len(self.events_multiplicity_)
        if number_events < 4:
            raise ValueError("The number of events has to be larger than 4")

        # create four sub samples for the error determination
        event_sample_A = []
        event_sample_B = []
        event_sample_C = []
        event_sample_D = []

        # distribute the numbers evenly
        for i, multiplicity in enumerate(self.events_multiplicity_):
            if multiplicity < 0:
                raise ValueError(
                    "Multiplicity in 'events_multiplicity' is negative"
                )
            if i % 4 == 0:
                event_sample_A.append(multiplicity)
            elif i % 4 == 1:
                event_sample_B.append(multiplicity)
            elif i % 4 == 2:
                event_sample_C.append(multiplicity)
            elif i % 4 == 3:
                event_sample_D.append(multiplicity)

        event_sample_A = sorted(event_sample_A, reverse=True)
        event_sample_B = sorted(event_sample_B, reverse=True)
        event_sample_C = sorted(event_sample_C, reverse=True)
        event_sample_D = sorted(event_sample_D, reverse=True)

        MinRecord = int(number_events / 4 * self.centrality_bins_[0] / 100.0)
        for i in range(1, len(self.centrality_bins_)):
            MaxRecord = int(
                number_events / 4 * self.centrality_bins_[i] / 100.0
            )

            AvgA = np.mean(event_sample_A[MinRecord:MaxRecord])
            AvgB = np.mean(event_sample_B[MinRecord:MaxRecord])
            AvgC = np.mean(event_sample_C[MinRecord:MaxRecord])
            AvgD = np.mean(event_sample_D[MinRecord:MaxRecord])

            Avg = (AvgA + AvgB + AvgC + AvgD) / 4.0
            Err = np.sqrt(
                (
                    (AvgA - Avg) ** 2
                    + (AvgB - Avg) ** 2
                    + (AvgC - Avg) ** 2
                    + (AvgD - Avg) ** 2
                )
                / 3.0
            )

            self.dNchdetaAvg_.append(Avg)
            self.dNchdetaAvgErr_.append(Err)

            MinRecord = MaxRecord

        # sort events by multiplicity and determine boundaries of centrality
        # classes
        global_event_record = sorted(self.events_multiplicity_, reverse=True)

        MinRecord = int(number_events * self.centrality_bins_[0] / 100.0)
        for i in range(1, len(self.centrality_bins_)):
            MaxRecord = int(number_events * self.centrality_bins_[i] / 100.0)

            self.dNchdetaMax_.append(global_event_record[MinRecord])
            self.dNchdetaMin_.append(global_event_record[MaxRecord - 1])

            MinRecord = MaxRecord

    def get_centrality_class(self, dNchdEta: float) -> int:
        """
        This function determines the index of the centrality bin for a given
        multiplicity value based on the predefined centrality classes.

        In the case that the multiplicity input exceeds the largest or smallest
        value of the multiplicity used to determine the centrality classes, the
        function returns the index of the most central or most peripheral bin,
        respectively.

        Parameters
        ----------
        dNchdEta : float
            Multiplicity value.

        Returns
        -------
        int
            Index of the centrality bin.
        """
        # check if the multiplicity is in the most central bin
        if dNchdEta >= self.dNchdetaMin_[0]:
            return 0
        # check if the multiplicity is in the most peripheral bin
        elif dNchdEta < self.dNchdetaMin_[len(self.dNchdetaMin_) - 2]:
            return len(self.dNchdetaMin_) - 1
        # check if the multiplicity is in one of the intermediate bins
        else:
            for i in range(1, len(self.dNchdetaMin_) - 1):
                if (dNchdEta >= self.dNchdetaMin_[i]) and (
                    dNchdEta < self.dNchdetaMin_[i - 1]
                ):
                    return i
        # default case to satisfy mypy's requirement for a return statement
        return -1

    def output_centrality_classes(self, fname: str) -> None:
        """
        Write centrality class information to a file.

        Parameters
        ----------
        fname : str
            Name of the output file.

        Raises
        ------
        TypeError
            If :code:`fname` is not a string.

        Notes
        -----
        This function writes the centrality class information, including minimum,
        maximum, average multiplicities, and average errors, to the specified file.
        """
        # Check if fname is a string
        if not isinstance(fname, str):
            raise TypeError("'fname' should be a string.")
        # Write the information to the file
        with open(fname, "w") as out_stream:
            out_stream.write(
                "# CentralityMin CentralityMax dNchdEtaMin dNchdEtaMax dNchdEtaAvg dNchdEtaAvgErr\n"
            )

            for i in range(1, len(self.dNchdetaMin_)):
                out_stream.write(
                    f"{self.centrality_bins_[i - 1]} - {self.centrality_bins_[i]} "
                    f"{self.dNchdetaMin_[i - 1]} {self.dNchdetaMax_[i - 1]} "
                    f"{self.dNchdetaAvg_[i - 1]} {self.dNchdetaAvgErr_[i - 1]}\n"
                )
