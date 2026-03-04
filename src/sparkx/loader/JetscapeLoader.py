# ===================================================
#
#    Copyright (c) 2024-2026
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from sparkx.loader.BaseLoader import BaseLoader
from sparkx.Filter import *
from typing import Dict, List, Tuple, Union, Any


class JetscapeLoader(BaseLoader):
    """
    A class to load and process JETSCAPE data files.

    Attributes
    ----------
    PATH_JETSCAPE_ : str
        The path to the JETSCAPE data file.
    particle_type_ : str
        The type of particles to load ('hadron' or 'parton').

    Methods
    -------
    load(**kwargs)
        Loads the data from the JETSCAPE file based on the specified optional arguments.
    get_last_line(file_path)
        Returns the last line of a file.
    get_sigmaGen()
        Retrieves the sigmaGen values with error from the last line of a file.
    get_particle_type_defining_string()
        Returns the string which defines the particle type in the JETSCAPE file.
    get_particle_type()
        Returns the particle type of the JETSCAPE file.
    get_event_header_information()
        Returns the event header information of the JETSCAPE file.
    """

    def __init__(self, JETSCAPE_FILE: str):
        """
        Constructor for the JetscapeLoader class.

        This method initializes an instance of the JetscapeLoader class.
        It checks if the provided file ends with '.dat' and that the last
        line contains the string 'sigmaGen'.

        Parameters
        ----------
        JETSCAPE_FILE : str
            The path to the JETSCAPE data file.

        Raises
        ------
        FileNotFoundError
            If the file is not found or does not end with '.dat'.
        ValueError
            If the last line of the JETSCAPE file does not contain the string 'sigmaGen'.

        Returns
        -------
        None
        """
        if not ".dat" in JETSCAPE_FILE:
            raise FileNotFoundError("File not found or does not end with .dat")
        # Check that the last line contains the string 'sigmaGen'
        if "sigmaGen" not in self.get_last_line(JETSCAPE_FILE):
            raise ValueError(
                "The last line of the Jetscape file does not contain the string 'sigmaGen'"
            )

        self.PATH_JETSCAPE_ = JETSCAPE_FILE
        self.particle_type_ = "hadron"
        self.particle_type_defining_string_ = "N_hadrons"
        self.optional_arguments_: Any = {}
        self.event_end_lines_: List[str] = []
        self.num_output_per_event_: np.ndarray = np.array([])
        self.event_header_information_: List[Dict[str, Union[int, float]]] = []
        self.num_events_: int = 0

    def load(
        self, **kwargs: Any
    ) -> Tuple[List[List[Particle]], int, np.ndarray, List[str]]:
        """
        Loads the data from the JETSCAPE file based on the specified optional arguments.

        This method reads the JETSCAPE data file and applies any filters specified in the 'filters' key of the kwargs dictionary. It also adjusts the number of events and the number of output lines per event based on the 'events' key in the kwargs dictionary. If the 'particletype' key is specified, it sets the particle type to either 'hadron' or 'parton'. If any other keys are specified in the kwargs dictionary, it raises a ValueError.

        Parameters
        ----------
        kwargs : dict
            A dictionary of optional arguments. The following keys are recognized:

            - 'events': Either a tuple of two integers specifying the range of events to load, or a single integer specifying a single event to load.
            - 'filters': A list of filters to apply to the data.
            - 'particletype': A string specifying the type of particles to load ('hadron' or 'parton').

        Raises
        ------
        ValueError
            If an unknown keyword argument is used, if the first value of the 'events' tuple is larger than the second value, if an event number is negative, or if the 'particletype' is not 'hadron' or 'parton'.
        TypeError
            If the 'events' key is not a tuple or an integer, or if the 'particletype' key is not a string.

        Returns
        -------
        tuple
            A tuple containing the list of Particle objects loaded from the JETSCAPE data file, the number of events, and the number of output lines per event.
        """
        self.optional_arguments_ = kwargs
        self.event_end_lines_ = []

        for keys in self.optional_arguments_.keys():
            if keys not in ["events", "filters", "particletype"]:
                raise ValueError("Unknown keyword argument used in constructor")

        if "events" in self.optional_arguments_.keys() and isinstance(
            self.optional_arguments_["events"], tuple
        ):
            self._check_that_tuple_contains_integers_only(
                self.optional_arguments_["events"]
            )
            if (
                self.optional_arguments_["events"][0]
                > self.optional_arguments_["events"][1]
            ):
                raise ValueError(
                    "First value of event number tuple must be smaller than second value"
                )
            elif (
                self.optional_arguments_["events"][0] < 0
                or self.optional_arguments_["events"][1] < 0
            ):
                raise ValueError("Event numbers must be non-negative")
        elif "events" in self.optional_arguments_.keys() and isinstance(
            self.optional_arguments_["events"], int
        ):
            if self.optional_arguments_["events"] < 0:
                raise ValueError("Event number must be non-negative")
        elif "events" in self.optional_arguments_.keys():
            raise TypeError(
                'Value given as flag "events" is not of type '
                + "int or a tuple of two int values"
            )

        if "particletype" in self.optional_arguments_.keys() and isinstance(
            self.optional_arguments_["particletype"], str
        ):
            if (self.optional_arguments_["particletype"] == "hadron") or (
                self.optional_arguments_["particletype"] == "parton"
            ):
                self.particle_type_ = self.optional_arguments_["particletype"]
            else:
                raise ValueError(
                    "'particletype' has to be 'hadron' or 'parton'"
                )
        elif (
            "particletype" in self.optional_arguments_.keys()
            and not isinstance(self.optional_arguments_["particletype"], str)
        ):
            raise TypeError("'particletype' is not given as a string value")

        if self.particle_type_ == "hadron":
            self.particle_type_defining_string_ = "N_hadrons"
        else:
            self.particle_type_defining_string_ = "N_partons"

        return self._single_pass_load(kwargs)

    # PRIVATE CLASS METHODS

    def _single_pass_load(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[List[List[Particle]], int, np.ndarray, List[str]]:
        """
        Loads all data from the JETSCAPE file in a single pass.

        This method reads the file once, simultaneously collecting event
        metadata (event headers, particle counts) and building Particle
        objects.

        When the ``events`` keyword is specified, events outside the
        requested range are skipped without creating Particle objects.
        Validation of event ranges is done on the fly: if the file ends
        before the requested events are found, an IndexError is raised.

        Parameters
        ----------
        kwargs : dict
            A dictionary of optional arguments. The following keys are recognized:

            - 'events': Either a tuple of two integers specifying the range of
              events to load, or a single integer specifying a single event.
            - 'filters': A dictionary of filters to apply per event.

        Raises
        ------
        IndexError
            If the requested event range exceeds the file contents.
        ValueError
            If the first event header line is malformed.

        Returns
        -------
        tuple
            A tuple of (particle_list, num_events, num_output_per_event, []).
        """
        # Determine which events to keep
        event_start: int = 0
        event_end: Union[int, None] = None  # None means read all
        if "events" in self.optional_arguments_:
            ev_arg = self.optional_arguments_["events"]
            if isinstance(ev_arg, int):
                event_start = ev_arg
                event_end = ev_arg
            elif isinstance(ev_arg, tuple):
                event_start = ev_arg[0]
                event_end = ev_arg[1]

        has_filters = "filters" in self.optional_arguments_

        particle_list: List[List[Particle]] = []
        event_output: List[List[int]] = []
        event_header_information: List[Dict[str, Union[int, float]]] = []
        data: List[Particle] = []
        cut_events: int = 0
        current_event_idx: int = -1  # 0-based index of current event in file
        in_requested_range: bool = False
        first_event_line: bool = True

        with open(self.PATH_JETSCAPE_, "r") as f:
            # Skip the very first header line
            f.readline()

            for line in f:
                # End-of-file line with sigmaGen
                if "#" in line and "sigmaGen" in line:
                    # Finalize the last event if we were collecting it
                    if in_requested_range and current_event_idx >= 0:
                        particle_list, cut_events = self._finalize_event(
                            data,
                            particle_list,
                            cut_events,
                            event_output,
                            has_filters,
                            kwargs,
                        )
                        data = []
                    # We've reached the end of the file
                    break

                # Event header line
                if "#" in line and self.particle_type_defining_string_ in line:
                    # Finalize previous event if we were collecting it
                    if in_requested_range and current_event_idx >= 0:
                        particle_list, cut_events = self._finalize_event(
                            data,
                            particle_list,
                            cut_events,
                            event_output,
                            has_filters,
                            kwargs,
                        )
                        data = []

                    current_event_idx += 1

                    # Parse header
                    line_str = (
                        line.replace("\n", "").replace("\t", " ").split(" ")
                    )
                    line_str = line_str[1:]  # ignore the leading '#'
                    event_header_data: Dict[str, Union[int, float]] = {}
                    for i in range(0, len(line_str), 2):
                        if (line_str[i] == "Event") or (
                            line_str[i] == self.particle_type_defining_string_
                        ):
                            event_header_data[line_str[i]] = int(
                                line_str[i + 1]
                            )
                        else:
                            event_header_data[line_str[i]] = float(
                                line_str[i + 1]
                            )

                    # Check if this event is in the requested range
                    if current_event_idx < event_start:
                        in_requested_range = False
                        continue
                    if event_end is not None and current_event_idx > event_end:
                        in_requested_range = False
                        # We're past the requested range; we can stop reading
                        break

                    in_requested_range = True
                    first_event_line = True

                    event_num = int(event_header_data["Event"])
                    num_output = int(
                        event_header_data[self.particle_type_defining_string_]
                    )
                    event_output.append([event_num, num_output])
                    event_header_information.append(event_header_data)
                    continue

                # Data line (particle)
                if in_requested_range:
                    if first_event_line:
                        first_event_line = False
                    line_list = (
                        line.replace("\n", "").replace("\t", " ").split(" ")
                    )
                    particle = Particle("JETSCAPE", np.asarray(line_list))
                    data.append(particle)

        # Validate that the requested events were found
        if event_end is not None and current_event_idx < event_end:
            raise IndexError(
                f"Requested events up to {event_end} but the file only "
                f"contains {current_event_idx + 1} events."
            )

        # When filters cut some events, renumber survivors sequentially
        # starting from 1 (Jetscape convention) so that post-constructor
        # filters in BaseStorer continue to work correctly.
        if cut_events > 0 and len(event_output) > 0:
            for i in range(len(event_output)):
                event_output[i][0] = i + 1

        # Build final arrays
        if len(event_output) > 0:
            self.num_output_per_event_ = np.array(event_output, dtype=np.int32)
        else:
            self.num_output_per_event_ = np.array([], dtype=np.int32)
        self.num_events_ = len(event_output)
        self.event_header_information_ = event_header_information

        if particle_list == []:
            particle_list = [[]]

        return (
            particle_list,
            self.num_events_,
            self.num_output_per_event_,
            [],
        )

    def _finalize_event(
        self,
        data: List[Particle],
        particle_list: List[List[Particle]],
        cut_events: int,
        event_output: List[List[int]],
        has_filters: bool,
        kwargs: Dict[str, Any],
    ) -> Tuple[List[List[Particle]], int]:
        """
        Finalize an event by applying filters and appending to particle_list.

        Parameters
        ----------
        data : list
            List of Particle objects for the current event.
        particle_list : list
            Accumulated list of events.
        cut_events : int
            Number of events cut so far.
        event_output : list
            List of [event_num, num_output] pairs.
        has_filters : bool
            Whether filters should be applied.
        kwargs : dict
            Original kwargs containing filter definitions.

        Returns
        -------
        tuple
            Updated (particle_list, cut_events).
        """
        old_data_len = len(data)
        if has_filters:
            data = self.__apply_kwargs_filters([data], kwargs["filters"])[0]

        if len(data) != 0 or old_data_len == 0:
            # Update the particle count for this event
            event_output[-1][1] = len(data)
            particle_list.append(data)
        else:
            # Event was entirely filtered out
            event_output.pop()
            cut_events += 1

        return particle_list, cut_events

    def event_end_lines(self) -> List[str]:
        """
        Returns the list of event end lines.

        This method returns the list of strings that mark the end of events
        in the Jetscape data.

        Returns
        -------
        List[str]
            A list of strings that mark the end of events in the Jetscape data.
        """
        return self.event_end_lines_

    def __apply_kwargs_filters(
        self, event: List[List[Particle]], filters_dict: Any
    ) -> List[List[Particle]]:
        """
        Applies the specified filters to the event.

        This method applies a series of filters to the event based on the keys
        in the filters_dict dictionary. The filters include
        'charged_particles', 'uncharged_particles',
        'particle_species', 'remove_particle_species',
        'lower_event_energy_cut', 'pT_cut', 'mT_cut', 'rapidity_cut',
        'pseudorapidity_cut', 'multiplicity_cut',
        'particle_status', 'keep_hadrons', 'keep_leptons', 'keep_quarks',
        'keep_mesons', 'keep_baryons', 'keep_up', 'keep_down', 'keep_strange',
        'keep_charm', 'keep_bottom', 'keep_top' and 'remove_photons'.
        If a key in the filters_dict dictionary does not
        match any of these filters, a ValueError is raised.

        Parameters
        ----------
        event : list
            The event to which the filters are applied.
        filters_dict : dict
            A dictionary of filters to apply to the event.
            The keys are the names of the filters and the values are the
            parameters for the filters.

        Raises
        ------
        ValueError
            If a key in the filters_dict dictionary does not match any of the
            supported filters.

        Returns
        -------
        event : list
            The event after the filters have been applied.
        """
        if not isinstance(filters_dict, dict) or len(filters_dict.keys()) == 0:
            return event
        for i in filters_dict.keys():
            if i == "charged_particles":
                if filters_dict["charged_particles"]:
                    event = charged_particles(event)
            elif i == "uncharged_particles":
                if filters_dict["uncharged_particles"]:
                    event = uncharged_particles(event)
            elif i == "particle_species":
                event = particle_species(
                    event, filters_dict["particle_species"]
                )
            elif i == "remove_particle_species":
                event = remove_particle_species(
                    event, filters_dict["remove_particle_species"]
                )
            elif i == "lower_event_energy_cut":
                event = lower_event_energy_cut(
                    event, filters_dict["lower_event_energy_cut"]
                )
            elif i == "pT_cut":
                event = pT_cut(event, filters_dict["pT_cut"])
            elif i == "mT_cut":
                event = mT_cut(event, filters_dict["mT_cut"])
            elif i == "rapidity_cut":
                event = rapidity_cut(event, filters_dict["rapidity_cut"])
            elif i == "pseudorapidity_cut":
                event = pseudorapidity_cut(
                    event, filters_dict["pseudorapidity_cut"]
                )
            elif i == "multiplicity_cut":
                event = multiplicity_cut(
                    event, filters_dict["multiplicity_cut"]
                )
            elif i == "particle_status":
                event = particle_status(event, filters_dict["particle_status"])
            elif i == "keep_hadrons":
                if filters_dict["keep_hadrons"]:
                    event = keep_hadrons(event)
            elif i == "keep_leptons":
                if filters_dict["keep_leptons"]:
                    event = keep_leptons(event)
            elif i == "keep_quarks":
                if filters_dict["keep_quarks"]:
                    event = keep_quarks(event)
            elif i == "keep_mesons":
                if filters_dict["keep_mesons"]:
                    event = keep_mesons(event)
            elif i == "keep_baryons":
                if filters_dict["keep_baryons"]:
                    event = keep_baryons(event)
            elif i == "keep_up":
                if filters_dict["keep_up"]:
                    event = keep_up(event)
            elif i == "keep_down":
                if filters_dict["keep_down"]:
                    event = keep_down(event)
            elif i == "keep_strange":
                if filters_dict["keep_strange"]:
                    event = keep_strange(event)
            elif i == "keep_charm":
                if filters_dict["keep_charm"]:
                    event = keep_charm(event)
            elif i == "keep_bottom":
                if filters_dict["keep_bottom"]:
                    event = keep_bottom(event)
            elif i == "keep_top":
                if filters_dict["keep_top"]:
                    event = keep_top(event)
            elif i == "remove_photons":
                if filters_dict["remove_photons"]:
                    event = remove_photons(event)
            else:
                raise ValueError("The cut is unknown!")

        return event

    # PUBLIC CLASS METHODS

    def get_last_line(self, file_path: str) -> str:
        """
        Returns the last line of a file.

        Parameters
        ----------
        file_path : str
            The path to the file.

        Returns
        -------
        str
            The last line of the file, stripped of leading and trailing whitespace.
        """
        with open(file_path, "rb") as file:
            file.seek(-2, 2)
            while file.read(1) != b"\n":
                file.seek(-2, 1)
            last_line = file.readline().decode().strip()
            return last_line

    def get_sigmaGen(self) -> Tuple[float, float]:
        """
        Retrieves the sigmaGen values with error from the last line of a file.

        Returns
        -------
        tuple
            A tuple containing the first and second sigmaGen values as floats.
        """
        last_line = self.get_last_line(self.PATH_JETSCAPE_)
        words = last_line.split()
        numbers = []

        for word in words:
            try:
                number = float(word)
                numbers.append(number)
                if len(numbers) == 2:
                    break
            except ValueError:
                continue

        return (numbers[0], numbers[1])

    def get_particle_type_defining_string(self) -> str:
        """
        Returns the string which defines the particle type in the Jetscape file.

        Returns
        -------
        string
            The particle type defining string.
        """
        return self.particle_type_defining_string_

    def get_event_header_information(
        self,
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Returns the event header information of the Jetscape file. They are read
        as a list of dictionaries from event header lines of the file.

        Returns
        -------
        list
            A list containing the event header information in a dictionary for
            each event.
            The keys of the dictionary are the names of the header information.
        """
        return self.event_header_information_

    def get_particle_type(self) -> str:
        """
        Returns the particle type of the Jetscape file.

        Returns
        -------
        string
            The particle type.
        """
        return self.particle_type_
