# ===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from sparkx.loader.BaseLoader import BaseLoader
from sparkx.Filter import *
from sparkx.Particle import Particle
import os
from typing import Dict, List, Tuple, Optional, Union, Any


class OscarLoader(BaseLoader):
    """
    OscarLoader is a class for loading and processing OSCAR format data files.

    This class extends the BaseLoader and provides functionality to load, filter, and process data from files in the OSCAR format. It supports different versions of the OSCAR format and allows for event-based data loading with optional filtering.

    Attributes
    ----------
    PATH_OSCAR_ : str
        The path to the OSCAR data file.
    oscar_format_ : str
        The format of the OSCAR data file.
    optional_arguments_ : dict
        A dictionary of optional arguments passed to the load method.
    event_end_lines_ : list
        A list to store the end lines of events in the OSCAR data file.
    num_events_ : int
        The number of events in the OSCAR data file.
    num_output_per_event_ : numpy.ndarray
        An array containing the number of particles per event in the OSCAR data file.

    Methods
    -------
    __init__(OSCAR_FILE)
        Initializes the OscarLoader with the provided OSCAR file path.
    load(**kwargs)
        Loads the OSCAR data with optional event ranges and filters.
    _get_num_skip_lines()
        Calculates the number of initial lines to skip in the OSCAR file.
    set_num_events()
        Sets the number of events in the OSCAR data file.
    set_oscar_format()
        Determines and sets the format of the OSCAR data file.
    oscar_format()
        Returns the OSCAR format of the data file.
    impact_parameter()
        Returns the impact parameter of the events in the OSCAR data file.
    event_end_lines()
        Returns the event end lines in the OSCAR data file.
    __get_num_read_lines()
        Calculates the number of lines to read based on the specified events.
    __apply_kwargs_filters(event, filters_dict)
        Applies filters to the event data based on the provided filter dictionary.
    set_particle_list(kwargs)
        Sets the list of particles from the OSCAR data file.
    set_num_output_per_event_and_event_footers()
        Determines the number of output lines per event and the event footers in the OSCAR data file.
    """

    PATH_OSCAR_: str
    oscar_format_: Optional[str]
    optional_arguments_: Dict[str, Any]
    event_end_lines_: List[str]
    num_events_: int
    num_output_per_event_: np.ndarray
    custom_attr_list: List[str]

    def __init__(self, OSCAR_FILE: str):
        """
        Constructor for the OscarLoader class.

        This method initializes an instance of the OscarLoader class. It checks if the provided file is in the OSCAR format (i.e., it has the '.oscar' extension). If not, it raises a FileNotFoundError.

        Parameters
        ----------
        OSCAR_FILE : str
            The path to the OSCAR data file.

        Raises
        ------
        FileNotFoundError
            If the input file does not have the '.oscar' or '.dat' extension.

        Returns
        -------
        None
        """
        if not ".oscar" in OSCAR_FILE and not ".dat" in OSCAR_FILE:
            raise FileNotFoundError(
                "Input file is not in the OSCAR format. Input "
                "file must have the ending .oscar or .dat"
            )

        self.PATH_OSCAR_ = OSCAR_FILE
        self.oscar_format_ = None
        self.custom_attr_list = []

    def load(
        self, **kwargs: Dict[str, Any]
    ) -> Tuple[List[List[Particle]], int, np.ndarray, List[str]]:
        """
        Loads the OSCAR data from the specified file.

        This method accepts optional arguments that specify which events to load and any filters to apply. The 'events' argument can be either a tuple specifying a range of events or an integer specifying a single event. If the 'events' argument is provided and is not a tuple or an integer, or if any of the event numbers are negative, a ValueError is raised.

        Parameters
        ----------
        **kwargs : dict, optional
            A dictionary of optional arguments. The following keys are recognized:

            - 'events': Either a tuple of two integers specifying the range of events to load, or a single integer specifying a single event to load.
            - 'filters': A list of filters to apply to the data.

        Raises
        ------
        ValueError
            If an unrecognized keyword argument is used in the constructor, or if the 'events' argument is not a tuple or an integer, or if any of the event numbers are negative.

        Returns
        -------
        tuple
            A tuple containing the following elements:

            - A list of particles loaded from the OSCAR data.
            - The number of events in the OSCAR data.
            - The number of particles in each event in the OSCAR data.
        """
        self.optional_arguments_ = kwargs
        self.event_end_lines_ = []

        for keys in self.optional_arguments_.keys():
            if keys not in ["events", "filters"]:
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

        self.set_oscar_format()
        self.set_num_events()
        self.set_num_output_per_event_and_event_footers()
        return (
            self.set_particle_list(kwargs),
            self.num_events_,
            self.num_output_per_event_,
            self.custom_attr_list,
        )

    def _get_num_skip_lines(self) -> int:
        """
        Get number of initial lines in Oscar file that are header or comment
        lines and need to be skipped in order to read the particle output.

        Returns
        -------
        skip_lines : int
            Number of initial lines before data.

        """
        if (
            not self.optional_arguments_
            or "events" not in self.optional_arguments_.keys()
        ):
            skip_lines = 3
        elif isinstance(self.optional_arguments_["events"], int):
            if self.optional_arguments_["events"] == 0:
                skip_lines = 3
            else:
                cumulate_lines = 0
                for i in range(0, self.optional_arguments_["events"]):
                    cumulate_lines += self.num_output_per_event_[i, 1] + 2
                skip_lines = 3 + cumulate_lines
        elif isinstance(self.optional_arguments_["events"], tuple):
            line_start = self.optional_arguments_["events"][0]
            if line_start == 0:
                skip_lines = 3
            else:
                cumulate_lines = 0
                for i in range(0, line_start):
                    cumulate_lines += self.num_output_per_event_[i, 1] + 2
                skip_lines = 3 + cumulate_lines
        else:
            raise TypeError(
                'Value given as flag "events" is not of type '
                + "int or a tuple of two int values"
            )

        return skip_lines

    def set_num_events(self) -> None:
        """
        Sets the number of events in the OSCAR data file.

        This method reads the file in binary mode to search for the last line. This approach avoids the need to loop through the entire file, which can be time-consuming for large files. It then checks if the last line starts with a '#' and contains the word 'event'. If it does, it sets the number of events to the integer value in the third position of the last line. If the last line does not meet these conditions, it raises a TypeError.

        Parameters
        ----------
        None

        Raises
        ------
        TypeError
            If the last line of the file does not start with a '#' and contain the word 'event'.

        Returns
        -------
        None
        """
        with open(self.PATH_OSCAR_, "rb") as file:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b"\n":
                file.seek(-2, os.SEEK_CUR)
            last_line = file.readline().decode().split(" ")
        if last_line[0] == "#" and "event" in last_line:
            self.num_events_ = int(last_line[2]) + 1
        else:
            raise TypeError(
                "Input file does not end with a comment line "
                + "including the events. File might be incomplete "
                + "or corrupted."
            )

    def _set_custom_attr_list(self, header_line: List[str]) -> List[str]:
        self.custom_attr_list = []
        attr_map = {
            "t": "t",
            "x": "x",
            "y": "y",
            "z": "z",
            "mass": "mass",
            "p0": "E",
            "px": "px",
            "py": "py",
            "pz": "pz",
            "pdg": "pdg",
            "ID": "ID",
            "charge": "charge",
            "ncoll": "ncoll",
            "form_time": "form_time",
            "xsecfac": "xsecfac",
            "proc_id_origin": "proc_id_origin",
            "proc_type_origin": "proc_type_origin",
            "time_last_coll": "time_last_coll",
            "pdg_mother1": "pdg_mother1",
            "pdg_mother2": "pdg_mother2",
            "baryon_number": "baryon_number",
            "strangeness": "strangeness",
        }
        for i in range(0, len(header_line)):
            attr_name = attr_map.get(header_line[i])
            if attr_name is not None:
                self.custom_attr_list.append(attr_name)
        return self.custom_attr_list

    def set_oscar_format(self) -> None:
        """
        Sets the number of events in the OSCAR data file.

        This method reads the file in binary mode to search for the last line. This approach avoids the need to loop through the entire file, which can be time-consuming for large files. It then checks if the last line starts with a '#' and contains the word 'event'. If it does, it sets the number of events to the integer value in the third position of the last line. If the last line does not meet these conditions, it raises a TypeError.

        Parameters
        ----------
        None

        Raises
        ------
        TypeError
            If the last line of the file does not start with a '#' and contain the word 'event'.

        Returns
        -------
        None
        """
        with open(self.PATH_OSCAR_, "r") as file:
            first_line = file.readline()
            first_line_list = first_line.replace("\n", "").split(" ")

        if len(first_line_list) == 15 or first_line_list[0] == "#!OSCAR2013":
            self.oscar_format_ = "Oscar2013"
        elif (
            first_line_list[0] == "#!OSCAR2013Extended"
            and first_line_list[1] == "SMASH_IC"
        ):
            self.oscar_format_ = "Oscar2013Extended_IC"
        elif (
            first_line_list[0] == "#!OSCAR2013Extended"
            and first_line_list[1] == "Photons"
        ):
            self.oscar_format_ = "Oscar2013Extended_Photons"
        elif (
            len(first_line_list) == 23
            or first_line_list[0] == "#!OSCAR2013Extended"
        ):
            self.oscar_format_ = "Oscar2013Extended"
        elif first_line_list[0] == "#!ASCIICustom":
            self.oscar_format_ = "ASCIICustom"
            value_line = first_line_list[2:]
            self.custom_attr_list = self._set_custom_attr_list(value_line)
        else:
            raise TypeError(
                "Input file must follow the Oscar2013, "
                + "Oscar2013Extended, Oscar2013Extended_IC, Oscar2013Extended_Photons or ASCIICustom format. "
            )

    def oscar_format(self) -> Optional[str]:
        """
        Returns the OSCAR format string.

        This method returns the OSCAR format string that specifies the format
        of the OSCAR data being loaded.

        Returns
        -------
        Optional[str]
            The OSCAR format string, or None if the format is not set.
        """
        return self.oscar_format_

    def event_end_lines(self) -> List[str]:
        """
        Returns the list of event end lines.

        This method returns the list of strings that mark the end of events
        in the OSCAR data.

        Returns
        -------
        List[str]
            A list of strings that mark the end of events in the OSCAR data.
        """
        return self.event_end_lines_

    def impact_parameter(self) -> List[float]:
        """
        Returns the impact parameter of the events.

        This method extracts the impact parameter of the collision from the
        last line of each event in the OSCAR data file.

        Returns
        -------
        List[float]
            The impact parameter of the collisions.
        """
        impact_parameters: List[float] = []
        for line in self.event_end_lines_:
            line_split = line.split(" ")
            line_split = list(filter(None, line_split))
            impact_parameter = float(line_split[-3])
            impact_parameters.append(impact_parameter)

        # update the list with the num_output_per_event_ for the case
        # that only certain events are loaded from the file
        if self.num_output_per_event_.shape[0] == 0:
            impact_parameters = []
        else:
            idx_list = self.num_output_per_event_[:, 0]
            impact_parameters = [impact_parameters[i] for i in idx_list]

        return impact_parameters

    def __get_num_read_lines(self) -> int:
        """
        Calculates the number of lines to read from the OSCAR data file.

        This method determines the number of lines to read based on the 'events' key in the optional arguments. If 'events' is not specified, it sums the number of output lines for all events. If 'events' is an integer, it gets the number of output lines for the specified event. If 'events' is a tuple, it sums the number of output lines for the range of events specified. If 'events' is not an integer or a tuple, it raises a TypeError.

        Parameters
        ----------
        None

        Raises
        ------
        TypeError
            If the value given as flag events is not of type int or a tuple.

        Returns
        -------
        cumulated_lines : int
            The total number of lines to read from the OSCAR data file.
        """
        if (
            not self.optional_arguments_
            or "events" not in self.optional_arguments_.keys()
        ):
            cumulated_lines = np.sum(self.num_output_per_event_, axis=0)[1]
            # add number of comments
            cumulated_lines += int(2 * len(self.num_output_per_event_))

        elif isinstance(self.optional_arguments_["events"], int):
            read_event = self.optional_arguments_["events"]
            cumulated_lines = int(self.num_output_per_event_[read_event, 1] + 2)

        elif isinstance(self.optional_arguments_["events"], tuple):
            cumulated_lines = 0
            event_start = self.optional_arguments_["events"][0]
            event_end = self.optional_arguments_["events"][1]
            for i in range(event_start, event_end + 1):
                cumulated_lines += int(self.num_output_per_event_[i, 1] + 2)
        else:
            raise TypeError(
                "Value given as flag events is not of type int or a tuple"
            )

        return cumulated_lines

    def __apply_kwargs_filters(
        self, event: List[List[Particle]], filters_dict: Dict[str, Any]
    ) -> List[List[Particle]]:
        """
        Applies the specified filters to the given event.

        This method applies a series of filters to the event data based on the
        keys in the filters_dict dictionary. The filters include:
        'charged_particles', 'uncharged_particles',
        'particle_species', 'remove_particle_species', 'participants',
        'spectators', 'lower_event_energy_cut', 'spacetime_cut', 'pT_cut',
        'mT_cut', 'rapidity_cut', 'pseudorapidity_cut',
        'spacetime_rapidity_cut', 'multiplicity_cut', 'keep_hadrons',
        'keep_leptons', 'keep_mesons', 'keep_baryons', 'keep_up', 'keep_down',
        'keep_strange', 'keep_charm', 'keep_bottom', 'keep_top' and
        'remove_photons'.
        If a key in the filters_dict dictionary does not
        match any of these filters, a ValueError is raised.

        Parameters
        ----------
        event : list
            The event data to be filtered.
        filters_dict : dict
            A dictionary of filters to apply to the event data. The keys of the
            dictionary specify the filters to apply, and the values specify the
            parameters for the filters.

        Raises
        ------
        ValueError
            If a key in the filters_dict dictionary does not match any of the
            supported filters.

        Returns
        -------
        event : list
            The filtered event data.
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
            elif i == "participants":
                if filters_dict["participants"]:
                    event = participants(event)
            elif i == "spectators":
                if filters_dict["spectators"]:
                    event = spectators(event)
            elif i == "lower_event_energy_cut":
                event = lower_event_energy_cut(
                    event, filters_dict["lower_event_energy_cut"]
                )
            elif i == "spacetime_cut":
                if not isinstance(filters_dict["spacetime_cut"], list):
                    raise ValueError(
                        "The spacetime cut filter requires a list of two values."
                    )
                event = spacetime_cut(
                    event,
                    filters_dict["spacetime_cut"][0],
                    filters_dict["spacetime_cut"][1],
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
            elif i == "spacetime_rapidity_cut":
                event = spacetime_rapidity_cut(
                    event, filters_dict["spacetime_rapidity_cut"]
                )
            elif i == "multiplicity_cut":
                event = multiplicity_cut(
                    event, filters_dict["multiplicity_cut"]
                )
            elif i == "keep_hadrons":
                if filters_dict["keep_hadrons"]:
                    event = keep_hadrons(event)
            elif i == "keep_leptons":
                if filters_dict["keep_leptons"]:
                    event = keep_leptons(event)
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

    def set_particle_list(self, kwargs: Dict[str, Any]) -> List[List[Particle]]:
        """
        Sets the list of particles from the OSCAR data file.

        This method reads the OSCAR data file line by line and creates a list
        of Particle objects. It also applies any filters specified in the
        'filters' key of the kwargs dictionary. If the 'events' key is
        specified in the kwargs dictionary, it adjusts the number of events and
        the number of output lines per event accordingly.

        Parameters
        ----------
        kwargs : dict
            A dictionary of optional arguments. The following keys are recognized:

            - 'events': Either a tuple of two integers specifying the range of events to load, or a single integer specifying a single event to load.
            - 'filters': A list of filters to apply to the data.

        Raises
        ------
        IndexError
            If the number of events in the OSCAR file does not match the number
            of events specified by the comments in the OSCAR file, or if the
            index is out of range of the OSCAR file.
        ValueError
            If the first line of the event is not a comment line or does not
            contain "out", or if a comment line is unexpectedly found.

        Returns
        -------
        particle_list : list
            A list of Particle objects loaded from the OSCAR data file.
        """
        particle_list: List[List[Particle]] = []
        data: List[Particle] = []
        num_read_lines = self.__get_num_read_lines()
        cut_events = 0
        with open(self.PATH_OSCAR_, "r") as oscar_file:
            self._skip_lines(oscar_file)
            for i in range(0, num_read_lines):
                line = oscar_file.readline()
                if not line:
                    raise IndexError(
                        "Index out of range of OSCAR file. This most likely happened because "
                        + "the particle number specified by the comments in the OSCAR "
                        + "file differs from the actual number of particles in the event."
                    )
                elif i == 0 and "#" not in line and "out" not in line:
                    raise ValueError(
                        "First line of the event is not a comment "
                        + 'line or does not contain "out"'
                    )
                elif "event" in line and (
                    "out" in line or "in " in line or " start" in line
                ):
                    continue
                elif "#" in line and "end" in line:
                    old_data_len = len(data)
                    if "filters" in self.optional_arguments_.keys():
                        data = self.__apply_kwargs_filters(
                            [data], kwargs["filters"]
                        )[0]
                        if len(data) != 0 or old_data_len == 0:
                            self.num_output_per_event_[len(particle_list)] = (
                                len(particle_list),
                                len(data),
                            )
                        else:
                            self.num_output_per_event_ = np.atleast_2d(
                                np.delete(
                                    self.num_output_per_event_,
                                    len(particle_list),
                                    axis=0,
                                )
                            )
                            if self.num_output_per_event_.shape[0] == 0:
                                self.num_output_per_event_ = np.array([])
                            elif (
                                len(particle_list)
                                < self.num_output_per_event_.shape[0]
                            ):
                                self.num_output_per_event_[
                                    len(particle_list) :, 0
                                ] -= 1
                    if len(data) != 0 or old_data_len == 0:
                        particle_list.append(data)
                    else:
                        cut_events = cut_events + 1
                    data = []
                elif "#" in line:
                    raise ValueError("Comment line unexpectedly found: " + line)
                else:
                    line_list = np.asarray(line.replace("\n", "").split(" "))
                    if self.oscar_format_ == "ASCIICustom":
                        particle = Particle(
                            self.oscar_format_, line_list, self.custom_attr_list
                        )
                    else:
                        particle = Particle(self.oscar_format_, line_list)
                    data.append(particle)
        self.num_events_ = self.num_events_ - cut_events
        # Correct num_output_per_event and num_events
        if not kwargs or "events" not in self.optional_arguments_.keys():
            if len(particle_list) != self.num_events_:
                raise IndexError(
                    "Number of events in OSCAR file does not match the "
                    + "number of events specified by the comments in the "
                    + "OSCAR file!"
                )
        elif isinstance(kwargs["events"], int):
            update = self.num_output_per_event_[kwargs["events"]]
            self.num_output_per_event_ = np.array([update])
            self.num_events_ = int(1)
        elif isinstance(kwargs["events"], tuple):
            event_start = kwargs["events"][0]
            event_end = kwargs["events"][1]
            update = self.num_output_per_event_[event_start : event_end + 1]
            self.num_output_per_event_ = update
            self.num_events_ = int(event_end - event_start + 1)

        if particle_list == []:
            particle_list = [[]]

        return particle_list

    def set_num_output_per_event_and_event_footers(self) -> None:
        """
        Sets the number of output lines per event and the event footers in the
        OSCAR data file.

        This method reads the OSCAR data file line by line and determines the
        number of output lines for each event and the event footers. The method
        behaves differently depending on the OSCAR format of the data file.
        If the format is 'Oscar2013Extended_IC' or 'Oscar2013Extended_Photons',
        it counts the number of lines between 'in' and 'end' lines. Otherwise,
        it counts the number of lines between 'out' and 'end' lines.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        None
        """
        event_output: List[List[Union[str, int]]] = []

        with open(self.PATH_OSCAR_, "r") as oscar_file:
            line_counter: int
            event: Union[int, str]
            line_str: List[str]
            if (
                self.oscar_format_ != "Oscar2013Extended_IC"
                and self.oscar_format_ != "Oscar2013Extended_Photons"
            ):
                while True:
                    line = oscar_file.readline()
                    if not line:
                        break
                    elif "#" in line and " end " in line:
                        self.event_end_lines_.append(line)
                    elif "#" in line and " out " in line:
                        line_str = line.replace("\n", "").split(" ")
                        event = line_str[2]
                        num_output: int = int(line_str[4])
                        event_output.append([event, num_output])
                    else:
                        continue
            elif self.oscar_format_ == "Oscar2013Extended_IC":
                line_counter = 0
                event = 0
                while True:
                    line_counter += 1
                    line = oscar_file.readline()
                    if not line:
                        break
                    elif "#" in line and " end" in line:
                        self.event_end_lines_.append(line)
                        event_output.append([event, line_counter - 1])
                    elif "#" in line and (" in " in line or " start" in line):
                        line_str = line.replace("\n", "").split(" ")
                        event = int(line_str[2])
                        line_counter = 0
                    else:
                        continue

            elif self.oscar_format_ == "Oscar2013Extended_Photons":
                line_counter = 0
                event = 0
                line_memory: int = 0
                while True:
                    line_counter += 1
                    line_memory += 1
                    line = oscar_file.readline()
                    if not line:
                        break
                    elif "#" in line and " end " in line:
                        if line_memory == 1:
                            continue
                        self.event_end_lines_.append(line)
                        line_str = line.replace("\n", "").split(" ")
                        event = int(line_str[2])
                        event_output.append([event, line_counter - 1])
                    elif "#" in line and " out " in line:
                        line_counter = 0
                    else:
                        continue

        self.num_output_per_event_ = np.array(
            event_output, dtype=np.int32, ndmin=2
        )
