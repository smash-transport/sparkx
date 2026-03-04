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
    set_oscar_format()
        Determines and sets the format of the OSCAR data file.
    oscar_format()
        Returns the OSCAR format of the data file.
    impact_parameter()
        Returns the impact parameter of the events in the OSCAR data file.
    event_end_lines()
        Returns the event end lines in the OSCAR data file.
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
        elif "events" in self.optional_arguments_.keys():
            raise TypeError(
                'Value given as flag "events" is not of type '
                + "int or a tuple of two int values"
            )

        self.set_oscar_format()
        return self._single_pass_load(kwargs)

    def _single_pass_load(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[List[List[Particle]], int, np.ndarray, List[str]]:
        """
        Loads all data from the OSCAR file in a single pass.

        This method reads the file once, simultaneously collecting event
        metadata (event counts, particles-per-event, end lines, impact
        parameters) and building Particle objects. This replaces the previous
        This method reads the file once, simultaneously collecting event
        metadata (event headers, particle counts, event end lines) and
        building Particle objects.

        When the ``events`` keyword is specified, events outside the
        requested range are skipped without creating Particle objects.
        Validation of event ranges is done on the fly.

        Parameters
        ----------
        kwargs : dict
            A dictionary of optional arguments.

        Returns
        -------
        tuple
            A tuple of (particle_list, num_events, num_output_per_event,
            custom_attr_list).
        """
        # Determine which events to keep
        event_start: int = 0
        event_end: Union[int, None] = None
        if "events" in self.optional_arguments_:
            ev_arg = self.optional_arguments_["events"]
            if isinstance(ev_arg, int):
                event_start = ev_arg
                event_end = ev_arg
            elif isinstance(ev_arg, tuple):
                event_start = ev_arg[0]
                event_end = ev_arg[1]

        has_filters = "filters" in self.optional_arguments_
        is_ic = self.oscar_format_ == "Oscar2013Extended_IC"
        is_photons = self.oscar_format_ == "Oscar2013Extended_Photons"

        particle_list: List[List[Particle]] = []
        event_output: List[List[int]] = []
        all_event_end_lines: List[str] = []
        surviving_event_end_lines: List[str] = []
        data: List[Particle] = []
        cut_events: int = 0
        current_event_idx: int = -1  # 0-based index of events in file
        in_requested_range: bool = False
        total_events_in_file: int = 0
        # For Photons format: track whether we just saw an end line
        # (to skip duplicate end lines)
        last_was_end: bool = False

        with open(self.PATH_OSCAR_, "r") as f:
            # Skip header lines (format line + units line + version line)
            for _ in range(3):
                f.readline()

            for line in f:
                # Event start line: "# event N out M" or "# event N in M"
                # or "# event N start"
                if (
                    "#" in line
                    and "event" in line
                    and (" out " in line or " in " in line or " start" in line)
                ):
                    current_event_idx += 1

                    # Determine if we're in the requested range
                    if current_event_idx < event_start:
                        in_requested_range = False
                        if is_ic or is_photons:
                            last_was_end = False
                        continue
                    if event_end is not None and current_event_idx > event_end:
                        in_requested_range = False
                        if not is_ic and not is_photons:
                            pass
                        continue

                    in_requested_range = True

                    # Parse the "out" line for standard formats
                    if not is_ic and not is_photons:
                        line_str = line.replace("\n", "").split(" ")
                        event_num = BaseLoader._extract_integer_after_keyword(
                            line_str, "event"
                        )
                        num_output = BaseLoader._extract_integer_after_keyword(
                            line_str, "out"
                        )
                        event_output.append([event_num, num_output])
                    elif is_ic:
                        line_str = line.replace("\n", "").split(" ")
                        event_num = BaseLoader._extract_integer_after_keyword(
                            line_str, "event"
                        )
                        event_output.append([event_num, 0])
                    elif is_photons:
                        event_output.append([0, 0])
                        last_was_end = False
                    continue

                # Event end line: "# event N end ..."
                if "#" in line and " end" in line:
                    # For Photons: skip duplicate end lines
                    if is_photons and last_was_end:
                        continue
                    last_was_end = True

                    total_events_in_file += 1

                    if is_photons and in_requested_range:
                        line_str = line.replace("\n", "").split(" ")
                        event_num = BaseLoader._extract_integer_after_keyword(
                            line_str, "event"
                        )
                        event_output[-1][0] = event_num
                        event_output[-1][1] = len(data)

                    if is_ic and in_requested_range:
                        event_output[-1][1] = len(data)

                    # Always collect end lines for impact parameter lookup
                    all_event_end_lines.append(line)

                    if in_requested_range:
                        old_data_len = len(data)
                        if has_filters:
                            data = self.__apply_kwargs_filters(
                                [data], kwargs["filters"]
                            )[0]

                        if len(data) != 0 or old_data_len == 0:
                            event_output[-1][1] = len(data)
                            particle_list.append(data)
                            surviving_event_end_lines.append(line)
                        else:
                            event_output.pop()
                            cut_events += 1
                        data = []
                    continue

                # Comment line in unexpected position
                if "#" in line:
                    continue

                # Data line (particle)
                if in_requested_range:
                    last_was_end = False
                    line_list = np.asarray(line.replace("\n", "").split(" "))
                    if self.oscar_format_ == "ASCII":
                        particle = Particle(
                            self.oscar_format_, line_list, self.custom_attr_list
                        )
                    else:
                        particle = Particle(self.oscar_format_, line_list)
                    data.append(particle)
                else:
                    last_was_end = False

        # Validate that the file ended properly
        if total_events_in_file == 0 and current_event_idx < 0:
            raise TypeError(
                "Input file does not contain any events. "
                "File might be incomplete or corrupted."
            )

        # Validate requested event range
        if event_end is not None and current_event_idx < event_end:
            raise IndexError(
                f"Requested events up to {event_end} but the file only "
                f"contains {current_event_idx + 1} events."
            )

        # When filters cut some events, renumber survivors sequentially
        # starting from 0 (Oscar convention) so that post-constructor
        # filters in BaseStorer continue to work correctly.
        if cut_events > 0 and len(event_output) > 0:
            for i in range(len(event_output)):
                event_output[i][0] = i

        # Build final arrays
        if len(event_output) > 0:
            self.num_output_per_event_ = np.array(
                event_output, dtype=np.int32, ndmin=2
            )
        else:
            self.num_output_per_event_ = np.array([])
        self.num_events_ = len(event_output)
        self.event_end_lines_ = all_event_end_lines
        # Track end lines for only the surviving events (after filters)
        # so impact_parameter() can return correct values directly.
        self.surviving_event_end_lines_ = surviving_event_end_lines

        if particle_list == []:
            particle_list = [[]]

        return (
            particle_list,
            self.num_events_,
            self.num_output_per_event_,
            self.custom_attr_list,
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
        elif first_line_list[0] == "#!ASCII":
            self.oscar_format_ = "ASCII"
            value_line = first_line_list[2:]
            self.custom_attr_list = self._set_custom_attr_list(value_line)
        else:
            raise TypeError(
                "Input file must follow the Oscar2013, "
                + "Oscar2013Extended, Oscar2013Extended_IC, Oscar2013Extended_Photons or ASCII format. "
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
        end line of each surviving event in the OSCAR data file.

        Returns
        -------
        List[float]
            The impact parameter of the collisions.
        """
        if self.num_output_per_event_.shape[0] == 0:
            return []

        impact_parameters: List[float] = []
        for line in self.surviving_event_end_lines_:
            line_split = line.split(" ")
            line_split = list(filter(None, line_split))
            impact_parameter = float(line_split[-3])
            impact_parameters.append(impact_parameter)

        return impact_parameters

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
