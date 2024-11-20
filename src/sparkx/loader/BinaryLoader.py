# ===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from sparkx.loader.BinaryReader import BinaryReader

from sparkx.loader.BaseLoader import BaseLoader
from sparkx.Filter import *
import os
from typing import Dict, List, Tuple, Optional, Union, Any



class BinaryLoader(BaseLoader):
   
    PATH_BINARY_: str
    oscar_format_: Optional[str]
    optional_arguments_: Dict[str, Any]
    event_end_lines_: List[str]
    num_events_: int
    num_output_per_event_: List[int]

    def __init__(self, BINARY_FILE: str):
        if not ".bin" in BINARY_FILE:
            raise FileNotFoundError(
                "Input file is not in the BINARY format. Input "
                "file must have the ending .bin"
            )

        self.PATH_BINARY_ = BINARY_FILE
        self.oscar_format_ = None
    def __apply_kwargs_filters(
        self, event: List[List["Particle"]], filters_dict: Dict[str, Any]
    ) -> List[List["Particle"]]:
        """
        Applies the specified filters to the given event.

        This method applies a series of filters to the event data based on the
        keys in the filters_dict dictionary. The filters include:
        'charged_particles', 'uncharged_particles', 'strange_particles',
        'particle_species', 'remove_particle_species', 'participants',
        'spectators', 'lower_event_energy_cut', 'spacetime_cut', 'pT_cut',
        'rapidity_cut', 'pseudorapidity_cut', 'spacetime_rapidity_cut', and
        'multiplicity_cut'. If a key in the filters_dict dictionary does not
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
            elif i == "strange_particles":
                if filters_dict["strange_particles"]:
                    event = strange_particles(event)
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
            else:
                raise ValueError("The cut is unknown!")

        return event


    def oscar_format(self):

        return self.oscar_format_
    def event_end_lines(self):
        return self.event_end_lines_



    def read_binary(self,kwargs):
        reader = BinaryReader(self.PATH_BINARY_)
        if(reader.format_extended == 1):
            self.oscar_format_  = "Oscar2013Extended"
        else:
            self.oscar_format_  = "Oscar2013"
        data = []
        self.particle_list = []
        self.num_output_per_event_ = []
        end_line_template = '# event {} end 0 impact   {} scattering_projectile_target {}\n'
        
        for block in reader:
            if(block['type'] == 'p'):
                particles = block['part']
                for particle in particles:
                    data.append(Particle(self.oscar_format_,particle))
            if(block['type'] == 'f'):
                n_event = block["nevent"]
                b = block["b"]
                empty_event = block["empty_event"]
                
                if empty_event:
                    end_line = end_line_template.format(n_event,b,"yes")
                else:
                    end_line = end_line_template.format(n_event,b,"no")

                if "filters" in self.optional_arguments_.keys():
                    data = self.__apply_kwargs_filters(
                            [data], kwargs["filters"]
                        )[0]

                self.num_output_per_event_.append((
                    len(self.particle_list),
                    len(data))
                )

                self.particle_list.append(data)
                data = []
        return 
        
    def load(
        self, **kwargs: Dict[str, Any]
    ) -> Tuple[List[List["Particle"]], int, np.ndarray]:
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

        self.read_binary(kwargs)


        
        return (
            self.particle_list,
            self.getLastLineBinary()["nevent"] + 1,
            
            np.array(self.num_output_per_event_),
        )


    def getLastLineBinary(self):
        """Read the last 'f' block from the SMASH binary file."""
        encoding = 'utf-8'
        with open(self.PATH_BINARY_, "rb") as bfile:
            file_size = bfile.seek(0, 2)  # Get file size
            pos = file_size - 1  # Start from the end of the file
            while pos > 0:
                bfile.seek(pos)
                try:
                    block_type = bfile.read(1).decode(encoding)
                except:
                    block_type = "?"
                if block_type == 'f':
                    # Read 'f' block data
                    bfile.seek(pos + 1)  # Move past the block_type byte
                    n_event = np.fromfile(bfile, dtype='i4', count=1)[0]
                    impact_parameter = np.fromfile(bfile, dtype='d', count=1)[0]
                    empty_event = np.fromfile(bfile, dtype='B', count=1)[0]
                    return {'type': block_type,
                            'nevent': n_event,
                            'b': impact_parameter,
                            'empty_event': bool(empty_event)}
                
                pos -= 1  # Move to the previous byte

        return None  # Return None if no 'f' block is found




