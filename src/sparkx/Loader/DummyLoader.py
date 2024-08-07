#===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================

from sparkx.Loader.BaseLoader import BaseLoader
from sparkx.Filter import *

class DummyLoader(BaseLoader):
    def __init__(self, particle_list):

        """
        Initializes a new instance of the DummyLoader class.

        This method initializes a new instance of the DummyLoader class with the specified list of particles. It calls the superclass's constructor with the particle_list parameter and then sets the particle_list_ attribute to the particle_list parameter.

        Parameters
        ----------
        particle_list : list of lists of Particle objects
            The list of particles to load.

        Raises
        ------
        TypeError
            If the particle_list parameter is not a list of lists of Particle objects.

        Returns
        -------
        None
        """
        super().__init__(particle_list)
        self.particle_list_ = particle_list

    def load(self, **kwargs):
        """
        Loads the data from the dummy input based on the specified optional arguments.

        This method reads the dummy input and applies any filters specified in the 'filters' key of the kwargs dictionary. It also adjusts the number of events based on the 'events' key in the kwargs dictionary. If any other keys are specified in the kwargs dictionary, it raises a ValueError.

        Parameters
        ----------
        kwargs : dict
            A dictionary of optional arguments. The following keys are recognized:
            - 'events': Either a tuple of two integers specifying the range of events to load, or a single integer specifying a single event to load.
            - 'filters': A list of filters to apply to the data.

        Raises
        ------
        ValueError
            If an unknown keyword argument is used, if the first value of the 'events' tuple is larger than the second value, if an event number is negative.

        Returns
        -------
        tuple
            A tuple containing the list of Particle objects loaded from the dummy input, the number of events, and the number of output lines per event.
        """
        self.optional_arguments_ = kwargs
        self.num_events_ = len(self.particle_list_)

        for keys in self.optional_arguments_.keys():
            if keys not in ['events', 'filters']:
                raise ValueError('Unknown keyword argument used in constructor')

        if 'events' in self.optional_arguments_.keys() and isinstance(self.optional_arguments_['events'], tuple):
            self._check_that_tuple_contains_integers_only(self.optional_arguments_['events'])
            if self.optional_arguments_['events'][0] > self.optional_arguments_['events'][1]:
                raise ValueError('First value of event number tuple must be smaller than second value')
            elif self.optional_arguments_['events'][0] < 0 or self.optional_arguments_['events'][1] < 0:
                raise ValueError('Event numbers must be positive')
        elif 'events' in self.optional_arguments_.keys() and isinstance(self.optional_arguments_['events'], int):
            if self.optional_arguments_['events'] < 0:
                raise ValueError('Event number must be positive')

        self.set_num_output_per_event()
        return (self.set_particle_list(kwargs),  self.num_events_,self.num_output_per_event_)
    
    def set_num_output_per_event(self):
        """
        Set the number of output particles per event based on the filters applied.
        """
        self.num_output_per_event_ = []
        for i in range(0, self.num_events_):
            self.num_output_per_event_.append(len(self.particle_list_[i]))
        return self.num_output_per_event_
    
    def __apply_kwargs_filters(self, event, filters_dict):
        """
        Applies the specified filters to the event.

        This method applies a series of filters to the event based on the keys in the filters_dict dictionary. The filters include 'charged_particles', 'uncharged_particles', 'strange_particles', 'particle_species', 'remove_particle_species', 'participants', 'spectators', 'lower_event_energy_cut', 'spacetime_cut', 'pt_cut', 'rapidity_cut', 'pseudorapidity_cut', 'spatial_rapidity_cut', and 'multiplicity_cut'. If a key is not recognized, it raises a ValueError.

        Parameters
        ----------
        event : list
            The event to which the filters are applied.
        filters_dict : dict
            A dictionary of filters to apply to the event. The keys are the names of the filters and the values are the parameters for the filters.

        Raises
        ------
        ValueError
            If a key in the filters_dict dictionary is not recognized.

        Returns
        -------
        event : list
            The event after the filters have been applied.
        """
        if not isinstance(filters_dict, dict) or len(filters_dict.keys()) == 0:
            return event
        for i in filters_dict.keys():
            if i == 'charged_particles':
                if filters_dict['charged_particles']:
                    event = charged_particles(event)
            elif i == 'uncharged_particles':
                if filters_dict['uncharged_particles']:
                    event = uncharged_particles(event)
            elif i == 'strange_particles':
                if filters_dict['strange_particles']:
                    event = strange_particles(event)
            elif i == 'particle_species':
                event = particle_species(event, filters_dict['particle_species'])
            elif i == 'remove_particle_species':
                event = remove_particle_species (event, filters_dict['remove_particle_species'])
            elif i == 'participants':
                if filters_dict['participants']:
                    event = participants(event)
            elif i == 'spectators':
                if filters_dict['spectators']:
                    event = spectators(event)
            elif i == 'lower_event_energy_cut':
                event = lower_event_energy_cut(event, filters_dict['lower_event_energy_cut'])
            elif i == 'spacetime_cut':
                event = spacetime_cut(event, filters_dict['spacetime_cut'][0],filters_dict['spacetime_cut'][1])
            elif i == 'pt_cut':
                event = pt_cut(event, filters_dict['pt_cut'])
            elif i == 'rapidity_cut':
                event = rapidity_cut(event, filters_dict['rapidity_cut'])
            elif i == 'pseudorapidity_cut':
                event = pseudorapidity_cut(event, filters_dict['pseudorapidity_cut'])
            elif i == 'spatial_rapidity_cut':
                event = spatial_rapidity_cut(event, filters_dict['spatial_rapidity_cut'])
            elif i == 'multiplicity_cut':
                event = multiplicity_cut(event, filters_dict['multiplicity_cut'])
            else:
                raise ValueError('The cut is unknown!')

        return event
    
    # PUBLIC CLASS METHODS

    def set_particle_list(self, kwargs):
        """
        Set the particle list based on the filters applied.

        Parameters
        ----------
        kwargs : dict
            Dictionary containing the filters to be applied. The following keys are recognized:
            - 'events': Either a tuple of two integers specifying the range of events to load, or a single integer specifying a single event to load.
            - 'filters': A list of filters to apply to the data.

        Returns
        -------
        list
            List of particle objects.
        """
        if 'events' in kwargs.keys():
            if isinstance(kwargs['events'], int):
                self.particle_list_ = [self.particle_list_[kwargs['events']]]
            elif isinstance(kwargs['events'], tuple):
                event_start = kwargs['events'][0]
                event_end = kwargs['events'][1]
                self.particle_list_ = self.particle_list_[event_start : event_end + 1]

        if 'filters' in kwargs.keys():
            self.particle_list_ = [self.__apply_kwargs_filters([event], kwargs['filters'])[0] for event in self.particle_list_]

        return self.particle_list_
