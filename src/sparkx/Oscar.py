from sparkx.Particle import Particle
from sparkx.Filter import *
import numpy as np
import csv
import warnings
import os

class Oscar:
    """
    Defines an Oscar object.

    The Oscar class contains a single .oscar file including all or only chosen
    events in either the Oscar2013 or Oscar2013Extended format. It's methods
    allow to directly act on all contained events as applying acceptance filters
    (e.g. un/charged particles, spectators/participants) to keep/romove particles
    by their PDG codes or to apply cuts (e.g. multiplicity, pseudo/rapidity, pT).
    Once these filters are applied, the new data set can be saved as a

    1) nested list containing all quantities of the Oscar format
    2) list containing Particle objects from the ParticleClass

    or it may be printed to a file complying with the input format.

    Parameters
    ----------
    OSCAR_FILE : str
        Path to Oscar file

    Other Parameters
    ----------------
    **kwargs : properties, optional
        kwargs are used to specify optional properties like a chunk reading
        and must be used like :code:`'property'='value'` where the possible
        properties are specified below.

        .. list-table::
            :header-rows: 1
            :widths: 25 75

            * - Property
              - Description
            * - :code:`events` (int)
              - From the input Oscar file load only a single event by |br|
                specifying :code:`events=i` where i is event number i.
            * - :code:`events` (tuple)
              - From the input Oscar file load only a range of events |br|
                given by the tuple :code:`(first_event, last_event)` |br|
                by specifying :code:`events=(first_event, last_event)` |br|
                where last_event is included.

        .. |br| raw:: html

           <br />

    Attributes
    ----------
    PATH_OSCAR_ : str
        Path to the Oscar file
    oscar_format_ : str
        Input Oscar format "Oscar2013" or "Oscar2013Extended" (set automatically)
    num_output_per_event_ : numpy.array
        Array containing the event number and the number of particles in this
        event as num_output_per_event_[event i][num_output in event i] (updated
        when filters are applied)
    num_events_ : int
        Number of events contained in the Oscar object (updated when filters
        are applied)
    event_end_lines_ : list
        List containing all comment lines at the end of each event as str.
        Needed to print the Oscar object to a file.


    Methods
    -------
    particle_list:
        Returns current Oscar data as nested list
    particle_objects_list:
        Returns current Oscar data as nested list of ParticleClass objects
    num_events:
        Get number of events
    num_output_per_event:
        Get number of particles in each event
    oscar_format:
        Get Oscar format of the input file
    particle_species:
        Keep only particles with given PDG ids
    remove_particle_species:
        Remove particles with given PDG ids
    participants:
        Keep participants only
    spectators:
        Keep spectators only
    lower_event_energy_cut:
        Filters out events with total energy lower than a threshold.
    charged_particles:
        Keep charged particles only
    uncharged_particles:
        Keep uncharged particles only
    strange_particles:
        Keep strange particles only
    spacetime_cut:
        Apply spacetime cut to all particles
    pt_cut:
        Apply pT cut to all particles
    rapidity_cut:
        Apply rapidity cut to all particles
    pseudorapidity_cut:
        Apply pseudorapidity cut to all particles
    spatial_rapidity_cut:
        Apply spatial rapidity (space-time rapidity) cut to all particles
    multiplicity_cut:
        Apply multiplicity cut to all particles
    print_particle_lists_to_file:
        Print current particle data to file with same format

    Examples
    --------

    **1. Initialization**

    To create an Oscar object, the path to the Oscar file has to be passed.
    By default the Oscar object will contain all events of the input file. If
    the Oscar object should only contain certain events, the keyword argument
    "events" must be used.

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Oscar import Oscar
        >>>
        >>> OSCAR_FILE_PATH = [Oscar_directory]/particle_lists.oscar
        >>>
        >>> # Oscar object containing all events
        >>> oscar1 = Oscar(OSCAR_FILE_PATH)
        >>>
        >>> # Oscar object containing only the first event
        >>> oscar2 = Oscar(OSCAR_FILE_PATH, events=0)
        >>>
        >>> # Oscar object containing only events 2, 3, 4 and 5
        >>> oscar3 = Oscar(OSCAR_FILE_PATH, events=(2,5))

    **2. Method Usage**

    All methods that apply filters to the Oscar data return :code:`self`. This
    means that methods can be concatenated. To access the Oscar data as list to
    store it into a variable, the method :code:`particle_list()` or
    :code:`particle_objects_list` must be called in the end.
    Let's assume we only want to keep participant pions in events with a
    multiplicity > 500:

        >>> oscar = Oscar("path_to_file")
        >>>
        >>> pions = oscar.multiplicity_cut(500).participants().particle_species((211, -211, 111))
        >>>
        >>> # save the pions of all events as nested list
        >>> pions_list = pions.particle_list()
        >>>
        >>> # save the pions as list of Particle objects
        >>> pions_particle_objects = pions.particle_objects_list()
        >>>
        >>> # print the pions to an oscar file
        >>> pions.print_particle_lists_to_file('./particle_lists.oscar')

    """

    def __init__(self, OSCAR_FILE, **kwargs):
        """
        Parameters
        ----------
        OSCAR_FILE : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None
        """

        if not '.oscar' in OSCAR_FILE:
            raise TypeError('Input file is not in the OSCAR format. Input '
                            'file must have the ending .oscar')

        self.PATH_OSCAR_ = OSCAR_FILE
        self.oscar_format_ = None
        self.num_output_per_event_ = None
        self.num_events_ = None
        self.particle_list_ = None
        self.optional_arguments_ = kwargs
        self.event_end_lines_ = []

        if 'events' in self.optional_arguments_.keys() and isinstance(self.optional_arguments_['events'], tuple):
            if self.optional_arguments_['events'][0] > self.optional_arguments_['events'][1]:
                raise ValueError('First value of event number tuple must be smaller than second value')
            elif self.optional_arguments_['events'][0] <0 or self.optional_arguments_['events'][1] < 0:
                raise ValueError('Event numbers must be positive')
        elif 'events' in self.optional_arguments_.keys() and isinstance(self.optional_arguments_['events'], int):
            if self.optional_arguments_['events'] < 0:
                raise ValueError('Event number must be positive')

        self.set_oscar_format()
        self.set_num_events()
        self.set_num_output_per_event_and_event_footers()
        self.set_particle_list(kwargs)

    # PRIVATE CLASS METHODS

    def __get_num_skip_lines(self):
        """
        Get number of initial lines in Oscar file that are header or comment
        lines and need to be skipped in order to read the particle output.

        Returns
        -------
        skip_lines : int
            Number of initial lines before data.

        """
        if not self.optional_arguments_ or 'events' not in self.optional_arguments_.keys():
            skip_lines = 3
        elif isinstance(self.optional_arguments_['events'], int):
            if self.optional_arguments_['events'] == 0:
                skip_lines = 3
            else:
                cumulate_lines = 0
                for i in range(0, self.optional_arguments_['events']):
                    cumulate_lines += self.num_output_per_event_[i,1] + 2
                skip_lines = 3 + cumulate_lines
        elif isinstance(self.optional_arguments_['events'], tuple):
            line_start = self.optional_arguments_['events'][0]
            if line_start == 0:
                skip_lines = 3
            else:
                cumulate_lines = 0
                for i in range(0, line_start):
                    cumulate_lines += self.num_output_per_event_[i,1] + 2
                skip_lines = 3 + cumulate_lines
        else:
            raise TypeError('Value given as flag "events" is not of type ' +\
                            'int or a tuple of two int values')

        return skip_lines


    def __skip_lines(self, fname):
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
        num_skip = self.__get_num_skip_lines()
        for i in range(0, num_skip):
            fname.readline()


    def __get_num_read_lines(self):
        if not self.optional_arguments_ or 'events' not in self.optional_arguments_.keys():
            cumulated_lines = np.sum(self.num_output_per_event_, axis=0)[1]
            # add number of comments
            cumulated_lines += int(2 * len(self.num_output_per_event_))

        elif isinstance(self.optional_arguments_['events'], int):
            read_event = self.optional_arguments_['events']
            cumulated_lines = int(self.num_output_per_event_[read_event,1] + 2)

        elif isinstance(self.optional_arguments_['events'], tuple):
            cumulated_lines = 0
            event_start = self.optional_arguments_['events'][0]
            event_end = self.optional_arguments_['events'][1]
            for i in range(event_start, event_end+1):
                cumulated_lines += int(self.num_output_per_event_[i, 1] + 2)
        else:
            raise TypeError('Value given as flag events is not of type int or a tuple')

        return cumulated_lines


    def __particle_as_list(self, particle):
        particle_list = []
        particle_list.append(float(particle.t))
        particle_list.append(float(particle.x))
        particle_list.append(float(particle.y))
        particle_list.append(float(particle.z))
        particle_list.append(float(particle.mass))
        particle_list.append(float(particle.E))
        particle_list.append(float(particle.px))
        particle_list.append(float(particle.py))
        particle_list.append(float(particle.pz))
        particle_list.append(int(particle.pdg))
        particle_list.append(int(particle.ID))
        particle_list.append(int(particle.charge))

        if self.oscar_format_ == 'Oscar2013Extended'  or self.oscar_format_ == 'Oscar2013Extended_IC' or self.oscar_format_ == 'Oscar2013Extended_Photons':
            particle_list.append(int(particle.ncoll))
            particle_list.append(float(particle.form_time))
            particle_list.append(float(particle.xsecfac))
            particle_list.append(int(particle.proc_id_origin))
            particle_list.append(int(particle.proc_type_origin))
            particle_list.append(float(particle.t_last_coll))
            particle_list.append(int(particle.pdg_mother1))
            particle_list.append(int(particle.pdg_mother2))
            if self.oscar_format_ != 'Oscar2013Extended_Photons':
                if particle.baryon_number != np.nan:
                    particle_list.append(int(particle.baryon_number))
                if particle.strangeness != np.nan:
                    particle_list.append(int(particle.strangeness))
            else:
                if particle.weight != np.nan:
                    particle_list.append(int(particle.weight))

        elif self.oscar_format_ != 'Oscar2013' and self.oscar_format_ != 'Oscar2013Extended' and self.oscar_format_ != 'Oscar2013Extended_IC' and self.oscar_format_ != 'Oscar2013Extended_Photons':
            raise TypeError('Input file not in OSCAR2013, OSCAR2013Extended or Oscar2013Extended_IC format')

        return particle_list

    def __apply_kwargs_filters(self, event, filters_dict):
        for i in filters_dict.keys():
            if i == 'charged_particles':
                if filters_dict['charged_particles']:
                    event=charged_particles(event)
                    # do something
            #elif
              #other filters
                

            # Check if user given key is contained in allowed keys
            #'charged_particles, uncharged_particles, strange_particles, particle_species (int,tuple/list/array),'
            #'remove_particle_species (int,tuple/list/array), participants, spectators,'
            #'lower_event_energy_cut (int,float), spacetime_cut (tuple), pt_cut (tuple),'
            #'rapidity_cut (float,tuple), pseudorapidity_cut (float,tuple), spatial_rapidity_cut (float,tuple)'
            #'multiplicity_cut (int)'
        return event

    # PUBLIC CLASS METHODS


    def set_particle_list(self, kwargs):
        particle_list = []
        data = []
        num_read_lines = self.__get_num_read_lines()
        with open(self.PATH_OSCAR_, 'r') as oscar_file:
            self.__skip_lines(oscar_file)
            for i in range(0, num_read_lines):
                line = oscar_file.readline()
                if not line:
                    raise IndexError('Index out of range of OSCAR file. This most likely happened because ' +\
                                     'the particle number specified by the comments in the OSCAR ' +\
                                     'file differs from the actual number of particles in the event.')
                elif i == 0 and '#' not in line and 'out' not in line:
                    raise ValueError('First line of the event is not a comment ' +\
                                     'line or does not contain "out"')
                elif 'event' in line and ('out' in line or 'in ' in line):
                    continue
                elif '#' in line and 'end' in line:
                    if kwargs['filters']:
                        data[0] = self.__apply_kwargs_filters([data],kwargs['filters'])
                    particle_list.append(data)
                    data = []
                elif '#' in line:
                    raise ValueError('Comment line unexpectedly found: '+line)
                else:
                    line = line.replace('\n','').split(' ')
                    particle = Particle(self.oscar_format_, line)
                    data.append(particle)

        # Correct num_output_per_event and num_events
        if not kwargs or 'events' not in self.optional_arguments_.keys():
            if len(particle_list) != self.num_events_:
                raise IndexError('Number of events in OSCAR file does not match the '+\
                                 'number of events specified by the comments in the '+\
                                    'OSCAR file!')
        elif isinstance(kwargs['events'], int):
            update = self.num_output_per_event_[kwargs['events']]
            self.num_output_per_event_ = update
            self.num_events_ = int(1)
        elif isinstance(kwargs['events'], tuple):
            event_start = kwargs['events'][0]
            event_end = kwargs['events'][1]
            update = self.num_output_per_event_[event_start : event_end+1]
            self.num_output_per_event_ = update
            self.num_events_ = int(event_end - event_start+1)

        if not kwargs or 'events' not in self.optional_arguments_.keys():
            self.particle_list_ = particle_list
        elif isinstance(kwargs['events'], int):
            self.particle_list_ = particle_list[0]
        else:
            self.particle_list_ = particle_list
            


    def set_oscar_format(self):
        with open(self.PATH_OSCAR_, 'r') as file:
            first_line = file.readline()
            first_line = first_line.replace('\n', '').split(' ')

        if len(first_line) == 15 or first_line[0] == '#!OSCAR2013':
            self.oscar_format_ = 'Oscar2013'
        elif first_line[0] == '#!OSCAR2013Extended' and first_line[1]=='SMASH_IC':
            self.oscar_format_ = 'Oscar2013Extended_IC'
        elif first_line[0] == '#!OSCAR2013Extended' and first_line[1]=='Photons':
            self.oscar_format_ = 'Oscar2013Extended_Photons'
        elif len(first_line) == 23 or first_line[0] == '#!OSCAR2013Extended':
            self.oscar_format_ = 'Oscar2013Extended'
        else:
            raise TypeError('Input file must follow the Oscar2013, '+\
                            'Oscar2013Extended, Oscar2013Extended_IC or Oscar2013Extended_Photons format. ')


    def set_num_output_per_event_and_event_footers(self):
        with open(self.PATH_OSCAR_, 'r') as oscar_file:
            event_output = []
            if(self.oscar_format_ != 'Oscar2013Extended_IC' and self.oscar_format_ != 'Oscar2013Extended_Photons'):
                while True:
                    line = oscar_file.readline()
                    if not line:
                        break
                    elif '#' in line and 'end ' in line:
                        self.event_end_lines_.append(line)
                    elif '#' in line and 'out' in line:
                        line_str = line.replace('\n','').split(' ')
                        event = line_str[2]
                        num_output = line_str[4]
                        event_output.append([event, num_output])
                    else:
                        continue
            elif (self.oscar_format_ == 'Oscar2013Extended_IC'):
                line_counter=0
                event=0
                while True:
                    line_counter+=1
                    line = oscar_file.readline()
                    if not line:
                        break
                    elif '#' in line and 'end' in line:
                        self.event_end_lines_.append(line)
                        event_output.append([event, line_counter-2])
                    elif '#' in line and 'in' in line:
                        line_str = line.replace('\n','').split(' ')
                        event = line_str[2]
                        line_counter=0
                    else:
                        continue
    
            elif (self.oscar_format_ == 'Oscar2013Extended_Photons'):
                line_counter=0
                event=0
                line_memory=0
                while True:
                    line_counter+=1
                    line_memory+=1
                    line = oscar_file.readline()
                    if not line:
                        break
                    elif '#' in line and 'end' in line:
                        if(line_memory==1):
                            continue
                        self.event_end_lines_.append(line)
                        line_str = line.replace('\n','').split(' ')
                        event = line_str[2]
                        event_output.append([event, line_counter-1])
                    elif '#' in line and 'out' in line:
                        line_counter=0
                    else:
                        continue
        self.num_output_per_event_ = np.asarray(event_output, dtype=np.int32)


    def set_num_events(self):
        # Read the file in binary mode to search for last line. In this way one
        # does not need to loop through the whole file
        with open(self.PATH_OSCAR_, "rb") as file:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
            last_line = file.readline().decode().split(' ')
        if last_line[0] == '#' and 'event' in last_line:
            self.num_events_ = int(last_line[2]) + 1
        else:
            raise TypeError('Input file does not end with a comment line '+\
                            'including the events. File might be incomplete '+\
                            'or corrupted.')

    def particle_list(self):
        """
        Returns a nested python list containing all quantities from the
        current Oscar data as numerical values with the following shape:

            | Single Event:    [output_line][particle_quantity]
            | Multiple Events: [event][output_line][particle_quantity]

        Returns
        -------
        list
            Nested list containing the current Oscar data

        """
        num_events = self.num_events_
        
        if num_events == 1:
            num_particles = self.num_output_per_event_[0,1]
        else:
            num_particles = self.num_output_per_event_[:,1]

        particle_array = []

        if num_events == 1:
            for i_part in range(0, num_particles):
                particle = self.particle_list_[0][i_part]
                particle_array.append(self.__particle_as_list(particle))
        else:
            for i_ev in range(0, num_events):
                event = []
                for i_part in range(0, num_particles[i_ev]):
                    particle = self.particle_list_[i_ev][i_part]
                    event.append(self.__particle_as_list(particle))
                particle_array.append(event)

        return particle_array


    def particle_objects_list(self):
        """
        Returns a nested python list containing all particles from
        the Oscar2013/Oscar2013Extended output as particle objects
        from ParticleClass:

           | Single Event:    [particle_object]
           | Multiple Events: [event][particle_object]

        Returns
        -------
        particle_list_ : list
            List of particle objects from ParticleClass
        """
        return self.particle_list_


    def oscar_format(self):
        """
        Get the Oscar format of the input file.

        Returns
        -------
        oscar_format_ : str
            Oscar format of the input Oscar file as string ("Oscar2013" or
            "Oscar2013Extended")

        """
        return self.oscar_format_


    def num_output_per_event(self):
        """
        Returns a numpy array containing the event number (starting with 1)
        and the corresponding number of particles created in this event as

        num_output_per_event[event_n, numer_of_particles_in_event_n]

        num_output_per_event is updated with every manipulation e.g. after
        applying cuts.

        Returns
        -------
        num_output_per_event_ : numpy.ndarray
            Array containing the event number and the corresponding number of
            particles
        """
        return self.num_output_per_event_


    def num_events(self):
        """
        Returns the number of events in particle_list

        num_events is updated with every manipulation e.g. after
        applying cuts.

        Returns
        -------
        num_events_ : int
            Number of events in particle_list
        """
        return self.num_events_


    def charged_particles(self):
        """
        Keep only charged particles in particle_list

        Returns
        -------
        self : Oscar object
            Containing charged particles in every event only
        """

        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                        if (elem.charge != 0 and elem.charge != np.nan)]
            new_length = len(self.particle_list_[i])
            self.num_output_per_event_[i, 1] = new_length

        return self


    def uncharged_particles(self):
        """
        Keep only uncharged particles in particle_list

        Returns
        -------
        self : Oscar object
            Containing uncharged particles in every event only
        """

        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                        if (elem.charge == 0 and elem.charge != np.nan)]
            new_length = len(self.particle_list_[i])
            self.num_output_per_event_[i, 1] = new_length

        return self


    def strange_particles(self):
        """
        Keep only strange particles in particle_list

        Returns
        -------
        self : Oscar object
            Containing strange particles in every event only
        """

        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                        if (elem.strangeness()!=0 and elem.strangeness != np.nan)]
            new_length = len(self.particle_list_[i])
            self.num_output_per_event_[i, 1] = new_length

        return self


    def particle_species(self, pdg_list):
        """
        Keep only particle species given by their PDG ID in every event

        Parameters
        ----------
        pdg_list : int
            To keep a single particle species only, pass a single PDG ID

        pdg_list : tuple/list/array
            To keep multiple particle species, pass a tuple or list or array
            of PDG IDs

        Returns
        -------
        self : Oscar object
            Containing only particle species specified by pdg_list for every event

        """
        if not isinstance(pdg_list, (str, int, list, np.integer, np.ndarray, tuple, float)):
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, float, np.integer, list ' +\
                            'of str, list of int, list of int np.ndarray, tuple')

        elif isinstance(pdg_list, (int, float, str, np.integer)):
            pdg_list = int(pdg_list)
            
            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (int(elem.pdg) == pdg_list and elem.pdg != np.nan)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(pdg_list, (list, np.ndarray, tuple)):
            pdg_list = np.asarray(pdg_list, dtype=np.int64)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (int(elem.pdg) in pdg_list and elem.pdg != np.nan)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        else:
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, float, np.integer, list ' +\
                            'of str, list of int, list of float, np.ndarray, tuple')
        return self


    def remove_particle_species(self, pdg_list):
        """
        Remove particle species from particle_list by their PDG ID in every
        event

        Parameters
        ----------
        pdg_list : int
            To remove a single particle species only, pass a single PDG ID

        pdg_list : tuple/list/array
            To remove multiple particle species, pass a tuple or list or array
            of PDG IDs

        Returns
        -------
        self : Oscar object
            Containing all but the specified particle species in every event

        """
        if not isinstance(pdg_list, (str, int, float, list, np.integer, np.ndarray, tuple)):
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, float, np.integer, list ' +\
                            'of str, list of int, list of float, np.ndarray, tuple')

        elif isinstance(pdg_list, (int, str, np.integer)):
            pdg_list = int(pdg_list)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (int(elem.pdg) != pdg_list and elem.pdg != np.nan)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(pdg_list, (list, np.ndarray, tuple)):
            pdg_list = np.asarray(pdg_list, dtype=np.int64)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (not int(elem.pdg) in pdg_list and elem.pdg != np.nan)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        else:
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, float, np.integer, list ' +\
                            'of str, list of int, llst of float, np.ndarray, tuple')
        return self


    def participants(self):
        """
        Keep only participants in particle_list

        Returns
        -------
        self : Oscar oject
            Containing participants in every event only
        """

        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                      if (elem.ncoll != 0 and elem.ncoll != np.nan)]
            new_length = len(self.particle_list_[i])
            self.num_output_per_event_[i, 1] = new_length

        return self


    def spectators(self):
        """
        Keep only spectators in particle_list

        Returns
        -------
        self : Oscar object
            Containing spectators in every event only
        """


        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                        if (elem.ncoll == 0 and elem.ncoll != np.nan)]
            new_length = len(self.particle_list_[i])
            self.num_output_per_event_[i, 1] = new_length

        return self

    def lower_event_energy_cut(self,minimum_event_energy):
        """
        Filters out events with total energy lower than a threshold.

        Parameters
        ----------
        minimum_event_energy : int or float
            The minimum event energy threshold. Should be a positive integer or float.

        Returns
        -------
        self: Oscar object
            The updated instance of the class contains only events above the
            energy threshold.

        Raises
        ------
        TypeError
            If the minimum_event_energy parameter is not an integer or float.
        ValueError
            If the minimum_event_energy parameter is less than or equal to 0.
        """
        if not isinstance(minimum_event_energy, (int, float)):
            raise TypeError('Input value for lower event energy cut has not ' +\
                            'one of the following types: int, float')
        if minimum_event_energy <= 0.:
            raise ValueError('The lower event energy cut value should be positive')

        updated_particle_list = []
        for event_particles in self.particle_list_:
            total_energy = sum(particle.E for particle in event_particles if particle.E != np.nan)
            if total_energy >= minimum_event_energy:
                updated_particle_list.append(event_particles)
        self.particle_list_ = updated_particle_list
        self.num_output_per_event_ = np.array([[i+1, len(event_particles)] \
                    for i, event_particles in enumerate(updated_particle_list)],\
                    dtype=np.int32)
        self.num_events_ = len(updated_particle_list)

        if self.num_events_ == 0:
            warnings.warn('There are no events left after low energy cut')
            self.particle_list_ = [[]]
            self.num_output_per_event_ = np.asarray([[None, None]])

        return self

    def spacetime_cut(self, dim, cut_value_tuple):
        """
        Apply spacetime cut to all events by passing an acceptance range by
        ::code`cut_value_tuple`. All particles outside this range will
        be removed.

        Parameters
        ----------
        dim : string
            String naming the dimension on which to apply the cut.
            Options: 't','x','y','z'
        cut_value_tuple : tuple
            Tuple with the upper and lower limits of the coordinate space
            acceptance range :code:`(cut_min, cut_max)`. If one of the limits 
            is not required, set it to :code:`None`, i.e.
            :code:`(None, cut_max)` or :code:`(cut_min, None)`.

        Returns
        -------
        self : Oscar object
            Containing only particles complying with the spacetime cut for all events
        """

        if not isinstance(cut_value_tuple, tuple):
            raise TypeError('Input value must be a tuple')
        elif cut_value_tuple[0] is None and cut_value_tuple[1] is None:
            raise ValueError('At least one cut limit must be a number')
        elif dim == "t" and cut_value_tuple[0]<0:
            raise ValueError('Time boundary must be positive or zero.')
        if dim not in ("x","y","z","t"):
            raise ValueError('Only "t, x, y and z are possible dimensions.')

        if cut_value_tuple[0] is None:
            if(dim != "t"):
                lower_cut = float('-inf')
            else:
                lower_cut = 0.0
        else:
            lower_cut = cut_value_tuple[0]
        if cut_value_tuple[1] is None:
            upper_cut = float('inf')
        else:
            upper_cut = cut_value_tuple[1]

        if upper_cut < lower_cut:
            raise ValueError('The upper cut is smaller than the lower cut!')

        for i in range(0, self.num_events_):
            if (dim == "t"):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                        (lower_cut <= elem.t <= upper_cut and elem.t != np.nan)]
            elif (dim == "x"):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                        (lower_cut <= elem.x <= upper_cut and elem.x != np.nan)]
            elif (dim == "y"):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                        (lower_cut <= elem.y <= upper_cut and elem.y != np.nan)]
            else:
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                        (lower_cut <= elem.z <= upper_cut and elem.z != np.nan)]
            new_length = len(self.particle_list_[i])
            self.num_output_per_event_[i, 1] = new_length

        return self

    def pt_cut(self, cut_value_tuple):
        """
        Apply p_t cut to all events by passing an acceptance range by
        ::code`cut_value_tuple`. All particles outside this range will
        be removed.

        Parameters
        ----------
        cut_value_tuple : tuple
            Tuple with the upper and lower limits of the pT acceptance
            range :code:`(cut_min, cut_max)`. If one of the limits is not
            required, set it to :code:`None`, i.e. :code:`(None, cut_max)`
            or :code:`(cut_min, None)`.

        Returns
        -------
        self : Oscar object
            Containing only particles complying with the p_t cut for all events
        """

        if not isinstance(cut_value_tuple, tuple):
            raise TypeError('Input value must be a tuple containing either '+\
                            'positive numbers or None')
        elif (cut_value_tuple[0] is not None and cut_value_tuple[0]<0) or \
             (cut_value_tuple[1] is not None and cut_value_tuple[1]<0):
                 raise ValueError('The cut limits must be positiv or None')
        elif cut_value_tuple[0] is None and cut_value_tuple[1] is None:
            raise ValueError('At least one cut limit must be a number')

        if cut_value_tuple[0] is None:
            lower_cut = 0.0
        else:
            lower_cut = cut_value_tuple[0]
        if cut_value_tuple[1] is None:
            upper_cut = float('inf')
        else:
            upper_cut = cut_value_tuple[1]

        if upper_cut < lower_cut:
            raise ValueError('The upper cut is smaller than the lower cut!')

        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                        (lower_cut <= elem.pt_abs() <= upper_cut and elem.pt_abs() != np.nan)]
            new_length = len(self.particle_list_[i])
            self.num_output_per_event_[i, 1] = new_length

        return self


    def rapidity_cut(self, cut_value):
        """
        Apply rapidity cut to all events and remove all particles with rapidity
        not complying with cut_value

        Parameters
        ----------
        cut_value : float
            If a single value is passed, the cut is applied symmetrically
            around 0.
            For example, if cut_value = 1, only particles with rapidity in
            [-1.0, 1.0] are kept.

        cut_value : tuple
            To specify an asymmetric acceptance range for the rapidity
            of particles, pass a tuple (cut_min, cut_max)

        Returns
        -------
        self : Oscar object
            Containing only particles complying with the rapidity cut
            for all events
        """
        if isinstance(cut_value, tuple) and cut_value[0] > cut_value[1]:
            warn_msg = warn_msg = 'Lower limit {} is greater that upper limit {}. Switched order is assumed in the following.'.format(cut_value[0], cut_value[1])
            warnings.warn(warn_msg)

        if not isinstance(cut_value, (int, float, tuple)):
            raise TypeError('Input value must be a number or a tuple ' +\
                            'with the cut limits (cut_min, cut_max)')

        elif isinstance(cut_value, tuple) and len(cut_value) != 2:
            raise TypeError('The tuple of cut limits must contain 2 values')

        elif isinstance(cut_value, (int, float)):
            # cut symmetrically around 0
            limit = np.abs(cut_value)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                            (-limit<=elem.momentum_rapidity_Y()<=limit 
                                             and elem.momentum_rapidity_Y() != np.nan)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(cut_value, tuple):
            lim_max = max(cut_value[0], cut_value[1])
            lim_min = min(cut_value[0], cut_value[1])

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                            (-lim_min<=elem.momentum_rapidity_Y()<=lim_max 
                                             and elem.momentum_rapidity_Y() != np.nan)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        else:
            raise TypeError('Input value must be a number or a tuple ' +\
                            'with the cut limits (cut_min, cut_max)')
        return self


    def pseudorapidity_cut(self, cut_value):
        """
        Apply pseudo-rapidity cut to all events and remove all particles with
        pseudo-rapidity not complying with cut_value

        Parameters
        ----------
        cut_value : float
            If a single value is passed, the cut is applyed symmetrically
            around 0.
            For example, if cut_value = 1, only particles with pseudo-rapidity
            in [-1.0, 1.0] are kept.

        cut_value : tuple
            To specify an asymmetric acceptance range for the pseudo-rapidity
            of particles, pass a tuple (cut_min, cut_max)

        Returns
        -------
        self : Oscar object
            Containing only particles complying with the pseudo-rapidity cut
            for all events
        """
        if isinstance(cut_value, tuple) and cut_value[0] > cut_value[1]:
            warn_msg = 'Cut limits in wrong order: '+str(cut_value[0])+' > '+\
                        str(cut_value[1])+'. Switched order is assumed in ' +\
                       'the following.'
            warnings.warn(warn_msg)

        if not isinstance(cut_value, (int, float, tuple)):
            raise TypeError('Input value must be a number or a tuple ' +\
                            'with the cut limits (cut_min, cut_max)')

        elif isinstance(cut_value, tuple) and len(cut_value) != 2:
            raise TypeError('The tuple of cut limits must contain 2 values')

        elif isinstance(cut_value, (int, float)):
            # cut symmetrically around 0
            limit = np.abs(cut_value)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                            -limit<=elem.pseudorapidity()<=limit]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(cut_value, tuple):
            lim_max = max(cut_value[0], cut_value[1])
            lim_min = min(cut_value[0], cut_value[1])

            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       (lim_min<=elem.pseudorapidity()<=lim_max
                                        and elem.pseudorapidity() != np.nan)]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              (lim_min<=elem.pseudorapidity()<=lim_max
                                                and elem.pseudorapidity() != np.nan)]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length

        else:
            raise TypeError('Input value must be a number or a tuple ' +\
                            'with the cut limits (cut_min, cut_max)')
        return self


    def spatial_rapidity_cut(self, cut_value):
        """
        Apply spatial rapidity (space-time rapidity) cut to all events and
        remove all particles with spatial rapidity not complying with cut_value

        Parameters
        ----------
        cut_value : float
            If a single value is passed, the cut is applied symmetrically
            around 0.
            For example, if cut_value = 1, only particles with spatial rapidity
            in [-1.0, 1.0] are kept.

        cut_value : tuple
            To specify an asymmetric acceptance range for the spatial rapidity
            of particles, pass a tuple (cut_min, cut_max)

        Returns
        -------
        self : Oscar object
            Containing only particles complying with the spatial rapidity cut
            for all events
        """
        if isinstance(cut_value, tuple) and cut_value[0] > cut_value[1]:
            warn_msg = 'Cut limits in wrong order: '+str(cut_value[0])+' > '+\
                        str(cut_value[1])+'. Switched order is assumed in ' +\
                       'the following.'
            warnings.warn(warn_msg)

        if not isinstance(cut_value, (int, float, tuple)):
            raise TypeError('Input value must be a number or a tuple ' +\
                            'with the cut limits (cut_min, cut_max)')

        elif isinstance(cut_value, tuple) and len(cut_value) != 2:
            raise TypeError('The tuple of cut limits must contain 2 values')

        elif isinstance(cut_value, (int, float)):
            # cut symmetrically around 0
            limit = np.abs(cut_value)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                            (-limit<=elem.spatial_rapidity()<=limit
                                             and elem.spatial_rapidity() != np.nan)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(cut_value, tuple):
            lim_max = max(cut_value[0], cut_value[1])
            lim_min = min(cut_value[0], cut_value[1])

            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       (lim_min<=elem.spatial_rapidity()<=lim_max
                                        and elem.spatial_rapidity() != np.nan)]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              (lim_min<=elem.spatial_rapidity()<=lim_max
                                                and elem.spatial_rapidity() != np.nan)]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length

        else:
            raise TypeError('Input value must be a number or a tuple ' +\
                            'with the cut limits (cut_min, cut_max)')
        return self

    def multiplicity_cut(self, min_multiplicity):
        """
        Apply multiplicity cut. Remove all events with a multiplicity lower
        than min_multiplicity

        Parameters
        ----------
        min_multiplicity : int
            Lower bound for multiplicity. If the multiplicity of an event is
            lower than min_multiplicity, this event is discarded.

        Returns
        -------
        self : Oscar object
            Containing only events with a multiplicity >= min_multiplicity
        """
        if not isinstance(min_multiplicity, int):
            raise TypeError('Input value for multiplicity cut must be an int')
        if min_multiplicity < 0:
            raise ValueError('Minimum multiplicity must >= 0')

        idx_keep_event = []
        for idx, multiplicity in enumerate(self.num_output_per_event_[:, 1]):
            if multiplicity >= min_multiplicity:
                idx_keep_event.append(idx)

        self.particle_list_ = [self.particle_list_[idx] for idx in idx_keep_event]
        self.num_output_per_event_ = np.asarray([self.num_output_per_event_[idx] for idx in idx_keep_event])
        number_deleted_events = self.num_events_- len(idx_keep_event)
        self.num_events_ -= number_deleted_events

        return self

    def print_particle_lists_to_file(self, output_file):
        """
        Prints the current Oscar data to an output file specified by :code:`output_file`
        with the same format as the input file

        Parameters
        ----------
        output_file : str
            Path to the output file like :code:`[output_directory]/particle_lists.oscar`

        """
        header = []
        event_footer = ''
        format_oscar2013 = '%g %g %g %g %g %.9g %.9g %.9g %.9g %d %d %d'
        format_oscar2013_extended = '%g %g %g %g %g %.9g %.9g %.9g %.9g %d %d %d %d %g %g %d %d %g %d %d %d'

        with open(self.PATH_OSCAR_,'r') as oscar_file:
            counter_line = 0
            while True:
                line = oscar_file.readline()
                line_splitted = line.replace('\n','').split(' ')
    
                if counter_line < 3:
                    header.append(line)
                elif line_splitted[0] == '#' and line_splitted[3] == 'end':
                    event_footer = line
                    break
                elif counter_line > 1000000:
                    err_msg = 'Unable to find the end of an event in the original' +\
                              'Oscar file within the first 1000000 lines'
                    raise RuntimeError(err_msg)
                counter_line += 1

        event_footer = event_footer.replace('\n','').split(' ')
        with open(output_file, "w") as f_out:
            for i in range(3):
                f_out.write(header[i])

        with open(output_file, "a") as f_out:
            if(self.num_events_>1):
                for i in range(self.num_events_):
                    event = self.num_output_per_event_[i,0]
                    num_out = self.num_output_per_event_[i,1]
                    particle_output = np.asarray(self.particle_list()[i])

                    f_out.write('# event '+ str(event)+' out '+ str(num_out)+'\n')
                    if self.oscar_format_ == 'Oscar2013':
                        np.savetxt(f_out, particle_output, delimiter=' ', newline='\n', fmt=format_oscar2013)
                    elif self.oscar_format_ == 'Oscar2013Extended'  or self.oscar_format_ == 'Oscar2013Extended_IC' or self.oscar_format_ == 'Oscar2013Extended_Photons':
                        np.savetxt(f_out, particle_output, delimiter=' ', newline='\n', fmt=format_oscar2013_extended)
                    f_out.write(self.event_end_lines_[event])
            else:
                event = 0
                num_out = self.num_output_per_event_
                particle_output = np.asarray(self.particle_list())
                f_out.write('# event '+ str(event)+' out '+ str(num_out)+'\n')
                if self.oscar_format_ == 'Oscar2013':
                    np.savetxt(f_out, particle_output, delimiter=' ', newline='\n', fmt=format_oscar2013)
                elif self.oscar_format_ == 'Oscar2013Extended'  or self.oscar_format_ == 'Oscar2013Extended_IC' or self.oscar_format_ == 'Oscar2013Extended_Photons':
                    np.savetxt(f_out, particle_output, delimiter=' ', newline='\n', fmt=format_oscar2013_extended)
                f_out.write(self.event_end_lines_[event])
        f_out.close()