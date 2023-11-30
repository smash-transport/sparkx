from sparkx.Particle import Particle
import numpy as np
import csv
import warnings
import os

class Jetscape:
    """
    Defines a Jetscape object.

    The Jetscape class contains a single Jetscape hadron output file including
    all or only chosen events. It's methods allow to directly act on all
    contained events as applying acceptance filters (e.g. un/charged particles)
    to keep/romove particles by their PDG codes or to apply cuts
    (e.g. multiplicity, pseudo/rapidity, pT).
    Once these filters are applied, the new data set can be saved 1) as a nested
    list containing all quantities of the Jetscape format 2) as a list containing
    Particle objects from the ParticleClass or it can be printed to a file
    complying with the input format.

    Parameters
    ----------
    JETSCAPE_FILE : str
        Path to Jetscape file

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
              - From the input Jetscape file load only a single event by |br|
                specifying :code:`events=i` where i is event number i.
            * - :code:`events` (tuple)
              - From the input Jetscape file load only a range of events |br|
                given by the tuple :code:`(first_event, last_event)` |br|
                by specifying :code:`events=(first_event, last_event)` |br|
                where last_event is included.

        .. |br| raw:: html

           <br />

    Attributes
    ----------
    PATH_JETSCAPE_ : str
        Path to the Jetscape file
    num_output_per_event_ : numpy.array
        Array containing the event number and the number of particles in this
        event as num_output_per_event_[event i][num_output in event i] (updated
        when filters are applied)
    num_events_ : int
        Number of events contained in the Jetscape object (updated when filters
        are applied)

    Methods
    -------
    particle_list:
        Returns current Jetscape data as nested list
    particle_objects_list:
        Returns current Jetscape data as nested list of ParticleClass objects
    num_events:
        Get number of events
    num_output_per_event:
        Get number of particles in each event
    get_sigmaGen:
        Retrieves the sigmaGen values with error
    particle_species:
        Keep only particles with given PDG ids
    remove_particle_species:
        Remove particles with given PDG ids
    charged_particles:
        Keep charged particles only
    uncharged_particles:
        Keep uncharged particles only
    strange_particles:
        Keep strange particles only
    particle_status:
        Keep only particles with a given status flag
    pt_cut:
        Apply pT cut to all particles
    rapidity_cut:
        Apply rapidity cut to all particles
    pseudorapidity_cut:
        Apply pseudorapidity cut to all particles
    multiplicity_cut:
        Apply multiplicity cut to all particles
    lower_event_energy_cut:
        Filters out events with total energy lower than a threshold.
    print_particle_lists_to_file:
        Print current particle data to file with same format

    Examples
    --------

    **1. Initialization**

    To create a Jetscape object, the path to the Jetscape file has to be passed.
    By default the Jetscape object will contain all events of the input file. If
    the Jetscape object should only contain certain events, the keyword argument
    "events" must be used.

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Jetscape import Jetscape
        >>>
        >>> JETSCAPE_FILE_PATH = [Jetscape_directory]/particle_lists.dat
        >>>
        >>> # Jetscape object containing all events
        >>> jetscape1 = Jetscape(JETSCAPE_FILE_PATH)
        >>>
        >>> # Jetscape object containing only the first event
        >>> jetscape2 = Jetscape(JETSCAPE_FILE_PATH, events=0)
        >>>
        >>> # Jetscape object containing only events 2, 3, 4 and 5
        >>> jetscape3 = Jetscape(JETSCAPE_FILE_PATH, events=(2,5))

    **2. Method Usage**

    All methods that apply filters to the Jetscape data return :code:`self`. This
    means that methods can be concatenated. To access the Jetscape data as list to
    store it into a variable, the method :code:`particle_list()` or
    :code:`particle_objects_list` must be called in the end.
    Let's assume we only want to keep participant pions in events with a
    multiplicity > 500:

        >>> jetscape = Jetscape("path_to_file")
        >>>
        >>> pions = jetscape.multiplicity_cut(500).participants().particle_species((211, -211, 111))
        >>>
        >>> # save the pions of all events as nested list
        >>> pions_list = pions.particle_list()
        >>>
        >>> # save the pions as list of Particle objects
        >>> pions_particle_objects = pions.particle_objects_list()
        >>>
        >>> # print the pions to an Jetscape file
        >>> pions.print_particle_lists_to_file('./particle_lists.dat')

    """
    def __init__(self, JETSCAPE_FILE, **kwargs):
        if '.dat' in JETSCAPE_FILE:
            None
        else:
            raise TypeError('Input file is not in the JETSCAPE format. Input '
                            'file must have the ending .dat')

        self.PATH_JETSCAPE_ = JETSCAPE_FILE
        self.num_output_per_event_ = None
        self.num_events_ = None
        self.particle_list_ = None
        self.optional_arguments_ = kwargs

        self.set_num_output_per_event()
        self.set_particle_list(kwargs)
        
    # PRIVATE CLASS METHODS

    def __get_num_skip_lines(self):
        """
        Get number of initial lines in Jetscape file that are header or comment
        lines and need to be skipped in order to read the particle output.

        Returns
        -------
        skip_lines : int
            Number of initial lines before data.

        """
        if not self.optional_arguments_ or 'events' not in self.optional_arguments_.keys():
            skip_lines = 1
        elif isinstance(self.optional_arguments_['events'], int):
            if self.optional_arguments_['events'] == 0:
                skip_lines = 1
            else:
                cumulate_lines = 0
                for i in range(0, self.optional_arguments_['events']):
                    cumulate_lines += self.num_output_per_event_[i,1] + 1
                skip_lines = 1 + cumulate_lines
        elif isinstance(self.optional_arguments_['events'], tuple):
            line_start = self.optional_arguments_['events'][0]
            if line_start == 0:
                skip_lines = 1
            else:
                cumulate_lines = 0
                for i in range(0, line_start):
                    cumulate_lines += self.num_output_per_event_[i,1] + 1
                skip_lines = 1 + cumulate_lines
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
            Name of the variable for the file opend with the :code:`open()`
            command.

        """
        num_skip = self.__get_num_skip_lines()
        for i in range(0, num_skip):
            fname.readline()


    def __get_num_read_lines(self):
        if not self.optional_arguments_ or 'events' not in self.optional_arguments_.keys():
            cumulated_lines = np.sum(self.num_output_per_event_, axis=0)[1]
            # add number of comments
            cumulated_lines += int(len(self.num_output_per_event_))

        elif isinstance(self.optional_arguments_['events'], int):
            read_event = self.optional_arguments_['events']
            cumulated_lines = int(self.num_output_per_event_[read_event,1] + 1)

        elif isinstance(self.optional_arguments_['events'], tuple):
            cumulated_lines = 0
            event_start = self.optional_arguments_['events'][0]
            event_end = self.optional_arguments_['events'][1]
            for i in range(event_start, event_end+1):
                cumulated_lines += int(self.num_output_per_event_[i, 1] + 1)
        else:
            raise TypeError('Value given as flag events is not of type int or a tuple')

        # +1 for the end line in Jetscape format
        return cumulated_lines + 1


    def __particle_as_list(self, particle):
        particle_list = [0.0]*7
        particle_list[0] = int(particle.ID)
        particle_list[1]  = int(particle.pdg)
        particle_list[2]  = int(particle.status)
        particle_list[3]  = float(particle.E)
        particle_list[4]  = float(particle.px)
        particle_list[5]  = float(particle.py)
        particle_list[6]  = float(particle.pz)

        return particle_list

    # PUBLIC CLASS METHODS

    def set_particle_list(self, kwargs):
        particle_list = []
        data = []
        num_read_lines = self.__get_num_read_lines()
        fname = open(self.PATH_JETSCAPE_, 'r')
        self.__skip_lines(fname)

        for i in range(0, num_read_lines):
            line = fname.readline()
            if not line:
                raise IndexError('Index out of range of JETSCAPE file')
            elif '#' in line and 'sigmaGen' in line:
                particle_list.append(data)
            elif i == 0 and '#' not in line and 'weight' not in line:
                raise ValueError('First line of the event is not a comment ' +\
                                 'line or does not contain "weight"')
            elif 'Event' in line and 'weight' in line:
                data_line = line.replace('\n','').replace('\t',' ').split(' ')
                first_event_header = 1
                if 'events' in self.optional_arguments_.keys():
                    first_event_header += int(kwargs['events'][0])
                if int(data_line[2]) == first_event_header:
                    continue
                else:
                    particle_list.append(data)
                    data = []
            else:
                data_line = line.replace('\n','').replace('\t',' ').split(' ')
                particle = Particle("JETSCAPE", data_line)
                data.append(particle)
        fname.close()

        # Correct num_output_per_event and num_events
        if not kwargs or 'events' not in self.optional_arguments_.keys():
            None
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

    def set_num_output_per_event(self):
        file = open(self.PATH_JETSCAPE_ , 'r')
        event_output = []

        while True:
            line = file.readline()
            if not line:
                break
            elif '#' in line and 'N_hadrons' in line:
                line_str = line.replace('\n','').replace('\t',' ').split(' ')
                event = line_str[2]
                num_output = line_str[8]
                event_output.append([event, num_output])
            else:
                continue
        file.close()

        self.num_output_per_event_ = np.asarray(event_output, dtype=np.int32)
        self.num_events_ = len(event_output)

    def particle_list(self):
        """
        Returns a nested python list containing all quantities from the
        current Jetscape data as numerical values with the following shape:

            | Single Event:    [output_line][particle_quantity]
            | Multiple Events: [event][output_line][particle_quantity]

        Returns
        -------
        list
            Nested list containing the current Jetscape data

        """
        num_events = self.num_events_
        
        if num_events == 1:
            num_particles = self.num_output_per_event_[1]
        else:
            num_particles = self.num_output_per_event_[:,1]

        particle_array = []

        if num_events == 1:
            for i_part in range(0, num_particles):
                particle = self.particle_list_[i_part]
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
        Returns a nested python list containing all quantities from the
        current Jetscape data as numerical values with the following shape:

            | Single Event:    [output_line][particle_quantity]
            | Multiple Events: [event][output_line][particle_quantity]

        Returns
        -------
        list
            Nested list containing the current Oscar data
        """
        return self.particle_list_

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
        self : Jetscape object
            Containing charged particles in every event only
        """
        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                        if (elem.charge != 0 and elem.charge != None)]
            new_length = len(self.particle_list_[i])
            self.num_output_per_event_[i, 1] = new_length

        return self

    def uncharged_particles(self):
        """
        Keep only uncharged particles in particle_list

        Returns
        -------
        self : Jetscape object
            Containing uncharged particles in every event only
        """
        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                        if (elem.charge == 0 and elem.charge != None)]
            new_length = len(self.particle_list_[i])
            self.num_output_per_event_[i, 1] = new_length

        return self

    def strange_particles(self):
        """
        Keep only strange particles in particle_list

        Returns
        -------
        self : Jetscape object
            Containing strange particles in every event only
        """
        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                        if elem.is_strange()]
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
        self : Jetscape object
            Containing only particle species specified by pdg_list for every event

        """
        if not isinstance(pdg_list, (str, int, list, np.integer, np.ndarray, tuple)):
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')

        elif isinstance(pdg_list, (int, str, np.integer)):
            pdg_list = int(pdg_list)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (int(elem.pdg) == pdg_list and elem.pdg != None)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(pdg_list, (list, np.ndarray, tuple)):
            pdg_list = np.asarray(pdg_list, dtype=np.int64)


            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (int(elem.pdg) in pdg_list and elem.pdg != None)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        else:
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')
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
        self : Jetscape object
            Containing all but the specified particle species in every event

        """
        if not isinstance(pdg_list, (str, int, list, np.integer, np.ndarray, tuple)):
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')

        elif isinstance(pdg_list, (int, str, np.integer)):
            pdg_list = int(pdg_list)


            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (int(elem.pdg) != pdg_list and elem.pdg != None)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(pdg_list, (list, np.ndarray, tuple)):
            pdg_list = np.asarray(pdg_list, dtype=np.int64)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (not int(elem.pdg) in pdg_list and elem.pdg != None)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        else:
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')
        return self

    def particle_status(self, status_list):
        """
        Keep only particles with a given particle status

        Parameters
        ----------
        status_list : int
            To keep a particles with a single status only, pass a single status

        status_list : tuple/list/array
            To keep hadrons with different hadron status, pass a tuple or list
            or array

        Returns
        -------
        self : Jetscape object
            Containing only hadrons with status specified by status_list for
            every event

        """
        if not isinstance(status_list, (str, int, list, np.integer, np.ndarray, tuple)):
            raise TypeError('Input value for status codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')

        elif isinstance(status_list, (int, str, np.integer)):
            status_list = int(status_list)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (int(elem.status) == status_list and elem.status != None)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(status_list, (list, np.ndarray, tuple)):
            status_list = np.asarray(status_list, dtype=np.int64)

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i]
                                            if (int(elem.status) in status_list and elem.status != None)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        else:
            raise TypeError('Input value for status flag has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')
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
        self: Jetscape object
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
            raise ValueError('The lower event energ cut value should be positive')

        updated_particle_list = []
        for event_particles in self.particle_list_:
            total_energy = sum(particle.E for particle in event_particles if particle.E != None)
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
        self : Jetscape object
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

        for i in range(0, self.num_events_):
            self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                        (lower_cut <= elem.pt_abs() <= upper_cut and elem.pt_abs() != None)]
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
            If a single value is passed, the cut is applyed symmetrically
            around 0.
            For example, if cut_value = 1, only particles with rapidity in
            [-1.0, 1.0] are kept.

        cut_value : tuple
            To specify an asymmetric acceptance range for the rapidity
            of particles, pass a tuple (cut_min, cut_max)

        Returns
        -------
        self : Jetscape object
            Containing only particles complying with the rapidity cut
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
                                            (-limit<=elem.momentum_rapidity_Y()<=limit 
                                             and elem.momentum_rapidity_Y() != None)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(cut_value, tuple):
            lim_max = max(cut_value[0], cut_value[1])
            lim_min = min(cut_value[0], cut_value[1])

            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                            (lim_min<=elem.momentum_rapidity_Y()<=lim_max
                                             and elem.momentum_rapidity_Y() != None)]
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
        self : Jetscape object
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
                                            (-limit<=elem.pseudorapidity()<=limit
                                             and elem.pseudorapidity() != None)]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length

        elif isinstance(cut_value, tuple):
            lim_max = max(cut_value[0], cut_value[1])
            lim_min = min(cut_value[0], cut_value[1])

            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       (lim_min<=elem.pseudorapidity()<=lim_max
                                        and elem.pseudorapidity() != None)]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              (lim_min<=elem.pseudorapidity()<=lim_max
                                                and elem.pseudorapidity() != None)]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length

        else:
            raise TypeError('Input value must be a number or a tuple ' +\
                            'with the cut limits (cut_min, cut_max)')
        return self

    def multiplicity_cut(self,min_multiplicity):
        """
        Apply multiplicity cut. Remove all events with a multiplicity lower
        than min_multiplicity

        Parameters
        ----------
        min_multiplicity : float
            Lower bound for multiplicity. If the multiplicity of an event is
            lower than min_multiplicity, this event is discarded.

        Returns
        -------
        self : Jetscape object
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

    def __get_last_line(self,file_path):
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
        with open(file_path, 'rb') as file:
            file.seek(-2, 2)
            while file.read(1) != b'\n':
                file.seek(-2, 1)
            last_line = file.readline().decode().strip()
            return last_line

    def get_sigmaGen(self):
        """
        Retrieves the sigmaGen values with error from the last line of a file.

        Returns
        -------
        tuple
            A tuple containing the first and second sigmaGen values as floats.
        """
        last_line = self.__get_last_line(self.PATH_JETSCAPE_)
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

        return (numbers[0],numbers[1])

    def print_particle_lists_to_file(self, output_file):
        """
        Prints the current Jetscape data to an output file specified by :code:`output_file`
        with the same format as the input file

        Parameters
        ----------
        output_file : str
            Path to the output file like :code:`[output_directory]/particle_lists.dat`

        """
        line_in_initial_file = open(self.PATH_JETSCAPE_,'r')
        header_file = line_in_initial_file.readline()
        line_in_initial_file.close()

        # Open the output file with buffered writing
        with open(output_file, "w") as f_out:
            header_file_written = False
            data_to_write = []

            for i in range(self.num_events_):
                event = self.num_output_per_event_[i, 0]
                num_out = self.num_output_per_event_[i, 1]
                particle_output = np.asarray(self.particle_list()[i])

                # Write the header if not already written
                if not header_file_written:
                    header_file_written = True
                    data_to_write.append(header_file)

                # Header for each event
                header = f'#\tEvent\t{event}\tweight\t1\tEPangle\t0\tN_hadrons\t{num_out}\n'
                data_to_write.append(header)

                # Convert particle data to formatted strings
                particle_lines = [f"{int(row[0])} {int(row[1])} {int(row[2])} {row[3]:g} {row[4]:g} {row[5]:g} {row[6]:g}\n" for row in particle_output]

                # Append particle data to the list
                data_to_write.extend(particle_lines)

            # Write all accumulated data to the file
            f_out.writelines(data_to_write)

            # Write the last line
            last_line = self.__get_last_line(self.PATH_JETSCAPE_)
            f_out.write(last_line)
