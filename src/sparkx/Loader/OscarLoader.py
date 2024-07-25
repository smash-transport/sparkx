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
import os

class OscarLoader(BaseLoader):
        
    def __init__(self, OSCAR_FILE):
        
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
        super().__init__(OSCAR_FILE)
        if not '.oscar' in OSCAR_FILE:
            raise FileNotFoundError('Input file is not in the OSCAR format. Input '
                                    'file must have the ending .oscar')

        self.PATH_OSCAR_ = OSCAR_FILE
        self.oscar_format_ = None

    def load(self, **kwargs):
        self.optional_arguments_ = kwargs
        self.event_end_lines_ = []

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

        self.set_oscar_format()
        self.set_num_events()
        self.set_num_output_per_event_and_event_footers()
        return (self.set_particle_list(kwargs),  self.num_events_, self.num_output_per_event_)
    
    def _get_num_skip_lines(self):
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
            raise TypeError('Value given as flag "events" is not of type ' +
                            'int or a tuple of two int values')

        return skip_lines
    
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
            raise TypeError('Input file does not end with a comment line '+
                            'including the events. File might be incomplete '+
                            'or corrupted.')

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
        
    def oscar_format(self):
        return self.oscar_format_

    def event_end_lines(self):
        return self.event_end_lines_

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

    def __apply_kwargs_filters(self, event, filters_dict):
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
        particle_list = []
        data = []
        num_read_lines = self.__get_num_read_lines()
        with open(self.PATH_OSCAR_, 'r') as oscar_file:
            self._skip_lines(oscar_file)
            for i in range(0, num_read_lines):
                line = oscar_file.readline()
                if not line:
                    raise IndexError('Index out of range of OSCAR file. This most likely happened because ' +
                                     'the particle number specified by the comments in the OSCAR ' +
                                     'file differs from the actual number of particles in the event.')
                elif i == 0 and '#' not in line and 'out' not in line:
                    raise ValueError('First line of the event is not a comment ' +
                                     'line or does not contain "out"')
                elif 'event' in line and ('out' in line or 'in ' in line):
                    continue
                elif '#' in line and 'end' in line:
                    if 'filters' in self.optional_arguments_.keys():
                        data = self.__apply_kwargs_filters([data],kwargs['filters'])[0]
                        self.num_output_per_event_[len(particle_list)]=(len(particle_list),len(data))
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
                raise IndexError('Number of events in OSCAR file does not match the '+
                                 'number of events specified by the comments in the '+
                                    'OSCAR file!')
        elif isinstance(kwargs['events'], int):
            update = self.num_output_per_event_[kwargs['events']]
            self.num_output_per_event_ = [update]
            self.num_events_ = int(1)
        elif isinstance(kwargs['events'], tuple):
            event_start = kwargs['events'][0]
            event_end = kwargs['events'][1]
            update = self.num_output_per_event_[event_start : event_end+1]
            self.num_output_per_event_ = update
            self.num_events_ = int(event_end - event_start+1)

        return particle_list
            

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
            raise TypeError('Input file must follow the Oscar2013, '+
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