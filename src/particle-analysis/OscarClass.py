from ParticleClass import Particle
import particle.data
import numpy as np
import csv
import warnings
import os

class Oscar:
    
    def __init__(self, OSCAR_FILE, **kwargs):
        #kwargs: 
        #   events:
        #        - int: load single event
        #        - tuple: (event_start, event_end) load all events in given range 
        
        if '.oscar' in OSCAR_FILE:
            None
        else:
            raise TypeError('Input file is not in the OSCAR format. Input '
                            'file must have the ending .oscar')
            
        self.PATH_OSCAR_ = OSCAR_FILE
        self.oscar_type_ = None
        self.num_output_per_event_ = None
        self.num_events_ = None
        self.particle_list_ = None
        self.list_of_all_valid_pdg_ids_ = None
        self.optional_arguments_ = kwargs
        self.event_end_lines_ = []
    
        
        self.set_oscar_type()
        self.set_num_events()
        self.set_num_output_per_event_and_event_footers()  
        self.set_list_of_all_valid_pdg_ids()
        self.set_particle_list(kwargs)
    
    # PRIVATE CLASS METHODS
    
    def __get_num_skip_lines(self):
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
        num_skip = self.__get_num_skip_lines()
        for i in range(0, num_skip):
            fname.readline()
            
            
    def __get_num_read_lines(self):
        if not self.optional_arguments_ or 'events' not in self.optional_arguments_.keys():
            cumulated_lines = np.sum(self.num_output_per_event_, axis=0)[1]
            # add number of comments
            cumulated_lines += int(2 * len(self.num_output_per_event_))
            if self.oscar_type_=="Oscar2013Extended IC":
                cumulated_lines-=0
            
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
        
        if self.oscar_type_ == 'Oscar2013Extended'  or self.oscar_type_ == 'Oscar2013Extended IC':
            particle_list.append(int(particle.ncoll))
            particle_list.append(float(particle.form_time))
            particle_list.append(int(particle.xsecfac))
            particle_list.append(int(particle.proc_id_origin))
            particle_list.append(int(particle.proc_type_origin))
            particle_list.append(float(particle.t_last_coll))
            particle_list.append(int(particle.pdg_mother1))
            particle_list.append(int(particle.pdg_mother2))
                                 
            if particle.baryon_number() != None:                         
                particle_list.append(int(particle.baryon_number))
            
            
        elif self.oscar_type_ != 'Oscar2013' and self.oscar_type_ != 'Oscar2013Extended' and self.oscar_type_ != 'Oscar2013Extended IC':
            raise TypeError('Input file not in OSCAR2013, OSCAR2013Extended or OSCAR2013Extended IC format')
            
        return particle_list
    
    
    def __check_if_pdg_is_valid(self, pdg_list):
        if isinstance(pdg_list, int):
            if not pdg_list in self.list_of_all_valid_pdg_ids_:
                raise ValueError('Invalid PDG ID given according to the following ' +\
                                 'data base: ' + self.list_of_all_valid_pdg_ids_[0] +\
                                 '\n Enter a valid PDG ID or update database.')
                    
        elif isinstance(pdg_list, np.ndarray):
            if not all(pdg in self.list_of_all_valid_pdg_ids_ for pdg in pdg_list):
                non_valid_elements = np.setdiff1d(pdg_list, self.list_of_all_valid_pdg_ids_)
                raise ValueError('One or more invalid PDG IDs given. The IDs ' +\
                                 str(non_valid_elements) +' are not contained in ' +\
                                 'the data base: ' + self.list_of_all_valid_pdg_ids_[0] +\
                                 '\n Enter valid PDG IDs or update database.')
        return True
    
        
    # PUBLIC CLASS METHODS
        
    
    def set_particle_list(self, kwargs):
        particle_list = []
        data = []
        num_read_lines = self.__get_num_read_lines()
        fname = open(self.PATH_OSCAR_, 'r')
        self.__skip_lines(fname)
        for i in range(0, num_read_lines):
            line = fname.readline()
            if not line:
                raise IndexError('Index out of range of OSCAR file')
            elif i == 0 and '#' not in line and 'out' not in line:
                raise ValueError('First line of the event is not a comment ' +\
                                 'line or does not contain "out"')
            elif 'event' in line and ('out' in line or 'in ' in line):
                continue
            elif '#' in line and 'end' in line:
                particle_list.append(data)
                data = []
            else:
                data_line = line.replace('\n','').split(' ')
                particle = Particle()
                if self.oscar_type_ == 'Oscar2013':
                    particle.set_quantities_OSCAR2013(data_line)
                elif self.oscar_type_ == 'Oscar2013Extended' or self.oscar_type_ == 'Oscar2013Extended IC' :
                    particle.set_quantities_OSCAR2013Extended(data_line)
                
                # Check for filters by method with a dictionary
                # and do not append if empty (Method: WantToKeep(particle, filter) -> True/False)
                
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
        
    
    def set_oscar_type(self):
        first_line = open(self.PATH_OSCAR_,'r')
        first_line = first_line.readline()
        first_line = first_line.replace('\n','').split(' ')
        
        if len(first_line) == 15 or first_line[0] == '#!OSCAR2013':
            self.oscar_type_ = 'Oscar2013'
        elif first_line[0] == '#!OSCAR2013Extended' and first_line[1]=='SMASH_IC':
            self.oscar_type_ = 'Oscar2013Extended IC'
        elif len(first_line) == 23 or first_line[0] == '#!OSCAR2013Extended':
            self.oscar_type_ = 'Oscar2013Extended'
        else:
            raise TypeError('Input file must follow the Oscar2013, '+\
                            'Oscar2013Extended or Oscar2013Extended IC format ')
                
                
    def set_num_output_per_event_and_event_footers(self):
        file = open(self.PATH_OSCAR_ , 'r')
        event_output = []
        if(self.oscar_type_ != 'Oscar2013Extended IC'):
            while True:
                line = file.readline()
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
        else:
            line_counter=0
            event=0
            while True:
                line_counter+=1
                line = file.readline()
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
        file.close()
        
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
                
                
    def set_list_of_all_valid_pdg_ids(self):
        path = particle.data.basepath / "particle2022.csv"
        valid_pdg_ids = []
        
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            counter_row = 0
            for row in csv_reader:
                if counter_row == 0:
                    valid_pdg_ids.append(row[0])
                elif 2 <= counter_row:
                    valid_pdg_ids.append(int(row[0]))
                counter_row += 1
        self.list_of_all_valid_pdg_ids_ = valid_pdg_ids
        
                
    def particle_list(self):
        """
        Returns a nested python list containing all quantities from 
        the Oscar2013/Oscar2013Extended output as numerical values 
        with the following shape:
            
            Single Event:    [output_line][particle_quantity]
            Multiple Events: [event][output_line][particle_quantity]

        Returns
        -------
        list of numerical particle quantities : list

        """
        if self.num_events_ == 1:
            
            num_particles = self.num_output_per_event_[1] 
            particle_array=[]
            
            for i in range(0, num_particles):
                particle = self.__particle_as_list(self.particle_list_[i])
                particle_array.append(particle)
                
        elif self.num_events_ > 1:
            num_particles = self.num_output_per_event_[:,1]
            num_events = self.num_events_
            
            particle_array = []
            
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
            
            Single Event:    [particle_object]
            Multiple Events: [event][particle_object]

        Returns
        -------
        List:
            List of particle objects
        """
        return self.particle_list_
                
                
    def oscar_type(self):
        """
        Returns the Oscar type of the given Oscar file as string
        "Oscar2013" or "Oscar2013Extended"
        """
        return self.oscar_type_
    
    
    def num_output_per_event(self):
        """
        Returns a numpy array containing the event number (starting with 1) 
        and the corresponding number of particles created in this event as
        
        num_output_per_event[event_n, numer_of_particles_in_event_n]
        
        num_output_per_event is updated with every manipulation e.g. after 
        applying cuts.

        Returns
        -------
        numpy.ndarray
            Array containing the event number and the number of particles
        """
        return self.num_output_per_event_
    
    
    def num_events(self):
        """
        Returns the number of events in particle_list
        
        num_events is updated with every manipulation e.g. after 
        applying cuts.

        Returns
        -------
        int:
            Number of events in particle_list
        """
        return self.num_events_
    
    
    def charged_particles(self):
        """
        Keep only charged particles in particle_list

        Returns
        -------
        Oscar oject:
            Containing charged particles in every event only
        """
        if self.num_events_ == 1:
            self.particle_list_ = [elem for elem in self.particle_list_ 
                                   if elem.charge != 0]
            new_length = len(self.particle_list_)
            self.num_output_per_event_[1] = new_length
        else:
            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                          if elem.charge != 0]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length
                
        return self


    def uncharged_particles(self):
        """
        Keep only uncharged particles in particle_list

        Returns
        -------
        Oscar oject:
            Containing uncharged particles in every event only
        """
        if self.num_events_ == 1:
            self.particle_list_ = [elem for elem in self.particle_list_ 
                                   if elem.charge == 0]
            new_length = len(self.particle_list_)
            self.num_output_per_event_[1] = new_length
        else:
            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                          if elem.charge == 0]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length
                
        return self
                
                
    def strange_particles(self):
        """
        Keep only strange particles in particle_list

        Returns
        -------
        Oscar oject:
            Containing strange particles in every event only
        """
        if self.num_events_ == 1:
            self.particle_list_ = [elem for elem in self.particle_list_ 
                                   if elem.is_strange() ]
            new_length = len(self.particle_list_)
            self.num_output_per_event_[1] = new_length
        else:
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
        Oscar object
            Containing only particle species specified by pdg_list for every event

        """
        if not isinstance(pdg_list, (str, int, list, np.integer, np.ndarray, tuple)):
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')
                
        elif isinstance(pdg_list, (int, str, np.integer)):
            pdg_list = int(pdg_list)
            
            self.__check_if_pdg_is_valid(pdg_list)
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ 
                                       if int(elem.pdg) == pdg_list]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                              if int(elem.pdg) == pdg_list]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length
                    
        elif isinstance(pdg_list, (list, np.ndarray, tuple)):
            pdg_list = np.asarray(pdg_list, dtype=np.int64)
            
            self.__check_if_pdg_is_valid(pdg_list)
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ 
                                       if int(elem.pdg) in pdg_list]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                              if int(elem.pdg) in pdg_list]
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
        Oscar object
            Containing all but the specified particle species for every event

        """
        if not isinstance(pdg_list, (str, int, list, np.integer, np.ndarray, tuple)):
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')
                
        elif isinstance(pdg_list, (int, str, np.integer)):
            pdg_list = int(pdg_list)
            
            self.__check_if_pdg_is_valid(pdg_list)
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ 
                                       if int(elem.pdg) != pdg_list]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                              if int(elem.pdg) != pdg_list]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length
                    
        elif isinstance(pdg_list, (list, np.ndarray, tuple)):
            pdg_list = np.asarray(pdg_list, dtype=np.int64)
            
            self.__check_if_pdg_is_valid(pdg_list)
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ 
                                       if not int(elem.pdg) in pdg_list]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                              if not int(elem.pdg) in pdg_list]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length     
                    
        else:
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple') 
        return self
    
    
    def participants(self):
        """
        Keep only participants in particle_list

        Returns
        -------
        Oscar oject:
            Containing participants in every event only
        """
        if self.num_events_ == 1:
            self.particle_list_ = [elem for elem in self.particle_list_ 
                                   if elem.ncoll != 0]
            new_length = len(self.particle_list_)
            self.num_output_per_event_[1] = new_length
            
        elif self.num_events_ > 1:
            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if elem.ncoll != 0]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length
                
        return self
    
    
    def spectators(self):
        """
        Keep only spectators in particle_list

        Returns
        -------
        Oscar oject:
            Containing spectators in every event only
        """
        if self.num_events_ == 1:
            self.particle_list_ = [elem for elem in self.particle_list_ 
                                   if elem.ncoll == 0 ]
            new_length = len(self.particle_list_)
            self.num_output_per_event_[1] = new_length
        elif self.num_events_ > 1:
            
            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                          if elem.ncoll == 0 ]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length
                
        return self
        
              
    def pt_cut(self, cut_value):
        """
        Apply p_t cut to all events and remove all particles with a transverse 
        momentum not complying with cut_value

        Parameters
        ----------
        cut_value : float
            If a single value is passed, the cut is applyed symmetrically 
            around 0.
            For example, if cut_value = 1, only particles with p_t in 
            [-1.0, 1.0] are kept.
            
        pdg_list : tuple
            To specify an asymmetric acceptance range for the transverse 
            momentum of particles, pass a tuple (cut_min, cut_max)

        Returns
        -------
        Oscar object
            Containing only particles complying with the p_t cut for all events
        """
        
        if not isinstance(cut_value, (int, float, np.number)) or cut_value < 0:
            raise TypeError('Input value must be a positive number')
                
        elif isinstance(cut_value, (int, float, np.number)):
            # cut symmetrically around 0
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       elem.pt_abs() > cut_value]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              elem.pt_abs() > cut_value]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length   
        else:
            raise TypeError('Input value must be a positive number')        
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
            
        pdg_list : tuple
            To specify an asymmetric acceptance range for the rapidity
            of particles, pass a tuple (cut_min, cut_max)

        Returns
        -------
        Oscar object
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
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       -limit<=elem.momentum_rapidity_Y()<=limit]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              -limit<=elem.momentum_rapidity_Y()<=limit]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length 
                    
        elif isinstance(cut_value, tuple):
            lim_max = max(cut_value[0], cut_value[1])
            lim_min = min(cut_value[0], cut_value[1])
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       lim_min<=elem.momentum_rapidity_Y()<=lim_max]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              lim_min<=elem.momentum_rapidity_Y()<=lim_max]
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
            
        pdg_list : tuple
            To specify an asymmetric acceptance range for the pseudo-rapidity
            of particles, pass a tuple (cut_min, cut_max)

        Returns
        -------
        Oscar object
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
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       -limit<=elem.pseudorapidity()<=limit]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[0,1] = new_length
            else:
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
                                       lim_min<=elem.pseudorapidity()<=lim_max]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              lim_min<=elem.pseudorapidity()<=lim_max]
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
            If a single value is passed, the cut is applyed symmetrically 
            around 0.
            For example, if cut_value = 1, only particles with spatial rapidity 
            in [-1.0, 1.0] are kept.
            
        pdg_list : tuple
            To specify an asymmetric acceptance range for the spatial rapidity
            of particles, pass a tuple (cut_min, cut_max)

        Returns
        -------
        Oscar object
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
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       -limit<=elem.spatial_rapidity()<=limit]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              -limit<=elem.spatial_rapidity()<=limit]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length 
                    
        elif isinstance(cut_value, tuple):
            lim_max = max(cut_value[0], cut_value[1])
            lim_min = min(cut_value[0], cut_value[1])
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       lim_min<=elem.spatial_rapidity()<=lim_max]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              lim_min<=elem.spatial_rapidity()<=lim_max]
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
        min_multiplicity : float
            Lower bound for multiplicity. If the multiplicity of an event is 
            lower than min_multiplicity, this event is discarded.

        Returns
        -------
        Oscar object
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
        header = []
        event_footer = ''
        format_oscar2013 = '%g %g %g %g %g %.9g %.9g %.9g %.9g %d %d %d'
        format_oscar2013_extended = '%g %g %g %g %g %.9g %.9g %.9g %.9g %d %d %d %d %g %g %d %d %g %d %d %d'
        
        line_in_initial_file = open(self.PATH_OSCAR_,'r')
        counter_line = 0
        while True:
            line = line_in_initial_file.readline()
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
        line_in_initial_file.close()
        
        event_footer = event_footer.replace('\n','').split(' ')
        
        output = open(output_file, "w")
        for i in range(3):
            output.write(header[i])
        output.close()
        
        with open(output_file, "a") as f_out:
            if self.num_events_ == 1:
                event = self.num_output_per_event_[0]
                num_out = self.num_output_per_event_[1]
                particle_output = np.asarray(self.particle_list())
             
                f_out.write('# event '+ str(event)+' out '+ str(num_out)+'\n')
                if self.oscar_type_ == 'Oscar2013':
                    np.savetxt(f_out, particle_output, delimiter=' ', newline='\n', fmt=format_oscar2013)
                elif self.oscar_type_ == 'Oscar2013Extended' or self.oscar_type_ == 'Oscar2013Extended IC':
                    np.savetxt(f_out, particle_output, delimiter=' ', newline='\n', fmt=format_oscar2013_extended)
                f_out.write(self.event_end_lines_[event])
            else:
                for i in range(self.num_events_):
                    event = self.num_output_per_event_[i,0]
                    num_out = self.num_output_per_event_[i,1]
                    particle_output = np.asarray(self.particle_list()[i])
                 
                    f_out.write('# event '+ str(event)+' out '+ str(num_out)+'\n')
                    if self.oscar_type_ == 'Oscar2013':
                        np.savetxt(f_out, particle_output, delimiter=' ', newline='\n', fmt=format_oscar2013)
                    elif self.oscar_type_ == 'Oscar2013Extended'  or self.oscar_type_ == 'Oscar2013Extended IC':
                        np.savetxt(f_out, particle_output, delimiter=' ', newline='\n', fmt=format_oscar2013_extended)
                    f_out.write(self.event_end_lines_[event])
        f_out.close()
