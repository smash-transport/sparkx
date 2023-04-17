from ParticleClass import Particle
import numpy as np
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
        self.optional_arguments_ = kwargs
        
        self.set_oscar_type()
        self.set_num_events()
        self.set_num_output_per_event()        
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
        particle_list.append(float(particle.t()))
        particle_list.append(float(particle.x()))
        particle_list.append(float(particle.y()))
        particle_list.append(float(particle.z()))
        particle_list.append(float(particle.mass()))
        particle_list.append(float(particle.E()))
        particle_list.append(float(particle.px()))
        particle_list.append(float(particle.py()))
        particle_list.append(float(particle.pz()))
        particle_list.append(int(particle.pdg()))
        particle_list.append(int(particle.ID()))
        particle_list.append(int(particle.charge()))
        
        if self.oscar_type_ == 'Oscar2013Extended':
            particle_list.append(int(particle.ncoll()))
            particle_list.append(float(particle.form_time()))
            particle_list.append(int(particle.xsecfac()))
            particle_list.append(int(particle.proc_id_origin()))
            particle_list.append(int(particle.proc_type_origin()))
            particle_list.append(float(particle.t_last_coll()))
            particle_list.append(int(particle.pdg_mother1()))
            particle_list.append(int(particle.pdg_mother2()))
                                 
            if particle.baryon_number() != None:                         
                particle_list.append(int(particle.baryon_number()))
            
            
        elif self.oscar_type_ != 'Oscar2013' and self.oscar_type_ != 'Oscar2013Extended':
            raise TypeError('Input file not in OSCAR2013 or OSCAR2013Extended format')
            
        return particle_list
    
        
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
            elif 'event' in line and 'out' in line:
                continue
            elif '#' in line and 'end' in line:
                particle_list.append(data)
                data = []
            else:
                data_line = line.replace('\n','').split(' ')
                particle = Particle()
                
                if self.oscar_type_ == 'Oscar2013':
                    particle.set_quantities_OSCAR2013(data_line)
                elif self.oscar_type_ == 'Oscar2013Extended':
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
        elif len(first_line) == 23 or first_line[0] == '#!OSCAR2013Extended':
            self.oscar_type_ = 'Oscar2013Extended'
        else:
            raise TypeError('Input file must follow the Oscar2013 or '+\
                            'Oscar2013Extended format ')
                
                
    def set_num_output_per_event(self):
        file = open(self.PATH_OSCAR_ , 'r')
        event_output = []
        
        while True:
            line = file.readline()
            if not line:
                break
            elif '#' in line and 'out' in line:
                line_str = line.replace('\n','').split(' ')
                event = line_str[2]
                num_output = line_str[4]
                event_output.append([event, num_output])
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
        the Oscar2013/Oscar2013Extended output as particle Objects 
        from ParticleClass:
            
            Single Event:    [particle_object]
            Multiple Events: [event][particle_object]

        Returns
        -------
        list of particle objects : list

        """
        return self.particle_list_
                
                
    def oscar_type(self):
        return self.oscar_type_
    
    
    def num_output_per_event(self):
        return self.num_output_per_event_
    
    
    def num_events(self):
        return self.num_events_
    
    
    def charged_particles(self):
        if self.num_events_ == 1:
            self.particle_list_ = [elem for elem in self.particle_list_ 
                                   if elem.charge() != 0]
            new_length = len(self.particle_list_)
            self.num_output_per_event_[1] = new_length
        else:
            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                          if elem.charge() != 0]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length
                
        return self


    def uncharged_particles(self):
        if self.num_events_ == 1:
            self.particle_list_ = [elem for elem in self.particle_list_ 
                                   if elem.charge() == 0]
            new_length = len(self.particle_list_)
            self.num_output_per_event_[1] = new_length
        else:
            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                          if elem.charge() == 0]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length
                
        return self
                
                
    def strange_particles(self):
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
        if not isinstance(pdg_list, (str, int, list, np.integer, np.ndarray, tuple)):
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')
                
        elif isinstance(pdg_list, (int, str, np.integer)):
            pdg_list = int(pdg_list)
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ 
                                       if int(elem.pdg()) == pdg_list]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                              if int(elem.pdg()) == pdg_list]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length
                    
        elif isinstance(pdg_list, (list, np.ndarray, tuple)):
            pdg_list = np.asarray(pdg_list, dtype=np.int64)
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ 
                                       if int(elem.pdg()) in pdg_list]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                              if int(elem.pdg()) in pdg_list]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length     
                    
        else:
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple') 
        return self
    
    
    def exclude_particle_species(self, pdg_list):
        if not isinstance(pdg_list, (str, int, list, np.integer, np.ndarray, tuple)):
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple')
                
        elif isinstance(pdg_list, (int, str, np.integer)):
            pdg_list = int(pdg_list)
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ 
                                       if int(elem.pdg()) != pdg_list]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                              if int(elem.pdg()) != pdg_list]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length
                    
        elif isinstance(pdg_list, (list, np.ndarray, tuple)):
            pdg_list = np.asarray(pdg_list, dtype=np.int64)
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ 
                                       if not int(elem.pdg()) in pdg_list]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                              if not int(elem.pdg()) in pdg_list]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length     
                    
        else:
            raise TypeError('Input value for pgd codes has not one of the ' +\
                            'following types: str, int, np.integer, list ' +\
                            'of str, list of int, np.ndarray, tuple') 
        return self
    
    
    def participants(self):
        if self.num_events_ == 1:
            self.particle_list_ = [elem for elem in self.particle_list_ 
                                   if elem.ncoll() != 0]
            new_length = len(self.particle_list_)
            self.num_output_per_event_[1] = new_length
            
        elif self.num_events_ > 1:
            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] if elem.ncoll() != 0]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length
                
        return self
    
    
    def spectators(self):
        if self.num_events_ == 1:
            self.particle_list_ = [elem for elem in self.particle_list_ 
                                   if elem.ncoll() == 0 ]
            new_length = len(self.particle_list_)
            self.num_output_per_event_[1] = new_length
        elif self.num_events_ > 1:
            
            for i in range(0, self.num_events_):
                self.particle_list_[i] = [elem for elem in self.particle_list_[i] 
                                          if elem.ncoll() == 0 ]
                new_length = len(self.particle_list_[i])
                self.num_output_per_event_[i, 1] = new_length
                
        return self
        
              
    def pt_cut(self, cut_value):
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
                self.num_output_per_event_[1] = new_length
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


  
PATH = '/Users/nils/smash-devel/build/data/0/particle_lists.oscar'
aaa = Oscar(PATH, events=(0,1))
print(aaa.num_output_per_event())
aaa.rapidity_cut((0.1,-0.1))
print(aaa.num_output_per_event())