from ParticleClass import Particle
import numpy as np
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
                cummulate_lines = 0
                for i in range(0, self.optional_arguments_['events']):
                    cummulate_lines += self.num_output_per_event_[i,1] + 2
                skip_lines = 3 + cummulate_lines
        elif isinstance(self.optional_arguments_['events'], tuple):
            line_start = self.optional_arguments_['events'][0]
            if line_start == 0:
                skip_lines = 3
            else:
                cummulate_lines = 0
                for i in range(0, line_start):
                    cummulate_lines += self.num_output_per_event_[i,1] + 2
                skip_lines = 3 + cummulate_lines
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
            cummulated_lines = np.sum(self.num_output_per_event_, axis=0)[1]
            # add number of comments
            cummulated_lines += int(2 * len(self.num_output_per_event_))
            
        elif isinstance(self.optional_arguments_['events'], int):
            read_event = self.optional_arguments_['events']
            cummulated_lines = int(self.num_output_per_event_[read_event,1] + 2)
            
        elif isinstance(self.optional_arguments_['events'], tuple):
            cummulated_lines = 0
            event_start = self.optional_arguments_['events'][0]
            event_end = self.optional_arguments_['events'][1]
            for i in range(event_start, event_end+1):
                cummulated_lines += int(self.num_output_per_event_[i, 1] + 2)
        else:
            raise TypeError('Value given as flag events is not of type int or a tuple')
            
        return cummulated_lines
    
    
    def __particle_as_array(self, particle):
        if self.oscar_type_ == 'Oscar2013Extended':
            particle_array = np.zeros(21)
            particle_array[0]  = float(particle.t())
            particle_array[1]  = float(particle.x())
            particle_array[2]  = float(particle.y())
            particle_array[3]  = float(particle.z())
            particle_array[4]  = float(particle.mass())
            particle_array[5]  = float(particle.E())
            particle_array[6]  = float(particle.px())
            particle_array[7]  = float(particle.py())
            particle_array[8]  = float(particle.pz())
            particle_array[9]  = int(particle.pdg())
            particle_array[10] = int(particle.ID())
            particle_array[11] = int(particle.charge())
            particle_array[12] = int(particle.ncoll())
            particle_array[13] = float(particle.form_time())
            particle_array[14] = int(particle.xsecfac())
            particle_array[15] = int(particle.proc_id_origin())
            particle_array[16] = int(particle.proc_type_origin())
            particle_array[17] = float(particle.t_last_coll())
            particle_array[18] = int(particle.pdg_mother1())
            particle_array[19] = int(particle.pdg_mother2())
            particle_array[20] = int(particle.baryon_number())
        
        elif self.oscar_type_ == 'Oscar2013':
            particle_array = np.zeros(12)
            particle_array[0]  = float(particle.t())
            particle_array[1]  = float(particle.x())
            particle_array[2]  = float(particle.y())
            particle_array[3]  = float(particle.z())
            particle_array[4]  = float(particle.mass())
            particle_array[5]  = float(particle.E())
            particle_array[6]  = float(particle.px())
            particle_array[7]  = float(particle.py())
            particle_array[8]  = float(particle.pz())
            particle_array[9]  = int(particle.pdg())
            particle_array[10] = int(particle.ID())
            particle_array[11] = int(particle.charge())
            
        else:
            raise TypeError('Input file not in OSCAR2013 or OSCAR2013Extended format')
            
        return particle_array
    
        
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
        if self.num_events_ == 1:
            
            num_particles = self.num_output_per_event_[1]
            
            if self.oscar_type_ == 'Oscar2013Extended':
                particle_array = np.zeros((num_particles, 21))
            elif self.oscar_type_ == 'Oscar2013':
                particle_array = np.zeros((num_particles, 12))
                
            for i in range(0, num_particles):
                particle_array[i] = self.__particle_as_array(self.particle_list_[i])
                
        elif self.num_events_ > 1:
            num_particles = self.num_output_per_event_[:,1]
            num_events = self.num_events_
            
            if self.oscar_type_ == 'Oscar2013Extended':
                particle_array = np.zeros((num_events, num_particles, 21))
            elif self.oscar_type_ == 'Oscar2013':
                particle_array = np.zeros((num_events, num_particles, 12))
            
            for i_ev in range(0, num_events):
                for i_part in range(0, num_particles):
                    particle_array[i_ev, i_part] = self.__particle_as_array(self.particle_list_[i_ev, i_part])
                    
        return particle_array
                
                
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
                
                
    def rapidity_cut(self, cut_value):
        if not isinstance(cut_value, (int, float, tuple)):
            raise TypeError('Input value must be a number or a tuple ' +\
                            'with the cut limits (cut_min, cut_max)')
                
        elif isinstance(cut_value, tuple) and len(cut_value) != 2:
            raise TypeError('The tuple containing the cut limits take 2 values')
                
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
