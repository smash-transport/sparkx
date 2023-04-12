from ParticleClass import Particle
import numpy as np
import warnings
import os

class Jetscape:
    
    def __init__(self, JETSCAPE_FILE, **kwargs):
        #kwargs: 
        #   events:
        #        - int: load single event
        #        - tuple: (event_start, event_end) load all events in given range 
        
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
        particle_list[0] = int(particle.ID())
        particle_list[1]  = int(particle.pdg())
        particle_list[2]  = int(particle.status())
        particle_list[3]  = float(particle.E())
        particle_list[4]  = float(particle.px())
        particle_list[5]  = float(particle.py())
        particle_list[6]  = float(particle.pz())
            
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
                if int(data_line[2]) == int(kwargs['events'][0])+1:
                    continue
                else:
                    particle_list.append(data)
                    data = []
            else:
                data_line = line.replace('\n','').replace('\t',' ').split(' ')
                particle = Particle()
                
                particle.set_quantities_JETSCAPE(data_line)
                
                # Check for filters by method with a dictionary
                # and do not append if empty (Method: WantToKeep(particle, filter) -> True/False)
                
                data.append(particle)
        print(len(particle_list))
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
            elif '#' in line and 'weight' in line:
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
        print(len(self.particle_list_))
        if self.num_events_ == 1:
            
            num_particles = self.num_output_per_event_[1]
             
            particle_array = []
            
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
            
              
    def pt_cut(self, cut_value):
        if not isinstance(cut_value, (int, float, np.number)) or cut_value < 0:
            raise TypeError('Input value must be a positive number')
                
        elif isinstance(cut_value, (int, float, np.number)):
            # cut symmetrically around 0
            
            if self.num_events_ == 1:
                self.particle_list_ = [elem for elem in self.particle_list_ if
                                       elem.pt_abs() <= cut_value]
                new_length = len(self.particle_list_)
                self.num_output_per_event_[1] = new_length
            else:
                for i in range(0, self.num_events_):
                    self.particle_list_[i] = [elem for elem in self.particle_list_[i] if
                                              elem.pt_abs() <= cut_value]
                    new_length = len(self.particle_list_[i])
                    self.num_output_per_event_[i, 1] = new_length   
        else:
            raise TypeError('Input value must be a positive number')        
        return self    
        
        
    def rapidity_cut(self, cut_value):
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
    
    def particle_objects_list(self):
        return self.particle_list_