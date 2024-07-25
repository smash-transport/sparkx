#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
from sparkx.Filter import *
import numpy as np
from sparkx.Loader.DummyLoader import DummyLoader
from sparkx.Storer.BaseStorer import BaseStorer

class Dummy(BaseStorer):
    """
    Defines a Dummy object.

    This is a wrapper for a list of Particle objects. It's methodsallow to 
    directly act on all contained events as applying acceptance filters
    (e.g. un/charged particles, spectators/participants) to keep/remove particles
    by their PDG codes or to apply cuts (e.g. multiplicity, pseudo/rapidity, pT).
    Once these filters are applied, the new data set can be accessed as a

    1) nested list containing all quantities
    2) list containing Particle objects from the Particle class

    or it may be printed to a file complying with the input format.

    .. note::
        If filters are applied, be aware that not all cuts commute.

    Parameters
    ----------
    None

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
              - From the input particle object list load only a single event by |br|
                specifying :code:`events=i` where i is event number i.
            * - :code:`events` (tuple)
              - From the input particle object list load only a range of events |br|
                given by the tuple :code:`(first_event, last_event)` |br|
                by specifying :code:`events=(first_event, last_event)` |br|
                where last_event is included.
            * - :code:`filters` (dict)
              - Apply filters on an event-by-event basis to directly filter the |br|
                particles after the read in of one event. This method saves |br|
                memory. The names of the filters are the same as the names of |br|
                the filter methods. All filters are applied in the order in |br|
                which they appear in the dictionary.

        .. |br| raw:: html

           <br />

    Attributes
    ----------
    num_output_per_event_ : numpy.array
        Array containing the event number and the number of particles in this
        event as num_output_per_event_[event i][num_output in event i] (updated
        when filters are applied)
    num_events_ : int
        Number of events contained in the particle object list (updated when filters
        are applied)


    Methods
    -------
    spacetime_cut:
        Apply spacetime cut to all particles
    particle_status:
        Keep only particles with a given status flag
    print_particle_lists_to_file:
        Print current particle data to file with same format

    Examples
    --------

    TBD
    
    """
    def __init__(self, particle_object_list, **kwargs):
        super().__init__(particle_object_list,**kwargs)  
        del self.loader_

    def create_loader(self, particle_object_list):
        self.loader_= DummyLoader(particle_object_list)

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
        particle_list.append(int(particle.ncoll))
        particle_list.append(float(particle.form_time))
        particle_list.append(float(particle.xsecfac))
        particle_list.append(int(particle.proc_id_origin))
        particle_list.append(int(particle.proc_type_origin))
        particle_list.append(float(particle.t_last_coll))
        particle_list.append(int(particle.pdg_mother1))
        particle_list.append(int(particle.pdg_mother2))
        particle_list.append(int(particle.baryon_number))
        particle_list.append(int(particle.strangeness))
        particle_list.append(int(particle.weight))
        particle_list.append(int(particle.status))

        return particle_list

    def particle_list(self):
        """
        Returns a nested python list containing all quantities from the
        current data as numerical values with the following shape:

            | Single Event:    [[output_line][particle_quantity]]
            | Multiple Events: [event][output_line][particle_quantity]

        Returns
        -------
        list
            Nested list containing the current data

        """
        num_events = self.num_events_
        
        if num_events == 1:
            num_particles = self.num_output_per_event_[0][1]
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
    
    #TODO: get num_events, num_output_per_event from LOADER

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

        self.particle_list_ = spacetime_cut(self.particle_list_, dim, cut_value_tuple)
        self.__update_num_output_per_event_after_filter()

        return self