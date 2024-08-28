# ===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from sparkx.Filter import *
import numpy as np
from sparkx.loader.ParticleObjectLoader import ParticleObjectLoader
from sparkx.BaseStorer import BaseStorer
from typing import List, Dict, Tuple, Optional, Union


class ParticleObjectStorer(BaseStorer):
    """
    Defines a ParticleObjectStorer object, which saves particle object lists.

    This is a wrapper for a list of Particle objects. It's methods allow to
    directly act on all contained events as applying acceptance filters
    (e.g. un/charged particles, spectators/participants) to keep/remove particles
    by their PDG codes or to apply cuts (e.g. multiplicity, pseudo/rapidity, pT).
    Once these filters are applied, the new data set can be accessed as a

    1) nested list containing all quantities
    2) list containing Particle objects from the Particle class

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
    particle_status:
        Keep only particles with a given status flag
    print_particle_lists_to_file:
        Print current particle data to file with same format

    Examples
    --------
    Create a Dummy object and access particle data as a nested list

    >>> from sparkx.Particle import Particle
    >>> particle1 = Particle()
    >>> particle1.t = 1.0
    >>> ...
    >>> particle2 = Particle()
    >>> particle2.t = 1.0
    >>> ...
    >>> particles = [particle1, particle2]
    >>> dummy = DummyStorer(particles)
    >>> nested_list = dummy.particle_list()
    >>> print(nested_list)
    [[[1.0,...],[1.0,...]]]

    """

    num_output_per_event_: np.ndarray
    num_events_: int

    def __init__(
        self,
        particle_object_list: List[List["Particle"]],
        **kwargs: Dict[str, Optional[Tuple[int, int]]]
    ) -> None:
        """
        Initializes a new instance of the DummyStorer class.

        This method initializes a new instance of the DummyStorer class with the specified list of particle objects and optional arguments. It calls the superclass's constructor with the particle_object_list and kwargs parameters and then deletes the loader_ attribute.

        Parameters
        ----------
        particle_object_list : list of Particle objects
            The list of particle objects to store.
        kwargs : dict
            A dictionary of optional arguments.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        super().__init__(particle_object_list, **kwargs)
        del self.loader_

    def create_loader(
        self, particle_object_list: Union[str, List[List["Particle"]]]
    ) -> None:
        """
        Creates a new ParticleObjectLoader object.

        This method creates a new ParticleObjectLoader object with the specified list of particle objects and assigns it to the loader_ attribute.

        Parameters
        ----------
        particle_object_list : list of Particle objects
            The list of particle objects to load.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        self.loader_ = ParticleObjectLoader(particle_object_list)

    def _particle_as_list(self, particle: "Particle") -> List[float]:
        """
        Converts a Particle object into a list.

        This method takes a Particle object and converts it into a list of its attributes. The attributes are added to the list in the following order: t, x, y, z, mass, E, px, py, pz, pdg, ID, charge, ncoll, form_time, xsecfac, proc_id_origin, proc_type_origin, t_last_coll, pdg_mother1, pdg_mother2, baryon_number, strangeness, weight, status.

        Parameters
        ----------
        particle : Particle
            The Particle object to convert into a list.

        Raises
        ------
        None

        Returns
        -------
        particle_list : list
            A list of the attributes of the Particle object.
        """
        particle_list: List[float] = [
            particle.t,
            particle.x,
            particle.y,
            particle.z,
            particle.mass,
            particle.E,
            particle.px,
            particle.py,
            particle.pz,
            particle.pdg,
            particle.ID,
            particle.charge,
            particle.ncoll,
            particle.form_time,
            particle.xsecfac,
            particle.proc_id_origin,
            particle.proc_type_origin,
            particle.t_last_coll,
            particle.pdg_mother1,
            particle.pdg_mother2,
            particle.baryon_number,
            particle.strangeness,
            particle.weight,
            particle.status,
        ]
        return particle_list

    def print_particle_lists_to_file(self, filename: str) -> None:
        """
        Prints the current particle data to a file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        with open(filename, "w") as f:
            if not isinstance(self.particle_list_, list):
                raise TypeError(
                    "The particle_list must be a list of lists of Particle objects"
                )
            for event in self.particle_list_:
                for particle in event:
                    # Extract the attributes from the particle object
                    particle_data: List[float] = [
                        particle.t,
                        particle.x,
                        particle.y,
                        particle.z,
                        particle.mass,
                        particle.E,
                        particle.px,
                        particle.py,
                        particle.pz,
                        particle.pdg,
                        particle.ID,
                        particle.charge,
                        particle.ncoll,
                        particle.form_time,
                        particle.xsecfac,
                        particle.proc_id_origin,
                        particle.proc_type_origin,
                        particle.t_last_coll,
                        particle.pdg_mother1,
                        particle.pdg_mother2,
                        particle.status,
                        particle.baryon_number,
                    ]
                    f.write(",".join(map(str, particle_data)) + "\n")
