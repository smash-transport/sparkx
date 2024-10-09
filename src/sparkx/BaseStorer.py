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
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Optional
from sparkx.loader.BaseLoader import BaseLoader


class BaseStorer(ABC):
    """
    Defines a generic BaseStorer object.

    Attributes
    ----------
    num_output_per_event_ : numpy.array
        Array containing the event number and the number of particles in this
        event as num_output_per_event_[event i][num_output in event i] (updated
        when filters are applied)
    num_events_ : int
        Number of events contained in the Oscar object (updated when filters
        are applied)
    loader_: Loader object
        Loader object that loads the data


    Methods
    -------
    load:
        Load data
    particle_list:
        Returns current events data as nested list
    particle_objects_list:
        Returns current events data as nested list of Particle objects
    num_events:
        Get number of events
    num_output_per_event:
        Get number of particles in each event
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
    pT_cut:
        Apply pT cut to all particles
    rapidity_cut:
        Apply rapidity cut to all particles
    pseudorapidity_cut:
        Apply pseudorapidity cut to all particles
    spacetime_rapidity_cut:
        Apply spacetime rapidity cut to all particles
    multiplicity_cut:
        Apply multiplicity cut to all particles
     spacetime_cut:
        Apply spacetime cut to all particles
    """

    def __init__(
        self, path: Union[str, List[List["Particle"]]], **kwargs
    ) -> None:
        """
        Parameters
        ----------
        data : list
            Nested list containing the event data
        num_events : int
            Number of events in the  data
        num_output_per_event : numpy.array
            Array containing the event number and the number of particles in this
            event as num_output_per_event_[event i][num_output in event i]

        Returns
        -------
        None
        """
        self.loader_: Optional[BaseLoader] = None
        self.num_output_per_event_: Optional[np.ndarray] = None
        self.num_events_: Optional[int] = None
        self.particle_list_: Optional[List] = None
        self.create_loader(path)
        if self.loader_ is not None:
            (
                self.particle_list_,
                self.num_events_,
                self.num_output_per_event_,
            ) = self.loader_.load(**kwargs)
        else:
            raise ValueError("Loader has not been created properly")

    @abstractmethod
    def create_loader(self, arg: Union[str, List[List["Particle"]]]) -> None:
        raise NotImplementedError("This method is not implemented yet")

    def num_output_per_event(self) -> Optional[np.ndarray]:
        """
        Returns a numpy array containing the event number (starting with 1)
        and the corresponding number of particles created in this event as

        num_output_per_event[event_n, number_of_particles_in_event_n]

        num_output_per_event is updated with every manipulation e.g. after
        applying cuts.

        Returns
        -------
        num_output_per_event_ : numpy.ndarray
            Array containing the event number and the corresponding number of
            particles
        """
        return self.num_output_per_event_

    def num_events(self) -> Optional[int]:
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

    def particle_objects_list(self) -> Optional[List]:
        """
        Returns a nested python list containing all particles from
        the data as particle objects
        from Particle:

           | Single Event:    [particle_object]
           | Multiple Events: [event][particle_object]

        Returns
        -------
        particle_list_ : list
            List of particle objects from Particle
        """
        return self.particle_list_

    @abstractmethod
    def _particle_as_list(self, particle: "Particle") -> List:
        raise NotImplementedError("This method is not implemented yet")

    def particle_list(self) -> List:
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

        if self.num_output_per_event_ is None:
            raise ValueError("num_output_per_event_ is not set")
        if self.particle_list_ is None:
            raise ValueError("particle_list_ is not set")
        if num_events is None:
            raise ValueError("num_events_ is not set")

        if num_events == 1:
            num_particles = self.num_output_per_event_[0][1]
        else:
            num_particles = self.num_output_per_event_[:, 1]

        particle_array: List[List] = []

        if num_events == 1:
            for i_part in range(0, num_particles):
                particle = self.particle_list_[0][i_part]
                particle_array.append(self._particle_as_list(particle))
        else:
            for i_ev in range(0, num_events):
                event = []
                for i_part in range(0, num_particles[i_ev]):
                    particle = self.particle_list_[i_ev][i_part]
                    event.append(self._particle_as_list(particle))
                particle_array.append(event)

        return particle_array

    def charged_particles(self) -> "BaseStorer":
        """
        Keep only charged particles in particle_list

        Returns
        -------
        self : BaseStorer object
            Containing charged particles in every event only
        """

        self.particle_list_ = charged_particles(self.particle_list_)

        self._update_num_output_per_event_after_filter()
        return self

    def uncharged_particles(self) -> "BaseStorer":
        """
        Keep only uncharged particles in particle_list

        Returns
        -------
        self : BaseStorer object
            Containing uncharged particles in every event only
        """

        self.particle_list_ = uncharged_particles(self.particle_list_)
        self._update_num_output_per_event_after_filter()

        return self

    def strange_particles(self) -> "BaseStorer":
        """
        Keep only strange particles in particle_list

        Returns
        -------
        self : BaseStorer object
            Containing strange particles in every event only
        """

        self.particle_list_ = strange_particles(self.particle_list_)
        self._update_num_output_per_event_after_filter()

        return self

    def particle_species(
        self, pdg_list: Union[int, Union[Tuple[int], List[int], np.ndarray]]
    ) -> "BaseStorer":
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
        self : BaseStorer object
            Containing only particle species specified by pdg_list for every event

        """

        self.particle_list_ = particle_species(self.particle_list_, pdg_list)
        self._update_num_output_per_event_after_filter()

        return self

    def remove_particle_species(
        self, pdg_list: Union[int, Union[Tuple[int], List[int], np.ndarray]]
    ) -> "BaseStorer":
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
        self : BaseStorer object
            Containing all but the specified particle species in every event

        """

        self.particle_list_ = remove_particle_species(
            self.particle_list_, pdg_list
        )
        self._update_num_output_per_event_after_filter()

        return self

    def participants(self) -> "BaseStorer":
        """
        Keep only participants in particle_list

        Returns
        -------
        self : BaseStorer object
            Containing participants in every event only
        """

        self.particle_list_ = participants(self.particle_list_)
        self._update_num_output_per_event_after_filter()

        return self

    def spectators(self) -> "BaseStorer":
        """
        Keep only spectators in particle_list

        Returns
        -------
        self : BaseStorer object
            Containing spectators in every event only
        """

        self.particle_list_ = spectators(self.particle_list_)
        self._update_num_output_per_event_after_filter()

        return self

    def lower_event_energy_cut(
        self, minimum_event_energy: Union[int, float]
    ) -> "BaseStorer":
        """
        Filters out events with total energy lower than a threshold.

        Parameters
        ----------
        minimum_event_energy : int or float
            The minimum event energy threshold. Should be a positive integer or float.

        Returns
        -------
        self: BaseStorer object
            The updated instance of the class contains only events above the
            energy threshold.

        Raises
        ------
        TypeError
            If the minimum_event_energy parameter is not an integer or float.
        ValueError
            If the minimum_event_energy parameter is less than or equal to 0.
        """

        self.particle_list_ = lower_event_energy_cut(
            self.particle_list_, minimum_event_energy
        )
        self._update_num_output_per_event_after_filter()

        return self

    def pT_cut(
        self, cut_value_tuple: Tuple[Union[float, None], Union[float, None]]
    ) -> "BaseStorer":
        """
        Apply transverse momentum cut to all events by passing an acceptance
        range by :code:`cut_value_tuple`. All particles outside this range will
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
        self : BaseStorer object
            Containing only particles complying with the transverse momentum
            cut for all events
        """

        self.particle_list_ = pT_cut(self.particle_list_, cut_value_tuple)
        self._update_num_output_per_event_after_filter()

        return self

    def rapidity_cut(
        self, cut_value: Union[float, Tuple[float, float]]
    ) -> "BaseStorer":
        """
        Apply rapidity cut to all events and remove all particles with rapidity
        not complying with cut_value

        Parameters
        ----------
        cut_value : float
            If a single value is passed, the cut is applied symmetrically
            around 0.
            For example, if :code:`cut_value = 1`, only particles with rapidity in
            :code:`[-1.0, 1.0]` are kept.

        cut_value : tuple
            To specify an asymmetric acceptance range for the rapidity
            of particles, pass a tuple :code:`(cut_min, cut_max)`

        Returns
        -------
        self : BaseStorer object
            Containing only particles complying with the rapidity cut
            for all events
        """

        self.particle_list_ = rapidity_cut(self.particle_list_, cut_value)
        self._update_num_output_per_event_after_filter()

        return self

    def pseudorapidity_cut(
        self, cut_value: Union[float, Tuple[float, float]]
    ) -> "BaseStorer":
        """
        Apply pseudo-rapidity cut to all events and remove all particles with
        pseudo-rapidity not complying with cut_value

        Parameters
        ----------
        cut_value : float
            If a single value is passed, the cut is applied symmetrically
            around 0.
            For example, if :code:`cut_value = 1`, only particles with rapidity in
            :code:`[-1.0, 1.0]` are kept.

        cut_value : tuple
            To specify an asymmetric acceptance range for the pseudo-rapidity
            of particles, pass a tuple :code:`(cut_min, cut_max)`

        Returns
        -------
        self : BaseStorer object
            Containing only particles complying with the pseudo-rapidity cut
            for all events
        """

        self.particle_list_ = pseudorapidity_cut(self.particle_list_, cut_value)
        self._update_num_output_per_event_after_filter()

        return self

    def spatial_rapidity_cut(
        self, cut_value: Union[float, Tuple[float, float]]
    ) -> "BaseStorer":
        """
        Apply spacetime rapidity (space-time rapidity) cut to all events and
        remove all particles with spcetime rapidity not complying with cut_value

        Parameters
        ----------
        cut_value : float
            If a single value is passed, the cut is applied symmetrically
            around 0.
            For example, if :code:`cut_value = 1`, only particles with spacetime
            rapidity in :code:`[-1.0, 1.0]` are kept.

        cut_value : tuple
            To specify an asymmetric acceptance range for the spacetime rapidity
            of particles, pass a tuple :code:`(cut_min, cut_max)`

        Returns
        -------
        self : BaseStorer object
            Containing only particles complying with the spacetime rapidity cut
            for all events
        """

        self.particle_list_ = spacetime_rapidity_cut(
            self.particle_list_, cut_value
        )
        self._update_num_output_per_event_after_filter()

        return self

    def multiplicity_cut(self, min_multiplicity: int) -> "BaseStorer":
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
        self : BaseStorer object
            Containing only events with a :code:`multiplicity >= min_multiplicity`
        """

        self.particle_list_ = multiplicity_cut(
            self.particle_list_, min_multiplicity
        )
        self._update_num_output_per_event_after_filter()

        return self

    def spacetime_cut(self, dim, cut_value_tuple: Tuple[float, float]):
        """
        Apply spacetime cut to all events by passing an acceptance range by
        :code:`cut_value_tuple`. All particles outside this range will
        be removed.

        Parameters
        ----------
        dim : string
            String naming the dimension on which to apply the cut.
            Options: :code:`t`,:code:`x`,:code:`y`,:code:`z`
        cut_value_tuple : tuple
            Tuple with the upper and lower limits of the coordinate space
            acceptance range :code:`(cut_min, cut_max)`. If one of the limits
            is not required, set it to :code:`None`, i.e.
            :code:`(None, cut_max)` or :code:`(cut_min, None)`.

        Returns
        -------
        self : BaseStorer object
            Containing only particles complying with the spacetime cut for all events
        """
        self.particle_list_ = spacetime_cut(
            self.particle_list_, dim, cut_value_tuple
        )
        self._update_num_output_per_event_after_filter()

        return self

    def _update_num_output_per_event_after_filter(self) -> None:
        if self.num_output_per_event_ is None:
            raise ValueError("num_output_per_event_ is not set")
        if self.particle_list_ is None:
            raise ValueError("particle_list_ is not set")

        if self.num_output_per_event_.ndim == 1:
            # Handle the case where num_output_per_event_ is a one-dimensional array
            self.num_output_per_event_[1] = len(self.particle_list_[0])
        elif self.num_output_per_event_.ndim == 2:
            # Handle the case where num_output_per_event_ is a two-dimensional array
            for event in range(len(self.particle_list_)):
                self.num_output_per_event_[event][1] = len(
                    self.particle_list_[event]
                )
        else:
            raise ValueError(
                "num_output_per_event_ has an unexpected number of dimensions"
            )

    @abstractmethod
    def print_particle_lists_to_file(self, output_file: str) -> None:
        """
        Prints the particle lists to a specified file.

        This method should be implemented by subclasses to print the particle
        lists to the specified output file. The method raises a NotImplementedError
        if it is not overridden by a subclass.

        Parameters
        ----------
        output_file : str
            The path to the file where the particle lists will be printed.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Returns
        -------
        None
        """
        raise NotImplementedError("This method is not implemented yet")
