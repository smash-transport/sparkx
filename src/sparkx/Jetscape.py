# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from sparkx.Filter import *
import numpy as np
from sparkx.loader.JetscapeLoader import JetscapeLoader
from sparkx.Particle import Particle
from sparkx.BaseStorer import BaseStorer
from typing import List, Tuple, Union, Dict, Optional


class Jetscape(BaseStorer):
    """
    Defines a Jetscape object.

    The Jetscape class contains a single Jetscape hadron output file including
    all or only chosen events. It's methods allow to directly act on all
    contained events as applying acceptance filters (e.g. un/charged particles)
    to keep/romove particles by their PDG codes or to apply cuts
    (e.g. multiplicity, pseudo/rapidity, pT).
    Once these filters are applied, the new data set can be saved 1) as a nested
    list containing all quantities of the Jetscape format 2) as a list containing
    Particle objects from the Particle or it can be printed to a file
    complying with the input format.

    .. note::
        If filters are applied, be aware that not all cuts commute.

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
            * - :code:`filters` (dict)
              - Apply filters on an event-by-event basis to directly filter the |br|
                particles after the read in of one event. This method saves |br|
                memory. The names of the filters are the same as the names of |br|
                the filter methods. All filters are applied in the order in |br|
                which they appear in the dictionary.
            * - :code:`particletype` (str)
              - This parameter allows to switch from the standard hadron file |br|
                to the parton output of JETSCAPE. The parameter can be set to |br|
                :code:`particletype='hadron'` (default) or :code:`particletype='parton'`.
                Quark charges are multiplied by 3 to make them integer values.

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
    particle_status:
        Keep only particles with a given status flag
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

        >>> jetscape = Jetscape(JETSCAPE_FILE_PATH)
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

    **3. Constructor cuts**

    Cuts can be performed directly in the constructor by passing a dictionary. This
    has the advantage that memory is saved because the cuts are applied after reading
    each single event. This is achieved by the keyword argument :code:`filters`, which
    contains the filter dictionary. Filters are applied in the order in which they appear.
    Let's assume we only want to keep pions in events with a
    multiplicity > 500:

        >>> jetscape = Jetscape(JETSCAPE_FILE_PATH, filters={'multiplicity_cut':500, 'particle_species':(211, -211, 111)}})
        >>>
        >>> # print the pions to a jetscape file
        >>> jetscape.print_particle_lists_to_file('./particle_lists.dat')

    Notes
    -----
    All filters with the keyword argument :code:`filters` need the usual
    parameters for the filter functions in the dictionary.
    All filter functions without arguments need a :code:`True` in the dictionary.
    """

    def __init__(
        self,
        JETSCAPE_FILE: str,
        **kwargs: Dict[
            str,
            Union[int, Tuple[int, int], Dict[str, Union[int, Tuple[int, int]]]],
        ],
    ):
        super().__init__(JETSCAPE_FILE, **kwargs)
        if not isinstance(self.loader_, JetscapeLoader):
            raise TypeError("The loader must be an instance of JetscapeLoader.")
        self.sigmaGen_: Tuple[float, float] = self.loader_.get_sigmaGen()
        self.particle_type_: str = self.loader_.get_particle_type()
        self.JETSCAPE_FILE: str = JETSCAPE_FILE
        self.particle_type_defining_string_: str = (
            self.loader_.get_particle_type_defining_string()
        )
        self.last_line_: str = self.loader_.get_last_line(JETSCAPE_FILE)
        del self.loader_

    def create_loader(
        self, JETSCAPE_FILE: Union[str, List[List[Particle]]]
    ) -> None:
        """
        Creates a new JetscapeLoader object.

        This method initializes a new JetscapeLoader object with the specified JETSCAPE file
        and assigns it to the loader attribute.

        Parameters
        ----------
        JETSCAPE_FILE : Union[str, List[List[Particle]]]
            The path to the JETSCAPE file to be loaded. Must be a string.

        Raises
        ------
        TypeError
            If JETSCAPE_FILE is not a string.

        Returns
        -------
        None
        """
        if not isinstance(JETSCAPE_FILE, str):
            raise TypeError("The JETSCAPE_FILE must be a path.")
        self.loader_ = JetscapeLoader(JETSCAPE_FILE)

    # PRIVATE CLASS METHODS
    def _particle_as_list(self, particle: Particle) -> List[Union[int, float]]:
        particle_list: List[Union[int, float]] = [0.0] * 7
        particle_list[0] = int(particle.ID)
        particle_list[1] = int(particle.pdg)
        particle_list[2] = int(particle.status)
        particle_list[3] = float(particle.E)
        particle_list[4] = float(particle.px)
        particle_list[5] = float(particle.py)
        particle_list[6] = float(particle.pz)

        return particle_list
    
    def _update_after_merge(self, other: BaseStorer) -> None:
        """
        Updates the current instance after merging with another Jetscape instance.

        This method is called after merging two Jetscape instances to update the
        attributes of the current instance based on the attributes of the other instance.
        The last line and filename are taken from the left-hand instance.
        :code:`sigmaGen` is averaged.

        Parameters
        ----------
        other : Jetscape
            The other Jetscape instance that was merged with the current instance.

        Raises
        ------
        UserWarning
            If the Jetscape :code:`particle_type_` or :code:`particle_type_defining_string_` of the two instances do not match, a warning is issued.
        """
        if not isinstance(other, Jetscape):
            raise TypeError("Can only add Jetscape objects to Jetscape.")
        if self.particle_type_ != other.particle_type_:
            raise TypeError("particle_types of the merged instances do not match.")
        if self.particle_type_defining_string_ != other.particle_type_defining_string_:
            raise TypeError("particle_type_defining_string of the merged instances do not match.")
        
        self.sigmaGen_ = ((self.sigmaGen_[0] + other.sigmaGen_[0])/2.0, 0.5*np.sqrt(self.sigmaGen_[1]**2 + other.sigmaGen_[1]**2))
    
    # PUBLIC CLASS METHODS
    def participants(self) -> "Jetscape":
        """
        Raises an error because participants are not defined for Jetscape 
        events.

        Returns
        -------
        NotImplementedError
            Always, because participants are not defined for Jetscape events.
        """
        raise NotImplementedError(
            "Participants are not defined for Jetscape events."
        )

    def spectators(self) -> "Jetscape":
        """
        Raises an error because spectators are not defined for Jetscape 
        events.

        Returns
        -------
        NotImplementedError
            Always, because spectators are not defined for Jetscape events.
        """
        raise NotImplementedError(
            "Spectators are not defined for Jetscape events."
        )

    def spacetime_cut(
        self, dim: str, cut_value_tuple: Tuple[float, float]
    ) -> "Jetscape":
        """
        Raises an error because spacetime cuts are not possible for Jetscape 
        events.

        Parameters
        ----------
        dim : str
            The dimension to apply the cut.
        cut_value_tuple : tuple
            The values to apply the cut.

        Raises
        ------
        NotImplementedError
            Always, because spacetime cuts are not possible for Jetscape events.
        """
        raise NotImplementedError(
            "Spacetime cuts are not possible for Jetscape events."
        )

    def spacetime_rapidity_cut(
        self, cut_value: Union[float, Tuple[float, float]]
    ) -> "Jetscape":
        """
        Raises an error because spacetime rapidity cuts are not possible for 
        Jetscape events.

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

        Raises
        ------
        NotImplementedError
            Always, because spacetime rapidity cuts are not possible for 
            Jetscape events.
        """
        raise NotImplementedError(
            "Spacetime rapidity cuts are not possible for Jetscape events."
        )

    def get_sigmaGen(self) -> Tuple[float, float]:
        """
        Returns the value of sigmaGen and the uncertainty in a tuple.

        Returns
        -------
        float
            The value of sigmaGen.
        """
        return self.sigmaGen_

    def print_particle_lists_to_file(self, output_file: str) -> None:
        """
        Prints the current Jetscape data to an output file specified by
        :code:`output_file` with the same format as the input file.
        For empty events, only the event header is printed.

        Parameters
        ----------
        output_file : str
            Path to the output file like :code:`[output_directory]/particle_lists.dat`

        """
        with open(self.JETSCAPE_FILE, "r") as jetscape_file:
            header_file = jetscape_file.readline()
        if self.particle_list_ is None:
            raise ValueError("The particle list is empty.")
        if self.num_output_per_event_ is None:
            raise ValueError("The number of output per event is empty.")
        if self.num_events_ is None:
            raise ValueError("The number of events is empty.")

        # Open the output file with buffered writing (25 MB)
        with open(output_file, "w", buffering=25 * 1024 * 1024) as f_out:
            f_out.write(header_file)

            list_of_particles = self.particle_list()
            if self.num_events_ > 1:
                for i in range(self.num_events_):
                    event = self.num_output_per_event_[i, 0]
                    num_out = self.num_output_per_event_[i, 1]
                    particle_output = np.asarray(list_of_particles[i])

                    # Header for each event
                    header = f"#\tEvent\t{event}\tweight\t1\tEPangle\t0\t{self.particle_type_defining_string_}\t{num_out}\n"
                    f_out.write(header)

                    # Write the particle data to the file
                    if particle_output.shape[0] != 0:
                        np.savetxt(
                            f_out, particle_output, fmt="%d %d %d %g %g %g %g"
                        )
            else:
                event = 0
                num_out = self.num_output_per_event_[0][1]
                particle_output = np.asarray(list_of_particles)

                # Header for each event
                header = f"#\tEvent\t{event}\tweight\t1\tEPangle\t0\t{self.particle_type_defining_string_}\t{num_out}\n"
                f_out.write(header)

                # Write the particle data to the file
                if particle_output.shape[0] != 0:
                    np.savetxt(
                        f_out, particle_output, fmt="%d %d %d %g %g %g %g"
                    )

            # Write the last line
            last_line = self.last_line_ + "\n"
            f_out.write(last_line)
