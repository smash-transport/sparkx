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
from sparkx.loader.OscarLoader import OscarLoader
from sparkx.BaseStorer import BaseStorer
from typing import Any, List, Optional, Union


class Oscar(BaseStorer):
    """
        Defines an Oscar object.

        The Oscar class contains a single .oscar file including all or only chosen
        events in either the Oscar2013 or Oscar2013Extended format. It's methods
        allow to directly act on all contained events as applying acceptance filters
        (e.g. un/charged particles, spectators/participants) to keep/remove particles
        by their PDG codes or to apply cuts (e.g. multiplicity, pseudo/rapidity, pT).
        Once these filters are applied, the new data set can be accessed as a

        1) nested list containing all quantities of the Oscar format
        2) list containing Particle objects from the Particle class

        or it may be printed to a file complying with the input format.

        .. note::
            If filters are applied, be aware that not all cuts commute.

        Parameters
        ----------
        OSCAR_FILE : str
            Path to Oscar file

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
                  - From the input Oscar file load only a single event by |br|
                    specifying :code:`events=i` where i is event number i.
                * - :code:`events` (tuple)
                  - From the input Oscar file load only a range of events |br|
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
        PATH_OSCAR_ : str
            Path to the Oscar file
        oscar_format_ : str
            Input Oscar format "Oscar2013" or "Oscar2013Extended" (set automatically)
        num_output_per_event_ : numpy.array
            Array containing the event number and the number of particles in this
            event as num_output_per_event_[event i][num_output in event i] (updated
            when filters are applied)
        num_events_ : int
            Number of events contained in the Oscar object (updated when filters
            are applied)
        event_end_lines_ : list
            List containing all comment lines at the end of each event as str.
            Needed to print the Oscar object to a file.


        Methods
        -------
        oscar_format:
            Get Oscar format of the input files
        print_particle_lists_to_file:
            Print current particle data to file with same format

        Examples
        --------

        **1. Initialization**

        To create an Oscar object, the path to the Oscar file has to be passed.
        By default the Oscar object will contain all events of the input file. If
        the Oscar object should only contain certain events, the keyword argument
        "events" must be used.

        .. highlight:: python
        .. code-block:: python
            :linenos:

            >>> from sparkx.Oscar import Oscar
            >>>
            >>> OSCAR_FILE_PATH = [Oscar_directory]/particle_lists.oscar
            >>>
            >>> # Oscar object containing all events
            >>> oscar1 = Oscar(OSCAR_FILE_PATH)
            >>>
            >>> # Oscar object containing only the first event
            >>> oscar2 = Oscar(OSCAR_FILE_PATH, events=0)
            >>>
            >>> # Oscar object containing only events 2, 3, 4 and 5
            >>> oscar3 = Oscar(OSCAR_FILE_PATH, events=(2,5))

        **2. Method Usage**

        All methods that apply filters to the Oscar data return :code:`self`. This
        means that methods can be concatenated. To access the Oscar data as list to
        store it into a variable, the method :code:`particle_list()` or
        :code:`particle_objects_list` must be called in the end.
        Let's assume we only want to keep participant pions in events with a
        multiplicity > 500:

            >>> oscar = Oscar(OSCAR_FILE_PATH)
            >>>
            >>> pions = oscar.multiplicity_cut(500).participants().particle_species((211, -211, 111))
            >>>
            >>> # save the pions of all events as nested list
            >>> pions_list = pions.particle_list()
            >>>
            >>> # save the pions as list of Particle objects
            >>> pions_particle_objects = pions.particle_objects_list()
            >>>
            >>> # print the pions to an oscar file
            >>> pions.print_particle_lists_to_file('./particle_lists.oscar')

        **3. Constructor cuts**

        Cuts can be performed directly in the constructor by passing a dictionary. This
        has the advantage that memory is saved because the cuts are applied after reading
        each single event. This is achieved by the keyword argument :code:`filters`, which
        contains the filter dictionary. Filters are applied in the order in which they appear.
        Let's assume we only want to keep participant pions in events with a
        multiplicity > 500:

            >>> oscar = Oscar(OSCAR_FILE_PATH, filters={'multiplicity_cut':500, 'participants':True, 'particle_species':(211, -211, 111)})
            >>>
            >>> # print the pions to an oscar file
            >>> oscar.print_particle_lists_to_file('./particle_lists.oscar')

        Notes
        -----
        If the :code:`filters` keyword with the :code:`spacetime_cut` is used, then a list
    <<<<<<< HEAD
        specifying the dimension to be cut in the first entry and the tuple with the cut
    =======
        specifying the dimensiAR_FILE, **kwargs):
            super().__init__(OSCAR_FILE,**kwargs)
            self.PATH_OSCAR_ = OSCAR_FILE
            self.oscar_format_=self.loader_.oscar_format()
            self.event_end_lines_ = self.loader_.event_end_lines()
            del self.loader_

        def create_loader(self, OSCAR_FILE):
            self.loader_= OscarLoader(OSCAR_FILE)on to be cut in the first entry and the tuple with the cut
    >>>>>>> 14df26b (Formatting)
        boundaries in the second entry is needed. For all other filters, the dictionary
        only needs the filter name as key and the filter argument as value.
        All filter functions without arguments need a :code:`True` in the dictionary.

    """

    def __init__(self, OSCAR_FILE: str, **kwargs: Any) -> None:
        super().__init__(OSCAR_FILE, **kwargs)
        self.PATH_OSCAR_: str = OSCAR_FILE
        if not isinstance(self.loader_, OscarLoader):
            raise TypeError("The loader must be an instance of OscarLoader.")
        self.oscar_format_: Union[str | None] = self.loader_.oscar_format()
        self.event_end_lines_: List[str] = self.loader_.event_end_lines()
        del self.loader_

    def create_loader(self, OSCAR_FILE: str) -> None:  # type: ignore[override]
        self.loader_ = OscarLoader(OSCAR_FILE)

    def _particle_as_list(self, particle: Any) -> List[Union[float, int]]:
        particle_list: List[Union[float, int]] = []
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

        if (
            self.oscar_format_ == "Oscar2013Extended"
            or self.oscar_format_ == "Oscar2013Extended_IC"
            or self.oscar_format_ == "Oscar2013Extended_Photons"
        ):
            particle_list.append(int(particle.ncoll))
            particle_list.append(float(particle.form_time))
            particle_list.append(float(particle.xsecfac))
            particle_list.append(int(particle.proc_id_origin))
            particle_list.append(int(particle.proc_type_origin))
            particle_list.append(float(particle.t_last_coll))
            particle_list.append(int(particle.pdg_mother1))
            particle_list.append(int(particle.pdg_mother2))
            if self.oscar_format_ != "Oscar2013Extended_Photons":
                if not np.isnan(particle.baryon_number):
                    particle_list.append(int(particle.baryon_number))
                if not np.isnan(particle.strangeness):
                    particle_list.append(int(particle.strangeness))
            else:
                if not np.isnan(particle.weight):
                    particle_list.append(int(particle.weight))

        elif (
            self.oscar_format_ != "Oscar2013"
            and self.oscar_format_ != "Oscar2013Extended"
            and self.oscar_format_ != "Oscar2013Extended_IC"
            and self.oscar_format_ != "Oscar2013Extended_Photons"
        ):
            raise TypeError(
                "Input file not in OSCAR2013, OSCAR2013Extended or Oscar2013Extended_IC format"
            )

        return particle_list

    def oscar_format(self) -> Union[str, None]:
        """
        Get the Oscar format of the input file.

        Returns
        -------
        oscar_format_ : str
            Oscar format of the input Oscar file as string ("Oscar2013" or
            "Oscar2013Extended")

        """
        return self.oscar_format_

    def print_particle_lists_to_file(self, output_file: str) -> None:
        """
        Prints the current Oscar data to an output file specified by
        :code:`output_file` with the same format as the input file.
        For empty events, only the event header and footer are printed.

        Parameters
        ----------
        output_file : str
            Path to the output file like :code:`[output_directory]/particle_lists.oscar`

        """
        header: List[str] = []
        format_oscar2013: str = "%g %g %g %g %g %.9g %.9g %.9g %.9g %d %d %d"
        format_oscar2013_extended: str = "%g %g %g %g %g %.9g %.9g %.9g %.9g %d %d %d %d %g %g %d %d %g %d %d"

        with open(self.PATH_OSCAR_, "r") as oscar_file:
            counter_line = 0
            while True:
                line = oscar_file.readline()
                line_splitted = line.replace("\n", "").split(" ")

                if counter_line < 3:
                    header.append(line)
                elif line_splitted[0] == "#" and line_splitted[3] == "end":
                    event_footer = line
                    break
                elif counter_line > 1000000:
                    err_msg = (
                        "Unable to find the end of an event in the original"
                        + "Oscar file within the first 1000000 lines"
                    )
                    raise RuntimeError(err_msg)
                counter_line += 1

        with open(output_file, "w") as f_out:
            for i in range(3):
                f_out.write(header[i])

        # Open the output file with buffered writing (25 MB)
        with open(output_file, "a", buffering=25 * 1024 * 1024) as f_out:
            if self.particle_list_ is None:
                raise ValueError("The particle list is empty.")
            list_of_particles = self.particle_list()
            if self.num_output_per_event_ is None:
                raise ValueError("The number of output per event is empty.")
            if self.num_events_ is None:
                raise ValueError("The number of events is empty.")
            if self.num_events_ > 1:
                for i in range(self.num_events_):
                    event = self.num_output_per_event_[i, 0]
                    num_out = self.num_output_per_event_[i, 1]
                    particle_output = np.asarray(list_of_particles[i])
                    f_out.write(
                        "# event " + str(event) + " out " + str(num_out) + "\n"
                    )
                    if len(particle_output) == 0:
                        f_out.write(self.event_end_lines_[event])
                        continue
                    elif (
                        i == 0
                        and len(particle_output[0]) > 20
                        and self.oscar_format_ == "Oscar2013Extended"
                    ):
                        format_oscar2013_extended = (
                            format_oscar2013_extended
                            + (len(particle_output[0]) - 20) * " %d"
                        )
                    if self.oscar_format_ == "Oscar2013":
                        np.savetxt(
                            f_out,
                            particle_output,
                            delimiter=" ",
                            newline="\n",
                            fmt=format_oscar2013,
                        )
                    elif (
                        self.oscar_format_ == "Oscar2013Extended"
                        or self.oscar_format_ == "Oscar2013Extended_IC"
                        or self.oscar_format_ == "Oscar2013Extended_Photons"
                    ):
                        np.savetxt(
                            f_out,
                            particle_output,
                            delimiter=" ",
                            newline="\n",
                            fmt=format_oscar2013_extended,
                        )
                    f_out.write(self.event_end_lines_[event])
            else:
                event = 0
                num_out = self.num_output_per_event_[0][1]
                particle_output = np.asarray(list_of_particles)
                f_out.write(
                    "# event " + str(event) + " out " + str(num_out) + "\n"
                )
                if len(particle_output) == 0:
                    f_out.write(self.event_end_lines_[event])
                    f_out.close()
                    return
                elif (
                    len(particle_output[0]) > 20
                    and self.oscar_format_ == "Oscar2013Extended"
                ):
                    format_oscar2013_extended = (
                        format_oscar2013_extended
                        + (len(particle_output[0]) - 20) * " %d"
                    )
                if self.oscar_format_ == "Oscar2013":
                    np.savetxt(
                        f_out,
                        particle_output,
                        delimiter=" ",
                        newline="\n",
                        fmt=format_oscar2013,
                    )
                elif (
                    self.oscar_format_ == "Oscar2013Extended"
                    or self.oscar_format_ == "Oscar2013Extended_IC"
                    or self.oscar_format_ == "Oscar2013Extended_Photons"
                ):
                    np.savetxt(
                        f_out,
                        particle_output,
                        delimiter=" ",
                        newline="\n",
                        fmt=format_oscar2013_extended,
                    )
                f_out.write(self.event_end_lines_[event])
        f_out.close()
