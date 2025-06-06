# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import numpy as np
import math
from particle import PDGID
import warnings
from typing import Optional, Union, List


class Particle:
    """Defines a particle object.

    The member variables of the Particle class are the quantities in the
    Oscar2013/Oscar2013Extended/Oscar2013Extended_IC/Oscar2013Extended_Photons/
    ASCII or JETSCAPE hadron/parton output. If they are not set,
    they stay :code:`np.nan` to throw an error if one tries to access a non existing
    quantity.
    If a particle with an unknown PDG is provided, a warning is thrown and and
    :code:`np.nan` is returned for charge, spin, and spin degeneracy.

    Attributes
    ----------
    t : float
        The time of the particle.
    x : float
        The x coordinate of the position.
    y : float
        The y coordinate of the position.
    z : float
        The z coordinate of the position.
    mass : float
        The mass of the particle.
    E : float
        The energy of the particle.
    px : float
        The x component of the momentum.
    py : float
        The y component of the momentum.
    pz : float
        The z component of the momentum.
    pdg : int
        The PDG code of the particle.
    pdg_valid : bool
        Is the PDG code valid?
    ID : int
        The ID of the particle (unique label of each particle in an event).
    charge : int
        Electric charge of the particle.
    ncoll : int
        Number of collisions undergone by the particle.
    form_time : double
        Formation time of the particle.
    xsecfac : double
        Scaling factor of the cross section.
    proc_id_origin : int
        ID for the process the particle stems from.
    proc_type_origin : int
        Type information for the process the particle stems from.
    t_last_coll : double
        Time of the last collision.
    pdg_mother1 : int
        PDG code of the mother particle 1.
    pdg_mother2 : int
        PDG code of the mother particle 2.
    status : int
        Status code of the particle.
    baryon_number : int
        Baryon number of the particle.
    strangeness : int
        Strangeness quantum number of the particle.
    weight : float
        Weight of the particle.


    Methods
    -------
    t:
        Get/set t
    x:
        Get/set x
    y:
        Get/set y
    z:
        Get/set z
    mass:
        Get/set mass
    E:
        Get/set E
    px:
        Get/set px
    py:
        Get/set py
    pz:
        Get/set pz
    pdg:
        Get/set pdg
    ID:
        Get/set ID
    charge:
        Get/set charge
    ncoll:
        Get/set ncoll
    form_time:
        Get/set form_time
    xsecfac:
        Get/set xsecfactor
    proc_id_origin:
        Get/set proc_id_origin
    proc_type_origin:
        Get/set proc_type_origin
    t_last_coll:
        Get/set t_last_coll
    pdg_mother1:
        Get/set pdg_mother1
    pdg_mother2:
        Get/set pdg_mother2
    status:
        Get/set status
    baryon_number:
        Get/set baryon_number
    strangeness:
        Get/set strangeness
    print_particle:
        Print the particle as CSV to terminal
    angular_momentum:
        Compute angular momentum
    rapidity:
        Compute momentum rapidity
    p_abs:
        Compute absolute momentum
    pT_abs:
        Compute absolute value of transverse momentum
    phi:
        Compute azimuthal angle
    theta:
        Compute polar angle
    pseudorapidity:
        Compute pseudorapidity
    spacetime_rapidity:
        Compute space-time rapidity
    proper_time:
        Compute proper time
    mass_from_energy_momentum:
        Compute mass from energy momentum relation
    charge_from_pdg:
        Compute charge from PDG code
    mT:
        Compute transverse mass
    is_quark:
        Is the particle a quark?
    is_lepton:
        Is the particle a lepton?
    is_meson:
        Is the particle a meson?
    is_baryon:
        Is the particle a baryon?
    is_hadron:
        Is the particle a hadron?
    is_heavy_flavor:
        Is the particle a heavy flavor particle?
    has_down:
        Does the particle have a down quark?
    has_up:
        Does the particle have an up quark?
    has_strange:
        Does the particle have a strange quark?
    has_charm:
        Does the particle have a charm quark?
    has_bottom:
        Does the particle have a bottom quark?
    has_top:
        Does the particle have a top quark?
    weight:
        What is the weight of the particle?
    spin:
        Total spin :math:`J` of the particle.
    spin_degeneracy:
        Total spin :math:`2J + 1` of the particle.


    Examples
    --------
    To use the particle class a particle has to be created and the attributes
    can be set or obtained with the corresponding functions.

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Particle import Particle
        >>>
        >>> particle = Particle()
        >>> particle.t = 1.0
        >>> print(particle.t)
        1.0

    The class can be used to construct a particle from different input formats.
    Supported formats include:

    * "Oscar2013"

    * "Oscar2013Extended"

    * "Oscar2013Extended_IC"

    * "Oscar2013Extended_Photons"

    * "ASCII"

    * "JETSCAPE"

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> particle_quantity_JETSCAPE = np.array([0,2114,11,2.01351754,1.30688601,-0.422958786,-0.512249773])
        >>> particle = Particle(input_format="JETSCAPE", particle_array=particle_quantity_JETSCAPE)

    Notes
    -----
    If a member of the Particle class is not set or a quantity should be computed
    and the needed member variables are not set, then :code:`np.nan` is returned by default.
    All quantities are saved in a numpy array member variable :code:`data_`. The datatype
    of this array is float, therefore casting is required when int or bool values are
    required.

    When JETSCAPE creates particle objects, which are partons, the charge is multiplied
    by 3 to make it an integer.
    """

    __slots__ = ["data_"]

    def __init__(
        self,
        input_format: Optional[str] = None,
        particle_array: Optional[np.ndarray] = None,
        attribute_list: List[str] = [],
    ) -> None:
        self.data_: np.ndarray = np.array(25 * [np.nan], dtype=float)
        self.pdg_valid: bool = False

        if ((input_format is not None) and (particle_array is None)) or (
            (input_format is None) and (particle_array is not None)
        ):
            raise ValueError("'input_format' or 'particle_array' not given")

        if attribute_list is None and input_format == "ASCII":
            raise ValueError("'attribute_list' not given")

        if attribute_list != [] and input_format != "ASCII":
            raise ValueError(
                "'ASCII' format requires 'attribute_list' to be given."
            )

        if (input_format is not None) and (particle_array is not None):
            self.__initialize_from_array(
                input_format, particle_array, attribute_list
            )

    def __initialize_from_array(
        self,
        input_format: str,
        particle_array: np.ndarray,
        attribute_list: List[str],
    ) -> None:
        """
        Initialize instance attributes based on the provided input format and array, and optionally an attribute list.

        Parameters
        ----------
        input_format : str
            The format of the input data. Supported formats include:
            - "Oscar2013"
            - "Oscar2013Extended"
            - "Oscar2013Extended_IC"
            - "Oscar2013Extended_Photons"
            - "ASCII"
            - "JETSCAPE"

        particle_array : numpy.ndarray
            An array containing particle information.

        attribute_list: List[str]
            A list containing coded information which element of particle_array corresponds to which attribute of the Particle instance.

        Raises
        ------
        ValueError
            If the input format is unsupported or the array length is invalid.

        Notes
        -----
        For each supported input format, this method expects a specific order
        of elements in the particle_array and assigns them to the corresponding
        attributes of the Particle instance.

        If the input format is "JETSCAPE," additional attributes (mass_ and
        charge_) are computed based on energy-momentum and PDG code.
        If the JETSCAPE class is created with parton output, then the charge is
        multiplied by 3 to make it an integer.

        """
        # first entry: index in data array
        # second entry: index in line
        attribute_mapping = {
            "Allfields": {
                "t": [0, 0],
                "x": [1, 0],
                "y": [2, 0],
                "z": [3, 0],
                "mass": [4, 0],
                "E": [5, 0],
                "px": [6, 0],
                "py": [7, 0],
                "pz_": [8, 0],
                "pdg": [9, 0],
                "ID": [11, 0],
                "charge": [12, 0],
                "ncoll": [13, 0],
                "form_time": [14, 0],
                "xsecfac": [15, 0],
                "proc_id_origin": [16, 0],
                "proc_type_origin": [17, 0],
                "t_last_coll": [18, 0],
                "pdg_mother1": [19, 0],
                "pdg_mother2": [20, 0],
                "status_": [21, 0],
                "baryon_number": [22, 0],
                "strangeness": [23, 0],
            },
            "Oscar2013": {
                "t_": [0, 0],
                "x_": [1, 1],
                "y_": [2, 2],
                "z_": [3, 3],
                "mass_": [4, 4],
                "E_": [5, 5],
                "px_": [6, 6],
                "py_": [7, 7],
                "pz_": [8, 8],
                "pdg_": [9, 9],
                "ID_": [11, 10],
                "charge_": [12, 11],
            },
            "Oscar2013Extended": {
                "t_": [0, 0],
                "x_": [1, 1],
                "y_": [2, 2],
                "z_": [3, 3],
                "mass_": [4, 4],
                "E_": [5, 5],
                "px_": [6, 6],
                "py_": [7, 7],
                "pz_": [8, 8],
                "pdg_": [9, 9],
                "ID_": [11, 10],
                "charge_": [12, 11],
                "ncoll_": [13, 12],
                "form_time_": [14, 13],
                "xsecfac_": [15, 14],
                "proc_id_origin_": [16, 15],
                "proc_type_origin_": [17, 16],
                "t_last_coll_": [18, 17],
                "pdg_mother1_": [19, 18],
                "pdg_mother2_": [20, 19],
                "baryon_number_": [22, 20],
                "strangeness_": [23, 21],
            },
            "Oscar2013Extended_IC": {
                "t_": [0, 0],
                "x_": [1, 1],
                "y_": [2, 2],
                "z_": [3, 3],
                "mass_": [4, 4],
                "E_": [5, 5],
                "px_": [6, 6],
                "py_": [7, 7],
                "pz_": [8, 8],
                "pdg_": [9, 9],
                "ID_": [11, 10],
                "charge_": [12, 11],
                "ncoll_": [13, 12],
                "form_time_": [14, 13],
                "xsecfac_": [15, 14],
                "proc_id_origin_": [16, 15],
                "proc_type_origin_": [17, 16],
                "t_last_coll_": [18, 17],
                "pdg_mother1_": [19, 18],
                "pdg_mother2_": [20, 19],
                "baryon_number_": [22, 20],
                "strangeness_": [23, 21],
            },
            "Oscar2013Extended_Photons": {
                "t_": [0, 0],
                "x_": [1, 1],
                "y_": [2, 2],
                "z_": [3, 3],
                "mass_": [4, 4],
                "E_": [5, 5],
                "px_": [6, 6],
                "py_": [7, 7],
                "pz_": [8, 8],
                "pdg_": [9, 9],
                "ID_": [11, 10],
                "charge_": [12, 11],
                "ncoll_": [13, 12],
                "form_time_": [14, 13],
                "xsecfac_": [15, 14],
                "proc_id_origin_": [16, 15],
                "proc_type_origin_": [17, 16],
                "t_last_coll_": [18, 17],
                "pdg_mother1_": [19, 18],
                "pdg_mother2_": [20, 19],
                "weight_": [24, 20],
            },
            "JETSCAPE": {
                "ID_": [11, 0],
                "pdg_": [9, 1],
                "status_": [21, 2],
                "E_": [5, 3],
                "px_": [6, 4],
                "py_": [7, 5],
                "pz_": [8, 6],
            },
        }
        if input_format == "ASCII":
            mapping_dict = {}
            for attr in attribute_list:
                mapping_dict[attr] = [
                    attribute_mapping["Allfields"][attr][0],
                    list(attribute_list).index(attr),
                ]
            attribute_mapping["ASCII"] = mapping_dict
        if input_format in attribute_mapping or input_format == "ASCII":
            if (
                input_format == "ASCII"
                or len(particle_array) == len(attribute_mapping[input_format])
                or (
                    input_format
                    in ["Oscar2013Extended", "Oscar2013Extended_IC"]
                    and len(particle_array)
                    <= len(attribute_mapping[input_format])
                    and len(particle_array)
                    >= len(attribute_mapping[input_format]) - 2
                )
            ):
                for attribute, index in attribute_mapping[input_format].items():
                    if len(particle_array) <= (index[1]):
                        continue
                    if input_format == "ASCII":
                        attribute = attribute + "_"
                    # Type casting for specific attributes. Although everything is saved as a float, we will only read in int data for int fields
                    # to ensure similar behaving as if we were reading in data
                    # into ints.
                    if attribute in [
                        "t_",
                        "x_",
                        "y_",
                        "z_",
                        "mass_",
                        "E_",
                        "px_",
                        "py_",
                        "pz_",
                        "form_time_",
                        "xsecfac_",
                        "t_last_coll_",
                        "weight_",
                    ]:
                        self.data_[index[0]] = float(particle_array[index[1]])
                    elif attribute in [
                        "pdg_",
                        "ID_",
                        "ncoll_",
                        "proc_id_origin_",
                        "proc_type_origin_",
                        "pdg_mother1_",
                        "pdg_mother2_",
                        "status_",
                    ]:
                        self.data_[index[0]] = int(particle_array[index[1]])
                    else:
                        self.data_[index[0]] = int(particle_array[index[1]])

                # It is important for JETSCAPE particles to compute pdg_valid
                # here because the charge_from_pdg function depends on
                # it.
                if np.isnan(self.pdg):
                    self.pdg_valid = False
                else:
                    self.pdg_valid = PDGID(self.pdg).is_valid

                if input_format == "JETSCAPE":
                    self.mass = self.mass_from_energy_momentum()
                    self.charge = self.charge_from_pdg()
                    if self.pdg_valid == False and np.isnan(self.charge):
                        warnings.warn(
                            "The PDG code "
                            + str(int(self.pdg))
                            + " is not known by PDGID, charge could not be computed. Consider setting it by hand."
                        )
            else:
                raise ValueError(
                    "The input file is corrupted! "
                    + "A line with wrong number of columns "
                    + str(len(particle_array))
                    + " was found."
                )
        else:
            raise ValueError(f"Unsupported input format '{input_format}'")

        if not self.pdg_valid:
            if np.isnan(self.pdg):
                warnings.warn(
                    "No PDG code given! "
                    + "All properties extracted from the PDG are set to default values."
                )
            else:
                warnings.warn(
                    "The PDG code "
                    + str(int(self.pdg))
                    + " is not valid. "
                    + "All properties extracted from the PDG are set to default values."
                )

    @property
    def t(self) -> float:
        """Get or set the time of the particle.

        Returns
        -------
        t : float
        """
        return self.data_[0]

    @t.setter
    def t(self, value: float) -> None:
        self.data_[0] = value

    @property
    def x(self) -> float:
        """Get or set the x-position of the particle.

        Returns
        -------
        x : float
        """
        return self.data_[1]

    @x.setter
    def x(self, value: float) -> None:
        self.data_[1] = value

    @property
    def y(self) -> float:
        """Get or set the y-position of the particle.

        Returns
        -------
        y : float
        """
        return self.data_[2]

    @y.setter
    def y(self, value: float) -> None:
        self.data_[2] = value

    @property
    def z(self) -> float:
        """Get or set the z-position of the particle.

        Returns
        -------
        z : float
        """
        return self.data_[3]

    @z.setter
    def z(self, value: float) -> None:
        self.data_[3] = value

    @property
    def mass(self) -> float:
        """Get or set the mass of the particle.

        Returns
        -------
        mass : float
        """
        return self.data_[4]

    @mass.setter
    def mass(self, value: float) -> None:
        self.data_[4] = value

    @property
    def E(self) -> float:
        """Get or set the energy of the particle.

        Returns
        -------
        E : float
        """
        return self.data_[5]

    @E.setter
    def E(self, value: float) -> None:
        self.data_[5] = value

    @property
    def px(self) -> float:
        """Get or set the momentum x-component of the particle.

        Returns
        -------
        px : float
        """
        return self.data_[6]

    @px.setter
    def px(self, value: float) -> None:
        self.data_[6] = value

    @property
    def py(self) -> float:
        """Get or set the momentum y-component of the particle.

        Returns
        -------
        py : float
        """
        return self.data_[7]

    @py.setter
    def py(self, value: float) -> None:
        self.data_[7] = value

    @property
    def pz(self) -> float:
        """Get or set the momentum z-component of the particle.

        Returns
        -------
        pz : float
        """
        return self.data_[8]

    @pz.setter
    def pz(self, value: float) -> None:
        self.data_[8] = value

    @property
    def pdg(self) -> Union[int, float]:
        """Get or set the PDG code of the particle.

        Returns
        -------
        pdg : int
        """
        if np.isnan(self.data_[9]):
            return np.nan
        return int(self.data_[9])

    @pdg.setter
    def pdg(self, value: float) -> None:
        self.data_[9] = value
        self.pdg_valid = PDGID(self.pdg).is_valid

        if not self.pdg_valid:
            warnings.warn(
                "The PDG code "
                + str(int(self.pdg))
                + " is not valid. "
                + "All properties extracted from the PDG are set to nan."
            )

    @property
    def ID(self) -> Union[int, float]:
        """Get or set the ID of the particle.

        This is a unique number in SMASH.

        Returns
        -------
        ID : int
        """
        if np.isnan(self.data_[11]):
            return np.nan
        return int(self.data_[11])

    @ID.setter
    def ID(self, value: float) -> None:
        self.data_[11] = value

    @property
    def charge(self) -> Union[int, float]:
        """Get or set the electrical charge of the particle.

        Returns
        -------
        charge : int
        """
        if np.isnan(self.data_[12]):
            return np.nan
        return int(self.data_[12])

    @charge.setter
    def charge(self, value: float) -> None:
        # this is for the case a parton is created from the JETSCAPE reader
        # handle quarks with 3 times the charge to make it integer
        if np.abs(value) < 1:
            value *= 3
        self.data_[12] = value

    @property
    def ncoll(self) -> Union[int, float]:
        """Get or set the number of collisions of the particle.

        Returns
        -------
        ncoll : int
        """
        if np.isnan(self.data_[13]):
            return np.nan
        return int(self.data_[13])

    @ncoll.setter
    def ncoll(self, value: float) -> None:
        self.data_[13] = value

    @property
    def form_time(self) -> float:
        """Get or set the formation time of the particle.

        Returns
        -------
        form_time : float
        """
        return self.data_[14]

    @form_time.setter
    def form_time(self, value: float) -> None:
        self.data_[14] = value

    @property
    def xsecfac(self) -> float:
        """Get or set the cross section scaling factor of the particle.

        Returns
        -------
        xsecfac : float
        """
        return self.data_[15]

    @xsecfac.setter
    def xsecfac(self, value: float) -> None:
        self.data_[15] = value

    @property
    def proc_id_origin(self) -> Union[int, float]:
        """Get or set the process ID of the particle's origin.

        Returns
        -------
        proc_id_origin : int
        """
        if np.isnan(self.data_[16]):
            return np.nan
        return int(self.data_[16])

    @proc_id_origin.setter
    def proc_id_origin(self, value: float) -> None:
        self.data_[16] = value

    @property
    def proc_type_origin(self) -> Union[int, float]:
        """Get or set the process type of the particle's origin.

        Returns
        -------
        proc_type_origin : int
        """
        if np.isnan(self.data_[17]):
            return np.nan
        return int(self.data_[17])

    @proc_type_origin.setter
    def proc_type_origin(self, value: float) -> None:
        self.data_[17] = value

    @property
    def t_last_coll(self) -> float:
        """Get or set the last time of a collision of the particle.

        Returns
        -------
        t_last_coll : float
        """
        return self.data_[18]

    @t_last_coll.setter
    def t_last_coll(self, value: float) -> None:
        self.data_[18] = value

    @property
    def pdg_mother1(self) -> Union[int, float]:
        """Get the PDG code of the first mother particle.

        Returns
        -------
        pdg_mother1 : int
        """
        if np.isnan(self.data_[19]):
            return np.nan
        return int(self.data_[19])

    @pdg_mother1.setter
    def pdg_mother1(self, value: float) -> None:
        self.data_[19] = value

    @property
    def pdg_mother2(self) -> Union[int, float]:
        """Get the PDG code of the second mother particle.

        Returns
        -------
        pdg_mother2 : int
        """
        if np.isnan(self.data_[20]):
            return np.nan
        return int(self.data_[20])

    @pdg_mother2.setter
    def pdg_mother2(self, value: float) -> None:
        self.data_[20] = value

    @property
    def status(self) -> Union[int, float]:
        """
        Get the hadron status (stores information on the module origin of
        a JETSCAPE hadron).

        Returns
        -------
        status : int
        """
        if np.isnan(self.data_[21]):
            return np.nan
        return int(self.data_[21])

    @status.setter
    def status(self, value: float) -> None:
        self.data_[21] = value

    @property
    def baryon_number(self) -> Union[int, float]:
        """Get the baryon number of the particle.

        Returns
        -------
        baryon_number : int
        """
        if np.isnan(self.data_[22]):
            return np.nan
        return int(self.data_[22])

    @baryon_number.setter
    def baryon_number(self, value: float) -> None:
        self.data_[22] = value

    @property
    def strangeness(self) -> Union[int, float]:
        """Get the strangeness of the particle.

        Returns
        -------
        strangeness : int
        """
        if np.isnan(self.data_[23]):
            return np.nan
        return int(self.data_[23])

    @strangeness.setter
    def strangeness(self, value: float) -> None:
        self.data_[23] = value

    @property
    def weight(self) -> float:
        """Get the weight of the particle.

        Returns
        -------
        weight : float
        """
        return self.data_[24]

    @weight.setter
    def weight(self, value: float) -> None:
        self.data_[24] = value

    @property
    def pdg_valid(self) -> bool:
        """Get the validity of the PDG code of the particle.

        Returns
        -------
        pdg_valid : bool
        """
        return bool(self.data_[10])

    @pdg_valid.setter
    def pdg_valid(self, value: bool) -> None:
        self.data_[10] = 1 if value else 0

    def print_particle(self) -> None:
        """Print the whole particle information as csv string.

        This function prints a header line with the different quantities.
        All particle quantities are then printed in the next line separated by
        a comma.
        """

        def int_isnan(value: float) -> Union[int, float]:
            if np.isnan(value):
                return np.nan
            else:
                return int(value)

        print(
            "t,x,y,z,mass,E,px,py,pz,pdg,ID,charge,ncoll,form_time,xsecfac,\
              proc_id_origin,proc_type_origin,t_last_coll,pdg_mother1,\
              pdg_mother2,status,baryon_number,strangeness,weight"
        )
        print(
            f"{self.t},{self.x},{self.y},{self.z},{self.mass},{self.E},\
              {self.px},{self.py},{self.pz},{int_isnan(self.pdg)},{int_isnan(self.ID)},\
              {int_isnan(self.charge)},{int_isnan(self.ncoll)},{self.form_time},{self.xsecfac},\
              {int_isnan(self.proc_id_origin)},{int_isnan(self.proc_type_origin)}\
              ,{self.t_last_coll},{int_isnan(self.pdg_mother1)},{int_isnan(self.pdg_mother2)},\
              {int_isnan(self.status)},{int_isnan(self.baryon_number)},{int_isnan(self.strangeness)},{self.weight}"
        )

    def angular_momentum(self) -> Union[np.ndarray, float]:
        """
        Compute the angular momentum :math:`\\vec{L}=\\vec{r} \\times \\vec{p}` of a particle.

        Returns
        -------
        angular_momentum : numpy.ndarray
            Array containing all three components of the
            angular momentum as :math:`[L_x, L_y, L_z]`.

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        """
        if (
            np.isnan(self.x)
            or np.isnan(self.y)
            or np.isnan(self.z)
            or np.isnan(self.px)
            or np.isnan(self.py)
            or np.isnan(self.pz)
        ):
            return np.nan
        else:
            r = [self.x, self.y, self.z]
            p = [self.px, self.py, self.pz]
            return np.cross(r, p)

    def rapidity(self) -> float:
        """
        Compute the momentum rapidity :math:`Y=\\frac{1}{2}\\ln\\left(\\frac{E+p_z}{E-p_z}\\right)` of the particle.

        Returns
        -------
        float
            momentum rapidity

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        For numerical stability, we compute the rapidity as follows: :math:`\\sinh^{-1}\\left(\\frac{p_z}{m_T}\\right)`
        This is equivalent to the above formula, but avoids numerical issues when :math:`E \approx p_z`.
        If the transverse mass :math:`m_T` is close to zero, we fall back to the original formula.
        """
        mT = self.mT()
        if np.isnan(self.pz):
            return np.nan
        elif not np.isnan(mT) and mT > 1e-16:
            # If mT is positive, we can compute rapidity
            # using the arcsinh function for numerical stability
            return np.arcsinh(self.pz / mT)
        elif not np.isnan(self.E):
            # If mT is close to zero or NaN, we fall back to the original formula
            numer = self.E + self.pz
            denom = self.E - self.pz

            if denom <= 0 or numer <= 0:
                return np.nan
            else:
                return 0.5 * np.log(numer / denom)
        else:
            # If E is also NaN, we cannot compute rapidity
            return np.nan

    def p_abs(self) -> float:
        """
        Compute the absolute momentum :math:`|\\vec{p}|=\\sqrt{p_x^2+p_y^2+p_z^2}` of the particle.

        Returns
        -------
        float
            absolute momentum

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        """
        if np.isnan(self.px) or np.isnan(self.py) or np.isnan(self.pz):
            return np.nan
        else:
            return np.sqrt(self.px**2.0 + self.py**2.0 + self.pz**2.0)

    def pT_abs(self) -> float:
        """
        Compute the absolute transverse momentum :math:`|\\vec{p}_{\\mathrm{T}}|=\\sqrt{p_x^2+p_y^2}` of the particle.

        Returns
        -------
        float
            absolute transverse momentum

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        """
        if np.isnan(self.px) or np.isnan(self.py):
            return np.nan
        else:
            return np.sqrt(self.px**2.0 + self.py**2.0)

    def phi(self) -> float:
        """
        Compute the azimuthal angle of the particle.

        Returns
        -------
        float
            azimuthal angle

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        """
        if np.isnan(self.px) or np.isnan(self.py):
            return np.nan
        else:
            if (np.abs(self.px) < 1e-6) and (np.abs(self.py) < 1e-6):
                return 0.0
            else:
                return math.atan2(self.py, self.px)

    def theta(self) -> float:
        """
        Compute the polar angle of the particle.

        Returns
        -------
        float
            polar angle

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        """
        if np.isnan(self.px) or np.isnan(self.py) or np.isnan(self.pz):
            return np.nan
        else:
            if self.p_abs() == 0.0:
                return 0.0
            else:
                return np.arccos(self.pz / self.p_abs())

    def pseudorapidity(self) -> float:
        """
        Compute the pseudorapidity :math:`\\eta=\\frac{1}{2}\\ln\\left(\\frac{|\\vec{p}|+p_z}{|\\vec{p}|-p_z}\\right)` of the particle.

        Returns
        -------
        float
            pseudorapidity

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        """
        if np.isnan(self.px) or np.isnan(self.py) or np.isnan(self.pz):
            return np.nan
        else:
            if abs(self.p_abs() - self.pz) < 1e-10:
                denominator = (
                    self.p_abs() - self.pz
                ) + 1e-10  # Adding a small positive value
            else:
                denominator = self.p_abs() - self.pz

            return 0.5 * np.log((self.p_abs() + self.pz) / denominator)

    def spacetime_rapidity(self) -> float:
        """
        Compute the space-time rapidity :math:`\\eta_s=\\frac{1}{2}\\ln\\left(\\frac{t+z}{t-z}\\right)` of the particle.

        Returns
        -------
        float
            space-time rapidity

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        """
        if np.isnan(self.t) or np.isnan(self.z):
            return np.nan
        else:
            if self.t > np.abs(self.z):
                return 0.5 * np.log((self.t + self.z) / (self.t - self.z))
            else:
                raise ValueError("|z| < t not fulfilled")

    def proper_time(self) -> float:
        """
        Compute the proper time :math:`\\tau=\\sqrt{t^2-z^2}` of the particle.

        Returns
        -------
        float
            proper time

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        """
        if np.isnan(self.t) or np.isnan(self.z):
            return np.nan
        else:
            if self.t > np.abs(self.z):
                return np.sqrt(self.t**2.0 - self.z**2.0)
            else:
                raise ValueError("|z| < t not fulfilled")

    def mass_from_energy_momentum(self) -> float:
        """
        Compute the mass from the energy momentum relation.

        This function is called automatically if a JETSCAPE file is read in.

        We consider particles with the following PDG codes as massless:
        photons (22), gluons (21), e-neutrinos (12, -12),
        mu-neutrinos (14, -14), tau-neutrinos (16, -16),
        tau-prime-neutrinos (18, -18).

        Returns
        -------
        float
            mass

        Notes
        -----
        If one of the needed particle quantities (four-momentum) is not given,
        then :code:`np.nan` is returned.
        """
        # photons and gluons are massless, consider neutrinos as massless
        massless_pdg = [22, 21, 12, -12, 14, -14, 16, -16, 18, -18]
        if (
            np.isnan(self.E)
            or np.isnan(self.px)
            or np.isnan(self.py)
            or np.isnan(self.pz)
        ):
            return np.nan
        elif self.pdg in massless_pdg:
            return 0.0
        else:
            mass_squared = self.E**2.0 - self.p_abs() ** 2.0
            if mass_squared >= 0:
                return np.sqrt(mass_squared)
            elif abs(mass_squared) < 1e-16:
                return 0.0  # numerical precision
            else:
                warnings.warn(
                    "|E| >= |p| not fulfilled! The mass is set to nan."
                )
                return np.nan

    def charge_from_pdg(self) -> float:
        """
        Compute the charge from the PDG code.

        This function is called automatically if a JETSCAPE file is read in.

        Returns
        -------
        float
            charge

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).charge

    def mT(self) -> float:
        """
        Compute the transverse mass :math:`m_{T}=\\sqrt{E^2-p_z^2}` of the particle.

        Returns
        -------
        float
            transverse mass

        Notes
        -----
        If one of the needed particle quantities is not given, then :code:`np.nan`
        is returned.
        """
        if np.isnan(self.E) or np.isnan(self.pz):
            return np.nan
        else:
            mT_squared = self.E**2.0 - self.pz**2.0
            if mT_squared >= 0:
                return np.sqrt(mT_squared)
            elif abs(mT_squared) < 1e-16:
                return 0.0  # numerical precision
            else:
                warnings.warn(
                    "|E| >= |pz| not fulfilled! "
                    "The transverse mass is set to nan."
                )
                return np.nan

    def is_quark(self) -> Union[bool, float]:
        """
        Is the particle a quark?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).is_quark

    def is_lepton(self) -> Union[bool, float]:
        """
        Is the particle a lepton?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).is_lepton

    def is_meson(self) -> Union[bool, float]:
        """
        Is the particle a meson?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).is_meson

    def is_baryon(self) -> Union[bool, float]:
        """
        Is the particle a baryon?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).is_baryon

    def is_hadron(self) -> Union[bool, float]:
        """
        Is the particle a hadron?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).is_hadron

    def is_heavy_flavor(self) -> Union[bool, float]:
        """
        Is the particle a heavy flavor hadron?

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        if (
            PDGID(self.pdg).has_charm
            or PDGID(self.pdg).has_bottom
            or PDGID(self.pdg).has_top
        ):
            return True
        else:
            return False

    def has_down(self) -> Union[bool, float]:
        """
        Does the particle contain a down quark? Does not work with partons.

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).has_down

    def has_up(self) -> Union[bool, float]:
        """
        Does the particle contain an up quark?  Does not work with partons.

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).has_up

    def has_strange(self) -> Union[bool, float]:
        """
        Does the particle contain a strange quark?  Does not work with partons.

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).has_strange

    def has_charm(self) -> Union[bool, float]:
        """
        Does the particle contain a charm quark?  Does not work with partons.

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).has_charm

    def has_bottom(self) -> Union[bool, float]:
        """
        Does the particle contain a bottom quark?  Does not work with partons.

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).has_bottom

    def has_top(self) -> Union[bool, float]:
        """
        Does the particle contain a top quark?  Does not work with partons.

        Returns
        -------
        bool
            True, False

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).has_top

    def spin(self) -> float:
        """
        Get the total spin :math:`J` of the particle.

        Returns
        -------
        float
            Total spin :math:`J`

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).J

    def spin_degeneracy(self) -> Union[int, float]:
        """
        Get the number of all possible spin projections (:math:`2J + 1`).

        Returns
        -------
        int
            Spin degeneracy :math:`2J + 1`

        Notes
        -----
        If the PDG ID is not known by `PDGID`, then :code:`np.nan` is returned.
        """
        if not self.pdg_valid:
            return np.nan
        return PDGID(self.pdg).j_spin
