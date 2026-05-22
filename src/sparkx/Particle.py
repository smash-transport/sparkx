# ===================================================
#
#    Copyright (c) 2023-2026
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import array as _array
import numpy as np
import math
from particle import PDGID
import warnings
from typing import Dict, Optional, Tuple, Union, List


class Particle:
    """Defines a particle object.

    The member variables of the Particle class are the quantities in the
    Oscar2013/Oscar2013Extended/Oscar2013Extended_IC/Oscar2013Extended_Photons/
    ASCII or JETSCAPE hadron/parton output. If they are not set,
    they stay :code:`np.nan` to throw an error if one tries to access a non existing
    quantity.
    If a particle with an unknown PDG is provided, a warning is thrown and
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
    Internally, values are stored in a compact :py:class:`array.array` named
    :code:`data_` (typecode ``'d'``, i.e. C double) that **grows by one element
    each time a new field is first written**.  A companion byte array
    :code:`_idx_map` (25 entries, one per canonical slot) maps each canonical
    slot index to the actual position inside :code:`data_`.  The sentinel value
    255 means the slot has never been written; reading such a slot returns
    :code:`np.nan`.  Writing NaN to an unallocated slot is treated as a no-op
    because the slot already reads back as NaN.  Integer and bool values are
    still encoded as ``float`` so that NaN can act as the "not set" sentinel.

    When JETSCAPE creates particle objects, which are partons, the charge is multiplied
    by 3 to make it an integer.

    PDG lookups (validity, charge, spin, etc.) are cached at the class level so that
    particles sharing the same PDG code reuse previous results. The attribute mapping
    that converts input format columns to internal data indices is also pre-compiled
    once per format. Both optimizations significantly reduce loading time for large
    files.
    """

    __slots__ = ["data_", "_idx_map"]

    # ------------------------------------------------------------------
    # Class-level caches for PDG lookups.
    #
    # During file loading, thousands of particles share a small set of
    # unique PDG codes (e.g. pions, protons).  Caching avoids repeated
    # PDGID object creation, is_valid checks, and charge lookups.
    # ------------------------------------------------------------------
    _pdgid_cache: Dict[int, PDGID] = {}
    _pdg_valid_cache: Dict[int, bool] = {}
    _pdg_charge_cache: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Class-level attribute mapping and pre-compiled fast index arrays.
    #
    # The mapping is defined once and shared by all Particle instances.
    # For each format a list of (data_index, line_index, use_float)
    # tuples is pre-compiled so that __initialize_from_array can iterate
    # a flat tuple list instead of doing dict iteration + string
    # membership checks on every particle.
    # ------------------------------------------------------------------
    _FLOAT_ATTRS: frozenset = frozenset(
        [
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
        ]
    )

    _ATTRIBUTE_MAPPING: Dict[str, Dict[str, list]] = {
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
            "weight": [24, 0],
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

    # Cache of pre-compiled (data_idx, line_idx, use_float) tuples per format.
    _fast_index_cache: Dict[
        Union[str, Tuple[str, ...]], Tuple[Tuple[int, int, bool], ...]
    ] = {}

    @classmethod
    def _get_fast_indices(
        cls,
        input_format: str,
        attribute_list: Optional[List[str]] = None,
    ) -> Tuple[Tuple[int, int, bool], ...]:
        """Return pre-compiled index tuples for a given format.

        Each tuple contains ``(data_index, line_index, use_float)`` so that
        the per-particle loop can iterate directly without dictionary lookups
        or string-membership checks.

        Parameters
        ----------
        input_format : str
            The particle format name.
        attribute_list : list of str, optional
            Required only for the ``"ASCII"`` format.

        Returns
        -------
        tuple of (int, int, bool)
            Pre-compiled index tuples.
        """
        # For ASCII the mapping depends on the attribute_list, so we include
        # it in the cache key.
        cache_key: Union[str, Tuple[str, ...]]
        if input_format == "ASCII":
            cache_key = ("ASCII",) + tuple(attribute_list or [])
        else:
            cache_key = input_format

        try:
            return cls._fast_index_cache[cache_key]
        except KeyError:
            pass

        if input_format == "ASCII":
            allfields = cls._ATTRIBUTE_MAPPING["Allfields"]
            mapping = {}
            if attribute_list is None:
                raise ValueError(
                    "'attribute_list' must be provided for ASCII format"
                )
            for i, attr in enumerate(attribute_list):
                mapping[attr + "_"] = [allfields[attr][0], i]
        else:
            mapping = cls._ATTRIBUTE_MAPPING[input_format]

        float_attrs = cls._FLOAT_ATTRS
        indices = tuple(
            (data_idx, line_idx, attr in float_attrs)
            for attr, (data_idx, line_idx) in mapping.items()
        )
        cls._fast_index_cache[cache_key] = indices
        return indices

    @classmethod
    def _cached_pdgid(cls, pdg_code: int) -> PDGID:
        """Return a cached PDGID instance for the given PDG code.

        Parameters
        ----------
        pdg_code : int
            The PDG code.

        Returns
        -------
        PDGID
            The cached PDGID instance.
        """
        try:
            return cls._pdgid_cache[pdg_code]
        except KeyError:
            pdgid = PDGID(pdg_code)
            cls._pdgid_cache[pdg_code] = pdgid
            return pdgid

    @classmethod
    def _is_pdg_valid(cls, pdg_code: int) -> bool:
        """Return cached validity check for a PDG code.

        Parameters
        ----------
        pdg_code : int
            The PDG code.

        Returns
        -------
        bool
            Whether the PDG code is valid.
        """
        try:
            return cls._pdg_valid_cache[pdg_code]
        except KeyError:
            valid = cls._cached_pdgid(pdg_code).is_valid
            cls._pdg_valid_cache[pdg_code] = valid
            return valid

    @classmethod
    def _get_pdg_charge(cls, pdg_code: int) -> float:
        """Return cached charge for a PDG code.

        Parameters
        ----------
        pdg_code : int
            The PDG code.

        Returns
        -------
        float
            The electric charge, or ``np.nan`` if the PDG code is invalid.
        """
        try:
            return cls._pdg_charge_cache[pdg_code]
        except KeyError:
            if cls._is_pdg_valid(pdg_code):
                charge = float(cls._cached_pdgid(pdg_code).charge)
            else:
                charge = np.nan
            cls._pdg_charge_cache[pdg_code] = charge
            return charge

    # ------------------------------------------------------------------
    # Per-instance helpers for the grow-by-one storage layout.
    #
    # These are called by every property getter/setter.  Keeping them as
    # thin instance methods (rather than inlining the logic everywhere)
    # makes the code readable while still being fast enough for
    # interactive use.  The hot loading path in __initialize_from_array
    # inlines the same logic to avoid per-element call overhead.
    # ------------------------------------------------------------------

    def _get(self, idx: int) -> float:
        """Return the value at canonical slot *idx*, or NaN if unset."""
        pos = self._idx_map[idx]
        return self.data_[pos] if pos != 255 else float('nan')

    def _set(self, idx: int, val: float) -> None:
        """Write *val* to canonical slot *idx*, growing *data_* if needed.

        Writing NaN to an **unallocated** slot is a no-op: the slot already
        reads back as NaN via _get, so allocating a new element would waste
        memory without any observable effect.  Writing NaN to an
        **already-allocated** slot is allowed and stores NaN in place.
        """
        pos = self._idx_map[idx]
        if pos == 255:
            # Only allocate a new element for non-NaN values.
            if not math.isnan(val):
                self._idx_map[idx] = len(self.data_)
                self.data_.append(val)
        else:
            self.data_[pos] = val

    def __init__(
        self,
        input_format: Optional[str] = None,
        particle_array: Optional[np.ndarray] = None,
        attribute_list: Optional[List[str]] = None,
    ) -> None:
        # Start with an empty value array and a fully-unset index map.
        # data_   : compact float64 storage; grows by one per new field written.
        # _idx_map: 25-byte map from canonical slot index to position in data_;
        #           255 means "slot not yet written".
        self.data_: _array.array = _array.array('d')
        self._idx_map: _array.array = _array.array('B', [255] * 25)
        # pdg_valid (canonical slot 10) must be set before __initialize_from_array
        # so that its slot is pre-allocated and the setter is always safe to call.
        self.pdg_valid: bool = False

        if ((input_format is not None) and (particle_array is None)) or (
            (input_format is None) and (particle_array is not None)
        ):
            raise ValueError("'input_format' or 'particle_array' not given")

        if attribute_list is None and input_format == "ASCII":
            raise ValueError("'attribute_list' not given")

        if attribute_list is not None and input_format != "ASCII":
            raise ValueError(
                "'ASCII' format requires 'attribute_list' to be given."
            )

        if attribute_list is None:
            attribute_list = []

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
        if input_format == "ASCII":
            fast_indices = Particle._get_fast_indices(
                input_format, attribute_list
            )
            expected_len = None  # ASCII has no fixed length
        elif input_format in Particle._ATTRIBUTE_MAPPING:
            fast_indices = Particle._get_fast_indices(input_format)
            expected_len = len(Particle._ATTRIBUTE_MAPPING[input_format])
        else:
            raise ValueError(f"Unsupported input format '{input_format}'")

        # Validate particle array length
        arr_len = len(particle_array)
        if expected_len is not None and input_format != "ASCII":
            if not (
                arr_len == expected_len
                or (
                    input_format
                    in ("Oscar2013Extended", "Oscar2013Extended_IC")
                    and expected_len - 2 <= arr_len <= expected_len
                )
            ):
                raise ValueError(
                    "The input file is corrupted! "
                    + "A line with wrong number of columns "
                    + str(arr_len)
                    + " was found."
                )

        # Fill data_ using pre-compiled index tuples (data_idx, line_idx, use_float).
        # The _set logic is inlined here to avoid per-element method-call overhead
        # on the hot loading path (potentially millions of particles).
        idx_map = self._idx_map
        data = self.data_
        for data_idx, line_idx, use_float in fast_indices:
            if line_idx >= arr_len:
                continue
            val = float(particle_array[line_idx]) if use_float else float(int(particle_array[line_idx]))
            pos = idx_map[data_idx]
            if pos == 255:
                idx_map[data_idx] = len(data)
                data.append(val)
            else:
                data[pos] = val

        # Validate PDG code and cache the result.
        if np.isnan(self.pdg):
            self.pdg_valid = False
        else:
            pdg_int = int(self.pdg)
            self.pdg_valid = Particle._is_pdg_valid(pdg_int)

        if input_format == "JETSCAPE":
            self.mass = self.mass_from_energy_momentum()
            if self.pdg_valid:
                self.charge = Particle._get_pdg_charge(pdg_int)
            else:
                self.charge = np.nan
                if not np.isnan(self.pdg):
                    warnings.warn(
                        "The PDG code "
                        + str(int(self.pdg))
                        + " is not known by PDGID, charge could not be computed. Consider setting it by hand."
                    )

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

    def __copy__(self) -> "Particle":
        new = object.__new__(Particle)
        new.data_ = _array.array('d', self.data_)
        new._idx_map = _array.array('B', self._idx_map)
        return new

    def __deepcopy__(self, memo: dict) -> "Particle":
        new = object.__new__(Particle)
        new.data_ = _array.array('d', self.data_)
        new._idx_map = _array.array('B', self._idx_map)
        memo[id(self)] = new
        return new

    @property
    def t(self) -> float:
        """Get or set the time of the particle.

        Returns
        -------
        t : float
        """
        return self._get(0)

    @t.setter
    def t(self, value: float) -> None:
        self._set(0, value)

    @property
    def x(self) -> float:
        """Get or set the x-position of the particle.

        Returns
        -------
        x : float
        """
        return self._get(1)

    @x.setter
    def x(self, value: float) -> None:
        self._set(1, value)

    @property
    def y(self) -> float:
        """Get or set the y-position of the particle.

        Returns
        -------
        y : float
        """
        return self._get(2)

    @y.setter
    def y(self, value: float) -> None:
        self._set(2, value)

    @property
    def z(self) -> float:
        """Get or set the z-position of the particle.

        Returns
        -------
        z : float
        """
        return self._get(3)

    @z.setter
    def z(self, value: float) -> None:
        self._set(3, value)

    @property
    def mass(self) -> float:
        """Get or set the mass of the particle.

        Returns
        -------
        mass : float
        """
        return self._get(4)

    @mass.setter
    def mass(self, value: float) -> None:
        self._set(4, value)

    @property
    def E(self) -> float:
        """Get or set the energy of the particle.

        Returns
        -------
        E : float
        """
        return self._get(5)

    @E.setter
    def E(self, value: float) -> None:
        self._set(5, value)

    @property
    def px(self) -> float:
        """Get or set the momentum x-component of the particle.

        Returns
        -------
        px : float
        """
        return self._get(6)

    @px.setter
    def px(self, value: float) -> None:
        self._set(6, value)

    @property
    def py(self) -> float:
        """Get or set the momentum y-component of the particle.

        Returns
        -------
        py : float
        """
        return self._get(7)

    @py.setter
    def py(self, value: float) -> None:
        self._set(7, value)

    @property
    def pz(self) -> float:
        """Get or set the momentum z-component of the particle.

        Returns
        -------
        pz : float
        """
        return self._get(8)

    @pz.setter
    def pz(self, value: float) -> None:
        self._set(8, value)

    @property
    def pdg(self) -> Union[int, float]:
        """Get or set the PDG code of the particle.

        Returns
        -------
        pdg : int
        """
        val = self._get(9)
        if np.isnan(val):
            return np.nan
        return int(val)

    @pdg.setter
    def pdg(self, value: float) -> None:
        self._set(9, value)
        if np.isnan(self.pdg):
            self.pdg_valid = False
            return
        self.pdg_valid = Particle._is_pdg_valid(int(self.pdg))

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
        val = self._get(11)
        if np.isnan(val):
            return np.nan
        return int(val)

    @ID.setter
    def ID(self, value: float) -> None:
        self._set(11, value)

    @property
    def charge(self) -> Union[int, float]:
        """Get or set the electrical charge of the particle.

        Returns
        -------
        charge : int
        """
        val = self._get(12)
        if np.isnan(val):
            return np.nan
        return int(val)

    @charge.setter
    def charge(self, value: float) -> None:
        # this is for the case a parton is created from the JETSCAPE reader
        # handle quarks with 3 times the charge to make it integer
        if np.abs(value) < 1:
            value *= 3
        self._set(12, value)

    @property
    def ncoll(self) -> Union[int, float]:
        """Get or set the number of collisions of the particle.

        Returns
        -------
        ncoll : int
        """
        val = self._get(13)
        if np.isnan(val):
            return np.nan
        return int(val)

    @ncoll.setter
    def ncoll(self, value: float) -> None:
        self._set(13, value)

    @property
    def form_time(self) -> float:
        """Get or set the formation time of the particle.

        Returns
        -------
        form_time : float
        """
        return self._get(14)

    @form_time.setter
    def form_time(self, value: float) -> None:
        self._set(14, value)

    @property
    def xsecfac(self) -> float:
        """Get or set the cross section scaling factor of the particle.

        Returns
        -------
        xsecfac : float
        """
        return self._get(15)

    @xsecfac.setter
    def xsecfac(self, value: float) -> None:
        self._set(15, value)

    @property
    def proc_id_origin(self) -> Union[int, float]:
        """Get or set the process ID of the particle's origin.

        Returns
        -------
        proc_id_origin : int
        """
        val = self._get(16)
        if np.isnan(val):
            return np.nan
        return int(val)

    @proc_id_origin.setter
    def proc_id_origin(self, value: float) -> None:
        self._set(16, value)

    @property
    def proc_type_origin(self) -> Union[int, float]:
        """Get or set the process type of the particle's origin.

        Returns
        -------
        proc_type_origin : int
        """
        val = self._get(17)
        if np.isnan(val):
            return np.nan
        return int(val)

    @proc_type_origin.setter
    def proc_type_origin(self, value: float) -> None:
        self._set(17, value)

    @property
    def t_last_coll(self) -> float:
        """Get or set the last time of a collision of the particle.

        Returns
        -------
        t_last_coll : float
        """
        return self._get(18)

    @t_last_coll.setter
    def t_last_coll(self, value: float) -> None:
        self._set(18, value)

    @property
    def pdg_mother1(self) -> Union[int, float]:
        """Get the PDG code of the first mother particle.

        Returns
        -------
        pdg_mother1 : int
        """
        val = self._get(19)
        if np.isnan(val):
            return np.nan
        return int(val)

    @pdg_mother1.setter
    def pdg_mother1(self, value: float) -> None:
        self._set(19, value)

    @property
    def pdg_mother2(self) -> Union[int, float]:
        """Get the PDG code of the second mother particle.

        Returns
        -------
        pdg_mother2 : int
        """
        val = self._get(20)
        if np.isnan(val):
            return np.nan
        return int(val)

    @pdg_mother2.setter
    def pdg_mother2(self, value: float) -> None:
        self._set(20, value)

    @property
    def status(self) -> Union[int, float]:
        """
        Get the hadron status (stores information on the module origin of
        a JETSCAPE hadron).

        Returns
        -------
        status : int
        """
        val = self._get(21)
        if np.isnan(val):
            return np.nan
        return int(val)

    @status.setter
    def status(self, value: float) -> None:
        self._set(21, value)

    @property
    def baryon_number(self) -> Union[int, float]:
        """Get the baryon number of the particle.

        Returns
        -------
        baryon_number : int
        """
        val = self._get(22)
        if np.isnan(val):
            return np.nan
        return int(val)

    @baryon_number.setter
    def baryon_number(self, value: float) -> None:
        self._set(22, value)

    @property
    def strangeness(self) -> Union[int, float]:
        """Get the strangeness of the particle.

        Returns
        -------
        strangeness : int
        """
        val = self._get(23)
        if np.isnan(val):
            return np.nan
        return int(val)

    @strangeness.setter
    def strangeness(self, value: float) -> None:
        self._set(23, value)

    @property
    def weight(self) -> float:
        """Get the weight of the particle.

        Returns
        -------
        weight : float
        """
        return self._get(24)

    @weight.setter
    def weight(self, value: float) -> None:
        self._set(24, value)

    @property
    def pdg_valid(self) -> bool:
        """Get the validity of the PDG code of the particle.

        Returns
        -------
        pdg_valid : bool
        """
        # bool(float('nan')) would be True, but slot 10 is always explicitly
        # initialised to 0.0 (False) in __init__, so this never returns nan.
        return bool(self._get(10))

    @pdg_valid.setter
    def pdg_valid(self, value: bool) -> None:
        self._set(10, 1.0 if value else 0.0)

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
        This is equivalent to the above formula, but avoids numerical issues when :math:`E \\approx p_z`.
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

        Electrons (11) and positrons (-11) are assigned their PDG mass
        of 0.00051099895 GeV because their mass falls below the numerical
        precision threshold used for the energy-momentum relation.

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
        # electron/positron mass in GeV (below numerical threshold)
        electron_pdg = [11, -11]
        electron_mass = 0.00051099895
        if (
            np.isnan(self.E)
            or np.isnan(self.px)
            or np.isnan(self.py)
            or np.isnan(self.pz)
        ):
            return np.nan
        elif self.pdg in massless_pdg:
            return 0.0
        elif self.pdg in electron_pdg:
            return electron_mass
        else:
            mass_squared = self.E**2.0 - self.p_abs() ** 2.0
            if abs(mass_squared) < 1e-16:
                return 0.0
            elif mass_squared > 0:
                return np.sqrt(mass_squared)
            else:
                pdg_str = (
                    str(int(self.pdg)) if not np.isnan(self.pdg) else "nan"
                )
                warnings.warn(
                    "|E| >= |p| not fulfilled! "
                    f"PDG = {pdg_str}, "
                    f"mass_squared = {mass_squared}. "
                    "The mass is set to nan."
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
        return Particle._get_pdg_charge(int(self.pdg))

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
            if abs(mT_squared) < 1e-16:
                return 0.0
            elif mT_squared > 0:
                return np.sqrt(mT_squared)
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
        return Particle._cached_pdgid(int(self.pdg)).is_quark

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
        return Particle._cached_pdgid(int(self.pdg)).is_lepton

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
        return Particle._cached_pdgid(int(self.pdg)).is_meson

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
        return Particle._cached_pdgid(int(self.pdg)).is_baryon

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
        return Particle._cached_pdgid(int(self.pdg)).is_hadron

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
        pdgid = Particle._cached_pdgid(int(self.pdg))
        if pdgid.has_charm or pdgid.has_bottom or pdgid.has_top:
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
        return Particle._cached_pdgid(int(self.pdg)).has_down

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
        return Particle._cached_pdgid(int(self.pdg)).has_up

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
        return Particle._cached_pdgid(int(self.pdg)).has_strange

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
        return Particle._cached_pdgid(int(self.pdg)).has_charm

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
        return Particle._cached_pdgid(int(self.pdg)).has_bottom

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
        return Particle._cached_pdgid(int(self.pdg)).has_top

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
        return Particle._cached_pdgid(int(self.pdg)).J

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
        return Particle._cached_pdgid(int(self.pdg)).j_spin
