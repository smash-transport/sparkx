# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import numpy as np
from sparkx.Particle import Particle
import warnings
from typing import Tuple, List, Union, Optional


def __ensure_tuple_is_valid_else_raise_error(
    value_tuple: Tuple[
        Optional[Union[int, float]], Optional[Union[float, int]]
    ],
    allow_none: bool = False,
) -> None:
    """
    Validates a tuple for specific conditions.

    This function checks if the input is a tuple of length two, where both
    elements are either numbers or None. If allow_none is set to False, the
    function raises an error if any of the elements is None. If allow_none is
    set to True, the function raises an error if both elements are None.

    Parameters
    ----------
    value_tuple : tuple
        The tuple to be validated. Expected to be of length two.
    allow_none : bool, optional
        Determines whether None values are allowed in the tuple. Default is False.

    Raises
    ------
    TypeError
        If the input value is not a tuple or if it's not of length two.
    ValueError
        If non-numeric value is found in the tuple, or if None values are not
        allowed and found, or if both elements are None when allow_none is True.
    """
    if not isinstance(value_tuple, tuple) or len(value_tuple) != 2:
        raise TypeError("Input value must be a tuple of length two")

    elif any(
        val is not None and not isinstance(val, (int, float))
        for val in value_tuple
    ):
        raise ValueError("Non-numeric value found in given tuple")

    elif (value_tuple[0] is not None and value_tuple[1] is not None) and (
        value_tuple[0] >= value_tuple[1]
    ):
        warn_msg = (
            "Lower limit {} is greater than upper limit {}. "
            "Switched order is assumed in the following."
        ).format(value_tuple[0], value_tuple[1])
        warnings.warn(warn_msg)

    elif not allow_none:
        if value_tuple[0] is None or value_tuple[1] is None:
            raise ValueError("At least one value in the tuple is None")

    elif allow_none:
        if value_tuple[0] is None and value_tuple[1] is None:
            raise ValueError("At least one cut limit must be set to a number")


def charged_particles(
    particle_list: List[List[Particle]],
) -> List[List[Particle]]:
    """
    Keep only charged particles in :code:`particle_list`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if (elem.charge != 0 and not np.isnan(elem.charge))
        ]
    return particle_list


def uncharged_particles(
    particle_list: List[List[Particle]],
) -> List[List[Particle]]:
    """
    Keep only uncharged particles in :code:`particle_list`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if (elem.charge == 0 and not np.isnan(elem.charge))
        ]
    return particle_list


def particle_species(
    particle_list: List[List[Particle]],
    pdg_list: Union[int, Tuple[int, ...], List[int], np.ndarray],
) -> List[List[Particle]]:
    """
    Keep only particle species given by their PDG ID in every event.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    pdg_list : int
        To keep a single particle species only, pass a single PDG ID

    pdg_list : tuple/list/array
        To keep multiple particle species, pass a tuple or list or array
        of PDG IDs

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(
        pdg_list, (str, int, list, np.integer, np.ndarray, tuple, float)
    ):
        raise TypeError(
            "Input value for pgd codes has not one of the "
            + "following types: str, int, float, np.integer, list "
            + "of str, list of int, list of int np.ndarray, tuple"
        )

    if isinstance(pdg_list, float):
        if np.isnan(pdg_list):
            raise ValueError("Input value for PDG code is NaN")

    elif isinstance(pdg_list, (list, np.ndarray, tuple)):
        if np.isnan(pdg_list).any():
            raise ValueError("Input value for PDG codes contains NaN values")

    if isinstance(pdg_list, (int, float, str, np.integer)):
        pdg_list = int(pdg_list)
        for i in range(0, len(particle_list)):
            particle_list[i] = [
                elem
                for elem in particle_list[i]
                if (int(elem.pdg) == pdg_list and not np.isnan(elem.pdg))
            ]

    elif isinstance(pdg_list, (list, np.ndarray, tuple)):
        pdg_list = np.asarray(pdg_list, dtype=np.int64)

        for i in range(0, len(particle_list)):
            particle_list[i] = [
                elem
                for elem in particle_list[i]
                if (int(elem.pdg) in pdg_list and not np.isnan(elem.pdg))
            ]

    else:
        raise TypeError(
            "Input value for pgd codes has not one of the "
            + "following types: str, int, float, np.integer, list "
            + "of str, list of int, list of float, np.ndarray, tuple"
        )
    return particle_list


def remove_particle_species(
    particle_list: List[List[Particle]],
    pdg_list: Union[int, Tuple[int, ...], List[int], np.ndarray],
) -> List[List[Particle]]:
    """
    Remove particle species from :code:`particle_list` by their PDG ID in every
    event.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    pdg_list : int
        To remove a single particle species only, pass a single PDG ID

    pdg_list : tuple/list/array
        To remove multiple particle species, pass a tuple or list or array
        of PDG IDs

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(
        pdg_list, (str, int, float, list, np.integer, np.ndarray, tuple)
    ):
        raise TypeError(
            "Input value for pgd codes has not one of the "
            + "following types: str, int, float, np.integer, list "
            + "of str, list of int, list of float, np.ndarray, tuple"
        )

    if isinstance(pdg_list, float):
        if np.isnan(pdg_list):
            raise ValueError("Input value for PDG code is NaN")

    elif isinstance(pdg_list, (list, np.ndarray, tuple)):
        if np.isnan(pdg_list).any():
            raise ValueError("Input value for PDG codes contains NaN values")

    if isinstance(pdg_list, (int, float, str, np.integer)):
        pdg_list = int(pdg_list)
        for i in range(0, len(particle_list)):
            particle_list[i] = [
                elem
                for elem in particle_list[i]
                if (int(elem.pdg) != pdg_list and not np.isnan(elem.pdg))
            ]

    elif isinstance(pdg_list, (list, np.ndarray, tuple)):
        pdg_list = np.asarray(pdg_list, dtype=np.int64)

        for i in range(0, len(particle_list)):
            particle_list[i] = [
                elem
                for elem in particle_list[i]
                if (int(elem.pdg) not in pdg_list and not np.isnan(elem.pdg))
            ]

    else:
        raise TypeError(
            "Input value for pgd codes has not one of the "
            + "following types: str, int, float, np.integer, list "
            + "of str, list of int, list of float, np.ndarray, tuple"
        )
    return particle_list


def participants(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only participants in :code:`particle_list`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if (elem.ncoll != 0 and not np.isnan(elem.ncoll))
        ]

    return particle_list


def spectators(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only spectators in :code:`particle_list`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if (elem.ncoll == 0 and not np.isnan(elem.ncoll))
        ]

    return particle_list


def lower_event_energy_cut(
    particle_list: List[List[Particle]], minimum_event_energy: Union[int, float]
) -> List[List[Particle]]:
    """
    Filters out events with total energy lower than a threshold.
    Events with smaller energies are removed and not kept as empty events.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    minimum_event_energy : int or float
        The minimum event energy threshold. Should be a positive integer or
        float.

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event

    Raises
    ------
    TypeError
        If the :code:`minimum_event_energy` parameter is not an integer or float.
    ValueError
        If the :code:`minimum_event_energy` parameter is less than or equal to 0.
    ValueError
        If the :code:`minimum_event_energy` parameter is :code:`NaN`.
    """
    if not isinstance(minimum_event_energy, (int, float)):
        raise TypeError(
            "Input value for lower event energy cut has not "
            + "one of the following types: int, float"
        )
    if np.isnan(minimum_event_energy):
        raise ValueError("Input value should not be NaN")
    if minimum_event_energy <= 0.0:
        raise ValueError("The lower event energy cut value should be positive")

    updated_particle_list = []
    for event_particles in particle_list:
        total_energy = sum(
            particle.E
            for particle in event_particles
            if not np.isnan(particle.E)
        )
        if total_energy >= minimum_event_energy:
            updated_particle_list.append(event_particles)
    particle_list = updated_particle_list

    if len(particle_list) == 0:
        particle_list = [[]]

    return particle_list


def spacetime_cut(
    particle_list: List[List[Particle]],
    dim: str,
    cut_value_tuple: Tuple[Optional[float], Optional[float]],
) -> List[List[Particle]]:
    """
    Apply spacetime cut to all events by passing an acceptance range by
    ::code`cut_value_tuple`. All particles outside this range will
    be removed.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

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
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(cut_value_tuple, tuple):
        raise TypeError(
            "Input value must be a tuple containing either "
            + "positive numbers or None of length two"
        )

    __ensure_tuple_is_valid_else_raise_error(cut_value_tuple, allow_none=True)

    if dim not in ("x", "y", "z", "t"):
        raise ValueError('Only "t, x, y and z are possible dimensions.')

    if cut_value_tuple[0] is None:
        lower_cut = float("-inf")
    else:
        lower_cut = cut_value_tuple[0]

    if cut_value_tuple[1] is None:
        upper_cut = float("inf")
    else:
        upper_cut = cut_value_tuple[1]

    # Ensure cut values are in the correct order
    lim_max = max(upper_cut, lower_cut)
    lim_min = min(upper_cut, lower_cut)

    updated_particle_list = []
    for i in range(0, len(particle_list)):
        if dim == "t":
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (lim_min <= elem.t <= lim_max and not np.isnan(elem.t))
            ]
        elif dim == "x":
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (lim_min <= elem.x <= lim_max and not np.isnan(elem.x))
            ]
        elif dim == "y":
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (lim_min <= elem.y <= lim_max and not np.isnan(elem.y))
            ]
        else:
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (lim_min <= elem.z <= lim_max and not np.isnan(elem.z))
            ]
        updated_particle_list.append(particle_list_tmp)

    return updated_particle_list


def pT_cut(
    particle_list: List[List[Particle]],
    cut_value_tuple: Tuple[Optional[float], Optional[float]],
) -> List[List[Particle]]:
    """
    Apply pT cut to all events by passing an acceptance range by
    ::code`cut_value_tuple`. All particles outside this range will
    be removed.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value_tuple : tuple
        Tuple with the upper and lower limits of the pT acceptance
        range :code:`(cut_min, cut_max)`. If one of the limits is not
        required, set it to :code:`None`, i.e. :code:`(None, cut_max)`
        or :code:`(cut_min, None)`.

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(cut_value_tuple, tuple):
        raise TypeError(
            "Input value must be a tuple containing either "
            + "positive numbers or None of length two"
        )

    __ensure_tuple_is_valid_else_raise_error(cut_value_tuple, allow_none=True)

    # Check if the cut limits are positive if they are not None
    if (cut_value_tuple[0] is not None and cut_value_tuple[0] < 0) or (
        cut_value_tuple[1] is not None and cut_value_tuple[1] < 0
    ):
        raise ValueError("The cut limits must be positive or None")

    # Assign numerical values to the cut limits. Even though we check for
    # non negative values, we send a left None value to -inf for numerical
    # reasons. This is still consistent with the logic of the cut, as the
    # lower cut applies to the absolute value of the pT, which is limited to
    # positive values.
    if cut_value_tuple[0] is None:
        lower_cut = float("-inf")
    else:
        lower_cut = cut_value_tuple[0]

    if cut_value_tuple[1] is None:
        upper_cut = float("inf")
    else:
        upper_cut = cut_value_tuple[1]

    # Ensure cut values are in the correct order
    lim_max = max(upper_cut, lower_cut)
    lim_min = min(upper_cut, lower_cut)

    updated_particle_list = []
    for i in range(0, len(particle_list)):
        particle_list_tmp = [
            elem
            for elem in particle_list[i]
            if (
                lim_min <= elem.pT_abs() <= lim_max
                and not np.isnan(elem.pT_abs())
            )
        ]
        updated_particle_list.append(particle_list_tmp)

    return updated_particle_list


def mT_cut(
    particle_list: List[List[Particle]],
    cut_value_tuple: Tuple[Optional[float], Optional[float]],
) -> List[List[Particle]]:
    """
    Apply transverse mass cut to all events by passing an acceptance range by
    ::code`cut_value_tuple`. All particles outside this range will
    be removed.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value_tuple : tuple
        Tuple with the upper and lower limits of the mT acceptance
        range :code:`(cut_min, cut_max)`. If one of the limits is not
        required, set it to :code:`None`, i.e. :code:`(None, cut_max)`
        or :code:`(cut_min, None)`.

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(cut_value_tuple, tuple):
        raise TypeError(
            "Input value must be a tuple containing either "
            + "positive numbers or None of length two"
        )

    __ensure_tuple_is_valid_else_raise_error(cut_value_tuple, allow_none=True)

    # Check if the cut limits are positive if they are not None
    if (cut_value_tuple[0] is not None and cut_value_tuple[0] < 0) or (
        cut_value_tuple[1] is not None and cut_value_tuple[1] < 0
    ):
        raise ValueError("The cut limits must be positive or None")

    # Assign numerical values to the cut limits. Even though we check for
    # non negative values, we send a left None value to -inf for numerical
    # reasons. This is still consistent with the logic of the cut, as the
    # lower cut applies to the absolute value of the pT, which is limited to
    # positive values.
    if cut_value_tuple[0] is None:
        lower_cut = float("-inf")
    else:
        lower_cut = cut_value_tuple[0]

    if cut_value_tuple[1] is None:
        upper_cut = float("inf")
    else:
        upper_cut = cut_value_tuple[1]

    # Ensure cut values are in the correct order
    lim_max = max(upper_cut, lower_cut)
    lim_min = min(upper_cut, lower_cut)

    updated_particle_list = []
    for i in range(0, len(particle_list)):
        particle_list_tmp = [
            elem
            for elem in particle_list[i]
            if (lim_min <= elem.mT() <= lim_max and not np.isnan(elem.mT()))
        ]
        updated_particle_list.append(particle_list_tmp)

    return updated_particle_list


def rapidity_cut(
    particle_list: List[List[Particle]],
    cut_value: Union[int, float, Tuple[float, float]],
) -> List[List[Particle]]:
    """
    Apply rapidity cut to all events and remove all particles with rapidity
    not complying with cut_value.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value : float
        If a single value is passed, the cut is applied symmetrically
        around 0.
        For example, if :code:`cut_value = 1`, only particles with rapidity in
        [-1.0, 1.0] are kept.

    cut_value : tuple
        To specify an asymmetric acceptance range for the rapidity
        of particles, pass a tuple :code:`(cut_min, cut_max)`

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if isinstance(cut_value, tuple):
        __ensure_tuple_is_valid_else_raise_error(cut_value, allow_none=False)

        # Ensure cut values are in the correct order
        lim_max = max(cut_value[0], cut_value[1])
        lim_min = min(cut_value[0], cut_value[1])

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (
                    lim_min <= elem.rapidity() <= lim_max
                    and not np.isnan(elem.rapidity())
                )
            ]
            updated_particle_list.append(particle_list_tmp)

    elif isinstance(cut_value, (int, float)):
        # cut symmetrically around 0
        limit = np.abs(cut_value)

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (
                    -limit <= elem.rapidity() <= limit
                    and not np.isnan(elem.rapidity())
                )
            ]
            updated_particle_list.append(particle_list_tmp)

    else:
        raise TypeError(
            "Input value must be a number or a tuple "
            + "with the cut limits (cut_min, cut_max)"
        )

    return updated_particle_list


def pseudorapidity_cut(
    particle_list: List[List[Particle]],
    cut_value: Union[int, float, Tuple[float, float]],
) -> List[List[Particle]]:
    """
    Apply pseudo-rapidity cut to all events and remove all particles with
    pseudo-rapidity not complying with :code:`cut_value`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value : float
        If a single value is passed, the cut is applied symmetrically
        around 0.
        For example, if :code:`cut_value = 1`, only particles with
        pseudo-rapidity in [-1.0, 1.0] are kept.

    cut_value : tuple
        To specify an asymmetric acceptance range for the pseudo-rapidity
        of particles, pass a tuple :code:`(cut_min, cut_max)`

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if isinstance(cut_value, tuple):
        __ensure_tuple_is_valid_else_raise_error(cut_value, allow_none=False)

        # Ensure cut values are in the correct order
        lim_max = max(cut_value[0], cut_value[1])
        lim_min = min(cut_value[0], cut_value[1])

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (
                    lim_min <= elem.pseudorapidity() <= lim_max
                    and not np.isnan(elem.pseudorapidity())
                )
            ]
            updated_particle_list.append(particle_list_tmp)

    elif isinstance(cut_value, (int, float)):
        # cut symmetrically around 0
        limit = np.abs(cut_value)

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (
                    -limit <= elem.pseudorapidity() <= limit
                    and not np.isnan(elem.pseudorapidity())
                )
            ]
            updated_particle_list.append(particle_list_tmp)

    else:
        raise TypeError(
            "Input value must be a number or a tuple "
            + "with the cut limits (cut_min, cut_max)"
        )

    return updated_particle_list


def spacetime_rapidity_cut(
    particle_list: List[List[Particle]],
    cut_value: Union[int, float, Tuple[float, float]],
) -> List[List[Particle]]:
    """
    Apply space-time rapidity cut to all events and remove all particles with
    space-time rapidity not complying with :code:`cut_value`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value : float
        If a single value is passed, the cut is applied symmetrically
        around 0.
        For example, if :code:`cut_value = 1`, only particles with spacetime
        rapidity in [-1.0, 1.0] are kept.

    cut_value : tuple
        To specify an asymmetric acceptance range for the space-time rapidity
        of particles, pass a tuple :code:`(cut_min, cut_max)`

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if isinstance(cut_value, tuple):
        __ensure_tuple_is_valid_else_raise_error(cut_value, allow_none=False)

        # Ensure cut values are in the correct order
        lim_max = max(cut_value[0], cut_value[1])
        lim_min = min(cut_value[0], cut_value[1])

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (
                    lim_min <= elem.spacetime_rapidity() <= lim_max
                    and not np.isnan(elem.spacetime_rapidity())
                )
            ]
            updated_particle_list.append(particle_list_tmp)

    elif isinstance(cut_value, (int, float)):
        # cut symmetrically around 0
        limit = np.abs(cut_value)

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (
                    -limit <= elem.spacetime_rapidity() <= limit
                    and not np.isnan(elem.spacetime_rapidity())
                )
            ]
            updated_particle_list.append(particle_list_tmp)

    else:
        raise TypeError(
            "Input value must be a number or a tuple "
            + "with the cut limits (cut_min, cut_max)"
        )

    return updated_particle_list


def multiplicity_cut(
    particle_list: List[List[Particle]], cut_value_tuple: tuple
) -> List[List[Particle]]:
    """
    Apply multiplicity cut. Remove all events with a multiplicity not complying
    with cut_value_tuple.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    cut_value_tuple : tuple
        Upper and lower bound for multiplicity. If the multiplicity of an event is
        not in this range, the event is discarded. The range is inclusive on the
        lower bound and exclusive on the upper bound.

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(cut_value_tuple, tuple):
        raise TypeError(
            "Input value must be a tuple containing either "
            + "positive numbers or None of length two"
        )

    __ensure_tuple_is_valid_else_raise_error(cut_value_tuple, allow_none=True)

    # Check if the cut limits are positive if they are not None
    if (cut_value_tuple[0] is not None and cut_value_tuple[0] < 0) or (
        cut_value_tuple[1] is not None and cut_value_tuple[1] < 0
    ):
        raise ValueError("The cut limits must be positive or None")

    if cut_value_tuple[0] is None:
        lower_cut = float("-inf")
    else:
        lower_cut = cut_value_tuple[0]

    if cut_value_tuple[1] is None:
        upper_cut = float("inf")
    else:
        upper_cut = cut_value_tuple[1]

    # Ensure cut values are in the correct order
    lim_max = max(upper_cut, lower_cut)
    lim_min = min(upper_cut, lower_cut)

    idx_keep_event = []
    for idx, event_particles in enumerate(particle_list):
        multiplicity = len(event_particles)
        if multiplicity >= lim_min and multiplicity < lim_max:
            idx_keep_event.append(idx)

    particle_list = [particle_list[idx] for idx in idx_keep_event]
    if len(particle_list) == 0:
        particle_list = [[]]
    return particle_list


def particle_status(
    particle_list: List[List[Particle]],
    status_list: Union[int, np.ndarray, List[int], Tuple[int, ...]],
) -> List[List[Particle]]:
    """
    Keep only particles with a given particle status.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    status_list : int
        To keep a particles with a single status only, pass a single status

    status_list : tuple/list/array
        To keep hadrons with different hadron status, pass a tuple or list
        or array

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    if not isinstance(status_list, (int, list, tuple, np.ndarray)):
        raise TypeError(
            "Input for status codes must be int, or  "
            + "list/tuple/array of int values"
        )
    if isinstance(status_list, (list, tuple, np.ndarray)):
        if any(not isinstance(val, int) for val in status_list):
            raise TypeError("status_list contains non int value")

    if isinstance(status_list, int):
        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (elem.status == status_list and not np.isnan(elem.status))
            ]
            updated_particle_list.append(particle_list_tmp)

    elif isinstance(status_list, (list, np.ndarray, tuple)):
        status_list = np.asarray(status_list, dtype=np.int64)

        updated_particle_list = []
        for i in range(0, len(particle_list)):
            particle_list_tmp = [
                elem
                for elem in particle_list[i]
                if (elem.status in status_list and not np.isnan(elem.status))
            ]
        updated_particle_list.append(particle_list_tmp)

    particle_list = updated_particle_list

    return particle_list


def keep_hadrons(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only hadrons in :code:`particle_list`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.is_hadron() and not np.isnan(elem.is_hadron())
        ]
    return particle_list


def keep_leptons(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only leptons in :code:`particle_list`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.is_lepton() and not np.isnan(elem.is_lepton())
        ]
    return particle_list


def keep_quarks(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only quarks in particle_list. This function can be used to filter all
    gluons out if the input is a parton list.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.is_quark() and not np.isnan(elem.is_quark())
        ]
    return particle_list


def keep_mesons(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only mesons in :code:`particle_list`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.is_meson() and not np.isnan(elem.is_meson())
        ]
    return particle_list


def keep_baryons(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only baryons in :code:`particle_list`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.is_baryon() and not np.isnan(elem.is_baryon())
        ]
    return particle_list


def keep_up(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only hadrons containing up quarks in :code:`particle_list`.
    This filter does not work for partons.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.has_up() and not np.isnan(elem.has_up())
        ]
    return particle_list


def keep_down(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only hadrons containing down quarks in :code:`particle_list`.
    This filter does not work for partons.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.has_down() and not np.isnan(elem.has_down())
        ]
    return particle_list


def keep_strange(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only hadrons containing strange quarks in :code:`particle_list`.
    This filter does not work for partons.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.has_strange() and not np.isnan(elem.has_strange())
        ]
    return particle_list


def keep_charm(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only hadrons containing charm quarks in :code:`particle_list`.
    This filter does not work for partons.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.has_charm() and not np.isnan(elem.has_charm())
        ]
    return particle_list


def keep_bottom(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only hadrons containing bottom quarks in :code:`particle_list`.
    This filter does not work for partons.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.has_bottom() and not np.isnan(elem.has_bottom())
        ]
    return particle_list


def keep_top(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Keep only hadrons containing top quarks in :code:`particle_list`.
    This filter does not work for partons.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if elem.has_top() and not np.isnan(elem.has_top())
        ]
    return particle_list


def remove_photons(particle_list: List[List[Particle]]) -> List[List[Particle]]:
    """
    Remove all photons from :code:`particle_list`.

    Parameters
    ----------
    particle_list:
        List with lists containing particle objects for the events

    Returns
    -------
    list of lists
        Filtered list of lists containing particle objects for each event
    """
    for i in range(0, len(particle_list)):
        particle_list[i] = [
            elem
            for elem in particle_list[i]
            if not np.isnan(elem.pdg) and int(elem.pdg) != 22
        ]
    return particle_list
