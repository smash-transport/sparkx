# ===================================================
#
#    Copyright (c) 2024-2025
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from sparkx.Histogram import Histogram
from sparkx.Particle import Particle
from typing import List, Tuple, Union, Optional, Iterator

import warnings


# This class ensures a list to be read-only
class ReadOnlyList:
    def __init__(self, nested_list: List[List[Particle]]) -> None:
        self._nested_list = nested_list  # Store reference to the original list

    def __getitem__(self, index: int) -> List[Particle]:
        return self._nested_list[index]  # Allow read access to list items

    def __len__(self) -> int:
        return len(self._nested_list)  # Allow getting the length of the list

    def __iter__(self) -> Iterator[List[Particle]]:
        return iter(self._nested_list)  # Allow iteration over the list

    def __repr__(self) -> str:
        return repr(self._nested_list)  # Allow printing the list

    # Prevent modification by raising exceptions
    def __setitem__(self, index: int, value: List[Particle]) -> None:
        raise TypeError("This list is read-only.")

    def append(self, value: List[Particle]) -> None:
        raise TypeError("This list is read-only.")

    def extend(self, values: List[List[Particle]]) -> None:
        raise TypeError("This list is read-only.")

    def insert(self, index: int, value: List[Particle]) -> None:
        raise TypeError("This list is read-only.")

    def remove(self, value: List[Particle]) -> None:
        raise TypeError("This list is read-only.")

    def pop(self, index: int = -1) -> None:
        raise TypeError("This list is read-only.")

    def clear(self) -> None:
        raise TypeError("This list is read-only.")


class BulkObservables:
    """
    Class to calculate bulk observables from a list of Particle objects. It is
    assumed that all necessary cuts were performed to the particle list before.

    Attributes
    ----------
    particle_objects: ReadOnlyList
        A read-only list of lists of Particle objects.

    Methods
    -------
    dNdy:
        Calculate the event averaged yield :math:`\\frac{dN}{dy}`.
    dNdpT:
        Calculate the event averaged yield :math:`\\frac{dN}{dp_T}`.
    dNdEta:
        Calculate the event averaged yield :math:`\\frac{dN}{d\\eta}`
    dNdmT:
        Calculate the event averaged yield :math:`\\frac{dN}{dm_T}`.
    mid_rapidity_yield:
        Calculate the event-averaged particle yield at mid-rapidity.
    mid_rapidity_mean_pT:
        Calculate the event-averaged mean transverse momentum :math:`p_T` at mid-rapidity.
    mid_rapidity_mean_mT:
        Calculate the event-averaged mean transverse mass :math:`m_T` at mid-rapidity.

    Examples
    --------

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.BulkObservables import BulkObservables

        >>> # Initialize the BulkObservables class
        >>> bulk_observables = BulkObservables(particle_objects_list)

        >>> # Calculate dN/dy
        >>> histogram_dNdy = bulk_observables.dNdy()

        >>> # Calculate dN/dpT
        >>> histogram_dNdpT = bulk_observables.dNdpT()

        >>> # Calculate dN/dη
        >>> histogram_dNdEta = bulk_observables.dNdEta()

        >>> # Calculate dN/dmT
        >>> histogram_dNdmT = bulk_observables.dNdmT()

        >>> # Calculate mid-rapidity yield
        >>> mid_rapidity_yield = bulk_observables.mid_rapidity_yield()
        >>> print(mid_rapidity_yield)

        >>> # Calculate mid-rapidity mean pT
        >>> mid_rapidity_mean_pT = bulk_observables.mid_rapidity_mean_pT()
        >>> print(mid_rapidity_mean_pT)

        >>> # Calculate mid-rapidity mean mT
        >>> mid_rapidity_mean_mT = bulk_observables.mid_rapidity_mean_mT()
        >>> print(mid_rapidity_mean_mT)

    """

    def __init__(self, particle_objects_list: List[List[Particle]]) -> None:
        # Wrapping in ReadOnlyList prevents particles from being modified
        self.particle_objects = ReadOnlyList(particle_objects_list)

    # PRIVATE CLASS METHODS
    """
    This is the base method for all differential yields. It is called by all 
    corresponding methods for each specific differential yield
    """

    def _differential_yield(
        self,
        quantity: str,
        bin_properties: Union[
            Tuple[Union[int, float], Union[int, float], int],
            List[Union[int, float]],
        ],
    ) -> Histogram:
        if not isinstance(quantity, str):
            raise TypeError("quantity must be of type str")
        elif not isinstance(bin_properties, (tuple, list)):
            raise TypeError("bin_properties must be of type tuple or list")
        # Check if bin_properties is a tuple
        if isinstance(bin_properties, tuple):
            if (
                len(bin_properties) != 3
                or not isinstance(bin_properties[0], (int, float))
                or not isinstance(bin_properties[1], (int, float))
                or not isinstance(bin_properties[2], int)
            ):
                raise ValueError(
                    "If bin_properties is a tuple, it must be of the form (int/float, int/float, int)."
                )

        # Check if bin_properties is a list
        elif isinstance(bin_properties, list):
            if not all(isinstance(x, (int, float)) for x in bin_properties):
                raise ValueError(
                    "If bin_properties is a list, all elements must be of type int or float"
                )

        hist = Histogram(bin_properties)
        num_events = len(self.particle_objects)
        inverse_bin_width = 1 / hist.bin_width()

        # Fill histograms
        for event in range(num_events):
            for particle in self.particle_objects[event]:
                """
                Call the corresponding method in Particle given by
                'quantity' as string
                """
                particle_method = getattr(particle, quantity)
                if callable(particle_method):
                    hist.add_value(particle_method())
                else:
                    raise AttributeError(
                        f"'{quantity}' is not a callable method of Particle"
                    )

            # Do not add an empty histogram at the end
            if num_events > 1 and not event == (num_events - 1):
                hist.add_histogram()

        hist.average()
        hist.scale_histogram(inverse_bin_width)
        return hist

    # PUBLIC CLASS METHODS
    def dNdy(
        self,
        bin_properties: Optional[
            Union[
                Tuple[Union[int, float], Union[int, float], int],
                List[Union[int, float]],
            ]
        ] = None,
    ) -> Histogram:
        """
        Calculate the event averaged yield :math:`\\frac{dN}{dy}`

        Parameters
        ----------
        bin_properties: tuple, list
          Optional tuple (start, stop, num) for histogram binning.
          If not given, a default will be used

        Returns
        -------
        Histogram
          1D histogram containing the event averaged particle counts per
          rapidity bin.
        """
        if bin_properties is None:
            return self._differential_yield("rapidity", (-2, 2, 11))
        else:
            return self._differential_yield("rapidity", bin_properties)

    def dNdpT(
        self,
        bin_properties: Optional[
            Union[
                Tuple[Union[int, float], Union[int, float], int],
                List[Union[int, float]],
            ]
        ] = None,
    ) -> Histogram:
        """
        Calculate the event averaged yield :math:`\\frac{dN}{dp_T}`

        Parameters
        ----------
        bin_properties: tuple, list
          Optional tuple (start, stop, num) for histogram binning.
          If not given, a default will be used

        Returns
        -------
        Histogram
          1D histogram containing the event averaged particle counts per
          transverse momentum bin.
        """

        if isinstance(bin_properties, tuple) and (
            bin_properties[0] < 0 or bin_properties[1] < 0
        ):
            warn_msg = "Bins must be positive for dNdpT! All negative bins will be empty."
            warnings.warn(warn_msg)

        elif isinstance(bin_properties, list) and any(
            bin_edge < 0 for bin_edge in bin_properties
        ):
            warn_msg = "Bins must be positive for dNdpT! All negative bins will be empty."
            warnings.warn(warn_msg)

        if bin_properties is None:
            return self._differential_yield("pT_abs", (0, 4, 11))
        else:
            return self._differential_yield("pT_abs", bin_properties)

    def dNdEta(
        self,
        bin_properties: Optional[
            Union[
                Tuple[Union[int, float], Union[int, float], int],
                List[Union[int, float]],
            ]
        ] = None,
    ) -> Histogram:
        """
        Calculate the event averaged yield :math:`\\frac{dN}{d\\eta}`

        Parameters
        ----------
        bin_properties: tuple, list
          Optional tuple (start, stop, num) for histogram binning.
          If not given, a default will be used

        Returns
        -------
        Histogram
          1D histogram containing the event averaged particle counts per
          pseudo-rapidity bin.
        """
        if bin_properties is None:
            return self._differential_yield("pseudorapidity", (-2, 2, 11))
        else:
            return self._differential_yield("pseudorapidity", bin_properties)

    def dNdmT(
        self,
        bin_properties: Optional[
            Union[
                Tuple[Union[int, float], Union[int, float], int],
                List[Union[int, float]],
            ]
        ] = None,
    ) -> Histogram:
        """
        Calculate the event averaged yield :math:`\\frac{dN}{dm_T}`

        Parameters
        ----------
        bin_properties: tuple, list
          Optional tuple (start, stop, num) for histogram binning.
          If not given, a default will be used

        Returns
        -------
        Histogram
          1D histogram containing the event averaged particle counts per
          transverse mass bin.
        """
        if isinstance(bin_properties, tuple) and (
            bin_properties[0] < 0 or bin_properties[1] < 0
        ):
            warn_msg = "Bins must be positive for dNdmT! All negative bins will be empty."
            warnings.warn(warn_msg)

        elif isinstance(bin_properties, list) and any(
            bin_edge < 0 for bin_edge in bin_properties
        ):
            warn_msg = "Bins must be positive for dNdmT! All negative bins will be empty."
            warnings.warn(warn_msg)

        if bin_properties is None:
            return self._differential_yield("mT", (0, 4, 11))
        else:
            return self._differential_yield("mT", bin_properties)

    def _compute_mid_rapidity(
        self, y_width: float, quantity: str, property_func=None
    ) -> float:
        """
        Generalized function to compute mid-rapidity yields and averages.

        Parameters
        ----------
        y_width : float
            The rapidity window width, centered at 0.
        quantity : str
            The rapidity-related method name (rapidity, pseudorapidity, etc.).
        property_func : callable, optional
            Function to apply to each selected particle.
            If None, counts particles instead.

        Returns
        -------
        float
            Event-averaged particle yield or mean property.
        """
        if not isinstance(y_width, (int, float)):
            raise TypeError("y_width must be of type int or float")

        if y_width <= 0:
            raise ValueError("y_width must be a positive number.")

        num_events = len(self.particle_objects)
        if num_events == 0:
            return 0

        particle_method = getattr(self.particle_objects[0][0], quantity)
        if not callable(particle_method):
            raise AttributeError(f"'{quantity}' is not a callable method of Particle")

        
        value = 0 
        for event in self.particle_objects:
            for particle in event:
                if -y_width / 2 <= getattr(particle, quantity)() <= y_width / 2:
                    value += property_func(particle)
        return value / num_events


    def mid_rapidity_yield(self, y_width: float = 1.0, quantity: str = "rapidity") -> float:
        return self._compute_mid_rapidity(y_width, quantity, lambda p: 1)

    def mid_rapidity_mean_pT(self, y_width: float = 1.0, quantity: str = "rapidity") -> float:
        return self._compute_mid_rapidity(y_width, quantity, lambda p: p.pT_abs())

    def mid_rapidity_mean_mT(self, y_width: float = 1.0, quantity: str = "rapidity") -> float:
        return self._compute_mid_rapidity(y_width, quantity, lambda p: p.mT())
