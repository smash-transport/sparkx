# ===================================================
#
#    Copyright (c) 2024-2025
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List


class BaseLoader(ABC):
    """
    Abstract base class for all loader classes.

    This class provides a common interface for all loader classes.
    Loader classes are responsible for loading data from a file.

    Attributes
    ----------
    None

    Methods
    -------
    __init__(self, path):
        Abstract constructor method that must be implemented by any concrete (i.e., non-abstract) subclass.

    load(self, **kwargs):
        Abstract method intended to be used to load data from the file specified in the constructor.

    _check_that_tuple_contains_integers_only(self, events_tuple):
        Checks if all elements inside the event tuple are integers.

    _extract_integer_after_keyword(tokens, keyword):
        Extracts the integer that follows the given keyword in a tokenized
        comment line.
    """

    @abstractmethod
    def __init__(self, path: str) -> None:
        """
        Abstract constructor method.

        Parameters
        ----------
        path : str
            The path to the file to be loaded.

        """
        pass

    @abstractmethod
    def load(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Abstract method for loading data.

        Raises
        ------
        NotImplementedError
            If this method is not overridden in a concrete subclass.
        """
        raise NotImplementedError("load method is not implemented")

    def _check_that_tuple_contains_integers_only(
        self, events_tuple: Tuple[int, ...]
    ) -> None:
        """
        Checks if all elements inside the event tuple are integers.

        Parameters
        ----------
        events_tuple : tuple
            Tuple containing event boundary events for read in.

        Raises
        ------
        TypeError
            If one or more elements inside the event tuple are not integers.
        """
        if not all(isinstance(event, int) for event in events_tuple):
            raise TypeError(
                "All elements inside the event tuple must be integers."
            )

    @staticmethod
    def _extract_integer_after_keyword(tokens: List[str], keyword: str) -> int:
        """
        Extracts the integer that follows the given keyword in a tokenized comment line.

        Expected format: tokens = ["#", "keyword", "<number>", ...]

        Parameters
        ----------
        tokens : List[str]
            The tokenized line to parse (already split).
            keyword : str
            The keyword to look for.

        Returns
        -------
        int
            The integer found after the keyword.

        Raises
        ------
        ValueError
            If the keyword is not found, or if it is found but not followed by a valid integer.
        """
        if not tokens or tokens[0] != "#":
            raise ValueError(
                f"Expected first token to be '#', got: '{tokens[0] if tokens else None}'"
            )

        for i in range(len(tokens) - 1):
            if tokens[i] == keyword:
                try:
                    return int(tokens[i + 1])
                except ValueError:
                    raise ValueError(
                        f"Keyword '{keyword}' found but not followed by an integer: '{tokens[i + 1]}'"
                    )

        # Keyword not found at all
        raise ValueError(f"Keyword '{keyword}' not found in tokens: {tokens}")
