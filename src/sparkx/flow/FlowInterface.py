# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Any, Tuple
from sparkx.Particle import Particle


class FlowInterface(ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def integrated_flow(
        self, particle_data: List[List[Particle]], *args: Any, **kwargs: Any
    ) -> Any:
        pass

    @abstractmethod
    def differential_flow(
        self,
        particle_data: List[List[Particle]],
        bins: Union[np.ndarray, List[float]],
        flow_as_function_of: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass
