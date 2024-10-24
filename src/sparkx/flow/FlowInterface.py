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
from typing import List, Union
from sparkx.Particle import Particle

class FlowInterface(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def integrated_flow(self, particle_data: List[List[Particle]], *args, 
                        **kwargs):
        pass

    @abstractmethod
    def differential_flow(
            self,
            particle_data: List[List[Particle]],
            bins: Union[np.ndarray, List[float]],
            flow_as_function_of: str,
            *args,
            **kwargs):
        pass
