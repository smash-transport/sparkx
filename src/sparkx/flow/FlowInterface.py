#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
from abc import ABC, abstractmethod

class FlowInterface(ABC):
    @abstractmethod
    def __init__(self,*args, **kwargs):
        pass

    @abstractmethod
    def integrated_flow(self, particle_data, *args, **kwargs):
        pass

    @abstractmethod
    def differential_flow(self, particle_data, bins, flow_as_function_of, *args, **kwargs):
        pass