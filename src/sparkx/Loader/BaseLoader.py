#===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================

from abc import ABC, abstractmethod

class BaseLoader(ABC):
    
    @abstractmethod
    def __init__(self, path):
        pass

    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError("load method is not implemented")