import numpy as np
import warnings

class Flow:
    '''
    flow_type:
        "EventPlane", "ScalarProduct", "QCumulants", "LeeYangZeros"
    '''
    def __init__(self,flow_type=None,n=None):
        if flow_type == None:
            warnings.warn('No flow type selected, set QCumulants as default')
            self.flow_type = "QCumulants"
        elif not isinstance(flow_type, str):
            raise TypeError('flow_type has to be a string')
        elif flow_type not in ["EventPlane", "ScalarProduct", "QCumulants", "LeeYangZeros"]:
            raise ValueError("Invalid flow_type given, choose one of the following flow_types: 'EventPlane', 'ScalarProduct', 'QCumulants', 'LeeYangZeros'")
        else:
            self.flow_type = flow_type

        if n = None:
            warnings.warn("No 'n' for the flow harmonic given, default set to 2")
            self.n = 2
        elif not isinstance(flow_type, int):
            raise TypeError('n has to be int')
        elif n <= 0:
            raise ValueError('n-th harmonic with value n<=0 can not be computed')
        else:
            self.n = n

        self.integrated_flow = 0.
        self.differential_flow = []

    def __integrated_flow_event_plane(self):
        return 0
    
    def __integrated_flow_scalar_product(self):
        return 0
    
    def __integrated_flow_Q_cumulants(self):
        return 0
    
    def __integrated_flow_LeeYang_zeros(self):
        return 0

    def integrated_flow(self,particle_data):

        return self.integrated_flow

    def __differential_flow_event_plane(self):
        return 0
    
    def __differential_flow_scalar_product(self):
        return 0
    
    def __differential_flow_Q_cumulants(self):
        return 0
    
    def __differential_flow_LeeYang_zeros(self):
        return 0

    def differential_flow(self,particle_data):
        return self.differential_flow

        

