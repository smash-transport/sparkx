# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from sparkx.flow.FlowInterface import FlowInterface

class DummyFlow(FlowInterface):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def integrated_flow(self, particle_data, *args, **kwargs):
        return "integrated_flow called"

    def differential_flow(self, particle_data, bins, flow_as_function_of, *args, **kwargs):
        return "differential_flow called"
    

def test_integrated_flow():
    flow = DummyFlow()
    result = flow.integrated_flow("particle_data")
    assert result == "integrated_flow called"

def test_differential_flow():
    flow = DummyFlow()
    result = flow.differential_flow("particle_data", "bins", "flow_as_function_of")
    assert result == "differential_flow called"