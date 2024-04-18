#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
import os
import pytest
from sparkx.Jetscape import Jetscape
from sparkx.flow.EventPlaneFlow import EventPlaneFlow
from sparkx.flow.ScalarProductFlow import ScalarProductFlow
from sparkx.flow.ReactionPlaneFlow import ReactionPlaneFlow
from sparkx.flow.LeeYangZeroFlow import LeeYangZeroFlow
from sparkx.flow.QCumulantFlow import QCumulantFlow
from sparkx.flow.PCAFlow import PCAFlow

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), 'test_files')
TEST_JETSCAPE_DAT = os.path.join(TEST_FILES_DIR, 'test_jetscape.dat')

@pytest.fixture
def test_data():
    return Jetscape(TEST_JETSCAPE_DAT).particle_objects_list()

def test_EventPlaneFlow_initialization():
    with pytest.raises(TypeError):
        # Test invalid type for n
        EventPlaneFlow(n="not_an_int", weight="pt2", pseudorapidity_gap=0.1)
    
    with pytest.raises(ValueError):
        # Test n <= 0
        EventPlaneFlow(n=0, weight="pt2", pseudorapidity_gap=0.1)
    
    with pytest.raises(TypeError):
        # Test invalid type for weight
        EventPlaneFlow(n=2, weight=123, pseudorapidity_gap=0.1)
    
    with pytest.raises(ValueError):
        # Test invalid weight
        EventPlaneFlow(n=2, weight="invalid_weight", pseudorapidity_gap=0.1)
    
    with pytest.raises(TypeError):
        # Test invalid type for pseudorapidity_gap
        EventPlaneFlow(n=2, weight="pt2", pseudorapidity_gap="not_a_float")
    
    with pytest.raises(ValueError):
        # Test pseudorapidity_gap < 0
        EventPlaneFlow(n=2, weight="pt2", pseudorapidity_gap=-0.1)

def test_EventPlaneFlow_integrated_flow_errors(test_data):
    with pytest.raises(TypeError):
        # Test invalid type for self_corr
        EventPlaneFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).integrated_flow(test_data, test_data, self_corr="not_a_bool")

def test_EventPlaneFlow_differential_flow_errors(test_data):
    with pytest.raises(TypeError):
        # Test invalid type for self_corr
        EventPlaneFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).differential_flow(test_data, [0, 1, 2], "pt", test_data, self_corr="not_a_bool")

    with pytest.raises(TypeError):
        # Test invalid type for bins
        EventPlaneFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).differential_flow(test_data, "not_a_list", "pt", test_data)

    with pytest.raises(TypeError):
        # Test invalid type for flow_as_function_of
        EventPlaneFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).differential_flow(test_data, [0, 1, 2], 123, test_data)

    with pytest.raises(ValueError):
        # Test invalid value for flow_as_function_of
        EventPlaneFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).differential_flow(test_data, [0, 1, 2], "invalid_value", test_data)

def test_ScalarProductFlow_initialization():
    with pytest.raises(TypeError):
        # Test invalid type for n
        ScalarProductFlow(n="not_an_int", weight="pt2", pseudorapidity_gap=0.1)
    
    with pytest.raises(ValueError):
        # Test n <= 0
        ScalarProductFlow(n=0, weight="pt2", pseudorapidity_gap=0.1)
    
    with pytest.raises(TypeError):
        # Test invalid type for weight
        ScalarProductFlow(n=2, weight=123, pseudorapidity_gap=0.1)
    
    with pytest.raises(ValueError):
        # Test invalid weight
        ScalarProductFlow(n=2, weight="invalid_weight", pseudorapidity_gap=0.1)
    
    with pytest.raises(TypeError):
        # Test invalid type for pseudorapidity_gap
        ScalarProductFlow(n=2, weight="pt2", pseudorapidity_gap="not_a_float")
    
    with pytest.raises(ValueError):
        # Test pseudorapidity_gap < 0
        ScalarProductFlow(n=2, weight="pt2", pseudorapidity_gap=-0.1)

def test_ScalarProductFlow_integrated_flow_errors(test_data):
    with pytest.raises(TypeError):
        # Test invalid type for self_corr
        ScalarProductFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).integrated_flow(test_data, test_data, self_corr="not_a_bool")

def test_ScalarProductFlow_differential_flow_errors(test_data):
    with pytest.raises(TypeError):
        # Test invalid type for self_corr
        ScalarProductFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).differential_flow(test_data, [0, 1, 2], "pt", test_data, self_corr="not_a_bool")

    with pytest.raises(TypeError):
        # Test invalid type for bins
        ScalarProductFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).differential_flow(test_data, "not_a_list", "pt", test_data)

    with pytest.raises(TypeError):
        # Test invalid type for flow_as_function_of
        ScalarProductFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).differential_flow(test_data, [0, 1, 2], 123, test_data)

    with pytest.raises(ValueError):
        # Test invalid value for flow_as_function_of
        ScalarProductFlow(n=2, weight="pt2", pseudorapidity_gap=0.1).differential_flow(test_data, [0, 1, 2], "invalid_value", test_data)

def test_ReactionPlaneFlow_initialization():
    with pytest.raises(TypeError):
        # Test invalid type for n
        ReactionPlaneFlow(n="not_an_int")

    with pytest.raises(ValueError):
        # Test n <= 0
        ReactionPlaneFlow(n=0)

def test_ReactionPlaneFlow_differential_flow_errors(test_data):
    with pytest.raises(TypeError):
        # Test invalid type for bins
        ReactionPlaneFlow(n=2).differential_flow(test_data, "not_a_list", "pt")

    with pytest.raises(TypeError):
        # Test invalid type for flow_as_function_of
        ReactionPlaneFlow(n=2).differential_flow(test_data, [0, 1, 2], 123)

    with pytest.raises(ValueError):
        # Test invalid value for flow_as_function_of
        ReactionPlaneFlow(n=2).differential_flow(test_data, [0, 1, 2], "invalid_value")

def test_LeeYangZeroFlow_initialization():
    with pytest.raises(TypeError):
        # Test invalid type for n
        LeeYangZeroFlow(vmin=0, vmax=1, vstep=0.1, n="not_an_int")

    with pytest.raises(ValueError):
        # Test n <= 0
        LeeYangZeroFlow(vmin=0, vmax=1, vstep=0.1, n=0)

def test_LeeYangZeroFlow_differential_flow_errors(test_data):
    lyz_flow = LeeYangZeroFlow(vmin=0, vmax=1, vstep=0.1, n=2)
    # Test invalid type for flow_as_function_of
    with pytest.raises(TypeError):
        lyz_flow.differential_flow(test_data, [0, 1, 2], 123, poi_pdg=[1, 2])

    # Test invalid type for poi_pdg
    with pytest.raises(TypeError):
        lyz_flow.differential_flow(test_data, [0, 1, 2], "pt", poi_pdg="not_a_list_or_ndarray")

    # Test poi_pdg containing non-list or non-ndarray
    with pytest.raises(TypeError):
        lyz_flow.differential_flow(test_data, [0, 1, 2], "pt", poi_pdg=123)

    # Test poi_pdg elements not being integers
    with pytest.raises(TypeError):
        lyz_flow.differential_flow(test_data, [0, 1, 2], "pt", poi_pdg=[1, 2, "not_an_int"])

    # Test flow_as_function_of not in allowed values
    with pytest.raises(ValueError):
        lyz_flow.differential_flow(test_data, [0, 1, 2], "invalid_value", poi_pdg=[1, 2])

def test_QCumulant_flow_initialization():
    # Test invalid type for n
    with pytest.raises(TypeError):
        QCumulantFlow(n='not_an_int', k=2, imaginary='zero')

    # Test n with invalid value
    with pytest.raises(ValueError):
        QCumulantFlow(n=-2, k=2, imaginary='zero')

    # Test invalid type for k
    with pytest.raises(TypeError):
        QCumulantFlow(n=2, k='not_an_int', imaginary='zero')

    # Test k with invalid value
    with pytest.raises(ValueError):
        QCumulantFlow(n=2, k=3, imaginary='zero')

    # Test invalid type for imaginary
    with pytest.raises(TypeError):
        QCumulantFlow(n=2, k=2, imaginary=123)

    # Test imaginary with invalid value
    with pytest.raises(ValueError):
        QCumulantFlow(n=2, k=2, imaginary='invalid_value')

    # Test valid initialization
    qcf = QCumulantFlow(n=2, k=2, imaginary='zero')
    assert qcf.n_ == 2
    assert qcf.k_ == 2
    assert qcf.imaginary_ == 'zero'

def test_QCumulant_flow_differential_flow_errors(test_data):
    qcf = QCumulantFlow(n=2, k=2, imaginary='zero')
    
    # Test invalid type for flow_as_function_of
    with pytest.raises(TypeError):
        qcf.differential_flow(test_data, [0, 1, 2], 123, poi_pdg=[1, 2])

    # Test invalid type for poi_pdg
    with pytest.raises(TypeError):
        qcf.differential_flow(test_data, [0, 1, 2], "pt", poi_pdg="not_a_list_or_ndarray")

    # Test poi_pdg containing non-list or non-ndarray
    with pytest.raises(TypeError):
        qcf.differential_flow(test_data, [0, 1, 2], "pt", poi_pdg=123)

    # Test poi_pdg elements not being integers
    with pytest.raises(TypeError):
        qcf.differential_flow(test_data, [0, 1, 2], "pt", poi_pdg=[1, 2, "not_an_int"])

    # Test flow_as_function_of not in allowed values
    with pytest.raises(ValueError):
        qcf.differential_flow(test_data, [0, 1, 2], "invalid_value", poi_pdg=[1, 2])

    # Test k value not in allowed values
    with pytest.raises(ValueError):
        QCumulantFlow(n=2, k=3, imaginary='zero').differential_flow(test_data, [0, 1, 2], "pt", poi_pdg=[1, 2])

def test_PCAFlow_initialization():
    # Test invalid type for n
    with pytest.raises(TypeError):
        PCAFlow(n='not_an_int', alpha=2, number_subcalc=4)

    # Test n with invalid value
    with pytest.raises(ValueError):
        PCAFlow(n=-2, alpha=2, number_subcalc=4)

    # Test invalid type for alpha
    with pytest.raises(TypeError):
        PCAFlow(n=2, alpha='not_an_int', number_subcalc=4)

    # Test alpha with invalid value
    with pytest.raises(ValueError):
        PCAFlow(n=2, alpha=0, number_subcalc=4)

    # Test invalid type for number_subcalc
    with pytest.raises(TypeError):
        PCAFlow(n=2, alpha=2, number_subcalc='not_an_int')

    # Test number_subcalc with invalid value
    with pytest.raises(ValueError):
        PCAFlow(n=2, alpha=2, number_subcalc=1)

    # Test valid initialization
    pca_flow = PCAFlow(n=2, alpha=2, number_subcalc=4)
    assert pca_flow.n_ == 2
    assert pca_flow.alpha_ == 2
    assert pca_flow.number_subcalc_ == 4

def test_PCAFlow_differential_flow_errors(test_data):
    pca_flow = PCAFlow(n=2, alpha=2, number_subcalc=4)

    # Test invalid type for bins
    with pytest.raises(TypeError):
        pca_flow.differential_flow(test_data, 1, 'pt')

    # Test invalid type for flow_as_function_of
    with pytest.raises(TypeError):
        pca_flow.differential_flow(test_data, [0, 1, 2], 123)

    # Test flow_as_function_of not in allowed values
    with pytest.raises(ValueError):
        pca_flow.differential_flow(test_data, [0, 1, 2], "invalid_value")