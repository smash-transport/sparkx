#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
import pytest
import numpy as np
import os
from sparkx.CentralityClasses import CentralityClasses


@pytest.fixture
def centrality_obj():
    bins = [0,10,20,30,40,50,60,70,80,90,100]
    numbers_sequence = range(1,101)
    multiplicities = [num for num in numbers_sequence for _ in range(100)]
    return CentralityClasses(events_multiplicity=multiplicities,
                              centrality_bins=bins)

def test_init_with_invalid_input():
    with pytest.raises(TypeError):
        CentralityClasses(events_multiplicity=10, centrality_bins=[0, 25, 50, 75, 100])
    with pytest.raises(TypeError):
        CentralityClasses(events_multiplicity=[0,10,20,30,40,50,60,70,80,90,100], 
                          centrality_bins=0)
    
    numbers_sequence = range(1,101)
    multiplicities = [num for num in numbers_sequence for _ in range(100)]
    with pytest.raises(ValueError):
        CentralityClasses(events_multiplicity=multiplicities,centrality_bins=[0,10,20,30,40,50,60,70,80,90,100,110])
    with pytest.raises(ValueError):
        CentralityClasses(events_multiplicity=multiplicities,centrality_bins=[-10,0,10,20,30,40,50,60,70,80,90,100])
    with pytest.warns(UserWarning, match=r"'centrality_bins' contains duplicate values. They are removed automatically."):
        a = CentralityClasses(events_multiplicity=multiplicities,centrality_bins=[0,10,20,30,40,40,50,60,70,80,90,100])
        assert a.centrality_bins_ == [0,10,20,30,40,50,60,70,80,90,100]

def test_create_centrality_classes(centrality_obj): 
    # Assuming there are 10 bins, so there should be 10 minimum values
    assert len(centrality_obj.dNchdetaMin_) == 10
    assert len(centrality_obj.dNchdetaMax_) == 10
    assert len(centrality_obj.dNchdetaAvg_) == 10
    assert len(centrality_obj.dNchdetaAvgErr_) == 10

def test_get_centrality_class(centrality_obj):
    assert centrality_obj.get_centrality_class(99) == 0
    assert centrality_obj.get_centrality_class(105) == 0
    assert centrality_obj.get_centrality_class(0) == 9

    assert centrality_obj.get_centrality_class(1) == 9
    assert centrality_obj.get_centrality_class(2) == 9
    assert centrality_obj.get_centrality_class(3) == 9
    assert centrality_obj.get_centrality_class(4) == 9
    assert centrality_obj.get_centrality_class(5) == 9
    assert centrality_obj.get_centrality_class(6) == 9
    assert centrality_obj.get_centrality_class(7) == 9
    assert centrality_obj.get_centrality_class(8) == 9
    assert centrality_obj.get_centrality_class(9) == 9
    assert centrality_obj.get_centrality_class(10) == 9
    assert centrality_obj.get_centrality_class(11) == 8
    assert centrality_obj.get_centrality_class(19) == 8
    assert centrality_obj.get_centrality_class(20) == 8
    assert centrality_obj.get_centrality_class(21) == 7
    assert centrality_obj.get_centrality_class(29) == 7

def test_output_centrality_classes(centrality_obj, tmp_path):
    output_file = os.path.join(tmp_path,"centrality_output.txt")
    centrality_obj.output_centrality_classes(str(output_file))
    assert os.path.isfile(output_file)

    # Check content of the output file
    with open(output_file, 'r') as f:
        lines = f.readlines()
        assert lines[0].startswith("# CentralityMin CentralityMax")
        assert len(lines) == 10

def test_create_centrality_classes_error():
    with pytest.raises(ValueError):
        CentralityClasses(events_multiplicity=[1, 2, 3], centrality_bins=[0, 25, 50, 75, 100])

    with pytest.raises(ValueError):
        CentralityClasses(events_multiplicity=[10, 15, -20, 25], centrality_bins=[0, 25, 50, 75, 100])

def test_output_centrality_classes_with_invalid_fname(centrality_obj):
    with pytest.raises(TypeError):
        centrality_obj.output_centrality_classes(123)