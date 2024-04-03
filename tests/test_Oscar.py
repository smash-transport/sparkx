from sparkx.Oscar import Oscar
import filecmp
import numpy as np
import pytest
import os

@pytest.fixture
def oscar_file_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'particle_lists.oscar')

@pytest.fixture
def output_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'test_output.oscar')

@pytest.fixture
def oscar_extended_file_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'particle_lists_extended.oscar')

@pytest.fixture
def oscar_old_extended_file_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'particle_lists_extended_old.oscar')


def test_constructor_invalid_initialization(oscar_file_path):
    # Initialization with invalid input file
    invalid_input_file = "./test_files/not_existing_file"
    with pytest.raises(FileNotFoundError):
        Oscar(invalid_input_file)
        
    # Initalization with invalid kwargs: events not a number
    with pytest.raises(TypeError):
        Oscar(oscar_file_path, events=np.nan)
        
    with pytest.raises(TypeError):
        Oscar(oscar_file_path, events=("a", "b"))
        
    # Initalization with invalid kwargs: events negative
    with pytest.raises(ValueError):
        Oscar(oscar_file_path, events=-1)
        
    with pytest.raises(ValueError):
        Oscar(oscar_file_path, events=(-4, -1))
        
    # Initalization with invalid kwargs: events out of boundaries
    with pytest.raises(IndexError):
        Oscar(oscar_file_path, events=5)
        
    with pytest.raises(IndexError):
        Oscar(oscar_file_path, events=(5, 10))
        
def test_extended_oscar_print(oscar_extended_file_path, output_path):
    oscar = Oscar(oscar_extended_file_path)
    oscar.print_particle_lists_to_file(output_path)
    assert filecmp.cmp(oscar_extended_file_path, output_path)
    os.remove(output_path) 

def test_old_extended_oscar_print(oscar_old_extended_file_path, output_path):
    oscar = Oscar(oscar_old_extended_file_path)
    oscar.print_particle_lists_to_file(output_path)
    assert filecmp.cmp(oscar_old_extended_file_path, output_path)
    os.remove(output_path)  

def test_standard_oscar_print(oscar_file_path, output_path):
    oscar = Oscar(oscar_file_path)
    oscar.print_particle_lists_to_file(output_path)
    assert filecmp.cmp(oscar_file_path, output_path)
    os.remove(output_path) 