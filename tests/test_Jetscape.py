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
import random
import numpy as np
import filecmp
from sparkx.Jetscape import Jetscape
from sparkx.Particle import Particle

@pytest.fixture
def output_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'test_output.dat')

@pytest.fixture
def jetscape_file_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'test_jetscape.dat')

def create_temporary_jetscape_file(path, num_events, output_per_event_list=None):
    """
    This function creates a sample Jetscape file "jetscape_test.dat" in the 
    temporary directory, containing data for the specified number of events.

    Parameters:
    - tmp_path: The temporary directory path where the file will be created.
    - num_events: The number of events to generate in the file.
    - output_per_event_list: An optional list specifying the number of outputs per event. If provided, it must have the same length as num_events.

    Returns:
    - str: The path to the created OSCAR file as a string.
    """
    # Validate output_per_event_list
    if output_per_event_list is not None:
        if not isinstance(output_per_event_list, list):
            raise TypeError("output_per_event_list must be a list")
        if len(output_per_event_list) != num_events:
            raise ValueError("output_per_event_list must have the same length as num_events")

    # Define the header content
    header = "#	JETSCAPE_FINAL_STATE	v2	|	N	pid	status	E	Px	Py	Pz\n"
    data = (0, 111, 27, 1.08566, 0.385059, 0.292645, 0.962134)

    # Construct the file path
    jetscape_file = path / "jetscape_test.dat"

    # Open the file for writing
    with jetscape_file.open("w") as f:
        # Write the header
        f.write(header)

        # Loop through the specified number of events
        for event_number in range(num_events):
            # Write starting line for the event
            if output_per_event_list is None:
                num_outputs = random.randint(10, 20)
            else:
                num_outputs = output_per_event_list[event_number]

            event_info = f"#	Event	{event_number}	weight	1	EPangle	0	N_hadrons	{num_outputs}\n"
            f.write(event_info)

            # Write particle data line with white space separation
            particle_line = ' '.join(map(str, data)) + '\n'

            # Write particle data lines
            for _ in range(num_outputs):
                f.write(particle_line)

        # Write ending comment line
        ending_comment_line = f"#	sigmaGen	0.000314633	sigmaErr	6.06164e-07\n"
        f.write(ending_comment_line)

    return str(jetscape_file)

def test_constructor_invalid_initialization(jetscape_file_path):
     # Initialization with invalid input file
    invalid_input_file = "./test_files/not_existing_file"
    with pytest.raises(FileNotFoundError):
        Jetscape(invalid_input_file)

    # Initalization with invalid kwargs: events not a number
    with pytest.raises(TypeError):
        Jetscape(jetscape_file_path, events=np.nan)
        
    with pytest.raises(TypeError):
        Jetscape(jetscape_file_path, events=("a", "b"))
        
    # Initalization with invalid kwargs: events negative
    with pytest.raises(ValueError):
        Jetscape(jetscape_file_path, events=-1)
        
    with pytest.raises(ValueError):
        Jetscape(jetscape_file_path, events=(-4, -1))
        
    # Initalization with invalid kwargs: events out of boundaries
    with pytest.raises(IndexError):
        Jetscape(jetscape_file_path, events=5)
        
    with pytest.raises(IndexError):
        Jetscape(jetscape_file_path, events=(5, 10))

def test_Jetscape_initialization(jetscape_file_path):
    jetscape = Jetscape(jetscape_file_path)
    assert jetscape is not None

def test_num_output_per_event(jetscape_file_path):
    num_output_jetscape = [[1, 25],[2, 41],[3, 46],[4, 43],[5, 45]]
    
    jetscape = Jetscape(jetscape_file_path)
    assert (jetscape.num_output_per_event() == num_output_jetscape).all()

def test_num_events(jetscape_file_path):
    jetscape = Jetscape(jetscape_file_path)
    assert jetscape.num_events() == 5

def test_particle_list(jetscape_file_path):
    dummy_particle_list = [[0, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [1, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [2, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [3, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [4, 111, 27, 0.138, 0.0, 0.0, 0.0], 
                           [5, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [6, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [7, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [8, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [9, 111, 27, 0.138, 0.0, 0.0, 0.0]]
    
    dummy_particle_list_nested = [[[0, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [1, 111, 27, 0.138, 0.0, 0.0, 0.0]],
                           [[2, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [3, 111, 27, 0.138, 0.0, 0.0, 0.0]],
                           [[4, 111, 27, 0.138, 0.0, 0.0, 0.0], 
                           [5, 111, 27, 0.138, 0.0, 0.0, 0.0]],
                           [[6, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [7, 111, 27, 0.138, 0.0, 0.0, 0.0]],
                           [[8, 111, 27, 0.138, 0.0, 0.0, 0.0],
                           [9, 111, 27, 0.138, 0.0, 0.0, 0.0]]]
    
    particle_objects = []
    for particle_data in dummy_particle_list:
        particle_objects.append(Particle("JETSCAPE", particle_data))
        
    # Reshape particle objects list such that we have 5 events with 
    # 2 particle objects each and turn it back into a python list
    particle_objects = (np.array(particle_objects).reshape((5, 2))).tolist()

    jetscape = Jetscape(jetscape_file_path)
    jetscape.particle_list_ = particle_objects
    jetscape.num_events_ = 5
    jetscape.num_output_per_event_ = np.array([[0,2],[1,2],[2,2],[3,2],[4,2]])
    
    assert (jetscape.particle_list() == dummy_particle_list_nested)

def test_Jetscape_print(jetscape_file_path, output_path):
    jetscape = Jetscape(jetscape_file_path)
    jetscape.print_particle_lists_to_file(output_path)
    assert filecmp.cmp(jetscape_file_path, output_path)
    os.remove(output_path)

def test_Jetscape_get_sigmaGen(jetscape_file_path):
    jetscape = Jetscape(jetscape_file_path)
    assert jetscape.get_sigmaGen() == (0.000314633,6.06164e-07)
    