#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
from sparkx.Oscar import Oscar
from sparkx.Particle import Particle
import filecmp
import numpy as np
import pytest
import os
import random

@pytest.fixture
def output_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'test_output.oscar')

@pytest.fixture
def oscar_file_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'particle_lists.oscar')

@pytest.fixture
def oscar_extended_file_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'particle_lists_extended.oscar')

@pytest.fixture
def oscar_old_extended_file_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'particle_lists_extended_old.oscar')

def create_temporary_oscar_file(path, num_events, oscar_format, output_per_event_list=None):
    """
    This function creates a sample oscar file "particle_lists.oscar" in the temporary directory,
    containing data for the specified number of events.

    Parameters:
    - tmp_path: The temporary directory path where the OSCAR file will be created.
    - num_events: The number of events to generate in the OSCAR file.
    - output_per_event_list: An optional list specifying the number of outputs per event. If provided, it must have the same length as num_events.
    - oscar_format: The format of the OSCAR file. Can be "Oscar2013" or "Oscar2013Extended".

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
    if oscar_format == "Oscar2013Extended":
        header_lines = [
            "#!OSCAR2013Extended particle_lists t x y z mass p0 px py pz pdg ID charge ncoll form_time xsecfac proc_id_origin proc_type_origin time_last_coll pdg_mother1 pdg_mother2 baryon_number strangeness\n",
            "# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e none fm none none none fm none none none none\n",
            "# SMASH-3.1rc-23-g59a05e65f\n"
        ]
        data = (200, 1.1998, 2.4656, 66.6003, 0.938, 0.9690, -0.00624, -0.0679, 0.2335, 2112, 0, 0, 0, -5.769, 1, 0, 0, 0, 0, 0, 1, 0)
    elif oscar_format == "Oscar2013":
        header_lines = [
            "#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge\n",
            "# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e\n",
            "# SMASH-3.1rc-23-g59a05e65f\n"
        ]
        data = (200, 1.1998, 2.4656, 66.6003, 0.938, 0.9690, -0.0062, -0.0679, 0.2335, 2112, 0, 0)
    else:
        raise ValueError("Invalid value for 'oscar_format'. Allowed values are 'Oscar2013Extended' and 'Oscar2013'.")

    header = ''.join(header_lines)

    # Construct the file path
    oscar_file = path / "particle_lists.oscar"

    # Open the file for writing
    with oscar_file.open("w") as f:
        # Write the header
        f.write(header)

        # Loop through the specified number of events
        for event_number in range(num_events):
            # Write starting line for the event
            if output_per_event_list is None:
                num_outputs = random.randint(10, 20)
            else:
                num_outputs = output_per_event_list[event_number]

            event_info = f"# event {event_number} out {num_outputs}\n"
            f.write(event_info)

            # Write particle data line with white space separation
            particle_line = ' '.join(map(str, data)) + '\n'

            # Write particle data lines
            for _ in range(num_outputs):
                f.write(particle_line)

            # Write ending comment line
            ending_comment_line = f"# event {event_number} end 0 impact   0.000 scattering_projectile_target yes\n"
            f.write(ending_comment_line)

    return str(oscar_file)

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
 
def test_oscar_initialization(oscar_file_path):
    oscar = Oscar(oscar_file_path)
    assert oscar is not None

def test_oscar_extended_initialization(oscar_extended_file_path):
    oscar_extended = Oscar(oscar_extended_file_path)
    assert oscar_extended is not None
    
def test_oscar_old_extended_initialization(oscar_old_extended_file_path):
    oscar_old_extended = Oscar(oscar_old_extended_file_path)
    assert oscar_old_extended is not None
        
def test_oscar_format(oscar_file_path):
    oscar = Oscar(oscar_file_path)
    assert oscar.oscar_format() == 'Oscar2013' 
    
def test_oscar_format_extended(oscar_extended_file_path, oscar_old_extended_file_path):
    oscar_extended = Oscar(oscar_extended_file_path)
    assert oscar_extended.oscar_format() == 'Oscar2013Extended' 
    
    oscar_old_extended = Oscar(oscar_old_extended_file_path)
    assert oscar_old_extended.oscar_format() == 'Oscar2013Extended' 
    
def test_num_output_per_event(oscar_file_path, oscar_extended_file_path, 
                              oscar_old_extended_file_path):
    num_output_oscar = [[0, 32],[1, 32],[2, 32],[3, 32],[4, 32]]
    num_output_oscar_extended = [[0, 32],[1, 32],[2, 32],[3, 32],[4, 32]]
    num_output_oscar_old_extended = [[0, 4],[1, 0]]
    
    oscar = Oscar(oscar_file_path)
    assert (oscar.num_output_per_event() == num_output_oscar).all()
    
    oscar_extended = Oscar(oscar_extended_file_path)
    assert (oscar_extended.num_output_per_event() == num_output_oscar_extended).all()
    
    oscar_old_extended = Oscar(oscar_old_extended_file_path)
    assert (oscar_old_extended.num_output_per_event() == num_output_oscar_old_extended).all()
    
def test_num_events(oscar_file_path, oscar_extended_file_path, 
                    oscar_old_extended_file_path):
    oscar = Oscar(oscar_file_path)
    assert oscar.num_events() == 5
    
    oscar_extended = Oscar(oscar_extended_file_path)
    assert oscar_extended.num_events() == 5
    
    oscar_old_extended = Oscar(oscar_old_extended_file_path)
    assert oscar_old_extended.num_events() == 2
    
def test_particle_list(oscar_file_path):
    dummy_particle_list = [[200,  1.19,   2.46,  66.66, 0.938, 0.96,  -0.006, -0.067,  0.23,  2112, 0,  0],
                           [200,  0.825, -1.53,  0.0,   0.139, 0.495,  0.012, -0.059,  0.0,    321, 1,  1],
                           [150,  0.999,  3.14, -2.72,  0.511, 1.25,   0.023,  0.078, -0.22,    11, 2, -1],
                           [150,  0.123,  0.987, 4.56,  0.105, 0.302, -0.045,  0.019,  0.053,   22, 3,  0],
                           [100, -2.1,    1.2,  -0.5,   1.776, 3.0,    0.25,  -0.15,   0.1,   2212, 4,  1], 
                           [100,  0.732,  0.824, 3.14,  0.938, 2.0,    0.1,    0.05,  -0.05,  -211, 5, -1],
                           [50,  -1.5,    2.5,  -3.0,   0.511, 2.5,    0.3,   -0.2,    0.1,    211, 6,  1],
                           [50,   1.0,   -2.0,   1.5,   0.105, 0.3,    0.02,  -0.05,   0.03,    22, 7,  0],
                           [0,   -0.5,    0.0,  -1.0,   0.938, 1.0,    0.1,    0.0,   -0.1,    -13, 8, -1],
                           [0,    1.5,    0.8,   0.0,   0.511, 0.75,  -0.1,    0.05,   0.0,     22, 9,  0]]
    
    dummy_particle_list_nested = [[[200,  1.19,   2.46,  66.66, 0.938, 0.96,  -0.006, -0.067,  0.23,  2112, 0,  0],
                                   [200,  0.825, -1.53,  0.0,   0.139, 0.495,  0.012, -0.059,  0.0,    321, 1,  1]],
                                  [[150,  0.999,  3.14, -2.72,  0.511, 1.25,   0.023,  0.078, -0.22,    11, 2, -1],
                                   [150,  0.123,  0.987, 4.56,  0.105, 0.302, -0.045,  0.019,  0.053,   22, 3,  0]],
                                  [[100, -2.1,    1.2,  -0.5,   1.776, 3.0,    0.25,  -0.15,   0.1,   2212, 4,  1], 
                                   [100,  0.732,  0.824, 3.14,  0.938, 2.0,    0.1,    0.05,  -0.05,  -211, 5, -1]],
                                  [[50,  -1.5,    2.5,  -3.0,   0.511, 2.5,    0.3,   -0.2,    0.1,    211, 6,  1],
                                   [50,   1.0,   -2.0,   1.5,   0.105, 0.3,    0.02,  -0.05,   0.03,    22, 7,  0]],
                                  [[0,   -0.5,    0.0,  -1.0,   0.938, 1.0,    0.1,    0.0,   -0.1,    -13, 8, -1],
                                   [0,    1.5,    0.8,   0.0,   0.511, 0.75,  -0.1,    0.05,   0.0,     22, 9,  0]]]
    
    particle_objects = []
    for particle_data in dummy_particle_list:
        particle_objects.append(Particle("Oscar2013", particle_data))
        
    # Reshape particle objects list such that we have 5 events with 
    # 2 particle objects each and turn it back into a python list
    particle_objects = (np.array(particle_objects).reshape((5, 2))).tolist()

    oscar = Oscar(oscar_file_path)
    oscar.particle_list_ = particle_objects
    oscar.num_events_ = 5
    oscar.num_output_per_event_ = np.array([[0,2],[1,2],[2,2],[3,2],[4,2]])
    
    assert (oscar.particle_list() == dummy_particle_list_nested)
    
    
    
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
    