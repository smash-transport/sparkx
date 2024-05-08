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
import copy
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

            event_info = f"#	Event	{event_number+1}	weight	1	EPangle	0	N_hadrons	{num_outputs}\n"
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

def test_loading_defined_events_and_checking_event_length(tmp_path):
    # To make sure that the correct number of lines are skipped when loading 
    # only a subset of events, we create a JETSCAPE file with a known number of 
    # events and then load a subset of events from it. We then check if the 
    # number of events loaded is equal to the number of events requested.
    num_events = 8
    num_output_per_event = [3, 1, 8, 4, 7, 11, 17, 2]
    
    tmp_jetscape_file = create_temporary_jetscape_file(tmp_path, num_events, 
                                                num_output_per_event)
    
    # Single events
    for event in range(num_events):
        jetscape = Jetscape(tmp_jetscape_file, events=event)
        assert jetscape.num_events() == 1
        assert len(jetscape.particle_objects_list()[0]) == num_output_per_event[event]
        del(jetscape)

    # Multiple events
    for event_start in range(num_events):
        for event_end in range(event_start, num_events):
            jetscape = Jetscape(tmp_jetscape_file, events=(event_start, event_end))

            assert jetscape.num_events() == event_end - event_start + 1

            for event in range(event_end - event_start + 1):
                assert len(jetscape.particle_objects_list()[event]) == \
                       num_output_per_event[event + event_start]
            del(jetscape)

def test_num_output_per_event(jetscape_file_path):
    num_output_jetscape = [[1, 25],[2, 41],[3, 46],[4, 43],[5, 45]]
    
    jetscape = Jetscape(jetscape_file_path)
    assert (jetscape.num_output_per_event() == num_output_jetscape).all()

def test_num_events(tmp_path,jetscape_file_path):
    jetscape = Jetscape(jetscape_file_path)
    assert jetscape.num_events() == 5

    number_of_events = [1, 5, 17, 3, 44, 101, 98]

    for events in number_of_events:
        # Create temporary Jetscape files
        tmp_jetscape_file = create_temporary_jetscape_file(tmp_path, events)
        jetscape = Jetscape(tmp_jetscape_file)
        assert jetscape.num_events() == events
        del(jetscape)
        del(tmp_jetscape_file)

def test_set_num_events(tmp_path):
    number_of_events = [1, 3, 7, 14, 61, 99]

    # Create multiple temporary Jetscape files with different numbers of events
    for events in number_of_events:
        tmp_jetscape_file = create_temporary_jetscape_file(tmp_path, events)
        jetscape = Jetscape(tmp_jetscape_file)
        assert jetscape.num_events() == events
        del(jetscape)
        del(tmp_jetscape_file)

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

# In the next tests we test a subset of filters only, as the filters are tested
# separately and completely. This test is therefore seen as an integration
# test in the Jetscape class.
@pytest.fixture
def particle_list_strange():
    particle_list1 = []
    for _ in range(5):
        p = Particle()
        p.pdg = 321
        particle_list1.append(p)
    for _ in range(5):
        p = Particle()
        p.pdg = 211
        particle_list1.append(p)
    particle_list2 = []
    for _ in range(7):
        p = Particle()
        p.pdg = 321
        particle_list2.append(p)
    for _ in range(15):
        p = Particle()
        p.pdg = 211
        particle_list2.append(p)
    return [particle_list1, particle_list2]

def test_filter_strangeness_in_Jetscape(tmp_path,particle_list_strange):
    tmp_jetscape_file = create_temporary_jetscape_file(tmp_path, 2)
    jetscape = Jetscape(tmp_jetscape_file)
    jetscape.particle_list_ = particle_list_strange
    jetscape.strange_particles()

    assert np.array_equal(jetscape.num_output_per_event(), np.array([[1,5],[2,7]]))

def test_filter_status_in_Jetscape(jetscape_file_path):
    jetscape = Jetscape(jetscape_file_path).particle_status(1)
    
    assert jetscape.num_events() == 5
    assert (jetscape.num_output_per_event() == np.array([[1, 0],[2, 0],[3, 0],[4, 0],[5, 0]])).all()

def test_filter_rapidity_in_Jetscape_constructor(tmp_path,jetscape_file_path):
    # In this test we test the integration of filters in the Jetscape constructor,
    # which are selected with keyword arguments while initializing the Jetscape object.

    pi0_y0 = (0, 111, 27, 0.138, 0.0, 0.0, 0.0)
    pi0_y1 = (1, 111, 27, 0.138, 0.0, 0.0, 0.10510)

    data_pi0_y0 = ' '.join(map(str, pi0_y0)) + '\n'
    data_pi0_y1 = ' '.join(map(str, pi0_y1)) + '\n'

    # Create a particle list to be loaded into the Jetscape object
    jetscape_file = str(tmp_path / "jetscape_test.dat")
    header_line = "#	JETSCAPE_FINAL_STATE	v2	|	N	pid	status	E	Px	Py	Pz\n"

    # Create a Jetscape file with two events
    # Event 1: 6 pi0 with Y=0, 12 with Y=1
    # Event 2: 10 pi0 with Y=0, 10 with Y=1
    with open(jetscape_file, "w") as f:
        f.write(header_line)
        f.write("#	Event	1	weight	1	EPangle	0	N_hadrons	18\n")
        for _ in range(6):
            f.write(data_pi0_y0)
            f.write(data_pi0_y1)
            f.write(data_pi0_y1)
        f.write("#	Event	2	weight	1	EPangle	0	N_hadrons	20\n")
        for _ in range(10):
            f.write(data_pi0_y0)
            f.write(data_pi0_y1)
        f.write("#	sigmaGen	0.000314633	sigmaErr	6.06164e-07\n")

    # print content of temporary file
    with open(jetscape_file, "r") as f:
        print(f.read())

    # Test filtering for the mid-rapidity particles
    jetscape = Jetscape(jetscape_file, filters={'rapidity_cut': 0.5})

    print(jetscape.num_output_per_event_)
    print(len(jetscape.particle_list()[0]), len(jetscape.particle_list()[1]))
    assert jetscape.num_events() == 2
    assert np.array_equal(jetscape.num_output_per_event(), np.array([[1,6],[2,10]]))

def test_filter_status_in_Jetscape_constructor(jetscape_file_path):
    # Perform status filtering such that everything should be filtered out with 
    # the given jetscape events in the file.
    jetscape1 = Jetscape(jetscape_file_path, filters={'particle_status': 1})
    
    assert jetscape1.num_events() == 5
    assert np.array_equal(jetscape1.num_output_per_event(), np.array([[1, 0],[2, 0],[3, 0],[4, 0],[5, 0]]))

@pytest.fixture
def particle_list_charge():
    particle_list1 = []
    for _ in range(5):
        p = Particle()
        p.charge = 1
        particle_list1.append(p)
    for _ in range(3):
        p = Particle()
        p.charge = 0
        particle_list1.append(p)
    particle_list2 = []
    for _ in range(7):
        p = Particle()
        p.charge = 0
        particle_list2.append(p)
    for _ in range(15):
        p = Particle()
        p.charge = 1
        particle_list2.append(p)
    return [particle_list1, particle_list2]

def test_filter_charge_in_Jetscape(tmp_path,particle_list_charge):
    tmp_jetscape_file = create_temporary_jetscape_file(tmp_path, 2)
    jetscape1 = Jetscape(tmp_jetscape_file)
    jetscape2 = Jetscape(tmp_jetscape_file)
    jetscape1.particle_list_ = particle_list_charge
    jetscape2.particle_list_ = copy.deepcopy(particle_list_charge)
    jetscape1.charged_particles()
    jetscape2.uncharged_particles()

    assert np.array_equal(jetscape1.num_output_per_event(), np.array([[1,5],[2,15]]))
    assert np.array_equal(jetscape2.num_output_per_event(), np.array([[1,3],[2,7]]))

@pytest.fixture
def particle_list_pdg():
    particle_list1 = []
    for _ in range(5):
        p = Particle()
        p.pdg = 211
        particle_list1.append(p)
    for _ in range(3):
        p = Particle()
        p.pdg = 22
        particle_list1.append(p)
    particle_list2 = []
    for _ in range(7):
        p = Particle()
        p.pdg = 22
        particle_list2.append(p)
    for _ in range(15):
        p = Particle()
        p.pdg = 221
        particle_list2.append(p)
    return [particle_list1, particle_list2]

def test_filter_pdg_in_Jetscape(tmp_path,particle_list_pdg):
    tmp_jetscape_file = create_temporary_jetscape_file(tmp_path, 2)
    jetscape1 = Jetscape(tmp_jetscape_file)
    jetscape2 = Jetscape(tmp_jetscape_file)
    jetscape1.particle_list_ = particle_list_pdg
    jetscape2.particle_list_ = copy.deepcopy(particle_list_pdg)
    jetscape1.particle_species([211,221])
    jetscape2.remove_particle_species(22)

    assert np.array_equal(jetscape1.num_output_per_event(), np.array([[1,5],[2,15]]))
    assert np.array_equal(jetscape2.num_output_per_event(), np.array([[1,5],[2,15]]))

def test_Jetscape_print(jetscape_file_path, output_path):
    jetscape = Jetscape(jetscape_file_path)
    jetscape.print_particle_lists_to_file(output_path)
    assert filecmp.cmp(jetscape_file_path, output_path)
    os.remove(output_path)

def test_Jetscape_get_sigmaGen(jetscape_file_path):
    jetscape = Jetscape(jetscape_file_path)
    assert jetscape.get_sigmaGen() == (0.000314633,6.06164e-07)
    
def test_Jetscape_charge_filter_one_event(jetscape_file_path):
    jetscape = Jetscape(jetscape_file_path, events=0).charged_particles()
    assert jetscape.num_events() == 1
    assert (jetscape.num_output_per_event() == np.array([[1, 14]])).all()
