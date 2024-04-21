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
import copy

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
    
def test_loading_defined_events_and_checking_event_length(tmp_path):
    # To make sure that the correct number of lines are skipped when loading
    # only a subset of events, we create an OSCAR file with a known number of
    # events and then load a subset of events from it. We then check if the
    # number of events loaded is equal to the number of events requested.
    num_events = 8
    num_output_per_event = [3, 1, 8, 4, 7, 11, 17, 2]

    tmp_oscar_file = create_temporary_oscar_file(tmp_path, num_events,
                                                 "Oscar2013", num_output_per_event)
    # Single events
    for event in range(num_events):
        oscar = Oscar(tmp_oscar_file, events=event)
        assert oscar.num_events() == 1
        assert len(oscar.particle_objects_list()[0]) == num_output_per_event[event]
        del(oscar)
    
    # Multiple events
    for event_start in range(num_events):
        for event_end in range(event_start, num_events):
            oscar = Oscar(tmp_oscar_file, events=(event_start, event_end))

            assert oscar.num_events() == event_end - event_start + 1

            for event in range(event_end - event_start + 1):
                assert len(oscar.particle_objects_list()[event]) == \
                       num_output_per_event[event + event_start]
            del(oscar)

def test_oscar_format(tmp_path):
    tmp_oscar_file = create_temporary_oscar_file(tmp_path, 2,"Oscar2013")
    oscar = Oscar(tmp_oscar_file)
    assert oscar.oscar_format() == "Oscar2013"

    tmp_oscar_extended_file = create_temporary_oscar_file(tmp_path, 2,"Oscar2013Extended")
    oscar = Oscar(tmp_oscar_extended_file)
    assert oscar.oscar_format() == "Oscar2013Extended"

def test_num_output_per_event(tmp_path, oscar_old_extended_file_path):
    num_events = 7
    num_output_per_event = [1, 35, 27, 0, 9, 17, 3]

    tmp_oscar_file = create_temporary_oscar_file(tmp_path, num_events,
                                                 "Oscar2013", num_output_per_event)
    tmp_oscar_file_extended = create_temporary_oscar_file(tmp_path, num_events,
                                                 "Oscar2013Extended", num_output_per_event)

    num_output = [[0, 1],[1, 35],[2, 27],[3, 0],[4, 9],[5, 17],[6, 3]]
    num_output_oscar_old_extended = [[0, 4],[1, 0]]

    oscar = Oscar(tmp_oscar_file)
    assert (oscar.num_output_per_event() == num_output).all()

    oscar_extended = Oscar(tmp_oscar_file_extended)
    assert (oscar_extended.num_output_per_event() == num_output).all()

    oscar_old_extended = Oscar(oscar_old_extended_file_path)
    assert (oscar_old_extended.num_output_per_event() == num_output_oscar_old_extended).all()

def test_num_events(tmp_path, oscar_old_extended_file_path):
    number_of_events = [1, 5, 17, 3, 44, 101, 98]

    for events in number_of_events:
        # Create temporary Oscar2013 files
        tmp_oscar_file = create_temporary_oscar_file(tmp_path, events,"Oscar2013")
        oscar = Oscar(tmp_oscar_file)
        assert oscar.num_events() == events
        del(oscar)
        del(tmp_oscar_file)

    for events in number_of_events:
        # Create temporary Oscar2013Extended files
        tmp_oscar_file = create_temporary_oscar_file(tmp_path, events,"Oscar2013Extended")
        oscar = Oscar(tmp_oscar_file)
        assert oscar.num_events() == events
        del(oscar)
        del(tmp_oscar_file)

    oscar_old_extended = Oscar(oscar_old_extended_file_path)
    assert oscar_old_extended.num_events() == 2

def test_set_num_events(tmp_path):
    number_of_events = [1, 3, 7, 14, 61, 99]

    # Create multiple temporary Oscar2013 files with different numbers of events
    for events in number_of_events:
        tmp_oscar_file = create_temporary_oscar_file(tmp_path, events,"Oscar2013")
        oscar = Oscar(tmp_oscar_file)
        assert oscar.num_events() == events
        del(oscar)
        del(tmp_oscar_file)

    # Create multiple temporary Oscar2013Extended files with different numbers of events
    for events in number_of_events:
        tmp_oscar_file = create_temporary_oscar_file(tmp_path, events,"Oscar2013Extended")
        oscar = Oscar(tmp_oscar_file)
        assert oscar.num_events() == events
        del(oscar)
        del(tmp_oscar_file)

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
    # 2 particle objects each and turn it back into a python list.
    particle_objects = (np.array(particle_objects).reshape((5, 2))).tolist()

    oscar = Oscar(oscar_file_path)
    oscar.particle_list_ = particle_objects
    oscar.num_events_ = 5
    oscar.num_output_per_event_ = np.array([[0,2],[1,2],[2,2],[3,2],[4,2]])

    assert (oscar.particle_list() == dummy_particle_list_nested)

def test_filter_in_oscar(tmp_path):
    # In this test we test a subset of filters only, as the filters are tested
    # separately and completely. This test is therefore seen as an integration
    # test in Oscar.

    # Create a list of different particles
    proton_spectator = Particle("Oscar2013Extended", [200, 5.73, -4.06, -2.02, 0.93, 90.86, 0.07, -0.11, 90.86, 2212, 172, 1, 0, -2.00, 1, 0, 0, 0, 0, 0, 1, 0])
    proton_participant = Particle("Oscar2013Extended", [200, 5.73, -4.06, -2.02, 0.93, 90.86, 0.07, -0.11, 90.86, 2212, 172, 1, 1, -2.00, 1, 0, 0, 0, 0, 0, 1, 0])
    pi_0_spectator = Particle("Oscar2013Extended", [200, -3.91, -3.58, -199.88, 0.14, 9.79, 0.08, -0.18, -9.78, 111, 16220, 0, 0, 200, 1, 8884, 5, 200, -213, 0, 0, 0])
    pi_0_participant = Particle("Oscar2013Extended", [200, -3.91, -3.58, -199.88, 0.14, 9.79, 0.08, -0.18, -9.78, 111, 16220, 0, 1, 200, 1, 8884, 5, 200, -213, 0, 0, 0])
    Kaon_0_spectator = Particle("Oscar2013Extended", [200, 2.57, -1.94, -9.83, 0.49, 3.47, 0.08, -0.26, -3.42, 311, 3228, 0, 0, 49.12, 0, 369, 45, 0.06, 2112, 2212, 0, 1])
    Kaon_0_participant = Particle("Oscar2013Extended", [200, 2.57, -1.94, -9.83, 0.49, 3.47, 0.08, -0.26, -3.42, 311, 3228, 0, 1, 49.12, 0, 369, 45, 0.06, 2112, 2212, 0, 1])

    # Create a temporary particle list with 2 events to generate temporary OSCAR
    # objects whose particle lists will be manually replaced with the ones created
    # in this test. This gives us full control of the particle properties and
    # allows us to test the filter function.
    tmp_oscar_file = create_temporary_oscar_file(tmp_path, 2, "Oscar2013Extended")

    # Test filter: un/charged particles
    oscar_charged = Oscar(tmp_oscar_file)
    oscar_uncharged = Oscar(tmp_oscar_file)
    oscar_empty = Oscar(tmp_oscar_file)
    oscar_charged_participants = Oscar(tmp_oscar_file)
    oscar_uncharged_specators = Oscar(tmp_oscar_file)
    event_1 = []
    event_2 = []

    # Event 1: 6 proton spectators, 12 proton participants and 18 pi0 participants
    for i in range(6):
        event_1.append(proton_spectator)
        event_1.append(proton_participant)
        event_1.append(proton_participant)
        event_1.append(pi_0_participant)
        event_1.append(pi_0_participant)
        event_1.append(pi_0_participant)
    # Event 2: 10 proton participants, 10 K0 participants and 10 K0 spectators
    for i in range(10):
        event_2.append(proton_participant)
        event_2.append(Kaon_0_participant)
        event_2.append(Kaon_0_spectator)

    particle_objects=[event_1, event_2]
    oscar_charged.particle_list_ = particle_objects
    oscar_uncharged.particle_list_ = copy.deepcopy(particle_objects)
    oscar_empty.particle_list_ = copy.deepcopy(particle_objects)
    oscar_charged_participants.particle_list_ = copy.deepcopy(particle_objects)
    oscar_uncharged_specators.particle_list_ = copy.deepcopy(particle_objects)

    oscar_charged.charged_particles()
    oscar_uncharged.uncharged_particles()
    oscar_empty.charged_particles().uncharged_particles()
    oscar_charged_participants.charged_particles().participants()
    oscar_uncharged_specators.uncharged_particles().spectators()

    assert np.array_equal(oscar_charged.num_output_per_event(), np.array([[0, 18],[1, 10]]))
    assert np.array_equal(oscar_uncharged.num_output_per_event(), np.array([[0, 18],[1, 20]]))
    assert np.array_equal(oscar_empty.num_output_per_event(), np.array([[0, 0],[1, 0]]))
    assert np.array_equal(oscar_charged_participants.num_output_per_event(), np.array([[0, 12],[1, 10]]))
    assert np.array_equal(oscar_uncharged_specators.num_output_per_event(), np.array([[0, 0],[1, 10]]))
    del event_1, event_2, oscar_empty

    # Test filter: spectators/participants
    oscar_participants = Oscar(tmp_oscar_file)
    oscar_spectators = Oscar(tmp_oscar_file)
    oscar_empty = Oscar(tmp_oscar_file)
    oscar_spectators_strange = Oscar(tmp_oscar_file)
    oscar_participants_strange = Oscar(tmp_oscar_file)
    event_1 = []
    event_2 = []
    # Event 1: 5 pi0 spectators, 10 K0 participants and 15 K0 spectators
    for i in range(5):
        event_1.append(pi_0_spectator)
        event_1.append(Kaon_0_participant)
        event_1.append(Kaon_0_participant)
        event_1.append(Kaon_0_spectator)
        event_1.append(Kaon_0_spectator)
        event_1.append(Kaon_0_spectator)
    # Event 2: 11 pi0 participants and 22 K0 spectators
    for i in range(11):
        event_2.append(pi_0_participant)
        event_2.append(Kaon_0_spectator)
        event_2.append(Kaon_0_spectator)

    particle_objects=[event_1, event_2]
    oscar_participants.particle_list_ = particle_objects
    oscar_spectators.particle_list_ = copy.deepcopy(particle_objects)
    oscar_empty.particle_list_ = copy.deepcopy(particle_objects)
    oscar_spectators_strange.particle_list_ = copy.deepcopy(particle_objects)
    oscar_participants_strange.particle_list_ = copy.deepcopy(particle_objects)

    oscar_participants.participants()
    oscar_spectators.spectators()
    oscar_empty.participants().spectators()
    oscar_spectators_strange.spectators().strange_particles()
    oscar_participants_strange.participants().strange_particles()

    assert np.array_equal(oscar_participants.num_output_per_event(), np.array([[0, 10],[1, 11]]))
    assert np.array_equal(oscar_spectators.num_output_per_event(), np.array([[0, 20],[1, 22]]))
    assert np.array_equal(oscar_empty.num_output_per_event(), np.array([[0, 0],[1, 0]]))
    assert np.array_equal(oscar_spectators_strange.num_output_per_event(), np.array([[0, 15],[1, 22]]))
    assert np.array_equal(oscar_participants_strange.num_output_per_event(), np.array([[0, 10],[1, 0]]))

def test_filter_in_oscar_constructor(tmp_path):
    # In this test we test the integration of filters in the Oscar constructor,
    # which are selected with keyword arguments while initializing the Oscar object.

    # Particle information to generate a particle list for Oscar
    proton_spectator = (200, 5.73, -4.06, -2.02, 0.93, 90.86, 0.07, -0.11, 90.86, 2212, 172, 1, 0, -2.00, 1, 0, 0, 0, 0, 0, 1, 0)
    proton_participant = (200, 5.73, -4.06, -2.02, 0.93, 90.86, 0.07, -0.11, 90.86, 2212, 172, 1, 1, -2.00, 1, 0, 0, 0, 0, 0, 1, 0)
    pi_0_spectator = (200, -3.91, -3.58, -199.88, 0.14, 9.79, 0.08, -0.18, -9.78, 111, 16220, 0, 0, 200, 1, 8884, 5, 200, -213, 0, 0, 0)
    pi_0_participant = (200, -3.91, -3.58, -199.88, 0.14, 9.79, 0.08, -0.18, -9.78, 111, 16220, 0, 1, 200, 1, 8884, 5, 200, -213, 0, 0, 0)
    Kaon_0_spectator = (200, 2.57, -1.94, -9.83, 0.49, 3.47, 0.08, -0.26, -3.42, 311, 3228, 0, 0, 49.12, 0, 369, 45, 0.06, 2112, 2212, 0, 1)
    Kaon_0_participant = (200, 2.57, -1.94, -9.83, 0.49, 3.47, 0.08, -0.26, -3.42, 311, 3228, 0, 1, 49.12, 0, 369, 45, 0.06, 2112, 2212, 0, 1)

    data_proton_spectator = ' '.join(map(str, proton_spectator)) + '\n'
    data_proton_participant = ' '.join(map(str, proton_participant)) + '\n'
    data_pi_0_spectator = ' '.join(map(str, pi_0_spectator)) + '\n'
    data_pi_0_participant = ' '.join(map(str, pi_0_participant)) + '\n'
    data_Kaon_0_spectator = ' '.join(map(str, Kaon_0_spectator)) + '\n'
    data_Kaon_0_participant = ' '.join(map(str, Kaon_0_participant)) + '\n'

    # Create a particle list to be loaded into the Oscar object
    oscar_file = str(tmp_path / "particle_lists.oscar")
    header_lines = [
            "#!OSCAR2013Extended particle_lists t x y z mass p0 px py pz pdg ID charge ncoll form_time xsecfac proc_id_origin proc_type_origin time_last_coll pdg_mother1 pdg_mother2 baryon_number strangeness\n",
            "# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e none fm none none none fm none none none none\n",
            "# SMASH-3.1rc-23-g59a05e65f\n"
    ]
    header = ''.join(header_lines)

    # Create an Oscar file with 2 events
    # Event 1: 6 proton spectators, 12 proton participants and 18 pi0 participants
    # Event 2: 10 proton participants, 10 K0 participants and 10 K0 spectators
    with open(oscar_file, "w") as f:
        f.write(header)
        f.write("# event 0 out 36\n")
        for i in range(6):
            f.write(data_proton_spectator)
            f.write(data_proton_participant)
            f.write(data_proton_participant)
            f.write(data_pi_0_participant)
            f.write(data_pi_0_participant)
            f.write(data_pi_0_participant)
        f.write("# event 0 end 0 impact   0.000 scattering_projectile_target yes\n")
        f.write("# event 1 out 30\n")
        for i in range(10):
            f.write(data_proton_participant)
            f.write(data_Kaon_0_participant)
            f.write(data_Kaon_0_spectator)
        f.write("# event 1 end 0 impact   0.000 scattering_projectile_target yes\n")

    # Test filter: un/charged particles
    oscar_charged = Oscar(oscar_file, filters={'charged_particles':True})
    oscar_uncharged = Oscar(oscar_file, filters={'uncharged_particles': True})
    oscar_empty = Oscar(oscar_file, filters={'charged_particles': True, 'uncharged_particles': True})
    oscar_charged_participants = Oscar(oscar_file, filters={'charged_particles': True, 'participants': True})
    oscar_uncharged_specators = Oscar(oscar_file, filters={'uncharged_particles': True, 'spectators': True})

    assert np.array_equal(oscar_charged.num_output_per_event(), np.array([[0, 18],[1, 10]]))
    assert np.array_equal(oscar_uncharged.num_output_per_event(), np.array([[0, 18],[1, 20]]))
    assert np.array_equal(oscar_empty.num_output_per_event(), np.array([[0, 0],[1, 0]]))
    assert np.array_equal(oscar_charged_participants.num_output_per_event(), np.array([[0, 12],[1, 10]]))
    assert np.array_equal(oscar_uncharged_specators.num_output_per_event(), np.array([[0, 0],[1, 10]]))
    del oscar_file, oscar_empty

    # Create a particle list to be loaded into the Oscar object
    oscar_file = str(tmp_path / "particle_lists.oscar")
    header_lines = [
            "#!OSCAR2013Extended particle_lists t x y z mass p0 px py pz pdg ID charge ncoll form_time xsecfac proc_id_origin proc_type_origin time_last_coll pdg_mother1 pdg_mother2 baryon_number strangeness\n",
            "# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e none fm none none none fm none none none none\n",
            "# SMASH-3.1rc-23-g59a05e65f\n"
    ]
    header = ''.join(header_lines)

    # Create an Oscar file with 2 events
    # Event 1: 5 pi0 spectators, 10 K0 participants and 15 K0 spectators
    # Event 2: 11 pi0 participants and 22 K0 spectators
    with open(oscar_file, "w") as f:
        f.write(header)
        f.write("# event 0 out 30\n")
        for i in range(5):
            f.write(data_pi_0_spectator)
            f.write(data_Kaon_0_participant)
            f.write(data_Kaon_0_participant)
            f.write(data_Kaon_0_spectator)
            f.write(data_Kaon_0_spectator)
            f.write(data_Kaon_0_spectator)
        f.write("# event 0 end 0 impact   0.000 scattering_projectile_target yes\n")
        f.write("# event 1 out 33\n")
        for i in range(11):
            f.write(data_pi_0_participant)
            f.write(data_Kaon_0_spectator)
            f.write(data_Kaon_0_spectator)
        f.write("# event 1 end 0 impact   0.000 scattering_projectile_target yes\n")

    # Test filter: spectators/participants
    oscar_participants = Oscar(oscar_file, filters={'participants': True})
    oscar_spectators = Oscar(oscar_file, filters={'spectators': True})
    oscar_empty = Oscar(oscar_file, filters={'participants': True, 'spectators': True})
    oscar_spectators_strange = Oscar(oscar_file, filters={'spectators': True, 'strange_particles': True})
    oscar_participants_strange = Oscar(oscar_file, filters={'participants': True, 'strange_particles': True})

    assert np.array_equal(oscar_participants.num_output_per_event(), np.array([[0, 10],[1, 11]]))
    assert np.array_equal(oscar_spectators.num_output_per_event(), np.array([[0, 20],[1, 22]]))
    assert np.array_equal(oscar_empty.num_output_per_event(), np.array([[0, 0],[1, 0]]))
    assert np.array_equal(oscar_spectators_strange.num_output_per_event(), np.array([[0, 15],[1, 22]]))
    assert np.array_equal(oscar_participants_strange.num_output_per_event(), np.array([[0, 10],[1, 0]]))

def test_standard_oscar_print(tmp_path, output_path):
    tmp_oscar_file = create_temporary_oscar_file(tmp_path, 5, "Oscar2013", [1, 7, 0, 36, 5])
    oscar = Oscar(tmp_oscar_file)
    oscar.print_particle_lists_to_file(output_path)
    assert filecmp.cmp(tmp_oscar_file, output_path)
    os.remove(output_path)

def test_extended_oscar_print(tmp_path, output_path):
    tmp_oscar_file = create_temporary_oscar_file(tmp_path, 5, "Oscar2013Extended", [4, 1, 42, 0, 3])
    oscar = Oscar(tmp_oscar_file)
    oscar.print_particle_lists_to_file(output_path)
    assert filecmp.cmp(tmp_oscar_file, output_path)
    os.remove(output_path)

def test_old_extended_oscar_print(oscar_old_extended_file_path, output_path):
    oscar = Oscar(oscar_old_extended_file_path)
    oscar.print_particle_lists_to_file(output_path)
    assert filecmp.cmp(oscar_old_extended_file_path, output_path)
    os.remove(output_path)
