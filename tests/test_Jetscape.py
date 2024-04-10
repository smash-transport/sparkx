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
from sparkx.Jetscape import Jetscape

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

