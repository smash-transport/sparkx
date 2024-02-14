import pytest
import os
from sparkx.Jetscape import Jetscape

@pytest.fixture
def jetscape_file_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(os.path.dirname(__file__), 'test_files', 'test_jetscape.dat')

def test_jetscape_initialization(jetscape_file_path):
    # Test if Jetscape object initializes correctly with a valid file
    jetscape = Jetscape(jetscape_file_path)
    assert jetscape is not None

def test_invalid_file_type():
    # Test if Jetscape raises TypeError for invalid file type
    with pytest.raises(TypeError):
        Jetscape("invalid_file.txt")

def test_invalid_event_number():
    # Test if Jetscape raises ValueError for invalid event numbers
    with pytest.raises(ValueError):
        Jetscape("valid_jetscape.dat", events=(2, 1))
    with pytest.raises(ValueError):
        Jetscape("valid_jetscape.dat", events=(-3, -2))
    with pytest.raises(ValueError):
        Jetscape("valid_jetscape.dat", events=-2)