from sparkx.Oscar import Oscar
import numpy as np
import pytest

OSCAR_FILE = "./test_files/particle_lists.oscar"
OSCAR_EXTENDED_FILE = "./test_files/particle_lists_extended.oscar"


def test_constructor_invalid_initialization():
    # Initialization with invalid input file
    invalid_input_file = "./test_files/not_existing_file"
    with pytest.raises(FileNotFoundError):
        Oscar(invalid_input_file)
        
    # Initalization with invalid kwargs: events not a number
    with pytest.raises(TypeError):
        Oscar(OSCAR_FILE, events=np.nan)
        
    with pytest.raises(TypeError):
        Oscar(OSCAR_FILE, events=("a", "b"))
        
    # Initalization with invalid kwargs: events negative
    with pytest.raises(ValueError):
        Oscar(OSCAR_FILE, events=-1)
        
    with pytest.raises(ValueError):
        Oscar(OSCAR_FILE, events=(-4, -1))
        
    # Initalization with invalid kwargs: events out of boundaries
    with pytest.raises(IndexError):
        Oscar(OSCAR_FILE, events=5)
        
    with pytest.raises(IndexError):
        Oscar(OSCAR_FILE, events=(5, 10))
        