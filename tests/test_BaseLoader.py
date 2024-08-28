from sparkx.Loader.BaseLoader import BaseLoader

import pytest
from io import StringIO
from typing import Dict, Any

# Assuming BaseLoader is imported from the appropriate module

# A concrete subclass for testing purposes
class ConcreteLoader(BaseLoader):
    def __init__(self, path: str) -> None:
        self.path = path

    def load(self, **kwargs: Dict[str, Any]) -> Any:
        # A dummy implementation for testing
        return {"data": "dummy data"}

    def _get_num_skip_lines(self) -> int:
        # Dummy implementation for testing
        return 3

# Fixtures for creating instances
@pytest.fixture
def concrete_loader():
    return ConcreteLoader(path="dummy_path")

# Test for the abstract method __init__
def test_abstract_init():
    with pytest.raises(TypeError):
        BaseLoader("dummy_path")

# Test for the abstract method load
def test_abstract_load():
    loader = ConcreteLoader("dummy_path")
    assert loader.load() == {"data": "dummy data"}

# Test for _check_that_tuple_contains_integers_only method
def test_check_that_tuple_contains_integers_only(concrete_loader):
    # Should not raise an exception
    concrete_loader._check_that_tuple_contains_integers_only((1, 2, 3))

    # Should raise TypeError
    with pytest.raises(TypeError):
        concrete_loader._check_that_tuple_contains_integers_only((1, 'a', 3))

# Test for _skip_lines method
def test_skip_lines(concrete_loader):
    # Simulate a file with multiple lines
    file_content = "header\ncomment\nparticle1\nparticle2\n"
    fname = StringIO(file_content)

    concrete_loader._skip_lines(fname)
    # Check if the correct line is now the first line to be read
    assert fname.readline().strip() == "particle2"

# Test for _get_num_skip_lines method
def test_get_num_skip_lines(concrete_loader):
    # This will test the dummy implementation of _get_num_skip_lines
    assert concrete_loader._get_num_skip_lines() == 3
