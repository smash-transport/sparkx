import pytest
import numpy as np
from typing import List, Union
from sparkx.Loader.BaseLoader import BaseLoader
from sparkx.BaseStorer import BaseStorer
from sparkx.Particle import Particle


class MockParticle:
    def __init__(self, charge):
        self.charge = charge


class MockLoader(BaseLoader):
    def __init__(self, path: str) -> None:
        # Simple implementation for testing purposes
        self.path = path

    def load(self, **kwargs):
        # Return mock data for testing purposes
        particle_list = [[[1, 2, 3], [4, 5, 6]]]  # Example particle data
        num_events = 1
        num_output_per_event = np.array([[1, len(particle_list[0])]])
        return particle_list, num_events, num_output_per_event


# Concrete subclass for testing purposes
class ConcreteStorer(BaseStorer):
    def create_loader(self, arg: Union[str, List[List["Particle"]]]) -> None:
        # Use the mock loader instead of the abstract BaseLoader
        if isinstance(arg, str):
            self.loader_ = MockLoader(arg)
        else:
            raise ValueError("Invalid argument type for create_loader")

    def _particle_as_list(self, particle: "Particle") -> List:
        # Example implementation for converting a Particle object to a list
        return [1, 2, 3]

    def print_particle_lists_to_file(self, output_file) -> None:
        # Example implementation for the abstract method
        with open(output_file, "w") as file:
            for event in self.particle_list():
                file.write(str(event) + "\n")


# Fixtures for creating instances
@pytest.fixture
def concrete_storer():
    # Using a mock path to create a loader
    return ConcreteStorer(path="dummy_path")


# Test for the abstract method __init__
def test_abstract_init():
    with pytest.raises(TypeError):
        BaseStorer("dummy_path")


# Test for the abstract method create_loader
def test_abstract_create_loader():
    storer = ConcreteStorer("dummy_path")
    assert storer.loader_ is not None


# Test for the abstract method _particle_as_list
def test_particle_as_list(concrete_storer):
    particle = [1, 2, 3]  # Mock Particle object
    assert concrete_storer._particle_as_list(particle) == [1, 2, 3]


# Test for num_output_per_event method
def test_num_output_per_event(concrete_storer):
    concrete_storer.num_output_per_event_ = np.array([[1, 2], [2, 3]])
    assert np.array_equal(
        concrete_storer.num_output_per_event(), np.array([[1, 2], [2, 3]])
    )


# Test for num_events method
def test_num_events(concrete_storer):
    concrete_storer.num_events_ = 5
    assert concrete_storer.num_events() == 5


# Test for particle_list method
def test_particle_list(concrete_storer):
    concrete_storer.num_output_per_event_ = np.array([[1, 2], [2, 3]])
    concrete_storer.particle_list_ = [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12], [13, 14, 15]],
    ]
    assert concrete_storer.particle_list() == [[1, 2, 3], [4, 5, 6]], [
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
    ]


# Test for print_particle_lists_to_file method
def test_print_particle_lists_to_file(concrete_storer, tmp_path):
    output_file = tmp_path / "output.txt"
    concrete_storer.print_particle_lists_to_file(output_file)
    with open(output_file, "r") as f:
        content = f.read()
    assert content == "[1, 2, 3]\n[4, 5, 6]\n"
