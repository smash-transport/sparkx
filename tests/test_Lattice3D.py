import pytest
import numpy as np
from sparkx.Lattice3D import Lattice3D

@pytest.fixture
def sample_lattice():
    return Lattice3D(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1,
                     num_points_x=10, num_points_y=10, num_points_z=10)

def test_initialization(sample_lattice):
    assert sample_lattice.num_points_x_ == 10
    assert sample_lattice.num_points_y_ == 10
    assert sample_lattice.num_points_z_ == 10
    assert sample_lattice.cell_volume_ == 0.001
    assert np.allclose(sample_lattice.x_values_, np.linspace(0, 1, 10))
    assert np.allclose(sample_lattice.y_values_, np.linspace(0, 1, 10))
    assert np.allclose(sample_lattice.z_values_, np.linspace(0, 1, 10))
    assert np.all(sample_lattice.grid_ == 0)
    assert sample_lattice.n_sigma_x_ == 3
    assert sample_lattice.n_sigma_y_ == 3
    assert sample_lattice.n_sigma_z_ == 3
    assert sample_lattice.spacing_x_ == 1/9
    assert sample_lattice.spacing_y_ == 1/9
    assert sample_lattice.spacing_z_ == 1/9
    assert sample_lattice.density_x_ == 0.1
    assert sample_lattice.density_y_ == 0.1
    assert sample_lattice.density_z_ == 0.1

def test_set_value_by_index(sample_lattice):
    sample_lattice.set_value_by_index(1, 2, 3, 5)
    assert sample_lattice.grid_[1, 2, 3] == 5

def test_set_value_by_index_invalid_indices(sample_lattice):
    with pytest.warns(UserWarning):
        sample_lattice.set_value_by_index(15, 20, 25, 5)

def test_is_valid_index(sample_lattice):
    assert sample_lattice._Lattice3D__is_valid_index(1, 2, 3) == True
    assert sample_lattice._Lattice3D__is_valid_index(15, 20, 25) == False

def test_get_value_by_index(sample_lattice):
    sample_lattice.set_value_by_index(1, 2, 3, 5)
    assert sample_lattice.get_value_by_index(1, 2, 3) == 5

def test_get_value_by_index_invalid_indices(sample_lattice):
    with pytest.warns(UserWarning):
        assert sample_lattice.get_value_by_index(15, 20, 25) is None

def test_get_index_within_range(sample_lattice):
    values = np.linspace(0, 1, 10)
    assert sample_lattice._Lattice3D__get_index(0.3, values, 10) == 2

def test_get_index_at_lower_bound(sample_lattice):
    values = np.linspace(0, 1, 10)
    assert sample_lattice._Lattice3D__get_index(0, values, 10) == 0

def test_get_index_at_upper_bound(sample_lattice):
    values = np.linspace(0, 1, 10)
    assert sample_lattice._Lattice3D__get_index(1, values, 10) == 8

def test_get_index_outside_range_raises_error(sample_lattice):
    values = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        sample_lattice._Lattice3D__get_index(2, values, 10)

def test_get_index_nearest_neighbor_within_range(sample_lattice):
    values = np.linspace(0, 1, 10)
    assert sample_lattice._Lattice3D__get_index_nearest_neighbor(0.3, values) == 3

def test_get_index_nearest_neighbor_at_lower_bound(sample_lattice):
    values = np.linspace(0, 1, 10)
    assert sample_lattice._Lattice3D__get_index_nearest_neighbor(0, values) == 0

def test_get_index_nearest_neighbor_at_upper_bound(sample_lattice):
    values = np.linspace(0, 1, 10)
    assert sample_lattice._Lattice3D__get_index_nearest_neighbor(1, values) == 9

def test_get_index_nearest_neighbor_outside_range_raises_error(sample_lattice):
    values = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        sample_lattice._Lattice3D__get_index_nearest_neighbor(2, values)

def test_get_indices_within_range(sample_lattice):
    assert sample_lattice._Lattice3D__get_indices(0.3, 0.4, 0.5) == (2, 3, 4)

def test_get_indices_at_lower_bounds(sample_lattice):
    assert sample_lattice._Lattice3D__get_indices(0, 0, 0) == (0, 0, 0)

def test_get_indices_at_upper_bounds(sample_lattice):
    assert sample_lattice._Lattice3D__get_indices(1, 1, 1) == (8, 8, 8)

def test_get_indices_outside_range_raises_error(sample_lattice):
    with pytest.raises(ValueError):
        sample_lattice._Lattice3D__get_indices(2, 2, 2)

def test_get_indices_nearest_neighbor_within_range(sample_lattice):
    assert sample_lattice._Lattice3D__get_indices_nearest_neighbor(0.3, 0.4, 0.5) == (3, 4, 4)

def test_get_indices_nearest_neighbor_at_lower_bounds(sample_lattice):
    assert sample_lattice._Lattice3D__get_indices_nearest_neighbor(0, 0, 0) == (0, 0, 0)

def test_get_indices_nearest_neighbor_at_upper_bounds(sample_lattice):
    assert sample_lattice._Lattice3D__get_indices_nearest_neighbor(1, 1, 1) == (9, 9, 9)

def test_get_indices_nearest_neighbor_outside_range_raises_error(sample_lattice):
    with pytest.raises(ValueError):
        sample_lattice._Lattice3D__get_indices_nearest_neighbor(2, 2, 2)

def test_set_value_within_range(sample_lattice):
    sample_lattice.set_value(0.3, 0.4, 0.5, 5)
    assert sample_lattice.get_value_by_index(2, 3, 4) == 5

def test_set_value_at_lower_bounds(sample_lattice):
    sample_lattice.set_value(0, 0, 0, 10)
    assert sample_lattice.get_value_by_index(0, 0, 0) == 10

def test_set_value_at_upper_bounds(sample_lattice):
    sample_lattice.set_value(1, 1, 1, 15)
    assert sample_lattice.get_value_by_index(8, 8, 8) == 15

def test_set_value_outside_range_raises_error(sample_lattice):
    with pytest.raises(ValueError):
        sample_lattice.set_value(2, 2, 2, 20)

def test_set_value_nearest_neighbor_within_range(sample_lattice):
    sample_lattice.set_value_nearest_neighbor(0.25, 0.35, 0.45, 7)
    assert sample_lattice.get_value_by_index(2, 3, 4) == 7

def test_set_value_nearest_neighbor_at_lower_bounds(sample_lattice):
    sample_lattice.set_value_nearest_neighbor(0, 0, 0, 12)
    assert sample_lattice.get_value_by_index(0, 0, 0) == 12

def test_set_value_nearest_neighbor_at_upper_bounds(sample_lattice):
    sample_lattice.set_value_nearest_neighbor(1, 1, 1, 18)
    assert sample_lattice.get_value_by_index(9, 9, 9) == 18

def test_set_value_nearest_neighbor_outside_range_raises_error(sample_lattice):
    with pytest.raises(ValueError):
        sample_lattice.set_value_nearest_neighbor(2, 2, 2, 25)

def test_get_value_within_range(sample_lattice):
    sample_lattice.set_value(0.3, 0.4, 0.5, 5)
    assert sample_lattice.get_value(0.3, 0.4, 0.5) == 5

def test_get_value_at_lower_bounds(sample_lattice):
    sample_lattice.set_value(0, 0, 0, 10)
    assert sample_lattice.get_value(0, 0, 0) == 10

def test_get_value_at_upper_bounds(sample_lattice):
    sample_lattice.set_value(1, 1, 1, 15)
    assert sample_lattice.get_value(1, 1, 1) == 15

def test_get_value_outside_range_returns_none(sample_lattice):
    with pytest.raises(ValueError):
        sample_lattice.get_value(2, 2, 2)

def test_get_value_nearest_neighbor_within_range(sample_lattice):
    sample_lattice.set_value(0.3, 0.4, 0.5, 5)
    assert sample_lattice.get_value_nearest_neighbor(0.25, 0.35, 0.45) == 5

def test_get_value_nearest_neighbor_at_lower_bounds(sample_lattice):
    sample_lattice.set_value(0, 0, 0, 10)
    assert sample_lattice.get_value_nearest_neighbor(0, 0, 0) == 10

def test_get_value_nearest_neighbor_at_upper_bounds(sample_lattice):
    sample_lattice.set_value(0.9, 0.9, 0.9, 15)
    assert sample_lattice.get_value_nearest_neighbor(0.9, 0.9, 0.9) == 15

def test_get_value_nearest_neighbor_outside_range_returns_none(sample_lattice):
    with pytest.raises(ValueError):
        sample_lattice.get_value_nearest_neighbor(2, 2, 2)
