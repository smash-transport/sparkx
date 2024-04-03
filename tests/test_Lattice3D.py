import pytest
import csv
import numpy as np
import copy
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
    assert sample_lattice._Lattice3D__get_index(1, values, 10) == 9

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
    assert sample_lattice._Lattice3D__get_indices(1, 1, 1) == (9, 9, 9)

def test_get_indices_outside_range_raises_error(sample_lattice):
    with pytest.raises(ValueError):
        sample_lattice._Lattice3D__get_indices(2, 2, 2)

def test_get_indices_nearest_neighbor_within_range(sample_lattice):
    assert sample_lattice._Lattice3D__get_indices_nearest_neighbor(0.3, 0.4, 0.499) == (3, 4, 4)

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
    assert sample_lattice.get_value_by_index(9, 9, 9) == 15

def test_set_value_outside_range_raises_error(sample_lattice):
    with pytest.raises(ValueError):
        sample_lattice.set_value(2, 2, 2, 20)

def test_set_value_nearest_neighbor_within_range(sample_lattice):
    sample_lattice.set_value_nearest_neighbor(0.3, 0.4, 0.49, 7)
    assert sample_lattice.get_value_by_index(3, 4, 4) == 7

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
    sample_lattice.set_value_nearest_neighbor(0.3, 0.4, 0.5, 5)
    assert sample_lattice.get_value_nearest_neighbor(0.3, 0.4, 0.5) == 5

def test_get_value_nearest_neighbor_at_lower_bounds(sample_lattice):
    sample_lattice.set_value_nearest_neighbor(0, 0, 0, 10)
    assert sample_lattice.get_value_nearest_neighbor(0, 0, 0) == 10

def test_get_value_nearest_neighbor_at_upper_bounds(sample_lattice):
    sample_lattice.set_value_nearest_neighbor(0.9, 0.9, 0.9, 15)
    assert sample_lattice.get_value_nearest_neighbor(0.9, 0.9, 0.9) == 15

def test_get_value_nearest_neighbor_outside_range_returns_none(sample_lattice):
    with pytest.raises(ValueError):
        sample_lattice.get_value_nearest_neighbor(2, 2, 2)

def test_find_closest_index(sample_lattice):
    values = np.linspace(0, 1, 10)
    assert sample_lattice._Lattice3D__find_closest_index(0.3, values) == 3

def test_find_closest_index_for_outside_range(sample_lattice):
    values = np.linspace(0, 1, 10)
    assert sample_lattice._Lattice3D__find_closest_index(2, values) == 9

def test_within_range(sample_lattice):
    assert sample_lattice._Lattice3D__is_within_range(0.3, 0, 1) == True
    assert sample_lattice._Lattice3D__is_within_range(2, 0, 1) == False

def test_interpolate_value_nearest(sample_lattice):
    sample_lattice.set_value_nearest_neighbor(0.3, 0.4, 0.49, 5.)
    assert sample_lattice.interpolate_value(0.29, 0.42, 0.48) == 5.
    assert sample_lattice.interpolate_value(0.29, 0.42, 0.56) ==0.

def test_add(sample_lattice):
    lattice1 = sample_lattice
    lattice2 = copy.deepcopy(lattice1)
    lattice3 = copy.deepcopy(lattice1)
    lattice1.set_value(0.3, 0.4, 0.5, 5)
    lattice2.set_value(0.3, 0.4, 0.5, 10)
    lattice1.set_value(0.1, 0.2, 0.3, 2)
    lattice3.set_value(0.3, 0.4, 0.5, 15)
    lattice3.set_value(0.1, 0.2, 0.3, 2)
    lattice1=lattice1+lattice2
    assert((lattice1.grid_==lattice3.grid_).all())

def test_average(sample_lattice):
    lattice1 = sample_lattice
    lattice2 = copy.deepcopy(lattice1)
    lattice3 = copy.deepcopy(lattice1)
    lattice1.set_value(0.3, 0.4, 0.5, 5)
    lattice2.set_value(0.3, 0.4, 0.5, 10)
    lattice3.set_value(0.1, 0.1, 0.1, 3)
    lattice1=lattice1.average(lattice2, lattice3)
    assert(lattice1.get_value(0.3, 0.4, 0.5)==5)
    assert(lattice1.get_value(0.1, 0.1, 0.1)==1)

def test_csv(sample_lattice, tmp_path):
    lattice1 = sample_lattice
    lattice1.set_value(0.3, 0.4, 0.5, 5)
    lattice1.save_to_csv(tmp_path / 'test.csv')
    lattice2=Lattice3D.load_from_csv(tmp_path / 'test.csv')
    assert((lattice1.grid_==lattice2.grid_).all())

def test_extract_slice(sample_lattice):
    lattice1 = sample_lattice
    lattice1.set_value(0.25, 0.0, 0.0, 5)
    with pytest.raises(ValueError):
        lattice1.extract_slice( 2, 2)
    with pytest.raises(ValueError):
        lattice1.extract_slice( "eta", 2)
    with pytest.raises(ValueError):
        lattice1.extract_slice( "x", 15)
    slice_data, slice_values, slice_label= lattice1.extract_slice( "x", 2)
    compare_data=np.array([5.]+99*[0.])
    assert((slice_data.flatten()==compare_data).all())

def test_extract_slice_to_csv(sample_lattice, tmp_path):
    lattice1 = sample_lattice
    filename= tmp_path / 'test.csv'
    lattice1.set_value(0.25, 0.0, 0.0, 5)
    lattice1.save_slice_to_csv("x", 2, filename)
    assert filename.is_file()
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        expected_headers = ['#Slice in Y-Z Plane at X = 0.2222222222222222. Ymin: 0 Ymax: 1 ny: 10 Zmin: 0 Zmax: 1 nz:10 ']
        assert headers == expected_headers

        # Check the content of the file
        rows = [row[:3] for row in reader]
        rows = rows[:3] # test only important part of content
        expected_rows = [
            ['5.000000000000000000e+00','0.000000000000000000e+00','0.000000000000000000e+00'],
            ['0.000000000000000000e+00','0.000000000000000000e+00','0.000000000000000000e+00'],
            ['0.000000000000000000e+00','0.000000000000000000e+00','0.000000000000000000e+00']
        ]
        assert rows == expected_rows

def test_interpolate_to_lattice(sample_lattice):
    # Create a Lattice3D object
    lattice = sample_lattice

    # Assume set_value_by_index and interpolate_value methods are properly defined
    for i in range(10):
        for j in range(10):
            for k in range(10):
                lattice.set_value_by_index(i, j, k, i + j + k)

    # Call interpolate_to_lattice
    new_lattice = lattice.interpolate_to_lattice(20, 20, 20)

    # Check that the values in the new lattice are as expected
    for i in range(20):
        for j in range(20):
            for k in range(20):
                if(i%2==1 or j%2==1 or k%2==1):
                    continue
                posx, posy, posz=lattice.get_coordinates(int(i/2), int(j/2), int(k/2))
                assert np.isclose(new_lattice.get_value_by_index(i, j, k), lattice.interpolate_value(posx,posy,posz))
                