import numpy as np
import pytest
import csv
from sparkx.Histogram import Histogram

def test_histogram_creation_with_tuple():
    # Test histogram creation with a tuple
    hist = Histogram((0, 10, 10))
    assert hist.number_of_bins_ == 10
    assert np.allclose(hist.bin_edges_, np.linspace(0, 10, num=11))

def test_histogram_creation_with_list():
    # Test histogram creation with a list
    hist = Histogram([0, 2, 4, 6, 8, 10])
    assert hist.number_of_bins_ == 5
    assert np.allclose(hist.bin_edges_, np.array([0, 2, 4, 6, 8, 10]))

def test_histogram_creation_with_invalid_input():
    # Test histogram creation with invalid input
    with pytest.raises(TypeError):
        # Passing incorrect input type
        hist = Histogram("invalid_input")

    with pytest.raises(ValueError):
        # Passing tuple with invalid values
        hist = Histogram((10, 0, 10))  # hist_min > hist_max
    
    with pytest.raises(ValueError):
        # Passing tuple with non-integer number of bins
        hist = Histogram((0, 10, 1.4))
    
    with pytest.raises(ValueError):
        hist = Histogram((0, 10, -1)) # num_bins <= 0

def test_set_error_with_invalid_input():
    # Test setting error with invalid input
    hist = Histogram((0, 10, 10))
    with pytest.raises(ValueError):
        # Passing incorrect length of error list
        hist.set_error([1, 2, 3])

def test_set_systematic_error_with_invalid_input():
    # Test setting systematic error with invalid input
    hist = Histogram((0, 10, 10))
    with pytest.raises(ValueError):
        # Passing incorrect length of systematic error list
        hist.set_systematic_error([1, 2, 3])

def test_add_value_single_number():
    # Test adding a single number to the histogram
    hist = Histogram((0, 10, 10))
    hist.add_value(4.5)
    assert np.allclose(hist.histogram(), np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))

def test_add_value_list():
    # Test adding a list of numbers to the histogram
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    assert np.allclose(hist.histogram(), np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))

def test_average():
    # Test averaging histograms
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist.add_histogram()
    hist.add_value([2, 4, 6, 8, 9])
    hist.average()
    assert np.allclose(hist.histogram(), np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]))

def test_write_to_file(tmp_path):
    # Test writing histograms to a file
    hist = Histogram((0, 4, 4))
    hist.add_value([1, 2, 3])
    hist.statistical_error()
    hist_labels = [{'bin_center': 'Bin Center', 'bin_low': 'Bin Low', 'bin_high': 'Bin High',
                    'distribution': 'Distribution', 'stat_err+': 'Stat Error+', 'stat_err-': 'Stat Error-',
                    'sys_err+': 'Sys Error+', 'sys_err-': 'Sys Error-'}]
    filename = tmp_path / "test_histograms.csv"
    hist.write_to_file(filename, hist_labels)

    # Check if the file exists
    assert filename.is_file()

    # Read the file and verify its content
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        expected_headers = ['Bin Center', 'Bin Low', 'Bin High', 'Distribution', 
                            'Stat Error+', 'Stat Error-', 'Sys Error+', 'Sys Error-']
        assert headers == expected_headers

        # Check the content of the file
        rows = [row for row in reader]
        rows = rows[:-1] # neglect the last empty line
        expected_rows = [
            ['0.5', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0'],
            ['1.5', '1.0', '2.0', '1.0', '1.0', '1.0', '0.0', '0.0'],
            ['2.5', '2.0', '3.0', '1.0', '1.0', '1.0', '0.0', '0.0'],
            ['3.5', '3.0', '4.0', '1.0', '1.0', '1.0', '0.0', '0.0']
        ]
        assert rows == expected_rows

def test_write_to_file_custom_columns(tmp_path):
    # Test writing histograms to a file
    hist = Histogram((0, 4, 4))
    hist.add_value([1, 2, 3])
    hist.statistical_error()
    hist_labels = [{'bin_center': 'Bin Center', 'bin_low': 'Bin Low', 'bin_high': 'Bin High',
                    'distribution': 'Distribution', 'stat_err+': 'Stat Error+', 'stat_err-': 'Stat Error-',
                    'sys_err+': 'Sys Error+', 'sys_err-': 'Sys Error-'}]
    columns = ['bin_center', 'bin_low', 'bin_high', 'distribution', 'stat_err+', 'stat_err-']
    filename = tmp_path / "test_histograms.csv"
    hist.write_to_file(filename, hist_labels, columns=columns)

    # Check if the file exists
    assert filename.is_file()

    # Read the file and verify its content
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        expected_headers = ['Bin Center', 'Bin Low', 'Bin High', 'Distribution', 
                            'Stat Error+', 'Stat Error-']
        assert headers == expected_headers

        # Check the content of the file
        rows = [row for row in reader]
        rows = rows[:-1] # neglect the last empty line
        expected_rows = [
            ['0.5', '0.0', '1.0', '0.0', '0.0', '0.0'],
            ['1.5', '1.0', '2.0', '1.0', '1.0', '1.0'],
            ['2.5', '2.0', '3.0', '1.0', '1.0', '1.0'],
            ['3.5', '3.0', '4.0', '1.0', '1.0', '1.0']
        ]
        assert rows == expected_rows

def test_average_weighted():
    # Test weighted averaging histograms
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist.add_histogram()
    hist.add_value([2, 4, 6, 8, 9])
    weights = np.array([0.5, 0.5])
    hist.average_weighted(weights)
    assert np.allclose(hist.histogram(), np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]))

def test_scale_histogram_single_factor():
    # Test scaling histogram with a single factor
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist.scale_histogram(2)
    assert np.allclose(hist.histogram(), np.array([0, 2, 0, 2, 0, 2, 0, 2, 0, 2]))

def test_scale_histogram_multiple_factors():
    # Test scaling histogram with multiple factors
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist.add_histogram()
    hist.add_value([2, 4, 6, 8, 9])
    with pytest.raises(ValueError):
        # Passing incorrect length of scaling list
        hist.scale_histogram([2, 0.5])
    hist.scale_histogram([2,2,2,2,2,2,2,2,2,2])
    assert np.allclose(hist.histogram(), np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 2, 0, 2, 0, 2, 0, 2, 2]]))
