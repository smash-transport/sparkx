# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import numpy as np
import pytest
import csv
import copy
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
        hist = Histogram((0, 10, -1))  # num_bins <= 0


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
    assert np.allclose(
        hist.histogram(), np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    )


def test_add_value_list():
    # Test adding a list of numbers to the histogram
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    assert np.allclose(
        hist.histogram(), np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    )


def test_add_value_single_number_to_existing_histogram():
    # Test adding a single number to an existing histogram
    hist = Histogram((0, 10, 10))
    hist.add_value(4.5)
    hist.add_value(5.5)
    assert np.allclose(
        hist.histogram(), np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    )


def test_add_value_single_number_to_multiple_histograms():
    # Test adding a single number to multiple histograms
    hist = Histogram((0, 10, 10))
    hist.add_value(4.5)
    hist.add_histogram()
    hist.add_value(5.5)
    assert np.allclose(
        hist.histogram(),
        np.array(
            [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
        ),
    )


def test_add_value_out_of_range():
    # Test adding values inside and outside the histograms range
    hist = Histogram((0, 20, 20))
    valid_values = [1, 2, 10, 18]
    for value in valid_values:
        hist.add_value(value)

    hist.add_histogram()

    histogram_before_outliers = hist.histogram().copy()

    # Test that the histogram does not change for out-of-range values
    outlier_values = [-1, 21, 40, 20]
    for value in outlier_values:
        hist.add_value(value)

    histogram_after_outliers = hist.histogram()
    assert np.allclose(histogram_before_outliers, histogram_after_outliers)


def test_remove_bin_out_of_range():
    # Test removing a bin at an index out of range
    hist = Histogram((0, 10, 10))
    with pytest.raises(ValueError):
        hist.remove_bin(11)


def test_remove_bin_from_existing_histogram():
    # Test removing a bin from an existing histogram
    hist = Histogram((0, 10, 10))
    hist.add_value(4.5)
    hist.remove_bin(4)
    assert np.allclose(hist.histogram(), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))


def test_remove_bin_from_multiple_histograms():
    # Test removing a bin from multiple histograms
    hist = Histogram((0, 10, 10))
    hist.add_value(4.5)
    hist.add_histogram()
    hist.add_value(5.5)
    hist.remove_bin(4)
    assert np.allclose(
        hist.histogram(),
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]]),
    )


def test_add_bin():
    # Test adding a bin to the histogram
    hist = Histogram((0, 10, 10))
    hist.add_bin(6, 5.5)
    assert np.allclose(
        hist.bin_edges_, np.array([0, 1, 2, 3, 4, 5, 5.5, 6, 7, 8, 9, 10])
    )
    assert np.allclose(
        hist.histogram(), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_add_bin_index_out_of_range():
    # Test adding a bin at an index out of range
    hist = Histogram((0, 10, 10))
    with pytest.raises(ValueError):
        hist.add_bin(11, 11)


def test_add_bin_non_monotonic():
    # Test adding a bin that would make the bin edges non-monotonic
    hist = Histogram((0, 10, 10))
    with pytest.raises(ValueError):
        hist.add_bin(6, 4.5)


def test_add_bin_to_existing_histogram():
    # Test adding a bin to an existing histogram
    hist = Histogram((0, 10, 10))
    hist.add_value(4.5)
    hist.add_bin(6, 5.5)
    assert np.allclose(
        hist.histogram(), np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    )


def test_add_bin_to_multiple_histograms():
    # Test adding a bin to multiple histograms
    hist = Histogram((0, 10, 10))
    hist.add_value(4.5)
    hist.add_histogram()
    hist.add_value(5.6)
    hist.add_bin(6, 5.5)
    hist.add_value(5.6)
    assert np.allclose(
        hist.histogram(),
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            ]
        ),
    )


def test_add_value_with_weight():
    hist = Histogram((0, 10, 10))
    hist.add_value(4.5, weight=0.5)
    assert np.allclose(
        hist.histogram(), np.array([0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0])
    )


def test_add_value_with_weight_NaN():
    with pytest.raises(ValueError):
        hist = Histogram((0, 10, 10))
        hist.add_value(4.5, weight=np.nan)


def test_add_value_with_two_weights():
    with pytest.raises(ValueError):
        hist = Histogram((0, 10, 10))
        hist.add_value(4.5, weight=[1.0, 1.0])


def test_add_value_list_weigths():
    hist = Histogram((0, 10, 10))
    hist.add_value([4.5, 5.5], weight=[0.5, 0.5])
    assert np.allclose(
        hist.histogram(), np.array([0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0])
    )


def test_add_value_list_weigths_error_length():
    with pytest.raises(ValueError):
        hist = Histogram((0, 10, 10))
        hist.add_value([4.5, 5.5], weight=0.5)


def test_add_value_nan():
    with pytest.raises(ValueError):
        hist = Histogram((0, 10, 10))
        hist.add_value(np.nan)


def test_add_value_nan():
    with pytest.raises(ValueError):
        hist = Histogram((0, 10, 10))
        hist.add_value([1.0, 2.0, np.nan])


def test_add_value_invalid_input_type():
    with pytest.raises(TypeError):
        hist = Histogram((0, 10, 10))
        hist.add_value((1.0, 2.0, np.nan))


def test_make_density_no_histogram_available():
    # Test case when make_density is called without any histogram available
    hist = Histogram((0, 10, 10))
    with pytest.raises(ValueError):
        hist.make_density()


def test_make_density_zero_integral():
    # Set up Histogram object with a histogram having zero integral
    histogram = Histogram((0.0, 10.0, 10))
    histogram.histograms_ = [np.zeros(4)]  # Histogram with zero values
    histogram.number_of_histograms_ = 1

    # Ensure ValueError is raised when integral is zero
    with pytest.raises(ValueError):
        histogram.make_density()


def test_make_density():
    # Initialize Histogram object
    hist = Histogram((0.0, 10.0, 10))
    hist.add_value(4.5)
    hist.add_value(5.5)

    # Call make_density method
    hist.make_density()

    # Check if the result is a numpy array
    assert isinstance(hist.histograms_, np.ndarray)

    # Check if the result is normalized
    integral = np.sum(hist.histograms_ * hist.bin_width())
    assert integral == pytest.approx(1.0, abs=1e-5)


def test_make_density_with_two_histograms():
    # Initialize Histogram object
    hist = Histogram((0.0, 10.0, 10))
    hist.add_value(4.5)
    hist.add_value(5.5)

    hist.make_density()
    integral1 = np.sum(hist.histograms_ * hist.bin_width())
    assert integral1 == pytest.approx(1.0, abs=1e-5)

    # Add another histogram
    hist.add_histogram()
    hist.add_value(6.5)
    hist.add_value(9.0)
    hist.add_value(3.0)
    hist.make_density()

    # Check if the result is a numpy array
    assert isinstance(hist.histograms_[-1], np.ndarray)
    integral2 = np.sum(hist.histograms_[-1] * hist.bin_width())
    assert integral2 == pytest.approx(1.0, abs=1e-5)


def test_average():
    # Test averaging histograms
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist.add_histogram()
    hist.add_value([2, 4, 6, 8, 9])
    counts_summed = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    hist.average()
    assert np.allclose(
        hist.histogram(),
        np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]),
    )
    # Test if the error is the standard error
    assert np.allclose(
        hist.error_, np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
    )
    assert isinstance(hist.scaling_, np.ndarray)
    assert all(isinstance(i, np.ndarray) for i in hist.scaling_)
    assert np.allclose(counts_summed, hist.histogram_raw_count_)


def test_average_weighted():
    # Test averaging histograms with different weights
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist.add_histogram()
    hist.add_value([2, 4, 6, 8, 9])
    counts_summed = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    hist.average_weighted([0.3, 0.7])
    assert np.allclose(
        hist.histogram(),
        np.array([0, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.7]),
    )
    # Test if the error is the standard error
    assert np.allclose(
        hist.error_, np.array([0, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.7])
    )
    assert isinstance(hist.scaling_, np.ndarray)
    assert not any(isinstance(i, np.ndarray) for i in hist.scaling_)
    assert np.allclose(counts_summed, hist.histogram_raw_count_)


def test_average_weighted_by_error():
    # Test averaging histograms with weights determined by error
    hist = Histogram((0, 3, 3))
    hist.add_value(
        [
            0,
            0,
            0,
        ]
    )
    hist.error_[0] = np.array([1.0, 2.0, 3.0])
    hist.add_histogram()
    hist.add_value([1.0, 1.0, 2.0])
    hist.error_[1] = np.array([2.0, 2.0, 3.0])
    counts_summed = [3.0, 2.0, 1.0]
    hist.average_weighted_by_error()
    assert np.allclose(hist.histogram(), np.array([2.4, 1.0, 0.5]), atol=0.01)
    # Test if the error is the standard error
    assert np.allclose(
        hist.error_, np.array([0.89442719, 1.41421356, 2.12132034]), atol=0.01
    )
    assert isinstance(hist.scaling_, np.ndarray)
    assert all(isinstance(i, np.ndarray) for i in hist.scaling_)
    assert np.allclose(counts_summed, hist.histogram_raw_count_)


def test_average_weighted_by_error_no_error_set():
    # Test that an error is thrown if the error is not set
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist.add_histogram()
    hist.add_value([2, 4, 6, 8, 9])
    with pytest.raises(TypeError):
        hist.average_weighted_by_error()


def test_average_over_single_histogram():
    # Test averaging histograms
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist2 = copy.deepcopy(hist)
    hist.average()
    assert np.allclose(hist.histogram(), hist2.histogram())


def test_write_to_file(tmp_path):
    # Test writing histograms to a file
    hist = Histogram((0, 4, 4))
    hist.add_value([1, 2, 3])
    hist.statistical_error()
    hist_labels = [
        {
            "bin_center": "Bin Center",
            "bin_low": "Bin Low",
            "bin_high": "Bin High",
            "distribution": "Distribution",
            "stat_err+": "Stat Error+",
            "stat_err-": "Stat Error-",
            "sys_err+": "Sys Error+",
            "sys_err-": "Sys Error-",
        }
    ]
    filename = tmp_path / "test_histograms.csv"
    hist.write_to_file(filename, hist_labels)

    # Check if the file exists
    assert filename.is_file()

    # Read the file and verify its content
    with open(filename, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        expected_headers = [
            "Bin Center",
            "Bin Low",
            "Bin High",
            "Distribution",
            "Stat Error+",
            "Stat Error-",
            "Sys Error+",
            "Sys Error-",
        ]
        assert headers == expected_headers

        # Check the content of the file
        rows = [row for row in reader]
        rows = rows[:-1]  # neglect the last empty line
        expected_rows = [
            ["0.5", "0.0", "1.0", "0.0", "0.0", "0.0", "0.0", "0.0"],
            ["1.5", "1.0", "2.0", "1.0", "1.0", "1.0", "0.0", "0.0"],
            ["2.5", "2.0", "3.0", "1.0", "1.0", "1.0", "0.0", "0.0"],
            ["3.5", "3.0", "4.0", "1.0", "1.0", "1.0", "0.0", "0.0"],
        ]
        assert rows == expected_rows


def test_write_to_file_custom_columns(tmp_path):
    # Test writing histograms to a file
    hist = Histogram((0, 4, 4))
    hist.add_value([1, 2, 3])
    hist.statistical_error()
    hist_labels = [
        {
            "bin_center": "Bin Center",
            "bin_low": "Bin Low",
            "bin_high": "Bin High",
            "distribution": "Distribution",
            "stat_err+": "Stat Error+",
            "stat_err-": "Stat Error-",
            "sys_err+": "Sys Error+",
            "sys_err-": "Sys Error-",
        }
    ]
    columns = [
        "bin_center",
        "bin_low",
        "bin_high",
        "distribution",
        "stat_err+",
        "stat_err-",
    ]
    filename = tmp_path / "test_histograms.csv"
    hist.write_to_file(filename, hist_labels, columns=columns)

    # Check if the file exists
    assert filename.is_file()

    # Read the file and verify its content
    with open(filename, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        expected_headers = [
            "Bin Center",
            "Bin Low",
            "Bin High",
            "Distribution",
            "Stat Error+",
            "Stat Error-",
        ]
        assert headers == expected_headers

        # Check the content of the file
        rows = [row for row in reader]
        rows = rows[:-1]  # neglect the last empty line
        expected_rows = [
            ["0.5", "0.0", "1.0", "0.0", "0.0", "0.0"],
            ["1.5", "1.0", "2.0", "1.0", "1.0", "1.0"],
            ["2.5", "2.0", "3.0", "1.0", "1.0", "1.0"],
            ["3.5", "3.0", "4.0", "1.0", "1.0", "1.0"],
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
    assert np.allclose(
        hist.histogram(),
        np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]),
    )


def test_scale_histogram_single_factor():
    # Test scaling histogram with a single factor
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist.scale_histogram(2)
    assert np.allclose(
        hist.histogram(), np.array([0, 2, 0, 2, 0, 2, 0, 2, 0, 2])
    )


def test_scale_histogram_multiple_factors():
    # Test scaling histogram with multiple factors
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 3, 5, 7, 9])
    hist.add_histogram()
    hist.add_value([2, 4, 6, 8, 9])

    # Passing incorrect length of scaling list
    with pytest.raises(ValueError):
        hist.scale_histogram([2, 0.5])

    # Scale histogram with valid scaling list
    hist.scale_histogram([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert np.allclose(
        hist.histogram(),
        np.array(
            [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 3, 0, 5, 0, 7, 0, 9, 10]]
        ),
    )


def test_scale_histogram_after_averaging():
    hist = Histogram((0, 10, 10))
    hist.add_value([1, 1, 1, 1, 3, 5, 7, 9])
    hist.add_histogram()
    hist.add_value([2, 4, 4.5, 6, 8, 8, 9])

    # Scale the averaged histogram
    hist.average()
    hist.scale_histogram(2.0)

    expected_histogram = np.array(
        [[0.0, 4.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0]]
    )

    assert hist.histogram().shape == expected_histogram.shape
    assert np.allclose(hist.histogram(), expected_histogram)
