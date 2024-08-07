# ===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from sparkx.Jackknife import Jackknife
import numpy as np
import pytest


def test_Jackknife_initialization():
    delete_fraction = 0.1
    number_samples = 10
    seed = 42

    jackknife = Jackknife(delete_fraction, number_samples, seed)

    assert jackknife.delete_fraction == delete_fraction
    assert jackknife.number_samples == number_samples
    assert jackknife.seed == seed

    # Test for ValueError on invalid delete_fraction and number_samples
    with pytest.raises(ValueError):
        Jackknife(-0.1, number_samples)
    with pytest.raises(ValueError):
        Jackknife(1.0, number_samples)
    with pytest.raises(ValueError):
        Jackknife(delete_fraction, 0)
    with pytest.raises(TypeError):
        Jackknife("0.1", number_samples)
    with pytest.raises(TypeError):
        Jackknife(delete_fraction, "10")
    with pytest.raises(TypeError):
        Jackknife(delete_fraction, number_samples, "42")


def generate_data(n, mean=0, std=1):
    # set the seed for reproducibility
    np.random.seed(42)
    return np.random.normal(mean, std, n)


def generate_2d_data(n, mean=0, std=1):
    # set the seed for reproducibility
    np.random.seed(42)
    return np.random.normal(mean, std, (n, 2))


def test_Jackknife_random_deletion():
    data = generate_data(100)
    delete_fraction = 0.5
    number_samples = 10
    seed = 42

    jackknife = Jackknife(delete_fraction, number_samples, seed)
    reduced_data = jackknife._randomly_delete_data(data)

    assert len(reduced_data) == int(len(data) * (1 - delete_fraction))
    elements_in_data = np.isin(data, reduced_data)
    # check the number of False in the boolean array
    assert np.sum(elements_in_data) == int(len(data) * delete_fraction)

    # Test for 2D data
    data_2d = generate_2d_data(100)
    reduced_data_2d = jackknife._randomly_delete_data(data_2d)
    assert reduced_data_2d.shape[0] == int(data_2d.shape[0] * (1 - delete_fraction))
    assert reduced_data_2d.shape[1] == data_2d.shape[1]


def test_Jackknife_apply_function():
    data = generate_data(100)
    delete_fraction = 0.1
    number_samples = 10
    seed = 42

    jackknife = Jackknife(delete_fraction, number_samples, seed)
    reduced_data = jackknife._randomly_delete_data(data)
    result = jackknife._apply_function_to_reduced_data(reduced_data, np.mean)

    assert isinstance(result, float)
    assert np.isclose(result, np.mean(reduced_data), atol=1e-6)


def test_Jackknife_single_sample():
    data = generate_data(100)
    delete_fraction = 0.1
    number_samples = 10
    seed = 42

    jackknife = Jackknife(delete_fraction, number_samples, seed)
    result = jackknife._compute_one_jackknife_sample(data, np.mean)

    assert isinstance(result, float)


def test_Jackknife_compute_jackknife_estimates():
    # generate data
    data = generate_data(100000)

    std_err_mean_data = np.std(data) / np.sqrt(len(data))

    delete_fraction = 0.5
    number_samples = 1000
    seed = 42

    jackknife = Jackknife(delete_fraction, number_samples, seed)
    std_jackknife = jackknife.compute_jackknife_estimates(data)
    assert np.isclose(std_jackknife, std_err_mean_data, atol=0.01)

    std_var_jackknife = jackknife.compute_jackknife_estimates(data, function=np.std)
    assert np.isclose(
        std_var_jackknife,
        np.std(data, ddof=1) / np.sqrt(2 * (len(data) - 1)),
        atol=0.01,
    )

    # Test for TypeError on invalid function output (function returns list
    # instead of float or int)
    with pytest.raises(TypeError):
        jackknife.compute_jackknife_estimates(data, function=lambda x: [1, 2])

    delete_fraction = 0.0000001
    jackknife1 = Jackknife(delete_fraction, number_samples, seed)
    with pytest.raises(ValueError):
        jackknife1.compute_jackknife_estimates(data)

    with pytest.raises(TypeError):
        jackknife.compute_jackknife_estimates([1, 2, 3])

    with pytest.raises(TypeError):
        jackknife.compute_jackknife_estimates(data, function=1)


def test_Jackknife_single_vs_two_cores():
    data = generate_data(100)
    delete_fraction = 0.5
    number_samples = 1000
    seed = 42

    jackknife = Jackknife(delete_fraction, number_samples, seed)
    std_single_core = jackknife.compute_jackknife_estimates(data, num_cores=1)
    std_two_cores = jackknife.compute_jackknife_estimates(data, num_cores=2)

    assert np.isclose(std_single_core, std_two_cores, atol=1e-6)


def custom_mean_function(input_data):
    return np.mean(input_data)


def test_Jacknife_2d_array_input():
    data = generate_2d_data(100)
    delete_fraction = 0.5
    number_samples = 1000
    seed = 42

    jackknife = Jackknife(delete_fraction, number_samples, seed)

    std = jackknife.compute_jackknife_estimates(data, function=custom_mean_function)
    assert isinstance(std, float)


def weighted_mean(data, weight):
    return np.mean(data) * weight


def test_jackknife_with_additional_argument():
    data = np.random.normal(0, 1, 100)
    delete_fraction = 0.2
    number_samples = 50
    seed = 42

    jackknife = Jackknife(delete_fraction, number_samples, seed)
    weight = 2.0
    jackknife_std_err = jackknife.compute_jackknife_estimates(
        data, function=weighted_mean, weight=weight
    )

    # this is just testing that the propagation of parameters for the function
    # works
    assert isinstance(jackknife_std_err, float)
