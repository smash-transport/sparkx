# ===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

from sparkx.MultiParticlePtCorrelations import MultiParticlePtCorrelations
from sparkx.Oscar import Oscar
from sparkx.Particle import Particle
import pytest
import os
import numpy as np

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")
TEST_OSCAR = os.path.join(TEST_FILES_DIR, "particle_lists_extended.oscar")


@pytest.fixture
def mpc_instance():
    return MultiParticlePtCorrelations(max_order=8)


def test_MultiParticlePtCorrelations_initialization():
    with pytest.raises(ValueError):
        MultiParticlePtCorrelations(max_order=9)
    with pytest.raises(ValueError):
        MultiParticlePtCorrelations(max_order=0)
    with pytest.raises(TypeError):
        MultiParticlePtCorrelations(max_order="9")


def test_MultiParticlePtCorrelations_mean_pt_correlations(mpc_instance):
    particle_list = Oscar(TEST_OSCAR).particle_objects_list()

    compute_error = True
    delete_fraction = 0.2
    number_samples = 100
    seed = 42

    with pytest.raises(TypeError):
        mpc_instance.mean_pt_correlations(
            particle_list_all_events=particle_list,
            compute_error="true",
            delete_fraction=delete_fraction,
            number_samples=number_samples,
            seed=seed,
        )
    with pytest.raises(TypeError):
        mpc_instance.mean_pt_correlations(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction="0.2",
            number_samples=number_samples,
            seed=seed,
        )
    with pytest.raises(TypeError):
        mpc_instance.mean_pt_correlations(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction=delete_fraction,
            number_samples="5",
            seed=seed,
        )
    with pytest.raises(ValueError):
        mpc_instance.mean_pt_correlations(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction=2.0,
            number_samples=number_samples,
            seed=seed,
        )
    with pytest.raises(ValueError):
        mpc_instance.mean_pt_correlations(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction=-1.0,
            number_samples=number_samples,
            seed=seed,
        )
    with pytest.raises(ValueError):
        mpc_instance.mean_pt_correlations(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction=delete_fraction,
            number_samples=0,
            seed=seed,
        )

    corr, corr_err = mpc_instance.mean_pt_correlations(
        particle_list_all_events=particle_list,
        compute_error=compute_error,
        delete_fraction=delete_fraction,
        number_samples=number_samples,
        seed=seed,
    )
    assert len(corr) == 8
    assert len(corr_err) == 8
    # go through the values and check that there are no NaNs
    assert not np.isnan(corr).any()
    assert not np.isnan(corr_err).any()


# similar test for mean_pt_cumulants


def test_MultiParticlePtCorrelations_mean_pt_cumulants(mpc_instance):
    particle_list = Oscar(TEST_OSCAR).particle_objects_list()

    compute_error = True
    delete_fraction = 0.2
    number_samples = 100
    seed = 42

    with pytest.raises(TypeError):
        mpc_instance.mean_pt_cumulants(
            particle_list_all_events=particle_list,
            compute_error="true",
            delete_fraction=delete_fraction,
            number_samples=number_samples,
            seed=seed,
        )
    with pytest.raises(TypeError):
        mpc_instance.mean_pt_cumulants(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction="0.2",
            number_samples=number_samples,
            seed=seed,
        )
    with pytest.raises(TypeError):
        mpc_instance.mean_pt_cumulants(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction=delete_fraction,
            number_samples="5",
            seed=seed,
        )
    with pytest.raises(ValueError):
        mpc_instance.mean_pt_cumulants(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction=2.0,
            number_samples=number_samples,
            seed=seed,
        )
    with pytest.raises(ValueError):
        mpc_instance.mean_pt_cumulants(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction=-1.0,
            number_samples=number_samples,
            seed=seed,
        )
    with pytest.raises(ValueError):
        mpc_instance.mean_pt_cumulants(
            particle_list_all_events=particle_list,
            compute_error=compute_error,
            delete_fraction=delete_fraction,
            number_samples=0,
            seed=seed,
        )

    cumulants, cumulants_err = mpc_instance.mean_pt_cumulants(
        particle_list_all_events=particle_list,
        compute_error=compute_error,
        delete_fraction=delete_fraction,
        number_samples=number_samples,
        seed=seed,
    )
    assert len(cumulants) == 8
    assert len(cumulants_err) == 8
    # go through the values and check that there are no NaNs
    assert not np.isnan(cumulants).any()
    assert not np.isnan(cumulants_err).any()


def test_MultiParticlePtCorrelations_mean_pt_correlations_and_cumulants(mpc_instance):
    # the first order must be the same for the correlations and cumulants

    particle_list = Oscar(TEST_OSCAR).particle_objects_list()

    compute_error = True
    delete_fraction = 0.2
    number_samples = 100
    seed = 42

    corr, corr_err = mpc_instance.mean_pt_correlations(
        particle_list_all_events=particle_list,
        compute_error=compute_error,
        delete_fraction=delete_fraction,
        number_samples=number_samples,
        seed=seed,
    )
    cumulants, cumulants_err = mpc_instance.mean_pt_cumulants(
        particle_list_all_events=particle_list,
        compute_error=compute_error,
        delete_fraction=delete_fraction,
        number_samples=number_samples,
        seed=seed,
    )
    assert corr[0] == cumulants[0]
    assert corr_err[0] == cumulants_err[0]


def test_MultiParticlePtCorrelations_mean_pt_correlations_and_cumulants_physics(
    mpc_instance,
):
    # test with a dummy particle list, where all particles have the same pt
    particle_list = []
    for _ in range(10):  # create 10 events
        event_particles = []
        for _ in range(100):  # create 100 particles with the same pt
            p_array = np.array(
                [0, 2114, 11, 2.01351754, 1.30688601, -0.422958786, -0.512249773]
            )
            p = Particle(input_format="JETSCAPE", particle_array=p_array)
            event_particles.append(p)
        particle_list.append(event_particles)

    compute_error = True
    delete_fraction = 0.2
    number_samples = 100
    seed = 42

    corr, corr_err = mpc_instance.mean_pt_correlations(
        particle_list_all_events=particle_list,
        compute_error=compute_error,
        delete_fraction=delete_fraction,
        number_samples=number_samples,
        seed=seed,
    )
    cumulants, cumulants_err = mpc_instance.mean_pt_cumulants(
        particle_list_all_events=particle_list,
        compute_error=compute_error,
        delete_fraction=delete_fraction,
        number_samples=number_samples,
        seed=seed,
    )
    assert corr[0] == cumulants[0]
    assert corr_err[0] == cumulants_err[0]

    assert not np.isnan(corr).any()
    assert not np.isnan(corr_err).any()

    assert not np.isnan(cumulants).any()
    assert not np.isnan(cumulants_err).any()

    # if there is no variation in pt, the uncertainty should be zero
    # check that the uncertainty is smaller 1e-10
    assert np.all(corr_err < 1e-10)
    assert np.all(cumulants_err < 1e-10)
