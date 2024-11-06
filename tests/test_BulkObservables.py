#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================

from sparkx.Particle import Particle
from sparkx.BulkObservables import BulkObservables
from sparkx.Oscar import Oscar

import os
import numpy as np
import pytest

@pytest.fixture
def oscar_extended_file_path():
    # Assuming your test file is in the same directory as test_files/
    return os.path.join(
        os.path.dirname(__file__), "test_files", "particle_lists_extended.oscar"
    )


def test_BulkObservables_valid_initialization(oscar_extended_file_path):
    oscar_extended = Oscar(oscar_extended_file_path)
    extended_particle_objects_list = oscar_extended.particle_objects_list()

    assert BulkObservables(extended_particle_objects_list)


def test_dNdy_invalid_input(oscar_extended_file_path):
    oscar_extended = Oscar(oscar_extended_file_path)
    extended_particle_objects_list = oscar_extended.particle_objects_list()

    bulk_obs = BulkObservables(extended_particle_objects_list)

    with pytest.raises(TypeError):
        bulk_obs.dNdy(1.2)

    with pytest.raises(TypeError):
        bulk_obs.dNdy("abc")

    with pytest.raises(ValueError):
        bulk_obs.dNdy((1.1, 2.2, 3.3, 4.4))

    with pytest.raises(ValueError):
        bulk_obs.dNdy((1.1, 2.2))

    with pytest.raises(ValueError):
        bulk_obs.dNdy((1.1, 2.2, 3.3))


def test_dNdy_valid(oscar_extended_file_path):
    # store the y values and the counts in that bin in two events to create 
    # particles accordingly    
    y_bins_and_counts = [
        [[-1.9, 5], [-0.6, 4], [-0.1, 3], [1.4, 3]],
        [[-1.7, 2], [0.4, 6],  [1.2, 5]],
        [[-0.9, 7], [0.8, 2],  [1.6, 1]]
    ]

    dNdy_expected = [
        7 / 3 / 0.5,   # Bin -2.0 to -1.5
        0,             # Bin -1.5 to -1.0
        11 / 3 / 0.5,  # Bin -1.0 to -0.5
        3 / 3 / 0.5,   # Bin -0.5 to 0.0
        6 / 3 / 0.5,   # Bin 0.0 to 0.5
        2 / 3 / 0.5,   # Bin 0.5 to 1.0
        8 / 3 / 0.5,   # Bin 1.0 to 1.5
        1 / 3 / 0.5    # Bin 1.5 to 2.0
    ]

    particle_objects_list = []

    for event in y_bins_and_counts:
        particle_objects_single_event = []

        for y_count_pairs in event:
            y = y_count_pairs[0]
            counts = y_count_pairs[1]

            # create particles according to counts
            for i in range(counts):
                p = Particle()
                # Given a p_z, the energy that will result in a
                # rapidity y, is given E = coth(y) * p_z
                p_z = np.random.random() * 200 - 100
                e = (1/np.tanh(y)) * p_z

                p.E = e
                p.pz = p_z

                particle_objects_single_event.append(p)

        particle_objects_list.append(particle_objects_single_event)

    bulk_obs = BulkObservables(particle_objects_list)

    # Call with tupel
    dNdy_tupel = bulk_obs.dNdy((-2, 2, 8)).histogram()[0]
    assert dNdy_tupel  == pytest.approx(dNdy_expected, rel=1e-6)
    
    # Call with bin list
    bins = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    dNdy_list = bulk_obs.dNdy(bins).histogram()[0]
    assert dNdy_list  == pytest.approx(dNdy_expected, rel=1e-6) 


def test_dNdpT_invalid_input(oscar_extended_file_path):
    oscar_extended = Oscar(oscar_extended_file_path)
    extended_particle_objects_list = oscar_extended.particle_objects_list()

    bulk_obs = BulkObservables(extended_particle_objects_list)

    with pytest.raises(TypeError):
        bulk_obs.dNdpT(1.2)

    with pytest.raises(TypeError):
        bulk_obs.dNdpT("abc")

    with pytest.raises(ValueError):
        bulk_obs.dNdpT((1.1, 2.2, 3.3, 4.4))

    with pytest.raises(ValueError):
        bulk_obs.dNdpT((1.1, 2.2))

    with pytest.raises(ValueError):
        bulk_obs.dNdpT((1.1, 2.2, 3.3))


def test_dNdpT_valid(oscar_extended_file_path):
    pT_bins_and_counts = [
    [[0.1, 8], [0.3, 1], [0.6, 5], [1.1, 10], [1.8, 4]],     # Event 1
    [[0.2, 6], [0.7, 2], [1.3, 7], [1.6, 3], [1.9, 9]],      # Event 2
    [[0.15, 9], [0.55, 3], [0.9, 1], [1.45, 6], [1.74, 2]],  # Event 3
    [[0.24, 4], [0.85, 5], [1.35, 8], [1.52, 2], [1.95, 6]]   # Event 4
    ]

    dNdpT_expected = [
    (8 + 6 + 9 + 4) / 4 / 0.25,   # Bin 0.0 to 0.25
    1 / 4 / 0.25,                 # Bin 0.25 to 0.5
    (2 + 3 + 5) / 4 / 0.25,       # Bin 0.5 to 0.75
    (5 + 1) / 4 / 0.25,           # Bin 0.75 to 1.0
    10 / 4 / 0.25,                # Bin 1.0 to 1.25
    (7 + 8 + 6) / 4 / 0.25,       # Bin 1.25 to 1.5
    (3 + 2 + 2) / 4 / 0.25,       # Bin 1.5 to 1.75
    (4 + 9 + 6) / 4 / 0.25        # Bin 1.75 to 2.0
    ]

    particle_objects_list = []

    for event in pT_bins_and_counts:
        particle_objects_single_event = []

        for pT_count_pairs in event:
            pT = pT_count_pairs[0]
            counts = pT_count_pairs[1]

            # create particles according to counts
            for i in range(counts):
                p = Particle()
                # Given a pT, we can sample px and py according to
                # px = sqrt(pT^2 - py^2)
                p_y = np.random.random() * 2 * pT - pT
                p_x = np.sqrt(pT**2 - p_y**2)

                p.px = p_x
                p.py = p_y

                particle_objects_single_event.append(p)

        particle_objects_list.append(particle_objects_single_event)

    bulk_obs = BulkObservables(particle_objects_list)

    # Call with tupel
    dNdpT_tupel = bulk_obs.dNdpT((0, 2, 8)).histogram()[0]
    assert dNdpT_tupel  == pytest.approx(dNdpT_expected, rel=1e-6)

    # Call with bin list
    bins = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    dNdpT_list = bulk_obs.dNdpT(bins).histogram()[0]
    assert dNdpT_list  == pytest.approx(dNdpT_expected, rel=1e-6)
