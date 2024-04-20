#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
import numpy as np
import copy
import pytest
from sparkx.Filter import *
from sparkx.Particle import Particle

@pytest.fixture
def particle_nan_quantities():
    particle_list = [[Particle() for _ in range(10)]]
    return particle_list

@pytest.fixture
def particle_list_charged_uncharged():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.charge = 1
        particle_list.append(p)
    for i in range(5):
        p = Particle()
        p.charge = 0
        particle_list.append(p)
    return [particle_list]

def test_charged_particles(particle_nan_quantities,particle_list_charged_uncharged):
    return_list = charged_particles(particle_nan_quantities)
    assert(len(return_list[0]) == 0)

    return_list = charged_particles(particle_list_charged_uncharged)
    assert(len(return_list[0]) == 5)

def test_uncharged_particles(particle_nan_quantities,particle_list_charged_uncharged):
    return_list = uncharged_particles(particle_nan_quantities)
    assert(len(return_list[0]) == 0)

    return_list = uncharged_particles(particle_list_charged_uncharged)
    assert(len(return_list[0]) == 5)

@pytest.fixture
def particle_list_strange():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.pdg = 321
        particle_list.append(p)
    for i in range(5):
        p = Particle()
        p.pdg = 211
        particle_list.append(p)
    return [particle_list]

def test_strange_particles(particle_nan_quantities,particle_list_strange):
    return_list = strange_particles(particle_nan_quantities)
    assert(len(return_list[0]) == 0)

    return_list = strange_particles(particle_list_strange)
    assert(len(return_list[0]) == 5)

def test_particle_species_filter(particle_list_strange): 
    return_list = particle_species(copy.deepcopy(particle_list_strange),'321')
    assert(len(return_list[0]) == 5)
    
    return_list = particle_species(copy.deepcopy(particle_list_strange),211)
    assert(len(return_list[0]) == 5)

    return_list = particle_species(copy.deepcopy(particle_list_strange),321.0)
    assert(len(return_list[0]) == 5)

    return_list = particle_species(copy.deepcopy(particle_list_strange),[211,321])
    assert(len(return_list[0]) == 10)

    return_list = particle_species(copy.deepcopy(particle_list_strange),np.array([211,321]))
    assert(len(return_list[0]) == 10)

    return_list = particle_species(copy.deepcopy(particle_list_strange),(211,321))
    assert(len(return_list[0]) == 10)

    with pytest.raises(ValueError):
        return_list = particle_species(particle_list_strange,np.nan)
    
    with pytest.raises(ValueError):
        return_list = particle_species(particle_list_strange,[np.nan,211])


def test_remove_particle_species_filter(particle_list_strange):
    return_list = remove_particle_species(copy.deepcopy(particle_list_strange),'321')
    assert(len(return_list[0]) == 5)
    
    return_list = remove_particle_species(copy.deepcopy(particle_list_strange),321)
    assert(len(return_list[0]) == 5)

    return_list = remove_particle_species(copy.deepcopy(particle_list_strange),321.0)
    assert(len(return_list[0]) == 5)

    return_list = remove_particle_species(copy.deepcopy(particle_list_strange),[211,321])
    assert(len(return_list[0]) == 0)

    return_list = remove_particle_species(copy.deepcopy(particle_list_strange),np.array([211,321]))
    assert(len(return_list[0]) == 0)

    return_list = remove_particle_species(copy.deepcopy(particle_list_strange),(211,321))
    assert(len(return_list[0]) == 0)

    with pytest.raises(ValueError):
        return_list = remove_particle_species(particle_list_strange,np.nan)
    
    with pytest.raises(ValueError):
        return_list = remove_particle_species(particle_list_strange,[np.nan,211])

@pytest.fixture
def particle_list_ncoll():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.ncoll = i
        particle_list.append(p)
    return [particle_list]

def test_participants(particle_nan_quantities,particle_list_ncoll):
    return_list = participants(particle_nan_quantities)
    assert(len(return_list[0]) == 0)

    return_list = participants(particle_list_ncoll)
    assert(len(return_list[0]) == 4)

def test_spectators(particle_nan_quantities,particle_list_ncoll):
    return_list = spectators(particle_nan_quantities)
    assert(len(return_list[0]) == 0)

    return_list = spectators(particle_list_ncoll)
    assert(len(return_list[0]) == 1)

@pytest.fixture
def particle_list_energies():
    particle_list1 = []
    for i in range(5):
        p = Particle()
        p.E = 1.0
        particle_list1.append(p)
    particle_list2 = []
    for i in range(5):
        p = Particle()
        p.E = 2.0
        particle_list2.append(p)
    final_list = [particle_list1]
    final_list.extend([particle_list2])
    return final_list

def test_lower_event_energy_cut(particle_nan_quantities,particle_list_energies):
    return_list = lower_event_energy_cut(particle_nan_quantities, 1.0)
    assert(len(return_list[0]) == 0)

    with pytest.raises(ValueError):
        lower_event_energy_cut(particle_nan_quantities, -1.)
    
    with pytest.raises(ValueError):
        lower_event_energy_cut(particle_nan_quantities, np.nan)

    with pytest.raises(TypeError):
        lower_event_energy_cut(particle_nan_quantities, '1.0')

    with pytest.raises(ValueError):
        return_list = lower_event_energy_cut(particle_list_energies, -1.0)
    
    return_list = lower_event_energy_cut(particle_list_energies, 8.0)
    assert(len(return_list[0]) == 5)
    assert(len(return_list) == 1)

@pytest.fixture
def particle_list_positions():
    particle_list = []
    p1 = Particle()
    p1.t = 1.
    p1.x = p1.y = p1.z = 0.
    particle_list.append(p1)

    p2 = Particle()
    p2.x = 1.
    p2.t = p2.y = p2.z = 0.
    particle_list.append(p2)

    p3 = Particle()
    p3.y = 1.
    p3.t = p3.x = p3.z = 0.
    particle_list.append(p3)

    p4 = Particle()
    p4.z = 1.
    p4.t = p4.x = p4.y = 0.
    particle_list.append(p4)

    return [particle_list]


def test_spacetime_cut(particle_list_positions):
    test_cases = [
        # Test cases for valid input
        ('t', (0.5, 1.5), [[particle_list_positions[0][0]]]),
        ('t', (1.5, 2.5), [[]]),
        ('x', (-0.5, 0.5), [[particle_list_positions[0][0], particle_list_positions[0][2], particle_list_positions[0][3]]]),
        ('y', (0.5, None), [[particle_list_positions[0][2]]]),
        ('z', (None, 0.5), [[particle_list_positions[0][0], particle_list_positions[0][1], particle_list_positions[0][2]]]),

        # Test cases for error conditions
        ('t', (None, None), ValueError),
        ('t', (1.5, 0.5), ValueError),
        ('t', (0.5,), TypeError),
        ('t', ('a', 1.5), ValueError),
        ('w', (0.5, 1.5), ValueError),
        ('x', (1.5, 0.5), ValueError),
    ]

    for dim, cut_value_tuple, expected_result in test_cases:
        if isinstance(expected_result, type) and issubclass(expected_result, Exception):
            # If expected_result is an Exception, we expect an error to be raised
            with pytest.raises(expected_result):
                spacetime_cut(particle_list_positions, dim, cut_value_tuple)
        else:
            # Apply the spacetime cut
            result = spacetime_cut(particle_list_positions, dim, cut_value_tuple)
            # Assert the result matches the expected outcome
            assert result == expected_result

@pytest.fixture
def particle_list_pt():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.px = i
        p.py = 0.
        particle_list.append(p)
    return [particle_list]

def test_pt_cut(particle_list_pt):
    test_cases = [
        # Test cases for valid input
        ((0.5, 1.5), [[particle_list_pt[0][1]]]),
        ((2.5, None), [[particle_list_pt[0][3], particle_list_pt[0][4]]]),
        ((None, 3.5), [[particle_list_pt[0][0], particle_list_pt[0][1], particle_list_pt[0][2], particle_list_pt[0][3]]]),

        # Test cases for error conditions
        ((None, None), ValueError),
        ((-1, 3), ValueError),
        (('a', 3), ValueError),
        ((3, 2), ValueError),
        ((None, None, None), TypeError),
    ]

    for cut_value_tuple, expected_result in test_cases:
        if isinstance(expected_result, type) and issubclass(expected_result, Exception):
            # If expected_result is an Exception, we expect an error to be raised
            with pytest.raises(expected_result):
                pt_cut(particle_list_pt, cut_value_tuple)
        else:
            # Apply the pt_cut
            result = pt_cut(particle_list_pt, cut_value_tuple)
            # Assert the result matches the expected outcome
            assert result == expected_result

@pytest.fixture
def particle_list_momentum_rapidity():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.E = 10.
        p.pz = i
        particle_list.append(p)
    return [particle_list]

def test_rapidity_cut(particle_list_momentum_rapidity):
    test_cases = [
        # Test cases for valid input
        (0.05,None,None, [[particle_list_momentum_rapidity[0][0]]]),
        ((-0.05,0.05),None,None, [[particle_list_momentum_rapidity[0][0]]]),
        # Test cases for invalid input
        ((0.1,1.0,2.0),None,TypeError,None),
        (True,None,TypeError,None),
        ((1.0,'a'),None,ValueError,None),
        ((1.0,0.5),UserWarning,None,None)
    ]

    for cut_value, expected_warning, expected_error, expected_result in test_cases:
        if expected_warning:
            with pytest.warns(expected_warning):
                result = rapidity_cut(particle_list_momentum_rapidity, cut_value)
        elif expected_error:
            with pytest.raises(expected_error):
                result = rapidity_cut(particle_list_momentum_rapidity, cut_value)
        else:
            result = rapidity_cut(particle_list_momentum_rapidity, cut_value)
            assert result == expected_result

@pytest.fixture
def particle_list_pseudorapidity():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.px = 1.
        p.py = 0.
        p.pz = i
        particle_list.append(p)
    return [particle_list]

def test_pseudorapidity_cut(particle_list_pseudorapidity):
    test_cases = [
        # Test cases for valid input
        (0.5,None,None, [[particle_list_pseudorapidity[0][0]]]),
        ((-0.5,0.5),None,None, [[particle_list_pseudorapidity[0][0]]]),
        # Test cases for invalid input
        ((0.1,1.0,2.0),None,TypeError,None),
        (True,None,TypeError,None),
        ((1.0,'a'),None,ValueError,None),
        ((1.0,0.5),UserWarning,None,None)
    ]

    for cut_value, expected_warning, expected_error, expected_result in test_cases:
        if expected_warning:
            with pytest.warns(expected_warning):
                result = pseudorapidity_cut(particle_list_pseudorapidity, cut_value)
        elif expected_error:
            with pytest.raises(expected_error):
                result = pseudorapidity_cut(particle_list_pseudorapidity, cut_value)
        else:
            result = pseudorapidity_cut(particle_list_pseudorapidity, cut_value)
            assert result == expected_result

@pytest.fixture
def particle_list_spatial_rapidity():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.t = i+1
        p.z = i
        particle_list.append(p)
    return [particle_list]

def test_rapidity_cut(particle_list_spatial_rapidity):
    test_cases = [
        # Test cases for valid input
        (0.5,None,None, [[particle_list_spatial_rapidity[0][0]]]),
        ((-0.5,0.5),None,None, [[particle_list_spatial_rapidity[0][0]]]),
        # Test cases for invalid input
        ((0.1,1.0,2.0),None,TypeError,None),
        (True,None,TypeError,None),
        ((1.0,'a'),None,ValueError,None),
        ((1.0,0.5),UserWarning,None,None)
    ]

    for cut_value, expected_warning, expected_error, expected_result in test_cases:
        if expected_warning:
            with pytest.warns(expected_warning):
                result = spatial_rapidity_cut(particle_list_spatial_rapidity, cut_value)
        elif expected_error:
            with pytest.raises(expected_error):
                result = spatial_rapidity_cut(particle_list_spatial_rapidity, cut_value)
        else:
            result = spatial_rapidity_cut(particle_list_spatial_rapidity, cut_value)
            assert result == expected_result

@pytest.fixture
def particle_list_multiplicity():
    particle_list1 = []
    for _ in range(5):
        p = Particle()
        particle_list1.append(p)
    particle_list2 = []
    for _ in range(10):
        p = Particle()
        particle_list2.append(p)
    final_list = [particle_list1]
    final_list.extend([particle_list2])
    return final_list

def test_multiplicity_cut(particle_list_multiplicity):
    # Test cases for valid input
    assert multiplicity_cut(particle_list_multiplicity, 7) == [particle_list_multiplicity[1]]
    assert multiplicity_cut(particle_list_multiplicity, 11) == []

    # Test cases for invalid input
    with pytest.raises(TypeError):
        multiplicity_cut(particle_list_multiplicity, min_multiplicity=3.5)

    with pytest.raises(ValueError):
        multiplicity_cut(particle_list_multiplicity, min_multiplicity=-1)

@pytest.fixture
def particle_list_status():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.status = 1
        particle_list.append(p)
        q = Particle()
        q.status = 0
        particle_list.append(q)
    return [particle_list]

def test_particle_status(particle_list_status):
    # Test for single status
    result = particle_status(particle_list_status, status_list=1)
    assert all(p.status == 1 for event in result for p in event)

    # Test for multiple statuses
    result = particle_status(particle_list_status, status_list=[0, 1])
    assert all(p.status in [0, 1] for event in result for p in event)

    # Test for invalid input
    with pytest.raises(TypeError):
        particle_status(particle_list_status, status_list='invalid')

    with pytest.raises(TypeError):
        particle_status(particle_list_status, status_list=[0, 'invalid'])
