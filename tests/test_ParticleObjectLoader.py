# ===================================================
#
#    Copyright (c) 2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import pytest
from sparkx.loader.ParticleObjectLoader import ParticleObjectLoader
from sparkx.Filter import *

# Create dummy particles for testing
particle1 = Particle()
particle2 = Particle()
particle3 = Particle()
particle4 = Particle()
particle5 = Particle()
particle1.charge = 1
particle2.charge = 0
particle3.charge = 1
particle4.charge = 1
particle5.charge = 0
particle1.strangeness = 0
particle2.strangeness = 0
particle3.strangeness = 0
particle4.strangeness = 0
particle5.strangeness = 1
particle_list = [[particle1, particle2], [particle3, particle4], [particle5]]


def test_particleobject_loader_initialization():
    loader = ParticleObjectLoader(particle_list)
    assert loader.particle_list_ == particle_list


def test_load_no_filters():
    loader = ParticleObjectLoader(particle_list)
    result = loader.load()
    assert result[0] == particle_list
    assert result[1] == len(particle_list)
    assert result[2] == [2, 2, 1]


def test_load_with_events_tuple():
    loader = ParticleObjectLoader(particle_list)
    result = loader.load(events=(0, 1))
    assert len(result[0]) == 2
    assert result[1] == len(particle_list)
    assert result[2] == [2, 2, 1]


def test_load_with_single_event():
    loader = ParticleObjectLoader(particle_list)
    result = loader.load(events=1)
    assert len(result[0]) == 1
    assert result[1] == len(particle_list)
    assert result[2] == [2, 2, 1]


def test_load_with_filters():
    filters = {"charged_particles": True}
    loader = ParticleObjectLoader(particle_list)
    result = loader.load(filters=filters)
    expected_particles = [
        [p for p in particles if p.charge == 1.0] for particles in particle_list
    ]
    for res_event, exp_event in zip(result[0], expected_particles):
        for res_particle, exp_particle in zip(res_event, exp_event):

            assert res_particle.charge == exp_particle.charge


def test_load_with_unknown_keyword():
    loader = ParticleObjectLoader(particle_list)
    with pytest.raises(
        ValueError, match="Unknown keyword argument used in constructor"
    ):
        loader.load(unknown_key="value")


def test_load_with_invalid_events_tuple():
    loader = ParticleObjectLoader(particle_list)
    with pytest.raises(
        ValueError,
        match="First value of event number tuple must be smaller than second value",
    ):
        loader.load(events=(2, 1))


def test_load_with_negative_event_number():
    loader = ParticleObjectLoader(particle_list)
    with pytest.raises(ValueError, match="Event number must be non-negative"):
        loader.load(events=-1)


def test_apply_kwargs_filters():
    filters = {"charged_particles": True}
    loader = ParticleObjectLoader(particle_list)
    filtered_events = [
        loader._ParticleObjectLoader__apply_kwargs_filters([event], filters)[0]
        for event in particle_list
    ]
    expected_particles = [
        [p for p in particles if p.charge == 1.0] for particles in particle_list
    ]
    assert filtered_events == expected_particles


def test_set_num_output_per_event():
    loader = ParticleObjectLoader(particle_list)
    result = loader.load()
    num_output_per_event = loader.set_num_output_per_event()
    assert num_output_per_event == [2, 2, 1]
