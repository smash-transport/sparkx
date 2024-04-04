import numpy as np
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

def test_strange_particles(particle_nan_quantities,particle_list_strange):
    return_list = strange_particles(particle_nan_quantities)
    assert(len(return_list[0]) == 0)

    return_list = strange_particles(particle_list_strange)
    assert(len(return_list[0]) == 5)