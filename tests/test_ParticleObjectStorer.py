import pytest
import numpy as np
from sparkx.Particle import Particle
from sparkx.Loader.ParticleObjectLoader import ParticleObjectLoader
from sparkx.ParticleObjectStorer import ParticleObjectStorer

# Utility function to create particles
def create_particle(t, x, y, z, mass, E, px, py, pz, pdg, ID, charge, ncoll, form_time, xsecfac, proc_id_origin, proc_type_origin, t_last_coll, pdg_mother1, pdg_mother2, baryon_number, strangeness, weight, status):
    particle = Particle()
    particle.t = t
    particle.x = x
    particle.y = y
    particle.z = z
    particle.mass = mass
    particle.E = E
    particle.px = px
    particle.py = py
    particle.pz = pz
    particle.pdg = pdg
    particle.ID = ID
    particle.charge = charge
    particle.ncoll = ncoll
    particle.form_time = form_time
    particle.xsecfac = xsecfac
    particle.proc_id_origin = proc_id_origin
    particle.proc_type_origin = proc_type_origin
    particle.t_last_coll = t_last_coll
    particle.pdg_mother1 = pdg_mother1
    particle.pdg_mother2 = pdg_mother2
    particle.baryon_number = baryon_number
    particle.strangeness = strangeness
    particle.weight = weight
    particle.status = status
    return particle

# Create test particles
particle1 = create_particle(1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 1, 0, 1.1, 1.2, 1, 1, 1.3, 0, 0, 1, 0, 1, 0)
particle2 = create_particle(2.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 2, 3, 0, 1, 2.1, 2.2, 2, 2, 2.3, 1, 1, 1, 1, 1, 1)

particle_object_list = [[particle1, particle2], [particle1], [particle2]]

# Tests

def test_particleobject_storer_initialization():
    storer = ParticleObjectStorer(particle_object_list)
    assert storer.particle_list_ == particle_object_list

def test_create_loader():
    storer = ParticleObjectStorer(particle_object_list)
    storer.create_loader(particle_object_list)
    assert isinstance(storer.loader_, ParticleObjectLoader)
    assert storer.loader_.particle_list_ == particle_object_list

def test_particle_as_list():
    storer = ParticleObjectStorer(particle_object_list)
    particle_list = storer._particle_as_list(particle1)
    expected_list = [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 1, 0, 1.1, 1.2, 1, 1, 1.3, 0, 0, 1, 0, 1, 0]
    assert particle_list == expected_list

def test_particle_list():
    storer = ParticleObjectStorer(particle_object_list)
    storer.num_events_ = 3
    storer.num_output_per_event_ = np.array([[0, 2], [1, 1], [2, 1]])
    particle_array = storer.particle_list()
    expected_array = [
        [[1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 1, 0, 1.1, 1.2, 1, 1, 1.3, 0, 0, 1, 0, 1, 0],
         [2.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 2, 3, 0, 1, 2.1, 2.2, 2, 2, 2.3, 1, 1, 1, 1, 1, 1]],
        [[1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 1, 0, 1.1, 1.2, 1, 1, 1.3, 0, 0, 1, 0, 1, 0]],
        [[2.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 2, 3, 0, 1, 2.1, 2.2, 2, 2, 2.3, 1, 1, 1, 1, 1, 1]]
    ]
    assert particle_array == expected_array

def test_particle_list_single_event():
    storer = ParticleObjectStorer([particle_object_list[0]])
    storer.num_events_ = 1
    storer.num_output_per_event_ = np.array([[0, 2]])
    particle_array = storer.particle_list()
    expected_array = [
        [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 1, 0, 1.1, 1.2, 1, 1, 1.3, 0, 0, 1, 0, 1, 0],
        [2.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 2, 3, 0, 1, 2.1, 2.2, 2, 2, 2.3, 1, 1, 1, 1, 1, 1]
    ]
    assert particle_array == expected_array

def test_print_particle_lists_to_file(tmp_path):
    storer = ParticleObjectStorer(particle_object_list)
    storer.num_events_ = 3
    storer.num_output_per_event_ = np.array([[0, 2], [1, 1], [2, 1]])
    
    filename = tmp_path / "particles.txt"
    storer.print_particle_lists_to_file(filename)
    
    with open(filename, 'r') as file:
        content = file.readlines()
    expected_content = [
        "1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,2,1,0,1.1,1.2,1,1,1.3,0,0,0,1\n",
        "2.0,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,2,3,0,1,2.1,2.2,2,2,2.3,1,1,1,1\n",
        "1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,2,1,0,1.1,1.2,1,1,1.3,0,0,0,1\n",
        "2.0,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,2,3,0,1,2.1,2.2,2,2,2.3,1,1,1,1\n"
    ]

    assert content == expected_content
