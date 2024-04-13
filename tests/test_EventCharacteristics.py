#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
import os
import pytest
import numpy as np
from sparkx.EventCharacteristics import EventCharacteristics
from sparkx.Lattice3D import Lattice3D
from sparkx.Oscar import Oscar
from sparkx.Particle import Particle

@pytest.fixture
def oscar_file_path():
    return os.path.join(os.path.dirname(__file__), 'test_files', 'particle_lists.oscar')

@pytest.fixture
def oscar_particle_objects_list(oscar_file_path):
    # grab the first event
    particle_list = Oscar(oscar_file_path,events=0).particle_objects_list()
    return particle_list[0]

@pytest.fixture
def test_Lattice3D_instance():
    lattice = Lattice3D(x_min=-3.,x_max=3.,y_min=-3.,y_max=3.,z_min=-3.,z_max=3.,
                     num_points_x=15,num_points_y=15,num_points_z=15)
    lattice.set_value_nearest_neighbor(0.,1.,0.,1.)
    lattice.set_value_nearest_neighbor(0.,-1.,0.,1.)
    return lattice

def test_EventCharacteristics_initialization(oscar_particle_objects_list,test_Lattice3D_instance):
    eve_char = EventCharacteristics(oscar_particle_objects_list)
    assert eve_char.has_lattice_ == False
    assert eve_char.event_data_ == oscar_particle_objects_list

    with pytest.raises(TypeError):
        add_number_to_list = oscar_particle_objects_list.append(1.0)
        EventCharacteristics(add_number_to_list)
    
    with pytest.raises(TypeError):
        add_string = 'test'
        EventCharacteristics(add_string)

    eve_char = EventCharacteristics(test_Lattice3D_instance)
    assert eve_char.has_lattice_ == True
    assert np.array_equal(eve_char.event_data_.grid_, test_Lattice3D_instance.grid_)

@pytest.fixture
def fake_particle_list_eccentricity():
    particle_list = []
    p1 = Particle()
    p1.x = 0.
    p1.y = 1.
    particle_list.append(p1)
    p2 = Particle()
    p2.x = 0.
    p2.y = -1.
    particle_list.append(p2)
    return particle_list

def test_eccentricity_from_particles(fake_particle_list_eccentricity):
    eve_char = EventCharacteristics(fake_particle_list_eccentricity)
    eps2 = eve_char.eccentricity(2,weight_quantity="number")
    assert np.allclose(eps2, (1.-0j), atol=1e-10)

    eps3 = eve_char.eccentricity(3,weight_quantity="number")
    assert np.allclose(eps3, (0.-0j), atol=1e-10)

    with pytest.raises(ValueError):
        eve_char.eccentricity(0,weight_quantity="number")
    
    with pytest.raises(ValueError):
        eve_char.eccentricity(2,harmonic_m=0,weight_quantity="number")

def test_eccentricity_from_lattice(test_Lattice3D_instance):
    eve_char = EventCharacteristics(test_Lattice3D_instance)
    eps2 = eve_char.eccentricity(2,weight_quantity="number")
    np.allclose(eps2, (1.-0j), atol=1e-10)

    eps3 = eve_char.eccentricity(3,weight_quantity="number")
    np.allclose(eps3, (0.-0j), atol=1e-10)

    with pytest.raises(ValueError):
        eve_char.eccentricity(0,weight_quantity="number")

    with pytest.raises(ValueError):
        eve_char.eccentricity(2,harmonic_m=0,weight_quantity="number")

def test_eBQS_densities_Milne_from_OSCAR_IC(oscar_particle_objects_list,test_Lattice3D_instance):
    x_min = y_min = z_min = -5.
    x_max = y_max = z_max = 5.
    Nx = Ny = Nz = 50
    n_sigma_x = n_sigma_y = n_sigma_z = 3
    sigma_smear = 0.2
    eta_range = 0.2
    output_filename = 'test_eBQS_densities_Milne_from_OSCAR_IC.dat'

    # test error if IC_info is not None and string
    with pytest.raises(TypeError):
        eve_char = EventCharacteristics(oscar_particle_objects_list)
        eve_char.generate_eBQS_densities_Milne_from_OSCAR_IC(x_min,x_max, 
                                                             y_min,y_max, 
                                                             z_min,z_max,
                                                             Nx,Ny,Nz, 
                                                             n_sigma_x,n_sigma_y,
                                                             n_sigma_z,sigma_smear, 
                                                             eta_range,
                                                             output_filename, 
                                                             IC_info=1)

    # test error if class is initialized with lattice
    with pytest.raises(TypeError):
        eve_char = EventCharacteristics(test_Lattice3D_instance)
        eve_char.generate_eBQS_densities_Milne_from_OSCAR_IC(x_min,x_max, 
                                                             y_min,y_max, 
                                                             z_min,z_max,
                                                             Nx,Ny,Nz, 
                                                             n_sigma_x,n_sigma_y,
                                                             n_sigma_z,sigma_smear, 
                                                             eta_range,
                                                             output_filename)
        
def test_eBQS_densities_Minkowski_from_OSCAR_IC(oscar_particle_objects_list,test_Lattice3D_instance):
    x_min = y_min = z_min = -5.
    x_max = y_max = z_max = 5.
    Nx = Ny = Nz = 50
    n_sigma_x = n_sigma_y = n_sigma_z = 3
    sigma_smear = 0.2
    eta_range = 0.2
    output_filename = 'test_eBQS_densities_Milne_from_OSCAR_IC.dat'

    # test error if IC_info is not None and string
    with pytest.raises(TypeError):
        eve_char = EventCharacteristics(oscar_particle_objects_list)
        eve_char.generate_eBQS_densities_Minkowski_from_OSCAR_IC(x_min,x_max, 
                                                             y_min,y_max, 
                                                             z_min,z_max,
                                                             Nx,Ny,Nz, 
                                                             n_sigma_x,n_sigma_y,
                                                             n_sigma_z,sigma_smear, 
                                                             eta_range,
                                                             output_filename, 
                                                             IC_info=1)

    # test error if class is initialized with lattice
    with pytest.raises(TypeError):
        eve_char = EventCharacteristics(test_Lattice3D_instance)
        eve_char.generate_eBQS_densities_Minkowski_from_OSCAR_IC(x_min,x_max, 
                                                             y_min,y_max, 
                                                             z_min,z_max,
                                                             Nx,Ny,Nz, 
                                                             n_sigma_x,n_sigma_y,
                                                             n_sigma_z,sigma_smear, 
                                                             eta_range,
                                                             output_filename)