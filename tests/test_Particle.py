from sparkx.Particle import Particle
import warnings
import numpy as np
import pytest

# test the getter and setter functions in the case a particle object is created
# by the user
def test_t():
    p = Particle()
    p.t = 1.0
    assert p.t == 1.0

def test_x():
    p = Particle()
    p.x = 1.0
    assert p.x == 1.0

def test_y():
    p = Particle()
    p.y = 1.0
    assert p.y == 1.0

def test_z():
    p = Particle()
    p.z = 1.0
    assert p.z == 1.0

def test_mass():
    p = Particle()
    p.mass = 0.138
    assert p.mass == 0.138

def test_E():
    p = Particle()
    p.E = 1.0
    assert p.E == 1.0

def test_px():
    p = Particle()
    p.px = 1.0
    assert p.px == 1.0

def test_py():
    p = Particle()
    p.py = 1.0
    assert p.py == 1.0

def test_pz():
    p = Particle()
    p.pz = 1.0
    assert p.pz == 1.0

def test_pdg():
    # check valid particle
    p = Particle()
    p.pdg = 211
    assert p.pdg == 211
    assert p.pdg_valid == True

    # check invalid particle
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        q = Particle()
        q.pdg = 99999999
        assert q.pdg == 99999999
        assert q.pdg_valid is False

    # Check if a warning is issued
    with pytest.warns(UserWarning, match=r'The PDG code 99999999 is not valid. All properties extracted from the PDG are set to None.'):
        q.pdg = 99999999

def test_ID():
    p = Particle()
    p.ID = 1
    assert p.ID == 1

def test_charge():
    p = Particle()
    p.charge = 1
    assert p.charge == 1

def test_ncoll():
    p = Particle()
    p.ncoll = 10
    assert p.ncoll == 10

def test_form_time():
    p = Particle()
    p.form_time = 1.0
    assert p.form_time == 1.0

def test_xsecfac():
    p = Particle()
    p.xsecfac = 1.0
    assert p.xsecfac == 1.0

def test_proc_id_origin():
    p = Particle()
    p.proc_id_origin = 1
    assert p.proc_id_origin == 1

def test_proc_type_origin():
    p = Particle()
    p.proc_id_origin = 1
    assert p.proc_id_origin == 1

def test_t_last_coll():
    p = Particle()
    p.t_last_coll = 10.0
    assert p.t_last_coll == 10.0

def test_pdg_mother1():
    p = Particle()
    p.pdg_mother1 = 211
    assert p.pdg_mother1 == 211

def test_pdg_mother2():
    p = Particle()
    p.pdg_mother2 = 211
    assert p.pdg_mother2 == 211

def test_baryon_number():
    p = Particle()
    p.baryon_number = 1
    assert p.baryon_number == 1

def test_strangeness():
    p = Particle()
    p.strangeness = 1
    assert p.strangeness == 1

def test_weight():
    p = Particle()
    p.weight = 1.0
    assert p.weight == 1.0

def test_status():
    p = Particle()
    p.status = 27
    assert p.status == 27

def test_initialize_from_array_valid_formats():
    format1 = "JETSCAPE"
    array1 = np.array([1,211,27,3.0,0.5,1.0,1.5])

    p1 = Particle(input_format=format1,particle_array=array1)
    assert p1.ID == 1
    assert p1.pdg == 211
    assert p1.status == 27
    assert p1.E == 3.0
    assert p1.px == 0.5
    assert p1.py == 1.0
    assert p1.pz == 1.5

    format2 = "Oscar2013"
    array2 = np.array([0.0,1.0,2.0,3.0,0.138,4.0,5.0,6.0,7.0,211,100,1])

    p2 = Particle(input_format=format2,particle_array=array2)
    assert p2.t == 0.0
    assert p2.x == 1.0
    assert p2.y == 2.0
    assert p2.z == 3.0
    assert p2.mass == 0.138
    assert p2.E == 4.0
    assert p2.px == 5.0
    assert p2.py == 6.0
    assert p2.pz == 7.0
    assert p2.pdg == 211
    assert p2.ID == 100
    assert p2.charge == 1

    format3 = "Oscar2013Extended"
    array3 = np.array([0.0,1.0,2.0,3.0,0.138,4.0,5.0,6.0,7.0,211,100,1,0,0.1,0.3,101,2,2.5,321,2212,1,-1])

    p3 = Particle(input_format=format3,particle_array=array3)
    assert p3.t == 0.0
    assert p3.x == 1.0
    assert p3.y == 2.0
    assert p3.z == 3.0
    assert p3.mass == 0.138
    assert p3.E == 4.0
    assert p3.px == 5.0
    assert p3.py == 6.0
    assert p3.pz == 7.0
    assert p3.pdg == 211
    assert p3.ID == 100
    assert p3.charge == 1
    assert p3.ncoll == 0
    assert p3.form_time == 0.1
    assert p3.xsecfac == 0.3
    assert p3.proc_id_origin == 101
    assert p3.proc_type_origin == 2
    assert p3.t_last_coll == 2.5
    assert p3.pdg_mother1 == 321
    assert p3.pdg_mother2 == 2212
    assert p3.baryon_number == 1
    assert p3.strangeness == -1

    format4 = "Oscar2013Extended_IC"
    array4 = array3

    p4 = Particle(input_format=format4,particle_array=array4)
    assert p4.t == 0.0
    assert p4.x == 1.0
    assert p4.y == 2.0
    assert p4.z == 3.0
    assert p4.mass == 0.138
    assert p4.E == 4.0
    assert p4.px == 5.0
    assert p4.py == 6.0
    assert p4.pz == 7.0
    assert p4.pdg == 211
    assert p4.ID == 100
    assert p4.charge == 1
    assert p4.ncoll == 0
    assert p4.form_time == 0.1
    assert p4.xsecfac == 0.3
    assert p4.proc_id_origin == 101
    assert p4.proc_type_origin == 2
    assert p4.t_last_coll == 2.5
    assert p4.pdg_mother1 == 321
    assert p4.pdg_mother2 == 2212
    assert p4.baryon_number == 1
    assert p4.strangeness == -1

    format5 = "Oscar2013Extended_Photons"
    array5 = np.array([0.0,1.0,2.0,3.0,0.138,4.0,5.0,6.0,7.0,211,100,1,0,0.1,0.3,101,2,2.5,321,2212,0.75])
    p5 = Particle(input_format=format5,particle_array=array5)
    assert p5.t == 0.0
    assert p5.x == 1.0
    assert p5.y == 2.0
    assert p5.z == 3.0
    assert p5.mass == 0.138
    assert p5.E == 4.0
    assert p5.px == 5.0
    assert p5.py == 6.0
    assert p5.pz == 7.0
    assert p5.pdg == 211
    assert p5.ID == 100
    assert p5.charge == 1
    assert p5.ncoll == 0
    assert p5.form_time == 0.1
    assert p5.xsecfac == 0.3
    assert p5.proc_id_origin == 101
    assert p5.proc_type_origin == 2
    assert p5.t_last_coll == 2.5
    assert p5.pdg_mother1 == 321
    assert p5.pdg_mother2 == 2212
    assert p5.weight == 0.75

def test_initialize_from_array_invalid_format():
    with pytest.raises(ValueError, match=r"Unsupported input format 'InvalidFormat'"):
        Particle(input_format="InvalidFormat", particle_array=np.array([1, 2, 3]))

def test_initialize_from_array_corrupted_data():
    format1 = "Oscar2013"
    # Provide an array with incorrect length to trigger ValueError
    with pytest.raises(ValueError, match=r"The input file is corrupted!"):
        Particle(input_format=format1, particle_array=np.array([1, 2, 3]))

def test_initialize_from_array_warning_invalid_pdg():
    format1 = "Oscar2013"
    array1 = np.array([0.0, 1.0, 2.0, 3.0, 0.138, 4.0, 5.0, 6.0, 7.0, 99999999, 100, 1])

    with pytest.warns(UserWarning, match=r"The PDG code 99999999 is not valid."):
        p1 = Particle(input_format=format1, particle_array=array1)

    assert p1.pdg_valid is False