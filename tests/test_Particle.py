#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
from sparkx.Particle import Particle
import warnings
import numpy as np
import pytest
import math
from particle import PDGID

# test the getter and setter functions in the case a particle object is created
# by the user
def test_t():
    p = Particle()
    p.t = 1.0
    assert p.t == 1.0
    assert isinstance(p.t, float)

def test_x():
    p = Particle()
    p.x = 1.0
    assert p.x == 1.0
    assert isinstance(p.x, float)

def test_y():
    p = Particle()
    p.y = 1.0
    assert p.y == 1.0
    assert isinstance(p.y, float)

def test_z():
    p = Particle()
    p.z = 1.0
    assert p.z == 1.0
    assert isinstance(p.z, float)

def test_mass():
    p = Particle()
    p.mass = 0.138
    assert p.mass == 0.138
    assert isinstance(p.mass, float)

def test_E():
    p = Particle()
    p.E = 1.0
    assert p.E == 1.0
    assert isinstance(p.E, float)

def test_px():
    p = Particle()
    p.px = 1.0
    assert p.px == 1.0
    assert isinstance(p.px, float)

def test_py():
    p = Particle()
    p.py = 1.0
    assert p.py == 1.0
    assert isinstance(p.py, float)

def test_pz():
    p = Particle()
    p.pz = 1.0
    assert p.pz == 1.0
    assert isinstance(p.pz, float)

def test_pdg():
    # check valid particle
    p = Particle()
    p.pdg = 211
    assert p.pdg == 211
    assert p.pdg_valid == True
    assert isinstance(p.pdg, int)

    # check invalid particle
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        q = Particle()
        q.pdg = 99999999
        assert q.pdg == 99999999
        assert q.pdg_valid is False

    # Check if a warning is issued
    with pytest.warns(UserWarning, match=r'The PDG code 99999999 is not valid. All properties extracted from the PDG are set to nan.'):
        q.pdg = 99999999

def test_ID():
    p = Particle()
    p.ID = 1
    assert p.ID == 1
    assert isinstance(p.ID, int)

def test_charge():
    p = Particle()
    p.charge = 1
    assert p.charge == 1
    assert isinstance(p.charge, int)

def test_ncoll():
    p = Particle()
    p.ncoll = 10
    assert p.ncoll == 10
    assert isinstance(p.ncoll, int)

def test_form_time():
    p = Particle()
    p.form_time = 1.0
    assert p.form_time == 1.0
    assert isinstance(p.form_time, float)

def test_xsecfac():
    p = Particle()
    p.xsecfac = 1.0
    assert p.xsecfac == 1.0
    assert isinstance(p.xsecfac, float)

def test_proc_id_origin():
    p = Particle()
    p.proc_id_origin = 1
    assert p.proc_id_origin == 1
    assert isinstance(p.proc_id_origin, int)

def test_proc_type_origin():
    p = Particle()
    p.proc_type_origin = 1
    assert p.proc_type_origin == 1
    assert isinstance(p.proc_type_origin, int)

def test_t_last_coll():
    p = Particle()
    p.t_last_coll = 10.0
    assert p.t_last_coll == 10.0
    assert isinstance(p.t_last_coll, float)

def test_pdg_mother1():
    p = Particle()
    p.pdg_mother1 = 211
    assert p.pdg_mother1 == 211
    assert isinstance(p.pdg_mother1, int)

def test_pdg_mother2():
    p = Particle()
    p.pdg_mother2 = 211
    assert p.pdg_mother2 == 211
    assert isinstance(p.pdg_mother2, int)

def test_baryon_number():
    p = Particle()
    p.baryon_number = 1
    assert p.baryon_number == 1
    assert isinstance(p.baryon_number, int)

def test_strangeness():
    p = Particle()
    p.strangeness = 1
    assert p.strangeness == 1
    assert isinstance(p.strangeness, int)

def test_weight():
    p = Particle()
    p.weight = 1.0
    assert p.weight == 1.0
    assert isinstance(p.weight, float)

def test_status():
    p = Particle()
    p.status = 27
    assert p.status == 27
    assert isinstance(p.status, int)

def test_initialize_from_array_valid_formats():
    format1 = "JETSCAPE"
    array1 = np.array([1,211,27,4.36557,3.56147,0.562961,2.45727])

    p1 = Particle(input_format=format1,particle_array=array1)
    assert p1.ID == 1
    assert p1.pdg == 211
    assert p1.status == 27
    assert p1.E == 4.36557
    assert p1.px == 3.56147
    assert p1.py == 0.562961
    assert p1.pz == 2.45727
    assert np.isclose(p1.mass, 0.137956238, rtol=1e-6)
    assert p1.pdg_valid == True
    assert p1.charge == 1


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

def test_initialize_from_array_Jetscape_invalid_pdg_warning():
    format1 = "JETSCAPE"
    array1 = np.array([1,99999999,27,4.36557,3.56147,0.562961,2.45727])

    # check that a warning is issued
    with pytest.warns(UserWarning, match=r"The PDG code 99999999 is not known by PDGID, charge could not be computed. Consider setting it by hand."):
        Particle(input_format=format1,particle_array=array1)


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

def test_angular_momentum_valid_values():
    p = Particle()
    p.x = 1.0
    p.y = 2.0
    p.z = 3.0
    p.px = 0.1
    p.py = 0.2
    p.pz = 0.3

    result = p.angular_momentum()

    expected_result = np.cross([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])

    assert np.array_equal(result, expected_result)

def test_angular_momentum_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.angular_momentum()

    assert np.isnan(result).all()

def test_angular_momentum_partial_missing_values():
    p = Particle()
    p.x = 1.0
    p.y = 2.0
    p.z = 3.0
    # Leave momentum values as np.nan

    result = p.angular_momentum()

    assert np.isnan(result).all()

def test_angular_momentum_zero_momentum():
    p = Particle()
    p.x = 1.0
    p.y = 2.0
    p.z = 3.0
    p.px = 0.0
    p.py = 0.0
    p.pz = 0.0

    result = p.angular_momentum()

    assert np.array_equal(result, [0.0, 0.0, 0.0])

def test_angular_momentum_zero_position():
    p = Particle()
    p.x = 0.0
    p.y = 0.0
    p.z = 0.0
    p.px = 1.0
    p.py = 2.0
    p.pz = 3.0

    result = p.angular_momentum()

    assert np.array_equal(result, [0.0, 0.0, 0.0])

def test_momentum_rapidity_Y_valid_values():
    p = Particle()
    p.E = 10.0
    p.pz = 5.0

    result = p.momentum_rapidity_Y()

    expected_result = 0.5 * np.log((10.0 + 5.0) / (10.0 - 5.0))

    assert np.isclose(result, expected_result)

def test_momentum_rapidity_Y_zero_denominator():
    p = Particle()
    p.E = 5.0
    p.pz = 5.0  # E == pz, should add a small positive value

    result = p.momentum_rapidity_Y()
    expected_result = 0.5 * np.log((5.0 + 5.0) / (1e-10))

    assert np.isclose(result, expected_result)

def test_momentum_rapidity_Y_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.momentum_rapidity_Y()

    assert np.isnan(result)

def test_p_abs_valid_values():
    p = Particle()
    p.px = 1.0
    p.py = 2.0
    p.pz = 3.0

    result = p.p_abs()

    expected_result = np.sqrt(1.0**2 + 2.0**2 + 3.0**2)

    assert np.isclose(result, expected_result)

def test_p_abs_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.p_abs()

    assert np.isnan(result)

def test_p_abs_zero_momentum():
    p = Particle()
    p.px = 0.0
    p.py = 0.0
    p.pz = 0.0

    result = p.p_abs()

    assert np.isclose(result, 0.0)

def test_pt_abs_valid_values():
    p = Particle()
    p.px = 1.0
    p.py = 2.0

    result = p.pt_abs()

    expected_result = np.sqrt(1.0**2 + 2.0**2)

    assert np.isclose(result, expected_result)

def test_pt_abs_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.pt_abs()

    assert np.isnan(result)

def test_pt_abs_zero_transverse_momentum():
    p = Particle()
    p.px = 0.0
    p.py = 0.0

    result = p.pt_abs()

    assert np.isclose(result, 0.0)

def test_phi_valid_values():
    p = Particle()
    p.px = 1.0
    p.py = 2.0

    result = p.phi()

    expected_result = math.atan2(2.0, 1.0)

    assert np.isclose(result, expected_result)

def test_phi_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.phi()

    assert np.isnan(result)

def test_phi_zero_transverse_momentum():
    p = Particle()
    p.px = 0.0
    p.py = 0.0

    result = p.phi()

    assert np.isclose(result, 0.0)

def test_phi_small_transverse_momentum():
    p = Particle()
    p.px = 1e-7
    p.py = 1e-7

    result = p.phi()

    assert np.isclose(result, 0.0)

def test_theta_valid_values():
    p = Particle()
    p.px = 1.0
    p.py = 2.0
    p.pz = 3.0

    result = p.theta()

    expected_result = np.arccos(3.0 / np.sqrt(1.0**2 + 2.0**2 + 3.0**2))

    assert np.isclose(result, expected_result)

def test_theta_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.theta()

    assert np.isnan(result)

def test_theta_zero_momentum():
    p = Particle()
    p.px = 0.0
    p.py = 0.0
    p.pz = 0.0

    result = p.theta()

    assert np.isclose(result, 0.0)

def test_pseudorapidity_valid_values():
    p = Particle()
    p.px = 1.0
    p.py = 2.0
    p.pz = 3.0

    result = p.pseudorapidity()

    expected_result = 0.5 * np.log((np.sqrt(1.0**2 + 2.0**2 + 3.0**2) + 3.0) / (np.sqrt(1.0**2 + 2.0**2 + 3.0**2) - 3.0))

    assert np.isclose(result, expected_result)

def test_pseudorapidity_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.pseudorapidity()

    assert np.isnan(result)

def test_pseudorapidity_zero_momentum():
    p = Particle()
    p.px = 0.0
    p.py = 0.0
    p.pz = 1.0

    result = p.pseudorapidity()
    expected_result = 0.5 * np.log(2./1e-10)

    assert np.isclose(result, expected_result)

def test_spatial_rapidity_valid_values():
    p = Particle()
    p.t = 5.0
    p.z = 3.0

    result = p.spatial_rapidity()

    expected_result = 0.5 * np.log((5.0 + 3.0) / (5.0 - 3.0))

    assert np.isclose(result, expected_result)

def test_spatial_rapidity_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.spatial_rapidity()

    assert np.isnan(result)

def test_spatial_rapidity_invalid_values():
    p = Particle()
    p.t = 3.0
    p.z = 5.0

    with pytest.raises(ValueError, match=r"|z| < t not fulfilled"):
        p.spatial_rapidity()

def test_proper_time_valid_values():
    p = Particle()
    p.t = 5.0
    p.z = 3.0

    result = p.proper_time()

    expected_result = np.sqrt(5.0**2 - 3.0**2)

    assert np.isclose(result, expected_result)

def test_proper_time_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.proper_time()

    assert np.isnan(result)

def test_proper_time_invalid_values():
    p = Particle()
    p.t = 3.0
    p.z = 5.0

    with pytest.raises(ValueError, match=r"|z| < t not fulfilled"):
        p.proper_time()

def test_compute_mass_from_energy_momentum_valid_values():
    p = Particle()
    p.E = 3.0
    p.px = 1.0
    p.py = 2.0
    p.pz = 2.0

    result = p.compute_mass_from_energy_momentum()

    expected_result = np.sqrt(3.0**2 - (1.0**2 + 2.0**2 + 2.0**2))

    assert np.isclose(result, expected_result)

def test_compute_mass_from_energy_momentum_missing_values():
    p = Particle()
    # Leave some values as np.nan

    result = p.compute_mass_from_energy_momentum()

    assert np.isnan(result)

def test_compute_mass_from_energy_momentum_zero_energy():
    p = Particle()
    p.E = 0.0
    p.px = 0.0
    p.py = 0.0
    p.pz = 0.0

    result = p.compute_mass_from_energy_momentum()

    assert np.isclose(result, 0.0)

def test_compute_charge_from_pdg_valid_values():
    p = Particle()
    p.pdg = 211  # Assuming PDG code for a positive pion

    result = p.compute_charge_from_pdg()

    expected_result = PDGID(211).charge

    assert result == expected_result

def test_compute_charge_from_pdg_invalid_values():
    p = Particle()
    p.pdg_valid = False
    # Leave pdg as an invalid value

    result = p.compute_charge_from_pdg()

    assert np.isnan(result)

def test_is_meson_from_pdg_valid_values():
    p = Particle()
    p.pdg = 211  # Assuming PDG code for a positive pion

    result = p.is_meson()

    expected_result = PDGID(211).is_meson

    assert result == expected_result

def test_is_meson_from_pdg_invalid_values():
    p = Particle()
    p.pdg_valid = False
    # Leave pdg as an invalid value

    result = p.is_meson()

    assert np.isnan(result)

def test_is_baryon_from_pdg_valid_values():
    p = Particle()
    p.pdg = 211  # Assuming PDG code for a positive pion

    result = p.is_baryon()

    expected_result = PDGID(211).is_baryon

    assert result == expected_result

def test_is_baryon_from_pdg_invalid_values():
    p = Particle()
    p.pdg_valid = False
    # Leave pdg as an invalid value

    result = p.is_baryon()

    assert np.isnan(result)

def test_is_hadron_from_pdg_valid_values():
    p = Particle()
    p.pdg = 211  # Assuming PDG code for a positive pion

    result = p.is_hadron()

    expected_result = PDGID(211).is_hadron

    assert result == expected_result

def test_is_hadron_from_pdg_invalid_values():
    p = Particle()
    p.pdg_valid = False
    # Leave pdg as an invalid value

    result = p.is_hadron()

    assert np.isnan(result)

def test_is_strange_from_pdg_valid_values():
    p = Particle()
    p.pdg = 211  # Assuming PDG code for a positive pion

    result = p.is_strange()

    expected_result = PDGID(211).has_strange

    assert result == expected_result

def test_is_strange_from_pdg_invalid_values():
    p = Particle()
    p.pdg_valid = False
    # Leave pdg as an invalid value

    result = p.is_strange()

    assert np.isnan(result)

def test_is_heavy_flavor_from_pdg_valid_values():
    p = Particle()
    p.pdg = 211  # Assuming PDG code for a positive pion

    result = p.is_heavy_flavor()

    expected_result = PDGID(211).has_charm or PDGID(211).has_bottom\
          or PDGID(211).has_top

    assert result == expected_result

def test_is_heavy_flavor_from_pdg_invalid_values():
    p = Particle()
    p.pdg_valid = False
    # Leave pdg as an invalid value

    result = p.is_heavy_flavor()

    assert np.isnan(result)

def test_spin_from_pdg_valid_values():
    p = Particle()
    p.pdg = 211  # Assuming PDG code for a positive pion

    result = p.spin()

    expected_result = PDGID(211).J

    assert result == expected_result

def test_spin_from_pdg_invalid_values():
    p = Particle()
    p.pdg_valid = False
    # Leave pdg as an invalid value

    result = p.spin()

    assert np.isnan(result)

def test_spin_degeneracy_from_pdg_valid_values():
    p = Particle()
    p.pdg = 211  # Assuming PDG code for a positive pion

    result = p.spin_degeneracy()

    expected_result = PDGID(211).j_spin

    assert result == expected_result

def test_spin_degeneracy_from_pdg_invalid_values():
    p = Particle()
    p.pdg_valid = False
    # Leave pdg as an invalid value

    result = p.spin_degeneracy()

    assert np.isnan(result)