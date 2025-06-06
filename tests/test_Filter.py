# ===================================================
#
#    Copyright (c) 2023-2025
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

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


def test_charged_particles(
    particle_nan_quantities, particle_list_charged_uncharged
):
    return_list = charged_particles(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = charged_particles(particle_list_charged_uncharged)
    assert len(return_list[0]) == 5


def test_uncharged_particles(
    particle_nan_quantities, particle_list_charged_uncharged
):
    return_list = uncharged_particles(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = uncharged_particles(particle_list_charged_uncharged)
    assert len(return_list[0]) == 5


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


def test_strange_particles(particle_nan_quantities, particle_list_strange):
    return_list = keep_strange(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_strange(particle_list_strange)
    assert len(return_list[0]) == 5


@pytest.fixture
def particle_list_baryons_electrons():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.pdg = 2212
        particle_list.append(p)
    for i in range(4):
        p = Particle()
        p.pdg = 11
        particle_list.append(p)
    return [particle_list]


def test_keep_hadrons(particle_nan_quantities, particle_list_baryons_electrons):
    return_list = keep_hadrons(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_hadrons(particle_list_baryons_electrons)
    assert len(return_list[0]) == 5


def test_keep_leptons(particle_nan_quantities, particle_list_baryons_electrons):
    return_list = keep_leptons(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_leptons(particle_list_baryons_electrons)
    assert len(return_list[0]) == 4


def test_keep_baryons(particle_nan_quantities, particle_list_baryons_electrons):
    return_list = keep_baryons(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_baryons(particle_list_baryons_electrons)
    assert len(return_list[0]) == 5


@pytest.fixture
def particle_list_quarks_electrons():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.pdg = 2
        particle_list.append(p)
    for i in range(4):
        p = Particle()
        p.pdg = 11
        particle_list.append(p)
    return [particle_list]


def test_keep_quarks(particle_nan_quantities, particle_list_quarks_electrons):
    return_list = keep_quarks(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_quarks(particle_list_quarks_electrons)
    assert len(return_list[0]) == 5


@pytest.fixture
def particle_list_baryons_mesons():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.pdg = 2212
        particle_list.append(p)
    for i in range(4):
        p = Particle()
        p.pdg = 211
        particle_list.append(p)
    return [particle_list]


def test_keep_mesons(particle_nan_quantities, particle_list_baryons_mesons):
    return_list = keep_mesons(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_mesons(particle_list_baryons_mesons)
    assert len(return_list[0]) == 4


@pytest.fixture
def particle_list_hadrons_with_up_down_quarks():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.pdg = 2212
        particle_list.append(p)
    for i in range(4):
        p = Particle()
        p.pdg = 11
        particle_list.append(p)
    return [particle_list]


def test_keep_up_quark_hadrons(
    particle_nan_quantities, particle_list_hadrons_with_up_down_quarks
):
    return_list = keep_up(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_up(particle_list_hadrons_with_up_down_quarks)
    assert len(return_list[0]) == 5


def test_keep_down_quark_hadrons(
    particle_nan_quantities, particle_list_hadrons_with_up_down_quarks
):
    return_list = keep_down(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_down(particle_list_hadrons_with_up_down_quarks)
    assert len(return_list[0]) == 5


@pytest.fixture
def particle_list_hadrons_with_charm_quarks():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.pdg = 2212
        particle_list.append(p)
    for i in range(4):
        p = Particle()
        p.pdg = 411
        particle_list.append(p)
    return [particle_list]


def test_keep_charm_quark_hadrons(
    particle_nan_quantities, particle_list_hadrons_with_charm_quarks
):
    return_list = keep_charm(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_charm(particle_list_hadrons_with_charm_quarks)
    assert len(return_list[0]) == 4


@pytest.fixture
def particle_list_hadrons_with_bottom_quarks():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.pdg = 2212
        particle_list.append(p)
    for i in range(4):
        p = Particle()
        p.pdg = 511
        particle_list.append(p)
    return [particle_list]


def test_keep_bottom_quark_hadrons(
    particle_nan_quantities, particle_list_hadrons_with_bottom_quarks
):
    return_list = keep_bottom(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_bottom(particle_list_hadrons_with_bottom_quarks)
    assert len(return_list[0]) == 4


def test_keep_top_quark(
    particle_nan_quantities, particle_list_hadrons_with_bottom_quarks
):
    return_list = keep_top(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = keep_top(particle_list_hadrons_with_bottom_quarks)
    assert len(return_list[0]) == 0


@pytest.fixture
def particle_list_hadrons_with_photons():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.pdg = 2212
        particle_list.append(p)
    for i in range(4):
        p = Particle()
        p.pdg = 22
        particle_list.append(p)
    return [particle_list]


def test_remove_photons(
    particle_nan_quantities, particle_list_hadrons_with_photons
):
    return_list = remove_photons(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = remove_photons(particle_list_hadrons_with_photons)
    assert len(return_list[0]) == 5


def test_particle_species_filter(particle_list_strange):
    return_list = particle_species(copy.deepcopy(particle_list_strange), "321")
    assert len(return_list[0]) == 5

    return_list = particle_species(copy.deepcopy(particle_list_strange), 211)
    assert len(return_list[0]) == 5

    return_list = particle_species(copy.deepcopy(particle_list_strange), 321.0)
    assert len(return_list[0]) == 5

    return_list = particle_species(
        copy.deepcopy(particle_list_strange), [211, 321]
    )
    assert len(return_list[0]) == 10

    return_list = particle_species(
        copy.deepcopy(particle_list_strange), np.array([211, 321])
    )
    assert len(return_list[0]) == 10

    return_list = particle_species(
        copy.deepcopy(particle_list_strange), (211, 321)
    )
    assert len(return_list[0]) == 10

    with pytest.raises(ValueError):
        return_list = particle_species(particle_list_strange, np.nan)

    with pytest.raises(ValueError):
        return_list = particle_species(particle_list_strange, [np.nan, 211])


def test_remove_particle_species_filter(particle_list_strange):
    return_list = remove_particle_species(
        copy.deepcopy(particle_list_strange), "321"
    )
    assert len(return_list[0]) == 5

    return_list = remove_particle_species(
        copy.deepcopy(particle_list_strange), 321
    )
    assert len(return_list[0]) == 5

    return_list = remove_particle_species(
        copy.deepcopy(particle_list_strange), 321.0
    )
    assert len(return_list[0]) == 5

    return_list = remove_particle_species(
        copy.deepcopy(particle_list_strange), [211, 321]
    )
    assert len(return_list[0]) == 0

    return_list = remove_particle_species(
        copy.deepcopy(particle_list_strange), np.array([211, 321])
    )
    assert len(return_list[0]) == 0

    return_list = remove_particle_species(
        copy.deepcopy(particle_list_strange), (211, 321)
    )
    assert len(return_list[0]) == 0

    with pytest.raises(ValueError):
        return_list = remove_particle_species(particle_list_strange, np.nan)

    with pytest.raises(ValueError):
        return_list = remove_particle_species(
            particle_list_strange, [np.nan, 211]
        )


@pytest.fixture
def particle_list_ncoll():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.ncoll = i
        particle_list.append(p)
    return [particle_list]


def test_participants(particle_nan_quantities, particle_list_ncoll):
    return_list = participants(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = participants(particle_list_ncoll)
    assert len(return_list[0]) == 4


def test_spectators(particle_nan_quantities, particle_list_ncoll):
    return_list = spectators(particle_nan_quantities)
    assert len(return_list[0]) == 0

    return_list = spectators(particle_list_ncoll)
    assert len(return_list[0]) == 1


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


def test_lower_event_energy_cut(
    particle_nan_quantities, particle_list_energies
):
    return_list = lower_event_energy_cut(particle_nan_quantities, 1.0)
    assert len(return_list[0]) == 0

    with pytest.raises(ValueError):
        lower_event_energy_cut(particle_nan_quantities, -1.0)

    with pytest.raises(ValueError):
        lower_event_energy_cut(particle_nan_quantities, np.nan)

    with pytest.raises(TypeError):
        lower_event_energy_cut(particle_nan_quantities, "1.0")

    with pytest.raises(ValueError):
        return_list = lower_event_energy_cut(particle_list_energies, -1.0)

    return_list = lower_event_energy_cut(particle_list_energies, 8.0)
    assert len(return_list[0]) == 5
    assert len(return_list) == 1


@pytest.fixture
def particle_list_positions():
    particle_list = []
    p1 = Particle()
    p1.t = 1.0
    p1.x = p1.y = p1.z = 0.0
    particle_list.append(p1)

    p2 = Particle()
    p2.x = 1.0
    p2.t = p2.y = p2.z = 0.0
    particle_list.append(p2)

    p3 = Particle()
    p3.y = 1.0
    p3.t = p3.x = p3.z = 0.0
    particle_list.append(p3)

    p4 = Particle()
    p4.z = 1.0
    p4.t = p4.x = p4.y = 0.0
    particle_list.append(p4)

    return [particle_list]


def test_spacetime_cut(particle_list_positions):
    test_cases = [
        # Test cases for valid input
        ("t", (0.5, 1.5), None, None, [[particle_list_positions[0][0]]]),
        ("t", (1.5, 2.5), None, None, [[]]),
        (
            "x",
            (-0.5, 0.5),
            None,
            None,
            [
                [
                    particle_list_positions[0][0],
                    particle_list_positions[0][2],
                    particle_list_positions[0][3],
                ]
            ],
        ),
        ("y", (0.5, None), None, None, [[particle_list_positions[0][2]]]),
        (
            "z",
            (None, 0.5),
            None,
            None,
            [
                [
                    particle_list_positions[0][0],
                    particle_list_positions[0][1],
                    particle_list_positions[0][2],
                ]
            ],
        ),
        # Test cases for error conditions
        ("t", (None, None), None, ValueError, None),
        ("t", (1.5, 0.5), UserWarning, None, [[particle_list_positions[0][0]]]),
        ("t", (0.5,), None, TypeError, None),
        ("t", ("a", 1.5), None, ValueError, None),
        ("w", (0.5, 1.5), None, ValueError, None),
        ("x", (1.5, 0.5), UserWarning, None, [[particle_list_positions[0][1]]]),
    ]

    for (
        dim,
        cut_value_tuple,
        expected_warning,
        expected_error,
        expected_result,
    ) in test_cases:
        if expected_warning:
            with pytest.warns(expected_warning):
                result = spacetime_cut(
                    particle_list_positions, dim, cut_value_tuple
                )
                assert result == expected_result

        elif expected_error:
            with pytest.raises(expected_error):
                result = spacetime_cut(
                    particle_list_positions, dim, cut_value_tuple
                )

        else:
            # Apply the spacetime cut
            result = spacetime_cut(
                particle_list_positions, dim, cut_value_tuple
            )
            # Assert the result matches the expected outcome
            assert result == expected_result


@pytest.fixture
def particle_list_pt():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.px = i
        p.py = 0.0
        particle_list.append(p)
    return [particle_list]


def test_pT_cut(particle_list_pt):
    test_cases = [
        # Test cases for valid input
        ((0.5, 1.5), None, None, [[particle_list_pt[0][1]]]),
        (
            (2.5, None),
            None,
            None,
            [[particle_list_pt[0][3], particle_list_pt[0][4]]],
        ),
        (
            (None, 3.5),
            None,
            None,
            [
                [
                    particle_list_pt[0][0],
                    particle_list_pt[0][1],
                    particle_list_pt[0][2],
                    particle_list_pt[0][3],
                ]
            ],
        ),
        # Test cases for error conditions
        ((None, None), None, ValueError, None),
        ((-1, 3), None, ValueError, None),
        (("a", 3), None, ValueError, None),
        (
            (3, 2),
            UserWarning,
            None,
            [[particle_list_pt[0][2], particle_list_pt[0][3]]],
        ),
        ((None, None, None), None, TypeError, None),
    ]

    for (
        cut_value_tuple,
        expected_warning,
        expected_error,
        expected_result,
    ) in test_cases:
        if expected_warning:
            with pytest.warns(expected_warning):
                result = pT_cut(particle_list_pt, cut_value_tuple)
                assert result == expected_result

        elif expected_error:
            with pytest.raises(expected_error):
                result = pT_cut(particle_list_pt, cut_value_tuple)

        else:
            # Apply the pT_cut
            result = pT_cut(particle_list_pt, cut_value_tuple)
            # Assert the result matches the expected outcome
            assert result == expected_result


@pytest.fixture
def particle_list_mT():
    particle_list = []
    mT_E_pz_pairs = [
        (3, 5, 4),
        (4, 5, 3),
        (5, 13, 12),
        (6, 10, 8),
        (7, 25, 24),
    ]
    for m_T, energy, p_z in mT_E_pz_pairs:
        p = Particle()
        p.E = energy
        p.pz = p_z
        particle_list.append(p)
    return [particle_list]


def test_mT_cut(particle_list_mT):
    # fmt: off
    test_cases = [
        # Test cases for valid input
        ((2.5, 3.5), None, None, [[particle_list_mT[0][0]]]),
        ((5.5, None), None, None, [[particle_list_mT[0][3], particle_list_mT[0][4]]]),
        ((None, 6.5), None, None, [[particle_list_mT[0][0],
                                    particle_list_mT[0][1],
                                    particle_list_mT[0][2],
                                    particle_list_mT[0][3]]]),
        # Test cases for error conditions
        ((None, None), None, ValueError, None),
        ((-1, 6), None, ValueError, None),
        (("a", 5), None, ValueError, None),
        ((5.7, 3.3), UserWarning, None, [[particle_list_mT[0][1], particle_list_mT[0][2]]],),
        ((None, None, None), None, TypeError, None),
    ]
    # fmt: on

    for (
        cut_value_tuple,
        expected_warning,
        expected_error,
        expected_result,
    ) in test_cases:
        if expected_warning:
            with pytest.warns(expected_warning):
                result = mT_cut(particle_list_mT, cut_value_tuple)
                assert result == expected_result

        elif expected_error:
            with pytest.raises(expected_error):
                result = mT_cut(particle_list_mT, cut_value_tuple)

        else:
            # Apply the mT_cut
            result = mT_cut(particle_list_mT, cut_value_tuple)
            # Assert the result matches the expected outcome
            assert result == expected_result


@pytest.fixture
def particle_list_rapidity():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.E = 10.0
        p.pz = i
        particle_list.append(p)
    return [particle_list]


def test_rapidity_cut(particle_list_rapidity):
    test_cases = [
        # Test cases for valid input
        (0.05, None, None, [[particle_list_rapidity[0][0]]]),
        ((-0.05, 0.05), None, None, [[particle_list_rapidity[0][0]]]),
        # Test cases for invalid input
        ((0.1, 1.0, 2.0), None, TypeError, None),
        (True, None, TypeError, None),
        ((1.0, "a"), None, ValueError, None),
        ((1.0, 0.5), UserWarning, None, None),
    ]

    for (
        cut_value,
        expected_warning,
        expected_error,
        expected_result,
    ) in test_cases:
        if expected_warning:
            with pytest.warns(expected_warning):
                result = rapidity_cut(particle_list_rapidity, cut_value)
        elif expected_error:
            with pytest.raises(expected_error):
                result = rapidity_cut(particle_list_rapidity, cut_value)
        else:
            result = rapidity_cut(particle_list_rapidity, cut_value)
            assert result == expected_result


@pytest.fixture
def particle_list_pseudorapidity():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.px = 1.0
        p.py = 0.0
        p.pz = i
        particle_list.append(p)
    return [particle_list]


def test_pseudorapidity_cut(particle_list_pseudorapidity):
    test_cases = [
        # Test cases for valid input
        (0.5, None, None, [[particle_list_pseudorapidity[0][0]]]),
        ((-0.5, 0.5), None, None, [[particle_list_pseudorapidity[0][0]]]),
        # Test cases for invalid input
        ((0.1, 1.0, 2.0), None, TypeError, None),
        (True, None, TypeError, None),
        ((1.0, "a"), None, ValueError, None),
        ((1.0, 0.5), UserWarning, None, None),
    ]

    for (
        cut_value,
        expected_warning,
        expected_error,
        expected_result,
    ) in test_cases:
        if expected_warning:
            with pytest.warns(expected_warning):
                result = pseudorapidity_cut(
                    particle_list_pseudorapidity, cut_value
                )
        elif expected_error:
            with pytest.raises(expected_error):
                result = pseudorapidity_cut(
                    particle_list_pseudorapidity, cut_value
                )
        else:
            result = pseudorapidity_cut(particle_list_pseudorapidity, cut_value)
            assert result == expected_result


@pytest.fixture
def particle_list_spacetime_rapidity():
    particle_list = []
    for i in range(5):
        p = Particle()
        p.t = i + 1
        p.z = i
        particle_list.append(p)
    return [particle_list]


def test_spacetime_rapidity_cut(particle_list_spacetime_rapidity):
    test_cases = [
        # Test cases for valid input
        (0.5, None, None, [[particle_list_spacetime_rapidity[0][0]]]),
        ((-0.5, 0.5), None, None, [[particle_list_spacetime_rapidity[0][0]]]),
        # Test cases for invalid input
        ((0.1, 1.0, 2.0), None, TypeError, None),
        (True, None, TypeError, None),
        ((1.0, "a"), None, ValueError, None),
        ((1.0, 0.5), UserWarning, None, None),
    ]

    for (
        cut_value,
        expected_warning,
        expected_error,
        expected_result,
    ) in test_cases:
        if expected_warning:
            with pytest.warns(expected_warning):
                result = spacetime_rapidity_cut(
                    particle_list_spacetime_rapidity, cut_value
                )
        elif expected_error:
            with pytest.raises(expected_error):
                result = spacetime_rapidity_cut(
                    particle_list_spacetime_rapidity, cut_value
                )
        else:
            result = spacetime_rapidity_cut(
                particle_list_spacetime_rapidity, cut_value
            )
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
    assert multiplicity_cut(particle_list_multiplicity, (7, None)) == [
        particle_list_multiplicity[1]
    ]
    assert multiplicity_cut(particle_list_multiplicity, (5, 10)) == [
        particle_list_multiplicity[0]
    ]
    assert multiplicity_cut(particle_list_multiplicity, (11, None)) == [[]]

    # Test cases for invalid input
    with pytest.raises(TypeError):
        multiplicity_cut(particle_list_multiplicity, cut_value=(-3.5, 4))

    with pytest.raises(TypeError):
        multiplicity_cut(particle_list_multiplicity, cut_value=(0, "a"))

    # The call with switched cut limits causes a warning which need to be caught
    with pytest.warns(UserWarning) as record:
        result = multiplicity_cut(particle_list_multiplicity, (10, 5))
    assert len(record) == 1
    assert (
        "Lower limit 10 is greater than upper limit 5. Switched order is assumed in the following."
        in str(record[0].message)
    )
    # Assert correct result
    assert result == [particle_list_multiplicity[0]]


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
        particle_status(particle_list_status, status_list="invalid")

    with pytest.raises(TypeError):
        particle_status(particle_list_status, status_list=[0, "invalid"])
