#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
from sparkx.Utilities import pdg_to_latex
import pytest

def test_pdg_to_latex_invalid_id():
    pdg_id = 9999  # An invalid PDG ID
    with pytest.raises(ValueError):
        pdg_to_latex(pdg_id)

def test_pdg_to_latex_single_id():
    pdg_id = 2112
    expected_latex = [r'n']
    assert pdg_to_latex(pdg_id) == expected_latex

def test_pdg_to_latex_multiple_ids():
    pdg_ids = [-3112, 4232]
    expected_latex = [r'\overline{\Sigma}^{+}', r'\Xi_{c}^{+}']
    assert pdg_to_latex(pdg_ids) == expected_latex