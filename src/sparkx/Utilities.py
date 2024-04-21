#===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
#===================================================
    
import particle.data
import csv

def pdg_to_latex(pdg_id):
    """
    Converts a given PDG ID or a list of PDG IDs into the corresponding LaTeX
    formatted particle name as sts or a list of LaTeX formatted particle names
    as str which has the same order as the input PDG IDs

    Parameters
    ----------
    pdg_id : int / list
        PDG ID or a list of PDG IDs

    Returns
    -------
    latex_names : str / list
        LaTeX formatted particle name or list of LaTeX formatted particle names
        in the same order as the input PDG IDs.

    Examples
    --------

    .. highlight:: python
    .. code-block:: python
        :linenos:

        >>> from sparkx.Utilities import pdg_to_latex
        >>>
        >>> pdg_ids = [2112, -3112, 4232]
        >>>
        >>> latex_names = pdg_to_latex(pdg_ids)
        >>> print(latex_names)

        ['n', '\overline{\Sigma}^{+}', '\Xi_{c}^{+}']
    """

    if isinstance(pdg_id, int):
        pdg_id = [pdg_id]

    path = particle.data.basepath / "particle2022.csv"
    latex_names = [0]*len(pdg_id)

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter_row = 0

        for row in csv_reader:
            if counter_row < 2:
                pass
            elif counter_row >= 2:
                if int(row[0]) in pdg_id:
                    index = pdg_id.index(int(row[0]))
                    latex_names[index] = row[17]
            counter_row += 1

    if 0 in latex_names:
        raise ValueError('pdg_id contains invalid PDG ID')

    return latex_names
