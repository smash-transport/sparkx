# ===================================================
#
#    Copyright (c) 2023-2024
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================

import math

"""
This module contains helper functions and global variables that are used in 
the testing scripts.
"""

# Global variables
small_value = 1e-9


# Helper functions

# Converts a string to a float or an integer, if applicable.
# Otherwise, returns the original item.
def convert_str_to_number(item):
    try:
        if '.' in item or 'e' in item or 'E' in item:
            return float(item)
        else:
            return int(item)
    except ValueError:
        return item


# Converts all string elements in a nested list to their respective
# numerical types.
def convert_nested_list_to_numerical(nested_list):
    return [[convert_str_to_number(item) for item in sublist] for sublist in nested_list]

# Compares two floating-point numbers within a given tolerance.
def compare_floats(a, b, tol=small_value):
    return math.isclose(a, b, abs_tol=tol)

# Compares two nested lists element-wise, with a tolerance for
# floating-point comparisons assuming the lists contain numerical values.
def compare_nested_lists(list1, list2, tol=small_value):
    if len(list1) != len(list2):
        return False
    
    for sublist1, sublist2 in zip(list1, list2):
        if len(sublist1) != len(sublist2):
            return False
        for item1, item2 in zip(sublist1, sublist2):
            if isinstance(item1, float) and isinstance(item2, float):
                if not compare_floats(item1, item2, tol):
                    return False
            else:
                if item1 != item2:
                    return False
    return True