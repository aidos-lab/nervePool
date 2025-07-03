import pickle

import torch

from boundary import (
    Simplices,
    boundary_from_simplices,
    letter_simplices_to_numbers,
    number_simplices_to_letters,
)
from complex import SComplex

#######################################################################
### Load fixtures
#######################################################################

# Original input data.
with open("./test/fixtures/simplices.pkl", "rb") as f:
    # List of strings
    simplices = pickle.load(f)


with open("./test/fixtures/boundaries.pkl", "rb") as f:
    # Tuple of np arrays.
    boundaries = pickle.load(f)
    boundaries = [torch.tensor(b) for b in boundaries]


def test_create_torch_sc_torch():
    """Tests if the torch version outputs the same values as the numpy version."""
    number_simplices_list = letter_simplices_to_numbers(simplices)
    number_simplices = Simplices(
        nodes=number_simplices_list[0],
        edges=number_simplices_list[1],
        cycles=number_simplices_list[2],
        tetra=number_simplices_list[3],
    )
    boundary_from_simplices(number_simplices, dim=3)
