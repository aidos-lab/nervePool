import pickle

import torch

from boundary import letter_simplices_to_numbers, number_simplices_to_letters
from complex import SComplex, pool_complex

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

with open("./test/fixtures/adjacencies.pkl", "rb") as f:
    adjacencies_np = pickle.load(f)
    adjacencies = []
    for a in adjacencies_np:
        if a is not None:
            adjacencies.append(torch.tensor(a))
        else:
            adjacencies.append(None)

# Pooled data.
with open("./test/fixtures/pooled_simplices.pkl", "rb") as f:
    # List of strings
    pooled_simplices = pickle.load(f)

with open("./test/fixtures/pooled_boundaries.pkl", "rb") as f:
    # Tuple of np arrays.
    boundaries_np = pickle.load(f)
    boundaries_pooled = []
    for b in boundaries_np:
        if b is not None:
            boundaries_pooled.append(torch.tensor(b))
        else:
            boundaries_pooled.append(None)

with open("./test/fixtures/pooled_adjacencies.pkl", "rb") as f:
    adjacencies_np = pickle.load(f)
    pooled_adjacencies = []
    for a in adjacencies_np:
        if a is not None:
            pooled_adjacencies.append(torch.tensor(a))
        else:
            pooled_adjacencies.append(None)


# def test_create_torch_sc_torch():
#     """Tests if the torch version outputs the same values as the numpy version."""
#     number_simplices = letter_simplices_to_numbers(simplices)
#     sc_torch = SComplex(simplices=number_simplices)
#
#     # Check that the new SC is the same as the numpy version.
#     # boundaries = [B1, B2, B3]
#     torch.equal(boundaries[0], sc_torch.boundaries.B1)
#     torch.equal(boundaries[1], sc_torch.boundaries.B2)
#     torch.equal(boundaries[2], sc_torch.boundaries.B3)
#
#     if isinstance(adjacencies[0], torch.Tensor) and isinstance(
#         sc_torch.adjacencies.A0, torch.Tensor
#     ):
#         torch.equal(adjacencies[0], sc_torch.adjacencies.A0)
#     elif adjacencies[0] is None and sc_torch.adjacencies.A0 is None:
#         pass
#     else:
#         raise ValueError("A0")
#
#     if isinstance(adjacencies[1], torch.Tensor) and isinstance(
#         sc_torch.adjacencies.A1, torch.Tensor
#     ):
#         torch.equal(adjacencies[1], sc_torch.adjacencies.A1)
#     elif adjacencies[1] is None and sc_torch.adjacencies.A1 is None:
#         pass
#     else:
#         raise ValueError("A1")
#
#     if isinstance(adjacencies[2], torch.Tensor) and isinstance(
#         sc_torch.adjacencies.A2, torch.Tensor
#     ):
#         torch.equal(adjacencies[2], sc_torch.adjacencies.A2)
#     elif adjacencies[2] is None and sc_torch.adjacencies.A2 is None:
#
#         pass
#     else:
#         raise ValueError("A2")
#
#     if isinstance(adjacencies[3], torch.Tensor) and isinstance(
#         sc_torch.adjacencies.A3, torch.Tensor
#     ):
#         torch.equal(adjacencies[3], sc_torch.adjacencies.A3)
#     elif adjacencies[3] is None and sc_torch.adjacencies.A3 is None:
#         pass
#     else:
#         raise ValueError("A3")
#
#
# def test_pool_torch_complex():
#     # Cluster Assignments
#     S0 = torch.tensor(
#         [
#             [1, 0, 0, 0],
#             [1, 0, 0, 0],
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 1, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [0, 0, 1, 0],
#             [0, 0, 1, 0],
#             [1, 0, 0, 0],
#             [0, 0, 0, 1],
#             [0, 0, 0, 1],
#         ],
#         dtype=torch.float,
#     )
#     sc_torch = SComplex(simplices=simplices)
#     sc_torch_pooled = pool_complex(sc_torch, S0)
#
#     # Check that the new SC is the same as the numpy version.
#     # boundaries = [B1, B2, B3]
#     torch.equal(sc_torch_pooled.boundaries.B1, boundaries_pooled[0])
#     torch.equal(sc_torch_pooled.boundaries.B2, boundaries_pooled[1])
#     assert sc_torch_pooled.boundaries.B3 == boundaries_pooled[2]
#
#     pass
