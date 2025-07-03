import pickle

import numpy as np

from boundary import letter_simplices_to_numbers
from original_pooling.complex import SComplex, pool_complex
from original_pooling.originalcomplex import SComplex as OriginalSComplex
from original_pooling.originalcomplex import pool_complex as original_pool_complex

vertex_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
edge_list = [
    "ab",
    "ac",
    "bc",
    "cd",
    "cj",
    "de",
    "df",
    "dg",
    "ef",
    "gh",
    "gi",
    "gj",
    "gk",
    "gl",
    "hi",
    "jk",
    "jl",
    "kl",
]
triangle_list = ["abc", "gjk", "gjl", "gkl", "jkl"]
tetrahedron_list = ["gjkl"]

# Cluster Assignments
S0 = np.array(
    [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]
)

# Gather these lists into a single list
simplices = list([vertex_list, edge_list, triangle_list, tetrahedron_list])


def test_main():
    OSC1 = OriginalSComplex(simplices)
    # number_simplices = letter_simplices_to_numbers(simplices)
    SC1 = SComplex(simplices=simplices)

    np.testing.assert_equal(SC1.boundaries.B1, OSC1.B1, verbose=True)
    np.testing.assert_equal(SC1.boundaries.B2, OSC1.B2, verbose=True)
    np.testing.assert_equal(SC1.boundaries.B3, OSC1.B3, verbose=True)

    SIMP = (
        SC1.simplices.nodes,
        SC1.simplices.edges,
        SC1.simplices.cycles,
        SC1.simplices.tetra,
    )
    B = (
        SC1.boundaries.B1,
        SC1.boundaries.B2,
        SC1.boundaries.B3,
    )
    S = (
        SC1.adjacencies.A0,
        SC1.adjacencies.A1,
        SC1.adjacencies.A2,
        SC1.adjacencies.A3,
    )
    with open("./test/fixtures/boundaries.pkl", "wb") as f:
        pickle.dump(B, f)

    with open("./test/fixtures/adjacencies.pkl", "wb") as f:
        pickle.dump(S, f)

    with open("./test/fixtures/simplices.pkl", "wb") as f:
        pickle.dump(SIMP, f)

    OSC1_pooled = original_pool_complex(OSC1, S0)
    SC1_pooled = pool_complex(SC1, S0)

    np.testing.assert_equal(SC1_pooled.boundaries.B1, OSC1_pooled.B1, verbose=True)
    np.testing.assert_equal(SC1_pooled.boundaries.B2, OSC1_pooled.B2, verbose=True)
    np.testing.assert_equal(SC1_pooled.boundaries.B3, OSC1_pooled.B3, verbose=True)

    SIMP = (
        SC1_pooled.simplices.nodes,
        SC1_pooled.simplices.edges,
        SC1_pooled.simplices.cycles,
        SC1_pooled.simplices.tetra,
    )
    B = (
        SC1_pooled.boundaries.B1,
        SC1_pooled.boundaries.B2,
        SC1_pooled.boundaries.B3,
    )
    S = (
        SC1_pooled.adjacencies.A0,
        SC1_pooled.adjacencies.A1,
        SC1_pooled.adjacencies.A2,
        SC1_pooled.adjacencies.A3,
    )
    with open("./test/fixtures/pooled_boundaries.pkl", "wb") as f:
        pickle.dump(B, f)

    with open("./test/fixtures/pooled_adjacencies.pkl", "wb") as f:
        pickle.dump(S, f)

    with open("./test/fixtures/pooled_simplices.pkl", "wb") as f:
        pickle.dump(S, f)

    if SC1_pooled.boundaries.B1 is not None:
        np.testing.assert_allclose(
            SC1_pooled.boundaries.B1, OSC1_pooled.B1, verbose=True
        )

    if SC1_pooled.boundaries.B2 is not None:
        np.testing.assert_allclose(
            SC1_pooled.boundaries.B2, OSC1_pooled.B2, verbose=True
        )

    if SC1_pooled.boundaries.B3 is not None:
        np.testing.assert_allclose(
            SC1_pooled.boundaries.B3, OSC1_pooled.B3, verbose=True
        )

    if SC1_pooled.simplices.nodes is not None:
        print("Testing Nodes")
        all(x == y for x, y in zip(SC1_pooled.simplices.nodes, OSC1_pooled.nodes))

    if SC1_pooled.simplices.edges is not None:
        print("Testing Edges")
        all(x == y for x, y in zip(SC1_pooled.simplices.edges, OSC1_pooled.edges))

    if SC1_pooled.simplices.cycles is not None:
        print("Testing Cycles")
        all(x == y for x, y in zip(SC1_pooled.simplices.cycles, OSC1_pooled.cycles))

    if SC1_pooled.simplices.tetra is not None:
        print("Testing Tetra")
        all(x == y for x, y in zip(SC1_pooled.tetra, OSC1_pooled.tetra))
