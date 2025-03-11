import numpy as np

from complex import SComplex, pool_complex
from originalcomplex import SComplex as OriginalSComplex
from originalcomplex import pool_complex as original_pool_complex

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

OSC1 = OriginalSComplex(simplices)
SC1 = SComplex(simplices)

np.testing.assert_equal(SC1.B1, OSC1.B1, verbose=True)
np.testing.assert_equal(SC1.B2, OSC1.B2, verbose=True)
np.testing.assert_equal(SC1.B3, OSC1.B3, verbose=True)


OSC1_pooled = original_pool_complex(OSC1, S0)
SC1_pooled = pool_complex(SC1, S0)

np.testing.assert_equal(SC1_pooled.B1, OSC1_pooled.B1, verbose=True)
np.testing.assert_equal(SC1_pooled.B2, OSC1_pooled.B2, verbose=True)
np.testing.assert_equal(SC1_pooled.B3, OSC1_pooled.B3, verbose=True)

if SC1_pooled.B1 is not None:
    np.testing.assert_allclose(SC1_pooled.B1, OSC1_pooled.B1, verbose=True)

if SC1_pooled.B2 is not None:
    np.testing.assert_allclose(SC1_pooled.B2, OSC1_pooled.B2, verbose=True)

if SC1_pooled.B3 is not None:
    np.testing.assert_allclose(SC1_pooled.B3, OSC1_pooled.B3, verbose=True)


if SC1_pooled.nodes is not None:
    print("Testing Nodes")
    all(x == y for x, y in zip(SC1_pooled.nodes, OSC1_pooled.nodes))

if SC1_pooled.edges is not None:
    print("Testing Edges")
    all(x == y for x, y in zip(SC1_pooled.edges, OSC1_pooled.edges))

if SC1_pooled.cycles is not None:
    print("Testing Cycles")
    all(x == y for x, y in zip(SC1_pooled.cycles, OSC1_pooled.cycles))

if SC1_pooled.tetra is not None:
    print("Testing Tetra")
    all(x == y for x, y in zip(SC1_pooled.tetra, OSC1_pooled.tetra))
