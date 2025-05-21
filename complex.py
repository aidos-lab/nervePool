"""
Simplicial Complex class with auxillary functions for pooling
Sarah McGuire 2022
"""

from math import comb

import numpy as np
import torch

from boundary import (
    Boundaries,
    Simplices,
    adjacency_from_boundaries,
    boundary_from_simplices,
    simplices_from_adjacencies,
)


class SComplex:
    def __init__(self, simplices=None, boundaries=None, label: int | None = None):
        self.label = label
        if simplices is not None and boundaries is not None:
            raise ValueError("Something went wrong")

        # List of simplices
        if simplices is not None:
            self.dim = len(simplices) - 1
            self.simplices = Simplices(*simplices)
            self.boundaries = boundary_from_simplices(self.simplices, self.dim)
            self.adjacencies = adjacency_from_boundaries(self.boundaries, self.dim)

            # TODO: Make boundaries non-oriented.

        # Array of boundary maps
        if boundaries is not None:
            self.dim = len(boundaries)
            self.boundaries = Boundaries(*boundaries)

            # TODO: Make boundaries non-oriented.

            self.adjacencies = adjacency_from_boundaries(self.boundaries, self.dim)
            self.simplices = simplices_from_adjacencies(self.adjacencies, self.dim)


def right_function(Scol0, SC):
    """
    NervePool right update function.
    Uses element-wise multiplication of cluster columns $\tilde{U_i}$ to compute intersections of cover elements.
    Full S matrix is row normalized.
    Output: S_p block diagonal matrices.
    """
    Scol0 = Scol0.numpy()

    # New Edges block column:
    # loop through all possible pairs of meta vertices to get new edge info
    n_nodes_new = Scol0.shape[1]
    Scol1 = np.zeros([Scol0.shape[0], comb(n_nodes_new, 2)])
    col = 0
    for i in range(0, n_nodes_new):
        for j in range(i + 1, n_nodes_new):
            Scol1[:, col] = Scol0[:, i] * Scol0[:, j]
            col += 1

    # Remove edges that are not in pooled complex (all zero cols)
    Scol1 = np.delete(Scol1, np.argwhere(np.all(Scol1[..., :] == 0, axis=0)), axis=1)

    # New Cycles block column:
    # loop through all possible triples of meta vertices to get new cycle info
    Scol0a = Scol0[len(SC.simplices.edges) :, :]
    Scol2 = np.zeros([Scol0a.shape[0], comb(n_nodes_new, 3)])
    col = 0
    for i in range(0, n_nodes_new):
        for j in range(i + 1, n_nodes_new):
            for k in range(j + 1, n_nodes_new):
                Scol2[:, col] = Scol0a[:, i] * Scol0a[:, j] * Scol0a[:, k]
                col += 1
    # Remove cycles that are not in pooled complex (all zero cols)
    Scol2 = np.delete(Scol2, np.argwhere(np.all(Scol2[..., :] == 0, axis=0)), axis=1)

    # New Tetra block column:
    # loop through all possible quadruplets of meta vertices to get new tetra info
    Scol0b = Scol0a[len(SC.simplices.cycles) :, :]
    Scol3 = np.zeros([Scol0b.shape[0], comb(n_nodes_new, 4)])
    col = 0
    for i in range(0, n_nodes_new):
        for j in range(i + 1, n_nodes_new):
            for k in range(j + 1, n_nodes_new):
                for l in range(k + 1, n_nodes_new):
                    Scol3[:, col] = (
                        Scol0b[:, i] * Scol0b[:, j] * Scol0b[:, k] * Scol0b[:, l]
                    )
                    col += 1

    # Remove tetra that are not in pooled complex (all zero cols)
    Scol3 = np.delete(Scol3, np.argwhere(np.all(Scol3[..., :] == 0, axis=0)), axis=1)
    # Normalize rows of S and select the diagonal sub-blocks for pooling
    if SC.dim >= 1 and Scol1.size != 0:
        S1 = Scol1[: len(SC.simplices.edges), :]
        Srow1 = np.concatenate((Scol0[: len(SC.simplices.edges)], S1), axis=1)
        Srow1_norm = Srow1 / Srow1.sum(axis=1)[:, np.newaxis]
        S1_norm = Srow1_norm[:, n_nodes_new:]
    else:
        S1_norm = None
    if SC.dim >= 2 and Scol2.size != 0:
        S2 = Scol2[: len(SC.simplices.cycles), :]
        idx_start = len(SC.simplices.edges)
        idx_end = idx_start + len(SC.simplices.cycles)
        Srow2 = np.concatenate(
            (Scol0[idx_start:idx_end], Scol1[idx_start:idx_end], S2), axis=1
        )
        Srow2_norm = Srow2 / Srow2.sum(axis=1)[:, np.newaxis]
        S2_norm = Srow2_norm[:, Scol0.shape[1] + Scol1.shape[1] :]
    else:
        S2_norm = None
    if SC.dim >= 3 and Scol3.size != 0:
        S3 = Scol3[: len(SC.simplices.tetra), :]
        idx_start = len(SC.simplices.cycles) + len(SC.simplices.edges)
        Srow3 = np.concatenate(
            (
                Scol0[idx_start:],
                Scol1[idx_start:],
                Scol2[len(SC.simplices.cycles) :],
                S3,
            ),
            axis=1,
        )
        Srow3_norm = Srow3 / Srow3.sum(axis=1)[:, np.newaxis]
        S3_norm = Srow3_norm[:, Scol0.shape[1] + Scol1.shape[1] + Scol2.shape[1] :]
    else:
        S3_norm = None

    if S1_norm is not None:
        S1_norm = torch.tensor(S1_norm, dtype=torch.float)
    if S2_norm is not None:
        S2_norm = torch.tensor(S2_norm, dtype=torch.float)
    if S3_norm is not None:
        S3_norm = torch.tensor(S3_norm, dtype=torch.float)
    return S1_norm, S2_norm, S3_norm


def down_function(S0, SC):
    """
    NervePool down update function.
    Extends vertex clusters $U_i$ to $\tilde(U_i)$.
    Columns are the union of stars of vertices for that cluster.
    """
    n = S0.shape[0]
    n_new = S0.shape[1]
    S01 = []
    S02 = []
    S03 = []
    for e in SC.simplices.edges:
        edge_arr = np.zeros(n_new)
        v0 = SC.simplices.nodes.index(e[0])
        v1 = SC.simplices.nodes.index(e[1])
        for v in range(n_new):
            if S0[v0, v] > 0 or S0[v1, v] > 0:
                edge_arr[v] = 1
        S01.append(edge_arr)

    for c in SC.simplices.cycles:
        cyc_arr = np.zeros(n_new)
        v0 = SC.simplices.nodes.index(c[0])
        v1 = SC.simplices.nodes.index(c[1])
        v2 = SC.simplices.nodes.index(c[2])
        for v in range(n_new):
            if S0[v0, v] > 0 or S0[v1, v] > 0 or S0[v2, v] > 0:
                cyc_arr[v] = 1
        S02.append(cyc_arr)

    for t in SC.simplices.tetra:
        t_arr = np.zeros(n_new)
        v0 = SC.simplices.nodes.index(t[0])
        v1 = SC.simplices.nodes.index(t[1])
        v2 = SC.simplices.nodes.index(t[2])
        v3 = SC.simplices.nodes.index(t[3])
        for v in range(n_new):
            if S0[v0, v] > 0 or S0[v1, v] > 0 or S0[v2, v] > 0 or S0[v3, v] > 0:
                t_arr[v] = 1
        S03.append(t_arr)

    if SC.dim >= 1:
        S01 = np.array(S01)
    else:
        S01 = None

    if SC.dim >= 2:
        S02 = np.array(S02)
    else:
        S02 = None

    if SC.dim >= 3:
        S03 = np.array(S03)
    else:
        S03 = None

    col0 = np.vstack([S01, S02, S03])
    return torch.tensor(col0)


def pool_complex(SC, S0):
    """
    Function to pool a simplicial complex using a partition of vertices
    Args:
        - SC : SComplex object to be pooled
        - S0 : array of size |v| x |v|', a partition of vertices

    Output:
        - SCpooled : SComplex object of the pooled complex
    """
    if S0.shape[0] == 0 or S0.shape[1] == 0:
        raise ValueError("Vertex cluster assignment matrix must be of size |v|x|v|")
    if S0.shape[0] != SC.adjacencies.A0.shape[0]:
        raise ValueError(
            f"Vertex cluster assignment size must match the number of vertices of the complex, "
            f"expected {SC.adjacencies.A0.shape}, but received {S0.shape}"
        )

    # Extend S0 to full S block matrix
    col0 = down_function(S0, SC)
    S1, S2, S3 = right_function(col0, SC)
    # print('S matrices are:', S1,'\n', S2, '\n', S3)
    # Use diagonal sub-blocks f S to pool boundary matrices

    BList = []
    if S1 is not None:
        BList.append(torch.abs(torch.matmul(torch.matmul(S0.T, SC.boundaries.B1), S1)))
    if S2 is not None:
        BList.append(torch.abs(torch.matmul(torch.matmul(S1.T, SC.boundaries.B2), S2)))
    if S3 is not None:
        BList.append(torch.abs(torch.matmul(torch.matmul(S2.T, SC.boundaries.B3), S3)))

    # Use new boundary matrices to construct pooled complex ... UNFINISHED
    Bds_new = tuple(BList)
    print(Bds_new)
    return SComplex(boundaries=Bds_new)
