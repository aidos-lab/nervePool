"""
Simplicial Complex class with auxillary functions for pooling
"""

import string
from math import comb

import numpy as np


class SComplex:
    def __init__(self, *args, dim: int = None, label: int = None):
        self.simplices = None
        self.boundaries = None

        for info in args:
            if isinstance(info, list):  # Input info is the list of simplices
                if len(info) == 0:
                    raise ValueError(
                        "List of simplices must contain at least one simplex."
                    )
                if dim is None:
                    dim = len(info) - 1
                if len(info) < dim + 1:
                    raise ValueError(
                        f"Missing simplices for the specified complex dim, "
                        f"expected {dim + 1}, but received {len(info)}"
                    )
                self.simplices = {i: info[i] for i in range(dim + 1)}
                self.nodes = info[0]
                if dim >= 1:
                    self.edges = info[1]
                else:
                    self.edges = None
                if dim >= 2:
                    self.cycles = info[2]
                else:
                    self.cycles = None
                if dim >= 3:
                    self.tetra = info[3]
                else:
                    self.tetra = None

            elif isinstance(
                info, np.ndarray
            ):  # Input info is the array of boundary matrices

                if len(info) == 0:
                    raise ValueError("Array of boundaries must contain at least B1.")
                if dim is None:
                    dim = len(info)
                if len(info) < dim:
                    raise ValueError(
                        f"Missing simplices for the specified complex dim, "
                        f"expected {dim}, but received {len(info)}"
                    )
                self.boundaries = {i + 1: info[i] for i in range(dim)}
                self.B1 = info[0]
                if dim >= 1:
                    self.B2 = info[1]
                else:
                    self.B2 = None
                if dim >= 2:
                    self.B3 = info[2]
                else:
                    self.B3 = None
            else:
                raise ValueError(
                    "Input arg type not supported. Use list of simplices or np.ndarray of boundary matrices to define the simplicial complex."
                )

        self.dim = dim
        self.label = label

        self._buildBoundaries()
        # Assuming non-oriented boundary matrices
        for b in self.boundaries:
            self.boundaries[b] = abs(self.boundaries[b])

        self._constructAdj()
        # self._buildSimplexListBDS()
        self._buildSimplexListADJ()
        # print("Simplices: \n", self.simplices)

    def _buildBoundaries(self):
        scomplex = self
        if scomplex.boundaries is None:
            # print("Using list of simplices to construct boundaries...")
            scomplex.B0 = None
            # Build B1
            if scomplex.dim >= 1:
                scomplex.B1 = np.zeros([len(self.simplices[0]), len(self.simplices[1])])
                for v0 in range(len(self.simplices[0])):
                    for v1 in range(len(self.simplices[0])):
                        simplex_up = self.simplices[0][v0] + self.simplices[0][v1]
                        if simplex_up in self.simplices[1]:
                            up_idx = self.simplices[1].index(simplex_up)
                            scomplex.B1[v0, up_idx] = 1
                            scomplex.B1[v1, up_idx] = 1
            else:
                scomplex.B1 = None

            # Build B2
            if scomplex.dim >= 2:
                scomplex.B2 = np.zeros([len(self.simplices[1]), len(self.simplices[2])])
                for cyc in range(len(self.simplices[2])):
                    v1 = self.simplices[2][cyc][0]
                    v2 = self.simplices[2][cyc][1]
                    v3 = self.simplices[2][cyc][2]
                    e1_idx = self.simplices[1].index(v1 + v2)
                    e2_idx = self.simplices[1].index(v2 + v3)
                    e3_idx = self.simplices[1].index(v1 + v3)
                    scomplex.B2[e1_idx, cyc] = 1
                    scomplex.B2[e2_idx, cyc] = 1
                    scomplex.B2[e3_idx, cyc] = 1
            else:
                scomplex.B2 = None

            # Build B3
            if scomplex.dim >= 3:
                scomplex.B3 = np.zeros([len(self.simplices[2]), len(self.simplices[3])])
                for t in range(len(self.simplices[3])):
                    v1 = self.simplices[3][t][0]
                    v2 = self.simplices[3][t][1]
                    v3 = self.simplices[3][t][2]
                    v4 = self.simplices[3][t][3]
                    c1_idx = self.simplices[2].index(v1 + v2 + v3)
                    c2_idx = self.simplices[2].index(v1 + v2 + v4)
                    c3_idx = self.simplices[2].index(v2 + v3 + v4)
                    c4_idx = self.simplices[2].index(v1 + v3 + v4)
                    scomplex.B3[c1_idx, t] = 1
                    scomplex.B3[c2_idx, t] = 1
                    scomplex.B3[c3_idx, t] = 1
                    scomplex.B3[c4_idx, t] = 1
            else:
                scomplex.B3 = None
            allBoundaries = [scomplex.B1, scomplex.B2, scomplex.B3]
            scomplex.boundaries = {i + 1: allBoundaries[i] for i in range(scomplex.dim)}

    def _buildSimplexListBDS(self):
        """
        Use boundary matrices to construct the corresponding list of simplices which represent the complex
        """
        scomplex = self
        if scomplex.simplices is None:
            # print("Using list of boundaries to construct simplices...")
            # Node list
            n_nodes = scomplex.B1.shape[0]
            scomplex.nodes = [str(x) for x in (string.ascii_lowercase[0:n_nodes])]

            # Build edge list
            if scomplex.dim >= 1:
                scomplex.edges = []
                n_edges = scomplex.B1.shape[1]
                for e in range(n_edges):
                    col = scomplex.B1[:, e]
                    # each edge has exactly 2 nodes on its boundary- a head and a tail
                    head_idx = np.where(col > 0)[0][0]
                    tail_idx = np.where(col < 0)[0][0]
                    scomplex.edges.append(
                        scomplex.nodes[head_idx] + scomplex.nodes[tail_idx]
                    )
            else:
                scomplex.edges = None

            # Build cycle list
            if scomplex.dim >= 2:
                scomplex.cycles = []
                n_cycles = scomplex.B2.shape[1]
                for cycle in range(n_cycles):
                    col = scomplex.B2[:, cycle]
                    # each cycle has exactly 2 edges on its boundary- 2 positive, 1 negative oriented
                    # Extract the 3 nodes labels that make up this cycle
                    pos_idx = np.where(col > 0)[0][0]
                    neg_idx = np.where(col < 0)[0][0]
                    scomplex.cycles.append(
                        scomplex.edges[pos_idx] + scomplex.edges[neg_idx][1]
                    )
            else:
                scomplex.cycles = None

            # Build tetrahedra list
            # print("scomplex.dim =", scomplex.dim)
            if scomplex.dim >= 3:
                scomplex.tetra = []
                n_tetra = scomplex.B3.shape[1]
                for tetra in range(n_tetra):
                    col = scomplex.B3[:, tetra]
                    # each tetrahedra has exactly 4 cycles on its boundary- 2 positive, 2 negative oriented
                    # Extract the 4 nodes labels that make up this cycle
                    pos_idx = np.where(col > 0)[0][0]
                    neg_idx = np.where(col < 0)[0][0]
                    node_idxs = [
                        scomplex.nodes.index(x)
                        for x in set(
                            scomplex.cycles[pos_idx] + scomplex.cycles[neg_idx]
                        )
                    ]
                    node_idxs.sort()
                    scomplex.tetra.append(
                        "".join([scomplex.nodes[x] for x in node_idxs])
                    )

            else:
                scomplex.tetra = None
            allsimplices = [
                scomplex.nodes,
                scomplex.edges,
                scomplex.cycles,
                scomplex.tetra,
            ]
            scomplex.simplices = {i: allsimplices[i] for i in range(scomplex.dim + 1)}

    def _buildSimplexListADJ(self):
        """
        Use adjacency matrices to construct the corresponding list of simplices which represent the complex
        """
        if self.simplices is None:
            # print("Using list of adjacency matrices to construct simplices...")
            # Node list
            n_nodes = self.A0.shape[0]
            self.nodes = [str(x) for x in (string.ascii_lowercase[0:n_nodes])]

            # Build edge list
            if self.dim >= 1:
                self.edges = []
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        if self.A0[i, j] > 0:
                            self.edges.append(self.nodes[i] + self.nodes[j])
                n_edges = len(self.edges)
            else:
                self.edges = None

            # Build triangle list  (cycles)
            if self.dim >= 2:
                self.cycles = []
                for i in range(n_edges):
                    for j in range(i + 1, n_edges):
                        for k in range(j + 1, n_edges):
                            if (
                                self.A1[i, j] > 0
                                and self.A1[i, k] > 0
                                and self.A1[j, k] > 0
                            ):
                                e1 = self.edges[i]
                                e2 = self.edges[j]
                                e3 = self.edges[k]
                                a = list(set(list(e1) + list(e2) + list(e3)))
                                if len(a) == 3:
                                    self.cycles.append("".join([str(i) for i in a]))
                n_cycles = len(self.cycles)
            else:
                self.cycles = None

            # Build tetrahedra list
            if self.dim >= 3:
                self.tetra = []
                for i in range(n_cycles):
                    for j in range(i + 1, n_cycles):
                        for k in range(j + 1, n_cycles):
                            for l in range(k + 1, n_cycles):
                                if (
                                    self.A2[i, j] > 0
                                    and self.A2[i, k] > 0
                                    and self.A2[i, l] > 0
                                    and self.A2[j, k] > 0
                                    and self.A2[j, l] > 0
                                    and self.A2[k, l] > 0
                                ):
                                    t1 = self.cycles[i]
                                    t2 = self.cycles[j]
                                    t3 = self.cycles[k]
                                    t4 = self.cycles[l]
                                    a = list(
                                        set(list(t1) + list(t2) + list(t3) + list(t4))
                                    )
                                    if len(a) == 4:
                                        self.tetra.append("".join([str(i) for i in a]))
                n_tetra = len(self.tetra)
            else:
                self.tetra = None

            allsimplices = [self.nodes, self.edges, self.cycles, self.tetra]
            self.simplices = {i: allsimplices[i] for i in range(self.dim + 1)}

    def _constructAdj(self):
        """
        Use boundary matrices to construct adjacency matrices via A = |D - (B * B^T)| or A = B * B^T
        """
        scomplex = self
        # A0
        if scomplex.dim >= 1:
            D0 = np.diag(np.sum(np.abs(scomplex.B1), axis=1))
            # print('D0 is:', D0)
            scomplex.A0 = np.abs(D0 - np.matmul(scomplex.B1, scomplex.B1.T))
            # scomplex.A0 = np.matmul(scomplex.B1,scomplex.B1.T)
        else:
            scomplex.A0 = None
        # A1
        if scomplex.dim >= 2:
            D1 = np.diag(np.sum(np.abs(scomplex.B2), axis=1))
            # print('D1 is:', D1)
            scomplex.A1 = np.abs(D1 - np.matmul(scomplex.B2, scomplex.B2.T))
            # scomplex.A1 = np.matmul(scomplex.B2,scomplex.B2.T)
        else:
            scomplex.A1 = None
        # A2
        if scomplex.dim >= 3:
            D2 = np.diag(np.sum(np.abs(scomplex.B3), axis=1))
            # print('D2 is:', D2)
            scomplex.A2 = np.abs(D2 - np.matmul(scomplex.B3, scomplex.B3.T))
            # scomplex.A2 = np.matmul(scomplex.B3,scomplex.B3.T)
        else:
            scomplex.A2 = None


def right_function(Scol0, SC):
    """
    NervePool right update function.
    Uses element-wise multiplication of cluster columns $\tilde{U_i}$ to compute intersections of cover elements.
    Full S matrix is row normalized.
    Output: S_p block diagonal matrices.
    """
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
    Scol0a = Scol0[len(SC.edges) :, :]
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
    Scol0b = Scol0a[len(SC.cycles) :, :]
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
        S1 = Scol1[: len(SC.edges), :]
        Srow1 = np.concatenate((Scol0[: len(SC.edges)], S1), axis=1)
        Srow1_norm = Srow1 / Srow1.sum(axis=1)[:, np.newaxis]
        S1_norm = Srow1_norm[:, n_nodes_new:]
    else:
        S1_norm = None
    if SC.dim >= 2 and Scol2.size != 0:
        S2 = Scol2[: len(SC.cycles), :]
        idx_start = len(SC.edges)
        idx_end = idx_start + len(SC.cycles)
        Srow2 = np.concatenate(
            (Scol0[idx_start:idx_end], Scol1[idx_start:idx_end], S2), axis=1
        )
        Srow2_norm = Srow2 / Srow2.sum(axis=1)[:, np.newaxis]
        S2_norm = Srow2_norm[:, Scol0.shape[1] + Scol1.shape[1] :]
    else:
        S2_norm = None
    if SC.dim >= 3 and Scol3.size != 0:
        S3 = Scol3[: len(SC.tetra), :]
        idx_start = len(SC.cycles) + len(SC.edges)
        Srow3 = np.concatenate(
            (Scol0[idx_start:], Scol1[idx_start:], Scol2[len(SC.cycles) :], S3), axis=1
        )
        Srow3_norm = Srow3 / Srow3.sum(axis=1)[:, np.newaxis]
        S3_norm = Srow3_norm[:, Scol0.shape[1] + Scol1.shape[1] + Scol2.shape[1] :]
    else:
        S3_norm = None
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
    for e in SC.edges:
        edge_arr = np.zeros(n_new)
        v0 = SC.nodes.index(e[0])
        v1 = SC.nodes.index(e[1])
        for v in range(n_new):
            if S0[v0, v] > 0 or S0[v1, v] > 0:
                edge_arr[v] = 1
        S01.append(edge_arr)

    for c in SC.cycles:
        cyc_arr = np.zeros(n_new)
        v0 = SC.nodes.index(c[0])
        v1 = SC.nodes.index(c[1])
        v2 = SC.nodes.index(c[2])
        for v in range(n_new):
            if S0[v0, v] > 0 or S0[v1, v] > 0 or S0[v2, v] > 0:
                cyc_arr[v] = 1
        S02.append(cyc_arr)

    for t in SC.tetra:
        t_arr = np.zeros(n_new)
        v0 = SC.nodes.index(t[0])
        v1 = SC.nodes.index(t[1])
        v2 = SC.nodes.index(t[2])
        v3 = SC.nodes.index(t[3])
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
    return col0


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
    if S0.shape[0] != SC.A0.shape[0]:
        raise ValueError(
            f"Vertex cluster assignment size must match the number of vertices of the complex, "
            f"expected {SC.A0.shape}, but received {S0.shape}"
        )

    # Extend S0 to full S block matrix
    col0 = down_function(S0, SC)
    S1, S2, S3 = right_function(col0, SC)
    # print('S matrices are:', S1,'\n', S2, '\n', S3)
    # Use diagonal sub-blocks f S to pool boundary matrices
    if S1 is None:
        B1_new = None
    else:
        B1_new = np.abs(np.matmul(np.matmul(S0.T, SC.B1), S1))
    if S2 is None:
        B2_new = None
    else:
        B2_new = np.abs(np.matmul(np.matmul(S1.T, SC.B2), S2))
    if S3 is None:
        B3_new = None
    else:
        B3_new = np.abs(np.matmul(np.matmul(S2.T, SC.B3), S3))

    # Pooled complex dimension
    if B1_new is None:
        newdim = 0
    elif B2_new is None:
        newdim = 1
    elif B3_new is None:
        newdim = 2
    else:
        newdim = 3

    # Use new boundary matrices to construct pooled complex ... UNFINISHED
    Bds_new = np.array([B1_new, B2_new, B3_new], dtype=object)
    return SComplex(Bds_new, dim=newdim)
