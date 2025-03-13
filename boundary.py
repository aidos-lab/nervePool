import string

import numpy as np


def newconstructAdj(B1, B2, B3, dim):
    if dim >= 1:
        D0 = np.diag(np.sum(np.abs(B1), axis=1))
        A0 = np.abs(D0 - np.matmul(B1, B1.T))
    else:
        A0 = None
    if dim >= 2:
        D1 = np.diag(np.sum(np.abs(B2), axis=1))
        A1 = np.abs(D1 - np.matmul(B2, B2.T))
    else:
        A1 = None
    if dim >= 3:
        D2 = np.diag(np.sum(np.abs(B3), axis=1))
        A2 = np.abs(D2 - np.matmul(B3, B3.T))
    else:
        A2 = None
    return A0, A1, A2


def constructAdj(scomplex):
    if scomplex.dim >= 1:
        D0 = np.diag(np.sum(np.abs(scomplex.B1), axis=1))
        scomplex.A0 = np.abs(D0 - np.matmul(scomplex.B1, scomplex.B1.T))
    else:
        scomplex.A0 = None
    if scomplex.dim >= 2:
        D1 = np.diag(np.sum(np.abs(scomplex.B2), axis=1))
        scomplex.A1 = np.abs(D1 - np.matmul(scomplex.B2, scomplex.B2.T))
    else:
        scomplex.A1 = None
    if scomplex.dim >= 3:
        D2 = np.diag(np.sum(np.abs(scomplex.B3), axis=1))
        scomplex.A2 = np.abs(D2 - np.matmul(scomplex.B3, scomplex.B3.T))
    else:
        scomplex.A2 = None
    return scomplex


def buildSimplexListADJ(scomplex):
    """
    Use adjacency matrices to construct the corresponding list of simplices which represent the complex
    """
    if scomplex.simplices is None:
        # print("Using list of adjacency matrices to construct simplices...")
        # Node list
        n_nodes = scomplex.A0.shape[0]
        scomplex.nodes = [str(x) for x in (string.ascii_lowercase[0:n_nodes])]

        # Build edge list
        if scomplex.dim >= 1:
            scomplex.edges = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if scomplex.A0[i, j] > 0:
                        scomplex.edges.append(scomplex.nodes[i] + scomplex.nodes[j])
            n_edges = len(scomplex.edges)
        else:
            scomplex.edges = None

        # Build triangle list  (cycles)
        if scomplex.dim >= 2:
            scomplex.cycles = []
            for i in range(n_edges):
                for j in range(i + 1, n_edges):
                    for k in range(j + 1, n_edges):
                        if (
                            scomplex.A1[i, j] > 0
                            and scomplex.A1[i, k] > 0
                            and scomplex.A1[j, k] > 0
                        ):
                            e1 = scomplex.edges[i]
                            e2 = scomplex.edges[j]
                            e3 = scomplex.edges[k]
                            a = list(set(list(e1) + list(e2) + list(e3)))
                            if len(a) == 3:
                                scomplex.cycles.append("".join([str(i) for i in a]))
            n_cycles = len(scomplex.cycles)
        else:
            scomplex.cycles = None

        # Build tetrahedra list
        if scomplex.dim >= 3:
            scomplex.tetra = []
            for i in range(n_cycles):
                for j in range(i + 1, n_cycles):
                    for k in range(j + 1, n_cycles):
                        for l in range(k + 1, n_cycles):
                            if (
                                scomplex.A2[i, j] > 0
                                and scomplex.A2[i, k] > 0
                                and scomplex.A2[i, l] > 0
                                and scomplex.A2[j, k] > 0
                                and scomplex.A2[j, l] > 0
                                and scomplex.A2[k, l] > 0
                            ):
                                t1 = scomplex.cycles[i]
                                t2 = scomplex.cycles[j]
                                t3 = scomplex.cycles[k]
                                t4 = scomplex.cycles[l]
                                a = list(set(list(t1) + list(t2) + list(t3) + list(t4)))
                                if len(a) == 4:
                                    scomplex.tetra.append("".join([str(i) for i in a]))
            n_tetra = len(scomplex.tetra)
        else:
            scomplex.tetra = None

        allsimplices = [scomplex.nodes, scomplex.edges, scomplex.cycles, scomplex.tetra]
        scomplex.simplices = {i: allsimplices[i] for i in range(scomplex.dim + 1)}
    return scomplex


def buildSimplexListBDS(scomplex):
    """
    Use boundary matrices to construct the corresponding list of simplices which represent the complex
    """
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
                    for x in set(scomplex.cycles[pos_idx] + scomplex.cycles[neg_idx])
                ]
                node_idxs.sort()
                scomplex.tetra.append("".join([scomplex.nodes[x] for x in node_idxs]))

        else:
            scomplex.tetra = None
        allsimplices = [
            scomplex.nodes,
            scomplex.edges,
            scomplex.cycles,
            scomplex.tetra,
        ]
        scomplex.simplices = {i: allsimplices[i] for i in range(scomplex.dim + 1)}
    return scomplex


def buildBoundaries(scomplex):
    if scomplex.boundaries is None:
        # print("Using list of simplices to construct boundaries...")
        scomplex.B0 = None
        # Build B1
        if scomplex.dim >= 1:
            scomplex.B1 = np.zeros(
                [len(scomplex.simplices[0]), len(scomplex.simplices[1])]
            )
            for v0 in range(len(scomplex.simplices[0])):
                for v1 in range(len(scomplex.simplices[0])):
                    simplex_up = scomplex.simplices[0][v0] + scomplex.simplices[0][v1]
                    if simplex_up in scomplex.simplices[1]:
                        up_idx = scomplex.simplices[1].index(simplex_up)
                        scomplex.B1[v0, up_idx] = 1
                        scomplex.B1[v1, up_idx] = 1
        else:
            scomplex.B1 = None

        # Build B2
        if scomplex.dim >= 2:
            scomplex.B2 = np.zeros(
                [len(scomplex.simplices[1]), len(scomplex.simplices[2])]
            )
            for cyc in range(len(scomplex.simplices[2])):
                v1 = scomplex.simplices[2][cyc][0]
                v2 = scomplex.simplices[2][cyc][1]
                v3 = scomplex.simplices[2][cyc][2]
                e1_idx = scomplex.simplices[1].index(v1 + v2)
                e2_idx = scomplex.simplices[1].index(v2 + v3)
                e3_idx = scomplex.simplices[1].index(v1 + v3)
                scomplex.B2[e1_idx, cyc] = 1
                scomplex.B2[e2_idx, cyc] = 1
                scomplex.B2[e3_idx, cyc] = 1
        else:
            scomplex.B2 = None

        # Build B3
        if scomplex.dim >= 3:
            scomplex.B3 = np.zeros(
                [len(scomplex.simplices[2]), len(scomplex.simplices[3])]
            )
            for t in range(len(scomplex.simplices[3])):
                v1 = scomplex.simplices[3][t][0]
                v2 = scomplex.simplices[3][t][1]
                v3 = scomplex.simplices[3][t][2]
                v4 = scomplex.simplices[3][t][3]
                c1_idx = scomplex.simplices[2].index(v1 + v2 + v3)
                c2_idx = scomplex.simplices[2].index(v1 + v2 + v4)
                c3_idx = scomplex.simplices[2].index(v2 + v3 + v4)
                c4_idx = scomplex.simplices[2].index(v1 + v3 + v4)
                scomplex.B3[c1_idx, t] = 1
                scomplex.B3[c2_idx, t] = 1
                scomplex.B3[c3_idx, t] = 1
                scomplex.B3[c4_idx, t] = 1
        else:
            scomplex.B3 = None
        allBoundaries = [scomplex.B1, scomplex.B2, scomplex.B3]
        scomplex.boundaries = {i + 1: allBoundaries[i] for i in range(scomplex.dim)}
    return scomplex
