import string
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Simplices:
    nodes: torch.Tensor | None = None
    edges: torch.Tensor | None = None
    cycles: torch.Tensor | None = None
    tetra: torch.Tensor | None = None


@dataclass
class Adjacencies:
    A0: torch.Tensor | None = None
    A1: torch.Tensor | None = None
    A2: torch.Tensor | None = None
    A3: torch.Tensor | None = None


@dataclass
class Boundaries:
    B1: torch.Tensor | None = None
    B2: torch.Tensor | None = None
    B3: torch.Tensor | None = None


def adjacency_from_boundaries(boundaries: Boundaries, dim):
    if dim >= 1:
        D0 = torch.diag(torch.sum(torch.abs(boundaries.B1), axis=1))
        A0 = torch.abs(D0 - torch.matmul(boundaries.B1, boundaries.B1.T))
    else:
        A0 = None
    if dim >= 2:
        D1 = torch.diag(torch.sum(torch.abs(boundaries.B2), axis=1))
        A1 = torch.abs(D1 - torch.matmul(boundaries.B2, boundaries.B2.T))
    else:
        A1 = None
    if dim >= 3:
        D2 = torch.diag(torch.sum(torch.abs(boundaries.B3), axis=1))
        A2 = torch.abs(D2 - torch.matmul(boundaries.B3, boundaries.B3.T))
    else:
        A2 = None
    return Adjacencies(A0, A1, A2)


def constructAdj(scomplex):
    if scomplex.dim >= 1:
        D0 = torch.diag(torch.sum(torch.abs(scomplex.B1), axis=1))
        scomplex.A0 = torch.abs(D0 - torch.matmul(scomplex.B1, scomplex.B1.T))
    else:
        scomplex.A0 = None
    if scomplex.dim >= 2:
        D1 = torch.diag(torch.sum(torch.abs(scomplex.B2), axis=1))
        scomplex.A1 = torch.abs(D1 - torch.matmul(scomplex.B2, scomplex.B2.T))
    else:
        scomplex.A1 = None
    if scomplex.dim >= 3:
        D2 = torch.diag(torch.sum(torch.abs(scomplex.B3), axis=1))
        scomplex.A2 = torch.abs(D2 - torch.matmul(scomplex.B3, scomplex.B3.T))
    else:
        scomplex.A2 = None
    return scomplex


def simplices_from_adjacencies(adjacencies, dim):
    """
    Use adjacency matrices to construct the corresponding list of simplices which represent the complex
    """
    # print("Using list of adjacency matrices to construct simplices...")
    # Node list
    n_nodes = adjacencies.A0.shape[0]
    nodes = [str(x) for x in (string.ascii_lowercase[0:n_nodes])]

    # Build edge list
    if dim >= 1:
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adjacencies.A0[i, j] > 0:
                    edges.append(nodes[i] + nodes[j])
        n_edges = len(edges)
    else:
        edges = None

    # Build triangle list  (cycles)
    if dim >= 2:
        cycles = []
        for i in range(n_edges):
            for j in range(i + 1, n_edges):
                for k in range(j + 1, n_edges):
                    if (
                        adjacencies.A1[i, j] > 0
                        and adjacencies.A1[i, k] > 0
                        and adjacencies.A1[j, k] > 0
                    ):
                        e1 = edges[i]
                        e2 = edges[j]
                        e3 = edges[k]
                        a = list(set(list(e1) + list(e2) + list(e3)))
                        if len(a) == 3:
                            cycles.append("".join([str(i) for i in a]))
        n_cycles = len(cycles)
    else:
        cycles = None

    # Build tetrahedra list
    if dim >= 3:
        tetra = []
        for i in range(n_cycles):
            for j in range(i + 1, n_cycles):
                for k in range(j + 1, n_cycles):
                    for l in range(k + 1, n_cycles):
                        if (
                            adjacencies.A2[i, j] > 0
                            and adjacencies.A2[i, k] > 0
                            and adjacencies.A2[i, l] > 0
                            and adjacencies.A2[j, k] > 0
                            and adjacencies.A2[j, l] > 0
                            and adjacencies.A2[k, l] > 0
                        ):
                            t1 = cycles[i]
                            t2 = cycles[j]
                            t3 = cycles[k]
                            t4 = cycles[l]
                            a = list(set(list(t1) + list(t2) + list(t3) + list(t4)))
                            if len(a) == 4:
                                tetra.append("".join([str(i) for i in a]))
        n_tetra = len(tetra)
    else:
        tetra = None

    return Simplices(nodes, edges, cycles, tetra)


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


def boundary_from_simplices(simplices: Simplices, dim) -> Boundaries:
    B0 = None
    if dim >= 1:
        B1 = torch.zeros([len(simplices.nodes), len(simplices.edges)])
        for v0 in range(len(simplices.nodes)):
            for v1 in range(len(simplices.nodes)):
                simplex_up = simplices.nodes[v0] + simplices.nodes[v1]
                if simplex_up in simplices.edges:
                    up_idx = simplices.edges.index(simplex_up)
                    B1[v0, up_idx] = 1
                    B1[v1, up_idx] = 1
    else:
        B1 = None

    # Build B2
    if dim >= 2:
        B2 = torch.zeros([len(simplices.edges), len(simplices.cycles)])
        for cyc in range(len(simplices.cycles)):
            v1 = simplices.cycles[cyc][0]
            v2 = simplices.cycles[cyc][1]
            v3 = simplices.cycles[cyc][2]
            e1_idx = simplices.edges.index(v1 + v2)
            e2_idx = simplices.edges.index(v2 + v3)
            e3_idx = simplices.edges.index(v1 + v3)
            B2[e1_idx, cyc] = 1
            B2[e2_idx, cyc] = 1
            B2[e3_idx, cyc] = 1
    else:
        B2 = None

    # Build B3
    if dim >= 3:
        B3 = torch.zeros([len(simplices.cycles), len(simplices.tetra)])
        for t in range(len(simplices.tetra)):
            v1 = simplices.tetra[t][0]
            v2 = simplices.tetra[t][1]
            v3 = simplices.tetra[t][2]
            v4 = simplices.tetra[t][3]
            c1_idx = simplices.cycles.index(v1 + v2 + v3)
            c2_idx = simplices.cycles.index(v1 + v2 + v4)
            c3_idx = simplices.cycles.index(v2 + v3 + v4)
            c4_idx = simplices.cycles.index(v1 + v3 + v4)
            B3[c1_idx, t] = 1
            B3[c2_idx, t] = 1
            B3[c3_idx, t] = 1
            B3[c4_idx, t] = 1
    else:
        B3 = None
    return Boundaries(B1, B2, B3)


def buildBoundaries(scomplex):
    if scomplex.boundaries is None:
        # print("Using list of simplices to construct boundaries...")
        scomplex.B0 = None
        # Build B1
        if scomplex.dim >= 1:
            scomplex.B1 = torch.zeros(
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
            scomplex.B2 = torch.zeros(
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
            scomplex.B3 = torch.zeros(
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
