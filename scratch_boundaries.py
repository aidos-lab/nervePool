from types import SimpleNamespace

import numpy as np
import torch


def boundary_from_simplices(simplices, dim=1):
    B1 = torch.zeros([len(simplices.nodes), len(simplices.edges)])
    if dim >= 1:
        for v0 in range(len(simplices.nodes)):
            for v1 in range(len(simplices.nodes)):
                simplex_up = simplices.nodes[v0] + simplices.nodes[v1]
                if simplex_up in simplices.edges:
                    up_idx = simplices.edges.index(simplex_up)
                    B1[v0, up_idx] = 1
                    B1[v1, up_idx] = 1
    return B1


def edge_index_to_boundaries(edge_index):
    num_nodes = edge_index.max() + 1
    num_edges = edge_index.shape[1]
    edge_list = edge_index.T.tolist()
    B1 = torch.zeros([num_nodes, num_edges])
    for v0 in range(num_nodes):
        for v1 in range(num_nodes):
            simplex_up = [v0, v1]
            if simplex_up in edge_list:
                up_idx = edge_list.index(simplex_up)
                B1[v0, up_idx] = 1
                B1[v1, up_idx] = 1
    return B1


if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    simplices = SimpleNamespace(nodes=["a", "b", "c"], edges=["ab", "bc", "ca"])
    print(edge_index_to_boundaries(edge_index))
    print(boundary_from_simplices(simplices))
