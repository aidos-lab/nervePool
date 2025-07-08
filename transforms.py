import torch
import torch_geometric

from boundary import Simplices


def edge_index_to_boundaries(edge_index, max_num_nodes, max_num_edges):
    num_nodes = edge_index.max() + 1
    edge_list = edge_index.T.tolist()

    B1 = torch.zeros([max_num_nodes, max_num_edges])
    for v0 in range(num_nodes):
        for v1 in range(num_nodes):
            simplex_up = [v0, v1]
            if simplex_up in edge_list:
                up_idx = edge_list.index(simplex_up)
                B1[v0, up_idx] = 1
                B1[v1, up_idx] = 1
    return B1


class EdgeIndexToBoundaryTransform:
    def __init__(self, max_num_nodes, max_num_edges):
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges

    def __call__(self, data):
        data.B1 = (
            edge_index_to_boundaries(
                data.edge_index,
                max_num_nodes=self.max_num_nodes,
                max_num_edges=self.max_num_edges,
            ).unsqueeze(0),
        )
        return data


class EdgeIndexToSimplicesTransform:
    def __call__(self, data):
        num_nodes = data.edge_index.max() + 1
        nodes = torch.arange(num_nodes)
        edges = data.edge_index.T
        simplices = Simplices(nodes, edges)
        data.simplices = simplices

        return data
