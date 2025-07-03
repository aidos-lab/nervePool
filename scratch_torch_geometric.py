import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from complex import SComplex, pool_complex
from transforms import EdgeIndexToBoundaryTransform

ds = TUDataset(
    root="./data",
    name="DD",
    transform=EdgeIndexToBoundaryTransform(max_num_nodes=700, max_num_edges=2000),
    use_node_attr=True,
    cleaned=False,
    force_reload=True,
)

# nodes = [g.num_nodes for g in ds]
# print(nodes)
# print(max(nodes))


dl = DataLoader(ds, batch_size=2)


for data in dl:
    # DataBatch(x=[32, 700, 4], adj=[32, 700, 700], mask=[32, 700], y=[32, 1])
    # x, mask = to_dense_batch(data.x, data.batch, max_num_nodes=200)
    # adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=200)
    # print(x.shape)
    # print(mask.shape)
    # print(adj.shape)
    s = torch.zeros(size=(700, 64))
    for boundaries in data.B1:
        sc_torch = SComplex(boundaries=boundaries)
        sc_torch_pooled = pool_complex(sc_torch, s)
    breakpoint()

    break
