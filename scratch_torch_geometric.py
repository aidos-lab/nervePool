import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from complex import SComplex, pool_complex
from transforms import EdgeIndexToSimplicesTransform

ds = TUDataset(
    root="./data",
    name="Letter-low",
    transform=EdgeIndexToSimplicesTransform(),
    use_node_attr=True,
    cleaned=False,
    force_reload=True,
)

# nodes = [g.num_nodes for g in ds]
# print(nodes)
# print(max(nodes))


dl = DataLoader(ds, batch_size=2)


for data in dl:
    for simp in data.simplices:
        s = torch.zeros(size=(len(simp.nodes), 4))
        s[:, 0] = 1
        sc_torch = SComplex(simplices=simp)
        sc_torch_pooled = pool_complex(sc_torch, s)
        print(sc_torch_pooled)
        breakpoint()
