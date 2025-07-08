import itertools

import torch


def pool_complex(node_features, edge_features, edge_index, cluster_assignments):
    num_virtual_nodes = cluster_assignments.shape[1]
    # Initialize
    virtual_edge_index = torch.tensor(
        list(
            itertools.product(
                range(num_virtual_nodes),
                range(num_virtual_nodes),
            )
        )
    ).T

    s = cluster_assignments

    # Down Function
    s_edges_virtual_nodes = cluster_assignments[edge_index].max(dim=0)[1]
    s = torch.vstack([s, s_edges_virtual_nodes])

    # Right function (naming convention by)
    s_all_virtual_edges = s[:, virtual_edge_index].min(dim=1)[1]
    s = torch.hstack([s, s_all_virtual_edges])

    non_empty_mask = s.abs().sum(dim=0).bool()
    s = s[:, non_empty_mask]

    features = torch.vstack([node_features, edge_features])

    pooled_features = s.T @ features

    return (
        pooled_features[:num_virtual_nodes],
        pooled_features[num_virtual_nodes:],
        virtual_edge_index[:, non_empty_mask],
    )


# |%%--%%| <7yiufeKhhg|DvnKTdS7kd>


edge_index = torch.tensor(
    [
        [0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8],
        [2, 3, 0, 1, 4, 3, 6, 8, 5, 7, 6, 5],
    ],
)
x = torch.tensor(
    [
        [0.6023, 2.9591],
        [1.8997, 2.7768],
        [0.7683, 0.6964],
        [0.6626, 1.6655],
        [1.8528, 0.7047],
        [0.6944, 2.8384],
        [2.2527, 0.5040],
        [2.1563, 2.8345],
        [0.6950, 0.4571],
    ]
)

edge_features = torch.rand(size=(edge_index.shape[1], x.shape[1]))

num_virtual_nodes = 5
rand_idx = torch.vstack(
    [
        torch.arange(x.shape[0]),
        torch.randint(0, num_virtual_nodes - 1, size=(x.shape[0], 1)).squeeze(),
    ]
)
cluster_assignments = torch.zeros(size=(x.shape[0], num_virtual_nodes))
cluster_assignments[rand_idx[0, :], rand_idx[1, :]] = 1


pool_complex(
    node_features=x,
    edge_features=edge_features,
    edge_index=edge_index,
    cluster_assignments=cluster_assignments,
)
