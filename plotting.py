import itertools
import string
from math import comb

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def drawComplex(SC, S0=None):
    """
    Draw simplicial complex from a list of simplices

    Input args:
    ----
        simplex_list: list of lists of characters

    Adapted from:
    ----------
    [1] https://github.com/iaciac/py-draw-simplicial-complex/blob/master/Draw%202d%20simplicial%20complex.ipynb

    """
    print("Drawing simplicial complex...")
    # print('Simplices: \n', SC.simplices)
    simplex_list = SC.simplices
    dim = SC.dim

    # List of 0-simplices
    if dim >= 0:
        nodes = list((simplex_list[0]))
    # List of 1-simplices
    if dim >= 1:
        edges = list(set(tuple(sorted((i, j))) for i, j in simplex_list[1]))
    # List of 2-simplices
    if dim >= 2:
        triangles = list(set(tuple(sorted((i, j, k))) for i, j, k in simplex_list[2]))
    # List of 3-simplices
    if dim >= 3:
        tetrahedra = list(
            set(tuple(sorted((i, j, k, l))) for i, j, k, l in simplex_list[3])
        )

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    if ax is None:
        ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis("off")

    # Create networkx Graph from edges
    G = nx.Graph()
    G.add_edges_from(edges)
    # Dictionary containing position of each node
    pos = nx.spring_layout(G)
    # Draw nodes
    # if dim >=0:
    #    for i in nodes:
    #        (a, b) = pos[i]
    #        dot = plt.Circle([ a, b ], radius = 0.02, zorder = 4, lw=1.0,
    #                         edgecolor = 'k', facecolor = 'k')
    # ax.add_patch(dot)
    # Draw edges
    if dim >= 1:
        for i, j in edges:
            (a0, b0) = pos[i]
            (a1, b1) = pos[j]
            line = plt.Line2D([a0, a1], [b0, b1], color="k", zorder=0, lw=1.0)
            ax.add_line(line)
    # Draw triangles
    if dim >= 2:
        for i, j, k in triangles:
            (a0, b0) = pos[i]
            (a1, b1) = pos[j]
            (a2, b2) = pos[k]
            tri = plt.Polygon(
                [[a0, b0], [a1, b1], [a2, b2]],
                edgecolor="k",
                facecolor="#BEC3C6",
                zorder=0,
                alpha=0.4,
                lw=1.0,
            )
            ax.add_patch(tri)
    # Draw tetrahedra
    if dim >= 3:
        for i, j, k, l in tetrahedra:

            (a0, b0) = pos[i]
            (a1, b1) = pos[j]
            (a2, b2) = pos[k]
            (a3, b3) = pos[l]
            # Use 4 triangles to fill in tetrahedra
            tri1 = plt.Polygon(
                [[a0, b0], [a1, b1], [a3, b3]],
                edgecolor="k",
                facecolor="#98989C",
                zorder=0,
                alpha=0.4,
                lw=1.0,
            )
            ax.add_patch(tri1)
            tri2 = plt.Polygon(
                [[a1, b1], [a2, b2], [a3, b3]],
                edgecolor="k",
                facecolor="#98989C",
                zorder=0,
                alpha=0.4,
                lw=1.0,
            )
            ax.add_patch(tri2)
            tri3 = plt.Polygon(
                [[a0, b0], [a1, b1], [a2, b2]],
                edgecolor="k",
                facecolor="#98989C",
                zorder=0,
                alpha=0.4,
                lw=1.0,
            )
            ax.add_patch(tri3)
            tri4 = plt.Polygon(
                [[a0, b0], [a2, b2], [a3, b3]],
                edgecolor="k",
                facecolor="#98989C",
                zorder=0,
                alpha=0.4,
                lw=1.0,
            )
            ax.add_patch(tri4)

    # nodes
    nx.draw(G, pos, node_color="k", with_labels=True)

    # node labels
    labels = {nodes[i]: nodes[i] for i in range(len(nodes))}
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color="whitesmoke")

    if S0 is not None:
        val_map = {}
        for v in range(len(nodes)):
            val_map[nodes[v]] = list(S0[v, :]).index(1)
        c_list = {}
        for k, v in val_map.items():
            c_list.setdefault(v, []).append(k)
        print("Vertex partition:", c_list)
        values = [val_map.get(node) for node in G.nodes()]
        cmap = plt.get_cmap("tab20")
        options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.5}
        nx.draw_networkx_nodes(G, pos, nodelist=G.nodes, node_color=values, **options)

    return


def visualizeA(SC):
    """
    Visualize the adjacency matrices
    """
    print("Visualizing ADJACENCY MATRICES...")
    dim = SC.dim
    fig, ax = plt.subplots(nrows=1, ncols=dim, figsize=(18, 18))

    simplices = SC.simplices
    for f in range(dim):
        if f == 0:
            adj = SC.A0
            title = "$A_{0}^{(\ell+1)}$"
            ticks = list(range(len(simplices[0])))
            labels = simplices[0]
        elif f == 1:
            adj = SC.A1
            title = "$A_{1}^{(\ell+1)}$"
            ticks = list(range(len(simplices[1])))
            labels = simplices[1]
        elif f == 2:
            adj = SC.A2
            title = "$A_{2}^{(\ell+1)}$"
            ticks = list(range(len(simplices[2])))
            labels = simplices[2]

        heatmap = ax[f].matshow(adj, cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax[f].set_title(title)
        ax[f].set_xticks(ticks)
        ax[f].set_xticklabels(labels, rotation=45)
        ax[f].set_yticks(ticks)
        ax[f].set_yticklabels(labels)
        color_bar = plt.colorbar(heatmap, ax=ax[f], fraction=0.046, pad=0.04)
        color_bar.minorticks_on()
        for (i, j), z in np.ndenumerate(adj):
            ax[f].text(j, i, "{:0.3f}".format(z), ha="center", va="center")
        plt.tight_layout()
    return


def visualizeB(SC):
    """
    Visualize the boundary matrices
    """
    print("Visualizing BOUNDARY MATRICES...")
    dim = SC.dim
    fig, ax = plt.subplots(nrows=1, ncols=dim, figsize=(18, 18))
    simplices = SC.simplices
    for f in range(dim):
        if f == 0:
            bd = SC.B1
            title = "$B_{1}^{(\ell+1)}$"
            ticksy = list(range(len(simplices[0])))
            labelsy = simplices[0]
            ticksx = list(range(len(simplices[1])))
            labelsx = simplices[1]

        elif f == 1:
            bd = SC.B2
            title = "$B_{2}^{(\ell+1)}$"
            ticksy = list(range(len(simplices[1])))
            labelsy = simplices[1]
            ticksx = list(range(len(simplices[2])))
            labelsx = simplices[2]
        elif f == 2:
            bd = SC.B3
            title = "$B_{3}^{(\ell+1)}$"
            ticksy = list(range(len(simplices[2])))
            labelsy = simplices[2]
            ticksx = list(range(len(simplices[3])))
            labelsx = simplices[3]

        heatmap = ax[f].matshow(bd, cmap=plt.cm.Blues, vmin=0, vmax=2)
        ax[f].set_title(title)
        ax[f].set_xticks(ticksx)
        ax[f].set_xticklabels(labelsx, rotation=45)
        ax[f].set_yticks(ticksy)
        ax[f].set_yticklabels(labelsy)
        color_bar = plt.colorbar(heatmap, ax=ax[f], fraction=0.046, pad=0.04)
        color_bar.minorticks_on()
        for (i, j), z in np.ndenumerate(bd):
            ax[f].text(j, i, "{:0.3f}".format(z), ha="center", va="center")
        plt.tight_layout()
    return
