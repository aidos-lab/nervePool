"""
Simplicial Complex class with auxillary functions for pooling

Sarah McGuire 2022
"""
import numpy as np
import string
from math import comb
import networkx as nx
import itertools
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt


class SComplex:
    """Class for a (attributed) simplicial complex. Currently supports up to dim=3 simplicial complex

        Args:
            - simplices: list of simplices which form the simplicial complex. 
            - dim: The max dimension of the simplicial complex.
            - label: int containing the complex's label (for a complex-level task)

    """
        
    def __init__(self, *args, dim: int = None, label: int = None):
        """
        If the input data to define the simplicial complex is a list of simplices, 
        use this info to define boundary matrices else, if the input data is the set 
        of boundary matrices, use this to define the list of simplices (via the adjacency matrices)
        """
        self.simplices = None
        self.boundaries = None

        for info in args:
            if isinstance(info, list): #Input info is the list of simplices
                if len(info) == 0:
                    raise ValueError('List of simplices must contain at least one simplex.')
                if dim is None:
                    dim = len(info) - 1
                if len(info) < dim + 1:
                    raise ValueError(f'Missing simplices for the specified complex dim, '
                             f'expected {dim + 1}, but received {len(info)}')
                self.simplices = {i: info[i] for i in range(dim + 1)}
                self.nodes = info[0]
                if dim >=1:
                    self.edges = info[1]
                else:
                    self.edges = None
                if dim >=2:
                    self.cycles = info[2]
                else:
                    self.cycles = None
                if dim >=3:
                    self.tetra = info[3]
                else:
                    self.tetra = None

            elif isinstance(info, np.ndarray): #Input info is the array of boundary matrices

                if len(info) == 0:
                    raise ValueError('Array of boundaries must contain at least B1.')
                if dim is None:
                    dim = len(info)
                if len(info) < dim :
                    raise ValueError(f'Missing simplices for the specified complex dim, '
                             f'expected {dim}, but received {len(info)}')
                self.boundaries = {i+1: info[i] for i in range(dim)}
                self.B1 = info[0]
                if dim >=1:
                    self.B2 = info[1]
                else:
                    self.B2 = None
                if dim >=2:
                    self.B3 = info[2]
                else:
                    self.B3 = None
            else:
                raise ValueError('Input arg type not supported. Use list of simplices or np.ndarray of boundary matrices to define the simplicial complex.')

        self.dim = dim
        self.label = label
        
        
        self._buildBoundaries()
        # Assuming non-oriented boundary matrices
        for b in self.boundaries:
            self.boundaries[b] = abs(self.boundaries[b])
            
        self._constructAdj()
        #self._buildSimplexListBDS()
        self._buildSimplexListADJ()
        print('Simplices: \n', self.simplices)
        
    def _buildBoundaries(self):
        """
        Use simplex lists to construct the corresponding boundary matrices which represent the complex
        """
        scomplex = self
        if scomplex.boundaries is None:
            print('Using list of simplices to construct boundaries...' )
            scomplex.B0 = None
            # Build B1
            if scomplex.dim >=1:
                scomplex.B1 = np.zeros([len(self.simplices[0]),len(self.simplices[1])])
                for v0 in range(len(self.simplices[0])):
                    for v1 in range(len(self.simplices[0])):
                        simplex_up = self.simplices[0][v0] + self.simplices[0][v1]
                        if (simplex_up in self.simplices[1]):
                            up_idx = self.simplices[1].index(simplex_up)
                            scomplex.B1[v0, up_idx] = 1
                            scomplex.B1[v1, up_idx] = 1
            else:
                scomplex.B1 = None

            # Build B2   
            if scomplex.dim >=2:
                scomplex.B2 = np.zeros([len(self.simplices[1]),len(self.simplices[2])])
                for cyc in range(len(self.simplices[2])):
                    v1 = self.simplices[2][cyc][0]
                    v2 = self.simplices[2][cyc][1]
                    v3 = self.simplices[2][cyc][2]
                    e1_idx = self.simplices[1].index(v1+v2)
                    e2_idx = self.simplices[1].index(v2+v3)
                    e3_idx = self.simplices[1].index(v1+v3)
                    scomplex.B2[e1_idx,cyc] = 1
                    scomplex.B2[e2_idx,cyc] = 1
                    scomplex.B2[e3_idx,cyc] = 1
            else:
                scomplex.B2 = None

            # Build B3    
            if scomplex.dim >=3:
                scomplex.B3 = np.zeros([len(self.simplices[2]),len(self.simplices[3])])
                for t in range(len(self.simplices[3])):
                    v1 = self.simplices[3][t][0]
                    v2 = self.simplices[3][t][1]
                    v3 = self.simplices[3][t][2]
                    v4 = self.simplices[3][t][3]
                    c1_idx = self.simplices[2].index(v1+v2+v3)
                    c2_idx = self.simplices[2].index(v1+v2+v4)
                    c3_idx = self.simplices[2].index(v2+v3+v4)
                    c4_idx = self.simplices[2].index(v1+v3+v4)
                    scomplex.B3[c1_idx,t] = 1
                    scomplex.B3[c2_idx,t] = 1
                    scomplex.B3[c3_idx,t] = 1
                    scomplex.B3[c4_idx,t] = 1
            else:
                scomplex.B3 = None
            allBoundaries = [scomplex.B1, scomplex.B2, scomplex.B3]
            scomplex.boundaries = {i+1: allBoundaries[i] for i in range(scomplex.dim)}
    def _buildSimplexListBDS(self):
        """
        Use boundary matrices to construct the corresponding list of simplices which represent the complex
        """
        scomplex = self
        if scomplex.simplices is None:
            print('Using list of boundaries to construct simplices...' )
            # Node list
            n_nodes = scomplex.B1.shape[0]
            scomplex.nodes = [str(x) for x in (string.ascii_lowercase[0:n_nodes])]

            # Build edge list
            if scomplex.dim >=1:
                scomplex.edges = []
                n_edges = scomplex.B1.shape[1]
                for e in range(n_edges):
                    col = scomplex.B1[:,e]
                    # each edge has exactly 2 nodes on its boundary- a head and a tail
                    head_idx = np.where(col>0)[0][0]
                    tail_idx = np.where(col<0)[0][0]
                    scomplex.edges.append(scomplex.nodes[head_idx]+scomplex.nodes[tail_idx])
            else:
                scomplex.edges = None

            # Build cycle list  
            if scomplex.dim >=2:
                scomplex.cycles = []
                n_cycles = scomplex.B2.shape[1]
                for cycle in range(n_cycles):
                    col = scomplex.B2[:,cycle]
                    # each cycle has exactly 2 edges on its boundary- 2 positive, 1 negative oriented
                    # Extract the 3 nodes labels that make up this cycle
                    pos_idx = np.where(col>0)[0][0]
                    neg_idx = np.where(col<0)[0][0]
                    scomplex.cycles.append(scomplex.edges[pos_idx]+scomplex.edges[neg_idx][1])
            else:
                scomplex.cycles = None

            # Build tetrahedra list 
            print('scomplex.dim =' , scomplex.dim)
            if scomplex.dim >=3:
                scomplex.tetra = []
                n_tetra = scomplex.B3.shape[1]
                for tetra in range(n_tetra):
                    col = scomplex.B3[:, tetra]
                    # each tetrahedra has exactly 4 cycles on its boundary- 2 positive, 2 negative oriented
                    # Extract the 4 nodes labels that make up this cycle
                    pos_idx = np.where(col>0)[0][0]
                    neg_idx = np.where(col<0)[0][0]
                    node_idxs = [scomplex.nodes.index(x) for x in set(scomplex.cycles[pos_idx]+scomplex.cycles[neg_idx])]
                    node_idxs.sort()
                    scomplex.tetra.append("".join([scomplex.nodes[x] for x in node_idxs]))

            else:
                scomplex.tetra = None
            allsimplices = [scomplex.nodes, scomplex.edges, scomplex.cycles, scomplex.tetra]
            scomplex.simplices = {i: allsimplices[i] for i in range(scomplex.dim+1)}
            
    def _buildSimplexListADJ(self):
        """
        Use adjacency matrices to construct the corresponding list of simplices which represent the complex
        """
        if self.simplices is None:
            print('Using list of adjacency matrices to construct simplices...' )
            # Node list
            n_nodes = self.A0.shape[0]
            self.nodes = [str(x) for x in (string.ascii_lowercase[0:n_nodes])]

            # Build edge list
            if self.dim >=1:
                self.edges = []
                for i in range(n_nodes):
                    for j in range(i+1,n_nodes):
                        if self.A0[i,j]>0:
                            self.edges.append(self.nodes[i]+self.nodes[j])
                n_edges = len(self.edges)
            else:
                self.edges = None

            # Build triangle list  
            if self.dim >=2:
                self.triangles = []
                for i in range(n_edges):
                    for j in range(i+1,n_edges):
                        for k in range(j+1,n_edges):
                            if self.A1[i,j]>0 and self.A1[i,k]>0 and self.A1[j,k]>0:
                                e1 = self.edges[i]
                                e2 = self.edges[j]
                                e3 = self.edges[k]
                                a = list(set(list(e1)+list(e2)+list(e3)))
                                if len(a)== 3:
                                    self.triangles.append(''.join([str(i) for i in a]))
                n_triangles = len(self.triangles)
            else:
                self.triangles = None

            # Build tetrahedra list 
            if self.dim >=3:
                self.tetra = []
                for i in range(n_triangles):
                    for j in range(i+1,n_triangles):
                        for k in range(j+1,n_triangles):
                            for l in range(k+1,n_triangles):
                                if self.A2[i,j]>0 and self.A2[i,k]>0 and self.A2[i,l]>0 and self.A2[j,k]>0 and self.A2[j,l]>0 and self.A2[k,l]>0 :
                                    t1 = self.triangles[i]
                                    t2 = self.triangles[j]
                                    t3 = self.triangles[k]
                                    t4 = self.triangles[l]
                                    a = list(set(list(t1)+list(t2)+list(t3)+list(t4)))
                                    if len(a) == 4:
                                        self.tetra.append(''.join([str(i) for i in a]))
                n_tetra = len(self.tetra)
            else:
                self.tetra = None
                
                
            allsimplices = [self.nodes, self.edges, self.triangles, self.tetra]
            self.simplices = {i: allsimplices[i] for i in range(self.dim+1)}
                    
    def _constructAdj(self):
        """
        Use boundary matrices to construct adjacency matrices via A = |D - (B * B^T)| or A = B * B^T
        """
        scomplex = self
        # A0
        if scomplex.dim >=1:
            D0 = np.diag(np.sum(np.abs(scomplex.B1), axis=1))
            #print('D0 is:', D0)
            scomplex.A0 = np.abs(D0 - np.matmul(scomplex.B1,scomplex.B1.T))
            #scomplex.A0 = np.matmul(scomplex.B1,scomplex.B1.T)
        else:
            scomplex.A0 = None
        # A1
        if scomplex.dim >=2:
            D1 = np.diag(np.sum(np.abs(scomplex.B2), axis=1))
            #print('D1 is:', D1)
            scomplex.A1 = np.abs(D1 - np.matmul(scomplex.B2,scomplex.B2.T))
            #scomplex.A1 = np.matmul(scomplex.B2,scomplex.B2.T)
        else:
            scomplex.A1 = None
        # A2
        if scomplex.dim >=3:
            D2 = np.diag(np.sum(np.abs(scomplex.B3), axis=1))
            #print('D2 is:', D2)
            scomplex.A2 = np.abs(D2 - np.matmul(scomplex.B3,scomplex.B3.T))
            #scomplex.A2 = np.matmul(scomplex.B3,scomplex.B3.T)
        else:
            scomplex.A2 = None
        
        


    def drawComplex(self, S0=None):
        """
        Draw simplicial complex from a list of simplices

        Input args:
        ----
            simplex_list: list of lists of characters

        Adapted from:
        ----------    
        [1] https://github.com/iaciac/py-draw-simplicial-complex/blob/master/Draw%202d%20simplicial%20complex.ipynb

        """
        print('Drawing simplicial complex...')
        #print('Simplices: \n', self.simplices)
        simplex_list = self.simplices
        dim = self.dim
        
        #List of 0-simplices
        if dim >=0:
            nodes =list((simplex_list[0]))
        #List of 1-simplices
        if dim >=1:
            edges = list(set(tuple(sorted((i,j))) for i,j in simplex_list[1]))
        #List of 2-simplices
        if dim >=2:
            triangles = list(set(tuple(sorted((i,j,k))) for i,j,k in simplex_list[2]))
        #List of 3-simplices
        if dim >=3:
            tetrahedra = list(set(tuple(sorted((i,j,k,l))) for i,j,k,l in simplex_list[3]))

        plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        if ax is None: ax = plt.gca()
        ax.set_xlim([-1.1, 1.1])      
        ax.set_ylim([-1.1, 1.1])
        ax.get_xaxis().set_ticks([])  
        ax.get_yaxis().set_ticks([])
        ax.axis('off')

        # Create networkx Graph from edges
        G = nx.Graph()
        G.add_edges_from(edges)
        # Dictionary containing position of each node
        pos = nx.spring_layout(G)
        #Draw nodes
        #if dim >=0:
        #    for i in nodes:
        #        (a, b) = pos[i]
        #        dot = plt.Circle([ a, b ], radius = 0.02, zorder = 4, lw=1.0,
         #                         edgecolor = 'k', facecolor = 'k')
                #ax.add_patch(dot)
        # Draw edges
        if dim >= 1:
            for i, j in edges:
                (a0, b0) = pos[i]
                (a1, b1) = pos[j]
                line = plt.Line2D([ a0, a1 ], [b0, b1 ],color = 'k', zorder = 0, lw=1.0)
                ax.add_line(line)
        # Draw triangles
        if dim >=2 :
            for i, j, k in triangles:
                (a0, b0) = pos[i]
                (a1, b1) = pos[j]
                (a2, b2) = pos[k]
                tri = plt.Polygon([ [ a0, b0 ], [ a1, b1 ], [ a2, b2 ] ],
                                      edgecolor = 'k', facecolor = u'#BEC3C6',
                                      zorder = 0, alpha=0.4, lw=1.0)
                ax.add_patch(tri)
        # Draw tetrahedra
        if dim >=3:
            for i,j,k,l in tetrahedra:

                (a0, b0) = pos[i]
                (a1, b1) = pos[j]
                (a2, b2) = pos[k]
                (a3, b3) = pos[l]
                # Use 4 triangles to fill in tetrahedra
                tri1 = plt.Polygon([ [ a0, b0 ], [ a1, b1 ], [ a3, b3 ] ],
                                      edgecolor = 'k', facecolor = u'#98989C',
                                      zorder = 0, alpha=0.4, lw=1.0)
                ax.add_patch(tri1)
                tri2 = plt.Polygon([ [ a1, b1 ], [ a2, b2 ], [ a3, b3 ] ],
                                      edgecolor = 'k', facecolor = u'#98989C',
                                      zorder = 0, alpha=0.4, lw=1.0)
                ax.add_patch(tri2)
                tri3 = plt.Polygon([ [ a0, b0 ], [ a1, b1 ], [ a2, b2 ] ],
                                      edgecolor = 'k', facecolor = u'#98989C',
                                      zorder = 0, alpha=0.4, lw=1.0)
                ax.add_patch(tri3)
                tri4 = plt.Polygon([ [ a0, b0 ], [ a2, b2 ], [ a3, b3 ] ],
                                      edgecolor = 'k', facecolor = u'#98989C',
                                      zorder = 0, alpha=0.4, lw=1.0)
                ax.add_patch(tri4)

            
        # nodes
        nx.draw(G, pos, node_color='k', with_labels= True)

        # node labels
        labels = {nodes[i]: nodes[i] for i in range(len(nodes))}
        nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color="whitesmoke")
        
        if S0 is not None:
            val_map = {}
            for v in range(len(nodes)):
                val_map[nodes[v]] = list(S0[v,:]).index(1)
            c_list = {}
            for k, v in val_map.items():
                c_list.setdefault(v, []).append(k)
            print('Vertex partition:', c_list)
            values = [val_map.get(node) for node in G.nodes()]
            cmap = plt.get_cmap('tab20')
            options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.5}
            nx.draw_networkx_nodes(G, pos, nodelist = G.nodes, node_color= values, **options)

        return

    def visualizeA(self):
        """
        Visualize the adjacency matrices
        """
        print('Visualizing ADJACENCY MATRICES...')
        dim = self.dim
        fig, ax = plt.subplots(nrows=1, ncols = dim, figsize = (8,8))
        simplices = self.simplices
        for f in range(dim):
            if f==0:
                adj = self.A0
                title = "$A_{0}^{(\ell+1)}$"
                ticks = list(range(len(simplices[0])))
                labels = simplices[0]
            elif f==1:
                adj = self.A1
                title = "$A_{1}^{(\ell+1)}$"
                ticks = list(range(len(simplices[1])))
                labels = simplices[1]
            elif f==2:
                adj = self.A2
                title = "$A_{2}^{(\ell+1)}$"
                ticks = list(range(len(simplices[2])))
                labels = simplices[2]

            heatmap = ax[f].matshow(adj, cmap=plt.cm.Blues,vmin=0,vmax=1)
            ax[f].set_title(title)
            ax[f].set_xticks(ticks)
            ax[f].set_xticklabels(labels,rotation = 45)
            ax[f].set_yticks(ticks)
            ax[f].set_yticklabels(labels)
            color_bar = plt.colorbar(heatmap, ax = ax[f],fraction=0.046, pad=0.04)
            color_bar.minorticks_on()
            for (i, j), z in np.ndenumerate(adj):
                ax[f].text(j, i, '{:0.3f}'.format(z), ha='center', va='center')    
            plt.tight_layout()
        return 

    def visualizeB(self):
        """
        Visualize the boundary matrices
        """
        print('Visualizing BOUNDARY MATRICES...')
        dim = self.dim
        fig, ax = plt.subplots(nrows=1, ncols = dim, figsize = (8,8))
        simplices = self.simplices
        for f in range(dim):
            if f==0:
                bd = self.B1
                title = "$B_{1}^{(\ell+1)}$"
                ticksy = list(range(len(simplices[0])))
                labelsy = simplices[0]
                ticksx = list(range(len(simplices[1])))
                labelsx = simplices[1]

            elif f==1:
                bd = self.B2
                title = "$B_{2}^{(\ell+1)}$"
                ticksy = list(range(len(simplices[1])))
                labelsy = simplices[1]
                ticksx = list(range(len(simplices[2])))
                labelsx = simplices[2]
            elif f==2:
                bd = self.B3
                title = "$B_{3}^{(\ell+1)}$"
                ticksy = list(range(len(simplices[2])))
                labelsy = simplices[2]
                ticksx = list(range(len(simplices[3])))
                labelsx = simplices[3]


            heatmap = ax[f].matshow(bd, cmap=plt.cm.Blues,vmin=0,vmax=2)
            ax[f].set_title(title)
            ax[f].set_xticks(ticksx)
            ax[f].set_xticklabels(labelsx,rotation = 45)
            ax[f].set_yticks(ticksy)
            ax[f].set_yticklabels(labelsy)
            color_bar = plt.colorbar(heatmap,ax = ax[f],fraction=0.046, pad=0.04)
            color_bar.minorticks_on()
            for (i, j), z in np.ndenumerate(bd):
                ax[f].text(j, i, '{:0.3f}'.format(z), ha='center', va='center')    
            plt.tight_layout()
        return

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
            if (S0[v0,v]>0 or S0[v1,v]>0):
                edge_arr[v]=1
        S01.append(edge_arr)

    
    for c in SC.cycles:
        cyc_arr = np.zeros(n_new)
        v0 = SC.nodes.index(c[0])
        v1 = SC.nodes.index(c[1])
        v2 = SC.nodes.index(c[2])
        for v in range(n_new):
            if (S0[v0,v]>0 or S0[v1,v]>0 or S0[v2,v]>0):
                cyc_arr[v]=1
        S02.append(cyc_arr)    
        
    for t in SC.tetra:
        t_arr = np.zeros(n_new)
        v0 = SC.nodes.index(t[0])
        v1 = SC.nodes.index(t[1])
        v2 = SC.nodes.index(t[2])
        v3 = SC.nodes.index(t[3])
        for v in range(n_new):
            if (S0[v0,v]>0 or S0[v1,v]>0 or S0[v2,v]>0 or S0[v3,v]>0):
                t_arr[v]=1
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


def right_function(Scol0,SC):
    """
        NervePool right update function. 
        Uses element-wise multiplication of cluster columns $\tilde{U_i}$ to compute intersections of cover elements. 
        Full S matrix is row normalized.
        Output: S_p block diagonal matrices.
    """
    # New Edges block column: 
    # loop through all possible pairs of meta vertices to get new edge info
    n_nodes_new = Scol0.shape[1]
    Scol1 = np.zeros([Scol0.shape[0],comb(n_nodes_new,2)])
    col = 0
    for i in range(0,n_nodes_new):
        for j in range(i+1,n_nodes_new):
            Scol1[:,col] =  Scol0[:,i] * Scol0[:,j]
            col+=1
    # Remove edges that are not in pooled complex (all zero cols)
    Scol1 = np.delete(Scol1, np.argwhere(np.all(Scol1[...,:] == 0, axis=0)), axis=1)
        
    # New Cycles block column: 
    # loop through all possible triples of meta vertices to get new cycle info
    Scol0a = Scol0[len(SC.edges):,:]
    Scol2 = np.zeros([Scol0a.shape[0],comb(n_nodes_new,3)])
    col = 0
    for i in range(0,n_nodes_new):
        for j in range(i+1,n_nodes_new):
            for k in range(j+1, n_nodes_new):
                Scol2[:,col] =  Scol0a[:,i] * Scol0a[:,j] *Scol0a[:,k]
                col+=1
    # Remove cycles that are not in pooled complex (all zero cols)
    Scol2 = np.delete(Scol2, np.argwhere(np.all(Scol2[...,:] == 0, axis=0)), axis=1)
    
    # New Tetra block column: 
    # loop through all possible quadruplets of meta vertices to get new tetra info
    Scol0b = Scol0a[len(SC.cycles):,:]
    Scol3 = np.zeros([Scol0b.shape[0],comb(n_nodes_new,4)])
    col = 0
    for i in range(0,n_nodes_new):
        for j in range(i+1,n_nodes_new):
            for k in range(j+1, n_nodes_new):
                for l in range(k+1, n_nodes_new):
                    Scol3[:,col] =  Scol0b[:,i] * Scol0b[:,j] * Scol0b[:,k] * Scol0b[:,l]
                    col+=1
    # Remove tetra that are not in pooled complex (all zero cols)
    Scol3 = np.delete(Scol3, np.argwhere(np.all(Scol3[...,:] == 0, axis=0)), axis=1)
    # Normalize rows of S and select the diagonal sub-blocks for pooling
    if SC.dim >=1 and Scol1.size!=0:
        S1 = Scol1[:len(SC.edges),:]
        Srow1 = np.concatenate((Scol0[:len(SC.edges)],S1),axis = 1)
        Srow1_norm = Srow1 / Srow1.sum(axis =1)[:,np.newaxis]
        S1_norm = Srow1_norm[:, n_nodes_new:]
    else:
        S1_norm = None
    if SC.dim >=2 and Scol2.size!=0:
        S2 = Scol2[:len(SC.cycles),:]
        idx_start = len(SC.edges)
        idx_end = idx_start+len(SC.cycles)
        Srow2 = np.concatenate((Scol0[idx_start:idx_end],Scol1[idx_start:idx_end],S2),axis = 1)
        Srow2_norm = Srow2 / Srow2.sum(axis =1)[:,np.newaxis]
        S2_norm = Srow2_norm[:, Scol0.shape[1] + Scol1.shape[1]:]
    else: 
        S2_norm = None
    if SC.dim >=3 and Scol3.size!=0:
        S3 = Scol3[:len(SC.tetra),:]
        idx_start = len(SC.cycles) + len(SC.edges)
        Srow3 = np.concatenate((Scol0[idx_start:],Scol1[idx_start:],Scol2[len(SC.cycles):], S3),axis = 1)
        Srow3_norm = Srow3 / Srow3.sum(axis =1)[:,np.newaxis]
        S3_norm = Srow3_norm[:, Scol0.shape[1] + Scol1.shape[1]+Scol2.shape[1]:]
    else: 
        S3_norm = None
    return S1_norm, S2_norm, S3_norm


def pool_complex(SC, S0):
    """
    Function to pool a simplicial complex using a partition of vertices
    Args:
        - SC : SComplex object to be pooled
        - S0 : array of size |v| x |v|', a partition of vertices 
        
    Output:
        - SCpooled : SComplex object of the pooled complex
    """
    if S0.shape[0]==0 or S0.shape[1]==0:
        raise ValueError('Vertex cluster assignment matrix must be of size |v|x|v|')
    if S0.shape[0] != SC.A0.shape[0]:
        raise ValueError(f'Vertex cluster assignment size must match the number of vertices of the complex, '
                                 f'expected {SC.A0.shape}, but received {S0.shape}')

    # Extend S0 to full S block matrix
    col0 = down_function(S0, SC)
    S1, S2, S3 = right_function(col0,SC)
    #print('S matrices are:', S1,'\n', S2, '\n', S3)
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
    
    #Pooled complex dimension
    if B1_new is None:
        newdim = 0
    elif B2_new is None:
        newdim = 1
    elif B3_new is None:
        newdim = 2
    else:
        newdim = 3
    # Use new boundary matrices to construct pooled complex ... UNFINISHED
    Bds_new = np.array([B1_new,B2_new,B3_new], dtype = object)
    return SComplex(Bds_new, dim = newdim)