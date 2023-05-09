#!/usr/bin/env python
# coding: utf-8

# ## Functions to cascade information from diffPool S cluster assignment matrix to higher dim-simplices

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd





# Collect new adjacency information and create new lists of simplices
# and cascade info down/across S block matrix
# Needs: S00, A0,A1, vertex_list, edge_list, triangle_list, tetrahedron_list
#
def cascade_all(S00,A0,A1,vertex_list,edge_list,triangle_list,tetrahedron_list):
    v = S00.shape[0]
    v_new = S00.shape[1]

    # construct new vertex list
    v_list_new = list(map(str,list(range(v_new))))
    print('new vertex list: ',v_list_new)

    # construct new edge list
    # -- requires S00 block to calculate new vertex adjacency matrix (which gives information about new edges)
    A0_new = np.matmul(np.matmul(S00.T, A0),S00)
    e_list_new = []
    for v0 in range(v_new):
        for v1 in range(v0+1,v_new):
            if A0_new[v0,v1]>0:
                e_list_new.append(",".join([str(v0),str(v1)]))
    e_new = len(e_list_new)
    print('new edge list: ',e_list_new)
    # Cascade info down first block column
    S10,S20,S30 = cascade_down(S00,vertex_list,edge_list,triangle_list,tetrahedron_list)
    # Cascade info across from first block column to second block col.
    S01 = cascade_across(S00,S00,e_list_new)
    S11 = cascade_across(S10,S10,e_list_new)
    S21 = cascade_across(S20,S20,e_list_new)
    S31 = cascade_across(S30,S30,e_list_new)

    # construct new triangle list 
    # -- requires S00 block to calculate new edge adjacency matrix (which gives information about new triangles)
    A1_new = np.matmul(np.matmul(S11.T, A1),S11)
    t_list_new = []
    for e0 in range(e_new):
        for e1 in range(e0+1,e_new):
            for e2 in range(e1+1,e_new):
                if (A1_new[e0,e1]>0 and A1_new[e0,e2]>0 and A1_new[e1,e2]>0):
                    li = e_list_new[e0].split(',') + e_list_new[e1].split(',') + e_list_new[e2].split(',')
                    reduce = []
                    [reduce.append(x) for x in li if x not in reduce]
                    t_list_new.append(",".join([reduce[0],reduce[1],reduce[2]]))
    t_new = len(t_list_new) 
    print('new triangle list: ',t_list_new)

    # Cascade info across from second block column to third block col.
    if t_list_new:
        S02 = cascade_across(S01,S00,t_list_new)
        S12 = cascade_across(S11,S10,t_list_new)
        S22 = cascade_across(S21,S20,t_list_new)
        S32 = cascade_across(S31,S30,t_list_new)
    else:
        S02 = np.array([])
        S12 = np.array([])
        S22 = np.array([])
        S32 = np.array([])
    return S10,S20,S30,S01,S11,S21,S31,S02,S12,S22,S32,v_list_new,e_list_new,t_list_new


#-----------------------------
# Function (+helpers) to construct an oriented BOUNDARY MATRIX
#-----------------------------------------------
def orientation(tau,sigma):
    #print (tau,' in ' ,cyclic_perms(sigma),'?')
    if list(tau.split(',')) in cyclic_perms(list(sigma.split(','))):
        value = 1
    elif list(tau.split(',')) in cyclic_perms_neg(list(sigma.split(','))):
        value = -1
    else:
        value = 0
    return value
def cyclic_perms_neg(a):
    n = len(a)
    B = [[a[j - i] for i in range(n)] for j in range(n)]
    sx_list = []
    for b in B:        
        sx_list.append(b[0:-1])
        sx_list.append(b[1:])    
    return sx_list
def cyclic_perms(a):
    n = len(a)
    B = [[a[i - j] for i in range(n)] for j in range(n)]
    sx_list = []
    for b in B:        
        sx_list.append(b[0:-1])
        sx_list.append(b[1:])    
    return sx_list
#------------------------------------------------------------------------
def signed_boundary(sx_list, sx_list_low):
    n = len(sx_list)
    n_low = len(sx_list_low)
    dim_p = len(sx_list[0].split(','))-1
    dim_p_low = dim_p-1
    Bp = np.zeros([n_low,n])
    
    #B1
    if dim_p==1:
        for v in range(n_low):
            for e in range(n):
                if (str(v) == sx_list[e].split(',')[0]):
                    Bp[v,e] = -1
                elif (str(v) == sx_list[e].split(',')[1]):
                    Bp[v,e] = 1
    #Bp for p>1
    if dim_p > 1:
        for t in range(n_low):
            tau = sx_list_low[t]
            for s in range(n):
                sigma = sx_list[s]
                #print(tau, sigma, orientation(tau,sigma))
                Bp[t,s] = orientation(tau,sigma)
        
    return Bp
#--------------------------------------------------------------------------


#-----------------------------------------------
# Function to construct an oriented BOUNDARY MATRIX
#-----------------------------------------------
def constructBp(sx_list,sx_list_low):
    n = len(sx_list)
    n_low = len(sx_list_low)
    dim_p = len(sx_list[0].split(','))-1
    dim_p_low = dim_p-1
    Bp = np.zeros([n_low,n])

    #B1
    if dim_p==1:
        for v in range(n_low):
            for e in range(n):
                if (str(v) == sx_list[e].split(',')[0]):
                    Bp[v,e] = -1
                elif (str(v) == sx_list[e].split(',')[1]):
                    Bp[v,e] = 1
    #B2
    if dim_p==2:  
        for f in range(n):
            v1 = sx_list[f].split(',')[0]
            v2 = sx_list[f].split(',')[1]
            v3 = sx_list[f].split(',')[2]
            e1_idx = sx_list_low.index(",".join([v1,v2]))
            e2_idx = sx_list_low.index(",".join([v2,v3]))
            e3_idx = sx_list_low.index(",".join([v1,v3]))
            
            Bp[e1_idx,f] = 1
            Bp[e2_idx,f] = 1
            Bp[e3_idx,f] = -1
    #B3
    if dim_p==3:  
        for t in range(n):
            v1 = sx_list[t].split(',')[0]
            v2 = sx_list[t].split(',')[1]
            v3 = sx_list[t].split(',')[2]
            v4 = sx_list[t].split(',')[3]
            f1_idx = sx_list_low.index(",".join([v1,v2,v3]))
            f2_idx = sx_list_low.index(",".join([v1,v2,v4]))
            f3_idx = sx_list_low.index(",".join([v2,v3,v4]))
            f4_idx = sx_list_low.index(",".join([v1,v3,v4]))

            Bp[f1_idx,t] = 1
            Bp[f2_idx,t] = -1
            Bp[f3_idx,t] = 1  
            Bp[f4_idx,t] = -1
    return Bp


def construct_B(vertex_list,edge_list,triangle_list,tetrahedron_list, diag0=[],diag1=[],diag2=[],diag3=[]):
    n0 = len(vertex_list)
    n1 = len(edge_list)
    n2 = len(triangle_list)
    n3 = len(tetrahedron_list)
    # boundary 1 matrix
    B01 = np.zeros([n0,n1])
    # boundary 1 star of simplex info 
    B02 = np.zeros([n0,n2])
    B03 = np.zeros([n0,n3])
    for v in range(n0):
        for e in range(n1):
            if (str(v) in edge_list[e].split(',')):
                B01[v,e] = 1
        for t in range(n2):
            if (str(v) in triangle_list[t].split(',')):
                B02[v,t] = 1
        for th in range(n3):
            if (str(v) in tetrahedron_list[th].split(',')):
                B03[v,th] = 1

    # boundary 2 matrix    
    B12 = np.zeros([n1,n2])
    for t in range(n2):
        v1 = triangle_list[t].split(',')[0]
        v2 = triangle_list[t].split(',')[1]
        v3 = triangle_list[t].split(',')[2]
        e1_idx = edge_list.index(",".join([v1,v2]))
        e2_idx = edge_list.index(",".join([v1,v3]))
        e3_idx = edge_list.index(",".join([v2,v3]))
        B12[e1_idx,t] = 1
        B12[e2_idx,t] = 1
        B12[e3_idx,t] = 1


    # boundary 3 matrix + boundary 2 star extra info    
    B23 = np.zeros([n2,n3])
    B13 = np.zeros([n1,n3])
    for th in range(n3):
        v1 = tetrahedron_list[th].split(',')[0]
        v2 = tetrahedron_list[th].split(',')[1]
        v3 = tetrahedron_list[th].split(',')[2]
        v4 = tetrahedron_list[th].split(',')[3]
        t1_idx = triangle_list.index(",".join([v1,v2,v3]))
        t2_idx = triangle_list.index(",".join([v1,v3,v4]))
        t3_idx = triangle_list.index(",".join([v1,v2,v4]))
        t4_idx = triangle_list.index(",".join([v2,v3,v4]))
        e1_idx = edge_list.index(",".join([v1,v2]))
        e2_idx = edge_list.index(",".join([v1,v3]))
        e3_idx = edge_list.index(",".join([v1,v4]))
        e4_idx = edge_list.index(",".join([v2,v3]))
        e5_idx = edge_list.index(",".join([v2,v4]))
        e6_idx = edge_list.index(",".join([v3,v4]))
        B23[t1_idx,th] = 1
        B23[t2_idx,th] = 1
        B23[t3_idx,th] = 1
        B23[t4_idx,th] = 1

        B13[e1_idx,th] = 1
        B13[e2_idx,th] = 1
        B13[e3_idx,th] = 1
        B13[e4_idx,th] = 1
        B13[e5_idx,th] = 1
        B13[e6_idx,th] = 1
      
       
    if diag0==[]:
        diag0 = np.eye(n0)
        diag1 = np.eye(n1)
        diag2 = np.eye(n2)
        diag3 = np.eye(n3)
    if n3>0: #If maxdim(K) is tetrahedron 
        # diag. blocks of B with adj matrices
        B_star= np.block([[diag0, B01 ,B02, B03],
                          [B01.T, diag1, B12, B13],
                          [B02.T, B12.T, diag2, B23],
                          [B03.T, B13.T, B23.T, diag3] ])
    elif n2>0:#If maxdim(K) is triangles
        B_star= np.block([[diag0, B01 ,B02],
                          [B01.T, diag1, B12],
                          [B02.T, B12.T, diag2] ])
    else:#If maxdim(K) is edges
        B_star= np.block([[diag0, B01],
                          [B01.T, diag1]])
        
    return B_star


# Function to plot old and new boundary matrices
def plot_B_matrices(B_star, B_new, simplex_list,block_lines, simplex_list_new, block_lines_new):
    n_total= len(simplex_list)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,10))

    fig0= ax[0].matshow(B_star, cmap=plt.cm.Blues,vmin=0,vmax=1)
    ax[0].set_title("\"Boundary\" Matrix")
    ax[0].set_xticks(list(range(n_total)))
    ax[0].set_xticklabels(simplex_list,rotation = 45)
    ax[0].set_yticks(list(range(n_total)))
    ax[0].set_yticklabels(simplex_list)
    ax[0].hlines(block_lines,*ax[0].get_xlim(),colors='k')
    ax[0].vlines(block_lines,*ax[0].get_ylim(),colors='k')
    color_bar = fig.colorbar(fig0,ax = ax[0],fraction=0.046, pad=0.04)
    color_bar.minorticks_on()
    if n_total<15:
        for (i, j), z in np.ndenumerate(B_star):
            ax[0].text(j, i, '{:0.01f}'.format(z), ha='center', va='center')

    fig1 = ax[1].matshow(B_new, cmap=plt.cm.Blues,vmin=0,vmax=1)
    ax[1].set_title("New \"Boundary\" matrix")
    ax[1].set_xticks(list(range(len(simplex_list_new))))
    ax[1].set_xticklabels(simplex_list_new,rotation = 45)
    ax[1].set_yticks(list(range(len(simplex_list_new))))
    ax[1].set_yticklabels(simplex_list_new)
    ax[1].hlines(block_lines_new,*ax[1].get_xlim(),colors='k')
    ax[1].vlines(block_lines_new,*ax[1].get_ylim(),colors='k')
    color_bar = fig.colorbar(fig1,ax = ax[1],fraction=0.046, pad=0.04)
    color_bar.minorticks_on()
    for (i, j), z in np.ndenumerate(B_new):
        ax[1].text(j, i, '{:0.01f}'.format(z), ha='center', va='center')

    return 


# In[ ]:


def plot_A_matrices(A0=[],A1=[],A2=[],A3=[],v_list=[],e_list=[],t_list=[],th_list=[]):
    if not isinstance(A3,list):
        N_fig = 4
    elif not isinstance(A2,list):
        N_fig = 3
    elif not isinstance(A1,list):
        N_fig = 2
    else:
        N_fig = 1

    fig, ax = plt.subplots(nrows=1, ncols = N_fig, figsize = (10,10))

    for f in range(N_fig):
        if f==0:
            adj = A0
            title = "$A_{0,up}^{(\ell+1)}$ Topological"
            ticks = list(range(len(v_list)))
            labels = v_list
        elif f==1:
            adj = A1
            title = "$A_{1,up}^{(\ell+1)}$ Topological"
            ticks = list(range(len(e_list)))
            labels = e_list
        elif f==2:
            adj = A2
            title = "$A_{2,up}^{(\ell+1)}$ Topological"
            ticks = list(range(len(t_list)))
            labels = t_list
        else:
            adj = A3
            title = "$A_3$ Tetr. Adj"
            ticks = list(range(len(th_list)))
            labels = th_list
            
        fig0 = ax[f].matshow(adj, cmap=plt.cm.Blues,vmin=-1,vmax=1)
        ax[f].set_title(title)
        ax[f].set_xticks(ticks)
        ax[f].set_xticklabels(labels,rotation = 45)
        ax[f].set_yticks(ticks)
        ax[f].set_yticklabels(labels)
        color_bar = fig.colorbar(fig0,ax = ax[f],fraction=0.046, pad=0.04)
        color_bar.minorticks_on()
        for (i, j), z in np.ndenumerate(adj):
            ax[f].text(j, i, '{:0.2f}'.format(z), ha='center', va='center')    
    #plt.tight_layout()
    return 



def apply_pooling(S00,vertex_list=[],edge_list=[],triangle_list=[],tetrahedron_list=[],A0=[],A1=[],A2=[],A3=[]):

    # number of simplices, total simplex_list, block line locations for B
    n0= len(vertex_list)
    n1= len(edge_list)
    n2= len(triangle_list)
    n3= len(tetrahedron_list)
    simplex_list = vertex_list+edge_list+triangle_list+tetrahedron_list
    n_total = len(simplex_list)
    if n3!=0:
        K_max_dim = 3
        #Plot adjacency matrixes for original complex
        plot_A_matrices(A0=A0,A1=A1,A2=A2,A3=A3,v_list=vertex_list,
                        e_list=edge_list,t_list=triangle_list, th_list= tetrahedron_list)
    elif n2!=0:
        K_max_dim = 2
        plot_A_matrices(A0=A0,A1=A1,A2=A2,v_list=vertex_list,e_list=edge_list,
                           t_list=triangle_list)
    else: # graph
        K_max_dim = 1
        plot_A_matrices(A0=A0,A1=A1,v_list=vertex_list,e_list=edge_list)
        
    block_lines = [n0-0.5, n0+n1-0.5, n0+n1+n2-0.5]


    
    # Construct B (adjacencies on diagonal, off-block diagonals are boundary matrices, off-off diagonals and further are incidence matrices indicating star(v)
    B_star = construct_B(vertex_list,edge_list,triangle_list,tetrahedron_list, A0,A1,A2,A3)
    S10,S20,S30,S01,S11,S21,S31,S02,S12,S22,S32,v_list_new,e_list_new,t_list_new = cascade_all(S00,A0,A1,vertex_list,edge_list,triangle_list,tetrahedron_list)

        
    # new max dim of pooled complex (up to triangles included)
    if t_list_new:
        K_max_dim_new = 2
    else:
        K_max_dim_new = 1

        
    # Piece together large S block matrix (according to new complex dimensions)
    # Normalize diagonal sub-blocks
    # Compute Ai_new = Si^T Ai Si
    # Plot the new adjacency matrices from pooling within one dimension
    A0_new = np.array([])
    A1_new = np.array([])
    A2_new = np.array([])

    if (K_max_dim==3 and K_max_dim_new == 2):
        S = np.block([[S00, S01, S02],
                  [S10, S11, S12],
                  [S20, S21, S22],
                  [S30, S31, S32]])
        S00n= S00/S00.sum(axis=0,keepdims=1)
        S11n= S11/S11.sum(axis=0,keepdims=1)
        S22n= S22/S22.sum(axis=0,keepdims=1)
        A0_new = np.matmul(S00n.T,np.matmul(A0.T,S00n))
        A1_new = np.matmul(S11n.T,np.matmul(A1.T,S11n))
        A2_new = np.matmul(S22n.T,np.matmul(A2.T,S22n))
        #Plot the new adjacency matrices
        plot_A_matrices(A0=A0_new,A1=A1_new,A2=A2_new,v_list=v_list_new,
                                e_list=e_list_new,t_list=t_list_new)
        
    elif (K_max_dim==3 and K_max_dim_new == 1):
        S = np.block([[S00, S01],
                  [S10, S11],
                  [S20, S21],
                  [S30, S31]])
        S00n= S00/S00.sum(axis=0,keepdims=1)
        S11n= S11/S11.sum(axis=0,keepdims=1)
        A0_new = np.matmul(S00n.T,np.matmul(A0.T,S00n))
        A1_new = np.matmul(S11n.T,np.matmul(A1.T,S11n))        
        plot_A_matrices(A0=A0_new,A1=A1_new,v_list=v_list_new,e_list=e_list_new)
    else: #K_max_dim==2 and K_max_dim_new == 1:
        S = np.block([[S00, S01],
                  [S10, S11],
                  [S20, S21]])
        S00n= S00/S00.sum(axis=0,keepdims=1)
        S11n= S11/S11.sum(axis=0,keepdims=1)
        A0_new = np.matmul(S00n.T,np.matmul(A0.T,S00n))
        A1_new = np.matmul(S11n.T,np.matmul(A1.T,S11n))
        plot_A_matrices(A0=A0_new,A1=A1_new,v_list=v_list_new,e_list=e_list_new)
    # normalize S so column sums are 1
    S = S/S.sum(axis=0,keepdims=1) 

    #New boundary matrix B= S^TBS and new block matrix lines for B plot
    B_new = np.matmul(S.T,np.matmul(B_star.T,S))
    simplex_list_new = v_list_new + e_list_new + t_list_new
    block_lines_new = [len(v_list_new)-0.5, len(v_list_new)+len(e_list_new)-0.5]
    #Plot the old and new boundary matrices for this example
    plot_B_matrices(B_star,B_new,simplex_list,block_lines, simplex_list_new,block_lines_new)
    
    return A0_new, A1_new, A2_new, v_list_new, e_list_new, t_list_new 

#--------------------------------------------------------
# Function to cascade down first block column of S 
# e.g.:
#      S0
#      |
#     S01
#     |
#    S02
#--------------------------------------------------------
def cascade_down(S0, vertex_list,edge_list,triangle_list,tetrahedron_list):
    n = S0.shape[0]
    n_new = S0.shape[1]
    S01 = []
    S02 = []
    S03 = []
    for e in edge_list:
        edge_arr = np.zeros(n_new)
        v_list = e.split(",")
        v0 = int(v_list[0])
        v1 = int(v_list[1])
        for v in range(n_new):
            if (S0[v0,v]>0 or S0[v1,v]>0):
                edge_arr[v]=1
        S01.append(edge_arr)

    
    for t in triangle_list:
        tri_arr = np.zeros(n_new)
        t_list = t.split(",")
        v0 = int(t_list[0])
        v1 = int(t_list[1])
        v2 = int(t_list[2])
        for v in range(n_new):
            if (S0[v0,v]>0 or S0[v1,v]>0 or S0[v2,v]>0):
                tri_arr[v]=1
        S02.append(tri_arr)    
        
    for th in tetrahedron_list:
        th_arr = np.zeros(n_new)
        th_list = th.split(",")
        v0 = int(th_list[0])
        v1 = int(th_list[1])
        v2 = int(th_list[2])
        v3 = int(th_list[2])
        for v in range(n_new):
            if (S0[v0,v]>0 or S0[v1,v]>0 or S0[v2,v]>0 or S0[v3,v]>0):
                th_arr[v]=1
        S03.append(th_arr)  
        
    if S01:
        S01 = np.array(S01)
    else:
        S01 = np.array([[0]])
            
    if S02:
        S02 = np.array(S02)
    else:
        S02 = np.array([[0]])
   
    if S03:
        S03 = np.array(S03)
    else:
        S03 = np.array([[0]])

    return S01, S02, S03

#--------------------------------------------------------
# Function to cascade across rows of S (only for intermediate blocks, not diags)
# S is always the first block matrix in the row
# e.g.:
#      S02-->S12
#      S03-->S13-->S23
#--------------------------------------------------------
def cascade_across_row(S, sx_list, sx_list_next):
    if S.any():
        # starting simplex dim
        nrows = S.shape[0]
        ncols = S.shape[1]
        n_new = len(sx_list_next)
        S_next = np.zeros([nrows,n_new])
        n_dim_new =  len(sx_list_next[0].split(',')) #dimension of the new simplices+1 (e.g. edge = 2)

        for j in range(n_new): # for each new simplex
            cols = np.zeros([nrows,n_dim_new]) 
            #which (n-1 simplices) are boundaries of n-simplex?
            for d in range(n_dim_new):
                v = int(sx_list_next[j].split(',')[d])
                cols[:,d] = S[:,v]
                #entry-wise multiplication of columns gives new matrix entries
            S_next[:,j]= np.prod(cols,axis=1)
    else:
        S_next = np.array([[0]])
    return S_next

#-----------------------------------------------
# Function to cascade across the last block in row to  a diagonal block 
# S is always the first block matrix in the row
# e.g.:
#      S01-->S1
#            S12--S2
#-----------------------------------------------
def cascade_across_last(S, sx_list):
    if S.any():
        # starting simplex dim
        nrows = S.shape[0]
        ncols = S.shape[1]
        n_new_poss= int((ncols*(ncols-1))/2)
        S_next = np.zeros([nrows,n_new_poss])
        n_dim =  len(sx_list[0].split(',')) #dimension of the simplices in list
        sx_list_next = []
        col = 0
        for i in range(ncols):
            for j in range(i+1,ncols):
                if n_dim==1: #pooled vertices to pooled edges
                            #entry-wise multiplication of the two cols for v_i and v_j
                    mult_col = S[:,i] * S[:,j]
                            #if the multiplied column has non-zeros, there is a new simplex, 
                            # so the column is added to S_next,the column count is updated,
                            # and the list of new simplices is updated
                            # Else: do nothing, that combination of simplices does 
                            # not have a simplex of higher dim in the pooled complex
                    if np.sum(mult_col)> 0:
                        S_next[:,col] = mult_col
                        col+=1
                        sx_list_next.append(",".join([str(i),str(j)]))
                elif n_dim > 1: #pooled edges to pooled triangles (or higher)
                    for k in range(j+1,ncols):
                        if n_dim==2:
                            #entry-wise multiplication of the 3 cols for v_i, v_j, v_k
                            mult_col = S[:,i] * S[:,j]* S[:,k]
                            if np.sum(mult_col)> 0:
                                S_next[:,col] = mult_col
                                col+=1
                                sx_list_next.append(",".join([str(i),str(j),str(k)]))
                        else: #pooled triangles to pooled tetrahedra
                            for l in range(k+1,ncols):
                             #entry-wise multiplication of the 4 cols for v_i, v_j, v_k, v_l
                                mult_col = S[:,i] * S[:,j]* S[:,k]* S[:,l]
                                if np.sum(mult_col)> 0:
                                    S_next[:,col] = mult_col
                                    col+=1
                                    sx_list_next.append(",".join([str(i),str(j),str(k), str(l)]))

        #delete columns of all zeros from S_next
        idx = np.argwhere(np.all(S_next[..., :] == 0, axis=0))
        S_next = np.delete(S_next, idx, axis=1)
    else:
        S_next = np.array([[0]])
        sx_list_next=[]
    return S_next, sx_list_next

#-----------------------------------------------
# Function to apply simplicial pooling, only using the lower triangular half of S matrix-- ADJACENCY VERSION
#-----------------------------------------------
def apply_pooling_lowerS(S0,vertex_list=[],edge_list=[],triangle_list=[],tetrahedron_list=[],A0=[],A1=[],A2=[],A3=[], C1=[], C2=[],C3=[]):
    if isinstance(A3,list):
        A3 = np.array([[0]])
    if isinstance(A2,list):
        A2 = np.array([[0]])
    if isinstance(A1,list):
        A1 = np.array([[0]])
    if isinstance(C1,list):
        C1 = np.array([[0]])
    if isinstance(C2,list):
        C2 = np.array([[0]])
    if isinstance(C3,list):
        C3 = np.array([[0]])
    # construct new vertex list
    v_list_new = list(map(str,list(range(S0.shape[1]))))
    
    n0= len(vertex_list)
    n1= len(edge_list)
    n2= len(triangle_list)
    n3= len(tetrahedron_list)
    simplex_list = vertex_list+edge_list+triangle_list+tetrahedron_list
    n_total = len(simplex_list)
    if n3!=0:
        K_max_dim = 3
        #Plot adjacency matrixes for original complex
        plot_A_matrices(A0=A0,A1=A1,A2=A2,A3=A3,v_list=vertex_list,
                        e_list=edge_list,t_list=triangle_list, th_list= tetrahedron_list)
    elif n2!=0:
        K_max_dim = 2
        plot_A_matrices(A0=A0,A1=A1,A2=A2,v_list=vertex_list,e_list=edge_list,
                           t_list=triangle_list)
    else: # graph
        K_max_dim = 1
        plot_A_matrices(A0=A0,A1=A1,v_list=vertex_list,e_list=edge_list)
        
    block_lines = [n0-0.5, n0+n1-0.5, n0+n1+n2-0.5]


    ## Cascade down
    S01, S02, S03 = cascade_down(S0, vertex_list,edge_list,triangle_list,tetrahedron_list)
    #print('S01 after cascade down is: ',S01)
    #print('S02 after cascade down is: ',S02)
    #print('S03 after cascade down is: ',S03)
    ## Cascade across
    S1, e_list_new = cascade_across_last(S01,v_list_new)
    #print('S1 after cascade across is: ',S1)
    S12 = cascade_across_row(S02,v_list_new, e_list_new)
    #print('S12 after cascade across is: ',S12) 
    S13 = cascade_across_row(S03,v_list_new,e_list_new)
    #print('S13 after cascade across is: ',S13)
    S2, t_list_new = cascade_across_last(S02,e_list_new)
    #print('S2 after cascade across is: ',S2)
    #print('t_list_new: ',t_list_new)
    S23 = cascade_across_row(S03,e_list_new, t_list_new)
    #print('S23 after cascade across is: ',S23)
    S3, th_list_new = cascade_across_last(S03,t_list_new)
    #print('S3 after cascade across is: ',S3)
    #print('th_list_new: ',th_list_new)
    # new max dim of pooled complex, normalize row blocks of S
    # using row_normalize()
    if th_list_new:
        K_max_dim_new = 3
        S3, S03, S13, S23 = row_normalize(Sn=S3, Sa=S03,Sb=S13,Sc=S23)
        S2, S02, S12, _ = row_normalize(Sn=S2, Sa=S02,Sb=S12)
        S1, S01,_,_ = row_normalize(Sn=S1, Sa=S01)
        S0,_,_,_ = row_normalize(Sn=S0)
    elif t_list_new:
        K_max_dim_new = 2
        S2, S02, S12, _ = row_normalize(Sn=S2, Sa=S02,Sb=S12)
        S1, S01,_,_ = row_normalize(Sn=S1, Sa=S01)
        S0,_,_,_ = row_normalize(Sn=S0)
    else:
        K_max_dim_new = 1
        S1, S01,_,_ = row_normalize(Sn=S1, Sa=S01)
        S0,_,_,_ = row_normalize(Sn=S0)


    # Compute Ai_new = Si^T Ai Si (Using the lower triangular columns of S)
    A0_new = np.array([])
    A1_new = np.array([])
    A2_new = np.array([])
    A0_new = np.matmul(S0.T,np.matmul(A0,S0)) + np.matmul(S01.T,np.matmul(A1,S01)) + np.matmul(S02.T,np.matmul(A2,S02)) + np.matmul(S03.T,np.matmul(A3,S03))
    A1_new = np.matmul(S1.T,np.matmul(A1,S1)) + np.matmul(S12.T,np.matmul(A2,S12))+ np.matmul(S13.T,np.matmul(A3,S13))
    A2_new = np.matmul(S2.T,np.matmul(A2,S2))+ np.matmul(S23.T,np.matmul(A3,S23))
    # Plot the new adjacency matrices from pooling
    # Compute Ci_new = Si^T Ci Si (Using the lower triangular columns of S)
    C1_new = np.array([])
    C2_new = np.array([])
    C1_new = np.matmul(S1.T,np.matmul(C1,S1)) + np.matmul(S12.T,np.matmul(C2,S12))+ np.matmul(S13.T,np.matmul(C3,S13))
    C2_new = np.matmul(S2.T,np.matmul(C2,S2))+ np.matmul(S23.T,np.matmul(C3,S23))
    
    # Normalize the new adjacency matrices
    A0_new = normalizeA(A0_new)
    A1_new = normalizeA(A1_new)
    A2_new = normalizeA(A2_new)
    
    # Normalize the new co-adjacency matrices
    C1_new = normalizeA(C1_new)
    C2_new = normalizeA(C2_new)
    
    
    #Plot the new adjacency matrices
    if (K_max_dim_new == 1):  
        plot_A_matrices(A0=A0_new,A1=A1_new,v_list=v_list_new,e_list=e_list_new)
    elif (K_max_dim_new == 2):  
        plot_A_matrices(A0=A0_new,A1=A1_new,A2=A2_new,v_list=v_list_new,
                                e_list=e_list_new,t_list=t_list_new)       
    else:
        plot_A_matrices(A0=A0_new,A1=A1_new,A2=A2_new,A3=A3_new,v_list=v_list_new,
                        e_list=e_list_new,t_list=t_list_new,th_list=th_list_new)

    simplex_list_new = v_list_new + e_list_new + t_list_new + th_list_new
    #block_lines_new = [len(v_list_new)-0.5, len(v_list_new)+len(e_list_new)-0.5]
    
    return A0_new, A1_new, A2_new, C1_new, C2_new, v_list_new, e_list_new, t_list_new 


#------------------------------------------------
# Function to row normalize sub-blocks of S 
# Sn is the diagonal block matrix of S
# S is a row of all block matrices to the left of Sn
# returns the block matrices back normalized
# [Sa] [Sb] [Sc] [Sn]
#------------------------------------------------
def row_normalize(Sn, Sa=[], Sb=[], Sc=[]):
    if isinstance(Sa, list):
        rowSum = Sn.sum(axis=-1,keepdims=1)
        rowSum = rowSum.astype(float)
        Sn = Sn/rowSum
        Sa = np.array([[0]])
        Sb = np.array([[0]])
        Sc = np.array([[0]])
    elif isinstance(Sb, list):
        rowSum = Sa.sum(axis=-1,keepdims=1)+Sn.sum(axis=-1,keepdims=1)
        rowSum = rowSum.astype(float)
        Sn = Sn/rowSum
        Sa = Sa/rowSum
        Sb = np.array([[0]])
        Sc = np.array([[0]])
    elif isinstance(Sc, list):
        rowSum = Sa.sum(axis=-1,keepdims=1)+ Sb.sum(axis=-1,keepdims=1)+Sn.sum(axis=-1,keepdims=1)
        rowSum = rowSum.astype(float)
        Sn = Sn/rowSum
        Sa = Sa/rowSum
        Sb = Sb/rowSum
        Sc = np.array([[0]])
    else:
        rowSum = Sa.sum(axis=-1,keepdims=1)+ Sb.sum(axis=-1,keepdims=1)+ Sc.sum(axis=-1,keepdims=1)+Sn.sum(axis=-1,keepdims=1)
        rowSum = rowSum.astype(float)
        Sn = Sn/rowSum
        Sa = Sa/rowSum
        Sb = Sb/rowSum
        Sc = Sc/rowSum
        
    return Sn, Sa, Sb, Sc


#---------------------------------------------
# Function to derive C_p from p-simplex list and p-1 adjacency 
#   simplex list: the list of simplices in dimension p
#   dim: p
#---------------------------------------------
def derive_coadj(simplex_list, dim):
    Cp = np.zeros([len(simplex_list), len(simplex_list)])
    if dim==1: #derive the edge co-adjacency matrix
        for e1_idx in range(len(simplex_list)):
            for e2_idx in range(e1_idx,len(simplex_list)):
                e1 = simplex_list[e1_idx]
                e2 = simplex_list[e2_idx]
                #print(e1, e1_idx, e2, e2_idx)
                v1 = int(e1.split(',')[0])
                v2 = int(e1.split(',')[1])
                va = int(e2.split(',')[0])
                vb = int(e2.split(',')[1])
                if v1==va or v1==vb or v2==va or v2==vb:
                #Then, the edges are co-adjacent
                    Cp[e1_idx, e2_idx] = 1
                if e1==e2:
                    Cp[e1_idx, e2_idx] = 0
    elif dim==2: #derive the triangle co-adjacency matrix
        for t1_idx in range(len(simplex_list)):
            for t2_idx in range(t1_idx,len(simplex_list)):
                t1 = simplex_list[t1_idx]
                t2 = simplex_list[t2_idx]

                v1 = int(t1.split(',')[0])
                v2 = int(t1.split(',')[1])
                v3 = int(t1.split(',')[2])
                va = int(t2.split(',')[0])
                vb = int(t2.split(',')[1])
                vc = int(t2.split(',')[2])
                tlist1 = [v1,v2,v3] 
                tlist2 = [va,vb,vc]
                overlap = [value for value in tlist1 if value in tlist2]
                # Two triangles are co-adjacent if they share two vertices in common
                if len(overlap)==2:
                    #Then, the triangles are co-adjacent
                    Cp[t1_idx, t2_idx] = 1         
    Cp = Cp + Cp.T

    return Cp

#---------------------------------------------
# Function to normalize adjacency matrix 
#   At = A+I
#   Dt = degree of At
#   renormA = Dt^(-1/2) At Dt^(-1/2)
#---------------------------------------------
def normalizeA(A):
    # Renormalization: (ala Kipf and Welling)
    n = np.size(A,0)
    At = A+np.eye(n)
    Dt = np.diag(np.sum(At,axis=1))
    renormA = np.matmul(np.matmul(np.linalg.inv(Dt/2),At),np.linalg.inv(Dt/2))
    return renormA




#-----------------------------------------------
# Function to update ADJACENCY MATRICES (after pooling)
# Using all of the lower triangular half of S matrix-- ADJACENCY VERSION
#-----------------------------------------------
def updateAdjacency(S0, S1, S2, S3, S01, S02, S03, S12, S13, S23, A0, A1, A2, A3):
    # Compute Ai_new = Si^T Ai Si (Using the lower triangular columns of S)
    A0_new = np.array([])
    A1_new = np.array([])
    A2_new = np.array([])
    A0_new = np.matmul(S0.T,np.matmul(A0,S0)) + np.matmul(S01.T,np.matmul(A1,S01)) + np.matmul(S02.T,np.matmul(A2,S02)) + np.matmul(S03.T,np.matmul(A3,S03))
    A1_new = np.matmul(S1.T,np.matmul(A1,S1)) + np.matmul(S12.T,np.matmul(A2,S12))+ np.matmul(S13.T,np.matmul(A3,S13))
    A2_new = np.matmul(S2.T,np.matmul(A2,S2))+ np.matmul(S23.T,np.matmul(A3,S23))
   
    # Normalize the new adjacency matrices
    A0_new = normalizeA(A0_new)
    A1_new = normalizeA(A1_new)
    A2_new = normalizeA(A2_new)
    
    return A0_new, A1_new, A2_new

#-----------------------------------------------
# Function to update BOUNDARY adj style MATRICES (after pooling)
#-----------------------------------------------
def updateBoundariesTogether(S0, S1, S2, S3, S01, S02, S03, S12, S13, S23, B1, B2, B3):

    # Compute Pooled Boundary Matrices
    
    A0_low = np.array([])
    A0_up = np.array([])
    
    A1_low = np.array([])
    A1_up = np.array([])
    
    A2_low = np.array([])
    A2_up = np.array([])
    
    #np.matmul(B1, B1.T)
    #np.matmul(B1.T, B1)
    #np.matmul(B2, B2.T)
    #np.matmul(B2.T, B2)
    #np.matmul(B3, B3.T)
    #np.matmul(B3.T, B3)
    
    #A0_low = empty
    A0_up = np.matmul(S0.T,np.matmul(np.matmul(B1, B1.T),S0))+ np.matmul(S01.T,np.matmul(np.matmul(B2, B2.T),S01)) + np.matmul(S02.T,np.matmul(np.matmul(B3, B3.T),S02))  
    A1_low = np.matmul(S1.T,np.matmul(np.matmul(B1.T, B1),S1)) 
    A1_up = np.matmul(S1.T,np.matmul(np.matmul(B2, B2.T),S1))

        
    A2_low = np.matmul(S2.T,np.matmul(np.matmul(B2.T, B2),S2))
    A2_up = np.matmul(S2.T,np.matmul(np.matmul(B3, B3.T),S2))
    
    A3_low = np.matmul(S3.T,np.matmul(np.matmul(B3.T, B3),S3))
    #A3_up = empty
   
    # Normalize the new adjacency matrices
    #A0_up = normalizeA(A0_up)
    #A1_low = normalizeA(A1_low)
    #A1_up = normalizeA(A1_up)
    #A2_low = normalizeA(A2_low)
    #A2_up = normalizeA(A2_up)
    #A3_low = normalizeA(A3_low)
    
    return A0_up, A1_low, A1_up, A2_low, A2_up, A3_low
#-----------------------------------------------
# Function to update BOUNDARY MATRICES (after pooling)
#-----------------------------------------------
def updateBoundaries(S0, S1, S2, S3, B1, B2, B3):

    # Compute Pooled Boundary Matrices
    # Bi_new = S_i^TB_iS_{i-1}
    B1_new = np.array([])
    B2_new = np.array([])
    B3_new = np.array([])
    B1_new = np.matmul(S0.T,np.matmul(B1,S1))
    B2_new = np.matmul(S1.T,np.matmul(B2,S2))
    B3_new = np.matmul(S2.T,np.matmul(B3,S3))

    
    return B1_new, B2_new, B3_new 



#-----------------------------------------------
# Function to apply simplicial pooling, 
#-----------------------------------------------
def extend_S_matrix(S0,vertex_list=[],edge_list=[],triangle_list=[],tetrahedron_list=[]):

    # construct new vertex list
    v_list_new = list(map(str,list(range(S0.shape[1]))))
    
    n0= len(vertex_list)
    n1= len(edge_list)
    n2= len(triangle_list)
    n3= len(tetrahedron_list)
    simplex_list = vertex_list+edge_list+triangle_list+tetrahedron_list
    n_total = len(simplex_list)

    ## Cascade down
    S01, S02, S03 = cascade_down(S0, vertex_list,edge_list,triangle_list,tetrahedron_list)

    ## Cascade across
    S1, e_list_new = cascade_across_last(S01,v_list_new)
    S12 = cascade_across_row(S02,v_list_new, e_list_new)
    S13 = cascade_across_row(S03,v_list_new,e_list_new)
    S2, t_list_new = cascade_across_last(S02,e_list_new)
    S23 = cascade_across_row(S03,e_list_new, t_list_new)
    S3, th_list_new = cascade_across_last(S03,t_list_new)
    
    # New max dim of pooled complex, normalize row blocks of S
    # using row_normalize()
    if th_list_new:
        K_max_dim_new = 3
        S3, S03, S13, S23 = row_normalize(Sn=S3, Sa=S03,Sb=S13,Sc=S23)
        S2, S02, S12, _ = row_normalize(Sn=S2, Sa=S02,Sb=S12)
        S1, S01,_,_ = row_normalize(Sn=S1, Sa=S01)
        S0,_,_,_ = row_normalize(Sn=S0)
    elif t_list_new:
        K_max_dim_new = 2
        S2, S02, S12, _ = row_normalize(Sn=S2, Sa=S02,Sb=S12)
        S1, S01,_,_ = row_normalize(Sn=S1, Sa=S01)
        S0,_,_,_ = row_normalize(Sn=S0)
    else:
        K_max_dim_new = 1
        S1, S01,_,_ = row_normalize(Sn=S1, Sa=S01)
        S0,_,_,_ = row_normalize(Sn=S0)

    
    return S01, S02, S03, S1, S12, S13, S2, S23, S3, v_list_new, e_list_new, t_list_new
