import torch

from complex import SComplex, pool_complex


def nerve_pool_dense_batch(x, boundaries, s):
    sc_torch = SComplex(boundaries=boundaries)
    sc_torch_pooled = pool_complex(sc_torch, s)
