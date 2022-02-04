import numpy as np
import torch

from . import _torch
import utils

def get_grid_size(Ls, shape, w_dims=[]):
    """
    returns scalar
    """
    return np.prod(np.array(Ls)/np.array([shape[i] if i in w_dims else shape[i]-1 for i in range(len(shape))]))

def get_dxs(Ls, shape, w_dims=[]):
    """
    returns np.ndarray
    """
    return np.array(Ls)/np.array([shape[i] if i in w_dims else shape[i]-1 for i in range(len(shape))])

def get_mids(shape, w_dims=[]):
    """
    Returns the index of the center of shape.
    Raises AssertionError if the center of shape does not coincide with a particular index
    """
    D = len(shape)
    assert all([shape[d] % 2 == 0 if d in w_dims else shape[d] % 2 == 1 for d in range(D)]) # odd number of neurons for symmetry
    return tuple([shape[d]//2 if d in w_dims else (shape[d]-1)//2 for d in range(D)])

def get_grid(Ls, shape, w_dims=[], device='cpu'):
    endpoints = [False if i in w_dims else True for i in range(len(Ls))]
    grids_per_dim = [_torch.linspace(-L/2,L/2,shape[i],endpoint=endpoints[i],device=device) for i, L in enumerate(Ls)]
    grid = torch.stack(torch.meshgrid(*grids_per_dim, indexing='ij'), dim=-1)
    return grid

def get_range_grid(ranges, shape, w_dims=[], device='cpu'):
    endpoints = [False if i in w_dims else True for i in range(len(ranges))]
    grids_per_dim = [_torch.linspace(ranges[i][0],ranges[i][1],shape[i],endpoint=endpoints[i],device=device) for i, L in enumerate(ranges)]
    grid = torch.stack(torch.meshgrid(*grids_per_dim, indexing='ij'), dim=-1)
    return grid

def get_int_grid(shape, device='cpu'):
    grids_per_dim = [torch.arange(shape[i], device=device) for i in range(len(shape))]
    grid = torch.stack(torch.meshgrid(*grids_per_dim, indexing='ij'), dim=-1)
    return grid

def meshgrid(tensors):
    """
    A generalization of torch.meshgrid
    Mesh together list of tensors of shapes (n_1_1,...,n_1_{M_1},N_1), (n_2_1,...,n_2_{M_2},N_2), ...
    Returns tensors of shapes (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,N_1), (n_2_1,...,n_2_{M_2},...,N_2)
    """
    sizes = [list(tensor.shape[:-1]) for tensor in tensors] # [[n_1,...,n_{M_1}],[n_1,...,.n_{M_2}],...]
    Ms = np.array([tensor.ndim - 1 for tensor in tensors]) # [M_1, M_2, ...]
    M_befores = np.cumsum(np.insert(Ms[:-1],0,0))
    M_afters = np.sum(Ms) - np.cumsum(Ms)
    Ns = [tensor.shape[-1] for tensor in tensors]
    shapes = [[1]*M_befores[i]+sizes[i]+[1]*M_afters[i]+[Ns[i]] for i, tensor in enumerate(tensors)]
    expanded_tensors = [tensor.reshape(shapes[i]).expand(utils.itertools.flatten_seq(sizes)+[Ns[i]]) for i, tensor in enumerate(tensors)]
    return expanded_tensors