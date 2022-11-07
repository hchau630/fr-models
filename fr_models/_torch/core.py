#  A module containing functions with pytorch backend that behave like numpy functions

import torch
import numpy as np

from fr_models import gridtools

import utils

__all__ = ['linspace', 'take', 'pad', 'block', 'tensor', 'isclose', 'allclose', 'isequal', 'isconst', 'nanstd', 'stderr', 'nanstderr']

def linspace(start, end, steps, endpoint=True, **kwargs):
    if endpoint:
        return torch.linspace(start, end, steps, **kwargs)
    else:
        return torch.linspace(start, end, steps+1, **kwargs)[:-1] # exclude endpoint
    
def take(tensor, indices, dim=None):
    """
    Current torch.take does not have dim argument like numpy
    """
    if dim is None:
        return torch.take(tensor, indices)
    else:
        assert isinstance(dim, int)
        dim = dim % tensor.ndim
        ind = tuple([slice(None)]*dim+[indices])
        return tensor[ind]
    
def pad(tensor, pad_width, mode='wrap', const=0):
    device = tensor.device
    
    if mode == 'const':
        result = torch.ones([tensor.shape[i]+pad_width[i][0]+pad_width[i][1] for i in range(tensor.ndim)], device=device, dtype=tensor.dtype)*const
        indices = tuple([slice(pad_width[i][0] if pad_width[i][0] != 0 else None,-pad_width[i][1] if pad_width[i][1] != 0 else None) for i in range(tensor.ndim)])
        result[indices] = tensor
        
        return result
    if mode == 'wrap':
        indices = gridtools.get_grid(
            [(-pad_width[i][0],tensor.shape[i]+pad_width[i][1]) for i in range(tensor.ndim)], 
            method='arange',
            device=device
        )
        indices = torch.moveaxis(indices, -1, 0)
        for dim in range(len(indices)):
            indices[dim] = indices[dim] % tensor.shape[dim]
        result = tensor[tuple(indices)]

        return result
    raise NotImplementedError()

def block(tensors):
    """
    similar to np.block, but treats the first n-2 dimensions as batch dimensions. Requires tensor.ndim >= 2 for each tensor in tensors
    and requires tensors to have ndim = 2
    """
    tensors = [torch.cat(tensor, dim=-1) for tensor in tensors]
    return torch.cat(tensors, dim=-2)
    
def tensor(data, **kwargs):
    """
    Mimics np.array in that this can take in a (nested) sequence of tensors and create a new tensor as such:
    
    >>> from fr_models import utils
    
    >>> A,B,C,D = [torch.ones((2,3))*i for i in range(4)]
    >>> X = utils._torch.tensor([[A,B,B,D],[C,D,A,B]])
    >>> print(X.shape)
    torch.Size([2, 4, 2, 3])
    
    This does not fully mimic np.array, since torch.Tensor does not support dtype=object.
    """
    try:
        return torch.tensor(data, **kwargs)
    except Exception as err:
        # try to recursively create tensors by stacking along the first dimension. 
        # stop recursing if the element is already a tensor.
        return torch.stack([elem if isinstance(elem, torch.Tensor) else tensor(elem) for elem in data], dim=0)
    
def isclose(x, y, rtol=1.0e-5, atol=1.0e-8):
    """
    A more flexible version of torch.isclose that allows for different atols and rtols for different elements of the tensor
    """
    return (x-y).abs() <= atol + rtol * y.abs()

def allclose(*args, **kwargs):
    return isclose(*args, **kwargs).all()

@utils.functools.deprecated
def isequal(x, dim=-1, rtol=1.0e-5, atol=1.0e-8):
    x = x.moveaxis(dim,-1)
    return isclose(x[...,:-1], x[...,1:], rtol=rtol, atol=atol).all(dim=-1)

def isconst(x, dim=None, **kwargs):
    if dim is None:
        x = x.reshape(-1)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted([d % x.ndim for d in dim])[::-1]
        for d in dim:
            x = x.movedim(d,-1)
        x = x.flatten(start_dim=-len(dim))
    return torch.isclose(x[...,:-1], x[...,1:], **kwargs).all(dim=-1)

def nanstd(x, *args, **kwargs):
    raise NotImplementedError()
    # return x[~torch.isnan(x)].std(*args, **kwargs) # does not work with dim

def stderr(x, dim=None, unbiased=True):
    if dim is None:
        numel = x.numel()
    else:
        numel = np.prod(x.shape[tuple(dim)])
    return x.std(dim=dim, unbiased=unbiased) / numel**0.5

def nanstderr(x, dim=None, unbiased=True):
    raise NotImplementedError()
    # return nanstd(x, dim=dim, unbiased=unbiased) / x[~torch.isnan(x)].numel()**0.5 # does not work with dim
