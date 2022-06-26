#  A module containing functions with pytorch backend that behave like numpy functions

import torch
from fr_models import gridtools

__all__ = ['linspace', 'take', 'pad', 'block', 'tensor', 'isclose', 'allclose', 'isequal', 'nanstd', 'stderr', 'nanstderr']

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
    
def pad(tensor, pad_width, mode='wrap'):
    if mode != 'wrap':
        raise NotImplementedError()
        
    device = tensor.device
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

def isequal(x, dim=-1, rtol=1.0e-5, atol=1.0e-8):
    x = x.moveaxis(dim,-1)
    return isclose(x[...,:-1], x[...,1:], rtol=rtol, atol=atol).all(dim=-1)

def nanstd(x, *args, **kwargs):
    return x[~torch.isnan(x)].std(*args, **kwargs)

def stderr(x):
    return x.std() / x.numel()**0.5

def nanstderr(x):
    return nanstd(x) / x[~torch.isnan(x)].numel()**0.5
